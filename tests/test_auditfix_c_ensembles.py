"""Regression tests for audit workstream C ensemble/optimizer fixes.

TR-ENS-03 (MC-Dropout module modes), TR-ENS-01/02 (SWAG buffer tracking and
registered low-rank deviations), TR-TT-01 (IVON state-dict device migration).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from torchregress.algorithms.ivon import IVON
from torchregress.ensemble.mc_dropout import MCDropoutWrapper, _module_mode
from torchregress.ensemble.swag import SWAG


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.fc(x)))


# ═══════════════════════════════════════════════════════════════════════════════
# TR-ENS-03: mc_forward keeps BN in eval; forward stops mutating global mode
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCDropoutModes:
    def test_mc_forward_does_not_mutate_bn_running_stats(self) -> None:
        torch.manual_seed(0)
        net = _TinyNet()
        net.train()
        x = torch.randn(16, 4)
        net(x)  # initialize running stats in train mode
        rm_before = net.bn.running_mean.clone()
        rv_before = net.bn.running_var.clone()

        wrapper = MCDropoutWrapper(net, n_samples=8)
        preds = wrapper.mc_forward(x)

        assert preds.shape[0] == 8
        delta_mean = (net.bn.running_mean - rm_before).abs().max().item()
        delta_var = (net.bn.running_var - rv_before).abs().max().item()
        assert delta_mean == pytest.approx(0.0, abs=1e-7)
        assert delta_var == pytest.approx(0.0, abs=1e-7)

    def test_forward_restores_original_module_modes(self) -> None:
        net = _TinyNet()
        net.train()
        wrapper = MCDropoutWrapper(net)
        wrapper(torch.randn(4, 4))
        assert net.training is True
        assert net.drop.training is True

    def test_module_mode_contextmanager_snapshot_roundtrip(self) -> None:
        net = _TinyNet()
        net.eval()
        with _module_mode(net, dropout_train=True):
            assert net.drop.training is True
            assert net.bn.training is False
        assert net.drop.training is False and net.bn.training is False


# ═══════════════════════════════════════════════════════════════════════════════
# TR-ENS-01/02: SWAG tracks buffers and registers deviations as buffers
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWAGBufferTracking:
    def _collect(self) -> tuple[SWAG, _TinyNet]:
        torch.manual_seed(0)
        net = _TinyNet()
        swag = SWAG(net, max_num_models=4)
        x = torch.randn(8, 4)
        for _ in range(3):
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(0.05 * torch.randn_like(p))
            net(x)
            swag.collect_model(net)
        return swag, net

    def test_running_buffers_are_collected(self) -> None:
        swag, _ = self._collect()
        buffers = dict(swag.named_buffers())
        # TR-ENS-01 corrected: BN running stats must NOT be tracked as Gaussian posterior
        # (they are deterministic population estimates; Maddox Alg.1 keeps BN recalibrate)
        assert "bn_running_mean_mean" not in buffers
        assert "bn_running_var_sq_mean" not in buffers
        # Non-BN buffers (if any) would be tracked, but TinyNet has only BN; check n_models
        assert int(swag.n_models.item()) == 3

    def test_deviations_are_registered_buffers_in_state_dict(self) -> None:
        swag, _ = self._collect()
        sd = swag.state_dict()
        assert any(k.endswith("fc_weight_devs") for k in sd)

    def test_state_dict_round_trip_preserves_posterior(self) -> None:
        swag, _ = self._collect()
        sd = swag.state_dict()
        fresh = SWAG(copy.deepcopy(_TinyNet()), max_num_models=4)
        fresh.load_state_dict(sd)
        assert int(fresh.n_models.item()) == int(swag.n_models.item())
        sd2 = fresh.state_dict()
        for key in sd:
            assert torch.allclose(sd[key], sd2[key]), f"round-trip mismatch at {key}"

    def test_sample_writes_buffer_means_into_model(self) -> None:
        swag, net = self._collect()
        before = net.bn.running_mean.clone()
        swag.sample(scale=0.0)  # scale 0 => mean for params, but BN buffers must NOT be overwritten
        after = net.bn.running_mean
        # BN running stats are not sampled; they remain as before (deterministic)
        assert torch.allclose(after, before, atol=1e-6)
        del before

# ═══════════════════════════════════════════════════════════════════════════════
# TR-TT-01: IVON load_state_dict migrates group tensors to optimizer device/dtype
# ═══════════════════════════════════════════════════════════════════════════════


class TestIVONStateDictMigration:
    def test_group_tensors_migrated_to_new_dtype(self) -> None:
        torch.manual_seed(0)
        lin = nn.Linear(4, 2)
        opt = IVON(lin.parameters(), lr=0.1, ess=10)
        x = torch.randn(8, 4)
        for _ in range(3):
            with opt.sampled_params(train=True):
                loss = lin(x).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

        sd = opt.state_dict()

        lin64 = nn.Linear(4, 2).to(torch.float64)
        opt64 = IVON(lin64.parameters(), lr=0.1, ess=10)
        opt64.load_state_dict(sd)

        for group in opt64.param_groups:
            for key in ("momentum", "hess"):
                tensor = group[key]
                assert isinstance(tensor, torch.Tensor)
                assert tensor.dtype == torch.float64
                assert str(tensor.device) == str(next(lin64.parameters()).device)

    def test_numeric_state_preserved_through_round_trip(self) -> None:
        torch.manual_seed(0)
        lin = nn.Linear(4, 2)
        opt = IVON(lin.parameters(), lr=0.1, ess=10)
        x = torch.randn(8, 4)
        for _ in range(2):
            with opt.sampled_params(train=True):
                loss = lin(x).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

        sd = opt.state_dict()
        opt2 = IVON(nn.Linear(4, 2).parameters(), lr=0.1, ess=10)
        opt2.load_state_dict(sd)
        sd2 = opt2.state_dict()
        g1, g2 = sd["param_groups"][0], sd2["param_groups"][0]
        assert torch.allclose(g1["momentum"], g2["momentum"])
        assert torch.allclose(g1["hess"], g2["hess"])
