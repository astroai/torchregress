"""Joint distributional test-time adaptation with conformal recalibration.

Safe-ordering protocol (adapt once -> freeze -> estimate weights from features
-> recalibrate last): importance weights are fixed BEFORE the adapted model's
calibration scores are computed, so the weighted NexCP coverage bounds of
:class:`~torchregress.losses.conformal.MultivariateScoreConformal` apply to the
adapted model (Barber et al. 2023, arXiv:2202.13415, Thms. 2-3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from ..losses.conformal import MultivariateScoreConformal
from .shift_weights import (
    DomainClassifierRatioEstimator,
    OTScoreWeightEstimator,
    estimate_label_shift_weights,
)
from .subspace import WeightedSubspaceMomentAligner

_VAR_EPS = 1e-6


def gaussian_outputs(model: Any, X: Tensor) -> tuple[Tensor, Tensor]:
    """Extract ``(mu, var)`` from a distributional regression model.

    Supported output contracts:
      * tuple ``(mu, var)`` where ``mu`` is ``[n, d]`` and ``var`` is a
        positive ``[n, d]`` diagonal or ``[n, d, d]`` covariance;
      * single tensor ``[n, 2d]`` interpreted as concatenated
        ``(mean, log_variance)`` columns (diagonal Gaussian).
    """
    out = model(X)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        mu, var = out[0], out[1]
        mu = mu.reshape(mu.shape[0], -1) if mu.dim() > 2 else mu
        if var.dim() > 2 and not torch.is_tensor(var):
            raise TypeError("var must be a tensor")
        return mu, var
    if torch.is_tensor(out):
        if out.dim() != 2 or out.shape[1] % 2 != 0:
            raise ValueError(
                f"model output must be [n, 2d] (mean, log_var), got {tuple(out.shape)}"
            )
        d = out.shape[1] // 2
        return out[:, :d], torch.exp(out[:, d:].clamp(min=-15.0, max=15.0))
    raise TypeError(f"unsupported model output type {type(out)!r}")


@dataclass
class JointTTAResult:
    """Outcome of :meth:`JointDistributionalTTA.adapt_and_calibrate`."""

    adapted_model: Any
    conformal: MultivariateScoreConformal
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class JointDistributionalTTA:
    """Adapt a distributional regressor at test time, then recalibrate jointly.

    Pipeline (ordered; the ordering IS the theorem assumption):
      1. optional SSA feature alignment (source features only);
      2. fit the density-ratio weight estimator on frozen source/target
         features -> ``w_cal`` fixed from now on;
      3. ``pseudo_label_rounds`` rounds of head-only fine-tuning on
         uncertainty-filtered target pseudo-labels;
      4. freeze the model, recompute source-calibration scores with it, and
         calibrate :class:`MultivariateScoreConformal` with the fixed weights.

    Args:
        alpha: Miscoverage level.
        weight_estimator: ``"domain_clf"`` | ``"ot"`` | ``"label_shift_em"``.
        align_features: Fit :class:`WeightedSubspaceMomentAligner` when the
            model exposes a ``feature_extractor`` attribute; skipped silently
            (with a diagnostics note) otherwise — architectures are never guessed.
        pseudo_label_rounds: Number of pseudo-label fine-tuning rounds
            (0 = calibrator-only adaptation, pure weighted-NexCP baseline).
        pseudo_label_quantile: Fraction of tightest-predictive target points
            kept as pseudo-labeled training data each round.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        weight_estimator: Literal["domain_clf", "ot", "label_shift_em"] = "domain_clf",
        align_features: bool = True,
        pseudo_label_rounds: int = 1,
        pseudo_label_quantile: float = 0.8,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if weight_estimator not in ("domain_clf", "ot", "label_shift_em"):
            raise ValueError(f"unknown weight_estimator {weight_estimator!r}")
        if not 0.0 < pseudo_label_quantile <= 1.0:
            raise ValueError("pseudo_label_quantile must be in (0, 1]")
        self.alpha = float(alpha)
        self.weight_estimator = weight_estimator
        self.align_features = bool(align_features)
        self.pseudo_label_rounds = int(pseudo_label_rounds)
        self.pseudo_label_quantile = float(pseudo_label_quantile)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _features(model: Any, X: Tensor) -> Optional[Tensor]:
        extractor = getattr(model, "feature_extractor", None)
        if extractor is None:
            return None
        was_training = getattr(model, "training", False)
        if was_training:
            model.eval()
        with torch.no_grad():
            feats = extractor(X)
        if was_training:
            model.train()
        if feats.dim() > 2:
            feats = feats.reshape(feats.shape[0], -1)
        return feats

    def _fit_weights(
        self,
        F_src: Tensor,
        F_tgt_aligned: Tensor,
        X_target_unlabeled: Tensor,
        model: Any,
        X_cal_src: Tensor,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """Fit the density-ratio estimator; weights are frozen on return."""
        note: Dict[str, Any] = {"estimator": self.weight_estimator}
        if self.weight_estimator == "domain_clf":
            est = DomainClassifierRatioEstimator().fit(F_src, F_tgt_aligned)
            w = est.weights(F_src)
            note.update(est.diagnostics_)
        elif self.weight_estimator == "ot":
            est = OTScoreWeightEstimator(n_steps=200).fit(F_src, F_tgt_aligned)
            w = est.weights(F_src)
            note.update(getattr(est, "diagnostics_", {}))
        else:  # label_shift_em on binned predictive means
            with torch.no_grad():
                mu_src, _ = gaussian_outputs(model, X_cal_src)
                mu_tgt, _ = gaussian_outputs(model, X_target_unlabeled)
            n_bins = min(10, max(2, F_src.shape[0] // 20))
            src_score = mu_src.detach().mean(dim=-1)
            tgt_score = mu_tgt.detach().mean(dim=-1)
            edges = torch.quantile(src_score, torch.linspace(0, 1, n_bins + 1))
            edges[0] -= 1.0
            edges[-1] += 1.0
            k = int(edges.numel())
            src_bins = torch.bucketize(src_score.contiguous(), edges).clamp(max=k - 1)
            tgt_bins = torch.bucketize(tgt_score.contiguous(), edges).clamp(max=k - 1)
            src_onehot = torch.nn.functional.one_hot(src_bins, k).double().numpy()
            tgt_onehot = torch.nn.functional.one_hot(tgt_bins, k).double().numpy()
            w_np, lse = estimate_label_shift_weights(src_onehot, tgt_onehot)
            # NexCP weights must align with SOURCE calibration points:
            # w(x_i) = sum_k p_s(k|x_i) * pi_t(k)/pi_s(k).
            ratios = np.clip(
                np.asarray(lse.target_prior) / np.asarray(lse.source_prior), 1e-12, 20.0
            )
            w_src = (src_onehot * ratios[None, :]).sum(axis=1)
            w = torch.as_tensor(w_src, device=F_src.device, dtype=F_src.dtype)
            note["em_target_prior"] = np.asarray(lse.target_prior).tolist()
        return w, note

    @staticmethod
    def _head_params(model: Any) -> list[torch.nn.Parameter]:
        head = getattr(model, "head", None)
        params = (
            [p for p in head.parameters()]
            if isinstance(head, torch.nn.Module)
            else list(model.parameters())[-2:]
        )
        if not params:
            raise RuntimeError("no trainable parameters discovered for adaptation")
        return params

    def _pseudo_label_filter(self, var_tgt: Tensor, var_src: Tensor) -> Tensor:
        """Keep the tightest-predictive fraction of target points.

        ponytail: the plan's literal 'conformal residual' vanishes identically
        for self-generated pseudo-labels (score(y_hat | x) = 0), so the filter
        uses predictive log-variance instead — the standard heteroscedastic
        confidence filter. Ceiling: ignores input-density; upgrade path is an
        OT-based residual proxy on aligned features.
        """
        u_tgt = var_tgt.log().sum(dim=-1)
        u_src = var_src.log().sum(dim=-1)
        tau = torch.quantile(u_src, self.pseudo_label_quantile)
        return u_tgt <= tau

    def _finetune_head(
        self,
        model: Any,
        X_tgt: Tensor,
        y_pl: Tensor,
        var_pl: Tensor,
        params: list[torch.nn.Parameter],
        diagnostics: Dict[str, Any],
    ) -> int:
        """Gaussian-NLL self-training toward frozen pseudo-label moments.

        ponytail: plain MSE toward own predictions has exactly-zero gradient
        at init (target == output); the log-var term of the NLL carries the
        actual adaptation signal (variance sharpening on kept points).
        Upgrade path: augmentation-consistency targets.
        """
        opt = torch.optim.Adam(params, lr=1e-4)
        best = float("inf")
        plateau = 0
        steps_run = 0
        was_training = getattr(model, "training", False)
        model.eval()  # BN/dropout frozen during TTA updates
        for step in range(100):
            opt.zero_grad(set_to_none=True)
            mu, var = gaussian_outputs(model, X_tgt)
            nll = 0.5 * (torch.log(var) + (mu - y_pl) ** 2 / var_pl.clamp_min(1e-6))
            loss = nll.mean()
            val = float(loss.detach())
            loss.backward()
            opt.step()
            steps_run = step + 1
            if val < best - 1e-6:
                best, plateau = val, 0
            else:
                plateau += 1
                if plateau >= 10:
                    break
        if was_training:
            model.train()
        diagnostics["head_finetune_steps"] = steps_run
        return steps_run

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def adapt_and_calibrate(
        self,
        model: Any,
        X_cal_src: Tensor,
        y_cal_src: Tensor,
        X_target_unlabeled: Tensor,
    ) -> JointTTAResult:
        """Run the safe-ordering adapt-then-recalibrate protocol.

        The input ``model`` is adapted in place; the returned result carries
        it together with the calibrated joint-marginal region.
        """
        diagnostics: Dict[str, Any] = {}
        was_training = getattr(model, "training", False)
        if was_training:
            model.eval()

        feats_src = self._features(model, X_cal_src)
        F_src = X_cal_src if feats_src is None else feats_src
        feats_tgt = self._features(model, X_target_unlabeled)
        F_tgt_raw = X_target_unlabeled if feats_tgt is None else feats_tgt
        F_tgt = F_tgt_raw
        if self.align_features and hasattr(model, "feature_extractor"):
            try:
                aligner = WeightedSubspaceMomentAligner().fit(
                    F_src.detach().cpu().numpy(),
                    y_cal_src.detach().cpu().numpy().reshape(-1),
                )
                aligned = aligner.transform(F_tgt_raw.detach().cpu().numpy())
                F_tgt = torch.as_tensor(aligned, device=F_src.device, dtype=F_src.dtype)
                diagnostics["feature_alignment"] = "ssa"
            except Exception as exc:  # alignment must never block calibration
                diagnostics["feature_alignment"] = f"skipped: {exc}"
        else:
            diagnostics["feature_alignment"] = "skipped: no feature_extractor"

        w_cal, w_note = self._fit_weights(F_src, F_tgt, X_target_unlabeled, model, X_cal_src)
        diagnostics["weights"] = w_note
        w_norm = w_cal / w_cal.sum().clamp_min(torch.finfo(w_cal.dtype).tiny)
        ess = float(1.0 / (w_norm * w_norm).sum())
        diagnostics["weight_ess"] = ess
        diagnostics["weight_max"] = float(w_norm.max())

        if self.pseudo_label_rounds > 0:
            params = self._head_params(model)
            diagnostics["n_head_params"] = sum(p.numel() for p in params)
            kept_total = 0
            n_tgt = int(X_target_unlabeled.shape[0])
            for _ in range(self.pseudo_label_rounds):
                with torch.no_grad():
                    mu_t, var_t = gaussian_outputs(model, X_target_unlabeled)
                    _, var_s = gaussian_outputs(model, X_cal_src)
                keep = self._pseudo_label_filter(var_t, var_s)
                if int(keep.sum()) < 2:
                    diagnostics["pseudo_label_note"] = "filter kept <2 points; round skipped"
                    break
                y_pl = mu_t[keep]
                var_pl = var_t[keep]
                self._finetune_head(
                    model, X_target_unlabeled[keep], y_pl, var_pl, params, diagnostics
                )
                kept_total += int(keep.sum())
            diagnostics["pseudo_label_yield"] = kept_total / max(
                1, self.pseudo_label_rounds * n_tgt
            )

        # Safe ordering: weights fixed above; NOW recompute scores with the
        # (possibly adapted) frozen model and calibrate last.
        with torch.no_grad():
            mu_c, var_c = gaussian_outputs(model, X_cal_src)
        conformal = MultivariateScoreConformal(self.alpha).calibrate(mu_c, var_c, y_cal_src, w_cal)
        tv_gap = float(0.5 * (w_norm - 1.0 / float(w_norm.numel())).abs().sum())
        diagnostics["coverage_bounds"] = (
            1.0 - self.alpha - 2.0 * tv_gap,
            1.0 - self.alpha + 2.0 * tv_gap + float(w_norm.max()),
        )

        if was_training:
            model.train()
        return JointTTAResult(adapted_model=model, conformal=conformal, diagnostics=diagnostics)

    def predict_intervals(self, result: JointTTAResult, X_test: Tensor) -> Dict[str, Tensor]:
        """Marginal per-target intervals from the joint ellipsoidal region.

        Returns ``{"mean", "lower", "upper", "radius"}``: target dimension d is
        covered by ``mean_d ± sqrt(radius * var_d)`` — the axis-aligned
        projection of the ellipsoid ``{y : score <= radius}``.
        """
        model = result.adapted_model
        was_training = getattr(model, "training", False)
        if was_training:
            model.eval()
        with torch.no_grad():
            mu, var = gaussian_outputs(model, X_test)
        if was_training:
            model.train()
        r = result.conformal.region_radius()
        half_width = torch.sqrt(r * var.clamp_min(_VAR_EPS))
        return {
            "mean": mu,
            "lower": mu - half_width,
            "upper": mu + half_width,
            "radius": torch.tensor(r, device=mu.device),
        }
