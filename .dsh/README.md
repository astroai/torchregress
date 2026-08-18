# torchregress × DeepSeek Harness (dsh)

Repo-local [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
configuration. dsh auto-discovers skills from `.dsh/skills/` at the project
root (top project priority).

> dsh is a **developer preview** — expect breaking changes. Requires a DeepSeek
> API key, or any OpenAI-compatible endpoint via `DEEPSEEK_BASE_URL`.

## Run

Web UI (needs localhost):

```bash
npx @deepseek-ai/dsh web --patch .dsh/cordis.patch.yml
```

Headless one-shot:

```bash
export DEEPSEEK_API_KEY=sk-...
npx @deepseek-ai/dsh --profile headless --patch .dsh/cordis.patch.yml \
  "Audit conformal/CQR coverage guarantees against empirical coverage in the test suite"
```

Environment knobs:

- `DEEPSEEK_API_KEY` — required.
- `DEEPSEEK_BASE_URL` — OpenAI-compatible proxy.
- `DSH_PERMISSION_MODE=danger-full-access` — for one-shot headless runs on a
  disposable VM (default `workspace-write` + ask). Never on a shared machine.
- `DSH_TOOLS_MODE=code` — Code Mode.

## Skills

- `science-core` — shared correctness / math / statistics / data-driven / science-impact discipline
- `torchregress-dev` — capabilities, repo boundaries (research/harness), gates

## Notes

- Keep paper manuscripts / NeurIPS / SAGE/SPT work in `torchregress-research`,
  and external-software parity in `torchregress-harness` — never here.
- Sessions are append-only logs (provenance); reuse a session id to continue a
  durable conversation with persistent bash state.
- `cordis.patch.yml`: 10-minute bash timeout + durable full-text session search.
