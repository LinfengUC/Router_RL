# RL Router (base / IRT-feature / IRT-hard-gate)

This repo is about trainning a RL agent in 3 modes:

- **base**: RL router with prompt+answer embeddings
- **irt**: base router + IRT prediction features appended to the observation
- **hard_gate**: base router + IRT “hard gate” (disallow calling a model when IRT predict 0)

**Important:** the default preset is a **smoke test** (fast).  
To get results as expectation, use `--preset full`.

## 1) Setup pip + virtualenv

```bash
python -m venv .router_rl
# macOS/Linux
source .router_rl/bin/activate
# Windows (PowerShell)
# .router_rl\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- PPO is run on **CPU** by default (`--ppo-device cpu`) because SB3 can emit GPU warnings.
- The text embedder can use **GPU** to speed up embedding (`--embedder-device cuda`).
- First run will download the HuggingFace embedding model (`sentence-transformers/all-mpnet-base-v2`).

## 2) Data layout (already normalized)

The `data/` directory is already in a consistent format:

```
data/
  labels/
    train.csv
    test.csv
  irt/
    train.csv
    test.csv
  responses/
    train/
      <model_name>.csv
    test/
      <model_name>.csv
```

**Labels columns**
- `prompt_id` (string)
- `prompt`
- `model_name`
- `label` (0/1)

**IRT columns**
- `prompt_id`
- `model_name`
- `s_bert_pred` (0/1)
- `m_bert_pred` (0/1)

**Responses columns**
- `prompt_id`
- `prompt`
- `output`
- `reasoning` (empty for non-reasoning models)

## 3) Run (single CLI)

### Smoke test (default)
Fast run to verify everything is wired correctly:

```bash
python run.py --mode base --preset smoke --profiles accuracy_first
```

You can also try the other variants:

```bash
python run.py --mode irt --preset smoke --profiles accuracy_first
python run.py --mode hard_gate --preset smoke --profiles accuracy_first
```

### Full experiment (expected-quality results)
This reproduces the original scale:
- `timesteps=300_000`
- `eval_freq=5_000`
- `n_eval_episodes=100`
- `n_runs=10` seeds per reward profile

```bash
python run.py --mode base --preset full --profiles all
```

Same for other modes:

```bash
python run.py --mode irt --preset full --profiles all
python run.py --mode hard_gate --preset full --profiles all
```

## 4) Outputs

Runs write to `outputs/<mode>/<profile>/<run_id>/`, including:

- `latest_model.zip` (final checkpoint)
- `best/best_model.zip` (best checkpoint by `mean_reward`)
- `run_config.json` (all settings, model list, seed, etc.)
- `metrics.jsonl` and `metrics.csv` (one row per evaluation)

Metrics include:
- `accuracy`
- `mean_reward`
- `mean_cost` (USD per question)
- `mean_latency` (seconds per question)
- `mean_calls` (avg # model calls per episode)
- per-model final usage rates (`final_usage__<model_name>`)

## 5) Use a trained router for prediction (routing)

Training produces a policy checkpoint (e.g. `latest_model.zip`) plus a
`run_config.json` that records the model list/order and embedding settings.

### Input format

For routing, you need a CSV with at least:

- `prompt_id`
- `prompt`

Example:

```
prompt_id,prompt
0,"What is the capital of France?"
1,"Solve: 17*23"
```

### IRT inputs

For `--mode irt` or `--mode hard_gate`, you may also provide IRT predictions as a CSV
with columns:

- `prompt_id`
- `model_name`
- `s_bert_pred` (0/1) **optional**
- `m_bert_pred` (0/1) **optional**

Important practical detail:

- If only one IRT model exists / reports (e.g. only `s_bert_pred`), the code will
  **copy that value into the missing column**
- If both are missing for a row, we default to `(1,1)` (“allowed”) to avoid hard failures.

### Demo prediction (offline, using stored responses)

This repo includes a **demo-only** predictor that uses `data/responses/<split>/...` as a
stand-in for real model endpoints:

```bash
python run.py predict \
  --checkpoint outputs/<mode>/<profile>/<run_id>/latest_model.zip \
  --input data/labels/test.csv \
  --output outputs/predictions.csv \
  --use-stored-responses --split test \
  --deterministic
```

Notes:
- The input can be `data/labels/test.csv` (it contains repeated prompt_id rows per model);
  predict mode deduplicates by `prompt_id`.
- For `irt` / `hard_gate`, by default we auto-use `data/irt/<split>.csv` if present.

The output CSV contains:
- chosen model
- #calls
- estimated cost/latency
- `action_trace` (actions taken; `N` means ACCEPT where `N=len(models)`)

### Production routing (online)

In a real system you do **not** have stored responses. Instead you:

1) call the `base_model_name` once to get an initial answer
2) give `(prompt, current_output, optional IRT)` to the policy
3) if the policy says “switch”, call that model, update `current_output`, and repeat

The core function is:

```python
from router.infer import route_prompt, IRTProvider, ModelCaller
```

See `router/infer.py` for:
- `DummySingleIRTProvider` (simulates the “only one IRT model exists” scenario)
- `StoredResponseCaller` (demo-only)

Minimal example (pseudo-production):

```python
from stable_baselines3 import PPO
from router.embedder import TextEmbedder
from router.infer import route_prompt, ModelCaller, IRTProvider


class MyModelCaller(ModelCaller):
    def call(self, *, model_name: str, prompt_id: str, prompt: str) -> str:
        # Replace with your real API call.
        # return requests.post(...).json()["output"]
        raise NotImplementedError


class MyIRTProvider(IRTProvider):
    def predict(self, *, model_name: str, prompt_id: str, prompt: str):
        # Return (s_bert_pred, m_bert_pred). If you only have one, return (pred, None)
        # and the router will mirror it.
        return (1, None)


sb3_model = PPO.load("/path/to/latest_model.zip", device="cpu")
embedder = TextEmbedder(device="cuda")
model_names = [...]  # must match training order (from run_config.json)

result = route_prompt(
    sb3_model=sb3_model,
    mode="hard_gate",
    embedder=embedder,
    model_names=model_names,
    model_caller=MyModelCaller(),
    prompt_id="123",
    prompt="Explain the Doppler effect",
    irt_provider=MyIRTProvider(),
)
print(result)
```

## 6) Adding a new candidate model

To add a new routable model `<model_name>`, you must provide:

1) **Responses** (required)
   - `data/responses/train/<model_name>.csv`
   - `data/responses/test/<model_name>.csv`
   - columns: `prompt_id, prompt, output, reasoning`
     - if you don’t have reasoning, use empty strings

2) **Labels** (required for training/eval)
   - add rows for `<model_name>` to:
     - `data/labels/train.csv`
     - `data/labels/test.csv`
   - columns: `prompt_id, prompt, model_name, label`

3) **Cost + latency config** (required)
   - add an entry to `router/config.py :: MODEL_COSTS`:
     - `cost_per_100` = average **USD per 100 questions**
     - `latency` = average seconds per call

4) **IRT predictions** (optional)
   - add rows for `<model_name>` to:
     - `data/irt/train.csv`
     - `data/irt/test.csv`
   - columns: `prompt_id, model_name, s_bert_pred?, m_bert_pred?`
   - If you only have one IRT signal, supply just that column; the loader mirrors it.

## 7) CLI reference

```bash
python run.py train --help
python run.py predict --help
```

Common overrides (useful when experimenting):

```bash
python run.py --mode irt --preset smoke --profiles all \
  --timesteps 50000 --eval-freq 5000 --n-eval-episodes 50 --runs 3
```

Device knobs:

```bash
python run.py --embedder-device cuda   # faster embeddings if you have a GPU
python run.py --ppo-device cpu         # recommended default
```
