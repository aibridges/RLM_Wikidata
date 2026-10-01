# Running the harness

This guide sets up the harness on any machine and runs it against any
OpenAI-compatible endpoint. Run the commands from the repository root, in bash.

## 1. Requirements

| Item | Requirement |
| :---- | :---- |
| Python | 3.13 |
| Package manager | [uv](https://docs.astral.sh/uv/) (recommended) or pip |
| Graph | A Wikidata graph in the format described in [graph-format.md](graph-format.md) |
| Model | Any OpenAI-compatible chat completions endpoint: a local server or a hosted API |

The harness itself needs no GPU. Only a locally served model does.

## 2. Install

```bash
uv sync
```

Without uv, install the dependencies listed in `pyproject.toml`:

```bash
python3.13 -m venv .venv
.venv/bin/pip install numpy openai pyarrow tantivy tenacity
```

## 3. Point the harness at the graph

```bash
export WIKIDATA_GRAPH_DIR=/path/to/graph
```

Without this variable, the harness looks for the graph in `data/graph/`. The
graph is opened read-only and memory-mapped, so several runs can share one copy.

## 4. Serve a model

The harness sends chat completion requests to `BASE_URL` and asks for
`MODEL_NAME`. Sub-calls (`llm_query`, `llm_map`) go to the same endpoint, with
the same model unless `SUBQ_MODEL_NAME` names another one.

### A local model with vLLM

This serves Qwen3.8-27B in FP8, the model of the evaluation in the README, on
one 80 GB GPU:

```bash
vllm serve unsloth/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

| Flag | Why |
| :---- | :---- |
| `--language-model-only` | The model is vision-language; the harness is text-only, so the vision encoder's memory goes to the KV cache |
| `--reasoning-parser qwen3` | Keeps the thinking out of the reply content. Required when thinking is on (`RLM_ENABLE_THINKING=1`) |
| `--speculative-config` | Multi-token prediction, for faster decoding. Optional |
| `--max-num-seqs` | How many requests the server runs at once. Match it to the number of workers you use |

Qwen3.8 needs vLLM 0.17.0 or later with transformers 5.8.0 or later. Model
loading and compilation take several minutes. Before a batch, check that the
server can generate, not only that it answers `/health`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B-FP8", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}'
```

Then point the harness at it:

```bash
export BASE_URL=http://127.0.0.1:8000/v1 API_KEY=EMPTY MODEL_NAME=Qwen3.8-27B-FP8
```

A local vLLM server accepts any non-empty `API_KEY`.

### A hosted API

Set the provider's endpoint, key and model name:

```bash
export BASE_URL=https://api.example.com/v1 API_KEY=<your key> MODEL_NAME=<model>
```

Most hosted APIs need some adjustments to the reference settings. Set them after loading the settings in step 5:

| Variable | Value | Why |
| :---- | :---- | :---- |
| `RLM_TEMPLATE_KWARGS` | `0` | Do not send vLLM's `chat_template_kwargs`, which some APIs reject |
| `RLM_ENABLE_THINKING` | `0` | The thinking switch is a vLLM template option. A hosted reasoning model reasons on its own; `RLM_REASONING_EFFORT` still sets its effort |
| `RLM_MAX_TOKENS_PARAM` | `max_completion_tokens` | Only for APIs that refuse `max_tokens`, such as OpenAI's recent models |
| `RLM_TOP_K` | empty | `top_k` is a vLLM extension that many hosted APIs reject. Set `RLM_TOP_K=` after loading the settings |

## 5. Load the settings

`config/runs/reference.env` holds the settings used for the evaluation. Export
them, then override any one of them in the shell:

```bash
set -a; source config/runs/reference.env; set +a
export RLM_TEMPERATURE=0.6        # optional: change one setting
```

| Variable | Reference | Effect |
| :---- | :---- | :---- |
| `RLM_ENABLE_THINKING` | `1` | Thinking on or off (vLLM template option) |
| `RLM_REASONING_EFFORT` | `low` | Reasoning effort sent with every request |
| `RLM_TEMPERATURE` | `1` | Sampling temperature |
| `RLM_TOP_P` | `0.95` | Nucleus sampling |
| `RLM_TOP_K` | `20` | Top-k sampling |
| `RLM_MAX_TOKENS` | `8192` | Output token budget per request, thinking included |
| `RLM_LLM_MAP` | `1` | Makes `llm_map` available to the model |
| `RLM_SEARCH_HINTS` | `1` | Explains an empty name search: which words matched nothing |
| `SUBQ_LIMIT` | `80` | Sub-call cap for a whole run |
| `RLM_PERSIST` | `1` | Adds "The question has an answer. Do not stop until you find one." and, on multi-hop questions, a note that the answer takes many steps |
| `RLM_TURNS` | `100` | Turn limit, replacing each task's own limit |
| `RLM_TRUNCATE` | `10000` | Characters of a cell's output that the model sees |
| `RLM_STOP` | `0` | No stop sequence. A reply with more than one code block is refused instead |

Every run records the settings it used under `generation` in `metadata.json`.

## 6. Run one question

```bash
.venv/bin/python scripts/rlm_loop.py \
  --task-file tasks/eval/TR_MH_010_Q1411902.json \
  --batch one-question
```

The run is archived under `results/runs/<YYYYMMDD-HHMM>_one-question/`.

## 7. Run the evaluation

```bash
.venv/bin/python scripts/bench/run_batch.py --families eval --workers 1 --tag eval
```

- `--families eval` runs every task in `tasks/eval/`. `--task-files` runs a
  chosen list instead, and `--limit N` only the first N.
- `--workers` sets how many questions run at once. With a local server, keep it
  at or below `--max-num-seqs`. Running several questions at once finishes the
  batch sooner, but each question takes longer.
- Each question runs in its own process, so a failed request costs one question,
  not the batch.

The batch directory is `results/runs/<YYYYMMDD-HHMM>_eval/`. At the end, the
runner scores every run and writes `scores.json`. For a summary:

```bash
.venv/bin/python scripts/bench/summary.py results/runs/<batch>
```

To score run directories again, one line per run:

```bash
.venv/bin/python scripts/bench/score.py results/runs/<batch>/*/
```

Scoring is exact and needs no model: the answer must match the reference after
canonicalisation (`scripts/bench/canonical.py`).

## 8. What a run keeps

Each run directory holds the whole trajectory:

| File | Contents |
| :---- | :---- |
| `question.txt` | The question |
| `system_prompt.txt` | The system prompt the model received |
| `messages.json` | The conversation: model replies and REPL outputs |
| `reasoning.json` | The model's thinking, per turn, when the endpoint returns it |
| `iterations.jsonl` | One line per turn: model time, execution time, token usage, reasoning, code, full output, sub-calls and graph reads |
| `read_log.json` | Every graph function call, with the identifiers it returned |
| `sub_queries.json` | Every sub-call: the text sent, the instruction and the answer (only when the run made sub-calls) |
| `answer.md` | The final answer |
| `metadata.json` | Status, turns, tokens, wall time and the settings used |

A batch directory adds `batch.json` (the task list, settings and code revision),
`scores.json` and `failures.json`.
