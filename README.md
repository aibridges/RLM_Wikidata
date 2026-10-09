# RLM_Wikidata

Built by [**Pleias**](https://pleias.fr) as part of the [**AI-BRIDGES**](https://ai-bridges.org) project at the University of London, with funding from Wikimedia Switzerland and in collaboration with Wikimedia Deutschland, **RLM_Wikidata** is a Recursive Language Model (RLM) harness over a frozen Wikidata graph. The model answers a question by writing Python in a REPL: it searches entities, follows relations, reads qualifiers and references, keeps intermediate results in variables, and returns an answer checked exactly against the graph.

Wikidata describes more than 100 million entities in over 300 languages, and AI systems have no good way to explore it. SPARQL takes expertise and pasting graph data into a context window degrades as the question grows. Recursive Language Models (RLMs; Zhang, Kraska & Khattab, 2026) fit this task better: the model explores the graph in code and calls itself on the parts that matter.

This harness generated **[Wikidata-Search-Traces](https://huggingface.co/datasets/AI-BRIDGES/wikidata-search-traces)**, an open corpus of 10,235 reasoning trajectories over Wikidata, available on Hugging Face.

## The harness

- **One cell per turn.** The model replies with one Python block. The harness runs it in a persistent namespace, so variables carry over between turns, and returns the printed output, truncated to 10,000 characters. The model ends the run with `FINAL(value)`.
- **Graph functions.** The model reads the graph only through 13 functions: `search_entity`, `search_property`, `claims`, `references`, `describe`, `labels`, `label`, `descriptions`, `edges`, `count_edges`, `degree`, `name` and `has`. Every call is logged.
- **Sub-calls.** `llm_query` and `llm_map` send text to a second instance of the model, for semantic judgements over material the run has already read. Sub-calls have no graph access.
- **Grounding checks.** The harness rejects a final answer whose shape does not match the answer format, or that names an entity the run never read.
- **Full archive.** Every run keeps the messages, reasoning, executed code, graph reads, sub-calls, timing and the final answer.
- **Exact scoring.** Every answer is an entity (QID), a year, a quantity, a number, a string or a list of entities, so a deterministic scorer compares it with the reference answer. No language model judges correctness.

We worked over a frozen Wikidata graph. The harness works with any OpenAI-compatible endpoint; the reference configuration serves Qwen3.8-27B-FP8 with vLLM on one GPU.

## Evaluation

We evaluated four models on 100 questions built with the same pipeline as the corpus: 50 single-entity and 50 multi-hop, in [`tasks/eval/`](tasks/eval/), scored by exact match with no judge. Each model answers either through this harness or as a tool-calling agent that calls the same 13 graph functions directly. With the model held fixed, the harness improves both models we ran under both interfaces: gpt-6-luna from 49 to 61 correct answers and Qwen3.8-27B from 60 to 74. The technical report gives the full protocol and analysis.

### Systems

| System | Model | Setup |
| :---- | :---- | :---- |
| Qwen3.8-27B + RLM | Qwen3.8-27B (FP8, served with vLLM on one H100) | RLM harness |
| gpt-6-luna + RLM | gpt-6-luna | RLM harness |
| Qwen3.8-27B agent | Qwen3.8-27B (FP8, served with vLLM on one H100) | tool-calling agent |
| glm-5.3-flash agent | glm-5.3-flash | tool-calling agent |
| gpt-6-luna agent | gpt-6-luna | tool-calling agent |
| gemini-3.1-flash-lite agent | gemini-3.1-flash-lite | tool-calling agent |

- **Shared:** every system uses the same 13 graph functions over the same frozen graph, with low reasoning effort. Every system gets the same instruction: "The question has an answer. Do not stop until you find one."
- **RLM harness:** the model writes Python in a REPL and keeps intermediate results in variables. It can call itself on sub-problems. Temperature 1, up to 100 turns and 80 sub-calls maximum.
- **Tool-calling agents:** openai-agents with LiteLLM. They call the graph functions directly, with tool output truncated at 10,000 characters, up to 100 calls and $0.50 per question maximum.

### Results

| Model | Interface | Single-entity | Multi-hop | Total | Cost / question | Tokens / question (median) | Seconds / question (median) |
| :---- | :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Qwen3.8-27B | RLM harness | 46/50 | 28/50 | 74/100 | $0.04 ¹ | 35,754 | 25 ² |
| gpt-6-luna | RLM harness | 42/50 | 19/50 | 61/100 | $0.013 | 24,942 | 18 |
| Qwen3.8-27B | tool-calling agent | 40/50 | 20/50 | 60/100 | $0.025 ¹ | 32,072 | 17 ² |
| glm-5.3-flash | tool-calling agent | 37/50 | 13/50 | 50/100 | $0.067 | 30,478 | 33 |
| gpt-6-luna | tool-calling agent | 40/50 | 9/50 | 49/100 | $0.005 | 24,924 | 16 |
| gemini-3.1-flash-lite | tool-calling agent | 31/50 | 10/50 | 41/100 | $0.19 | 460,426 | 104 |

For the API models, cost is input tokens × input price + output tokens × output price, with prices from LiteLLM's model price table; reasoning tokens count as output tokens.

¹ Estimated cost of batched serving on one H100 in Google Colab.
² Qwen3.8-27B served with vLLM on one H100, processing one question at a time.

## Repository

| Path | Contents |
| :---- | :---- |
| `scripts/rlm_loop.py` | The harness: runs one question and archives the trajectory |
| `scripts/wd_graph_env.py` | The graph environment: the functions the model calls |
| `scripts/bench/` | Batch runner, scorer and summary |
| `tasks/eval/` | The 100 evaluation questions with their reference answers |
| `config/runs/reference.env` | The settings used for the evaluation |
| `docs/running.md` | How to set up and run the harness |
| `docs/graph-format.md` | The graph files the environment reads |

## Running the harness

The harness needs Python 3.13, a Wikidata graph in the format described in [docs/graph-format.md](docs/graph-format.md), and any OpenAI-compatible endpoint: a local server such as vLLM, or a hosted API. [docs/running.md](docs/running.md) covers installation, serving a model, the settings, and how to run and score the evaluation.

```bash
uv sync
set -a; source config/runs/reference.env; set +a
export WIKIDATA_GRAPH_DIR=/path/to/graph
export BASE_URL=http://127.0.0.1:8000/v1 API_KEY=EMPTY MODEL_NAME=<served model name>

.venv/bin/python scripts/bench/run_batch.py --families eval --workers 1 --tag eval
.venv/bin/python scripts/bench/summary.py results/runs/<batch>
```
## What you can do with this

Everything here runs on open weights and modest hardware, so it can be used, inspected and adapted rather than only read about.

**Ask Wikidata complex questions without SPARQL** 
Point the harness at the graph
and ask in plain language. Answers come back traced to the statements, qualifiers
and references the run actually read, so you can check them. With Qwen3.8-27B on a
single GPU, an institution can run the whole stack in-house: no commercial API, no
data leaving your infrastructure, around $0.04 per question.

**Link your collection to Wikidata** 
Archives, libraries and museums can use the
same functions to match people, places and works in their own records to Wikidata
entities, a task usually done by hand.

**Build tooling for editors** 
The functions that answer questions also let a model
walk around an entity, read its references and compare neighbouring statements.
That makes this a starting point for tools that surface missing sources or
cross-language inconsistencies for human review. Nothing here edits Wikidata; the
agent reads and cites, people decide.

**Train and evaluate models** 
The 10,235 traces are CC0 and ready for fine-tuning.
The construction pipeline grows new certified questions for other domains,
languages or answer types, and any new model can be run in this harness and
compared against the table above.

**Note on scope** 
The harness currently runs against a frozen February 2026
snapshot. Connecting it to the live Wikidata API is next on our list.

## How to get involved 
We need your help with questions!
AI-BRIDGES is collecting the questions people working in galleries, libraries, archives, museums and the Humanities would ask Wikidata if the technical barrier disappeared. They will heop shape how the next phase is evaluated. You can share them via [this doc](https://docs.google.com/document/d/1ipIzwSwf-xCtU7R7ewNCCND7vk0yha6byWMNs3OyIEc/edit?tab=t.0) or send them via email to: contact@ai-bridges.org.

## About AI-BRIDGES
[AI-BRIDGES](https://ai-bridges.org) is a research project at the Digital Humanities Research Hub, School of Advanced Study, University of London, directed by Dr. Shani Evenstein Sigalov. It works on the connection between institutional data, Linked Open Data platforms (such as Wikidata & Wikibase) and AI systems, building methods and models that let cultural, scholarly and community knowledge be used well by AI, rather than flattened by it. 
This work is the first phase of its technical development strand.
AI-BRIDGES is funded by the European Commission through a Marie Skłodowska-Curie Postdoctoral Research Fellowship (Grant ID 101203096). This phase was funded by Wikimedia Switzerland (WMCH), delivered by Pleias, and carried out in close
collaboration with Wikimedia Deutschland (WMDE). Compute was provided through grant n°AD011014736R2 on the Jean Zay cluster.
