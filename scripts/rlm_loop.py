"""Run one JSON task through the Wikidata RLM harness.

The task file owns the question, available environment functions and
answer contract. The harness owns one stable execution protocol: the model
writes one Python block per turn, code runs in a persistent namespace, and
FINAL()/FINAL_VAR() ends the run.

Usage:
    python scripts/rlm_loop.py --task-file tasks/eval/TR_SE_....json
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wd_graph_env import WDGraphEnv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ROOT / "results" / "runs"
TRUNCATE = int(os.environ.get("RLM_TRUNCATE", "10000"))   # stdout chars the model sees per turn
# Sub-queries interpret text the root model already fetched. They use the
# served model unless SUBQ_MODEL_NAME names another one.
SUBQ_MODEL = os.environ.get("SUBQ_MODEL_NAME") or os.environ.get("MODEL_NAME", "")
# The cap stops a run that calls the helper once per item instead of batching;
# the model then receives a visible error rather than an unbounded cost.
SUBQ_LIMIT = int(os.environ.get("SUBQ_LIMIT", "20"))

# Generation settings, from the environment. The defaults are conservative
# (thinking off, temperature 0.1, 8192 output tokens). The published results use
# the values in config/runs/reference.env; export them before Python starts.
# Every run records the values it used under `generation` in metadata.json.
ENABLE_THINKING = os.environ.get("RLM_ENABLE_THINKING", "0") == "1"
REASONING_EFFORT = os.environ.get("RLM_REASONING_EFFORT") or None
MAX_TOKENS = int(os.environ.get("RLM_MAX_TOKENS", "8192"))
# OpenAI's recent models refuse max_tokens and want max_completion_tokens.
MAX_TOKENS_PARAM = os.environ.get("RLM_MAX_TOKENS_PARAM", "max_tokens")
TEMPERATURE = float(os.environ.get("RLM_TEMPERATURE", "0.1"))
TOP_P = os.environ.get("RLM_TOP_P") or None
TOP_K = os.environ.get("RLM_TOP_K") or None
# vLLM reads thinking on/off from chat_template_kwargs. Set this to 0 for an
# endpoint that rejects the unknown field.
TEMPLATE_KWARGS = os.environ.get("RLM_TEMPLATE_KWARGS", "1") == "1"

SUBQ_SYSTEM = (
    "You are a helper inside an RLM program. Reason only from the data supplied "
    "by the caller, follow its requested output format, and do not invent "
    "Wikidata identifiers."
)

CORE_PROMPT = """You are a Recursive Language Model (RLM) agent exploring a frozen Wikidata snapshot through Python functions in a persistent REPL.

## Interaction limit

{max_iters} turns total. Graph reads are measured but not capped. llm_query is
capped at {max_subq} calls for the whole run.

{function_docs}

## Rules

1. ONE ```python block per turn. Not two, not a plan cell followed by a code
   cell — ONE. Put all your reasoning in prose BEFORE the block, then write a
   single block that does the whole turn's work. A reply containing more than
   one block is rejected and the turn is wasted.
2. Keep data in variables and print only useful summaries. Output is truncated
   to the last {truncate} characters, but variables persist across turns.
3. Every final fact must come from a function call made during this run. Do not
   invent identifiers or rely on remembered Wikidata facts.
4. Reuse intermediate results instead of fetching the same data again.
5. llm_query is for judgement Python cannot make. BATCH: pass many items in one
   call and ask for one answer per item. One call per item exhausts the budget
   long before the task is done — about one call per ten items is the target.
6. Use FINAL(value) or FINAL_VAR(value) to finish.
"""

# Added to the system prompt when RLM_PERSIST=1.
HAS_ANSWER = "The question has an answer. Do not stop until you find one."
# Multi-hop questions only: keeps the model from giving up after a few calls.
MULTI_HOP_EFFORT = """## This question takes many steps

The entity it asks about is described only through a chain of other, unnamed
entities. Finding it takes many calls: work through the chain one link at a
time, and do not hesitate to make several calls in a turn or dozens in a run.
Keep going until the graph confirms the answer; do not stop early, and do not
answer with an empty list or a guess.

"""
PERSIST = os.environ.get("RLM_PERSIST", "0") == "1"
# Replaces every task's turn limit when set: the agents get 100 model calls,
# and one harness turn is one model call.
TURNS = int(os.environ["RLM_TURNS"]) if os.environ.get("RLM_TURNS") else None

NAVIGATION_GUIDE = """
## Graph navigation

Keep frontiers and parent maps in variables, intersect them in Python, and
expand only promising unseen nodes. Never print a complete large frontier.
Background knowledge may rank identifiers returned by the environment, but it
must never generate new identifiers.
"""

FUNCTION_DOCS = {
    "claims": (
        "- claims(qid, pid=None, all_ranks=False) -> LIST of {'property': "
        "'P..', 'property_label':.., 'value':.., 'value_label':.., "
        "'value_type':.., 'rank':.., 'qualifiers': [..]}. Not a dict — it has "
        "no .items(). value_type is one of 'wikibase-entityid' (value is a "
        "QID), 'string', 'time', 'quantity' — those four spellings and no "
        "others. The default keeps preferred statements when present, "
        "otherwise non-deprecated ones; all_ranks=True returns every statement. "
        "This rank filter can hide many values: to enumerate everything a node "
        "LINKS to, use edges(), which applies no rank filter."
    ),
    "references": (
        "- references(qid, pid=None) -> LIST of {'property': 'P..', "
        "'property_label':.., 'value':.., 'sources': [{'property':.., "
        "'property_label':.., 'value':..}]}. Only sourced statements appear; "
        "a statement absent from this list is unsourced."
    ),
    "describe": (
        "- describe(qid) -> {'id': 'Q..', 'instance_of': ['Q..'], 'languages': "
        "{'en': {'label':.., 'description':.., 'aliases':.., 'sitelink':..}, "
        "..}}. Sitelinks are NESTED per language and hold a page title or "
        "None — there is no top-level 'sitelinks' key and no 'enwiki' key."
    ),
    "count_edges": (
        "- count_edges(qid, direction='both', pid=None) -> {'out':N, 'in_':N, "
        "'total':N}: how many edges match, without returning them. Use this "
        "BEFORE a sweep: degree() reports a node's total across every "
        "predicate, which is the wrong number for one predicate. Then page "
        "the sweep with edges(qid, direction=.., pid=.., limit=L, offset=K) "
        "— offset skips the first K MATCHING edges, so successive pages cover "
        "the set exactly once. A pool or 'reviewed' field is the count of the "
        "candidate set the question describes, not the size of your answer."
    ),
    "labels": "- labels(qid) -> labels in the six target languages",
    "descriptions": "- descriptions(qid) -> descriptions in the six target languages",
    "label": "- label(qid, lang) -> one label without language fallback",
    "edges": (
        "- edges(qid, direction='both', pid=None, limit=None) -> outgoing and "
        "incoming graph edges as {'out': [...], 'in_': [...], ...}. Each edge "
        "is (pid, pid_label, other_qid, other_label); 'in_' is the incoming "
        "list. The exact keys are: 'id', 'label', 'deg_out', 'deg_in', 'out', "
        "'in_', 'truncated'. There are no others — a .get() on an invented key "
        "returns your default and silently poisons the result."
    ),
    "degree": "- degree(qid) -> (out_degree, in_degree); free, costs no read",
    "name": "- name(id) -> English label of an entity or property",
    "has": "- has(id) -> whether an identifier exists in the graph",
    "search_entity": (
        "- search_entity(text, limit=10, lang='en') -> (total_matches, "
        "[{'qid':'Q..', 'label':.., 'description':..}]). Finds ENTITIES by "
        "name across labels and aliases in six languages, accents optional. "
        "All words must match. total_matches is how many matched in all, so a "
        "large number means the name is ambiguous and the descriptions are how "
        "you tell them apart. It does NOT find properties. 1 read."
    ),
    "search_property": (
        "- search_property(text, limit=10) -> (total_matches, [{'pid':'P..', "
        "'label':.., 'lang':..}]). Finds a PROPERTY by name in any of the six "
        "languages: exact matches first, then prefixes, then substrings. "
        "1 read."
    ),
}

NAVIGATION_FUNCTIONS = {
    "edges", "degree",
}


class SubQueryBudgetExhausted(RuntimeError):
    """Raised when llm_query is called past the run's whole-run cap."""


class FinalAnswer(Exception):
    def __init__(self, value: str) -> None:
        self.value = value


def load_task(path: Path) -> dict:
    """Load and validate the single input contract accepted by the harness."""
    task = json.loads(path.read_text())
    if not isinstance(task, dict):
        raise TypeError("task file must contain one JSON object")
    missing = [key for key in (
        "id", "question", "taxonomy", "allowed_functions", "limits", "answer"
    )
               if key not in task]
    if missing:
        raise ValueError(f"task file is missing required fields: {', '.join(missing)}")
    if not isinstance(task["id"], str) or not task["id"].strip():
        raise ValueError("task.id must be a non-empty string")
    if not isinstance(task["question"], str) or not task["question"].strip():
        raise ValueError("task.question must be a non-empty string")
    allowed = task["allowed_functions"]
    if not isinstance(allowed, list) or not all(isinstance(name, str) for name in allowed):
        raise ValueError("task.allowed_functions must be a list of function names")
    unknown = sorted(set(allowed) - set(FUNCTION_DOCS))
    if unknown:
        raise ValueError(f"unknown allowed functions: {', '.join(unknown)}")
    taxonomy = task["taxonomy"]
    if not isinstance(taxonomy, dict) or not all(
        isinstance(taxonomy.get(field), str) and taxonomy[field]
        for field in ("category", "type")
    ):
        raise ValueError("task.taxonomy must contain non-empty category and type")
    limits = task["limits"]
    if not isinstance(limits, dict):
        raise TypeError("task.limits must be an object")
    if set(limits) != {"turns"}:
        raise ValueError("task.limits must contain only the turns limit")
    if not isinstance(limits["turns"], int) or limits["turns"] < 1:
        raise ValueError("task.limits.turns must be an integer >= 1")
    answer = task["answer"]
    if not isinstance(answer, dict):
        raise TypeError("task.answer must be an object")
    for field in ("schema", "format"):
        if not isinstance(answer.get(field), str) or not answer[field]:
            raise ValueError(f"task.answer.{field} must be a non-empty string")
    if "expected" not in answer:
        raise ValueError("task.answer.expected is required")
    if "reference_path" in task and not isinstance(task["reference_path"], list):
        raise ValueError("task.reference_path must be a list when provided")
    return task


def render_function_docs(allowed: list[str]) -> str:
    lines = ["## Available in the REPL", ""]
    seen = set()
    for name in allowed:
        doc = FUNCTION_DOCS[name]
        if doc and doc not in seen:
            lines.append(doc)
            seen.add(doc)
    lines.extend([
        "- llm_query(text, instruction) -> ask the same model to interpret or",
        "  summarize supplied data. It cannot access the graph: pass all needed",
        "  evidence in text, batch related items, and validate its output in code.",
    ])
    if LLM_MAP:
        lines.extend([
            "- llm_map({key: text, ...}, instruction) -> {key: answer, ...}:",
            "  judge many items at once. Batches for you and returns one",
            "  validated answer per key, so you filter in code instead of",
            "  reading the items yourself. A key the helper drops comes back",
            "  as 'NO ANSWER' rather than going missing.",
        ])
    lines.append("- FINAL(value) / FINAL_VAR(value) -> terminate the run")
    lines.extend(["", "No other function is available."])
    return "\n".join(lines)


def build_prompt(task: dict) -> str:
    limits = task["limits"]
    max_iters = limits["turns"]
    allowed = task["allowed_functions"]
    prompt = CORE_PROMPT.format(
        max_iters=max_iters,
        max_subq=SUBQ_LIMIT,
        function_docs=render_function_docs(allowed),
        truncate=TRUNCATE,
    )
    if NAVIGATION_FUNCTIONS.intersection(allowed):
        prompt += NAVIGATION_GUIDE
    if PERSIST:
        prompt += f"\n{HAS_ANSWER}\n"
        if task["taxonomy"]["category"] == "multi_hop":
            prompt += "\n" + MULTI_HOP_EFFORT
    prompt += f"\n\n## Answer format\n\n{task['answer']['format']}\n"
    return prompt


def make_client() -> OpenAI:
    base_url = os.environ.get("BASE_URL")
    # A local vLLM server accepts any non-empty value.
    api_key = os.environ.get("API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("BASE_URL and API_KEY must be configured")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=180.0,
        max_retries=0,
        default_headers={"User-Agent": "curl/8.5.0"},
    )


def extra_field(obj: object, name: str, default=None):
    """Read a field the OpenAI SDK does not declare.

    `reasoning_content` and `stop_reason` are vLLM additions. Depending on the
    SDK version they arrive as real attributes or only inside `model_extra`,
    and a plain getattr returns None for the second case without complaining,
    which would archive an empty reasoning trace for a model that is thinking.
    """
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return default if value is None else value


# vLLM 0.27.1 returns the thinking block as `reasoning`; other versions and
# other OpenAI-compatible servers call it `reasoning_content`. Both are tried,
# newest first. The endpoint's message carries ["content", "reasoning",
# "role"]; reading only `reasoning_content` archives nothing.
REASONING_FIELDS = ("reasoning", "reasoning_content")


def reasoning_text(message: object) -> str:
    """The thinking block, under whichever name this server uses."""
    for name in REASONING_FIELDS:
        value = extra_field(message, name, "")
        if value:
            return value
    return ""


def retryable_chat_error(error: BaseException) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    status = getattr(error, "status_code", None)
    return (
        status == 429
        or status >= 500
        or (status == 400 and "ContentPolicyViolation" in str(error))
    ) if isinstance(status, int) else False


def log_retry(retry_state) -> None:
    error = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep
    print(
        f"[chat] upstream error ({error}); retry "
        f"{retry_state.attempt_number + 1}/3 in {wait:.0f}s",
        file=sys.stderr,
    )


# The protocol is one cell per turn, and asking for it in the prompt only ever
# half worked: 41.9% of everything generated came AFTER the first closing
# fence, and eight messages carried a second block that cost the whole turn.
# Stopping generation at the closing fence makes the rule mechanical. The
# OPENING fence is "\n```python\n", so it does not match; the closing one is a
# bare "\n```" followed by a newline or the end of the message.
CELL_END = ["\n```\n", "\n``` "]


def turn_stop() -> list[str] | None:
    """The stop sequence for a root turn — None while thinking is on.

    With thinking enabled, a stop sequence empties the turn. `stop` is matched
    against the whole generated stream, the model
    drafts code inside <think>, and generation therefore ends at the first
    fence it writes there — before any answer exists. The turn arrives empty,
    which in an archive is indistinguishable from a model that said nothing.

    Dropping it is safe only because the reasoning parser splits the stream:
    `content` holds what follows </think>, so a fence written while thinking
    never reaches extract_code. Serving a thinking model WITHOUT
    --reasoning-parser puts the whole <think> block back into `content`, and
    with it every fence — so that combination is not supported. The one-cell
    rule falls back to the multi-block rejection path, which is what enforced
    it before the stop sequence existed.
    """
    # Some endpoints refuse `stop` (OpenAI's recent models); RLM_STOP=0 drops it
    # and a reply with several cells is then refused by the one-cell rule.
    return None if ENABLE_THINKING or os.environ.get("RLM_STOP", "1") == "0" else CELL_END


@retry(
    retry=retry_if_exception(retryable_chat_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=10, min=10, max=20),
    before_sleep=log_retry,
    reraise=True,
)
def chat(client: OpenAI, messages: list[dict], model: str,
         temperature: float | None = None, max_tokens: int | None = None,
         usage_state: dict | None = None, stop: list[str] | None = None,
         kind: str = "root") -> str:
    """Request one completion from the configured endpoint.

    Thinking is off unless RLM_ENABLE_THINKING is set.

    The call's duration, usage and finish reason are left in
    usage_state["last_call"], so callers can attach them to their own record.
    """
    if usage_state is not None:
        usage_state["model_calls"] += 1
    template: dict = {"enable_thinking": ENABLE_THINKING}
    extra: dict = {}
    if TOP_K is not None:
        extra["top_k"] = int(TOP_K)
    call_started = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE if temperature is None else temperature,
        **{MAX_TOKENS_PARAM: MAX_TOKENS if max_tokens is None else max_tokens},
        **({"top_p": float(TOP_P)} if TOP_P is not None else {}),
        # Sent whenever it is set: "none" is how an OpenAI model turns its hidden
        # thinking off, and it must reach the API with thinking off here.
        **({"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}),
        **({"stop": stop} if stop else {}),
        extra_body={**({"chat_template_kwargs": template} if TEMPLATE_KWARGS else {}),
                    **extra},
    )
    call_seconds = time.time() - call_started
    if usage_state is not None:
        usage_state["successful_model_calls"] += 1
        usage = response.usage
        details_p = getattr(usage, "prompt_tokens_details", None)
        details_c = getattr(usage, "completion_tokens_details", None)
        usage_state["last_call"] = {
            "kind": kind,
            "started_at": round(call_started, 3),
            "seconds": round(call_seconds, 3),
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cached_tokens": getattr(details_p, "cached_tokens", None),
            "reasoning_tokens": getattr(details_c, "reasoning_tokens", None),
        }
        if usage is not None:
            usage_state["input_tokens"] += usage.prompt_tokens or 0
            usage_state["output_tokens"] += usage.completion_tokens or 0
            usage_state["total_tokens"] += usage.total_tokens or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            usage_state["cached_tokens"] += (
                getattr(prompt_details, "cached_tokens", 0) or 0
            )
            usage_state["reasoning_tokens"] += (
                getattr(completion_details, "reasoning_tokens", 0) or 0
            )
    choice = response.choices[0]
    content = choice.message.content or ""
    reasoning = reasoning_text(choice.message)
    if usage_state is not None and reasoning:
        usage_state.setdefault("reasoning_log", [])
        # Kept out of `messages` on purpose: a root turn's SFT sample is the
        # history plus what the root EMITTED, and thinking is not part of it.
        # Archived beside the run so a wasted turn can still be explained.
        usage_state["reasoning_log"].append({
            "turn": usage_state.get("iteration"),
            "kind": kind,
            "seconds": round(call_seconds, 3),
            "finish_reason": choice.finish_reason,
            "reasoning": reasoning,
        })
    if usage_state is not None:
        usage_state["last_call"]["reasoning"] = reasoning
    # The stop sequence is matched against the whole generated stream, and the
    # reasoning block is part of it. A fenced draft inside <think> therefore
    # ends the turn before any cell exists: empty content, non-empty thinking,
    # finish_reason 'stop'. Counted rather than silently absorbed.
    if usage_state is not None and stop and reasoning and not content.strip() \
            and choice.finish_reason == "stop":
        usage_state["stopped_in_thinking"] = \
            usage_state.get("stopped_in_thinking", 0) + 1
    if stop and content.count("```") % 2 == 1:
        content += "\n```"          # the stop sequence ate the closing fence
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "</s>"):
        content = content.replace(token, "")
    return content


def extract_code(text: str) -> str | list[str] | None:
    """The single Python cell of one assistant message.

    One cell per turn is the protocol. Returning a LIST instead of a string
    means the message carried several executable cells: the caller rejects the
    turn rather than guessing which one the model meant. Interleaving plan
    cells with code cells lets a model narrate a computation it never ran —
    reasoning belongs in prose before the block.

    Fenced blocks that do not parse as Python are dropped, not counted: some
    models fence an imagined stdout. A trailing ``FINAL(<literal>)`` cell is
    the same artefact and is dropped too, so the common two-cell shape
    "real work, then invented answer" is executed as the one real cell.
    """
    blocks = re.findall(r"```(?:python|repl)?\s*\n(.*?)```", text, re.DOTALL)
    executable: list[tuple[str, ast.Module]] = []
    for block in blocks:
        if not block.strip():
            continue
        try:
            tree = ast.parse(block, "<model-cell>", "exec")
        except SyntaxError:
            # A fenced block of imagined output was never executed by the
            # environment and often is not Python at all
            # (e.g. "Processed 6 cast members.").
            continue
        executable.append((block, tree))
    if len(executable) > 1:
        executable = [item for item in executable
                      if not _literal_final_only(item[1])]
    if len(executable) > 1:
        return [block for block, _ in executable]
    return executable[0][0] if executable else None


def _literal_final_only(tree: ast.Module) -> bool:
    """Whether a cell only submits a fully literal answer."""
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return False
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) \
            or call.func.id not in {"FINAL", "FINAL_VAR"} \
            or len(call.args) != 1 or call.keywords:
        return False
    try:
        ast.literal_eval(call.args[0])
    except (ValueError, TypeError):
        return False
    return True


def serialize_final(value: object) -> str:
    """Preserve structured answers as strict JSON for mechanical scorers."""
    if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def map_over(query, items: object, instruction: str, batch: int = 10) -> dict:
    """Judge many items in batches and return one verdict per key.

    Why this exists beside llm_query: llm_query returns prose, so a root that
    uses it for many items must serialise them into a prompt and parse free
    text back into structure. When that plumbing costs more than the
    judgement, the model reads every item itself and decides from memory
    several turns later -- context rot, inside the corpus whose purpose is to
    prevent it. llm_map sends items in batches and returns one verdict per
    key, so a semantic filter over a large candidate set stays in Python.

    A PRIMITIVE, not a strategy: it does not decide when to delegate or what
    to ask. It removes plumbing -- batching, and one validated verdict per key.
    """
    if isinstance(items, dict):
        keys = [str(k) for k in items]
        values = [items[k] for k in items]
    else:
        listed = list(items)
        keys = [str(i) for i in range(len(listed))]
        values = listed
    if not keys:
        return {}
    out: dict[str, str] = {}
    size = max(1, int(batch))
    for start in range(0, len(keys), size):
        chunk = list(zip(keys[start:start + size], values[start:start + size]))
        listing = "\n".join(f"{k}\t{v}" for k, v in chunk)
        reply = query(
            listing,
            f"{instruction}\n\nThe data is one item per line as "
            f"KEY<TAB>VALUE. Answer with exactly one line per item, formatted "
            f"KEY<TAB>ANSWER, in the same order, and nothing else.")
        seen = {}
        for line in str(reply).splitlines():
            if "\t" not in line:
                continue
            key, _, verdict = line.partition("\t")
            seen[key.strip()] = verdict.strip()
        for key, _ in chunk:
            # A key the helper dropped is reported, never silently absent:
            # a missing verdict must not read as a negative one.
            out[key] = seen.get(key, "NO ANSWER")
    return out


# llm_map is the batched delegation primitive used by the published results, and
# it is on by default. Set RLM_LLM_MAP=0 to remove it from the prompt and the
# REPL namespace.
LLM_MAP = os.environ.get("RLM_LLM_MAP", "1") == "1"


def final(value: object) -> None:
    raise FinalAnswer(serialize_final(value))


def _compile_cell(code: str) -> tuple[object, object | None]:
    """Compile a cell so a TRAILING BARE EXPRESSION yields its value.

    The system prompt calls this a REPL and a REPL echoes the value of a final
    bare expression; exec() discards it. A model that writes
    `search_entity("Ivan Andreyevsky")` instead of wrapping it in print() would
    otherwise get "(no output)" and could not tell that the call had worked.

    FINAL(...) as the last expression still raises FinalAnswer out of eval,
    where the existing handler catches it.
    """
    tree = ast.parse(code, "<repl>", "exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        head = ast.Module(body=tree.body[:-1], type_ignores=[])
        tail = ast.Expression(body=tree.body[-1].value)
        return compile(head, "<repl>", "exec"), compile(tail, "<repl>", "eval")
    return compile(tree, "<repl>", "exec"), None


def execute_code(code: str, namespace: dict,
                 capture: dict | None = None) -> tuple[str, str | None]:
    """Execute one model turn and return feedback or a serialized final answer.

    The model sees stdout truncated to TRUNCATE; `capture["stdout"]`, when a
    dict is passed, receives the untruncated text for the archive.
    """
    output = io.StringIO()
    try:
        return _execute_code(code, namespace, output)
    finally:
        if capture is not None:
            capture["stdout"] = output.getvalue()


def _execute_code(code: str, namespace: dict,
                  output: io.StringIO) -> tuple[str, str | None]:
    try:
        body, last = _compile_cell(code)
        with contextlib.redirect_stdout(output):
            exec(body, namespace)  # noqa: S102
            if last is not None:
                value = eval(last, namespace)  # noqa: S307
                if value is not None:
                    print(repr(value))
    except FinalAnswer as answer:
        return "", answer.value
    except NameError as error:
        # a bare identifier that looks like a QID/PID means the model wrote
        # edges(Q312681) instead of edges("Q312681"). Python's message
        # reads as "this entity does not exist", and a model that believes that
        # can abandon the search.
        bare = re.search(r"name '([QP]\d+)' is not defined", str(error))
        partial = output.getvalue()
        prefix = f"Output (before error):\n{partial[-TRUNCATE // 2:]}\n" \
            if partial else ""
        if bare:
            ident = bare.group(1)
            return (prefix + f"NameError: {ident} was written as a bare Python "
                    f"name. Identifiers are STRINGS: write \"{ident}\" in "
                    f"quotes. This says nothing about whether {ident} exists in "
                    f"the graph — use has(\"{ident}\") to check that.", None)
        return prefix + "Error:\n" + traceback.format_exc(limit=3)[-3000:], None
    except SubQueryBudgetExhausted as error:
        partial = output.getvalue()
        prefix = f"Output (before cutoff):\n{partial[-TRUNCATE // 2:]}\n" if partial else ""
        return prefix + str(error), None
    except Exception:  # noqa: BLE001 - model-authored REPL code may raise anything
        partial = output.getvalue()
        prefix = f"Output (before error):\n{partial[-TRUNCATE // 2:]}\n" if partial else ""
        return prefix + "Error:\n" + traceback.format_exc(limit=3)[-3000:], None
    text = output.getvalue()
    return "Output:\n" + (text[-TRUNCATE:] if text else "(no output)"), None


def batch_dir(name: str | None, root: Path = DEFAULT_RUNS) -> Path:
    """A batch directory named `YYYYMMDD-HHMM_<name>`, so listing results/runs
    lists it in time order."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    label = (name or "run").strip("/ ")
    if re.match(r"^\d{8}-\d{4}_", label):
        return root / label                       # already stamped
    return root / f"{stamp}_{label}"


def create_run_dir(task_id: str, output_dir: Path) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / f"{task_id}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def checkpoint(run_dir: Path, messages: list[dict], env: WDGraphEnv,
               sub_query_log: list[dict] | None = None,
               reasoning_log: list[dict] | None = None) -> None:
    (run_dir / "messages.json").write_text(
        json.dumps(messages, indent=1, ensure_ascii=False)
    )
    env.save_log(run_dir / "read_log.json")
    if sub_query_log:
        (run_dir / "sub_queries.json").write_text(
            json.dumps(sub_query_log, indent=1, ensure_ascii=False))
    if reasoning_log:
        (run_dir / "reasoning.json").write_text(
            json.dumps(reasoning_log, indent=1, ensure_ascii=False))


def archive_run(run_dir: Path, task_path: Path, task: dict, model: str,
                messages: list[dict], env: WDGraphEnv, state: dict,
                started_at: float, batch_root: Path | None) -> dict:
    checkpoint(run_dir, messages, env, state.get("sub_query_log"),
               state.get("reasoning_log"))
    (run_dir / "answer.md").write_text(state["answer"] or "(no final answer)")
    metadata = {
        "run_id": run_dir.name,
        "task": task["id"],
        "model": model,
        "batch_dir": str(batch_root.resolve()) if batch_root else None,
        "task_file": str(task_path.resolve()),
        "status": state["status"],
        "iterations": state["iteration"],
        "sub_queries": state["sub_queries"],
        "shape_rejections": state["shape_rejections"],
        "wall_time_s": round(time.time() - started_at, 1),
        "reads": len(env.read_log),
        "charged": env._spent,
        "ids_seen": len(env.seen_ids()),
        "model_calls": state["model_calls"],
        "successful_model_calls": state["successful_model_calls"],
        "input_tokens": state["input_tokens"],
        "output_tokens": state["output_tokens"],
        "total_tokens": state["total_tokens"],
        "cached_tokens": state["cached_tokens"],
        "reasoning_tokens": state["reasoning_tokens"],
        # Some endpoints report reasoning_tokens=0 while the model is thinking,
        # so these two fields are computed from the text actually returned.
        "reasoning_chars": sum(len(entry.get("reasoning", ""))
                               for entry in state.get("reasoning_log", [])),
        "reasoning_turns": len(state.get("reasoning_log", [])),
        "stopped_in_thinking": state.get("stopped_in_thinking", 0),
        "generation": {
            "enable_thinking": ENABLE_THINKING,
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "truncate_chars": TRUNCATE,
            "llm_map": LLM_MAP,
            "persist": PERSIST,
            "turns_override": TURNS,
            "search_hints": os.environ.get("RLM_SEARCH_HINTS", "0") == "1",
            "subq_limit": SUBQ_LIMIT,
            # serving side, set by the job; recorded so a run is self-describing
            "mtp_tokens": os.environ.get("MTP_TOKENS"),
            "max_num_seqs": os.environ.get("MAX_NUM_SEQS"),
            "workers": os.environ.get("WORKERS"),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=1))
    return metadata


IDENTIFIER = re.compile(r"^[QP]\d+$")
IDENTIFIER_IN_TEXT = re.compile(r"(?<![A-Za-z0-9])[QP]\d+(?![A-Za-z0-9])")


def _identifier_fields(expected: dict, got: dict) -> list[str]:
    """Fields the contract fills with QIDs where the FINAL carries labels.

    Example: `occupations` as ["politician", "naval officer"] where the
    contract asks for ["Q82955", "Q10669499"]. Only the FORM of the strings is
    compared, never their content, so the check tells the model nothing about
    which identifiers are correct.
    """

    def strings(value):
        items = value if isinstance(value, list) else [value]
        return [item for item in items if isinstance(item, str)]

    bad = []
    for field, want in expected.items():
        want_strings = strings(want)
        got_strings = strings(got.get(field))
        if not want_strings or not got_strings:
            continue
        if all(IDENTIFIER.match(s) for s in want_strings) and \
                not any(IDENTIFIER.match(s) for s in got_strings):
            bad.append(field)
    return sorted(bad)


def ungrounded_identifiers(answer: str, question: str, env) -> list[str]:
    """Identifiers in a FINAL that this run never read and was never given.

    Covers properties as well as entities. A model that names P22 without ever
    seeing it in an edge listing answered from memory, which is exactly what a
    relation-discovery task is meant to rule out — and the read log records
    every property a call returned, so the check costs nothing.

    A well-formed answer can be written without a single environment call:
    invented identifiers pass every shape check.

    So the check is provenance, not shape: every identifier in the answer must
    appear in the read log, or have been supplied in the question. It is the
    one gate that a run of pure recall cannot pass.
    """
    seen = env.seen_ids()
    given = set(IDENTIFIER_IN_TEXT.findall(question))
    return sorted({identifier for identifier in IDENTIFIER_IN_TEXT.findall(answer)
                   if identifier not in seen and identifier not in given})


def recalled_identifiers(code: str, question: str, env) -> list[str]:
    """Identifiers written as literals in a cell that the run has not read.

    The provenance gate on FINAL catches a fabricated answer; it does not
    catch a fabricated PREMISE. Asked for "links that record who a person's
    parents are", the model's first cell was `if pid in ['P22', 'P25']` with a
    comment naming them — written before it had read anything. Wikidata
    property numbers are in pre-training, so rewording the question changes
    nothing; only reading the cell does.

    Literals only. An identifier the code builds at runtime came from data, and
    one the question supplied is fair game.
    """
    try:
        tree = ast.parse(code, "<grounding>", "exec")
    except SyntaxError:
        return []
    seen = env.seen_ids()
    given = set(IDENTIFIER_IN_TEXT.findall(question))
    return sorted({
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and IDENTIFIER.match(node.value)
        and node.value not in seen and node.value not in given})


def shape_mismatch(answer: str, expected) -> str | None:
    """Reject a FINAL whose SHAPE contradicts the contract, so the model can fix
    it instead of the run failing silently.

    Shape only — top-level type, key set, and whether identifier fields hold
    identifiers. No value is ever compared against the reference, so nothing
    about the answer leaks. A three-field object answered as a bare list, for
    example, is rejected instead of being recorded as complete.
    """
    if expected is None:
        return None
    body = answer.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.DOTALL)
    try:
        got = json.loads(body)
    except json.JSONDecodeError as error:
        return f"FINAL is not valid JSON ({error.msg})."
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return (f"FINAL must be a JSON object with keys "
                    f"{sorted(expected)}, not a {type(got).__name__}.")
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"unexpected {extra}")
            return "FINAL has the wrong keys: " + ", ".join(parts) + "."
        labelled = _identifier_fields(expected, got)
        if labelled:
            return (f"FINAL uses human-readable labels where the answer format "
                    f"asks for Wikidata identifiers, in: {labelled}. Report the "
                    f"QID of each item (the value you got from the graph, e.g. "
                    f"\"Q42\"), not its name.")
    elif isinstance(expected, list) and not isinstance(got, list):
        return f"FINAL must be a JSON array, not a {type(got).__name__}."
    return None


def run_loop(client: OpenAI, model: str, task: dict, env: WDGraphEnv,
             run_dir: Path, messages: list[dict], state: dict) -> None:
    """Run model/code turns until FINAL or the iteration limit."""
    expected_shape = task["answer"].get("expected")
    max_iters = task["limits"]["turns"]

    def llm_query(text: str, instruction: str) -> str:
        """Delegate interpretation of data already fetched by the root model.

        Every exchange is archived. A sub-query is a model call made from
        inside a Python cell, so it never reaches `messages.json`. A corpus
        whose subject is recursive delegation must keep these exchanges.
        """
        if state["sub_queries"] >= SUBQ_LIMIT:
            raise SubQueryBudgetExhausted(
                f"SUB-QUERY BUDGET EXHAUSTED ({SUBQ_LIMIT} calls). You are "
                f"calling llm_query once per item. Batch the remaining items "
                f"into one call — pass them all in the text and ask for one "
                f"line per item — or decide the rest in Python.")
        state["sub_queries"] += 1
        answer = chat(
            client,
            [
                {"role": "system", "content": SUBQ_SYSTEM},
                {"role": "user", "content": f"{instruction}\n\n---\n{text}"},
            ],
            SUBQ_MODEL or model,
            usage_state=state,
            kind="sub",
        )
        call = state.get("last_call") or {}
        state["sub_query_log"].append({
            "index": state["sub_queries"],
            "turn": state["iteration"],
            "instruction": instruction,
            "text": text,
            "response": answer,
            "seconds": call.get("seconds"),
            "usage": {k: call.get(k) for k in (
                "prompt_tokens", "completion_tokens", "reasoning_tokens",
                "cached_tokens", "total_tokens")},
            "finish_reason": call.get("finish_reason"),
            "reasoning": call.get("reasoning"),
        })
        return answer

    namespace = env.repl_namespace(allowed=task["allowed_functions"])
    if "qids" in task:
        namespace["task_qids"] = list(task["qids"])
    namespace["llm_query"] = llm_query

    def llm_map(items: object, instruction: str, batch: int = 10) -> dict:
        """Judge many items at once, with structured verdicts back.

        A thin binding of `map_over` to this run's llm_query, so the logic
        stays testable without a graph or a server.
        """
        return map_over(llm_query, items, instruction, batch)

    if LLM_MAP:
        namespace["llm_map"] = llm_map

    namespace["FINAL"] = final
    namespace["FINAL_VAR"] = namespace["FINAL"]

    run_started = state.setdefault("run_started", time.time())
    iterations_path = run_dir / "iterations.jsonl"

    def log_iteration(record: dict) -> None:
        """One JSON line per iteration: everything that happened in it."""
        record["total_s"] = round(time.time() - record.pop("_t0"), 3)
        record["elapsed_s"] = round(time.time() - run_started, 3)
        with iterations_path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    for iteration in range(1, max_iters + 1):
        state["iteration"] = iteration
        t0 = time.time()
        subq_before = len(state["sub_query_log"])
        reads_before = len(env.read_log)
        content = chat(client, messages, model, usage_state=state,
                       stop=turn_stop())
        root_call = dict(state.get("last_call") or {})
        messages.append({"role": "assistant", "content": content})
        code = extract_code(content)
        capture: dict = {}
        exec_s = None
        executed = False
        record = {
            "_t0": t0, "iteration": iteration,
            "started_at": round(t0, 3),
            "model_s": root_call.get("seconds"),
            "usage": {k: root_call.get(k) for k in (
                "prompt_tokens", "completion_tokens", "reasoning_tokens",
                "cached_tokens", "total_tokens")},
            "finish_reason": root_call.get("finish_reason"),
            "reasoning": root_call.get("reasoning"),
            "content": content,
            "code": code if isinstance(code, str) else None,
        }
        if code is None:
            # Some endpoints end a turn with no text when the model tries a
            # native tool call; asking for plain text gets it back.
            feedback = (
                "No Python code block found. Reply in plain text: prose, then "
                "one ```python block — never a function call. Call FINAL/"
                "FINAL_VAR inside the block to finish."
            )
            answer = None
        elif isinstance(code, list):
            feedback = (
                f"Your reply contained {len(code)} Python blocks. The protocol "
                f"is ONE block per turn and NOTHING was executed. Put your "
                f"reasoning in prose before the block, then send a single "
                f"block that does the whole turn's work.")
            answer = None
        else:
            recalled = (recalled_identifiers(code, task["question"], env)
                        if task.get("grounded_code") else [])
            if recalled:
                feedback = (
                    f"This cell was NOT executed. It names {recalled} as "
                    f"literals, and this run has neither read them nor been "
                    f"given them — they come from memory. Find them in the "
                    f"graph first: read a node's edges and look at which "
                    f"properties it actually carries, or search for the "
                    f"entity. Then write the identifiers the environment "
                    f"returned.")
                answer = None
            else:
                exec_started = time.time()
                feedback, answer = execute_code(code, namespace, capture)
                exec_s = round(time.time() - exec_started, 3)
                executed = True
        record.update({
            "code_status": ("no_code" if code is None else "multiple_blocks"
                            if isinstance(code, list) else "executed"
                            if executed else "refused_recalled_ids"),
            "exec_s": exec_s,
            "stdout_full": capture.get("stdout"),
            "sub_queries": [q["index"] for q in
                            state["sub_query_log"][subq_before:]],
            "sub_query_s": round(sum(q.get("seconds") or 0 for q in
                                     state["sub_query_log"][subq_before:]), 3),
            "reads": len(env.read_log) - reads_before,
            "read_log_range": [reads_before, len(env.read_log)],
        })
        if answer is not None:
            problem = shape_mismatch(answer, expected_shape)
            if problem is None:
                unread = ungrounded_identifiers(answer, task["question"], env)
                if unread:
                    problem = (
                        f"FINAL contains identifiers this run never read and "
                        f"the question never supplied: {unread[:8]}"
                        f"{' …' if len(unread) > 8 else ''}. Every identifier "
                        f"in the answer must come from a call you actually "
                        f"made. Fetch them, or drop them.")
            if problem is None:
                state["answer"] = answer
                state["status"] = "final"
                record.update({"final": "accepted", "answer": answer,
                               "feedback": None})
                log_iteration(record)
                print(f"[iter {iteration}] FINAL received", file=sys.stderr)
                return
            record.update({"final": "rejected", "final_problem": problem,
                           "answer": answer})
            state["shape_rejections"] += 1
            feedback = (f"{problem} The run is NOT finished. Re-read the answer "
                        f"format and call FINAL again with the full object.")
            answer = None
            print(f"[iter {iteration}] FINAL rejected: {problem}",
                  file=sys.stderr)
        suffix = f"\n[turn {iteration}/{max_iters}]"
        messages.append({"role": "user", "content": feedback + suffix})
        record.setdefault("final", None)
        record["feedback"] = feedback + suffix
        log_iteration(record)
        checkpoint(run_dir, messages, env, state["sub_query_log"],
                   state.get("reasoning_log"))
        print(
            f"[iter {iteration}] code={'yes' if code else 'NO'} "
            f"feedback={len(feedback)}ch subq={state['sub_queries']}",
            file=sys.stderr,
        )


def run_task(task_path: Path, output_root: Path, model: str) -> Path:
    task = load_task(task_path)
    if TURNS is not None:
        task["limits"]["turns"] = TURNS
    prompt = build_prompt(task)
    run_dir = create_run_dir(task["id"], output_root)
    env = WDGraphEnv()
    client = make_client()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {task['question']}"},
    ]
    state = {
        "answer": None,
        "status": "max_iters",
        "iteration": 0,
        "sub_queries": 0,
        "sub_query_log": [],
        "shape_rejections": 0,
        "model_calls": 0,
        "successful_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "reasoning_log": [],
        "stopped_in_thinking": 0,
    }
    started_at = time.time()
    (run_dir / "question.txt").write_text(task["question"])
    (run_dir / "system_prompt.txt").write_text(prompt)

    try:
        run_loop(client, model, task, env, run_dir, messages, state)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        raise
    except Exception as error:
        state["status"] = f"crashed:{type(error).__name__}"
        raise
    finally:
        metadata = archive_run(
            run_dir,
            task_path,
            task,
            model,
            messages,
            env,
            state,
            started_at,
            output_root if output_root != DEFAULT_RUNS else None,
        )
        print(json.dumps(metadata, indent=1))
        print(f"\nanswer:\n{state['answer']}\n\nrun dir: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    parser.add_argument("--batch", default=None,
                        help="batch name; the directory is stamped "
                             "YYYYMMDD-HHMM_<name> under results/runs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="explicit directory, used as given")
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("--model or MODEL_NAME is required")
    task_path = args.task_file if args.task_file.is_absolute() else ROOT / args.task_file
    if args.output_dir is not None:
        output_root = (args.output_dir if args.output_dir.is_absolute()
                       else ROOT / args.output_dir)
    else:
        output_root = batch_dir(args.batch)
    output_root.mkdir(parents=True, exist_ok=True)
    shown = output_root.relative_to(ROOT) if output_root.is_relative_to(ROOT) else output_root
    print(f"batch {shown}", file=sys.stderr)
    run_task(task_path, output_root, args.model)


if __name__ == "__main__":
    main()
