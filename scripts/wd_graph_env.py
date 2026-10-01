#!/usr/bin/env python3
"""
wd_graph_env.py — full-graph navigation environment over data/graph/
(nodes.parquet + edges_{fwd,bwd}.arrow).

The graph location is data/graph/ in this repository, or the directory named by
the WIKIDATA_GRAPH_DIR environment variable.

World: nodes are Wikidata items (Q-entities); edges are Q->Q statements
between nodes. The file layout is described in docs/graph-format.md.

Harness interface: repl_namespace(allowed=), read_log, seen_ids(), save_log().

Scale conventions (deliberate, documented for the prompt):
  - edges() returns a dict with degree counts and per-direction lists, capped
    at `limit` per direction (hubs here have 100k+ incoming edges; returning
    them all would blow the stdout budget instantly). Truncation is explicit.
  - every returned id is expandable by construction.

Memory: edge arrays are opened memory-mapped; lookups touch O(log n) pages.
Node labels stay in one Arrow table (compact strings), not Python dicts.
"""

import json
import os
import pathlib
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LANGS = ("en", "fr", "de", "zh", "ar", "ru")

# Explain an empty search instead of returning silence. Off by default in
# the code and on in the reference configuration (config/runs/reference.env).
SEARCH_HINTS = os.environ.get("RLM_SEARCH_HINTS", "0") == "1"


REPO = pathlib.Path(__file__).resolve().parent.parent
# The graph is large and often lives outside the checkout. WIKIDATA_GRAPH_DIR
# points at it without a symlink; the repository default is data/graph/.
DEFAULT_DIR = pathlib.Path(os.environ.get("WIKIDATA_GRAPH_DIR")
                           or REPO / "data" / "graph")


def _qint(id_):
    if isinstance(id_, (int, np.integer)):
        return int(id_)
    if isinstance(id_, str) and id_[:1] == "Q" and id_[1:].isdigit():
        return int(id_[1:])
    raise KeyError(
        f"{id_!r} is not a QID. Known ids only — do not guess QIDs; use the "
        f"ids returned by edges().")


class Struct(dict):
    """A fixed-schema result that refuses invented keys, even through .get().

    The most expensive failure mode here is not a wrong answer, it is a silent
    one: a model guesses a key name, writes ``result.get("out_count", 0)`` as a
    defensive reflex, receives the default, and every downstream count
    collapses to zero without anything being raised or logged.

    A KeyError the model can read and repair is strictly better than a plausible
    default it cannot detect. Only dictionaries with a CLOSED schema are wrapped:
    a mapping keyed by language stays an ordinary dict, because a missing
    language is real absence rather than a typo.
    """

    def __missing__(self, key):
        raise KeyError(
            f"{key!r} is not a key of this result. The keys are "
            f"{sorted(self)}. Use one of those — do not guess key names.")

    def get(self, key, default=None):
        if key not in self:
            raise KeyError(
                f"{key!r} is not a key of this result, so .get({key!r}, "
                f"{default!r}) would silently return {default!r} and corrupt "
                f"everything computed from it. The keys are {sorted(self)}.")
        return self[key]


_IDENT = re.compile(r"^[QP]\d+$")


class WDGraphEnv:
    def __init__(self, dir_path=DEFAULT_DIR, edge_limit=200):
        d = pathlib.Path(dir_path)
        self.dir = d
        # projection is load-bearing: nodes.parquet now carries the multilingual
        # entity layer (28 columns, ~12 GB in memory), so only ask for the four
        # navigation columns. Other languages are read on demand via labels().
        nodes = pq.read_table(
            d / "nodes.parquet",
            columns=["qid", "label_en", "deg_out", "deg_in"])
        self._qid = nodes.column("qid").to_numpy()          # sorted
        self._label = nodes.column("label_en")              # arrow, aligned
        self._deg_out = nodes.column("deg_out").to_numpy()
        self._deg_in = nodes.column("deg_in").to_numpy()
        edges = lambda n: pa.ipc.open_file(  # noqa: E731
            pa.memory_map(str(d / n), "rb")).read_all()
        fwd, bwd = edges("edges_fwd.arrow"), edges("edges_bwd.arrow")
        self._f_src, self._f_prop, self._f_dst = (
            fwd.column("src").to_numpy(), fwd.column("prop").to_numpy(),
            fwd.column("dst").to_numpy())
        self._b_src, self._b_prop, self._b_dst = (
            bwd.column("src").to_numpy(), bwd.column("prop").to_numpy(),
            bwd.column("dst").to_numpy())

        # properties.json maps a PID to a per-language label map
        raw = {k: v for k, v in json.loads(
            (d / "properties.json").read_text()).items() if k[:1] == "P"}
        self._plabels_all = {           # pid -> {lang: label}, for search_property
            k: (v if isinstance(v, dict) else {"en": v}) for k, v in raw.items()}
        self._plabels = {k: v.get("en") for k, v in self._plabels_all.items()}

        # Per-direction listing cap of edges(). Stdout safety is the harness's
        # print truncation, not this cap.
        self.edge_limit = edge_limit
        self._spent = 0              # charged reads, reported as `charged`
        self._seen = set()
        self._log = []
        self._expanded_full = set()  # qids fully expanded (both, no pid)
        self._eindex = None          # lazy: entity part lower bounds
        self._eparts = None          # lazy: entity part descriptors
        self._records = {}           # qid -> parsed entity record
        self._search = None          # lazy: (index, searcher, fields)

    # ---------- bookkeeping ----------

    # `degree` is not here because it is free unconditionally: charging it would
    # make a read count depend on whether a QID had been seen.
    _FREE_ON_SEEN = {"name", "has"}

    def _rec(self, fn, args, ids, free=False):
        self._log.append({"fn": fn, "args": args, "ids": sorted(set(ids))})
        new = [i for i in ids if i not in self._seen]
        self._seen.update(ids)
        if not free and not (fn in self._FREE_ON_SEEN and not new):
            self._spent += 1

    # ---------- lookups ----------

    def _pos(self, q):
        i = np.searchsorted(self._qid, q)
        if i < len(self._qid) and self._qid[i] == q:
            return int(i)
        return None

    def _name_of(self, q):
        i = self._pos(q)
        return self._label[i].as_py() if i is not None else None

    def _slice(self, key_arr, q):
        # dtype-matched scalar is load-bearing: a Python-int key against an
        # int32 mmap makes numpy cast-copy the ENTIRE array per call (~1s)
        k = np.asarray(q, dtype=key_arr.dtype)
        lo = key_arr.searchsorted(k, side="left")
        hi = key_arr.searchsorted(k, side="right")
        return int(lo), int(hi)

    # ---------- REPL-facing ----------

    def has(self, qid):
        """Cheap existence check: is this id a node of the graph? (1 read;
        free once seen)."""
        try:
            q = _qint(qid)
        except KeyError:
            self._rec("has", [qid], [])
            return False
        out = self._pos(q) is not None
        self._rec("has", [qid], [f"Q{q}"] if out else [])
        return out

    def name(self, qid):
        """English label of a node QID or a PID. Free for ids already seen.
        None if unknown."""
        s = str(qid)
        if s[:1] == "P" and s[1:].isdigit():
            out = self._plabels.get(s)
            self._rec("name", [s], [s] if out is not None else [])
            return out
        q = _qint(qid)
        out = self._name_of(q)
        self._rec("name", [s], [f"Q{q}"] if out is not None else [])
        return out

    def count_edges(self, qid, direction="both", pid=None):
        """How many edges match, WITHOUT returning any of them. 1 read.

        `degree()` reports a node's total degree across every predicate, which
        is the wrong number for planning a filtered sweep and misleads badly:
        Berlin has 83,417 incoming edges, but only 245 of them are
        `filming location`. A model told 83,417 has no reason to attempt a
        sweep it could finish in two pages.

        Charged, not free. A pool-size field is an anti-sampling check, and a
        free exact count would let a trajectory report the pool without ever
        enumerating it. Charging keeps the intent visible in the read log.
        """
        q = _qint(qid)
        if self._pos(q) is None:
            raise KeyError(f"Q{q} is not in the graph.")
        if direction == "in_":
            direction = "in"
        if direction not in {"both", "out", "in"}:
            raise ValueError("direction must be 'both', 'out', 'in', or 'in_'")
        want_pid = int(str(pid).lstrip("P")) if pid is not None else None

        def tally(srcarr, proparr, otherarr, key_is_src):
            lo, hi = self._slice(srcarr if key_is_src else otherarr, q)
            n = 0
            for j in range(lo, hi):
                p = int(proparr[j])
                o = int((otherarr if key_is_src else srcarr)[j])
                if o == q:
                    continue
                if want_pid is not None and p != want_pid:
                    continue
                n += 1
            return n

        out = tally(self._f_src, self._f_prop, self._f_dst, True) \
            if direction in ("both", "out") else 0
        inn = tally(self._b_src, self._b_prop, self._b_dst, False) \
            if direction in ("both", "in") else 0
        self._rec("count_edges", [f"Q{q}", direction, str(pid)], [f"Q{q}"])
        return {"out": out, "in_": inn, "total": out + inn}

    def degree(self, qid):
        """(deg_out, deg_in) of a node — how expensive expanding it would be.
        Always free: it is planning metadata, not content."""
        q = _qint(qid)
        i = self._pos(q)
        if i is None:
            raise KeyError(f"Q{q} is not in the graph.")
        self._rec("degree", [f"Q{q}"], [f"Q{q}"], free=True)
        return int(self._deg_out[i]), int(self._deg_in[i])

    def edges(self, qid, direction="both", pid=None, limit=None, offset=0):
        """Graph edges of a node, both directions in ONE call (1 read).
        Returns {id, label, deg_out, deg_in, out: [...], in_: [...],
        truncated}. Each entry: (pid, pqidlabel, other_qid, other_label).
        Directions are capped at `limit` (default env cap) — check deg_* and
        `truncated`; use pid= or direction= to narrow a hub instead of paging
        blindly."""
        # The returned mapping names its incoming list ``in_`` because ``in``
        # is a Python keyword. Models copy that key into the direction
        # argument, so accept the alias and reject true typos.
        direction = "in" if direction == "in_" else direction
        if direction not in {"both", "out", "in"}:
            raise ValueError("direction must be 'both', 'out', 'in', or 'in_'")
        q = _qint(qid)
        i = self._pos(q)
        if i is None:
            raise KeyError(
                f"Q{q} is not in the graph. Known ids only — do not guess "
                f"QIDs; use ids returned by edges().")
        lim = int(limit) if limit else self.edge_limit
        off = max(0, int(offset or 0))
        want_pid = int(str(pid).lstrip("P")) if pid is not None else None
        touched = [f"Q{q}"]
        out_entries, in_entries = [], []
        truncated = {"out": False, "in": False}

        def collect(srcarr, proparr, otherarr, key_is_src, acc, dkey):
            lo, hi = self._slice(srcarr if key_is_src else otherarr, q)
            n_kept, n_seen = 0, 0
            for j in range(lo, hi):
                p = int(proparr[j])
                o = int((otherarr if key_is_src else srcarr)[j])
                if o == q:
                    continue
                if want_pid is not None and p != want_pid:
                    continue
                # offset counts MATCHES, not raw array positions: skipping by
                # index would drift as soon as a row is filtered out, so page
                # N+1 would silently omit rows page N never showed.
                n_seen += 1
                if n_seen <= off:
                    continue
                if n_kept >= lim:
                    truncated[dkey] = True
                    break
                pl = self._plabels.get(f"P{p}")
                acc.append((f"P{p}", pl, f"Q{o}", self._name_of(o)))
                touched.extend([f"P{p}", f"Q{o}"])
                n_kept += 1

        if direction in ("both", "out"):
            collect(self._f_src, self._f_prop, self._f_dst, True,
                    out_entries, "out")
        if direction in ("both", "in"):
            collect(self._b_src, self._b_prop, self._b_dst, False,
                    in_entries, "in")
        # cache economics: re-expanding a FULLY expanded node returns data
        # the caller already holds -> free (same contract as name()/labels)
        repeat = q in self._expanded_full
        if direction == "both" and pid is None:
            self._expanded_full.add(q)
        self._rec("edges", [f"Q{q}", direction, pid, lim], touched,
                  free=repeat)
        return Struct(
            id=f"Q{q}", label=self._name_of(q),
            deg_out=int(self._deg_out[i]), deg_in=int(self._deg_in[i]),
            out=out_entries, in_=in_entries,
            truncated=truncated,
        )

    # ---------- entity record (data/graph/entities) ----------

    def _entity_index(self):
        """Lower bounds of the entity parts — ~38 KB, loaded once on first use.

        Ranges are disjoint by construction, so one searchsorted resolves a qid to exactly one part.
        """
        if self._eindex is None:
            path = self.dir / "entities_index.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"no entities_index.json in {self.dir}")
            index = json.loads(path.read_text())
            self._eparts = index["ranges"]
            self._eindex = np.array([p["qid_min"] for p in self._eparts],
                                    dtype=np.int64)
        return self._eindex

    def _record(self, q):
        """The entity's full record, parsed and cached. One part read."""
        if q in self._records:
            return self._records[q]
        bounds = self._entity_index()
        position = int(np.searchsorted(bounds, q, side="right")) - 1
        if position < 0 or q > self._eparts[position]["qid_max"]:
            raise KeyError(f"Q{q} is not in the graph.")
        table = pq.read_table(self.dir / "entities" /
                              self._eparts[position]["file"])
        rows = table.filter(pc.equal(table.column("qid"), q))
        if rows.num_rows == 0:
            raise KeyError(f"Q{q} is not in the graph.")
        row = rows.to_pylist()[0]
        self._records[q] = row
        return row

    @staticmethod
    def _zip(row, prefix, fields):
        """Parallel list columns share one order — read them back as tuples."""
        columns = [row.get(f"{prefix}_{f}") or [] for f in fields]
        return list(zip(*columns)) if columns[0] else []

    @staticmethod
    def _keep_best_rank(claims, ranks):
        """Per property: preferred if any exists, else everything not deprecated.

        Wikidata's rule for "the current value".
        """
        by_property = {}
        for claim in claims:
            by_property.setdefault(claim["property"], []).append(claim)
        out = []
        for statements in by_property.values():
            keep = [s for s in statements if s["rank"] == "preferred"] or \
                   [s for s in statements if s["rank"] != "deprecated"]
            out.extend(keep)
        return out

    def claims(self, qid, pid=None, all_ranks=False):
        """Everything asserted about an entity: properties, typed values and
        qualifiers — including the dates, quantities and identifiers that
        edges() cannot represent. Best-rank by default (preferred if any, else
        non-deprecated); all_ranks=True keeps every statement. Each entry
        carries property, value, value_label, value_type, rank and qualifiers.
        1 read; free afterwards for the same entity."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)

        ranks = {(p, v): r for p, v, r in
                 self._zip(record, "rank", ["property_id", "value_id", "rank"])}
        qualifiers = {}
        for p, v, qp, qv, ql in self._zip(
                record, "qual", ["property_id", "value_id",
                                 "qualifier_property_id", "qualifier_value_id",
                                 "qualifier_value_label"]):
            qualifiers.setdefault((p, v), []).append(
                {"property": qp, "property_label": self._plabels.get(qp),
                 "value": qv, "value_label": ql})

        out = [{"property": p, "property_label": self._plabels.get(p),
                "value": v, "value_label": vl, "value_type": vt,
                "rank": ranks.get((p, v), "normal"),
                "qualifiers": qualifiers.get((p, v), [])}
               for p, v, vl, vt in self._zip(
                   record, "claim", ["property_id", "value_id", "value_label",
                                     "value_type"])]
        if not all_ranks:
            out = self._keep_best_rank(out, ranks)
        if pid is not None:
            out = [c for c in out if c["property"] == str(pid)]
        # Everything the caller now holds an identifier for: the provenance
        # gates must accept a property, or a qualifier value, read here.
        touched = [f"Q{q}"] + [c["value"] for c in out
                               if c["value_type"] == "wikibase-entityid"]
        touched += [c["property"] for c in out]
        # a quantity's unit arrives inside the value string: "+2473 [Q11573]"
        touched += [u for c in out if c["value_type"] == "quantity"
                    for u in re.findall(r"Q\d+", str(c["value"]))]
        touched += [x for c in out for qual in c["qualifiers"]
                    for x in (qual["property"], qual["value"])
                    if _IDENT.match(str(x))]
        self._rec("claims", [f"Q{q}", pid, all_ranks], touched, free=cached)
        return out

    def references(self, qid, pid=None):
        """Which sources ground an entity's claims: [{property, value,
        sources: [{property, value}]}]. P248 'stated in' names a database,
        P143 'imported from' means a Wikipedia copy, P854 a URL. Free once the
        entity's record has been read."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        grouped = {}
        for p, v, sp, sv in self._zip(
                record, "ref", ["property_id", "value_id", "ref_property_id",
                                "ref_value_id"]):
            if pid is not None and p != str(pid):
                continue
            grouped.setdefault((p, v), []).append(
                {"property": sp, "property_label": self._plabels.get(sp),
                 "value": sv})
        out = [{"property": p, "property_label": self._plabels.get(p),
                "value": v, "sources": sources}
               for (p, v), sources in grouped.items()]
        touched = [f"Q{q}"] + [c["property"] for c in out]
        touched += [x for c in out for src in c["sources"]
                    for x in (src["property"], src["value"])
                    if _IDENT.match(str(x))]
        self._rec("references", [f"Q{q}", pid], touched, free=cached)
        return out

    def describe(self, qid, langs=None):
        """Labels, descriptions and aliases of an entity across the six target
        languages (en, fr, de, zh, ar, ru), plus its instance_of and sitelinks.
        Free once the entity's record has been read."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        wanted = list(langs) if langs else list(LANGS)
        out = Struct(id=f"Q{q}", instance_of=record.get("instance_of"),
                     languages={})
        for lang in wanted:
            entry = Struct((field, record.get(f"{field}_{lang}"))
                          for field in ("label", "description", "aliases",
                                        "sitelink"))
            if any(entry.values()):
                out["languages"][lang] = entry
        self._rec("describe", [f"Q{q}", wanted], [f"Q{q}"], free=cached)
        return out

    # ---------- convenience views over one record ----------

    def label(self, qid, lang="en"):
        """Label in ONE language. No fallback: None if that language is absent."""
        q = _qint(qid)
        cached = q in self._records
        value = self._record(q).get(f"label_{lang}")
        self._rec("label", [f"Q{q}", lang], [f"Q{q}"], free=cached)
        return value

    def labels(self, qid):
        """Labels in every target language that has one."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        self._rec("labels", [f"Q{q}"], [f"Q{q}"], free=cached)
        return {l: record.get(f"label_{l}") for l in LANGS
                if record.get(f"label_{l}")}

    def descriptions(self, qid, _free=False):
        """Descriptions in every target language that has one."""
        q = _qint(qid)
        cached = q in self._records
        record = self._record(q)
        self._rec("descriptions", [f"Q{q}"], [f"Q{q}"],
                  free=cached or _free)
        return {l: record.get(f"description_{l}") for l in LANGS
                if record.get(f"description_{l}")}

    # ---------- name search (data/graph/label_index) ----------

    # Depth is load-bearing because BM25 rewards short fields — Barack Obama
    # sits at raw rank 145 of 1689 for "obama", under dozens of entities whose
    # entire label is "Obama".
    SEARCH_DEPTH = 4000
    SEARCH_SITELINK_WEIGHT = 2.0
    SEARCH_DEGREE_WEIGHT = 4.0

    def _index(self):
        """The label index, opened once."""
        if self._search is None:
            import tantivy
            path = self.dir / "label_index"
            if not path.exists():
                raise RuntimeError(
                    f"no label index at {path}")
            index = tantivy.Index.open(str(path))
            index.reload()
            fields = json.loads((path / "fields.json").read_text())["text_fields"]
            self._search = (index, index.searcher(), fields)
        return self._search

    def search_property(self, text, limit=10):
        """Find a PROPERTY by name across the six languages. Returns
        (total_matches, [{pid, label, lang}]).

        Separate from search_entity because the two indexes are nothing alike:
        properties fit in memory and need no ranking beyond exact before
        prefix before substring, while entities need a full-text index.
        1 read."""
        key = str(text).strip().lower()
        if not key:
            return 0, []
        found = []
        for pid, labels in self._plabels_all.items():
            best = None
            for lang, label in labels.items():
                if not label:
                    continue
                low = label.lower()
                tier = (0 if low == key else 1 if low.startswith(key)
                        else 2 if key in low else None)
                if tier is not None and (best is None or tier < best[0]):
                    best = (tier, lang, label)
            if best:
                # lower PID numbers are the older, more fundamental properties
                found.append((best[0], int(pid[1:]), pid, best[2], best[1]))
        found.sort()
        out = [{"pid": pid, "label": label, "lang": lang}
               for _, _, pid, label, lang in found[:max(1, int(limit))]]
        self._rec("search_property", [key, limit], [d["pid"] for d in out])
        return len(found), out

    def search_entity(self, text, limit=10, lang="en"):
        """Find entities by name. Returns (total_matches, [dict, ...]) where
        each dict is {qid, label, description} — the label and description are
        what a reader needs to tell identical names apart.

        Matching spans labels and aliases in all six languages plus a
        diacritic-folded form, so 'gerhard schroder' finds 'Gerhard Schröder'.
        All query words must match. Results are ordered by text relevance
        weighted by how connected and how widely documented the entity is.
        1 read."""
        import math
        index, searcher, fields = self._index()
        # A name is plain words: "Tsubasa: Reservoir Chronicle" must not parse
        # "Tsubasa:" as a field, nor "Women's" as a quote, so only words remain.
        words = " ".join(re.findall(r"\w+", str(text))) or str(text)
        query = index.parse_query(words, fields, conjunction_by_default=True)
        found = searcher.search(query, self.SEARCH_DEPTH, count=True)
        # The text score ranks an entity with many aliases below its own
        # namesakes. A second query on the label fields alone brings the exact
        # names in, and an exact label ranks first, by popularity.
        labels = [f for f in fields if f.startswith("l_")]
        exact_hits = searcher.search(
            index.parse_query(f'"{words}"', labels), self.SEARCH_DEPTH).hits
        wanted = words.casefold()
        best = {}
        for score, address in found.hits + exact_hits:
            doc = searcher.doc(address)
            q = int(doc["qid"][0])
            weight = (1 + self.SEARCH_SITELINK_WEIGHT * doc["sl"][0]
                      + self.SEARCH_DEGREE_WEIGHT * math.log1p(doc["deg"][0]))
            exact = " ".join(re.findall(r"\w+", self._name_of(q) or "")).casefold() == wanted
            best[q] = max(best.get(q, (0, 0.0)), (1, weight) if exact else (0, score * weight))
        scored = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        out = []
        for q, _ in scored[:max(1, int(limit))]:
            record = self._record(q)
            out.append({
                "qid": f"Q{q}",
                "label": next((record.get(f"label_{l}") for l in (lang,) + LANGS
                               if record.get(f"label_{l}")), None),
                "description": next(
                    (record.get(f"description_{l}") for l in (lang,) + LANGS
                     if record.get(f"description_{l}")), None),
            })
        if not out and SEARCH_HINTS:
            self._explain_empty_search(str(text), index, searcher, fields)
        self._rec("search_entity", [str(text), limit], [d["qid"] for d in out])
        return found.count, out

    def _explain_empty_search(self, text, index, searcher, fields):
        """Say WHICH word matched nothing, when a search returns nothing.

        The index is conjunctive -- one unmatched word zeroes the whole query
        -- and an empty list cannot distinguish "this entity is absent" from
        "your third word is wrong". Without a hint the model rephrases
        blindly. The harness holds the information that lets the model
        correct itself, so it returns it. Costs nothing on the common path:
        this runs only when the result is already empty.
        """
        words = [w for w in str(text).split() if w][:8]
        if len(words) < 2:
            print(f"(no match for {text!r}. Matching spans labels and aliases "
                  f"in en/fr/de/zh/ar/ru; try a different name or spelling.)")
            return
        counts = []
        for word in words:
            try:
                q = index.parse_query(word, fields, conjunction_by_default=True)
                counts.append((word, searcher.search(q, 1, count=True).count))
            except Exception:
                counts.append((word, None))
        dead = [w for w, n in counts if n == 0]
        live = ", ".join(f"{w}={n}" for w, n in counts if n)
        note = (f"none of your words match nothing individually, but no entity "
                f"has them ALL" if not dead
                else f"these match nothing: {', '.join(dead)}")
        print(f"(no match for {text!r}. All query words must match, so one "
              f"bad word empties the result -- {note}. Words that do match: "
              f"{live or 'none'}. Search with fewer, more distinctive words.)")

    # ---------- harness side ----------

    def repl_namespace(self, allowed=None):
        ns = {"edges": self.edges, "count_edges": self.count_edges,
              "name": self.name, "has": self.has, "degree": self.degree,
              "claims": self.claims, "references": self.references,
              "describe": self.describe,
              "label": self.label, "labels": self.labels,
              "descriptions": self.descriptions,
              "search_entity": self.search_entity,
              "search_property": self.search_property}
        if allowed is not None:
            ns = {k: v for k, v in ns.items() if k in allowed}
        return ns

    @property
    def read_log(self):
        return list(self._log)

    def seen_ids(self):
        out = set()
        for e in self._log:
            out.update(e["ids"])
        return out

    def save_log(self, path):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._log, indent=1))


if __name__ == "__main__":
    import time
    t0 = time.time()
    env = WDGraphEnv()
    print(f"loaded in {time.time()-t0:.1f}s")
    t0 = time.time()
    e = env.edges("Q31")  # Belgium — a hub; exercises truncation
    print(f"edges(Q31) in {time.time()-t0:.2f}s: deg_out={e['deg_out']} "
          f"deg_in={e['deg_in']} out={len(e['out'])} in={len(e['in_'])} "
          f"truncated={e['truncated']}")
    print("sample out:", e["out"][:5])
    print("sample in:", e["in_"][:5])
    print("has(Q7186):", env.has("Q7186"), "| name(Q7186):", env.name("Q7186"))
    print(f"reads: {len(env.read_log)} | spent: {env._spent}")
