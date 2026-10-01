# Graph format

The graph environment (`scripts/wd_graph_env.py`) reads a frozen Wikidata graph
from one directory: `data/graph/` in the repository, or the directory named by
`WIKIDATA_GRAPH_DIR`. The graph is read-only at runtime. This page describes the
files that directory must hold, so a graph built from any Wikidata dump can be
used.

Identifiers are stored as integers without their prefix: `Q42` is `42`, `P31`
is `31`. Text fields exist for six languages, written `<lang>` below: `en`,
`fr`, `de`, `zh`, `ar`, `ru`.

| Item | Used for |
| :---- | :---- |
| `nodes.parquet` | The node list, English labels and degrees |
| `edges_fwd.arrow`, `edges_bwd.arrow` | Navigation in both directions |
| `properties.json` | Property labels |
| `entities/`, `entities_index.json` | One full record per entity: labels, claims, qualifiers, ranks, references |
| `label_index/` | Name search |

## `nodes.parquet`

One row per node, **sorted by `qid`**. The environment reads four columns:

| Column | Type | Meaning |
| :---- | :---- | :---- |
| `qid` | int64 | Item number |
| `label_en` | string | English label |
| `deg_out` | int32 | Number of outgoing edges in the edge files |
| `deg_in` | int32 | Number of incoming edges in the edge files |

Other columns may be present and are ignored.

## `edges_fwd.arrow` and `edges_bwd.arrow`

Arrow IPC files, opened memory-mapped. Both hold the same edges, one row per
item-to-item statement, deduplicated, with three columns:

| Column | Type | Meaning |
| :---- | :---- | :---- |
| `src` | int32 | Subject item |
| `prop` | int32 | Property |
| `dst` | int32 | Value item |

`edges_fwd.arrow` is sorted by `src` (then `dst`, then `prop`), and
`edges_bwd.arrow` by `dst` (then `src`, then `prop`). Both endpoints of every
edge must be nodes of `nodes.parquet`.

## `properties.json`

A JSON object mapping each property to its labels:

```json
{"P31": {"en": "instance of", "fr": "nature de l'élément", "de": "ist ein(e)"}}
```

## `entities/` and `entities_index.json`

`entities/` holds Parquet parts. Each part covers a contiguous range of item
numbers, and the ranges do not overlap. `entities_index.json` lists them, for
example:

```json
{"ranges": [{"file": "part_00000.parquet", "qid_min": 1, "qid_max": 18233}]}
```

Each part has one row per entity with these columns:

| Columns | Type | Meaning |
| :---- | :---- | :---- |
| `qid` | int64 | Item number |
| `label_<lang>`, `description_<lang>` | string | Label and description |
| `aliases_<lang>` | string | Aliases |
| `sitelink_<lang>` | string | Wikipedia article title |
| `instance_of` | list of strings | `P31` values, as `Q…` identifiers |
| `claim_property_id`, `claim_value_id`, `claim_value_label`, `claim_value_type` | list of strings | The claims, one list element per statement |
| `qual_property_id`, `qual_value_id`, `qual_qualifier_property_id`, `qual_qualifier_value_id`, `qual_qualifier_value_label` | list of strings | The qualifiers, one element per qualifier, keyed to their statement by property and value |
| `rank_property_id`, `rank_value_id`, `rank_rank` | list of strings | Statement ranks: `preferred`, `normal` or `deprecated`. A statement without an entry is `normal` |
| `ref_property_id`, `ref_value_id`, `ref_ref_property_id`, `ref_ref_value_id` | list of strings | The references, one element per reference value, keyed to their statement by property and value |

The lists that share a prefix (`claim_`, `qual_`, `rank_`, `ref_`) are parallel:
element `i` of each list describes the same statement or annotation. Properties
and items are written with their prefix (`P31`, `Q5`). `claim_value_type` is the
Wikidata datatype of the value, for example `wikibase-entityid`, `time`,
`quantity` or `string`. A quantity value carries its unit in brackets, for
example `+2473 [Q11573]`.

## `label_index/`

A [Tantivy](https://github.com/quickwit-oss/tantivy-py) index with one document
per node that has a label or an alias, and a `fields.json` file:

```json
{"text_fields": ["l_en", "a_en", "l_fr", "a_fr", "l_de", "a_de", "l_zh", "a_zh", "l_ar", "a_ar", "l_ru", "a_ru", "fold"]}
```

| Field | Kind | Content |
| :---- | :---- | :---- |
| `l_<lang>` | text | Label |
| `a_<lang>` | text | Aliases |
| `fold` | text | All labels and aliases, with accents folded to ASCII |
| `qid` | unsigned, stored, fast | Item number |
| `sl` | unsigned, stored, fast | Number of the six languages with a sitelink |
| `deg` | unsigned, stored, fast | `deg_out + deg_in` |

`search_entity` ranks matches by text relevance, weighted by `sl` and `deg`, so
a well-known entity comes first among its homonyms.
