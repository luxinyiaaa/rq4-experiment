# Teacher prompt: task-based multi-page question generation (question set S2)

Model role: teacher (GPT-6 Sol). One call per evidence bundle. Temperature 0. JSON output.

Question set S2 complements the single-passage set S1. Every S2 question needs information from at least two pages of the knowledge base and reflects a task of a researcher or data user.

---

## System

You write test questions for an AI assistant that answers researchers' questions from a knowledge base of web pages from European environmental research infrastructures (ENVRI). Each question must be something a researcher or data user would plausibly ask, must need the evidence pages you are given, and must have exactly one defensible expected answer or expected behaviour.

## User template

```
QUESTION TYPE: {type}
USER TASK: {user_task}

EVIDENCE PAGES:
[E1] {title_1} ({url_1})
<<<
{text_1}
>>>
[E2] {title_2} ({url_2})
<<<
{text_2}
>>>
...
{group_note}{s1_note}
Write one question of this type. Follow all rules. Return JSON only.
```

`{group_note}` is built by the program from the knowledge-base index, not by the teacher. It takes one of these forms:

set_all:

```
CATALOGUE NOTE: The knowledge base contains {N} records of the catalogue "{catalogue}". Exactly these {k} records have {field} = {value}: {record list}. No other record of this catalogue has this value.
```

unanswerable:

```
CATALOGUE NOTE: The knowledge base contains {N} records of the catalogue "{catalogue}". No record has both {field_1} = {value_1} and {field_2} = {value_2}. The closest records are: {record list with their values}.
```

ambiguity (one of the two):

```
AMBIGUITY NOTE: The knowledge base contains {k} different records that share the name "{name}": {record list with what distinguishes them}.
```

```
CONFLICT NOTE: These pages describe the same {object} and give different values for {attribute}: {value_1} ({page}), {value_2} ({page}).
```

`{s1_note}` (only when an evidence page was used in question set S1):

```
ALREADY ASKED: Another question set already asks the following about these pages. Do not ask these facts again:
- {S1 question}
```

## User tasks

- `find_data`: finding datasets or records that match a need (variable, place, period, organisation).
- `dataset_details`: details of a specific dataset or record (period, location, instrument, variables, format).
- `access`: how to search, obtain, download or use data or a service.
- `compare`: choosing between records or resources.
- `background`: understanding a programme, network, variable or method.

## Rules for every type

1. Ask what a researcher or data user would plausibly ask for the given USER TASK. Do not ask about event logistics (booth numbers, agenda times, venues), staff, page layout or website navigation. If the pages do not support such a question, skip. Users do not know how the assistant stores its information: in `question_user`, never write "knowledge base", "records in the database" or similar; ask what is available, for example "Which ... are available" or "Is there ...".
2. The question must need information from at least two evidence pages. A question that one page alone answers is invalid.
3. It must have exactly one defensible expected answer or behaviour. Avoid opinions, open "why" questions and questions whose answer depends on interpretation.
4. Make it self-contained: name the records, stations, organisations, programmes or documents it is about, so that someone searching the whole knowledge base could find them. Identify records by name, place, organisation, period or file name, never only by a URL fragment, hash sum or database identifier. Never write "this page", "the passages", "the evidence", "according to ..." or similar.
5. Do not ask about e-mail addresses, phone numbers, postal addresses or private individuals.
6. Ignore navigation text, image links, cookie notices and menus. The pages may contain gaps where text was removed during cleaning; do not build a question across such a gap.
7. Write two versions of the question:
   - `question_user`: how a researcher would naturally ask it, in their own words, without copying distinctive phrases or field labels from the pages;
   - `question_source`: the same question with the same answer, but using the pages' own wording for what is asked.
   Both versions name the same records and ask for the same facts. Neither version may contain the answer.
8. `expected_behaviour`: the behaviour a correct assistant shows, fixed by the question type and the note (see below): `list_all`, `answer`, `disambiguate`, `flag_conflict` or `state_absence`.
9. `keypoints`: the atomic facts a correct answer must contain, each checkable on its own. For each keypoint give:
   - `text`: the fact;
   - `decisive`: true if an answer without it, or with it stated wrongly, is wrong. A decisive keypoint contains only what the question asks for;
   - `source`: the evidence page it comes from (`E1`, `E2`, ...);
   - `quote`: an exact substring of that page, copied character for character with the original spelling, spacing and punctuation, the shortest that supports the keypoint.
   If a keypoint needs more than one page, for example "two records share this name", give `sources` and `quotes` as lists (one quote per page, in the same order) instead of `source` and `quote`. A keypoint that states that no such record is available needs no quote; set `quote` to an empty string. A keypoint that is a conclusion drawn from other keypoints (for example which record is earliest, that two records share a name, or that the values differ) sets `"derived": true`, has no `quote`, and must follow from the other keypoints of the question.
   Add at most two non-decisive keypoints, and only for useful context such as which record, unit or period.
10. `reference_answer`: one to four sentences, built only from the keypoints.
11. Do not ask about an attribute that a page states with two different values (except in the type `ambiguity`), about text that belongs to the page template and is the same on many records (such as a fixed keyword list, a licence or "No online assets found"), or about usage statistics such as downloads or previews.
12. If the pages do not allow a good question of this type, return `{"skip": true, "reason": "..."}`.
13. Return every field of the output format. Each keypoint needs `text` and `decisive`. Evidence-based keypoints also need `source`/`quote` or `sources`/`quotes`; derived conclusions must set `derived: true` and omit all quote fields; decisive absence keypoints may omit quote fields.

## Rules per type

- **set_all** (`expected_behaviour`: `list_all`): ask for all records that satisfy the condition in the CATALOGUE NOTE ({field} = {value}), phrased as a user's need, for example "Which ... datasets cover the Belgian coastal area?". The answer is exactly the record list in the CATALOGUE NOTE; do not add or drop records. Each item is supported by its own evidence page. Do not state the number of items in the question. Every item is a decisive keypoint.

- **multi_hop** (`expected_behaviour`: `answer`): the first page names an entity (an organisation, centre, network, programme or instrument) that the second page describes. Ask a question whose answer needs one fact from each page, where the first fact identifies what the second fact is about, for example which centre produces a record and then a property of that centre. Do not name the bridging entity in the question, but describe it so that it identifies exactly one entity in the evidence pages; if the first page links the description to several entities (for example a workshop hosted by three infrastructures), choose another description or skip. The question must still be something a researcher would ask: do not build it on a link that is only an event, a news item, a workshop or a project list. The second page alone must not be enough to answer it. Keypoints: the bridging fact (decisive) and the final fact (decisive).

- **comparison_multi** (`expected_behaviour`: `answer`): compare three records on one attribute that all three state with a definite value. Use an attribute that matters for choosing data: temporal coverage (start, end, length), spatial coverage, depth or height, variables or parameters, instruments, resolution. Do not compare file sizes, identifiers, version numbers, licences or usage statistics. Ask which record is highest, earliest, longest, or which records share a value. Keypoints: the value of each record (decisive, with its quote) and the conclusion (decisive, `derived`).

- **ambiguity**: the program gives either an AMBIGUITY NOTE or a CONFLICT NOTE.
  - With an AMBIGUITY NOTE (`expected_behaviour`: `disambiguate`): name the record only by the shared name, as a user who does not know there are several would, and ask for an attribute whose value differs between the records. A correct answer says that there are several records with this name and gives the value for each, or asks which one is meant. Keypoints: each record with its value (decisive, one keypoint per record, with its quote), and that there are several records with this name (decisive, `derived`).
  - With a CONFLICT NOTE (`expected_behaviour`: `flag_conflict`): ask for the attribute neutrally, without hinting that the pages disagree. A correct answer reports both values with their sources and says that they differ. Keypoints: each value with its source (decisive, with its quote), and the statement that the values differ (decisive, `derived`).
  Use only differences that are explicit in the quotes; a difference in wording with the same meaning is not a difference.

- **unanswerable** (`expected_behaviour`: `state_absence`): using the CATALOGUE NOTE, ask for a record that combines {field_1} = {value_1} with {field_2} = {value_2}, which no record has, although each value exists in other records. The evidence pages are the closest existing records. A correct answer says that no such record is available and may name the closest existing records. The question must be specific enough that general knowledge cannot answer it. Keypoints: the statement that no such record is available (decisive), and the closest existing records (non-decisive, with quotes).

## Output format

```json
{
  "skip": false,
  "type": "set_all | multi_hop | comparison_multi | ambiguity | unanswerable",
  "user_task": "find_data | dataset_details | access | compare | background",
  "question_user": "...",
  "question_source": "...",
  "expected_behaviour": "list_all | answer | disambiguate | flag_conflict | state_absence",
  "reference_answer": "...",
  "keypoints": [
    {"text": "...", "decisive": true, "source": "E1", "quote": "..."}
  ],
  "pages_used": ["E1", "E2"],
  "construction_note": "One sentence: how the question combines the pages."
}
```

## Examples

The examples use fictional records and only show the format. Never invent records; use only the evidence pages you are given.

### Example 1: set_all

Evidence: three pages titled "Soil respiration archive from Station North", "... from Station Lake" and "... from Station Ridge", each stating "Country | Sweden". CATALOGUE NOTE: the catalogue "Soil respiration archive" has 4 records; exactly these 3 have Country = Sweden (the fourth, Station Coast, is in Norway).

```json
{
  "skip": false,
  "type": "set_all",
  "user_task": "find_data",
  "question_user": "I need soil respiration data from Swedish sites. Which stations in the soil respiration archive series are located in Sweden?",
  "question_source": "Which records of the soil respiration archive series have Country Sweden?",
  "expected_behaviour": "list_all",
  "reference_answer": "Three stations of the series are in Sweden: Station North, Station Lake and Station Ridge.",
  "keypoints": [
    {"text": "Station North is in Sweden", "decisive": true, "source": "E1", "quote": "Country | Sweden"},
    {"text": "Station Lake is in Sweden", "decisive": true, "source": "E2", "quote": "Country | Sweden"},
    {"text": "Station Ridge is in Sweden", "decisive": true, "source": "E3", "quote": "Country | Sweden"}
  ],
  "pages_used": ["E1", "E2", "E3"],
  "construction_note": "Lists the catalogue records with Country = Sweden given in the catalogue note."
}
```

### Example 2: unanswerable

Evidence: "Methane time series from Station Hill (5 m)" and "Carbon dioxide time series from Station Port (14 m)". CATALOGUE NOTE: no record has both Gas = methane and Station = Port; the closest records are these two.

```json
{
  "skip": false,
  "type": "unanswerable",
  "user_task": "find_data",
  "question_user": "Is there a methane time series from Station Port, and at what sampling height was it measured?",
  "question_source": "What is the sampling height of the methane time series from Station Port?",
  "expected_behaviour": "state_absence",
  "reference_answer": "No methane time series from Station Port is available. There is a carbon dioxide time series from Station Port (14 m) and a methane time series from Station Hill (5 m).",
  "keypoints": [
    {"text": "No methane time series from Station Port is available", "decisive": true},
    {"text": "The closest records are the Station Port carbon dioxide series and the Station Hill methane series", "decisive": false, "source": "E1", "quote": "Methane time series from Station Hill"}
  ],
  "pages_used": ["E1", "E2"],
  "construction_note": "Combines the variable of one record with the station of another; no such record exists in the series."
}
```
