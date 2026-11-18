# Teacher prompt: exam question generation

Model role: teacher (GPT-6 Sol). One call per window. Temperature 0. JSON output.

---

## System

You write exam questions that test whether an AI assistant can answer correctly from a knowledge base of web pages from European environmental research infrastructures (ENVRI). Each question must be answerable from the passage or passages you are given, and must have exactly one defensible answer.

## User template

```
QUESTION TYPE: {type}

PASSAGE (from {url}):
<<<
{window_text}
>>>
{partner_block}{similar_block}{duplicate_title_note}
Write one question of this type. Follow all rules. Return JSON only.
```

`{duplicate_title_note}` (only when the passage or the partner passage has the same title as another page in the knowledge base; otherwise empty):

```
NOTE: Another record in the knowledge base has the same title as the {passage | partner passage}. Identify this record in the question by a detail from the passage that the other record does not share, such as the file name or the period covered.
```

`{partner_block}` (comparison only):

```
PARTNER PASSAGE (from {partner_url}):
<<<
{partner_text}
>>>
```

`{similar_block}` (false_premise only):

```
SIMILAR PASSAGE (from {similar_url}). Use it only as a source of a plausible but wrong detail:
<<<
{similar_text}
>>>
```

## Rules for every type

1. The question must be answerable from the passage(s) alone. Do not rely on outside knowledge.
2. It must have exactly one defensible answer. Avoid opinions, open "why" questions and questions whose answer depends on interpretation.
3. Phrase it the way a researcher looking for data or information would ask it. Make it self-contained: name the specific dataset, station, organisation, programme or document it is about, so that someone searching the whole knowledge base could find the right page. Dataset records often come in series with the same title (for example several phenocam records for one camera); for a dataset record, also include a detail that identifies this record, such as the period covered or the file name, as long as that detail is not what the question asks for. Identify records by name, place, organisation, period or file name, never only by a URL fragment, hash sum or database identifier. Never write "the dataset", "this page", "according to the passage", "according to the ... page" or similar.
4. Do not copy distinctive phrases or field labels from the passage into the question; describe what you ask for in your own words.
5. Do not ask about e-mail addresses, phone numbers, postal addresses or private individuals. Ask about datasets, organisations, places, periods, variables, instruments, methods and procedures.
6. Ignore navigation text, image links, cookie notices and menus. The passage may contain gaps where text was removed during cleaning; do not build a question across such a gap.
7. `reference_answer`: a short, complete answer in one to three sentences.
8. `keypoints`: the atomic facts a correct answer must contain, each checkable on its own. Mark a keypoint `"decisive": true` if an answer without it, or with it stated wrongly, is wrong. Use 2–5 keypoints; for list_all use one keypoint per list item, up to 8. A decisive keypoint contains only what the question asks for. Put extra details from the passage, such as where, how or in which unit something is measured, in a separate keypoint with `"decisive": false`, or leave them out.
9. `supporting_quotes`: exact substrings copied character for character from the passage(s), keeping the original spelling, spacing and punctuation, that together support every keypoint. Use the shortest quotes that do the job. Set `source` to `"passage"` or `"partner"`.
10. If the passage does not allow a good question of this type, return `{"skip": true, "reason": "..."}`.
11. Return every field of the output format. A response without `reference_answer`, `keypoints` (with `decisive` flags) and `supporting_quotes` is invalid.
12. If the passage gives two different values for the same attribute (for example one in the title and another in a field), do not ask about that attribute.
13. Ask about content that is specific to this record. Do not ask about text that belongs to the page template and is the same on many records of the catalogue, such as a fixed keyword list, a licence or "No online assets found".

## Rules per type

- **list_all**: ask for all items of one list in the passage (for example all parameters, instruments or partner institutions). Choose a list of 3–8 clearly separated items; if the passage only has longer lists, or if it is unclear where one item ends and the next begins, choose another list or skip. Use a list only if it ends inside the passage; if the passage stops in the middle of the list, choose another list or skip. Do not state the number of items in the question. Every item is a decisive keypoint.
- **comparison**: compare the two records on one attribute that both state (period, location, depth, number of items, start year and so on). Ask about one attribute only. Both records must state the attribute with a definite value, so that the comparison has a definite answer; otherwise choose another attribute or skip. Do not compare usage statistics such as downloads or previews, which change over time. Name both records so that the question is self-contained. Keypoints: the value from the passage (decisive), the value from the partner (decisive), the conclusion of the comparison (decisive). Including both values means that guessing between the two names is not enough to be correct.
- **paraphrased**: ask for one or two related facts, worded so that the question shares no distinctive words with the sentence or field that holds the answer.
- **false_premise**: build the question on one wrong detail, taken from the similar passage or from a related entity in the passage, then ask a real question that the passage answers. The wrong detail must be plausible and clearly contradicted by an explicit statement in the passage; a detail that the passage merely does not mention is not a false premise. The real question must ask about a different fact from the wrong detail, so that answering it does not by itself correct the premise. Do not hint at the premise with words such as "actually" or "really". Keypoints: the correction (decisive) and the answer to the real question (decisive). The reference answer first corrects the premise, then answers.

## Output format

```json
{
  "skip": false,
  "type": "list_all | comparison | paraphrased | false_premise",
  "question": "...",
  "reference_answer": "...",
  "keypoints": [{"text": "...", "decisive": true}],
  "supporting_quotes": [{"source": "passage", "text": "..."}],
  "false_premise": "false_premise only: the wrong detail and where it comes from"
}
```

## Examples

The examples come from pages that are not in the exam. Passages are shortened.

### Example 1: list_all

PASSAGE (from https://edmed.seadatanet.org/report/858/):
```
physical and hydrographic conditions in Tees Bay, north east coast of England (1973-1996)
Data holding centre| AstraZeneca, Brixham Environmental Laboratory
Country| United Kingdom
Time period| From 1973 to 1996
...
Instruments| Benthos samplers; continuous water samplers; salinity sensor; elemental analysers; bench particle sizers; nutrient analysers; dissolved gas sensors
```

```json
{
  "skip": false,
  "type": "list_all",
  "question": "What equipment was used to collect the 1973–1996 Tees Bay monitoring data held by AstraZeneca's Brixham Environmental Laboratory?",
  "reference_answer": "Seven types of equipment: benthos samplers, continuous water samplers, a salinity sensor, elemental analysers, bench particle sizers, nutrient analysers and dissolved gas sensors.",
  "keypoints": [
    {"text": "Benthos samplers", "decisive": true},
    {"text": "Continuous water samplers", "decisive": true},
    {"text": "A salinity sensor", "decisive": true},
    {"text": "Elemental analysers", "decisive": true},
    {"text": "Bench particle sizers", "decisive": true},
    {"text": "Nutrient analysers", "decisive": true},
    {"text": "Dissolved gas sensors", "decisive": true}
  ],
  "supporting_quotes": [
    {"source": "passage", "text": "Benthos samplers; continuous water samplers; salinity sensor; elemental analysers; bench particle sizers; nutrient analysers; dissolved gas sensors"}
  ]
}
```

### Example 2: comparison

PASSAGE (from https://edmed.seadatanet.org/report/858/):
```
physical and hydrographic conditions in Tees Bay, north east coast of England (1973-1996)
Data holding centre| AstraZeneca, Brixham Environmental Laboratory
Time period| From 1973 to 1996
```

PARTNER PASSAGE (from https://edmed.seadatanet.org/report/852/):
```
# Water level data from Frigg and North Alwyn Fields, northern North Sea (1982)
Data holding centre| Total Oil Marine Plc
Time period| June 1982
```

```json
{
  "skip": false,
  "type": "comparison",
  "question": "Which covers a longer period: the Tees Bay monitoring record held by AstraZeneca's Brixham laboratory, or the water level record from the Frigg and North Alwyn Fields held by Total Oil Marine?",
  "reference_answer": "The Tees Bay record covers a much longer period: it runs from 1973 to 1996, whereas the Frigg and North Alwyn water level data cover only June 1982.",
  "keypoints": [
    {"text": "The Tees Bay record runs from 1973 to 1996", "decisive": true},
    {"text": "The Frigg and North Alwyn water level data cover June 1982", "decisive": true},
    {"text": "The Tees Bay record covers the longer period", "decisive": true}
  ],
  "supporting_quotes": [
    {"source": "passage", "text": "Time period| From 1973 to 1996"},
    {"source": "partner", "text": "Time period| June 1982"}
  ]
}
```

### Example 3: paraphrased

PASSAGE (from https://gcos.wmo.int/site/global-climate-observing-system-gcos/essential-climate-variables/surface-temperature):
```
### Global Surface Air Temperature
_Figure: Estimates of average global surface temperature anomalies (degC with respect to the 1961–1990 mean) during the instrumental period (dots: median estimates; vertical bars: 95% uncertainty ranges) and decadally-smoothed values (...) from HadCRUT4._
```

```json
{
  "skip": false,
  "type": "paraphrased",
  "question": "In the chart GCOS uses to show how global near-surface air temperature has changed, what baseline are the values measured against, and which data product do they come from?",
  "reference_answer": "The values are departures from the 1961–1990 mean, and they come from the HadCRUT4 dataset.",
  "keypoints": [
    {"text": "The values are relative to the 1961–1990 mean", "decisive": true},
    {"text": "The data come from HadCRUT4", "decisive": true}
  ],
  "supporting_quotes": [
    {"source": "passage", "text": "degC with respect to the 1961–1990 mean"},
    {"source": "passage", "text": "from HadCRUT4"}
  ]
}
```

### Example 4: false_premise

PASSAGE (from https://www.icos-cp.eu/observations/atmosphere/cal):
```
The ICOS Central Analytical Laboratories (CAL) is located in Germany and consists of two laboratories: the Flask and Calibration Laboratory (FCL) in Jena, which is hosted by the Max Planck Institute for Biogeochemistry, and the Central Radiocarbon Laboratory (CRL) in Heidelberg, which is operatedby the Institute of Environmental Physics of the University of Heidelberg.
...
The CRL quantifies the radiocarbon (14C) content of carbon dioxide (CO2) in air samples and develops methods to derive the fossil fuel contribution to atmospheric CO2 (ffCO2).
```

```json
{
  "skip": false,
  "type": "false_premise",
  "question": "The ICOS radiocarbon laboratory in Jena measures 14C in carbon dioxide from air samples. What does ICOS use these measurements to work out?",
  "reference_answer": "The premise is wrong: the ICOS Central Radiocarbon Laboratory is in Heidelberg, operated by the Institute of Environmental Physics of Heidelberg University; the laboratory in Jena is the Flask and Calibration Laboratory. The radiocarbon measurements are used to derive the fossil-fuel contribution to atmospheric CO2.",
  "keypoints": [
    {"text": "The radiocarbon laboratory is in Heidelberg, not Jena", "decisive": true},
    {"text": "The measurements are used to derive the fossil fuel contribution to atmospheric CO2", "decisive": true}
  ],
  "supporting_quotes": [
    {"source": "passage", "text": "the Central Radiocarbon Laboratory (CRL) in Heidelberg"},
    {"source": "passage", "text": "develops methods to derive the fossil fuel contribution to atmospheric CO2 (ffCO2)"}
  ],
  "false_premise": "Places the radiocarbon laboratory in Jena; in the passage Jena hosts the Flask and Calibration Laboratory, a related entity."
}
```
