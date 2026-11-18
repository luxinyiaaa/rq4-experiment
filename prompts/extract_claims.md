# Page claim extraction

Extract the two to four most useful, standalone factual claims from each supplied knowledge-base page; return fewer when the page supports fewer, and never return more than eight. A claim must contain concrete information such as a named entity, number, year, place, property, or explicit relationship. Exclude navigation, generic definitions, promotional language, and facts that depend on missing context.

For every claim, return the exact quote copied character-for-character from one supplied chunk, its chunk ID, a concise normalized statement, a subject, predicate/property, value when present, and named entities. The quote must be a substring of the cited chunk. Do not combine information from different chunks into one claim. If no useful claims exist for a page, return an empty claim list.

Return JSON only:
{"pages":[{"url":"...","claims":[{"claim":"...","quote":"...","chunk_id":"...","subject":"...","property":"...","value":"...","entities":["..."]}]}]}

Only keep entities that are specific proper names: organizations, networks, programmes, instruments, stations, or variables. Exclude generic words such as research infrastructure, data, project, network, and service unless they are part of a specific proper name.

## User template

Extract claims from these pages and chunks:
