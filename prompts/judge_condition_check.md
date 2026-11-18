# Semantic condition verification

Judge whether each listed knowledge-base record satisfies the user's question, using the supplied page fields, temporal interval, and retrieved excerpt. Interpret time conditions semantically. A record covers a requested year or interval whenever its coverage interval overlaps the requested interval. “Ongoing”, “onwards”, “since”, “present”, and “current” mean that the end is open. Do not require literal string equality. For example, an interval beginning in 1983 and continuing onward covers 1989–1992.

For `set_all`, determine which catalogue records satisfy the condition and whether any retrieved full-KB record is an additional match. For `unanswerable`, decide whether any supplied catalogue or retrieved record satisfies the full combination. Treat an absent/missing field as unknown, not as a match. Return one decision per record ID, and list all records judged to satisfy the condition.

Return JSON only:
{"records":[{"record_id":"...","satisfies":true,"reason":"..."}],"matching_record_ids":["..."],"condition_is_semantically_clear":true,"reason":"..."}

## User template

Question: {question}
Question type: {type}

Candidate records:
