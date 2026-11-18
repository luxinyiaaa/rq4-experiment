# Response relevance

Judge sees the question and answer. Return JSON `{"score": <integer 1-5>, "reason": "<one sentence>"}`.

Rate response relevance on a scale of 1 to 5. Relevance measures only how much of the answer is about something other than the thing the question asks about. Do not consider correctness, completeness, length, or whether the answer abstains; other metrics measure these.

The thing the question asks about is the specific record, project, organisation, event or concept named in the question, including its parts and properties.

Sort the content of the answer into three kinds. Count each sentence, bullet point or table row as one unit.

(A) Addresses the question: content that answers what was asked, whether right or wrong; correcting a false premise; saying that the information was not found or is not in the documentation; explanations, background or further facts about the thing the question asks about; one sentence saying which organisation or project it belongs to.

(B) Ignore: source lines, URLs and citation markers; disclaimers such as 'unverified' or 'based on general knowledge'; offers of further help, requests for a link or source, and pointers to where the information might be found; headings and formatting.

(C) About something else: content about a different record, project, organisation, event or topic, including related projects and other activities of the parent organisation; general background about the field that is not about the thing asked; descriptions of what unrelated pages contain.

If unsure whether a unit is A or C, ask whether it is still about the thing the question asks about; if yes, it is A.

Score: 5 = no C; 4 = one or two units of C; 3 = three or more units of C, about as much as A; 2 = clearly more C than A, and A is at most one sentence; 1 = no A (the answer addresses a different question).

Give the score and a one-sentence reason that names the C content, if there is any.
