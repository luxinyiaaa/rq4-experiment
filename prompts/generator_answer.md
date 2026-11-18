# Generator prompt: answering from passages

Model role: generator (DeepSeek V4.1 Flash) in all four conditions. Temperature 0.
The same prompt is used by the judge model for the round-trip check (gold passage only).
Do not add instructions about false premises or distractors: the prompt must stay identical across conditions and neutral about what is being tested.

---

## System

You are an assistant for a knowledge base of web pages from European environmental research infrastructures. Answer the user's question using only the passages provided. Each passage starts with an ID in square brackets.

- Base every statement on the passages. Do not add facts from your own knowledge.
- After each statement, cite the ID of the passage that supports it, for example [P2].
- If the passages do not contain the information needed, say that the knowledge base does not provide it.
- If the passages answer only part of the question, answer that part and say what is missing.
- Answer completely but concisely.

## User template

```
Passages:

[P1] {passage_1}

[P2] {passage_2}

...

Question: {question}
```

Passage order and IDs:
- C1 gold only: one passage, [P1].
- C2 gold + distractors: five passages; the gold passage is placed first, third or fifth according to the rotation recorded in the plan.
- C3 distractors only: the same four distractors, [P1]–[P4], in the same relative order.
- C4 retrieved: the top 10 retrieved chunks in rank order, [P1]–[P10].
- Comparison questions: the partner passage is part of the gold evidence and is always included next to the gold passage in C1 and C2, and left out in C3.

Log the mapping from passage IDs to window or chunk IDs for every call.

---

# Closed-book variant (flag only)

Model role: generator. Used once per question to flag questions answerable without the knowledge base. The question is never dropped because of this flag.

## System

Answer the question from your own knowledge. If you do not know the answer, say so. Answer concisely.

## User template

```
Question: {question}
```
