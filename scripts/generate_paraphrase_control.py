"""Generate source-worded comparison questions only; retrieval is intentionally omitted."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
from typing import Any
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from scripts.build_index import _read_jsonl,_write_jsonl
from scripts.check_questions import _write_xlsx_minimal
from rq4.llm import call_llm

SYSTEM="""You are rewriting a factual question for a retrieval experiment. Write one alternative question that asks for exactly the same fact and has exactly the same answer, while using the source passage's own wording for the fact as much as possible. Preserve the named entity and scope. Do not add assumptions, ask a broader question, reveal the answer, or change the answer. Return JSON only: {\"source_worded_question\": \"...\"}."""

def main()->None:
    out=ROOT/"outputs"/"final";qs=[q for q in _read_jsonl(out/"questions_final.jsonl") if q.get("type")=="paraphrased" and q.get("status")=="generated"]
    if len(qs)!=14: raise ValueError(f"Expected 14 paraphrased questions; found {len(qs)}")
    h=hashlib.sha256(SYSTEM.encode()).hexdigest(); rows=[]
    for i,q in enumerate(qs,1):
        quotes="\n".join(str(x.get("text","")) for x in q.get("supporting_quotes",[]))
        user=f"Question: {q['question']}\n\nReference answer (must remain unchanged): {q['reference_answer']}\n\nSupporting source wording:\n{quotes}\n\nRelevant source passage:\n{q.get('text','')}"
        response=call_llm("teacher",[{"role":"system","content":SYSTEM},{"role":"user","content":user}],json_mode=True,temperature=0.0,prompt_sha256=h,log_messages=True)
        data=response.get("json")
        if not isinstance(data,dict) or not str(data.get("source_worded_question","")).strip(): raise ValueError(f"Teacher returned invalid JSON for {q['id']}: {response.get('text')}")
        rows.append({"id":q["id"],"type":q["type"],"original_question":q["question"],"source_worded_question":str(data["source_worded_question"]).strip(),"reference_answer_unchanged":q["reference_answer"],"supporting_quotes":quotes,"source_passage":q.get("text",""),"teacher_model_id":response.get("api_model",response.get("model")),"prompt_sha256":h,"human_decision":"","human_note":""})
        print(f"Rewording {i}/14: {q['id']}",flush=True)
    _write_xlsx_minimal(pd.DataFrame(rows),out/"paraphrase_control.xlsx")
    _write_jsonl(out/"paraphrase_control.jsonl",rows)
    print(f"Wrote {out/'paraphrase_control.xlsx'}; retrieval was not run.")

if __name__=="__main__":main()
