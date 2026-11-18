"""Create final keypoint-form and quote-splitting diagnostics."""
from __future__ import annotations
import csv,json,re,sys
from pathlib import Path
from typing import Any
import pandas as pd
from scripts.build_index import _read_jsonl
from scripts.score_answers import _score_number

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"/"final"

def keypoint_form(text:str)->str:
    words=re.findall(r"[A-Za-z]+",text)
    verbish=bool(re.search(r"\b(is|are|was|were|has|have|had|does|do|did|measures|evaluates|includes|contains|provides|uses|records|supports|allows|offers|covers|refers|represents|indicates|shows|describes|requires|consists|can|will|should|must|published|associated|starts|covers|run|runs|developed|intended)\b",text,re.I))
    if verbish: return "propositional_clause"
    if len(words)<=3 or (text.isupper() and len(text.split())<=3): return "entity_field_or_value"
    return "noun_phrase_or_action_label"

def main()->None:
    questions=_read_jsonl(OUT/"questions_final.jsonl"); scores=_read_jsonl(OUT/"scores_final.jsonl"); chunks={}
    with (ROOT/"data"/"processed"/"chunks_clean.csv").open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f): chunks[str(row["id"])]=row["content"]
    qmap={q["id"]:q for q in questions}; byid={}
    for s in scores: byid.setdefault(s["id"],{})[s["condition"]]=s
    kp_rows=[]; quote_rows=[]
    for q in questions:
        for n,kp in enumerate(q.get("keypoints",[]),1):
            text=str(kp.get("text","")); formats=keypoint_form(text)
            kp_rows.append({"id":q["id"],"type":q.get("type"),"keypoint_number":n,"decisive":bool(kp.get("decisive")),"format":formats,"word_count":len(text.split()),"text":text,"C1_correct":(byid.get(q["id"],{}).get("C1",{}).get("ragchecker") or {}).get("correctness"),"C2_correct":(byid.get(q["id"],{}).get("C2",{}).get("ragchecker") or {}).get("correctness"),"C4_correct":(byid.get(q["id"],{}).get("C4",{}).get("ragchecker") or {}).get("correctness")})
        supports=q.get("supporting_quotes",[]); split=any(len(x.get("chunk_ids",[]))>1 for x in supports)
        quote_rows.append({"id":q["id"],"type":q.get("type"),"quote_count":len(supports),"quote_split":split,"split_quotes":" || ".join(x.get("text","") for x in supports if len(x.get("chunk_ids",[]))>1),"gold_chunk_ids":";".join(q.get("gold_chunk_ids",[])),"C4_correct":(byid.get(q["id"],{}).get("C4",{}).get("ragchecker") or {}).get("correctness")})
    kpdf=pd.DataFrame(kp_rows); kpdf.to_csv(OUT/"keypoint_form_diagnostic.csv",index=False)
    qdf=pd.DataFrame(quote_rows); qdf.to_csv(OUT/"quote_split_diagnostic.csv",index=False)
    agg=qdf.groupby("quote_split",dropna=False).agg(n=("id","count"),correct=("C4_correct",lambda x:sum(v is True for v in x)),accuracy=("C4_correct",lambda x:sum(v is True for v in x)/len(x) if len(x) else None)).reset_index()
    from scripts.supplement_final_analysis import wilson
    lows=[]; highs=[]
    for _,r in agg.iterrows():
        lo,hi=wilson(int(r["correct"]),int(r["n"]));lows.append(lo);highs.append(hi)
    agg["wilson95_lower"]=lows;agg["wilson95_upper"]=highs;agg.to_csv(OUT/"quote_split_by_c4_accuracy.csv",index=False)
    reverse_path=OUT/"answer2response_checks.jsonl"
    if reverse_path.exists():
        rev=_read_jsonl(reverse_path); scores_out=[]; diff=[]
        from scripts.build_index import _write_jsonl
        _write_jsonl(OUT/"answer2response_preflight_examples.jsonl",rev[:3])
        score_map={(s["id"],s["condition"]):s for s in scores}
        for r in rev:
            old=(score_map[(r["id"],r["condition"])].get("ragchecker") or {}).get("correctness")
            scores_out.append({"id":r["id"],"condition":r["condition"],"extra_contradiction":r["extra_contradiction"],"correctness_v2":r["correctness_v2"],"original_correctness":old,"answer2response_claim_count":len(r["answer_claims"]),"contradiction_claims":" || ".join(str(c) for c,l in zip(r["answer_claims"],r["answer2response_labels"]) if l=="Contradiction")})
            if old!=r["correctness_v2"]: diff.append(scores_out[-1])
        pd.DataFrame(scores_out).to_csv(OUT/"correctness_v2_comparison.csv",index=False)
        pd.DataFrame(diff).to_csv(OUT/"correctness_disagreements.csv",index=False)
    print("Keypoint form counts:");print(kpdf.groupby(["type","format"]).size().rename("count").reset_index().to_string(index=False))
    print("Quote split x C4 correctness:");print(agg.to_string(index=False))

if __name__=="__main__":main()
