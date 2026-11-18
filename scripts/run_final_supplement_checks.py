"""Run missing final checks without regenerating experimental answers."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from scripts.build_index import _read_jsonl,_write_jsonl
from scripts.check_questions import GENERATOR_PROMPT,load_generator_systems,run_closed_book_only,ragchecker_response2answer,_write_xlsx_minimal
from scripts.score_answers import _make_ragas_client,_score_number
from rq4.llm import call_llm
import pandas as pd

OUT=ROOT/"outputs"/"final"

def run_reverse(scores:list[dict[str,Any]],questions:dict[str,dict[str,Any]])->list[dict[str,Any]]:
    from ragchecker import RAGChecker
    from ragchecker.container import RAGResult,RAGResults
    config=yaml.safe_load((ROOT/"configs"/"models.yaml").read_text(encoding="utf-8"))["judge"]
    out=[]
    targets=[s for s in scores if s.get("condition") in {"C1","C2","C4"}]
    for i,s in enumerate(targets,1):
        q=questions[s["id"]];traces=[]
        result=RAGResult(query_id=f"{s['id']}-{s['condition']}-answer2response",query=s["question"],gt_answer=s["reference_answer"],response=s["answer"],retrieved_context=[],gt_answer_claims=None)
        def judge(prompts:list[str])->list[str]:
            def request(prompt:str)->tuple[str,dict[str,Any]]:
                is_label="If the claim is supported by ANY passage" in prompt
                response=call_llm("judge",[{"role":"user","content":prompt}],temperature=0.0,extra_body={"enable_thinking":False},max_tokens=10 if is_label else 1024,log_messages=True)
                raw=str(response.get("text","")).strip()
                trace={"prompt":prompt,"raw_response":raw,"model_id":response.get("api_model",response.get("model")),"usage":response.get("usage"),"request_parameters":response.get("request_parameters"),"cache_hit":response.get("cache_hit"),"call_type":"label" if is_label else "claim_extraction"}
                return raw,trace
            with ThreadPoolExecutor(max_workers=6) as pool:
                results=list(pool.map(request,prompts))
            responses=[x[0] for x in results]
            traces.extend(x[1] for x in results)
            return responses
        checker=RAGChecker(extractor_name=f"openai/{config['model']}",checker_name=f"openai/{config['model']}",extractor_api_base=config["base_url"],checker_api_base=config["base_url"],extractor_max_new_tokens=1024,joint_check=False,custom_llm_api_func=judge)
        checker.check_claims(RAGResults(results=[result]),check_type="answer2response")
        labels=[str(x).strip() for x in (result.answer2response or [])]
        claims=list(result.response_claims or [])
        if len(labels)!=len(claims): raise ValueError(f"answer2response claim-label mismatch for {s['id']} {s['condition']}: {len(claims)} claims, {len(labels)} labels")
        allowed={"Entailment","Neutral","Contradiction"}
        if any(x not in allowed for x in labels): raise ValueError(f"Unexpected answer2response labels for {s['id']} {s['condition']}: {labels}")
        extra=any(x=="Contradiction" for x in labels)
        kp=[str(x).casefold() for x in (s.get("ragchecker") or {}).get("keypoint_labels",[])]
        qpoints=q.get("keypoints",[])
        decisive=[j for j,x in enumerate(qpoints) if x.get("decisive")]
        kp_entails=bool(decisive) and all(j<len(kp) and kp[j]=="entailment" for j in decisive)
        kp_no_contra=not any(x=="contradiction" for x in kp)
        v2=kp_entails and kp_no_contra and not extra
        out.append({"id":s["id"],"condition":s["condition"],"answer_claims":claims,"answer2response_labels":labels,"extra_contradiction":extra,"correctness_v2":v2,"decisive_keypoints_all_entailment":kp_entails,"no_keypoint_contradiction":kp_no_contra,"trace":traces})
        if i<=3:
            print(f"ANSWER2RESPONSE PREFLIGHT {i}: {s['id']} {s['condition']}")
            for j,t in enumerate(traces,1):
                print(f"RAW PROMPT {j} ({t['call_type']}):\n{t['prompt']}\nRAW RETURN:\n{t['raw_response']}\n")
        if i%10==0: print(f"Answer2response: {i}/{len(targets)}",flush=True)
    return out

def rerun_faithfulness(scores:list[dict[str,Any]],answers:dict[tuple[str,str],dict[str,Any]],questions:dict[str,dict[str,Any]])->dict[tuple[str,str],dict[str,Any]]:
    from ragas.metrics.collections import Faithfulness
    retry={("W004","C4"),("W016","C1"),("W016","C4"),("W061","C3"),("W073","C4")}
    repaired={}
    for i,key in enumerate(sorted(retry),1):
        s=next(x for x in scores if (x["id"],x["condition"])==key);a=answers[key];captured=[]
        client,llm=_make_ragas_client(f"faithfulness_retry_4096_{key[0]}_{key[1]}",captured,max_tokens=4096)
        metric=Faithfulness(llm=llm);ctx=[str(p.get("text","")) for p in a.get("passages",[])]
        async def work():
            try:
                statements=await metric._create_statements(s["question"],s["answer"])
                verdicts=await metric._create_verdicts(statements,"\n".join(ctx)) if statements else None
                score=metric._compute_score(verdicts) if verdicts is not None else float("nan")
                details=[] if verdicts is None else [{"statement":v.statement,"verdict":bool(v.verdict),"reason":getattr(v,"reason",None)} for v in verdicts.statements]
                return score,statements,details
            finally: await client.close()
        value,statements,details=asyncio.run(work())
        repaired[key]={"value":value,"status":"scored" if value==value else "NaN","reason":None if value==value else "No statements extracted at 4,096-token limit","statements":statements,"verdicts":details,"raw_calls":captured,"max_tokens":4096}
        print(f"Faithfulness retry {i}/5: {key[0]} {key[1]} -> {value}")
    return repaired

def prepare_human_faithfulness(answers:dict[tuple[str,str],dict[str,Any]],scores:dict[tuple[str,str],dict[str,Any]])->None:
    from ragas.metrics.collections import Faithfulness
    sample=_read_jsonl(OUT/"human_validation"/"faithfulness_sample.jsonl"); rows=[]
    for i,sel in enumerate(sample,1):
        key=(sel["id"],sel["condition"]);a=answers[key];s=scores[key]
        contexts=[str(p.get("text","")) for p in a.get("passages",[])];captured=[]
        client,llm=_make_ragas_client(f"human_faithfulness_{key[0]}_{key[1]}",captured,max_tokens=4096)
        metric=Faithfulness(llm=llm)
        async def work():
            try:
                claims=await metric._create_statements(str(a["question"]),str(a["answer"]))
                verdicts=await metric._create_verdicts(claims,"\n".join(contexts)) if claims else None
                return claims,verdicts
            finally: await client.close()
        claims,verdicts=asyncio.run(work())
        vals=[] if verdicts is None else verdicts.statements
        for j,claim in enumerate(claims):
            rows.append({"item_id":f"FV-{key[0]}-{key[1]}-{j+1}","question":a["question"],"answer":a["answer"],"passages_shown":"\n\n".join(contexts),"ragas_statement":claim,"ragas_judgment":bool(vals[j].verdict) if j<len(vals) else "Not returned","ragas_reason":getattr(vals[j],"reason","") if j<len(vals) else "","ragas_faithfulness_score":_score_number(s.get("faithfulness")),"human_agreement":"","human_rating_guide":"Enter agree or disagree."})
        print(f"Human Faithfulness detail {i}/{len(sample)}: {key[0]} {key[1]} ({len(claims)} statements)",flush=True)
    path=OUT/"human_validation";df=pd.DataFrame(rows);_write_jsonl(path/"faithfulness_rows.jsonl",rows)
    _write_xlsx_minimal(df,path/"faithfulness_check.xlsx");_write_xlsx_minimal(df.drop(columns=["human_agreement"]),path/"faithfulness_check_second_rater.xlsx")

def main()->None:
    questions=_read_jsonl(OUT/"questions_final.jsonl");scores=_read_jsonl(OUT/"scores_final.jsonl");answers=_read_jsonl(OUT/"answers_final.jsonl")
    qmap={q["id"]:q for q in questions}; amap={(a["id"],a["condition"]):a for a in answers}
    missing=[q for q in questions if q.get("closed_book_status")!="checked"]
    if missing:
        _,closed=load_generator_systems(GENERATOR_PROMPT)
        checked=run_closed_book_only(missing,closed)
        _write_jsonl(OUT/"closed_book_missing_checks.jsonl",checked)
        print(f"Closed-book supplementary checks completed: {sum(x.get('closed_book_status')=='checked' for x in checked)}/{len(checked)}")
    else: print("No missing closed-book checks.")
    reverse=run_reverse(scores,qmap)
    _write_jsonl(OUT/"answer2response_checks.jsonl",reverse)
    score_map={(s["id"],s["condition"]):s for s in scores};answer_map={(a["id"],a["condition"]):a for a in answers}
    prepare_human_faithfulness(answer_map,score_map)
    faith=rerun_faithfulness(scores,amap,qmap)
    _write_jsonl(OUT/"faithfulness_retry_4096.jsonl",[{"id":k[0],"condition":k[1],**v} for k,v in faith.items()])
    revised=[]
    revmap={(r["id"],r["condition"]):r for r in reverse}
    for original in scores:
        s=dict(original);key=(s["id"],s["condition"]);rev=revmap.get(key)
        if rev:
            s["answer2response"]={"claims":rev["answer_claims"],"labels":rev["answer2response_labels"]}
            s["extra_contradiction"]=rev["extra_contradiction"];s["correctness_v2"]=rev["correctness_v2"]
            s.setdefault("raw_outputs",{})["answer2response"]={"trace":rev["trace"]}
        if key in faith:
            s["faithfulness"]=faith[key]
            s.setdefault("raw_outputs",{})["ragas_faithfulness_retry_4096"]={k:v for k,v in faith[key].items() if k!="value"}
            s["faithfulness"]["value"]=faith[key]["value"]
        revised.append(s)
    _write_jsonl(OUT/"scores_final_revised.jsonl",revised)
    print(f"Answer2response checks: {len(reverse)}; correctness differences: {sum((r['correctness_v2'] != next(s for s in scores if s['id']==r['id'] and s['condition']==r['condition']).get('ragchecker',{}).get('correctness')) for r in reverse)}")

if __name__=="__main__":main()
