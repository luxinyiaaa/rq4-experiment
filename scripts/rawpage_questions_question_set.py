"""Build the raw-web question set from current pages, resuming page by page.

Credentials and the configured model endpoint are read into memory only. They are
never written to result files or printed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, re, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests, trafilatura, yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rawpage_questions_main_stage1 as s1
from rawpage_pilot_v2 import build_prompt as build_v2_prompt

OUT = ROOT / 'outputs/rawpage_questions/question_set'
RAW_WEB = ROOT / 'data/raw_web/main'
STAGE1_DIR = ROOT / 'outputs/rawpage_questions/main'
ORDER_PATH = STAGE1_DIR / 'sampling_order.csv'
STAGE1_PAGES = STAGE1_DIR / 'pages.csv'
PROMPT_PATH = ROOT / 'prompts/teacher_generate.md'
QUESTION_TARGETS = {'dataset_record': 60, 'about_science': 240}
SITE_CAP = 60
TYPE_PAIRS = [('list_all','paraphrased'), ('paraphrased','false_premise'), ('false_premise','list_all')]
UA = s1.UA

PAGE_FIELDS = s1.PAGE_FIELDS + ['type_pair_used']
QUESTION_FIELDS = ['id','page_id','url','site','page_type','live_in_kb','similarity_label','type','type_substituted','type_substitution_reason','question','reference_answer','source_location','keypoints','supporting_quotes','false_premise','quote_checks','quote_in_page','in_kb','in_kb_any','kb_other_urls','kb_best_match','needs_kb_check','table_based','duplicate','duplicate_of','truncated','teacher_model','teacher_prompt_sha256']


def read_jsonl(path: Path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def write_jsonl(path: Path, rows: list[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    os.replace(tmp,path)

def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)

def page_file_stem(page_id: str) -> str: return re.sub(r'[^A-Za-z0-9_-]','_',page_id)

def normalize_quote(text: str) -> str:
    return ' '.join(s1.tokens(text))

def question_quote_signature(q: dict[str,Any]) -> tuple[str,...]:
    return tuple(sorted(set(normalize_quote(str(x.get('text',''))) for x in q.get('supporting_quotes',[]) if normalize_quote(str(x.get('text',''))))))

def load_stage1_inputs():
    with ORDER_PATH.open(newline='',encoding='utf-8') as f: order=list(csv.DictReader(f))
    with STAGE1_PAGES.open(newline='',encoding='utf-8') as f: stage1_pages=list(csv.DictReader(f))
    attempted=[r for r in stage1_pages if r.get('http_status')!='not_fetched_site_cap']
    attempted_ids={r['id'] for r in attempted}
    ordered_by_id={r['id']:r for r in order}
    first=[ordered_by_id[r['id']] for r in attempted if r['id'] in ordered_by_id]
    remaining=[r for r in order if r['id'] not in attempted_ids]
    return order, stage1_pages, first+remaining

def load_saved_fetch(page: dict[str,Any]):
    stem=page_file_stem(str(page['id']))
    html_path=RAW_WEB/f'{stem}.html'; meta_path=RAW_WEB/f'{stem}.json'
    if not html_path.exists() or not meta_path.exists(): return None
    try: meta=json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception: return None
    return meta, html_path.read_bytes()

def fetch_new_page(page: dict[str,Any]):
    attempts=[]; last=None; body=b''
    for n in range(1,4):
        try:
            last=requests.get(page['url'],headers={'User-Agent':UA,'Accept':'text/html,application/xhtml+xml'},timeout=60,allow_redirects=True)
            body=last.content
            attempts.append({'attempt':n,'http_status':last.status_code})
            if last.status_code==200: break
        except requests.RequestException as exc:
            last=None; attempts.append({'attempt':n,'error':type(exc).__name__})
        if n<3: time.sleep(60)
    meta={'id':page['id'],'original_url':page['url'],'final_url':last.url if last is not None else '', 'http_status':last.status_code if last is not None else None,'content_type':last.headers.get('Content-Type','') if last is not None else '', 'fetch_time_utc':datetime.now(timezone.utc).isoformat(),'attempts':attempts,'retries':len(attempts)-1}
    if last is None: body=b''
    RAW_WEB.mkdir(parents=True,exist_ok=True); stem=page_file_stem(str(page['id']))
    (RAW_WEB/f'{stem}.html').write_bytes(body)
    (RAW_WEB/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return meta,body

def load_teacher_config():
    load_dotenv(ROOT/'.env',override=False)
    with (ROOT/'configs/models.yaml').open(encoding='utf-8') as f: allcfg=yaml.safe_load(f) or {}
    cfg=allcfg.get('judge_primary') or {}
    for key in ('model','base_url','api_key_env'):
        if not cfg.get(key): raise ValueError('judge_primary configuration is incomplete')
    key=os.getenv(str(cfg['api_key_env']))
    if not key: raise ValueError('judge_primary credential is missing from the local environment')
    endpoint=str(cfg['base_url']).rstrip('/')+'/chat/completions'
    return cfg,key,endpoint

def build_prompt():
    prompt=build_v2_prompt() + """

## Stage 2 overrides
- Follow all question content and quality rules above, but this stage's `source_location` field overrides any earlier instruction to place table location in `reference_answer`. Keep `reference_answer` to the answer only; put table row, column, and unit in `source_location`.
- Supporting quotes must be the shortest exact span (a sentence or part of one) that states the fact.
- Ask only about information specific to this page or record, not generic text repeated on all pages of the same kind.
- Do not include e-mail addresses, phone numbers, postal addresses, or private individuals in questions or answers.
- Output each question with a `source_location` string; use an empty string when not table-based.
"""
    addition='''\n\n## Raw-web question-set rules\n- For each usable page, use the assigned preferred pair of question types in the user message. If one preferred type is unsupported by this page, replace it with the remaining allowed type and give the reason in `type_reason`; the two actual question types must remain different.\n- Ask only about facts specific to this page or record. Do not ask about standard text repeated on every page of the same kind.\n- Ask what a researcher seeking data or information would ask: content, measured variables, temporal/spatial coverage, methods, access, or responsible organisation. Do not ask for a URL, identifier, file timestamp, or collection name as the answer. Skip pages without substantive content and explain why.\n- Each supporting quote must be the shortest exact span (a sentence or part of one) that states its fact.\n- `reference_answer` contains only the answer. For a table-based answer, put row, column, and unit in `source_location`, not in `reference_answer`; otherwise use an empty string.\n- Do not include e-mail addresses, phone numbers, postal addresses, or private individuals in questions or answers.\n- The two questions must rely on different decisive facts.\n- Return this JSON shape: {"skip_page":false,"skip_reason":"","questions":[{"type":"list_all|paraphrased|false_premise","type_reason":"...","question":"...","reference_answer":"...","source_location":"row; column; unit, or empty","keypoints":[{"text":"...","decisive":true}],"supporting_quotes":[{"source":"passage","text":"short exact quote"}],"false_premise":"... or empty"},{"same fields"}]}.\n'''
    return prompt+addition

def validate_generation(obj: Any, preferred: tuple[str,str], text: str):
    if not isinstance(obj,dict): return None,'response_not_object',None
    if obj.get('skip_page') is True:
        reason=obj.get('skip_reason')
        return ([],None,str(reason).strip()) if isinstance(reason,str) and reason.strip() else (None,'skip_page_missing_reason',None)
    qs=obj.get('questions')
    if not isinstance(qs,list) or len(qs)!=2: return None,'expected_two_questions',None
    errors=[]
    for i,q in enumerate(qs,1):
        if not isinstance(q,dict): errors.append(f'question_{i}_not_object'); continue
        if q.get('type') not in ('list_all','paraphrased','false_premise'): errors.append(f'question_{i}_invalid_type')
        for field in ('type_reason','question','reference_answer'):
            if not isinstance(q.get(field),str) or not q[field].strip(): errors.append(f'question_{i}_missing_{field}')
        if 'source_location' not in q: q['source_location']=''
        if not isinstance(q['source_location'],str): errors.append(f'question_{i}_invalid_source_location')
        points=q.get('keypoints')
        if not isinstance(points,list) or not points or any(not isinstance(k,dict) or not isinstance(k.get('text'),str) or not k['text'].strip() or not isinstance(k.get('decisive'),bool) for k in points): errors.append(f'question_{i}_invalid_keypoints')
        quotes=q.get('supporting_quotes')
        if not isinstance(quotes,list) or not quotes or any(not isinstance(z,dict) or z.get('source')!='passage' or not isinstance(z.get('text'),str) or not z['text'].strip() for z in quotes): errors.append(f'question_{i}_invalid_quotes')
        if q.get('type')=='false_premise' and not isinstance(q.get('false_premise'),str): errors.append(f'question_{i}_missing_false_premise')
        if isinstance(quotes,list):
            for z in quotes:
                if isinstance(z,dict) and z.get('text') and z['text'] not in text: errors.append(f'question_{i}_quote_not_verbatim')
            if any(re.search(r'\|[^\n|]+\|', str(z.get('text',''))) for z in quotes if isinstance(z,dict)) and not str(q.get('source_location','')).strip(): errors.append(f'question_{i}_table_source_location_missing')
        if s1.contains_sensitive_contact(str(q.get('question',''))+' '+str(q.get('reference_answer',''))): errors.append(f'question_{i}_sensitive_content')
    types=[q.get('type') for q in qs if isinstance(q,dict)]
    if len(set(types))!=2: errors.append('question_types_not_different')
    if len(types)==2 and set(types)!=set(preferred):
        if not all(isinstance(q.get('type_reason'),str) and q['type_reason'].strip() for q in qs): errors.append('type_substitution_missing_reason')
    if not errors:
        sig0={normalize_quote(z['text']) for z in qs[0]['supporting_quotes']}; sig1={normalize_quote(z['text']) for z in qs[1]['supporting_quotes']}
        dec0={normalize_quote(k['text']) for k in qs[0]['keypoints'] if k['decisive']}; dec1={normalize_quote(k['text']) for k in qs[1]['keypoints'] if k['decisive']}
        if sig0 & sig1 or dec0 & dec1: errors.append('questions_reuse_decisive_fact_or_quote')
    return (qs,None,None) if not errors else (None,';'.join(dict.fromkeys(errors)),None)

def call_teacher(endpoint,key,model,extra_body,prompt,page,text,prompt_hash,preferred):
    truncated=len(text)>15000; excerpt=text[:15000]
    user=('Generate two questions from the current page text below. Use only this text.\n'
          f"PAGE URL: {page['url']}\nPAGE TYPE: {page['page_type']}\n"
          f"PREFERRED TYPE PAIR (in this order): {preferred[0]}, {preferred[1]}.\n"
          'If a preferred type is unsupported, substitute the remaining allowed type and explain why in type_reason.\n'
          f"INPUT NOTE: {'Only the first 15,000 characters are supplied; truncated.' if truncated else 'Full extracted text supplied.'}\nPAGE TEXT:\n<<<\n{excerpt}\n>>>")
    messages=[{'role':'system','content':prompt},{'role':'user','content':user}]
    total=Counter(); call_count=0; response_format=True; last_error='invalid_response'; actual_model=model
    for correction in range(2):
        msgs=messages if correction==0 else messages+[{'role':'user','content':f'Your previous response failed validation: {last_error}. Return a corrected response in the exact required JSON shape, or skip the page with a reason.'}]
        payload=None
        for attempt in range(3):
            req={'model':model,'messages':msgs,'temperature':0}
            if response_format: req['response_format']={'type':'json_object'}
            req.update(extra_body or {}); call_count+=1
            try:
                resp=requests.post(endpoint,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},json=req,timeout=180)
                if response_format and resp.status_code in (400,422) and 'response_format' in resp.text.lower() and any(w in resp.text.lower() for w in ('unsupported','not supported','unknown parameter','unrecognized')):
                    response_format=False; continue
                if resp.status_code==429 or resp.status_code>=500:
                    if attempt<2: time.sleep(2**attempt); continue
                    return {'error':'teacher_http_'+str(resp.status_code),'calls':call_count,'usage':dict(total),'model':actual_model,'truncated':truncated}
                if not resp.ok: return {'error':'teacher_http_'+str(resp.status_code),'calls':call_count,'usage':dict(total),'model':actual_model,'truncated':truncated}
                payload=resp.json(); break
            except requests.RequestException as e:
                last_error=type(e).__name__
                if attempt<2: time.sleep(2**attempt); continue
                return {'error':'teacher_request_'+last_error,'calls':call_count,'usage':dict(total),'model':actual_model,'truncated':truncated}
        if payload is None: continue
        actual_model=str(payload.get('model') or model)
        usage=payload.get('usage') or {}
        for k in ('prompt_tokens','completion_tokens','total_tokens'): total[k]+=int(usage.get(k,0) or 0)
        try: content=payload['choices'][0]['message'].get('content') or ''; value=json.loads(content)
        except Exception: last_error='invalid_json'; continue
        questions,err,skip=validate_generation(value,preferred,text)
        if skip is not None: return {'skip_reason':skip,'calls':call_count,'usage':dict(total),'model':actual_model,'truncated':truncated}
        if err: last_error=err; continue
        return {'questions':questions,'calls':call_count,'usage':dict(total),'model':actual_model,'truncated':truncated}
    return {'error':'generation_failed:'+last_error,'calls':call_count,'usage':dict(total),'model':actual_model,'truncated':truncated}

def make_question(page,q,slot,text,raw_index,live_kb,sim_label,teacher,prompt_hash,earlier_signatures,preferred):
    checks=[]; all_other=[]
    for entry in q.get('supporting_quotes',[]):
        quote=entry['text']; qtokens=s1.tokens(quote); own=raw_index.matches_url(qtokens,page['url'])
        other=raw_index.other_urls(qtokens,page['url'],limit=3) if len(qtokens)>=8 and not own else []
        fuzzy,context=s1.fuzzy_score_and_context(quote,raw_index.chunks.get(page['url'],[]))
        checks.append({'source':entry.get('source','passage'),'quote':quote,'quote_in_page':quote in text,'in_kb':own,'quote_kb_fuzzy':round(fuzzy,2),'kb_best_match':'' if own else context,'in_kb_any':own or bool(other),'kb_other_urls':other,'word_tokens':len(qtokens)})
        all_other.extend(other)
    in_kb=s1.label_matches([c['in_kb'] for c in checks]); in_any=s1.label_matches([c['in_kb_any'] for c in checks])
    needs='yes' if in_kb!='all' or in_any!='all' else 'no'
    table=any(bool(re.search(r'\|[^\n|]+\|',c['quote'])) for c in checks)
    type_sub=q['type'] not in preferred
    sig=question_quote_signature(q)
    duplicate_of=''
    if sig:
        previous=earlier_signatures.get(sig)
        if previous and previous[0] != page['id']:
            duplicate_of=previous[1]
    lines=[]
    for k in q.get('keypoints',[]): lines.append(('[decisive] ' if k['decisive'] else '[non-decisive] ')+k['text'])
    lines.extend(f"[source: {z.get('source','passage')}] {z['text']}" for z in q.get('supporting_quotes',[]))
    best='\n'.join(f"{c['quote']} [fuzzy={c['quote_kb_fuzzy']:.2f}] {c['kb_best_match']}" for c in checks if not c['in_kb'])
    return {'id':f"{page['id']}_{slot}",'page_id':page['id'],'url':page['url'],'site':page['site'],'page_type':page['page_type'],'live_in_kb':live_kb,'similarity_label':sim_label,'type':q['type'],'type_substituted':'yes' if type_sub else 'no','type_substitution_reason':q.get('type_reason','') if type_sub else '', 'question':q['question'],'reference_answer':q['reference_answer'],'source_location':q.get('source_location',''),'keypoints':q['keypoints'],'supporting_quotes':q['supporting_quotes'],'false_premise':q.get('false_premise',''),'quote_checks':checks,'quote_in_page':all(c['quote_in_page'] for c in checks) if checks else False,'in_kb':in_kb,'in_kb_any':in_any,'kb_other_urls':list(dict.fromkeys(all_other))[:3],'kb_best_match':best,'needs_kb_check':needs,'table_based':'yes' if table else 'no','duplicate':'yes' if duplicate_of else 'no','duplicate_of':duplicate_of,'included_in_set':'yes','excluded_from_set_reason':'','truncated':teacher['truncated'],'teacher_model':teacher['model'],'teacher_prompt_sha256':prompt_hash}

def build_summary(page_rows,questions,elapsed):
    counts_status=Counter(str(p['http_status']) for p in page_rows); by_status=defaultdict(Counter)
    for p in page_rows: by_status[str(p['http_status'])][p['page_type']]+=1
    valid=[q for q in questions if q['duplicate']!='yes' and q.get('included_in_set','yes')=='yes']
    qtype=Counter(q['type'] for q in valid); bypt=Counter(q['page_type'] for q in valid); bysite=Counter((q['page_type'],q['site']) for q in valid)
    skips=Counter()
    for p in page_rows:
        if p.get('skip_reason'): skips[str(p['skip_reason']).split(':',1)[0]]+=1
    sim_counts=Counter((q['similarity_label'],q['in_kb']) for q in valid); sim_any=Counter((q['similarity_label'],q['in_kb_any']) for q in valid)
    duplicates=sum(q['duplicate']=='yes' for q in questions); substitutions=sum(q['type_substituted']=='yes' for q in valid)
    quote_fail=sum(not z.get('quote_in_page') for q in questions for z in q.get('quote_checks',[]))
    calls=sum(int(p.get('teacher_calls') or 0) for p in page_rows); pt=sum(int(p.get('prompt_tokens') or 0) for p in page_rows); ct=sum(int(p.get('completion_tokens') or 0) for p in page_rows); tt=sum(int(p.get('total_tokens') or 0) for p in page_rows)
    lines=['# Raw-Web Question Set — Stage 2 Summary','','## Pages processed by HTTP status and page type','','| HTTP status | dataset_record | about_science | total |','|---|---:|---:|---:|']
    for status in sorted(counts_status): lines.append(f"| {status} | {by_status[status]['dataset_record']} | {by_status[status]['about_science']} | {counts_status[status]} |")
    lines.append(f"| Total | {len([p for p in page_rows if p['page_type']=='dataset_record'])} | {len([p for p in page_rows if p['page_type']=='about_science'])} | {len(page_rows)} |")
    lines+=['',f"Usable pages: {sum(int(p.get('extracted_characters') or 0)>=600 and str(p.get('http_status'))=='200' for p in page_rows)}",'', '## Skips by reason','']
    lines += [f'- {k}: {v}' for k,v in sorted(skips.items())] or ['- None']
    lines+=['','','## Non-duplicate questions by page type','','| Page type | Questions | Target |','|---|---:|---:|']
    for p in ('dataset_record','about_science'): lines.append(f"| {p} | {bypt[p]} | {QUESTION_TARGETS[p]} |")
    lines+=['','','## Questions by type','','| Type | Count |','|---|---:|']+[f'| {k} | {v} |' for k,v in sorted(qtype.items())]
    lines+=['','','## Questions by page type and site','','| Page type | Site | Count |','|---|---|---:|']+[f'| {p} | {s} | {n} |' for (p,s),n in sorted(bysite.items())]
    lines+=['','','## Type substitutions','','| Page type | Substituted questions |','|---|---:|']
    for p in QUESTION_TARGETS: lines.append(f"| {p} | {sum(q['type_substituted']=='yes' and q['page_type']==p for q in valid)} |")
    lines+=['',f'Duplicates removed: {duplicates}','','## Question-level KB matches overall','','| Check | all | some | none |','|---|---:|---:|---:|']
    for check in ('in_kb','in_kb_any'):
        counts=Counter(q[check] for q in valid)
        lines.append(f"| {check} | {counts['all']} | {counts['some']} | {counts['none']} |")
    lines+=['','','## Question-level KB matches by similarity label','','| Similarity | Check | all | some | none |','|---|---|---:|---:|---:|']
    for label in ('unchanged','changed','very_different','unavailable'):
        for check,counter in [('in_kb',sim_counts),('in_kb_any',sim_any)]: lines.append(f"| {label} | {check} | {counter[(label,'all')]} | {counter[(label,'some')]} | {counter[(label,'none')]} |")
    lines+=['','','## Manual KB and table checks','','| Check | Questions |','|---|---:|',f"| needs_kb_check | {sum(q['needs_kb_check']=='yes' for q in valid)} |",f"| table_based | {sum(q['table_based']=='yes' for q in valid)} |",f"| Quote-check failures | {quote_fail} |",'','','## Teacher usage','',f'- Calls: {calls}',f'- Prompt tokens: {pt}',f'- Completion tokens: {ct}',f'- Total tokens: {tt}',f'- Run time seconds: {elapsed:.1f}','']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')

def run(network: bool,summary_only=False):
    start=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True)
    if summary_only:
        pages=list(csv.DictReader((OUT/'pages.csv').open(newline='',encoding='utf-8'))); questions=read_jsonl(OUT/'questions.jsonl')
        created=getattr(OUT.stat(),'st_birthtime',OUT.stat().st_ctime)
        build_summary(pages,questions,max(0,time.time()-created)); return
    order,stage1_rows,plan=load_stage1_inputs()
    existing_pages=[]
    if (OUT/'pages.csv').exists(): existing_pages=list(csv.DictReader((OUT/'pages.csv').open(newline='',encoding='utf-8')))
    completed={p['id'] for p in existing_pages}; rows_by_id={p['id']:p for p in existing_pages}
    questions=read_jsonl(OUT/'questions.jsonl')
    # Duplicate registry contains only earlier questions from different pages.
    signature_registry={}
    for q in questions:
        sig=question_quote_signature(q)
        if sig and q.get('duplicate')!='yes': signature_registry.setdefault(sig,(q['page_id'],q['id']))
    accepted=Counter(q['page_type'] for q in questions if q.get('duplicate')!='yes' and q.get('included_in_set','yes')=='yes')
    site_counts=Counter((q['page_type'],q['site']) for q in questions if q.get('duplicate')!='yes' and q.get('included_in_set','yes')=='yes')
    type_cycle=Counter()
    clean_texts=s1.load_chunk_texts(ROOT/'data/processed/chunks_clean.csv'); raw_index=s1.RawKBIndex(ROOT/'data/raw/site_pages_rows.csv')
    prompt=build_prompt(); prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
    endpoint=key=model=''; extra={}
    if network:
        cfg,key,endpoint=load_teacher_config(); model=str(cfg['model']); extra=dict(cfg.get('thinking_parameters') or {})
    # Stage 1 fetched pages must be regenerated from saved HTML; no refetch.
    stage1_map={r['id']:r for r in stage1_rows if r.get('http_status')!='not_fetched_site_cap'}
    pending=plan
    late_503=[]
    for p0 in pending:
        p={**p0,'id':str(p0['id']),'url':str(p0['url']),'page_type':str(p0['page_type']),'site':str(p0['site'])}
        typ=p['page_type']; target=QUESTION_TARGETS[typ]
        if accepted[typ]>=target: continue
        if p['id'] in completed:
            completed_row=rows_by_id.get(p['id'],{})
            if str(completed_row.get('http_status'))=='200' and int(completed_row.get('extracted_characters') or 0)>=600:
                type_cycle[typ]+=1
            continue
        if typ=='about_science' and site_counts[(typ,p['site'])]>=SITE_CAP:
            report={'id':p['id'],'url':p['url'],'final_url':'','site':p['site'],'page_type':typ,'http_status':'not_fetched_site_cap','retries':0,'extracted_characters':0,'truncated':'no','kb_in_live':'','live_in_kb':'','similarity_label':'','questions_generated':0,'questions_skipped':2,'skip_reason':'site_question_cap_reached','teacher_calls':0,'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'type_pair_used':''}
            existing_pages.append(report); completed.add(p['id']); rows_by_id[p['id']]=report
            write_csv(OUT/'pages.csv',existing_pages,PAGE_FIELDS); write_jsonl(OUT/'questions.jsonl',questions); continue
        saved=load_saved_fetch(p)
        if saved:
            meta,body=saved
        elif p['id'] in stage1_map:
            # Stage-1 attempted but no saved artifact: preserve its status and do not fetch again.
            old=stage1_map[p['id']]
            report={**old,'type_pair_used':''}
            existing_pages.append(report); completed.add(p['id']); rows_by_id[p['id']]=report
            write_csv(OUT/'pages.csv',existing_pages,PAGE_FIELDS); write_jsonl(OUT/'questions.jsonl',questions); continue
        else:
            if not network: continue
            meta,body=fetch_new_page(p)
        status=meta.get('http_status')
        text=(trafilatura.extract(body,output_format='markdown',include_tables=True,include_comments=False,include_links=True) or '') if status==200 else ''
        nchar=len(text); clean=clean_texts.get(p['url'],'')
        kb_live=s1.ngram_coverage(clean,text) if status==200 else None; live_kb=s1.ngram_coverage(text,clean) if status==200 else None
        label=s1.similarity_label(status,live_kb)
        pair=TYPE_PAIRS[type_cycle[typ]%len(TYPE_PAIRS)] if status==200 and nchar>=600 else None
        if pair is not None: type_cycle[typ]+=1
        teacher={'calls':0,'usage':{},'model':'','truncated':nchar>15000}; generated=[]; reason=''
        if status!=200:
            reason='fetch_unavailable'
            if status==503: late_503.append((p,meta))
        elif nchar<600: reason='extracted_text_under_600_characters'
        elif not network: reason='pending_network_approval'
        else:
            teacher=call_teacher(endpoint,key,model,extra,prompt,p,text,prompt_hash,pair)
            if teacher.get('error'): reason=str(teacher['error'])
            elif teacher.get('skip_reason'): reason='teacher_skip:'+teacher['skip_reason']
            else:
                for i,q in enumerate(teacher['questions'],1):
                    rec=make_question(p,q,i,text,raw_index,live_kb,label,teacher,prompt_hash,signature_registry,pair)
                    generated.append(rec)
                    sig=question_quote_signature(q)
                    if sig and not rec['duplicate_of']: signature_registry.setdefault(sig,(p['id'],rec['id']))
        if generated:
            # Keep both questions from the final page if they cross the page target by one.
            remaining=target-accepted[typ]
            take=generated[:max(0,min(len(generated),remaining+1))]
            for rec in take:
                if rec['duplicate']=='yes': continue
                if typ=='about_science' and site_counts[(typ,p['site'])]>=SITE_CAP:
                    rec['included_in_set']='no'; rec['excluded_from_set_reason']='site_question_cap_reached'
                elif accepted[typ]>=target+1:
                    rec['included_in_set']='no'; rec['excluded_from_set_reason']='target_reached'
                else:
                    accepted[typ]+=1; site_counts[(typ,p['site'])]+=1
                questions.append(rec)
            generated=take
        usage=teacher.get('usage') or {}
        report={'id':p['id'],'url':p['url'],'final_url':meta.get('final_url',''),'site':p['site'],'page_type':typ,'http_status':status if status is not None else 'request_error','retries':meta.get('retries',0),'extracted_characters':nchar,'truncated':'yes' if nchar>15000 else 'no','kb_in_live':kb_live if kb_live is not None else '', 'live_in_kb':live_kb if live_kb is not None else '', 'similarity_label':label,'questions_generated':len(generated),'questions_skipped':max(0,2-len(generated)),'skip_reason':reason,'teacher_calls':teacher.get('calls',0),'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),'total_tokens':usage.get('total_tokens',0),'type_pair_used':','.join(pair) if pair else ''}
        existing_pages.append(report); rows_by_id[p['id']]=report; completed.add(p['id'])
        write_csv(OUT/'pages.csv',existing_pages,PAGE_FIELDS); write_jsonl(OUT/'questions.jsonl',questions)
        time.sleep(1)
        if all(accepted[t]>=QUESTION_TARGETS[t] for t in QUESTION_TARGETS): break
    # Retry previously/newly fetched 503 pages once after ordinary sampling, only if targets remain unmet.
    if any(accepted[t]<QUESTION_TARGETS[t] for t in QUESTION_TARGETS):
        candidate_ids={p['id'] for p,_ in late_503}
        for old in stage1_rows:
            if str(old.get('http_status'))=='503' and old['id'] not in candidate_ids:
                candidate_ids.add(old['id']); p=next((x for x in order if x['id']==old['id']),None)
                if p: late_503.append((p,{'http_status':503}))
        for p0,_ in late_503:
            p={**p0,'id':str(p0['id']),'url':str(p0['url']),'page_type':str(p0['page_type']),'site':str(p0['site'])}; typ=p['page_type']
            if accepted[typ]>=QUESTION_TARGETS[typ] or (typ=='about_science' and site_counts[(typ,p['site'])]>=SITE_CAP): continue
            meta,body=fetch_new_page(p); status=meta.get('http_status')
            text=(trafilatura.extract(body,output_format='markdown',include_tables=True,include_comments=False,include_links=True) or '') if status==200 else ''
            nchar=len(text); clean=clean_texts.get(p['url'],''); kb_live=s1.ngram_coverage(clean,text) if status==200 else None; live_kb=s1.ngram_coverage(text,clean) if status==200 else None; label=s1.similarity_label(status,live_kb)
            pair=TYPE_PAIRS[type_cycle[typ]%3] if status==200 and nchar>=600 else None
            if pair is not None: type_cycle[typ]+=1
            teacher={'calls':0,'usage':{},'model':'','truncated':nchar>15000}; generated=[]; reason='fetch_unavailable'
            if status==200 and nchar>=600 and network:
                teacher=call_teacher(endpoint,key,model,extra,prompt,p,text,prompt_hash,pair)
                reason=teacher.get('error') or ('teacher_skip:'+teacher['skip_reason'] if teacher.get('skip_reason') else '')
                if not teacher.get('error') and not teacher.get('skip_reason'):
                    for i,q in enumerate(teacher['questions'],1):
                        rec=make_question(p,q,i,text,raw_index,live_kb,label,teacher,prompt_hash,signature_registry,pair); generated.append(rec)
                        sig=question_quote_signature(q)
                        if sig and not rec['duplicate_of']: signature_registry.setdefault(sig,(p['id'],rec['id']))
            if generated:
                remaining=QUESTION_TARGETS[typ]-accepted[typ]; generated=generated[:remaining+1]
                for rec in generated:
                    questions.append(rec)
                    if rec['duplicate']=='yes': continue
                    if typ=='about_science' and site_counts[(typ,p['site'])]>=SITE_CAP:
                        rec['included_in_set']='no'; rec['excluded_from_set_reason']='site_question_cap_reached'
                    elif accepted[typ]>=QUESTION_TARGETS[typ]+1:
                        rec['included_in_set']='no'; rec['excluded_from_set_reason']='target_reached'
                    else: accepted[typ]+=1; site_counts[(typ,p['site'])]+=1
            usage=teacher.get('usage') or {}
            report={'id':p['id'],'url':p['url'],'final_url':meta.get('final_url',''),'site':p['site'],'page_type':typ,'http_status':status if status is not None else 'request_error','retries':meta.get('retries',0),'extracted_characters':nchar,'truncated':'yes' if nchar>15000 else 'no','kb_in_live':kb_live if kb_live is not None else '','live_in_kb':live_kb if live_kb is not None else '','similarity_label':label,'questions_generated':len(generated),'questions_skipped':max(0,2-len(generated)),'skip_reason':reason,'teacher_calls':teacher.get('calls',0),'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),'total_tokens':usage.get('total_tokens',0),'type_pair_used':','.join(pair) if pair else ''}
            if p['id'] in rows_by_id:
                idx=next(i for i,x in enumerate(existing_pages) if x['id']==p['id']); existing_pages[idx]=report
            else: existing_pages.append(report)
            rows_by_id[p['id']]=report; write_csv(OUT/'pages.csv',existing_pages,PAGE_FIELDS); write_jsonl(OUT/'questions.jsonl',questions)
    write_csv(OUT/'pages.csv',existing_pages,PAGE_FIELDS); write_jsonl(OUT/'questions.jsonl',questions); build_summary(existing_pages,questions,time.monotonic()-start)

def mock_test():
    from types import SimpleNamespace
    import unittest.mock as mock
    page={'id':'MOCK-1','url':'https://example.invalid/page','site':'example.invalid','page_type':'dataset_record'}
    def item(qtype,text):
        return {'type':qtype,'type_reason':'The page supports this form.','question':'What fact is stated?','reference_answer':text,'source_location':'','keypoints':[{'text':text,'decisive':True}],'supporting_quotes':[{'source':'passage','text':text}],'false_premise':''}
    payload={'model':'mock-teacher','usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5},'choices':[{'message':{'content':json.dumps({'skip_page':False,'questions':[item('list_all','Alpha fact.'),item('paraphrased','Beta fact.')]})}}]}
    response=SimpleNamespace(status_code=200,ok=True,text='',json=lambda:payload)
    with mock.patch('requests.post',return_value=response) as post:
        got=call_teacher('mock-endpoint', 'mock-key', 'mock-model', {}, 'mock prompt', page, 'Alpha fact. Beta fact.', 'hash', ('list_all','paraphrased'))
    assert got.get('questions') and got['calls']==1 and post.call_count==1
    assert got['questions'][0]['type']=='list_all' and got['usage']['total_tokens']==5
    valid,err,skip=validate_generation({'skip_page':False,'questions':[item('list_all','Alpha fact.'),item('false_premise','Beta fact.')]},('list_all','paraphrased'),'Alpha fact. Beta fact.')
    assert err is None and skip is None and len(valid)==2
    assert valid[1]['type_reason']
    print('mock checks passed: teacher JSON parsing, preferred pair, and documented type substitution')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--summary-only',action='store_true'); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--mock-test',action='store_true'); args=ap.parse_args()
    if args.mock_test: mock_test()
    elif args.summary_only: run(False,True)
    elif args.prepare_only:
        order, stage1_pages, plan = load_stage1_inputs()
        print(f'Preflight passed: {len(order)} pages in the frozen sampling order; {len(stage1_pages)} stage-1 page records; {len(plan)} ordered candidates.')
    else: run(True,False)
if __name__=='__main__': main()
