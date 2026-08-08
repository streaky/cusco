#!/usr/bin/env python3
import json, os, statistics, sys, time, urllib.error, urllib.request
from pathlib import Path
BASE=os.environ.get('CUSCO_SMOKE_BASE_URL','http://127.0.0.1:8080').rstrip('/')
TOKEN=os.environ.get('CUSCO_SMOKE_TOKEN','smoke-report-token')
OUT=Path(os.environ.get('CUSCO_SMOKE_OUTPUT','/results/smoke-report.json'))

def call(method,path,payload=None):
    data=None if payload is None else json.dumps(payload,separators=(',',':')).encode()
    headers={'Authorization':f'Bearer {TOKEN}','Accept':'application/json'}
    if data is not None: headers['Content-Type']='application/json'
    start=time.perf_counter_ns(); req=urllib.request.Request(BASE+path,data=data,headers=headers,method=method)
    try:
        with urllib.request.urlopen(req,timeout=60) as r: status,ctype,raw=r.status,r.headers.get('content-type',''),r.read()
    except urllib.error.HTTPError as e: status,ctype,raw=e.code,e.headers.get('content-type',''),e.read()
    elapsed=(time.perf_counter_ns()-start)/1_000_000
    try: body=json.loads(raw) if raw else None
    except json.JSONDecodeError: body=raw.decode('utf-8','replace')
    return status,ctype,body,elapsed

def status_ok(s): assert s==200,f'expected HTTP 200, got {s}'
def has(status,body,key): status_ok(status); assert isinstance(body,dict) and key in body,f'missing {key}'
def list_field(status,body,key): has(status,body,key); assert isinstance(body[key],list),f'{key} is not a list'
def imported(status,body): has(status,body,'tokens'); assert body['tokens']==['smoke','context'],'wrong imported tokens'
def run():
    cases=[('openapi','GET','/openai/v1/openapi.json',None,lambda s,b:has(s,b,'paths')),('models','GET','/openai/v1/models',None,lambda s,b:list_field(s,b,'data')),('completion','POST','/openai/v1/completions',{'model':'gemma-4-e2b-it','prompt':'Reply with exactly: smoke-ok','max_tokens':8,'temperature':0},lambda s,b:list_field(s,b,'choices')),('chat','POST','/openai/v1/chat/completions',{'model':'gemma-4-e2b-it','messages':[{'role':'user','content':'Reply with exactly: smoke-ok'}],'max_tokens':8,'temperature':0},lambda s,b:list_field(s,b,'choices')),('responses','POST','/openai/v1/responses',{'model':'gemma-4-e2b-it','input':'Reply with exactly: smoke-ok','max_output_tokens':8},lambda s,b:has(s,b,'output')),('context_create','POST','/cusco/v1/contexts',{},lambda s,b:has(s,b,'id')),('context_import','POST','/cusco/v1/contexts/import',{'tokens':['smoke','context']},imported),('compaction_strategies','GET','/cusco/v1/compaction/strategies',None,lambda s,b:list_field(s,b,'strategies')),('status','GET','/cusco/v1/status',None,lambda s,b:status_ok(s))]
    results=[]; lat=[]; context_id=None; stats={'requests':0,'passed':0,'failed':0,'input_tokens':0,'output_tokens':0}
    for name,method,path,payload,check in cases:
        s,c,b,ms=call(method,path,payload); lat.append(ms); stats['requests']+=1; error=None
        if isinstance(b,dict) and isinstance(b.get('usage'),dict): stats['input_tokens']+=int(b['usage'].get('prompt_tokens',b['usage'].get('input_tokens',0)) or 0); stats['output_tokens']+=int(b['usage'].get('completion_tokens',b['usage'].get('output_tokens',0)) or 0)
        try: check(s,b); passed=True; stats['passed']+=1
        except AssertionError as e: passed=False; error=str(e); stats['failed']+=1
        if name == 'context_import' and passed:
            context_id = b['id']
        results.append({'name':name,'method':method,'path':path,'status':s,'latency_ms':round(ms,3),'passed':passed,'error':error})
    if context_id is not None:
        s,c,b,ms=call('POST','/openai/v1/completions',{'model':'gemma-4-e2b-it','prompt':'compact this context','context_id':context_id,'max_tokens':1,'compaction':{'strategy_preferences':['window_tail'],'target_tokens':1}})
        lat.append(ms); stats['requests']+=1; error=None
        try:
            has(s,b,'cusco'); result=b['cusco'].get('compaction_result'); assert result and result['success'] and result['selected_strategy_id']=='window_tail:v1','missing successful window-tail result'; passed=True; stats['passed']+=1
        except AssertionError as e: passed=False; error=str(e); stats['failed']+=1
        results.append({'name':'window_tail_compaction','method':'POST','path':'/openai/v1/completions','status':s,'latency_ms':round(ms,3),'passed':passed,'error':error})
        s,c,b,ms=call('POST','/openai/v1/completions',{'model':'gemma-4-e2b-it','prompt':'continue after compaction','context_id':context_id,'max_tokens':1})
        lat.append(ms); stats['requests']+=1; error=None
        if isinstance(b,dict) and isinstance(b.get('usage'),dict): stats['input_tokens']+=int(b['usage'].get('prompt_tokens',0) or 0); stats['output_tokens']+=int(b['usage'].get('completion_tokens',0) or 0)
        try: list_field(s,b,'choices'); passed=True; stats['passed']+=1
        except AssertionError as e: passed=False; error=str(e); stats['failed']+=1
        results.append({'name':'compacted_successor_continuation','method':'POST','path':'/openai/v1/completions','status':s,'latency_ms':round(ms,3),'passed':passed,'error':error})
    stats['latency_ms']={'count':len(lat),'min':round(min(lat),3),'median':round(statistics.median(lat),3),'max':round(max(lat),3)}
    report={'schema_version':1,'deterministic':True,'base_url':BASE,'scenarios':results,'stats':stats}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return 0 if stats['failed']==0 else 1
if __name__=='__main__': sys.exit(run())
