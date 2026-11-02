"""Local data orchestration for AI-authored evidence reviews; no model inference.

Content labels are authored by the current conversational AI, never by these
scripts. Scripts only anonymize, validate, preserve and aggregate those labels.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import random
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .common import ROOT, digest_file, digest_json, read_jsonl, write_json, write_jsonl
from .inference import append_result

SOURCE = ROOT / 'outputs/paper_eval/dev_supplement_20261028_164300'
SEED = 20261028
RULES = '''# AI证据核查预固定协议

本轮是单一当前会话AI评价者的逐题内容阅读，不是人工专家评价、独立多评审或模型训练。
评价者：Codex当前会话AI助手；系统角色说明模型家族为GPT-6。精确执行模型ID、快照/版本、采样参数：未记录，不能独立核实。
执行方式：本会话依次阅读材料并直接撰写结构化判断；本地Python只做匿名化、校验、持久化和统计。不启动Qwen、不调用新增付费外部模型、不使用关键词规则判定内容。
评价者在此前上下文已见模型身份、汇总F1和若干例子；本轮A/B标签不显示身份或F1，但并非完全盲评。批次、自检、重复阅读均仍为同一评价者。

## 筛选和随机化
只选原始answers.jsonl的v1_rag_t160与v2_rag_t160；要求同101个ID、202条，无重复，实际上下文和提示一致。异常完整保留在input_anomalies.json，不静默删除；缺失/上下文不同时禁止合并并报告。
按题目ID字典序，每题两份回答用random.Random(20261028)独立shuffle为A/B。私有映射单独保存，不在评价查看命令中展示；身份恢复仅在全部内容判断封存后进行。
先按ID排序前5题做完整试运行；通过结构/证据定位检查后依顺序继续，不按表现替换题目。每批及时保存；续跑要求输入、匿名包、协议及执行代码签名一致。

## 实际评价提示词（每题均适用）
你是一名单一AI证据核查者。只依据题目及实际提供给模型的材料作判断，不依赖模型名称、F1、答案长度、专业文风或预期胜者。
第一步：在查看A/B前先阅读问题与所有实际片段，判定资料充分性：足以回答/只能部分回答/不足以回答/无法判断。建立必要可核实要点；每点必须给出实际片段ID和逐字短摘录。只把材料能够确定的内容列为要点；缺失资料的请求部分另行说明，不编造要点。
第二步：独立审查A、B。每项要点判覆盖/部分覆盖/遗漏/无法判断，附理由。拆分全部实质陈述，逐项判被材料支持/与材料矛盾/材料未支持/无法判断；附证据定位或明确说明缺乏证据。材料未支持不等于科学事实错误。
拒答判没有拒答/恰当拒答/错误拒答/无法判断。材料已有答案而称材料不足是错误拒答；恰当拒答可能忠于证据，却不算完成原问题。未输出完整答案不自动视为拒答。
逐个完整引用标记，核查关联陈述是否确实由所引片段支持；编号存在不等于支持成立。记录无引用、局部支持、错指来源、矛盾及无法判断。复述问题不计覆盖要点。
分别记录是否直接回答及额外内容是否影响答案（无/无害扩展/有害或偏离/无法判断）。不因参考原句匹配就判事实正确。
公开原参考只在要点锁定后辅助阅读。判与材料一致/部分可核实/不能由所给材料核实/与材料冲突/无法判断，并说明依据；不修改原参考。
第三步：依据以上独立判断选A内容更好/B内容更好/内容基本等价/各有得失/无法判断。仅在核心事实与必要要点基本一致、且没有新增实质问题时选基本等价。
差异原因可多选：错误拒答变化、关键要点变化、无支持陈述变化、矛盾陈述变化、引用支持变化、仅措辞变化、额外内容增减、其他。所有实质判断有简短理由和可定位证据；证据不足保留不确定性。

## 统计约定
全部101题均进入分母表；不确定/不适用单列，不能当零。要点层面覆盖/部分/遗漏/无法判断分别计数；覆盖比例=(覆盖+0.5*部分)/(覆盖+部分+遗漏)，无法判断点剔除并另报；无可评要点则比例NA，不当0。
回答层面分别统计至少一个无支持陈述、矛盾陈述、错误拒答，以及引用缺失/局部/不支持/无法判断。引用支持还按标记计数，分母为实际已评标记；回答无引用另列。
两份内容判断与配对结论全部封存哈希后，才能关联原per_question_metrics.jsonl中的F1。按1e-12容差分升/降/不变；交叉表保留五类内容结果，混合得失不强行视为净损失。
“F1下降且v2更差”为明确净内容损失；“各有得失”另报其中是否有要点遗漏/无支持等局部损失；“基本等价且仅措辞/无害额外内容变化”为主要表达变化。F1升高但内容没有改善的等价/更差/混合分别报告，不把混合当纯改善。
本轮不计算置信区间：单一AI判断的主观误差未估计。描述性题数不构成人工事实准确率或独立因果效应。
无法判断必须列出原因；争议例子可含参考不一致、资料不充分、混合得失、无法判断与F1/内容不一致，不从所选案例估计总体错误率。

## 内容数据简写和来源
原始批次JSON由当前会话AI逐题撰写。S/P/I/U分别是资料充分/部分/不足/不确定；cov的C/P/O/U是覆盖/部分/遗漏/无法判断；cl与ct的S/C/N/U是支持/矛盾/未支持/无法判断。
要点p按顺序编号K1…；每点含文字、片段序号、逐字摘录。陈述cl的evidence可指向K序号列表，或显式对象{c:片段序号,q:逐字摘录}列表。ct关联人工（AI）指定的陈述序号，不由脚本推断关系。所有标签与理由均由AI输入。
原实验目录保持只读。保留输入全目录哈希、代码快照、命令、批次原注释、进度与封存哈希。仅当101份证据清单、202份回答审查和101份配对判断全部验证通过时才标记完成。
'''

def stamp(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path): return read_jsonl(path) if path.exists() else []
def log(out, action, extra=None):
    append_result(out/'progress.jsonl', {'time_utc':stamp(),'action':action,**(extra or {})})
def command(out):
    append_result(out/'commands.jsonl', {'time_utc':stamp(),'cwd':str(ROOT),'argv':[sys.executable,'-m','paper_eval.ai_content_audit',*sys.argv[1:]]})
def csvfile(path, data):
    if not data: return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data)

def init():
    date=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out=ROOT/'outputs/paper_eval'/f'ai_content_audit_{date}'
    out.mkdir(exist_ok=False)
    questions=read_jsonl(SOURCE/'selected_questions.jsonl')
    raw=read_jsonl(SOURCE/'answers.jsonl')
    metrics=read_jsonl(SOURCE/'per_question_metrics.jsonl') # Ingest but do not expose scores to review.
    wanted=[r for r in raw if r['condition'] in {'v1_rag_t160','v2_rag_t160'}]
    counts=collections.Counter((r['id'],r['condition']) for r in wanted)
    idx={(r['id'],r['condition']):r for r in wanted}
    qidx={q['id']:q for q in questions}
    anomalies=[]
    for k,n in counts.items():
        if n!=1: anomalies.append({'kind':'duplicate','key':k,'count':n})
    for qid in sorted(qidx):
        for c in ('v1_rag_t160','v2_rag_t160'):
            if (qid,c) not in idx: anomalies.append({'kind':'missing','id':qid,'condition':c})
        a,b=idx.get((qid,'v1_rag_t160')),idx.get((qid,'v2_rag_t160'))
        if a and b and (a['retrieved_context']!=b['retrieved_context'] or a['prompt']!=b['prompt']):
            anomalies.append({'kind':'prompt_or_actual_context_mismatch','id':qid})
    for r in wanted:
        if r['id'] not in qidx: anomalies.append({'kind':'unexpected_id','id':r['id']})
    if len(qidx)!=101 or len(questions)!=101 or len(wanted)!=202:
        anomalies.append({'kind':'count_mismatch','questions':len(questions),'unique_ids':len(qidx),'answers':len(wanted)})
    write_json(out/'input_anomalies.json',anomalies)
    write_jsonl(out/'selected_raw_records.jsonl',wanted)
    write_json(out/'source_hashes.json',{str(p.relative_to(SOURCE)):digest_file(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()})
    (out/'review_protocol.md').write_text(RULES,encoding='utf-8')
    if anomalies: raise ValueError(f'Input anomalies preserved in {out}; resolve explicitly before judging')
    rng=random.Random(SEED); packet=[]; evidence=[]; mapping={}
    for i,qid in enumerate(sorted(qidx)):
        stages=['v1_rag_t160','v2_rag_t160']; rng.shuffle(stages)
        source=idx[(qid,stages[0])]
        q=qidx[qid]
        contexts=[{k:h.get(k) for k in ('id','text','source','section','context_truncated')} for h in source['retrieved_context']]
        base={'index':i,'id':qid,'question':q['question'],'contexts':contexts}
        evidence.append(base)
        packet.append({**base,'answers':{label:idx[(qid,c)]['answer'] for label,c in zip(('A','B'),stages)},
            'auxiliary_original_reference':q['reference_answer_origin']})
        mapping[qid]={label:c for label,c in zip(('A','B'),stages)}
    write_jsonl(out/'anonymous_review_packet.jsonl',packet)
    write_jsonl(out/'evidence_only_packet.jsonl',evidence)
    write_json(out/'private_mapping.json',{'seed':SEED,'mapping':mapping,'private':True})
    shutil.copy2(Path(__file__),out/'audit_code_snapshot.py')
    signature={'source_hashes_sha256':digest_file(out/'source_hashes.json'),'protocol_sha256':digest_file(out/'review_protocol.md'),
        'packet_sha256':digest_file(out/'anonymous_review_packet.jsonl'),'evidence_packet_sha256':digest_file(out/'evidence_only_packet.jsonl'),
        'mapping_sha256':digest_file(out/'private_mapping.json'),'code_sha256':digest_file(Path(__file__))}
    write_json(out/'audit_manifest.json',{'created_at_utc':stamp(),'signature':signature,'source':str(SOURCE),
        'reviewer':{'name':'Codex当前会话AI助手','family':'系统角色说明为GPT-6','exact_model_id':'未记录','snapshot_version':'未记录','method':'single conversational AI, sequential direct content review; no inference API calls','independent_reviewers':1,'fully_blind':False},
        'questions':101,'answers':202,'original_metric_rows_ingested':len(metrics),'signature_sha256':digest_json(signature)})
    command(out); log(out,'prepared',{'questions':101,'answers':202,'anomalies':0,'pilot_indices':list(range(5))})
    print(out)

def check(out):
    m=load(out/'audit_manifest.json'); s=m['signature']
    checks={'protocol_sha256':'review_protocol.md','packet_sha256':'anonymous_review_packet.jsonl',
        'evidence_packet_sha256':'evidence_only_packet.jsonl','mapping_sha256':'private_mapping.json','source_hashes_sha256':'source_hashes.json'}
    assert all(digest_file(out/f)==s[k] for k,f in checks.items()),'Frozen review inputs changed'
    assert digest_file(Path(__file__))==s['code_sha256'],'Execution code changed; use original snapshot or new run'
    return m

def evidence_ref(x,packet,points=None):
    if isinstance(x,int):
        assert points and 1<=x<=len(points)
        return points[x-1]['evidence']
    assert isinstance(x,dict) and x['q'].strip()
    h=packet['contexts'][int(x['c'])-1]
    assert x['q'] in h['text'],f"Quote not exact: {packet['id']} {x['q']}"
    return {'chunk_id':h['id'],'quote':x['q']}

def commit_evidence(out,path):
    packet=rows(out/'evidence_only_packet.jsonl'); existing=rows(out/'_evidence_locked.jsonl')
    seen={r['id'] for r in existing}
    for annotation in load(path):
        p=packet[annotation['i']]
        assert p['id'] not in seen,'Evidence already locked; never silently replace'
        assert annotation['s'] in ('S','P','I','U') and annotation['why'].strip()
        points=[]
        for j,(text,c,quote) in enumerate(annotation['p'],1):
            points.append({'key':f'K{j}','text':text,'evidence':evidence_ref({'c':c,'q':quote},p)})
        record={'id':p['id'],'index':p['index'],'sufficiency':annotation['s'],'sufficiency_reason':annotation['why'],
                'necessary_points':points,'locked_before_answer_display_at_utc':stamp(),
                'requested_parts_not_verifiable':annotation.get('unverifiable',''),'author':'single_current_AI'}
        append_result(out/'_evidence_locked.jsonl',record); seen.add(p['id'])
    log(out,'evidence_committed',{'file':str(path),'sha256':digest_file(path),'completed':len(seen)})

def commit_review(out,path):
    packet=rows(out/'anonymous_review_packet.jsonl'); evidence={r['id']:r for r in rows(out/'_evidence_locked.jsonl')}
    reviews=rows(out/'answer_reviews.jsonl'); pairs=rows(out/'_paired_anonymous.jsonl')
    seen={r['id'] for r in pairs}
    pending_answers=[]; pending_pairs=[]
    for a in load(path):
        p=packet[a['i']]; qid=p['id']; ev=evidence[qid]; points=ev['necessary_points']
        assert qid not in seen,'Pair already committed; preserve original judgments'
        if p['index']>=5:
            assert len([r for r in pairs if r['index']<5])==5,'Complete fixed five-question pilot first'
        for label in ('A','B'):
            r=a[label]; assert len(r['cov'])==len(points)
            coverage=[]
            for point,(status,reason) in zip(points,r['cov']):
                assert status in ('C','P','O','U') and reason.strip()
                coverage.append({'key':point['key'],'status':status,'reason':reason,'evidence':point['evidence']})
            claims=[]
            for j,(text,status,refs,reason) in enumerate(r['cl'],1):
                assert status in ('S','C','N','U') and reason.strip()
                claims.append({'claim':j,'text':text,'status':status,'reason':reason,
                    'evidence':[evidence_ref(x,p,points) for x in refs]})
            cited=[]
            for chunk,claim_ids,status,reason in r['ct']:
                assert status in ('S','C','N','U') and reason.strip()
                marker=p['contexts'][chunk-1]['id'] if isinstance(chunk,int) else chunk
                assert f'[{marker}]' in p['answers'][label],f'Citation not in answer: {qid} {label} {marker}'
                assert all(1<=k<=len(claims) for k in claim_ids)
                cited.append({'marker':marker,'claims':claim_ids,'status':status,'reason':reason,
                    'evidence':[e for k in claim_ids for e in claims[k-1]['evidence']]})
            actual=re.findall(r'\[([^\[\]]+)\]',p['answers'][label])
            assert collections.Counter(actual)==collections.Counter(c['marker'] for c in cited),f'All citations must be reviewed: {qid} {label}'
            assert r['rf'][0] in ('none','appropriate','erroneous','uncertain')
            assert r['di'][0] in ('yes','partial','no','uncertain')
            assert r['ex'][0] in ('none','benign','harmful','uncertain')
            assert r['cs'][0] in ('supported','partial','unsupported','missing','uncertain','not_applicable')
            assert all(r[k][1].strip() for k in ('rf','di','ex','cs'))
            record={'id':qid,'index':p['index'],'label':label,'answer':p['answers'][label],
                'point_coverage':coverage,'claims':claims,'refusal':r['rf'][0],'refusal_reason':r['rf'][1],
                'citation_reviews':cited,'citation_summary':r['cs'][0],'citation_reason':r['cs'][1],
                'direct_answer':r['di'][0],'direct_answer_reason':r['di'][1],
                'extra_content':r['ex'][0],'extra_content_reason':r['ex'][1],
                'reviewed_at_utc':stamp(),'reviewer':'single_current_AI','human_score':None}
            assert not any(x['id']==qid and x['label']==label for x in reviews)
            pending_answers.append(record); reviews.append(record)
        verdict,causes,reason=a['cmp']
        assert verdict in ('A_better','B_better','equivalent','tradeoff','uncertain') and reason.strip()
        assert a['ref'][0] in ('consistent','partial','unverifiable','conflict','uncertain') and a['ref'][1].strip()
        pair={'id':qid,'index':p['index'],'verdict':verdict,'causes':causes,'reason':reason,
            'reference_status':a['ref'][0],'reference_reason':a['ref'][1],'reviewed_at_utc':stamp()}
        pending_pairs.append(pair); pairs.append(pair); seen.add(qid)
    # Validate the whole batch before writing any review; malformed annotations
    # cannot leave a half-written A/B pair. Each completed prior batch is retained.
    for record in pending_answers: append_result(out/'answer_reviews.jsonl',record)
    for record in pending_pairs: append_result(out/'_paired_anonymous.jsonl',record)
    log(out,'reviews_committed',{'file':str(path),'sha256':digest_file(path),'pairs':len(seen),'answers':len(reviews)})
    if len(seen)==5 and not (out/'pilot_validation.json').exists():
        write_json(out/'pilot_validation.json',{'passed':True,'questions':5,'answers':10,'indices':list(range(5)),
            'evidence_quotes_exact':True,'points_locked_before_answer_display':True,'complete_citation_accounting':True,'time_utc':stamp()})

def view(out,kind,start,count):
    packet=rows(out/('evidence_only_packet.jsonl' if kind=='evidence' else 'anonymous_review_packet.jsonl'))
    locked={r['id'] for r in rows(out/'_evidence_locked.jsonl')}
    for p in packet[start:start+count]:
        if kind=='answers': assert p['id'] in locked,'Lock evidence checklist before answer display'
        print(json.dumps(p,ensure_ascii=False))
    log(out,f'displayed_{kind}',{'start':start,'count':count})

def finalize(out):
    ev=rows(out/'_evidence_locked.jsonl'); ar=rows(out/'answer_reviews.jsonl'); pr=rows(out/'_paired_anonymous.jsonl')
    packet=rows(out/'anonymous_review_packet.jsonl'); ids={p['id'] for p in packet}
    assert len(ev)==len(pr)==101 and len(ar)==202
    assert {r['id'] for r in ev}=={r['id'] for r in pr}==ids
    assert len({(r['id'],r['label']) for r in ar})==202
    seal={n:digest_file(out/n) for n in ('_evidence_locked.jsonl','answer_reviews.jsonl','_paired_anonymous.jsonl')}
    write_json(out/'content_judgments_sealed.json',{'time_utc':stamp(),'files':seal,'all_content_complete_before_f1_join':True})
    mapping=load(out/'private_mapping.json')['mapping']
    metric=read_jsonl(SOURCE/'per_question_metrics.jsonl')
    f1={(r['id'],r['condition']):r['f1'] for r in metric if r['condition'] in ('v1_rag_t160','v2_rag_t160')}
    qidx={q['id']:q for q in read_jsonl(SOURCE/'selected_questions.jsonl')}
    eidx={r['id']:r for r in ev}; aidx={(r['id'],r['label']):r for r in ar}
    paired=[]; summaries=[]; disputed=[]
    unblind=[]
    for p in sorted(pr,key=lambda r:r['id']):
        qid=p['id']; label1=next(k for k,v in mapping[qid].items() if v=='v1_rag_t160'); label2=next(k for k,v in mapping[qid].items() if v=='v2_rag_t160')
        verdict=p['verdict']
        result='equivalent' if verdict=='equivalent' else 'tradeoff' if verdict=='tradeoff' else 'uncertain' if verdict=='uncertain' else ('v2_better' if verdict==f'{label2}_better' else 'v2_worse')
        delta=f1[qid,'v2_rag_t160']-f1[qid,'v1_rag_t160']; direction='increase' if delta>1e-12 else 'decrease' if delta< -1e-12 else 'unchanged'
        row={'id':qid,'source_group':qidx[qid]['source_group'],'sufficiency':eidx[qid]['sufficiency'],
            'anonymous_verdict':verdict,'v2_content_result':result,'difference_causes':p['causes'],'comparison_reason':p['reason'],
            'reference_status':p['reference_status'],'reference_reason':p['reference_reason'],
            'v1_f1':f1[qid,'v1_rag_t160'],'v2_f1':f1[qid,'v2_rag_t160'],'f1_delta':delta,'f1_direction':direction}
        for stage,label in (('v1',label1),('v2',label2)):
            r=aidx[qid,label]; unblind.append({**r,'stage':stage})
            cv=collections.Counter(x['status'] for x in r['point_coverage']); denom=sum(cv[x] for x in ('C','P','O'))
            row.update({f'{stage}_refusal':r['refusal'],f'{stage}_points_covered':cv['C'],f'{stage}_points_partial':cv['P'],
                f'{stage}_points_omitted':cv['O'],f'{stage}_points_unknown':cv['U'],f'{stage}_assessable_points':denom,
                f'{stage}_weighted_point_coverage':(cv['C']+.5*cv['P'])/denom if denom else None,
                f'{stage}_unsupported_claims':sum(c['status']=='N' for c in r['claims']),
                f'{stage}_contradicted_claims':sum(c['status']=='C' for c in r['claims']),
                f'{stage}_unknown_claims':sum(c['status']=='U' for c in r['claims']),
                f'{stage}_citation_summary':r['citation_summary'],f'{stage}_direct_answer':r['direct_answer'],f'{stage}_extra_content':r['extra_content']})
        paired.append(row)
        flags=[]
        if p['reference_status']!='consistent': flags.append('reference_'+p['reference_status'])
        if eidx[qid]['sufficiency']!='S': flags.append('evidence_'+eidx[qid]['sufficiency'])
        if result in ('tradeoff','uncertain'): flags.append(result)
        if row['v1_refusal']=='erroneous' or row['v2_refusal']=='erroneous': flags.append('erroneous_refusal')
        if direction=='decrease' and result=='equivalent': flags.append('F1_down_content_equivalent')
        if direction=='increase' and result!='v2_better': flags.append('F1_up_without_clear_content_gain')
        if row['v1_unsupported_claims'] or row['v2_unsupported_claims'] or row['v1_contradicted_claims'] or row['v2_contradicted_claims']: flags.append('claim_support_issue')
        if flags: disputed.append({**row,'flags':flags,'review_status':'requires_independent_verification','human_score':None})
    write_jsonl(out/'answer_reviews_unblinded.jsonl',unblind)
    write_jsonl(out/'question_evidence_checklist.jsonl',[{**eidx[p['id']],'auxiliary_reference_status':p['reference_status'],'auxiliary_reference_reason':p['reference_reason']} for p in sorted(pr,key=lambda r:r['id'])])
    csvfile(out/'paired_reviews.csv',paired); csvfile(out/'disputed_cases.csv',disputed)
    for stage in ('v1','v2'):
        rr=[r for r in unblind if r['stage']==stage]
        def add(metric,numerator,denominator=101,unknown=0,note=''):
            summaries.append({'stage':stage,'metric':metric,'numerator':numerator,'denominator':denominator,'unknown_or_na':unknown,'value':numerator/denominator if denominator else None,'note':note})
        for status in ('none','appropriate','erroneous','uncertain'): add('refusal_'+status,sum(r['refusal']==status for r in rr))
        for status in ('S','C','N','U'): add('answers_with_claim_status_'+status,sum(any(c['status']==status for c in r['claims']) for r in rr))
        for status in ('supported','partial','unsupported','missing','uncertain','not_applicable'): add('citation_answer_'+status,sum(r['citation_summary']==status for r in rr))
        cov=[c for r in rr for c in r['point_coverage']]; valid=[c for c in cov if c['status']!='U']
        for status in ('C','P','O','U'): add('point_coverage_'+status,sum(c['status']==status for c in cov),len(cov),sum(c['status']=='U' for c in cov),'Unit: necessary point; U not treated as missed')
        add('weighted_point_coverage',sum(1 if c['status']=='C' else .5 if c['status']=='P' else 0 for c in valid),len(valid),len(cov)-len(valid))
        add('answers_with_omission',sum(any(c['status']=='O' for c in r['point_coverage']) for r in rr),sum(bool(r['point_coverage']) for r in rr),sum(not r['point_coverage'] for r in rr))
        ct=[c for r in rr for c in r['citation_reviews']]
        for status in ('S','C','N','U'): add('citation_marker_'+status,sum(c['status']==status for c in ct),len(ct),sum(c['status']=='U' for c in ct))
        for status in ('yes','partial','no','uncertain'): add('direct_answer_'+status,sum(r['direct_answer']==status for r in rr))
    for status in ('v2_better','v2_worse','equivalent','tradeoff','uncertain'):
        summaries.append({'stage':'paired','metric':status,'numerator':sum(p['v2_content_result']==status for p in paired),'denominator':101,'unknown_or_na':0,'value':sum(p['v2_content_result']==status for p in paired)/101,'note':'Single AI judgment, not human accuracy'})
    for status in ('S','P','I','U'):
        summaries.append({'stage':'evidence','metric':status,'numerator':sum(r['sufficiency']==status for r in ev),'denominator':101,'unknown_or_na':0,'value':sum(r['sufficiency']==status for r in ev)/101,'note':'Sufficient/partial/insufficient/uncertain'})
    csvfile(out/'summary.csv',summaries)
    cross=[{'f1_direction':direction,'v2_content_result':result,'questions':sum(p['f1_direction']==direction and p['v2_content_result']==result for p in paired)}
        for direction in ('increase','decrease','unchanged') for result in ('v2_better','v2_worse','equivalent','tradeoff','uncertain')]
    csvfile(out/'f1_content_crosstab.csv',cross)
    changed=[name for name,h in load(out/'source_hashes.json').items() if not (SOURCE/name).exists() or digest_file(SOURCE/name)!=h]
    assert not changed,f'Original files changed: {changed}'
    write_report(out,paired,summaries,cross,ev,unblind)
    write_json(out/'completion.json',{'complete':True,'questions':101,'answer_reviews':202,'evidence_checklists':101,
        'paired_reviews':101,'time_utc':stamp(),'all_original_source_files_unchanged':True,'protected_files':len(load(out/'source_hashes.json')),
        'uncertain_pair_questions':sum(p['v2_content_result']=='uncertain' for p in paired),'fully_blind':False,'independent_AI_reviewers':1})
    log(out,'complete',{'questions':101,'answers':202})
    write_json(out/'delivery_hashes.json',{str(p.relative_to(out)):digest_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='delivery_hashes.json'})
    print(json.dumps(load(out/'completion.json'),ensure_ascii=False))

def write_report(out,paired,summary,cross,ev,answers):
    def n(stage,metric): return next(r['numerator'] for r in summary if r['stage']==stage and r['metric']==metric)
    count=collections.Counter(p['v2_content_result'] for p in paired)
    dec=[p for p in paired if p['f1_direction']=='decrease']; inc=[p for p in paired if p['f1_direction']=='increase']
    expression=[p for p in dec if p['v2_content_result']=='equivalent' and set(p['difference_causes'])<= {'wording','extra'}]
    err_up=sum(p['v1_refusal']!='erroneous' and p['v2_refusal']=='erroneous' for p in paired)
    err_down=sum(p['v1_refusal']=='erroneous' and p['v2_refusal']!='erroneous' for p in paired)
    lines=['# 现有模型回答的AI证据核查','',f'结果目录：`{out}`。已逐题审查101题、202份RAG回答，只比较v1_rag_t160与v2_rag_t160。','',
        '本轮是一个当前会话AI评价者的证据阅读与判断，不是人工专家评价。模型精确ID/版本未记录；系统角色说明家族为GPT-6。没有本地模型推理、训练或新增外部模型调用。评价者此前已知道身份和部分F1，故匿名A/B只能降低当前展示中的身份线索，不能称完全盲评。重复阅读与批次不是独立评审。','',
        '## 数据和执行','',
        '输入检查：两个条件各101个相同ID，共202份，无重复/缺失，逐题实际检索上下文及提示一致。原实验全目录哈希在开始前保存，并在结尾复核；原文件不改。','',
        '评价规则与提示词先保存在review_protocol.md。先对ID排序前5题完成资料→要点锁定→匿名回答→配对判断，检查通过后继续其余题目。每个要点有实际片段ID和逐字摘录；实质陈述和每个引用有支持/矛盾/未支持/无法判断的理由。脚本只做数据处理和统计，内容标签由当前AI逐题撰写。','',
        '全部101题的内容判断先封存content_judgments_sealed.json，随后才关联原F1；但此前已知汇总和部分案例的先验暴露仍是限制。数据筛选没有依据F1或胜负。','',
        '## 内容判断汇总','',
        f'资料充分性：足以回答{sum(e["sufficiency"]=="S" for e in ev)}，部分可答{sum(e["sufficiency"]=="P" for e in ev)}，不足{sum(e["sufficiency"]=="I" for e in ev)}，无法判断{sum(e["sufficiency"]=="U" for e in ev)}（分母101）。',
        f'配对结果：v2改善{count["v2_better"]}，变差{count["v2_worse"]}，基本等价{count["equivalent"]}，各有得失{count["tradeoff"]}，无法判断{count["uncertain"]}（总计101）。','',
        '| 指标 | v1 | v2 | 分母与说明 |','|---|---:|---:|---|']
    for label,metric in [('错误拒答','refusal_erroneous'),('恰当拒答','refusal_appropriate'),('有未支持陈述的回答','answers_with_claim_status_N'),('有矛盾陈述的回答','answers_with_claim_status_C'),('有无法判断陈述的回答','answers_with_claim_status_U'),('引用关联完整支持','citation_answer_supported'),('引用部分支持','citation_answer_partial'),('引用不支持或矛盾','citation_answer_unsupported'),('回答缺少应有引用','citation_answer_missing')]:
        lines.append(f'| {label} | {n("v1",metric)} | {n("v2",metric)} | 各101份回答；不同问题类型可重叠 |')
    lines += ['',f'错误拒答配对变化：新增{err_up}题，消除{err_down}题；有无变化由同题判断对应，不能从两组边际计数推断个案。','',
        '| 要点状态 | v1 | v2 |','|---|---:|---:|']
    for status,label in [('C','覆盖'),('P','部分覆盖'),('O','遗漏'),('U','无法判断')]: lines.append(f'| {label} | {n("v1","point_coverage_"+status)} | {n("v2","point_coverage_"+status)} |')
    for stage in ('v1','v2'):
        row=next(r for r in summary if r['stage']==stage and r['metric']=='weighted_point_coverage')
        lines.append(f'{stage}加权要点覆盖：{row["numerator"]}/{row["denominator"]}，未知/不适用{row["unknown_or_na"]}；部分覆盖权重0.5，仅为透明描述性约定。')
    lines += ['', '引用支持按实际标记的分母及所有状态另列summary.csv。无引用回答不自动算“引用全部支持”；引用合法与引用支持分开。恰当拒答也不表示问题已完成。未支持只相对本轮资料，不等于科学事实错误。','',
        '## F1与内容的关系','', '| F1变化 | v2改善 | v2变差 | 基本等价 | 各有得失 | 无法判断 |','|---|---:|---:|---:|---:|---:|']
    for d in ('increase','decrease','unchanged'):
        lines.append('| '+d+' | '+' | '.join(str(next(r['questions'] for r in cross if r['f1_direction']==d and r['v2_content_result']==v)) for v in ('v2_better','v2_worse','equivalent','tradeoff','uncertain'))+' |')
    lines += ['',f'F1下降共{len(dec)}题，其中v2净内容变差{sum(p["v2_content_result"]=="v2_worse" for p in dec)}题；各有得失{sum(p["v2_content_result"]=="tradeoff" for p in dec)}题另列，不能全部算净损失。基本等价且差异原因仅措辞/额外内容的{len(expression)}题，可描述为主要表达变化。',
        f'F1升高共{len(inc)}题，其中内容基本等价{sum(p["v2_content_result"]=="equivalent" for p in inc)}题、v2更差{sum(p["v2_content_result"]=="v2_worse" for p in inc)}题、各有得失{sum(p["v2_content_result"]=="tradeoff" for p in inc)}题；不能把F1升高统一解释为内容改善。',
        '所有交叉表均来自本次AI判断，没有将未知记为零。F1差的阈值为1e-12。公开参考只作辅助；参考可核验性逐题记录于清单和配对表，原参考未改。','',
        '## 证据与争议案例','',
        'question_evidence_checklist.jsonl与answer_reviews.jsonl保留全101题/202份判断，每项可沿chunk_id和quote回查实际文本；paired_reviews.csv保留同题差异原因。disputed_cases.csv列出需独立核验的资料不足、参考问题、错误拒答、支持问题和指标不一致，不用选定案例估计总体错误率。','',
        '## 可支持的观察与尚不能成立的结论','',
        f'本次单一AI审查支持的描述：v2改善/变差/等价/混合/未知为{count["v2_better"]}/{count["v2_worse"]}/{count["equivalent"]}/{count["tradeoff"]}/{count["uncertain"]}；错误拒答新增/消除为{err_up}/{err_down}。收益应结合要点、未支持内容、引用和直接回答逐项观察，不能单用胜负总数替代理由。',
        '仍需专业人员或独立来源验证：必要要点定义、问题语义和资料适用范围、拒答是否合理、科学事实正确性、引用是否充分蕴含每条陈述、参考答案是否妥当。片段可能截断、来源异质或使用宽泛问题，不能以本次资料缺少证明事实为假。','',
        '本轮不计算置信区间；即便做同题来源组bootstrap，也不能包含单一AI评价自身的判断误差。没有独立评审重复性或人类一致性估计。此前身份/F1暴露及主观要点设定可能影响结论。','',
        '是否设计新训练：值得把本轮高置信的错误拒答、遗漏和引用问题列为待验证假设，并先由专业人员盲核；本轮不足以直接证明应增加训练或替换正式模型。如果这些趋势独立复现，可设计等训练量、等采样权重且只改变条件引用目标的消融，并另用未参与选择的数据验证。正式最终模型、正式成绩和人工评分均不变。','',
        '所有题目均在总体分母101中；某些要点/陈述不能判断或不适用时明确列出，不当零。准确执行信息、原始批次注释、命令、签名和进度见同目录文件。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['init','evidence','answers','commit-evidence','commit-review','status','finalize'])
    p.add_argument('--out',type=Path); p.add_argument('--start',type=int,default=0); p.add_argument('--count',type=int,default=5); p.add_argument('--file',type=Path)
    args=p.parse_args()
    if args.action=='init': init(); return
    out=args.out.resolve(); check(out); command(out)
    if args.action in ('evidence','answers'): view(out,args.action,args.start,args.count)
    elif args.action=='commit-evidence': commit_evidence(out,args.file.resolve())
    elif args.action=='commit-review': commit_review(out,args.file.resolve())
    elif args.action=='finalize': finalize(out)
    else: print(json.dumps({'evidence':len(rows(out/'_evidence_locked.jsonl')),'answers':len(rows(out/'answer_reviews.jsonl')),'pairs':len(rows(out/'_paired_anonymous.jsonl'))}))

if __name__=='__main__': main()
