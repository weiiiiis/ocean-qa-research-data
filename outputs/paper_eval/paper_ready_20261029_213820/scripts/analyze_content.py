"""Recompute descriptive tables from saved AI labels only; no new evaluation.
Run: python analyze_content.py [project_root] [output_root]
"""
from pathlib import Path
import sys, json, hashlib, shutil
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv)>1 else Path(r'D:\weitiao\ocean-chat')
OUT = Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/paper_eval/ai_content_audit_20261028_175905'
DEV=ROOT/'outputs/paper_eval/dev_supplement_20261028_164300'
for folder in ['tables','audit','scripts']: (OUT/folder).mkdir(parents=True,exist_ok=True)
def jl(p): return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def write_csv(df,p): df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')
def md(df):
    vals=df.fillna('NA').astype(str)
    return '| '+' | '.join(vals.columns)+' |\n| '+' | '.join(['---']*len(vals.columns))+' |\n'+'\n'.join('| '+' | '.join(x.replace('|','\\|').replace('\n','<br>') for x in row)+' |' for row in vals.values.tolist())
paired=pd.read_csv(SRC/'paired_reviews.csv')
reviews=jl(SRC/'answer_reviews_unblinded.jsonl')
checklists=jl(SRC/'question_evidence_checklist.jsonl')
raw=jl(SRC/'selected_raw_records.jsonl')
packets=jl(SRC/'anonymous_review_packet.jsonl')
metrics=jl(DEV/'per_question_metrics.jsonl')
selected_questions=jl(DEV/'selected_questions.jsonl')
ri={(r['id'],r['stage']):r for r in reviews}
oi={(r['id'],r['stage']):r for r in raw}
mi={(r['id'],r['stage']):r for r in metrics if r['condition'] in ['v1_rag_t160','v2_rag_t160']}
ci={r['id']:r for r in checklists}
pi={r['id']:r for r in packets}
assert len(paired)==101 and paired.id.nunique()==101
assert len(reviews)==202 and len(ri)==202 and len(raw)==202
assert set(paired.id)==set(ci)==set(pi)
checks=[]
for _,p in paired.iterrows():
    for s in ['v1','v2']:
        a=ri[(p.id,s)]; o=oi[(p.id,s)]; m=mi[(p.id,s)]
        assert a['answer']==o['answer']==m['answer']
        assert abs(p[f'{s}_f1']-m['f1'])<1e-12
        assert p[f'{s}_refusal']==a['refusal']
        assert p[f'{s}_citation_summary']==a['citation_summary']
    assert oi[(p.id,'v1')]['prompt_sha256']==oi[(p.id,'v2')]['prompt_sha256']
    assert oi[(p.id,'v1')]['context_sha256']==oi[(p.id,'v2')]['context_sha256']
    contexts={c['id']:c['text'] for c in oi[(p.id,'v1')]['retrieved_context']}
    for point in ci[p.id]['necessary_points']:
        e=point['evidence']; checks.append({'id':p.id,'chunk_id':e['chunk_id'],'quote':e['quote'],'exact_substring':e['quote'] in contexts.get(e['chunk_id'],'')})

rows=[]
def add(stage,metric,num,den,unit='answer',na=0,note=''):
    rows.append(dict(stage=stage,metric=metric,numerator=num,denominator=den,unknown_or_na=na,value=num/den if den else np.nan,unit=unit,note=note,scope='development; v1/v2 RAG; 160-token cap; single AI; exploratory',source=str(SRC/'answer_reviews_unblinded.jsonl')))
for s in ['v1','v2']:
    ars=[r for r in reviews if r['stage']==s]
    for status in ['none','appropriate','erroneous','uncertain']:
        add(s,'refusal_'+status,sum(r['refusal']==status for r in ars),101)
    for status in ['S','C','N','U']:
        add(s,'answers_with_claim_status_'+status,sum(any(c['status']==status for c in r['claims']) for r in ars),101,note='Categories can overlap; N=not supported by supplied excerpts, not scientifically false; C includes false insufficiency metaclaims.')
    for status in ['supported','partial','unsupported','missing','uncertain','not_applicable']:
        add(s,'citation_answer_'+status,sum(r['citation_summary']==status for r in ars),101,note='Answer-level association judgment; not marker validity or total fact coverage.')
    pts=Counter(p['status'] for r in ars for p in r['point_coverage'])
    assert sum(pts.values())==141
    for status in ['C','P','O','U']:
        add(s,'point_coverage_'+status,pts[status],141,'AI-defined necessary point',note='C covered; P partial; O omitted; U unknown; original labels preserved.')
    add(s,'weighted_point_coverage',pts['C']+.5*pts['P'],pts['C']+pts['P']+pts['O'],'weighted point',na=4,note='Original aggregation: all points equal weight; P=0.5; 4 questions have no assessable point, NA not zero.')
    pc=paired[f'{s}_weighted_point_coverage'].dropna()
    add(s,'question_equal_mean_coverage',float(pc.sum()),len(pc),'sum of per-question coverage / assessable questions',na=101-len(pc),note='Sensitivity: equal question weights, each coverage=(C+0.5P)/(C+P+O); numerator fractional, not a success count.')
    add(s,'answers_with_omission',sum(r[f'{s}_points_omitted']>0 for _,r in paired.iterrows()),97,na=4)
    cts=Counter(c['status'] for r in ars for c in r['citation_reviews'])
    for status in ['S','C','N','U']:
        add(s,'citation_marker_'+status,cts[status],sum(cts.values()),'reviewed citation marker',note='Support judged against actual cited chunk and associated statement; no citation is not counted as full support.')
    for status in ['yes','partial','no','uncertain']:
        add(s,'direct_answer_'+status,sum(r['direct_answer']==status for r in ars),101)
    add(s,'answers_no_assessable_points',sum(r[f'{s}_assessable_points']==0 for _,r in paired.iterrows()),101,na=4)
for status in ['v2_better','v2_worse','equivalent','tradeoff','uncertain']:
    add('paired',status,sum(paired.v2_content_result==status),101,'question',note='Holistic AI verdict; worse can include refusal, omissions, citation defects, and content, not a count of core factual errors.')
for status in ['S','P','I','U']:
    add('evidence',status,sum(paired.sufficiency==status),101,'question',note='S sufficient; P partially answerable; I insufficient; U uncertain.')
t5=pd.DataFrame(rows)
write_csv(t5,OUT/'tables/table5_ai_content.csv')
# Compare all pre-existing metrics to label-based recomputation.
orig=pd.read_csv(SRC/'summary.csv')
diff=[]
for _,r in orig.iterrows():
    n=t5[(t5.stage==r.stage)&(t5.metric==r.metric)]
    if len(n)!=1: diff.append({'stage':r.stage,'metric':r.metric,'reason':'not uniquely recomputed'}); continue
    n=n.iloc[0]
    for f in ['numerator','denominator','value']:
        if abs(float(r[f])-float(n[f]))>1e-10: diff.append({'stage':r.stage,'metric':r.metric,'field':f,'original':r[f],'recomputed':n[f]})
write_csv(pd.DataFrame(diff,columns=['stage','metric','field','original','recomputed','reason']),OUT/'audit/content_summary_differences.csv')

translations={'refusal_erroneous':'错误拒答 / Erroneous refusal','refusal_appropriate':'恰当拒答 / Appropriate refusal','answers_with_claim_status_N':'含材料未支持陈述 / Any unsupported claim','answers_with_claim_status_C':'含与材料矛盾陈述 / Any contradicted claim','answers_with_claim_status_U':'含无法判断陈述 / Any uncertain claim','citation_answer_supported':'引用关联完整支持 / Fully supported citation associations','citation_answer_partial':'引用关联部分支持 / Partially supported associations','citation_answer_unsupported':'引用不支持或矛盾 / Unsupported/contradicted associations','citation_answer_not_applicable':'引用不适用（无标记）/ No citation, not applicable','weighted_point_coverage':'原报告要点加权覆盖 / Point-weighted coverage','question_equal_mean_coverage':'每题等权覆盖均值 / Equal-question coverage mean','answers_with_omission':'有遗漏的可评价回答 / Answers with omission','citation_marker_S':'支持的标记 / Supported citation markers','direct_answer_yes':'直接回答 / Direct answer'}
display=[]
for key,label in translations.items():
    dr={'指标 / Metric':label}
    for s in ['v1','v2']:
        r=t5[(t5.stage==s)&(t5.metric==key)].iloc[0]
        dr[s]=f"{r.numerator:.6g}/{r.denominator:g} ({100*r.value:.2f}%)"
    display.append(dr)
table5note='''表5 / Table 5. 开发集单 AI 内容审查（仅 v1/v2 RAG）。101 题、202 回答，91 来源组；160 token 上限。各回答指标分母 101；97 题有可评要点，共 141 点，另 4 题覆盖率 NA。覆盖计分 C=1、P=0.5、O=0；必要要点由 AI 主观定义，可能包含辅助细节。原报告将每个要点等权，敏感性分析则先逐题计算覆盖率再按 97 题等权，分子为比例之和。标记支持以 v1 的 116 个与 v2 的 113 个已评标记为分母，与回答层支持及可解析标记为不同指标。无标记不算全部支持；材料未支持不等于科学事实错误。矛盾包含“材料没有答案”之类元陈述。\n\nThis is descriptive, exploratory evidence from one conversational AI reviewer, whose exact model/version was unrecorded and who had prior exposure to identities and some results. Anonymous display did not make it fully independent or blind. No new labels, confidence intervals, expert scores, or significance tests were created. Point-weighted and question-weighted rates use distinct denominators and weights; the fractional numerators are coverage-score sums, not counts of correct answers. Citation support is relative to supplied excerpts, not independent-source verification.\n\n来源 / Sources: `ai_content_audit_20261028_175905/answer_reviews_unblinded.jsonl`, `question_evidence_checklist.jsonl`, `paired_reviews.csv`; exact source paths are in CSV. Original labels are preserved in `audit/ai_labels/`. Formal-test 40-question AI evaluation is separate (Table 2 supplement).\n'''
verdict=paired.v2_content_result.value_counts()
verdictdf=pd.DataFrame([{'v2 内容判断 / Content verdict':s,'题数 / Questions':f'{verdict.get(s,0)}/101 ({100*verdict.get(s,0)/101:.2f}%)'} for s in ['v2_better','v2_worse','equivalent','tradeoff','uncertain']])
(OUT/'tables/table5_ai_content.md').write_text('# 表5：AI 内容审查 / Table 5: AI content audit\n\n'+md(pd.DataFrame(display))+'\n\n'+md(verdictdf)+'\n\n'+table5note,encoding='utf-8')

cats=['v2_better','v2_worse','equivalent','tradeoff','uncertain']
cross=[]
for direction in ['increase','decrease','unchanged']:
    sub=paired[paired.f1_direction==direction]; row={'f1_direction':direction}
    row.update({c:int(sum(sub.v2_content_result==c)) for c in cats})
    row.update(row_total=len(sub),total_questions=101,unit='question',source=str(SRC/'paired_reviews.csv'))
    cross.append(row)
t6=pd.DataFrame(cross); write_csv(t6,OUT/'tables/table6_f1_content.csv')
origcross=pd.read_csv(SRC/'f1_content_crosstab.csv');
for r in cross:
    for c in cats:
        o=origcross[(origcross.f1_direction==r['f1_direction'])&(origcross.v2_content_result==c)].iloc[0]
        assert int(o['questions'])==r[c]
disp=[]
for r in cross:
    q={'F1变化 / Direction':r['f1_direction']}
    q.update({c:f"{r[c]}/{r['row_total']} ({100*r[c]/r['row_total']:.1f}%)" for c in cats})
    q['该行占全体 / All questions']=f"{r['row_total']}/101"
    disp.append(q)
(OUT/'tables/table6_f1_content.md').write_text('# 表6：F1变化与 AI 内容判断 / Table 6: F1 changes and AI verdicts\n\n'+md(pd.DataFrame(disp))+'''\n\n分母 / Denominators: cells use the corresponding F1-direction row (24 increased, 35 decreased, 42 unchanged); last column uses all 101 questions. All original five content labels are retained. Increase/decrease is v2 minus v1 citation-stripped original-reference token F1, tolerance 1e-12. F1 is lexical overlap, not factual accuracy. No new AI reassessment was performed. Mixed and uncertain outcomes are not pooled with gains or losses. These are the same 101 development questions under two RAG stages, not independent test observations; 91 source groups. No confidence intervals were computed under the audit protocol. A v2-worse verdict can reflect omissions, refusal, citation support or content, rather than a new core factual error.\n\n来源 / Sources: AI `paired_reviews.csv` linked to original `dev_supplement_20261028_164300/per_question_metrics.jsonl`; exact F1 values independently matched. Single AI, model/version unrecorded, prior result exposure; exploratory.\n''',encoding='utf-8')

trans=pd.crosstab(paired.v1_refusal,paired.v2_refusal).reindex(index=['none','appropriate','erroneous','uncertain'],columns=['none','appropriate','erroneous','uncertain'],fill_value=0)
translong=[]
for a in trans.index:
    for b in trans.columns:
        translong.append(dict(v1_refusal=a,v2_refusal=b,questions=int(trans.loc[a,b]),denominator=101))
write_csv(pd.DataFrame(translong),OUT/'tables/ai_refusal_paired_transitions.csv')
covcols=['id','source_group','sufficiency','v2_content_result','f1_delta','v1_weighted_point_coverage','v2_weighted_point_coverage','v1_refusal','v2_refusal','v1_assessable_points','v2_assessable_points']
cov=paired[covcols].copy(); cov['coverage_delta']=cov.v2_weighted_point_coverage-cov.v1_weighted_point_coverage
cov['coverage_change']=np.select([cov.coverage_delta.isna(),cov.coverage_delta>1e-12,cov.coverage_delta< -1e-12],['NA_no_assessable_points','increased','decreased'],default='unchanged')
write_csv(cov,OUT/'tables/ai_coverage_paired_changes.csv')
newref=paired[(paired.v1_refusal!='erroneous')&(paired.v2_refusal=='erroneous')]
assert len(newref)==5

case_info={
'A1234':('真实内容损失；新增错误拒答 / Lost content; new erroneous refusal','片段逐项列出四种方法；v1全部覆盖、v2遗漏四项并称无信息。该例直接支持局部内容损失，不能估计总体错误率。','高可定位性；仅材料范围，未独立验证来源科学正确性。'),
'A1726':('词面变化 / Lexical change','三类测定技术均被保留；instrument/procedure措辞及句式使F1下降。支持词面降分不一定对应内容损失。','沿用原AI equivalent标签；不将专业术语差异重新评分。'),
'A718':('指标上升但未改善 / Higher F1 without improvement','两者均错误拒答并遗漏同一要点，v2仍获更高F1。支持词面增分不能自动解释为内容改善。','原AI判equivalent，额外引用缺陷逐标记记录；综合判定主观。'),
'A331':('新增错误拒答 / New erroneous refusal','相关片段出现Blue Amazon名称；v1给出，v2错误宣称材料未提供。','高可定位性；仅依据实际提供片段。'),
'A1825':('内容损失且原答有引用缺陷 / Lost answer with pre-existing citation defect','C2明确公路运输占比最大。v1答对但错引片段；v2错误拒答。内容优势不能掩盖v1错引。','不把v1表述为引用完全可靠。'),
'A862':('新增错误拒答 / New erroneous refusal','片段描述SS-20 taut-leg系泊研究/设计；v1识别该配置，v2拒答。','片段为设计分析应用，不能扩写为真实物理安装的独立证明。'),
'A1109':('有争议的新增拒答 / Disputed new refusal','v1保留成本不确定性和历史单井例子；v2否认可见成本资料。AI新增错误拒答标签保留。','部分可答；总探索/开发成本与历史单井/勘探成本存在范围歧义，应独立专业核验，不与四个充分资料案例同等确定。'),
'A1428':('局部引用改善 / Local citation improvement','v2引用直接联系浅化旋回与孔隙度的片段；v1引用另一地层旋回描述。','支持的是局部引用关联改善，不是整体事实质量提高。'),
'A575':('综合胜负与核心事实变化有别 / Holistic verdict versus factual change','两者同样错误拒答，综合差异涉及额外矛盾引用，不能视为核心事实回答发生变化。','严格保留原AI比较与标记支持标签；需独立审核。'),
'A568':('要点遗漏 / Omitted point','v2保留老化管道/持续生产条件，但遗漏腐蚀/泄漏，显示F1变化需结合必要要点解释。','必要要点由单AI设定；是否为最低必要细节需专家确认。'),
'A1422':('实质解释差异待核验 / Substantive interpretation pending verification','高压变质年龄与地壳形成时代的替换涉及地质语义，原审查判损失。','应由专业地质人员复核，不作为独立科学事实裁决。')}
cases=[]
for short,(typ,explain,limits) in case_info.items():
    qid='PIRA-DEV-'+short; p=paired[paired.id==qid].iloc[0]; o1=oi[(qid,'v1')]; o2=oi[(qid,'v2')]
    pts=ci[qid]['necessary_points']
    ev=[dict(point=k['text'],**k['evidence']) for k in pts]
    contexts=o1['retrieved_context']
    # Retain all actual evidence windows, and separately highlight locked point quotes.
    cases.append(dict(id=qid,question=o1['question'],case_type=typ,source_group=p.source_group,evidence_sufficiency_original_label=p.sufficiency,necessary_points_and_quotes=json.dumps(ev,ensure_ascii=False),all_actual_evidence=json.dumps(contexts,ensure_ascii=False),evidence_source='; '.join(c['source']+' / '+c['id'] for c in contexts),v1_answer=o1['answer'],v2_answer=o2['answer'],v1_f1=p.v1_f1,v2_f1=p.v2_f1,f1_delta_pp=100*p.f1_delta,v1_output_tokens=o1['metrics']['output_tokens_including_eos'],v2_output_tokens=o2['metrics']['output_tokens_including_eos'],ai_verdict_original_label=p.v2_content_result,ai_comparison_reason=p.comparison_reason,ai_difference_causes_original_labels=p.difference_causes,v1_refusal_original_label=p.v1_refusal,v2_refusal_original_label=p.v2_refusal,v1_coverage=p.v1_weighted_point_coverage,v2_coverage=p.v2_weighted_point_coverage,v1_citation_support_original_label=p.v1_citation_summary,v2_citation_support_original_label=p.v2_citation_summary,reference_status_original_label=p.reference_status,reference_reason=p.reference_reason,specific_supported_interpretation=explain,uncertainty_or_dispute=limits,scope='same development questions; v1/v2 RAG t160; single existing AI',raw_source=str(SRC/'selected_raw_records.jsonl'),label_source=str(SRC/'answer_reviews_unblinded.jsonl')))
cdf=pd.DataFrame(cases); write_csv(cdf,OUT/'cases.csv')
for r in cases:
    r['v1_full_original_ai_review']=json.dumps(ri[(r['id'],'v1')],ensure_ascii=False)
    r['v2_full_original_ai_review']=json.dumps(ri[(r['id'],'v2')],ensure_ascii=False)
write_csv(pd.DataFrame(cases),OUT/'cases.csv')
case_display=[]
for r in cases:
    quotes=json.loads(r['necessary_points_and_quotes'])
    evidence='<br>'.join(e['chunk_id']+': '+e['quote'] for e in quotes)
    case_display.append({'ID / 类型':r['id']+'<br>'+r['case_type'],'问题 / Question':r['question'],'证据定位与摘录 / Evidence':evidence,'v1 / v2 回答':r['v1_answer']+'<br>→ '+r['v2_answer'],'F1与AI标签':f"{100*r['v1_f1']:.2f}% → {100*r['v2_f1']:.2f}% ({r['f1_delta_pp']:+.2f} pp); {r['ai_verdict_original_label']}; refusal {r['v1_refusal_original_label']}→{r['v2_refusal_original_label']}",'解释和限制':r['specific_supported_interpretation']+' '+r['uncertainty_or_dispute']})
(OUT/'cases.md').write_text('# 核实案例 / Verified case records\n\n'+md(pd.DataFrame(case_display))+'''\n\n案例为有目的选择，不能估计总体发生率。全部回答、完整实际检索窗口、必要要点的逐字证据与来源、原始AI标签见cases.csv。表中F1为去引用后的原参考词面重合，差值单位为百分点；不表示事实正确率。本轮只核对原数据与标签一致性，没有启动新内容评价。All cases are from the same 101-question development set under v1/v2 RAG with a 160-token cap. Original single-AI judgments are retained, not promoted to expert adjudication or independently verified scientific truth.\n''',encoding='utf-8')

labels=OUT/'audit/ai_labels';labels.mkdir(exist_ok=True)
for name in ['answer_reviews_unblinded.jsonl','question_evidence_checklist.jsonl','paired_reviews.csv','review_protocol.md','content_judgments_sealed.json','audit_manifest.json']:
    shutil.copy2(SRC/name,labels/name)
write_csv(pd.DataFrame(checks),OUT/'audit/content_evidence_quote_checks.csv')
assert all(c['exact_substring'] for c in checks)
hash_before=json.loads((SRC/'source_hashes.json').read_text(encoding='utf-8-sig'))
integrity=[]
for name,h in hash_before.items():
    p=DEV/name
    actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    integrity.append(dict(path=str(p),expected_sha256=h,actual_sha256=actual,match=actual==h))
write_csv(pd.DataFrame(integrity),OUT/'audit/content_source_hash_checks.csv')
summary={
 'scope':{'questions':101,'answers':202,'source_groups':paired.source_group.nunique(),'unique_expected_gold_supporting_text_ids':len({r['expected_source_id'] for r in selected_questions}),'unique_retrieved_source_sections_in_all_contexts':len({c['section'] for r in raw for c in r['retrieved_context']})},
 'verdicts':{c:int(sum(paired.v2_content_result==c)) for c in cats},
 'f1_content_crosstab':cross,
 'coverage':{s:{'point_weighted_numerator':float(t5[(t5.stage==s)&(t5.metric=='weighted_point_coverage')].iloc[0].numerator),'points':141,'point_weighted_mean':float(t5[(t5.stage==s)&(t5.metric=='weighted_point_coverage')].iloc[0].value),'question_equal_sum':float(paired[f'{s}_weighted_point_coverage'].sum()),'assessable_questions':97,'question_equal_mean':float(paired[f'{s}_weighted_point_coverage'].mean())} for s in ['v1','v2']},
 'coverage_changes':{k:int(v) for k,v in cov.coverage_change.value_counts().items()},
 'refusals':{'v1_erroneous':int(sum(paired.v1_refusal=='erroneous')),'v2_erroneous':int(sum(paired.v2_refusal=='erroneous')),'new_erroneous':len(newref),'resolved_erroneous':int(sum((paired.v1_refusal=='erroneous')&(paired.v2_refusal!='erroneous'))),'new_ids':newref.id.tolist()},
 'checks':{'answer_and_f1_linkage':'all 202 exact answers and F1 matched original records/metrics','same_prompt_and_context':'101 pairs matched','necessary_point_quotes_checked':len(checks),'quote_mismatches':sum(not c['exact_substring'] for c in checks),'original_summary_differences':len(diff),'source_hash_files':len(integrity),'source_hash_mismatches':sum(not r['match'] for r in integrity)},
 'reviewer_limitations':['Single conversational AI','Exact model/version unrecorded','Prior exposure to identities/partial F1 results; not fully independent blind review','AI-defined necessary points subjective','Excerpt support does not independently verify scientific truth','No new content labels or intervals generated']}
(OUT/'audit/content_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
audit=f'''# AI内容数据复核\n\n只读取并重汇总既有标签。101题/202回答，各为v1_rag_t160和v2_rag_t160；91来源组。全部202条回答逐字与原始记录和指标文件一致；101对提示及上下文哈希一致；141个必要要点摘录均可逐字定位。原summary.csv重算差异{len(diff)}项；原F1交叉表逐格一致。源数据清单{len(integrity)}个文件哈希不一致{sum(not r['match'] for r in integrity)}个。\n\n要点加权v1=104/141={104/141:.8f}，v2=93/141={93/141:.8f}。每题等权敏感性分析v1={paired.v1_weighted_point_coverage.sum():.12g}/97={paired.v1_weighted_point_coverage.mean():.8f}，v2={paired.v2_weighted_point_coverage.sum():.12g}/97={paired.v2_weighted_point_coverage.mean():.8f}。分子为逐题覆盖比例之和，非二元答对数；4题无要点保留NA。两种权重均方向下降，但其大小依赖AI要点定义。\n\n错误拒答3/101→8/101，新增5、消除0。新增A1234、A331、A1825、A862资料充分；A1109只有部分材料且成本范围有歧义，保留原标签并标争议。全体配对更好5、更差17、等价76、混合1、未知2；更差不可等同17条核心事实错误。F1下降35题中11更差、22等价、2更好；上升24题中3更好、5更差、14等价、1混合、1未知。\n\n引用标记支持97/116→95/113；回答层完整关联支持83/101→81/101。两种分母不同，不应宣称标记支持比例微升意味着回答证据使用改善。单一AI版本未记录、此前接触结果、要点主观、跨日续跑而精确快照不可核实。按原审查协议不计算区间；本轮不重审标签。该审查不覆盖Base或无RAG，亦不替代正式40题AI评价。\n\n交付：表5/6、配对拒答转移、逐题覆盖变化、11例原答和完整上下文、原始标签副本及脚本。所有新增文件仅写入本次paper_ready目录。\n'''
(OUT/'audit/content_audit.md').write_text(audit,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
