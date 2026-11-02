"""Read-only replay of development metrics and recorded paired cluster bootstrap.

Usage: python analyze_development.py [--root PROJECT] [--out DELIVERY]
Does not import model/inference libraries and writes only under --out.
"""
from pathlib import Path
import argparse, collections, hashlib, json, re, string
import numpy as np
import pandas as pd

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--root', type=Path, default=Path(r'D:\weitiao\ocean-chat'))
PARSER.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
ARGS = PARSER.parse_args()
ROOT, OUT = ARGS.root.resolve(), ARGS.out.resolve()
SRC = ROOT/'outputs/paper_eval/dev_supplement_20261028_164300'
for sub in ('tables','audit'): (OUT/sub).mkdir(parents=True, exist_ok=True)
def jl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def js(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(df,name): df.to_csv(OUT/'tables'/name,encoding='utf-8-sig',index=False)
CITE = re.compile(r'\[([A-Za-z0-9][A-Za-z0-9_.:/-]*(?:\s*[,;]\s*[A-Za-z0-9][A-Za-z0-9_.:/-]*)*)\]')
def norm(s): return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def score(a,b):
    a,b=norm(a),norm(b); at,bt=a.split(),b.split()
    if not at or not bt: return float(a==b),float(at==bt)
    common=sum((collections.Counter(at)&collections.Counter(bt)).values())
    return float(a==b),2*common/(len(at)+len(bt))
questions=jl(SRC/'selected_questions.jsonl'); Q={q['id']:q for q in questions}
raw=jl(SRC/'answers.jsonl'); original=jl(SRC/'per_question_metrics.jsonl')
INDEX={(r['condition'],r['id']):r for r in original}
assert len(raw)==len(INDEX)==1010 and len(Q)==101
assert len({q['source_group'] for q in questions})==91
assert all(q['split']=='dev' for q in questions)
rows=[]
for r in raw:
    q=Q[r['id']]; text=r['answer']; stripped=CITE.sub('',text).strip()
    ref=q.get('reference_answer_origin',q.get('reference_answer',''))
    em,f1=score(stripped,ref); _,rf1=score(text,ref)
    markers=[p.strip() for m in CITE.finditer(text) for p in re.split('[,;]',m.group(1))]
    available={c['id'] for c in r['retrieved_context']}
    unresolved=[m for m in markers if m not in available]
    expected=str(q.get('expected_source_id') or q.get('source_article_eid'))
    sourcehit=any(str(c.get('section',c.get('article_eid',c.get('source_article_eid'))))==expected for c in r['retrieved_context']) if r['rag'] else None
    record={'id':r['id'],'condition':r['condition'],'source_group':q['source_group'],'stage':r['stage'],'rag':r['rag'],'budget':r['budget'],
        'f1':f1,'raw_f1':rf1,'exact_match':em,'output_tokens':r['metrics']['output_tokens_including_eos'],
        'word_count':len(text.split()),'stripped_word_count':len(stripped.split()),'hit_token_limit':int(r['metrics']['hit_token_limit']),
        'citation_markers':len(markers),'resolved_markers':sum(m in available for m in markers),'has_citation':int(bool(markers)),
        'has_unresolved_citation':int(bool(unresolved)),'unclosed_bracket_count':max(0,text.count('[')-text.count(']')),
        'gold_source_retrieved':sourcehit}
    previous=INDEX[(r['condition'],r['id'])]
    for k,v in record.items():
        pv=previous[k]
        if isinstance(v,float): assert abs(pv-v)<1e-12,(r['id'],k,pv,v)
        else: assert pv==v,(r['id'],k,pv,v)
    assert r['source_group']==q['source_group'] and r['question']==q['question']
    assert r['finish_reason']==('length' if r['metrics']['hit_token_limit'] else 'eos')
    rows.append(record)
D=pd.DataFrame(rows); R=D.set_index(['condition','id'])
stats=js(SRC/'statistics.json'); ids=stats['question_order']; groups=[Q[i]['source_group'] for i in ids]
gkeys=stats['group_order']; draws=np.load(SRC/'bootstrap_source_group_draws.npy')
assert gkeys==sorted(set(groups)) and len(gkeys)==91 and draws.shape==(10000,91)
assert sha(SRC/'bootstrap_source_group_draws.npy')==stats['draws_file_sha256']
assert np.array_equal(draws,np.random.default_rng(20261028).integers(0,91,(10000,91)))
def interval(values,groups,draws):
    keys=sorted(set(groups)); values=np.asarray(values)
    sums=np.array([values[np.array(groups)==g].sum() for g in keys])
    sizes=np.array([groups.count(g) for g in keys])
    boots=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    return np.mean(values),*np.percentile(boots,[2.5,97.5])
S=pd.read_csv(SRC/'summary.csv'); summaries=[]
for _,old in S.iterrows():
    d=R.loc[old.condition].loc[ids]; assert len(d)==101 and d.index.is_unique
    value,lo,hi=interval(d.f1.to_numpy(),groups,draws)
    assert max(abs(value-old.f1),abs(lo-old.f1_ci95_low),abs(hi-old.f1_ci95_high))<1e-12
    for oldk,newk in [('mean_output_tokens_including_eos','output_tokens'),('mean_words','word_count'),('mean_stripped_words','stripped_word_count'),('raw_f1','raw_f1')]:
        assert abs(d[newk].mean()-old[oldk])<1e-12
    for oldk,newk in [('truncated_answers','hit_token_limit'),('answers_with_citations','has_citation'),('answers_with_unresolved_citations','has_unresolved_citation'),('citation_markers','citation_markers'),('resolved_markers','resolved_markers')]:
        assert int(d[newk].sum())==int(old[oldk])
    assert int((d.unclosed_bracket_count>0).sum())==int(old.answers_with_unclosed_brackets)
    z=old.to_dict(); z.update({'f1_sum':d.f1.sum(),'f1_denominator_questions':101,'mean_output_tokens_sum':int(d.output_tokens.sum()),
       'citation_answer_denominator':101,'truncation_denominator':101,'unresolved_answer_denominator':101,'unclosed_answer_denominator':101,
       'split':'development','analysis_status':'exploratory; pointwise unadjusted 95% intervals','source_file':str(SRC/'answers.jsonl'),
       'source_metrics_file':str(SRC/'per_question_metrics.jsonl'),'statistics_source':str(SRC/'statistics.json')})
    summaries.append(z)
SUM=pd.DataFrame(summaries)
# Reproduce every recorded comparison using exactly the stored common cluster draws.
P=pd.read_csv(SRC/'paired_differences.csv'); maxerr=0.0
for _,p in P.iterrows():
    vals=sum(coef*R.loc[condition].loc[ids,p.metric].to_numpy() for coef,condition in json.loads(p.terms))
    new=interval(vals,groups,draws)
    maxerr=max(maxerr,max(abs(a-b) for a,b in zip(new,(p.difference,p.ci95_low,p.ci95_high))))
assert maxerr<1e-12,maxerr
merged=js(SRC/'source_group_sensitivity.json'); mg=[merged['question_group_map'][i] for i in ids]
mdraws=np.load(SRC/'bootstrap_merged_source_draws.npy'); PM=pd.read_csv(SRC/'paired_differences_source_merged.csv')
assert len(set(mg))==90 and sha(SRC/'bootstrap_merged_source_draws.npy')==merged['draws_sha256']
maxerr_merged=0.0
for n,p in PM.iterrows():
    termrow=P[(P.comparison==p.comparison)&(P.metric==p.metric)].iloc[0]
    vals=sum(coef*R.loc[condition].loc[ids,p.metric].to_numpy() for coef,condition in json.loads(termrow.terms))
    new=interval(vals,mg,mdraws)
    maxerr_merged=max(maxerr_merged,max(abs(a-b) for a,b in zip(new,(p.difference,p.ci95_low,p.ci95_high))))
assert maxerr_merged<1e-12
# Check each actual prompt/context is common to every stage and budget in its RAG state.
for qid in ids:
    for rag in (False,True):
        rr=[r for r in raw if r['id']==qid and r['rag']==rag]
        assert len({r['prompt'] for r in rr})==1 and len({json.dumps(r['retrieved_context'],sort_keys=True) for r in rr})==1
        assert len({r['prompt_sha256'] for r in rr})==1 and len({r['context_sha256'] for r in rr})==1
csv(D,'development_recomputed_per_question.csv')
T3=SUM[SUM.max_new_tokens==160].copy().sort_values(['stage','rag'],key=lambda x:x.map({'base':0,'v1':1,'v2':2}) if x.name=='stage' else x)
T4=SUM[SUM.stage!='v1'].copy().sort_values(['max_new_tokens','stage','rag'],ascending=[False,True,True])
csv(T3,'table3_development.csv'); csv(T4,'table4_budget.csv')
P['difference_unit']=P.metric.map(lambda m:'fraction (multiply by 100 for percentage points)' if m in ['f1','hit_token_limit','has_citation','has_unresolved_citation'] else 'tokens/words/markers as metric')
P['source_file']=str(SRC/'paired_differences.csv'); P['analysis_status']='exploratory; pointwise, no multiplicity adjustment'
csv(P,'development_paired_contrasts.csv'); csv(PM,'development_merged_source_contrasts.csv')
notes='''
指标定义与样本范围：全部为已参与模型选择的英文开发集101题，91个公开EID来源组；90个不同支撑文本。每条件101条回答，不是独立测试。表3与表4共用四个160-token条件（404条物理记录），不能重复计数；总体10条件1010条输出。

去引用F1是按题等权的SQuAD词面重合均值：各题F1之和/101，不是答对题数/101，也不是事实正确率。仅使用原始公开参考；未把候选参考增为新题。去引用正则仅删除完整匹配的标记，未闭合编号仍留在评分文本。F1区间沿用并精确重放已存的配对来源组bootstrap：91组有放回抽样，保留组内题目，题加权均值，10,000次，seed=20261028，百分位区间；各项共用抽样矩阵。所有开发集区间均为探索性、逐项且未做多重比较调整。90组合并来源敏感性另见 development_merged_source_contrasts.csv。

实际输出token含EOS；词数按完整解码答案空白分隔，含引用；引用形式问题分别为：有完整标记的回答数/101、至少一个编号无法映射到当题提供片段的回答数/101、方括号未闭合回答数/101。标记可解析率为可映射的完整标记数/所有完整标记数；0/0记为不适用，不报告100%。无RAG时没有提供片段，因此其标记全部无法映射。编号有效不等于证据支持陈述；这两表没有AI或人工内容评分。截断为实际生成达到预算且未以EOS结束，不等于答案必然不完整。表4同上限也允许实际长度不同；96/160比较不能排除长度影响。

两阶段继承：Base → 英文grounded_v1/epoch_1 → 英文grounded_v2/epoch_1（正式FT）。v1配置训练了2 epoch并分别保存，但本处只用epoch_1；epoch_2不是v2祖先。v1以240例/48事实家族训练、LR=5e−5；v2对同240例使用评测对齐的用户提示结尾、LR=2e−5、closed_book三倍采样，共336次抽样、一epoch。这些变化与额外训练混杂，不能归因于单条引用规则。所有本轮推理使用同一evidence_v2提示、同题BM25 top3（上下文1000 token）和贪心生成，输入上限2048、重复惩罚1.05。
'''
def ppf(v): return f'{v*100:.2f}'
def ci(s): return f"{ppf(s.f1)} [{ppf(s.f1_ci95_low)}, {ppf(s.f1_ci95_high)}]"
def label(s): return {'base':'Base','v1':'v1','v2':'v2'}[s.stage]+('+RAG' if s.rag else '')
def md_table(df,title,budget=False):
    lines=[title,'', '| 条件 | 上限(token) | 去引用F1% [95% CI] | 原始F1% | 实际token均值 | 实际词数均值 | 有引用回答 | 可解析标记/全部标记 | 无法解析回答 | 方括号未闭合回答 | 截断 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in df.itertuples():
        lines.append(f'| {label(s)} | {int(s.max_new_tokens)} | {ci(s)} | {ppf(s.raw_f1)} | {s.mean_output_tokens_including_eos:.2f} | {s.mean_words:.2f} | {int(s.answers_with_citations)}/101 | {int(s.resolved_markers)}/{int(s.citation_markers)} | {int(s.answers_with_unresolved_citations)}/101 | {int(s.answers_with_unclosed_brackets)}/101 | {int(s.truncated_answers)}/101 |')
    lines+=['',notes,'','来源：'+str(SRC/'answers.jsonl')+'；逐题评分 '+str(SRC/'per_question_metrics.jsonl')+'；协议 '+str(SRC/'protocol.json')+'；统计 '+str(SRC/'statistics.json')+'。新脚本从原始回答独立重算指标并与原逐题和汇总逐字段核对。','']
    return '\n'.join(lines)
def contrast_md(comparisons):
    lines=['| 配对比较（后者相减） | 指标 | 差值 [95% CI] | 单位 | n/来源组 |','|---|---|---:|---|---:|']
    for comparison in comparisons:
        for metric in ['f1','output_tokens','has_citation','hit_token_limit']:
            r=P[(P.comparison==comparison)&(P.metric==metric)].iloc[0]
            scale=1 if metric=='output_tokens' else 100; unit='token' if scale==1 else '百分点'
            lines.append(f'| {comparison} | {metric} | {r.difference*scale:+.2f} [{r.ci95_low*scale:+.2f}, {r.ci95_high*scale:+.2f}] | {unit} | 101/91 |')
    return '\n'.join(lines)
t3md=md_table(T3,'# 表3｜开发集训练阶段比较（探索性）')
t3md+='\n## 配对差值\n\n'+contrast_md(['B_v1_minus_base_no_rag','B_v1_minus_base_rag','B_v2_minus_v1_no_rag','B_v2_minus_v1_rag','A_RAG_gain_base_160','B_RAG_gain_v1_160','A_RAG_gain_v2_160'])+'\n'
t4md=md_table(T4,'# 表4｜开发集生成长度预算敏感性（探索性）')
t4md+='\n## 配对差值\n\n'+contrast_md(['A_v2_minus_base_rag_160','A_v2_minus_base_rag_96','A_RAG_gap_change_96_minus160','A_budget96_minus160_base_no_rag','A_budget96_minus160_base_rag','A_budget96_minus160_v2_no_rag','A_budget96_minus160_v2_rag'])+'\n'
(OUT/'tables/table3_development.md').write_text(t3md,encoding='utf-8')
(OUT/'tables/table4_budget.md').write_text(t4md,encoding='utf-8')
audit={'source_directory':str(SRC),'questions':101,'source_groups':91,'unique_supporting_sources':len({q['expected_source_id'] for q in questions}),
 'raw_answers':len(raw),'conditions':10,'per_question_checks':{'matched_records':1010,'all_numeric_fields_match':True},
 'paired_statistics_rows_replayed':len(P),'max_primary_bootstrap_difference':maxerr,'merged_source_groups':len(set(mg)),
 'merged_comparison_rows_replayed':len(PM),'max_merged_bootstrap_difference':maxerr_merged,'prompts_and_contexts_equal_within_question_rag_state':True,
 'discrepancies':[],'summaries':summaries,'f1_contrasts':P[P.metric=='f1'].to_dict(orient='records'),
 'source_hashes':{str(SRC/n):sha(SRC/n) for n in ['answers.jsonl','per_question_metrics.jsonl','summary.csv','paired_differences.csv','paired_differences_source_merged.csv','protocol.json','statistics.json','selected_questions.jsonl','bootstrap_source_group_draws.npy','bootstrap_merged_source_draws.npy']}}
(OUT/'audit/dev_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
auditmd=f'''# 开发集复核

本次只读重算：101题、91 EID组、90支撑文本、10条件1010条原始输出；split全部dev。所有条件题目ID相同且无重复。按原始回答、原始参考和原引用正则重算F1、原始F1、EM、词数、实际token、截断及引用计数，与1010条已存逐题记录完全一致；与10行原汇总一致。\n
精确重放原存的10,000×91来源组抽样矩阵（seed=20261028），核对其SHA256与重生随机矩阵；全部{len(P)}行配对统计的均值与95%区间匹配，最大浮点差={maxerr:.3g}。同法核对90组合并来源敏感性的全部{len(PM)}行，最大浮点差={maxerr_merged:.3g}。同题同RAG状态的实际提示、完整检索上下文和顺序在阶段/预算之间完全一致。未重跑模型或新增评价。\n
未发现原始、逐题、summary或配对CSV之间的数值冲突。原报告对base+RAG预算下降F1写−0.01 pp，是−0.0133336 pp四舍五入；不是−1 pp。95%区间跨零均仅表示证据不足，不解释为零效应已被证明。

已明确表格分母、F1宏平均、可解析标记与支持的区别；表3/4对开发集和共享回答重复计数的边界见表注。

已核实训练继承，英文v1/epoch_1是英文v2/epoch_1的父适配器；v1配置中的epochs=2是原训练运行计划，不能据此将被评测v1/epoch_1写成训练2 epoch，也不能把epoch_2当祖先。训练提示结尾和采样权重与额外训练同时改变，第二阶段效应不识别单项规则的因果作用。

{notes}
'''
(OUT/'audit/dev_audit.md').write_text(auditmd,encoding='utf-8')
print(json.dumps({'records':len(raw),'summary_rows':len(SUM),'contrasts_replayed':len(P),'max_primary_error':maxerr,'max_merged_error':maxerr_merged,'table3':str(OUT/'tables/table3_development.csv'),'table4':str(OUT/'tables/table4_budget.csv')},ensure_ascii=False))
