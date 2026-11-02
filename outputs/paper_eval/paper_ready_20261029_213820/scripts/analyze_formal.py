"""Read-only audit of frozen predictions and adopted AI scores. No inference/ratings.

Run: python analyze_formal.py --project D:/weitiao/ocean-chat --out OUTPUT
Uses project scoring definitions after checking recorded hashes. Writes only OUTPUT.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys
import numpy as np
import pandas as pd

def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def md(df):
    def cell(x): return str(x).replace('|','\\|').replace('\n','<br>')
    return '| '+' | '.join(map(cell,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell,row))+' |' for row in df.itertuples(index=False,name=None))+'\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,default=Path('D:/weitiao/ocean-chat')); ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args()
    root,out=args.project.resolve(),args.out.resolve(); sys.path.insert(0,str(root))
    from paper_eval.english_report import score_record, COMPARISONS
    from paper_eval.official_english_report import summarize,paired_differences,slice_masks,metric_names
    from paper_eval.adopt_ai_evaluation import estimate, METRICS
    ev=root/'outputs/paper_eval'; conds=('base','base_rag','lora','lora_rag'); names=dict(zip(conds,['Base','Base+RAG','FT','FT+RAG']))
    checks=[]; scores={}; analyses={}; runpaths={}; metrics_long=[]; contrast_long=[]; wide=[]
    frozen=load(root/'paper_data/frozen_english_v1_20261017/freeze_manifest.json')
    hashes=[]
    for category in ['data_sha256','code_sha256','model_files_sha256']:
      for rel,expected in frozen[category].items():
        if category=='model_files_sha256' and not ('adapter_' in rel): continue
        p=root/rel; actual=digest(p) if p.exists() else None
        hashes.append(dict(category=category,path=str(p),expected=expected,actual=actual,status='match' if actual==expected else 'MISMATCH' if actual else 'missing'))
    protocol=root/'paper_data/frozen_english_v1_20261017/protocol.json'
    hashes.append(dict(category='protocol',path=str(protocol),expected=frozen['protocol_sha256'],actual=digest(protocol),status='match' if digest(protocol)==frozen['protocol_sha256'] else 'MISMATCH'))
    for task in ['mcqa','openqa']:
      run=ev/f'en_test_{task}_frozen_v1_20261017'; stored=load(ev/f'en_test_{task}_frozen_v1_report_20261017/analysis.json'); analyses[task]=stored; runpaths[task]=run
      raw=lines(run/'answers.jsonl'); qs=lines(run/'selected_questions.jsonl'); qmap={q['id']:q for q in qs}
      assert len(raw)==908 and len(qs)==227 and len({(r['condition'],r['id']) for r in raw})==908
      assert all(q['split']=='test' and q['original_split']=='test' for q in qs)
      assert all({r['id'] for r in raw if r['condition']==c}==set(qmap) for c in conds)
      for filename,expected in stored['source_artifacts_sha256'].items():
        p=run/filename; hashes.append(dict(category=task+'_run',path=str(p),expected=expected,actual=digest(p),status='match' if digest(p)==expected else 'MISMATCH'))
      scored=[]; oldindex={(r['condition'],r['id']):r for r in stored['per_answer_automatic_scores']}
      for r in raw:
        q=qmap[r['id']]; s=score_record(r,q,task)
        s.update(original_id=q['original_id'],corpus=q['corpus'],source_group_conservative=q['source_group_conservative'],original_assessments=q['original_assessments'],answer_has_citation=bool(s['citation_markers']),answer_has_at_least_one_and_all_markers_resolved=bool(s['citation_markers']>0 and s['citation_markers_resolved']==s['citation_markers']),hit_token_limit=r['metrics']['hit_token_limit'])
        if task=='mcqa': s['mc_format_invalid']=not s['mc_valid_format']
        old=oldindex[(r['condition'],r['id'])]
        for key in metric_names(task)+['citation_markers','citation_markers_resolved']:
          assert s[key]==old[key], (task,r['id'],r['condition'],key,s[key],old[key])
        s['output_tokens_including_eos']=r['metrics']['output_tokens_including_eos']; scored.append(s)
      scores[task]=scored
      for slicename,mask in slice_masks(qmap).items():
       subset=[r for r in scored if r['id'] in mask]
       for grouping in ['source_group','source_group_conservative']:
        st=stored['slices'][slicename]['groupings'][grouping]
        for c in conds:
         got=summarize([r for r in subset if r['condition']==c],grouping,metric_names(task),True,len(mask))
         assert got==st['by_condition'][c],(task,slicename,grouping,c)
         for key,z in got['metrics'].items():
          ci=z['ci95'] or [None,None]
          metrics_long.append(dict(dataset='official_test',task=task,slice=slicename,grouping=grouping,condition=names[c],condition_key=c,metric=key,value=z['mean'],ci_low=ci[0],ci_high=ci[1],numerator=z['numerator_total'],denominator=z['denominator_total'],n_questions=z['questions'],source_groups=z['source_groups'],source=str(ev/f'en_test_{task}_frozen_v1_report_20261017/analysis.json')))
        paired=paired_differences(scored,mask,conds,grouping,metric_names(task),dict.fromkeys(conds,True))
        assert paired==st['paired_differences'],(task,slicename,grouping,'paired')
        for comparison,z in paired.items():
         for metric,m in z['metrics'].items():
          ci=m['ci95'] or [None,None]
          contrast_long.append(dict(dataset='official_test',task=task,slice=slicename,grouping=grouping,comparison=comparison,metric=metric,difference=m['mean'],ci_low=ci[0],ci_high=ci[1],unit='proportion_difference',paired_questions=m['paired_question_denominator'],source_groups=m['source_groups']))
      pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in scored]).to_csv(out/'audit'/f'formal_{task}_per_answer.csv',index=False,encoding='utf-8-sig')
      checks.append(f'{task}: 908 raw records, 227 unique original questions, 4 complete conditions; all stored per-item metrics and all 4 slices × 2 groupings × summaries/paired bootstrap reproduced exactly.')
    assert {r['original_id'] for r in scores['mcqa']}=={r['original_id'] for r in scores['openqa']}
    for c in conds:
      m=analyses['mcqa']['slices']['all_227']['groupings']['source_group']['by_condition'][c]['metrics']; o=analyses['openqa']['slices']['all_227']['groupings']['source_group']['by_condition'][c]['metrics']
      raw=lines(runpaths['openqa']/'answers.jsonl'); rr=[r for r in raw if r['condition']==c]
      row=dict(condition=names[c],condition_key=c,n_questions=227,source_groups=181,conservative_groups=144)
      for label,key in [('option_accuracy','mc_prefix_sensitivity_accuracy'),('strict_letter_accuracy','accuracy')]:
        row[label+'_n']=int(m[key]['numerator_total']); row[label+'_denom']=227; row[label]=m[key]['mean']; row[label+'_ci_low'],row[label+'_ci_high']=m[key]['ci95']
      row.update(format_valid_n=227-int(m['mc_format_invalid']['numerator_total']),format_denom=227,format_compliance=1-m['mc_format_invalid']['mean'],format_ci_low=1-m['mc_format_invalid']['ci95'][1],format_ci_high=1-m['mc_format_invalid']['ci95'][0])
      for label,key in [('stripped_f1','citation_stripped_token_f1'),('raw_f1','token_f1'),('stripped_em','citation_stripped_exact_match')]:
        row[label]=o[key]['mean']; row[label+'_ci_low'],row[label+'_ci_high']=o[key]['ci95']; row[label+'_score_sum']=o[key]['numerator_total']; row[label+'_question_denom']=227
      for label,key in [('cited_answers','answer_has_citation'),('all_markers_resolved_answers','answer_has_at_least_one_and_all_markers_resolved'),('truncated_answers','hit_token_limit')]: row[label+'_n']=int(o[key]['numerator_total']); row[label+'_denom']=227
      row.update(resolved_markers_n=int(o['marker_resolution_ratio']['numerator_total']),markers_denom=int(o['marker_resolution_ratio']['denominator_total']),mean_output_tokens=float(np.mean([r['metrics']['output_tokens_including_eos'] for r in rr])),gold_source_hit_n=int(o['gold_source_recall_at_k']['numerator_total']) if o['gold_source_recall_at_k']['mean'] is not None else None,gold_source_hit_denom=int(o['gold_source_recall_at_k']['denominator_total']))
      wide.append(row)
    df=pd.DataFrame(wide); df.to_csv(out/'tables/table2_formal.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(metrics_long).to_csv(out/'tables/table2_formal_metrics_long.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(contrast_long).to_csv(out/'tables/table2_formal_contrasts.csv',index=False,encoding='utf-8-sig')
    display=[]
    for r in wide:
      def ratio(k): return f"{r[k+'_n']}/{r[k+'_denom']} ({100*r[k+'_n']/r[k+'_denom']:.1f}%)"
      def ci(k): return f"{100*r[k]:.1f} [{100*r[k+'_ci_low']:.1f}, {100*r[k+'_ci_high']:.1f}]"
      display.append({'条件':r['condition'],'MC选项准确率':ratio('option_accuracy')+'; CI '+ci('option_accuracy'),'严格单字母且正确':ratio('strict_letter_accuracy'),'格式遵循':f"{r['format_valid_n']}/227 ({100*r['format_compliance']:.1f}%)",'去引用F1 % [95% CI]':ci('stripped_f1'),'去引用EM':f"{int(r['stripped_em_score_sum'])}/227 ({100*r['stripped_em']:.1f}%)",'原始F1 %':f"{100*r['raw_f1']:.1f}",'带标记回答':ratio('cited_answers'),'可解析标记/全部':f"{r['resolved_markers_n']}/{r['markers_denom']}",'OpenQA截断':ratio('truncated_answers'),'实际输出token均值':f"{r['mean_output_tokens']:.2f}"})
    notes='''\n表2注：Pirá正式test同227题，MCQA与OpenQA为同题两种任务，不能相加为454个独立问题。FT为最终英文v2/epoch_1。MC选项准确率按冻结开头字母规则解析；严格指标同时要求单字母格式及答案正确；格式遵循只判单字母格式，不判答案。F1为去引用后与唯一原始参考答案的SQuAD式词面重合宏均值（分数和/227，非事实正确率），CSV保留分数和与题数。EM为归一化完全匹配。引用可解析按标记总数加权，0/0为未定义；其值不说明语义支持。输出token含EOS，截断按hit_token_limit。95%区间使用181来源组、2000次配对来源组bootstrap、seed=20261017；问题统计按题加权，标记比率按标记合计。144组敏感性及225/206/21题切片保存在long/contrasts CSV。所有数值与原报告逐项复算一致。\n\nSources: outputs/paper_eval/en_test_{mcqa,openqa}_frozen_v1_20261017/{answers,selected_questions}.jsonl；对应 _report_20261017/analysis.json；paper_data/frozen_english_v1_20261017/{protocol,freeze_manifest}.json。差值单位为百分点；差值用未舍入均值计算。\n'''
    (out/'tables/table2_formal.md').write_text('# 表2 正式测试四组结果\n\n'+md(pd.DataFrame(display))+notes,encoding='utf-8')
    # Reproduce adopted 40-question AI summaries and shared-draw contrasts without changing labels.
    ai_dir=ev/'ai_evaluation_adopted_20261018'; ai=load(ai_dir/'analysis.json'); ar=load(ai_dir/'per_answer_ai_scores.json')
    if isinstance(ar,dict): ar=ar.get('rows',ar.get('scores',ar))
    assert isinstance(ar,list) and len(ar)==160
    arows=[]; acontrast=[]
    for grouping in ['source_group','source_group_conservative']:
      groups=sorted({r[grouping] for r in ar}); draws=np.random.default_rng(20261018).integers(0,len(groups),(2000,len(groups))); boots={}; summ={}
      for c in conds:
        boots[c]={}; summ[c]={}
        for metric in METRICS:
          got,b=estimate([r for r in ar if r['condition']==c],metric,grouping,groups,draws); expected=ai['groupings'][grouping]['by_condition'][c][metric]
          assert got==expected,(c,metric,grouping)
          summ[c][metric]=got; boots[c][metric]=b; ci=got['ci95'] or [None,None]
          arows.append(dict(dataset='official_test_AI40',grouping=grouping,condition=names[c],condition_key=c,metric=metric,value=got['mean'],ci_low=ci[0],ci_high=ci[1],numerator=got['numerator'],denominator=got['denominator'],n_questions=40,source_groups=len(groups)))
      for l,r in COMPARISONS:
        for metric,expected in ai['groupings'][grouping]['paired_differences'][f'{l}_minus_{r}'].items():
          b=boots[l][metric]-boots[r][metric]; ci=np.quantile(b[np.isfinite(b)],[.025,.975]); delta=summ[l][metric]['mean']-summ[r][metric]['mean']; assert delta==expected['mean'] and np.allclose(ci,expected['ci95'],atol=1e-15)
          acontrast.append(dict(dataset='official_test_AI40',grouping=grouping,comparison=f'{l}_minus_{r}',metric=metric,difference=delta,ci_low=ci[0],ci_high=ci[1],paired_questions=40,unit='proportion_difference' if metric.startswith('citation_') else 'score_points_0_to_4'))
    adf=pd.DataFrame(arows); adf.to_csv(out/'tables/table2_ai40_separate.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(acontrast).to_csv(out/'tables/table2_ai40_contrasts.csv',index=False,encoding='utf-8-sig')
    (out/'tables/table2_ai40_separate.md').write_text('# 正式测试的独立呈现部分：40题AI评价\n\n'+md(adf[adf.grouping=='source_group'].drop(columns=['dataset','grouping','condition_key']))+'\n注：这不是新增独立测试集，也不是人评。40题为206道answer_in_text=1题中的预先选定子集，每组40，质量评分160条，两个RAG条件证据评分80条；40主来源组，32保守组。事实答题质量、必要完整性、忠实性为0—4均分；前两者分母40回答，忠实性仅RAG适用。citation_support为受支持的陈述—引用关联/所有关联；citation_coverage为有支持引用的陈述/需引用陈述，均使用合计数而非逐回答等权。已按原协议2000次、seed=20261018、共享来源组抽样逐项复现区间。区间不包含AI评价误差。单AI流程，无独立人类评分；精确评审模型版本未记录。\n\n来源：outputs/paper_eval/ai_evaluation_adopted_20261018/per_answer_ai_scores.json 与 analysis.json。\n',encoding='utf-8')
    checks.append('AI40: all 4-condition/5-metric means, denominators and 40/32-source-group CIs/paired differences reproduced exactly from original adopted item labels.')
    pd.DataFrame(hashes).to_csv(out/'audit/frozen_hash_validation.csv',index=False,encoding='utf-8-sig')
    summary={'formal_conditions':wide,'formal_checks':checks,'hash_mismatches':[h for h in hashes if h['status']!='match'],'ai40_scope':{k:ai[k] for k in ['questions','quality_answers','evidence_answers','separate_reference_reviews','scope','bootstrap']},'source_groups':{'formal':181,'formal_conservative':144,'ai40':40,'ai40_conservative':32},'ai40_primary_summary':ai['groupings']['source_group'],'formal_primary_paired':{task:analyses[task]['slices']['all_227']['groupings']['source_group']['paired_differences'] for task in analyses}}
    save(out/'audit/formal_summary.json',summary)
    (out/'audit/formal_audit.md').write_text('# 正式数据复核\n\n'+'\n\n'.join(checks)+'\n\n## 冻结哈希\n\n'+md(pd.DataFrame(hashes)[['category','path','status']])+'\n\n## 差异与解释\n\n逐题原始预测、原报告JSON和中文汇总的关键指标一致；中文汇总四舍五入至一位小数，不以舍入数直接相减。正式MCQA无效格式与严格答错分开；F1分子是分数和，不是正确题数。正式报告expert_review=pending保留原状态；后续采纳AI评分不能改写成人评完成。两任务ID交集为227，不可合并为454。\n\n原报告accuracy是严格单字母且正确；mc_prefix_sensitivity_accuracy为主选项准确率；不能把前者称为事实正确率。引用标记编号可解析不说明陈述得到证据支持。主来源组通过已知EID/归一化文本合并，无法保证整篇来源独立；144组为将corpus2合并的依赖敏感性假设。AI40三个分片属于同一AI流程，不是3位评审。\n\n没有重跑模型或改变任何标签；没有读取多GB基座权重；小适配器权重及数据、代码、协议按冻结记录逐项核验，完整值见frozen_hash_validation.csv。\n',encoding='utf-8')
    print(json.dumps({'formal':wide,'checks':checks,'hash_mismatches':summary['hash_mismatches']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

