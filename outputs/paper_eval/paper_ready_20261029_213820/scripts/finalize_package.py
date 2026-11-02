"""Apply editorial clarifications, validate data/exports and write delivery reports."""
from pathlib import Path
import re,json,hashlib,sys,platform
import numpy as np
import pandas as pd
from PIL import Image
import fitz
import xml.etree.ElementTree as ET
from datetime import datetime,timezone,timedelta
OUT=Path(__file__).resolve().parents[1]; ROOT=OUT.parents[2]
def write(p,s): (OUT/p).write_text(s,encoding='utf-8')
def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def english_spacing(s):
    # Preserve model/checkpoint names and numerical thousands separators.
    words='and|or|seed|Table|Tables|Fig|official|development|of|across|only|for|in|at|At|from|to|out|with|versus|over|by|local|the|same|a|no|was|were|is|are|into|using|contains|contained|than|respectively|remained|remain|numbered|between|but'
    s=re.sub(r'\b('+words+r')(?=\d)',r'\1 ',s)
    s=re.sub(r'\b(and|in|for|versus|to)(?=v[12]\b)',r'\1 ',s)
    s=re.sub(r'(?<=\d)pp\b',' pp',s)
    s=re.sub(r'\bFig\.(?=\d|S\d)','Fig. ',s)
    s=re.sub(r',(?=\d{1,2}(?!\d))',', ',s)
    s=re.sub(r'\b(seed)\s?(\d{8})',r'\1=\2',s)
    preserve={'Qwen','v','F','A','B','C','S','AI','fig','NF','FP'}
    s=re.sub(r'\b([A-Za-z]+)(?=\d)',lambda m:m[1] if m[1] in preserve else m[1]+' ',s)
    s=re.sub(r'([A-Za-z])(?=−\d)',r'\1 ',s)
    s=re.sub(r'([:;])(?=\d)',r'\1 ',s)
    return s
zh=(OUT/'manuscript_zh.md').read_text(encoding='utf-8')
zh=zh.replace('在独立列示的正式40题AI评价子集中','在正式生成前从206道answer_in_text=1问题中预选、单列报告的40题AI评价子集中')
zh=zh.replace('实际输出token均值在无RAG时','实际输出token（含EOS）均值在无RAG时')
zh=zh.replace('下降7.47个百分点（表5敏感性；图4b）。','下降7.47个百分点（表5敏感性；图4b）。75.125与67.875是97题覆盖比例之和，并非答对题数。')
zh=zh.replace('也不证明陈述经独立来源核验后为假。','也不证明陈述为科学事实错误；本轮未作独立来源核验。')
zh=zh.replace('两套内容评价均由AI流程完成，独立人工评分人数为0。','两套内容评价均由AI流程完成，独立人工评分人数为0，两套评价均未独立记录精确模型版本与解码参数。')
zh=zh.replace('现有证据支持按评价维度讨论有限的功能互补与续训取舍','现有证据支持按评价维度区分作用与续训取舍，功能互补仍须限定范围')
write('manuscript_zh.md',zh)
en=(OUT/'manuscript_en.md').read_text(encoding='utf-8')
en=en.replace('In the separately reported official40-question AI subset','In the separately reported official40-question AI subset preselected before generation from the206 questions with answer_in_text=1')
en=en.replace('Mean generated tokens decreased','Mean generated tokens, including EOS, decreased')
en=en.replace('a decrease of7.47pp (Table5 sensitivity; Fig.4b).','a decrease of7.47pp (Table5 sensitivity; Fig.4b). The numerators75.125 and67.875 are sums of97 question-level coverage proportions, not counts of correct answers.')
en=en.replace('did not establish scientific falsity after independent source verification.','did not establish that the statements were scientifically false; independent-source verification was not performed.')
if 'Neither independently recorded an exact judge build and decoding configuration.' not in en:
    en=en.replace('Both content evaluations were AI processes, with zero independent human raters.','Both content evaluations were AI processes, with zero independent human raters. Neither independently recorded an exact judge build and decoding configuration.')
en=en.replace('supports metric-specific functional complementarity and continuation trade-offs within this system','supports a metric-specific division of effects and continuation trade-offs within this system')
write('manuscript_en.md',english_spacing(en))
for p in [OUT/'tables/captions_en.md',OUT/'figures/captions_en.md']:
    p.write_text(english_spacing(p.read_text(encoding='utf-8')),encoding='utf-8')
# Keep a complete copy of legends with each language draft, while retaining standalone legends.
for lang in ['zh','en']:
    p=OUT/f'manuscript_{lang}.md';s=p.read_text(encoding='utf-8')
    marker='<!-- legends_appendix -->'
    s=s.split(marker)[0].rstrip()
    tables=(OUT/f'tables/captions_{lang}.md').read_text(encoding='utf-8').replace('# ', '## ',1)
    figures=(OUT/f'figures/captions_{lang}.md').read_text(encoding='utf-8').replace('# ','## ',1)
    p.write_text(s+'\n\n'+marker+'\n\n'+tables+'\n\n'+figures,encoding='utf-8')
# Verify arithmetic and identity at delivery boundaries, in addition to full source reanalyses.
checks=[]
def check(name,test,detail=''):
    checks.append(dict(check=name,passed=bool(test),detail=detail))
formal=pd.read_csv(OUT/'tables/table2_formal.csv')
check('official counts',formal.n_questions.eq(227).all() and formal.source_groups.eq(181).all() and formal.option_accuracy_n.tolist()==[185,200,180,204])
for num,den,val in [('option_accuracy_n','option_accuracy_denom','option_accuracy'),('strict_letter_accuracy_n','strict_letter_accuracy_denom','strict_letter_accuracy'),('format_valid_n','format_denom','format_compliance'),('stripped_f1_score_sum','stripped_f1_question_denom','stripped_f1')]:
    check('formal arithmetic '+val,np.allclose(formal[num]/formal[den],formal[val],rtol=0,atol=1e-12))
dev=pd.read_csv(OUT/'tables/table3_development.csv'); budget=pd.read_csv(OUT/'tables/table4_budget.csv')
check('dev six conditions',len(dev)==6 and dev.questions.eq(101).all() and dev.source_groups.eq(91).all())
check('budget eight conditions',len(budget)==8 and budget.questions.eq(101).all() and set(budget.max_new_tokens)=={96,160})
ai=pd.read_csv(OUT/'tables/table5_ai_content.csv');keep=ai.denominator.gt(0)
check('AI proportion arithmetic',np.allclose(ai.loc[keep,'numerator']/ai.loc[keep,'denominator'],ai.loc[keep,'value'],atol=1e-12,equal_nan=True))
cross=pd.read_csv(OUT/'tables/table6_f1_content.csv'); cols=['v2_better','v2_worse','equivalent','tradeoff','uncertain']
check('cross-tab total',cross[cols].to_numpy().sum()==101 and cross.row_total.tolist()==[24,35,42])
check('cross-tab verdict marginal',cross[cols].sum().tolist()==[5,17,76,1,2])
cases=pd.read_csv(OUT/'cases.csv');check('required cases',len(cases)==11 and all(any(cases.id.str.endswith(i)) for i in ['A1234','A1726','A718','A331','A1825','A862','A1109']))
check('case F1 arithmetic',np.allclose((cases.v2_f1-cases.v1_f1)*100,cases.f1_delta_pp,atol=1e-10))
claim=pd.read_csv(OUT/'claim_evidence_map.csv');check('claim map',len(claim)>=15 and not claim.isna().any().any(),str(claim.columns.tolist()))
source_hashes=pd.read_csv(OUT/'audit/source_files_sha256.csv');changed=[]
for r in source_hashes.itertuples():
    p=Path(r.path)
    if not p.is_file() or sha(p)!=r.sha256: changed.append(str(p))
check('source immutability',not changed,f'{len(source_hashes)} source files; changed={changed}')
formats=[]
for stem in ['fig1_formal','fig2_stages','fig3_f1_content','fig4_paired','figS1_ai40']:
    svg=OUT/'figures'/f'{stem}.svg'; tree=ET.parse(svg);texts=len(tree.findall('.//{http://www.w3.org/2000/svg}text'))
    with Image.open(OUT/'figures'/f'{stem}.png') as im: im.verify()
    with Image.open(OUT/'figures'/f'{stem}.png') as im: size=im.size;dpi=im.info.get('dpi')
    with fitz.open(OUT/'figures'/f'{stem}.pdf') as pdf:
        text=pdf[0].get_text(); pages=len(pdf)
        # PDF preview is rendered with Python only, preserving the selected backend contract.
        pdf[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/'audit'/f'{stem}_pdf_preview.png')
    check(stem+' exports',texts>10 and pages==1 and len(text)>100 and min(size)>1500 and dpi[0]>590,f'{size}; dpi={dpi}; SVG text objects={texts}; PDF text chars={len(text)}')
    source=OUT/'figures'/f'{stem}_source.csv';check(stem+' source CSV',source.exists() and len(pd.read_csv(source))>0)
    formats.append(dict(figure=stem,png_width=size[0],png_height=size[1],dpi=dpi[0],svg_text_objects=texts,pdf_pages=pages,pdf_text_chars=len(text)))
pd.DataFrame(formats).to_csv(OUT/'audit/figure_export_validation.csv',index=False,encoding='utf-8-sig')
# CSV source data and plot/table alignment.
fp=pd.read_csv(OUT/'figures/fig1_formal_source.csv'); check('figure1 table alignment',np.allclose(fp.stripped_f1,formal.stripped_f1) and np.allclose(fp.option_accuracy,formal.option_accuracy))
f2=pd.read_csv(OUT/'figures/fig2_stages_source.csv');check('figure2 table alignment',np.allclose(f2.f1,dev.f1))
f3=pd.read_csv(OUT/'figures/fig3_f1_content_source.csv');check('figure3 table alignment',np.array_equal(f3[cols].values,cross[cols].values))
f4=pd.read_csv(OUT/'figures/fig4_paired_source.csv');check('figure4 coverage alignment',np.allclose(f4[['v1_weighted_point_coverage','v2_weighted_point_coverage']].mean().values,[75.125/97,67.875/97]))
check('manuscript key numeric trace',all(s in zh for s in ['185/227','200/227','180/227','204/227','73.76%','65.96%','77.45%','69.97%','−3.71','−6.64','−1.02','3.15','3.13','129/207','77/101','124/157','76/101']))
checksdf=pd.DataFrame(checks);checksdf.to_csv(OUT/'audit/delivery_checks.csv',index=False,encoding='utf-8-sig')
write('audit/runtime.json',json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'completed_at_shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),'all_checks_passed':bool(checksdf.passed.all())},ensure_ascii=False,indent=2))
report='''# 复核与验收报告

本轮仅重算既有预测和既有AI标签，没有训练、付费评价、模型推理或正式测试重新评估。全部新增科研产物位于本目录。执行起点：2026-10-29 21:38:20（Asia/Shanghai）；完成时间见audit/runtime.json。

## 数据、范围与复现

- 正式MCQA/OpenQA各227题×4条件，任务共享227个原始ID；181主组/144保守组。逐题指标、均值、原区间及配对差值均复现，含225/206/21切片。
- 正式AI40从206题answer_in_text=1中事前预选，160质量/80证据评分；40主组/32保守组统计逐项复现，与AI101分开。
- 开发10条件各101题、共1010输出，91主组/90合并支撑文本组。逐条重新计算F1、长度、引用和截断，复现162项配对区间及162项敏感性区间；最大浮点误差约3.6e−15，不构成数值冲突。
- AI101只有v1/v2 RAG160条件，101配对/202回答；保留141要点、97可评题/4NA。原始标签逐条关联回答与F1；141点摘录均可在实际文本定位。统计未创建新内容标签或CI。
- 正式冻结协议、代码、数据和适配器哈希比对一致，见audit/frozen_hash_validation.csv。根清单保护178个源文件，交付时逐项复核无改变。没有对基础模型大权重重新计算哈希，不将其描述为本轮完整认证。

## 原材料差异及采用依据

没有发现原始预测、逐题指标和指定汇总中的实质数值冲突。保留以下口径区别，避免造成表观冲突：

1. 原差值表偶用“%”表示差值，本交付统一写百分点；原CSV部分以0—1保存，显示前×100。正式F1差4.2百分点与35.7−31.4得到4.3的表观差异来自先舍入：采用未舍入逐题差值4.2434百分点。
2. 正式FT/v2不是历史中文应用模型，亦非v1 epoch_2。开发补充重新统一实际评估提示，不把历史提示不同的成绩混入。
3. 开发91 EID组与90支撑文本合并组是同一101题的不同聚类；AI证据窗口还涉及更多检索来源，不能把检索来源数当独立问题组数。
4. 原AI报告覆盖采用要点等权104/141→93/141；本交付新增每题等权75.125/97→67.875/97为敏感性，不覆盖原标签。分子分别为加权覆盖分数和题级比例之和。
5. AI目录名含20261028，但实际审查跨2026-10-28/29续跑至完成；本交付如实注明，不据此假设同一模型快照。
6. MCQA内部字段accuracy实际上是严格单字母正确率，mc_prefix_sensitivity_accuracy为宽松开头选项正确率；主表已显式重命名。两者均以全部227题为分母。
7. 生成160上限下OpenQA与MCQA16上限的截断是不同任务指标。引用ID有效只检验完整标记，未闭合残片另计；不因100%完整标记有效而宣称引用无问题。

采用依据：原始回答及冻结解析协议优先，已有逐题分析用于双向核对；经逐项复现的配对区间优先于从四舍五入均值重算差值。AI标签原样保留，争议列明而不以本轮文字修改原判。

## 正文与结论审核

正式、开发、AI内容三个独立子任务对中英文正文进行了只读交叉审核，数字全部与所负责表格一致；主代理完成空格、分母和评价范围澄清。主要主张均有claim_evidence_map.csv条目，包含30条主张与不支持的推断。正文区分直接显示、提示与尚待验证；F1/格式/ID解析/片段支持/独立科学验证分开。未把开发集写成新测试集，未把AI写成人工评价，未将17个综合更差标签写成17个核心事实错误。

## 图与文件验收

5幅图各有SVG/PDF/600dpi PNG及底层CSV，图注中英齐备。PNG逐幅目视检查；SVG经XML解析并确认保留文本；PDF打开、提取文本并由Python渲染预览。数据范围、轴量纲与样本标注清楚，配对内容图不加虚构区间。图4细线重合不代表只有可见线数，已在图注明确。

自动验收逐项结果：

'''+checksdf[['check','passed','detail']].to_markdown(index=False)+'''

## 尚未解决的问题

精确AI版本/解码参数、先验接触、主观要点、缺少独立专家与完整来源验证、单模型单轨迹、额外训练与目标/提示/采样混杂、预算不等长、正式v1对照缺失，均保留为证据缺口。这些不阻止描述现有正式指标及开发行为取舍，但限制因果互补、普遍事实质量或效率论点。详细最小补充设计见limitations_and_gaps.md。本交付不是独立重复实验或AI标签真值认证。
'''
write('validation_report.md',report)
if not checksdf.passed.all(): raise AssertionError(checksdf[~checksdf.passed].to_string())
print(f'{len(checks)} validation checks passed; {len(source_hashes)} input files unchanged.')
# Human-readable file index and reproducible checksum manifest (exclude self-referential indices).
index=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in ['file_index.csv','delivery_sha256.csv'] and '__pycache__' not in str(p):
        index.append(dict(relative_path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
pd.DataFrame(index).to_csv(OUT/'file_index.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(index)[['relative_path','sha256']].to_csv(OUT/'delivery_sha256.csv',index=False,encoding='utf-8-sig')
