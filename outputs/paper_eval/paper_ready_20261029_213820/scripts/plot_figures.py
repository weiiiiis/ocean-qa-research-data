"""Render publication figures from verified delivery tables and preserved AI labels."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
T=OUT/'tables'; F=OUT/'figures'
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'savefig.facecolor':'white'})
C=['#788795','#427fa0','#b29779','#ad644a']
def save(fig,name):
    for ext in ['svg','pdf','png']: fig.savefig(F/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)
def source(df,name): df.to_csv(F/f'{name}_source.csv',index=False,encoding='utf-8-sig')
def panel(ax,letter,title): ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=9)
def points(ax,x,y,lo,hi,colors,digits=1):
    for i in range(len(x)):
        ax.errorbar(x[i],y[i],yerr=[[y[i]-lo[i]],[hi[i]-y[i]]],fmt='o',color=colors[i],capsize=3,ms=5,elinewidth=1)
        ax.annotate(f'{y[i]:.{digits}f}',(x[i],hi[i]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
    ax.grid(axis='y',color='#e6e9ec',lw=.5); ax.set_axisbelow(True)
formal=pd.read_csv(T/'table2_formal.csv'); source(formal,'fig1_formal')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
for ax,(letter,title,m,lo,hi) in zip(axs.flat,[('a','MCQA: parsed option accuracy','option_accuracy','option_accuracy_ci_low','option_accuracy_ci_high'),('b','MCQA: strict correct letter','strict_letter_accuracy','strict_letter_accuracy_ci_low','strict_letter_accuracy_ci_high'),('c','MCQA: format compliance','format_compliance','format_ci_low','format_ci_high'),('d','OpenQA: lexical overlap','stripped_f1','stripped_f1_ci_low','stripped_f1_ci_high')]):
    points(ax,list(range(4)),formal[m].values*100,formal[lo].values*100,formal[hi].values*100,C)
    ax.set_xticks(range(4),formal.condition,rotation=15); ax.set_ylabel('Percent (%)' if m!='stripped_f1' else 'Citation-stripped F1 (%)'); ax.set_ylim(0,110 if m!='stripped_f1' else 45);ax.set_xlim(-.45,3.45);panel(ax,letter,title)
fig.suptitle('Official test: 227 original questions, 181 source groups',fontsize=10)
save(fig,'fig1_formal')
dev=pd.read_csv(T/'table3_development.csv'); dev=dev[dev.max_new_tokens==160].copy(); source(dev,'fig2_stages')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
for j,rag in enumerate([False,True]):
    d=dev[dev.rag==rag].set_index('stage').loc[['base','v1','v2']]
    ax=axs[0,j]; points(ax,[0,1,2],d.f1.values*100,d.f1_ci95_low.values*100,d.f1_ci95_high.values*100,[C[0],C[2],C[3]])
    ax.plot(range(3),d.f1.values*100,c='#b9bec3',lw=.8,zorder=0);ax.set_xticks(range(3),['Base','v1','v2']);ax.set_ylabel('Citation-stripped F1 (%)');ax.set_ylim(0,50);ax.set_xlim(-.4,2.4);panel(ax,'ab'[j],('No RAG' if not rag else 'BM25 RAG')+' / lexical overlap')
    ax=axs[1,j];y=d.mean_output_tokens_including_eos.values
    ax.plot(range(3),y,c='#aab0b7',lw=1)
    for i,v in enumerate(y):ax.scatter(i,v,c=[C[0],C[2],C[3]][i],s=25);ax.annotate(f'{v:.1f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xticks(range(3),['Base','v1','v2']);ax.set_ylabel('Mean generated tokens (incl. EOS)');ax.set_ylim(0,90);ax.set_xlim(-.4,2.4);panel(ax,'cd'[j],('No RAG' if not rag else 'BM25 RAG')+' / actual length');ax.grid(axis='y',color='#e6e9ec',lw=.5)
fig.suptitle('Exploratory development: 101 questions; maximum 160 tokens',fontsize=10);save(fig,'fig2_stages')
paired=pd.read_csv(ROOT/'outputs/paper_eval/ai_content_audit_20261028_175905/paired_reviews.csv')
verdicts=['v2_better','v2_worse','equivalent','tradeoff','uncertain']; dirs=['increase','decrease','unchanged']
ct=pd.crosstab(paired.f1_direction,paired.v2_content_result).reindex(index=dirs,columns=verdicts,fill_value=0)
ct.rename_axis('F1_direction').reset_index().to_csv(F/'fig3_f1_content_source.csv',index=False,encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(7.2,3.1),layout='constrained');im=ax.imshow(ct.values,cmap='Blues',vmin=0,vmax=40,aspect='auto')
for i in range(3):
    for j in range(5):ax.text(j,i,str(ct.iloc[i,j]),ha='center',va='center',color='white' if ct.iloc[i,j]>20 else '#203449',fontsize=11)
ax.set_xticks(range(5),['v2 better','v2 worse','Equivalent','Trade-off','Uncertain']);ax.set_yticks(range(3),[f'F1 increased (n={ct.iloc[0].sum()})',f'F1 decreased (n={ct.iloc[1].sum()})',f'F1 unchanged (n={ct.iloc[2].sum()})']);ax.set_xlabel('Single-AI comparative content judgment');ax.set_title('Development AI audit: 101 paired RAG answers (v2 versus v1)',loc='left',fontsize=10,pad=12);fig.colorbar(im,ax=ax,label='Questions',shrink=.8);save(fig,'fig3_f1_content')
cover=paired[['id','source_group','v1_refusal','v2_refusal','v1_weighted_point_coverage','v2_weighted_point_coverage','v1_points_covered','v1_points_partial','v1_points_omitted','v1_assessable_points','v2_points_covered','v2_points_partial','v2_points_omitted','v2_assessable_points']].copy();source(cover,'fig4_paired')
trans=pd.crosstab(paired.v1_refusal.eq('erroneous'),paired.v2_refusal.eq('erroneous')).reindex(index=[False,True],columns=[False,True],fill_value=0);source(trans.rename_axis('v1_erroneous_refusal').reset_index(),'fig4_refusal')
cov=cover.dropna(subset=['v1_weighted_point_coverage','v2_weighted_point_coverage'])
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained');ax=axs[0];ax.imshow(trans.values,cmap='Blues',vmin=0,vmax=93)
for i in range(2):
    for j in range(2):ax.text(j,i,str(trans.iloc[i,j]),ha='center',va='center',fontsize=14,color='white' if trans.iloc[i,j]>40 else '#203449')
ax.set_xticks([0,1],['No','Yes']);ax.set_yticks([0,1],['No','Yes']);ax.set_xlabel('v2 erroneous refusal');ax.set_ylabel('v1 erroneous refusal');panel(ax,'a','Paired transitions (n=101)')
ax=axs[1]
for _,r in cov.iterrows():ax.plot([0,1],[100*r.v1_weighted_point_coverage,100*r.v2_weighted_point_coverage],c='#aebdca',alpha=.25,lw=.6)
means=cov[['v1_weighted_point_coverage','v2_weighted_point_coverage']].mean().values*100
ax.plot([0,1],means,c=C[3],lw=2,marker='o',ms=5,label='Equal-question mean')
for i,m in enumerate(means):ax.annotate(f'{m:.2f}%',(i,m),xytext=(0,9),textcoords='offset points',ha='center',color=C[3],fontweight='bold')
ax.set_xlim(-.3,1.3);ax.set_ylim(-3,110);ax.set_xticks([0,1],['v1 + RAG','v2 + RAG']);ax.set_ylabel('AI-defined point coverage (%)');panel(ax,'b',f'Paired coverage (n={len(cov)}; 4 NA)');ax.text(.5,5,'Point-weighted: 104/141 → 93/141',ha='center',fontsize=7);ax.grid(axis='y',color='#e6e9ec',lw=.5)
fig.suptitle('Development AI audit: descriptive changes after continuation',fontsize=10);save(fig,'fig4_paired')
ai=pd.read_csv(T/'table2_ai40_separate.csv');ai=ai[ai.grouping=='source_group'].copy();source(ai,'figS1_ai40')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
d=ai[ai.metric=='factual_quality'].set_index('condition').loc[['Base','Base+RAG','FT','FT+RAG']]
points(axs[0],list(range(4)),d.value.values,d.ci_low.values,d.ci_high.values,C,digits=3);axs[0].set_xticks(range(4),d.index,rotation=20);axs[0].set_ylim(0,4.3);axs[0].set_ylabel('AI answer quality (0–4)');panel(axs[0],'a','Answer quality (n=40 per condition)')
for k,(m,col) in enumerate([('citation_support',C[1]),('citation_coverage',C[3])]):
    d=ai[(ai.metric==m)&ai.condition.isin(['Base+RAG','FT+RAG'])].set_index('condition').loc[['Base+RAG','FT+RAG']]
    x=np.array([0,1])+(-.10 if k==0 else .10);y=d.value.values*100
    axs[1].errorbar(x,y,yerr=[y-d.ci_low.values*100,d.ci_high.values*100-y],fmt='o',capsize=3,color=col,label='Link support' if k==0 else 'Statement coverage',ms=4)
axs[1].set_xticks([0,1],['Base+RAG','FT+RAG']);axs[1].set_xlim(-.4,1.4);axs[1].set_ylim(0,105);axs[1].set_ylabel('AI-judged proportion (%)');axs[1].legend(loc='lower left',fontsize=7);panel(axs[1],'b','Evidence associations (80 answers)')
fig.suptitle('Official test AI subset: separate 40-question evaluation',fontsize=10);save(fig,'figS1_ai40')
print('Rendered five figures: SVG / PDF / 600dpi PNG; source CSVs retained.')
