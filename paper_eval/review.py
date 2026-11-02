from __future__ import annotations

import html
import json
import random
import re
import uuid
from pathlib import Path

from .common import digest_file, read_jsonl, write_json

TEMPLATE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>海洋问答专家盲评</title>
<style>body{font:16px/1.7 system-ui;max-width:1000px;margin:30px auto;padding:0 22px;color:#183040;background:#f6f8fa}header,article{background:white;padding:22px;margin:20px 0;border-radius:10px;border:1px solid #d5dfe5}button{cursor:pointer;padding:8px 16px}select,input,textarea{font:inherit;padding:5px}textarea{width:95%}pre{white-space:pre-wrap;font:inherit;background:#eef4f7;padding:15px}small{color:#526573}.point{padding:8px 0}nav{position:sticky;top:0;background:#f6f8fa;padding:10px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}label{margin-right:12px}details{margin:10px 0}select{max-width:100%}#status{font-weight:600}</style>
<header><h1>海洋问答专家盲评</h1><p id="intro"></p><p>独立评分，评分期间请勿查看模型身份或另一位评审的结果。参考要点是待复核草稿，若发现错误请记在备注中，由协调人修订后统一重评相关题。无数据时保持空白。浏览器在本机保存进度；导出JSON后交给协调人。请勿清空浏览器存储。</p><details><summary>评分标准</summary><p>事实正确性：4=无明显事实错误；3=主体正确，少量次要不严谨；2=正确与错误内容混合或有重要错误；1=大部分内容错误；0=完全错误或无实质回答。遗漏在完整性中另计。拒答可减少错误，但不能自动获得完整性分。</p><p>要点：1=明确正确覆盖；0.5=部分覆盖；0=未覆盖或错误。科普表达：4=适龄清晰且不牺牲科学准确性，0=明显难懂或误导；仅科普题填写。</p><p>证据忠实度：4=可核实陈述均受给定资料支持；3=少量次要外推；2=重要陈述缺支持；1=多数缺支持；0=与资料冲突或无支持。资料缺失且回答仅明确拒答时可选“不适用”，并说明。引用计数以“陈述—引用”关联为单位，同一编号支持两个陈述应分别核查。真实存在的编号也可能不支持陈述。</p></details></header>
<nav><button id="prev">上一项</button><button id="next">下一项</button><input id="jump" type="number" min="1" style="width:80px"><button id="go">跳转</button><span id="status"></span><button id="export">导出评分JSON</button></nav>
<article id="card"></article><script>
const PACKET=__PACKET__;
const storageKey='ocean-blind-'+PACKET.packet_id;
let saved=JSON.parse(localStorage.getItem(storageKey)||'{}'), index=0;
document.getElementById('intro').textContent=PACKET.kind==='quality'?'第一阶段：事实、要点及表达。引用标记已统一移除，以减少模型身份线索。':'第二阶段：核查原始引用及实际检索资料。此阶段仅评价具备检索配置的匿名答案。';
function el(tag,text,parent){let x=document.createElement(tag);if(text!==null)x.textContent=text;if(parent)parent.append(x);return x}
function persist(){localStorage.setItem(storageKey,JSON.stringify(saved));status()}
function isComplete(item,r){if(!r)return false;if(PACKET.kind==='quality')return r.factual!==undefined&&r.points?.length===item.key_points.length&&r.points.every(x=>x!==null)&& (item.mode!=='popular'||r.age!==undefined);return r.faithfulness!==undefined&&['citation_links','supported_links','claims_requiring_citation','claims_with_supported_citation'].every(k=>r[k]!==undefined)&& (r.faithfulness!==null||!!r.notes?.trim())}
function status(){let done=PACKET.items.filter(x=>isComplete(x,saved[x.token])).length;document.getElementById('status').textContent=`${index+1}/${PACKET.items.length} · 已完整填写 ${done}`;document.getElementById('jump').max=PACKET.items.length;document.getElementById('jump').value=index+1}
function score(parent,label,key,options){let wrap=el('div',null,parent);el('label',label+' ',wrap);let s=el('select',null,wrap);for(let [v,t]of [['','请选择'],...options]){let o=el('option',t,s);o.value=v}let r=saved[PACKET.items[index].token];if(r[key]!==undefined)s.value=r[key]===null?'NA':String(r[key]);s.onchange=()=>{if(s.value==='')delete r[key];else r[key]=s.value==='NA'?null:Number(s.value);persist()}}
function number(parent,label,key){let wrap=el('div',null,parent);el('label',label+' ',wrap);let n=el('input',null,wrap);n.type='number';n.min='0';n.step='1';let r=saved[PACKET.items[index].token];n.value=r[key]??'';n.oninput=()=>{if(n.value==='')delete r[key];else if(n.validity.valid)r[key]=Number(n.value);persist()}}
function render(){const item=PACKET.items[index], r=saved[item.token]??={points:Array(item.key_points.length).fill(null)};let card=document.getElementById('card');card.replaceChildren();el('h2','题目 '+item.question,card);el('small',item.mode==='popular'?'科普题':'专业题',card);el('h3','待评答案',card);el('pre',item.answer,card);let details=el('details',null,card);el('summary','参考要点与教材摘录（草稿）',details);item.key_points.forEach((p,i)=>el('div',(i+1)+'. '+p,details));for(let e of item.evidence)el('pre','教材 PDF p.'+e.pdf_page+'\n'+e.text,details);
if(PACKET.kind==='quality'){score(card,'事实正确性','factual',[0,1,2,3,4].map(x=>[String(x),String(x)]));el('h3','逐项完整性',card);item.key_points.forEach((p,i)=>{let d=el('div',null,card);d.className='point';el('span',p+' ',d);let s=el('select',null,d);for(let [v,t]of [['','请选择'],['0','0 未正确覆盖'],['0.5','0.5 部分覆盖'],['1','1 完整覆盖']]){let o=el('option',t,s);o.value=v}s.value=r.points[i]===null?'':String(r.points[i]);s.onchange=()=>{r.points[i]=s.value===''?null:Number(s.value);persist()}});if(item.mode==='popular')score(card,'适龄表达','age',[0,1,2,3,4].map(x=>[String(x),String(x)]));}
else{el('h3','本次实际提供的检索资料',card);if(!item.context.length)el('p','本次没有超过检索阈值的资料。',card);item.context.forEach(c=>el('pre','['+c.id+'] 教材 PDF p.'+c.pdf_page+'\n'+c.text,card));score(card,'证据忠实度','faithfulness',[...[0,1,2,3,4].map(x=>[String(x),String(x)]),['NA','不适用（备注说明）']]);number(card,'答案中的陈述—引用关联总数','citation_links');number(card,'其中被所引资料支持的关联数','supported_links');number(card,'需要引用的可核实陈述总数','claims_requiring_citation');number(card,'其中具有有效且支持该陈述的引用数','claims_with_supported_citation');}
el('h3','备注／发现的错误／参考答案修订建议',card);let note=el('textarea',null,card);note.rows=3;note.value=r.notes||'';note.oninput=()=>{r.notes=note.value;persist()};status()}
document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);render()};document.getElementById('next').onclick=()=>{index=Math.min(PACKET.items.length-1,index+1);render()};document.getElementById('go').onclick=()=>{index=Math.max(0,Math.min(PACKET.items.length-1,Number(document.getElementById('jump').value)-1));render()};
document.getElementById('export').onclick=()=>{let data={packet_id:PACKET.packet_id,kind:PACKET.kind,exported_at:new Date().toISOString(),ratings:PACKET.items.filter(x=>saved[x.token]).map(x=>({token:x.token,...saved[x.token],complete:isComplete(x,saved[x.token])}))};let blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ratings_'+PACKET.packet_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script></html>'''


def make_packets(run: Path) -> None:
    answers = read_jsonl(run / "answers.jsonl")
    questions = {r["id"]: r for r in read_jsonl(run / "selected_questions.jsonl")}
    target, private = run / "blind_review", run / "private"
    private.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    mapping_path = private / "blind_mapping.json"
    if mapping_path.exists():
        raise ValueError("Existing packets may have ratings; use them, do not regenerate anonymous IDs")
    mapping = {"answers_sha256": digest_file(run / "answers.jsonl"), "packets": {}}
    for reviewer in [1, 2]:
        for kind in ["quality", "evidence"]:
            packet_id = uuid.uuid4().hex
            relevant = [r for r in answers if kind == "quality" or r["condition"].endswith("rag")]
            random.Random(20261016 + reviewer * 100 + (kind == "evidence")).shuffle(relevant)
            items, links = [], {}
            for result in relevant:
                token = uuid.uuid4().hex
                q = questions[result["id"]]
                answer = result["answer"]
                if kind == "quality":
                    answer = re.sub(r"\[WB_p\d+_c\d+\]", "", answer)
                items.append({"token": token, "question": q["question"], "mode": q["mode"], "answer": answer,
                              "key_points": q["key_points"], "evidence": [{"pdf_page": e["pdf_page"], "text": e["text"]} for e in q["evidence"]],
                              "context": result["retrieved_context"] if kind == "evidence" else []})
                links[token] = {"id": result["id"], "condition": result["condition"], "source_group": result["source_group"]}
            packet = {"packet_id": packet_id, "kind": kind, "items": items}
            text = TEMPLATE.replace("__PACKET__", json.dumps(packet, ensure_ascii=False).replace("</", "<\\/"))
            folder = target / f"reviewer_{reviewer}"
            folder.mkdir(exist_ok=True)
            (folder / f"{kind}.html").write_text(text, encoding="utf-8")
            mapping["packets"][packet_id] = {"reviewer": f"reviewer_{reviewer}", "kind": kind, "items": links}
    write_json(mapping_path, mapping)
    (target / "使用说明.txt").write_text("两位专家分别领取 reviewer_1 / reviewer_2 文件夹，独立填写后导出JSON。先完成quality，再完成evidence。不要向专家分享private文件夹或原始结果（其中有模型身份）。协调人保存private映射表并合并结果。页面允许中途导出；未填项不会被算作0分。\n\n教材：Introduction to Oceanography by Paul Webb, CC BY 4.0, except where otherwise noted. https://rwu.pressbooks.pub/webboceanography/\n盲法局限：文本内容、长度和引用风格可能泄露条件。两阶段评审减少线索，不能保证专家无法推测。\n", encoding="utf-8")
    print(f"Created two independent expert packets in {target}; no expert scores fabricated.")
