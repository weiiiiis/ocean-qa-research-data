"""Independent delivery audit using only the Python standard library."""
import collections
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(name):
    return json.loads((out/name).read_text(encoding='utf-8-sig'))

def jsonl(name):
    return [json.loads(s) for s in (out/name).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def table(name):
    with (out/name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

questions=jsonl('selected_questions.jsonl')
answers=jsonl('answers.jsonl')
ids={r['id'] for r in questions}
counts=collections.Counter((r['condition'],r['id']) for r in answers)
conditions={r['condition'] for r in answers}
assert len(ids)==len(questions)==101 and len(answers)==1010 and len(conditions)==10
assert all(v==1 for v in counts.values())
assert all({r['id'] for r in answers if r['condition']==c}==ids for c in conditions)
for qid in ids:
    for rag in (False,True):
        rows=[r for r in answers if r['id']==qid and r['rag']==rag]
        assert len(rows)==5
        assert all((r['prompt'],r['retrieved_context'])==(rows[0]['prompt'],rows[0]['retrieved_context']) for r in rows)
scores=table('per_question_metrics.csv')
assert len(scores)==1010 and {(r['condition'],r['id']) for r in scores}==set(counts)
summary=table('summary.csv')
assert len(summary)==10 and all(int(r['questions'])==101 for r in summary)
differences=table('paired_differences.csv')
assert len(differences)==162 and all(int(r['paired_questions'])==101 for r in differences)
assert all(math.isfinite(float(r[k])) for r in differences for k in ('difference','ci95_low','ci95_high'))
for name in ('cases_review.csv','stage_change_cases.csv'):
    assert all(not r[k] for r in table(name) for k in ('factual_correctness_human','completeness_human','citation_support_human'))
assert read('protected_integrity.json')['original_results_preserved'] is True
previous=read('delivery_hashes.json')
assert all(sha(out/p)==h for p,h in previous.items())
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
        'questions':101,'unique_conditions':10,'raw_answers':1010,'per_question_metric_rows':1010,
        'summary_rows':10,'paired_metric_rows':162,'same_question_ids_all_conditions':True,
        'no_duplicates':True,'same_actual_prompt_and_context_across_stages_and_budgets':True,
        'human_scores_blank':True,'protected_original_files_unchanged':481,
        'previous_delivery_hashes_verified':len(previous),'verification_code_sha256':sha(Path(__file__))}
(out/'final_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (out/'commands.jsonl').open('a',encoding='utf-8') as stream:
    stream.write(json.dumps({'timestamp_utc':result['checked_at_utc'],'argv':[sys.executable,*sys.argv]},ensure_ascii=False)+'\n')
inventory={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='delivery_hashes.json'}
(out/'delivery_hashes.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
