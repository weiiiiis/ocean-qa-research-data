# English marine QA LoRA — development-selected candidate

Selected locally on 2026-10-17. The adapter is `epoch_1/` in this directory.

## Intended experiment

English marine-science source-grounded QA using the local Qwen3-4B-Instruct-2507 base, four-bit NF4 inference, and the frozen BM25 source-only retrieval corpus. This artifact is an experimental adapter. It has not received independent expert quality ratings.

## Reproducible ancestry

1. Fresh r=8, alpha=16, dropout=0.05 LoRA on `q_proj` and `v_proj`, initialized on the local original base. Train on 240 English examples (48 knowledge families) for one epoch, LR=5e-5. This is `../ocean_qwen3_english_grounded_v1/epoch_1`; the separately evaluated epoch_2 is not an ancestor of this final candidate.
2. Continue that epoch_1 on the same 240 examples with user-message endings aligned to the evaluation prompts, LR=2e-5, one epoch. Closed-book examples are repeated three times, giving 336 draws from 240 distinct examples. This does not add knowledge families.

The 60 validation examples cover 12 other families/sections from the same Webb textbook. The model has 2,949,120 trainable LoRA parameters. Training uses assistant-only loss, batch size 1, gradient accumulation 4, NF4 double quantization, FP16 compute and gradient checkpointing. Maximum allowed full-example length is 1024; actual v3 maximum is 528 and no example is truncated.

Source facts and targets are AI-authored from traceable English textbook passages and have source-anchor checks; expert approval is pending. Pira QA and supporting passages are excluded from SFT. See `training_config.json`, `training_script.py`, `train_data.jsonl` and `validation_data.jsonl` for exact inputs and hashes.

## Development results

The 101-question external English development subset is separated from official test by the known EID and normalized supporting-text checks. Its question-answer pairs are not used for training.

| Metric | No RAG | BM25 RAG |
|---|---:|---:|
| MC leading-option accuracy | 85.1% | 91.1% |
| MC strict single-letter accuracy | 84.2% | 91.1% |
| Open-QA original-reference Token F1, citations removed | 13.4% | 34.6% |
| Answers containing citation-like markers | 7/101 | 99/101 |
| Marker IDs resolved against actual supplied evidence | 0/7 | 113/113 |
| Output truncations | 0/101 | 0/101 |

F1 is lexical overlap, not factual correctness. Resolved IDs do not demonstrate claim support. Main and alternate answers are not pooled into additional independent questions. Full raw answers, prompts and resource measurements are in the two development run directories under `outputs/paper_eval/`.

## Known limitations before official testing

- Seven no-RAG development answers still contain unsupported citation-like markers. Raw outputs are retained and this is not described as a complete hallucination fix.
- Scientific correctness, essential completeness and citation entailment await actual independent human review.
- Training material contains only 48 factual families from one textbook; 240 format examples are not 240 independent knowledge points.
- Unknown base pretraining and the historical adapter chain cannot be certified contamination-free. New-source audits found no exact/normalized/0.82 semantic-threshold flags against the selected Pira development/test stems; this is screening, not proof.
- Retrieval uses 688 released supporting texts in 1507 windows, a restricted corpus. Results do not establish performance on the whole ocean-science literature, Chinese science education, or student learning outcomes.
- Natural-answer latency depends on answer length. No inference-speed improvement is claimed without equal-work measurements.

## Adapter identity

`epoch_1/adapter_model.safetensors` SHA256:

`8227de213d3467c63803996ea8dc8095549aed735b4709435521f1c62ef7a9f3`

The prior application adapter and the earlier trial checkpoints are retained separately for audit and comparison.
