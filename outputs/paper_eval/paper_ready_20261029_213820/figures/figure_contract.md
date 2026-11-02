# Figure contract (before rendering)

Backend: Python/matplotlib only. Quantitative grids; white background; method colors and direct labels. Export SVG with editable text, PDF with TrueType fonts, PNG at600dpi. Double-column width183mm; no inferred error bars. All panels link to source CSV, not rounded manuscript numbers.

1. Fig1: On the same227 official questions, retrieval changes option accuracy/lexical overlap and FT changes format. Four separate panels: parsed accuracy, strict correct-letter rate, format compliance, citation-stripped F1. Means and existing verified181-group2000-resample percentile95%CIs. Do not interpret F1 or format as facts.
2. Fig2: Development stage changes depend on RAG. F1 and actual output tokens are separate panels, each split by RAG.101 questions/91groups; F1 intervals verified10000-resample percentile. Actual lengths descriptive. Previously selected development sample, exploratory; no independent replication claim.
3. Fig3: F1 direction does not uniquely determine AI content judgment.3x5 count heatmap, denominator101 pairs. Single AI judge; v1/v2 RAG160 only. No CI.
4. Fig4: Refusal and point coverage can deteriorate after continuation in this audit. Paired binary erroneous-refusal transitions101 and97 individual paired coverage lines, with equal-question means. Also point-weighted coverage totals. No inferential interval, subjective AI-defined points,4NA excluded only from coverage.
5. FigS1: Official AI40 is distinct from devAI101. Quality/completeness/faithfulness0–4 and citation support/coverage are separated. Existing40-groupCIs exclude judge uncertainty.

Review risks: paired sampling, source dependence, no multiplicity correction on dev, undefined0/0 omitted, NA not zero, no scientist/human-rating implication, composite v2 worse labels are not counts of factual errors. Grayscale distinguishes method positions; heatmap prints exact counts. Figure captions will carry all denominators and protocols.
