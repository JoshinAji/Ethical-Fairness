1) Create a Manual Subset (30 min)
Goal: a representative 50–100 image slice that stresses ambiguity.
Sampling heuristic (approximate)
* 50% “ambiguous/uncertain” (ambiguous_mixed==1 or unknown_uncertain==1)
* 25% “edge skin tones” (bins 1–2 and 6–7)
* 25% “clear cases” (to sanity-check precision)
Script: /src/make_manual_subset.py
* Input: labels_v3.csv
* Output: v3_manual_subset.csv with columns: image_id, rel_path, existing_v3_fields… (copy read-only)
Command
python src/make_manual_subset.py --n 80 --seed 42

2) Manual Annotation Protocol (2–5 hrs)
Goal: ground-truth labels from a human pass.
Tool: Google Sheets or Excel. Freeze header row. Protect “existing_v3_*” columns.
Annotate these columns only
* race_ml_human (comma-separated; allow 1–3 labels)
* ambiguous_mixed_human (0/1)
* unknown_uncertain_human (0/1)
* skin_tone_bin_human (1–7)
* conf_race_human, conf_gender_human, conf_skin_human (0.0–1.0)
* cultural_markers_human (comma list; pick from your 8 types)
* annotation_notes (free text)
Micro-guide (print in the sheet)
* Multi-label if more than one race signals are equally plausible.
* Use unknown_uncertain_human=1 if you genuinely can’t decide.
* Confidence = your own certainty (not model’s).
* Skin tone: pick best bin; if between two, pick the closer.
Export to CSV ? data/v3_manual_annotated.csv.

3) (Optional but strong) Second Annotator for IAA (1–2 hrs)
If you can get a friend/peer to annotate the same subset:
* Save as v3_manual_annotated_r2.csv.
* You’ll compute inter-annotator agreement (IAA) to strengthen validity.

4) Compare Human vs Heuristic (45–60 min)
Goal: produce agreement metrics and a crisp validation report.
Script: /src/compare_manual_vs_heuristic.py
Input: labels_v3.csv, v3_manual_annotated.csv
Output:
* data/v3_validation_report.csv (per-image comparisons)
* reports/tables/v3_validation_summary.csv (aggregates)
* reports/figures/v3_validation_plots.png
Metrics (use these exact ones)
* Race multi-label:
o Jaccard Index per image (set overlap) + macro average
o Hamming loss (multi-label)
o Exact set match rate (strict)
* Ambiguous & Uncertain flags: Accuracy, Precision/Recall/F1
* Skin tone: MAE (absolute bin difference), ±1-bin accuracy
* Confidence fields: MAE; Spearman ? between human and heuristic
* Cultural markers: Jaccard Index per image; macro average
* Aggregate: macro Jaccard for (race + markers), mean of MAEs, etc.
Optional IAA (if Step 3 done)
* Cohen’s ? for the binary flags
* Krippendorff’s ? (nominal for races; ordinal for skin tone)
* Spearman ? for confidences
Success criteria (practical)
* Race Jaccard ≥ 0.70 (good), ≥0.80 (great)
* Ambiguous/Uncertain F1 ≥ 0.70
* Skin tone MAE ≤ 0.8 bins; ±1-bin acc ≥ 0.80
* Confidence MAE ≤ 0.20; ? ≥ 0.5

5) Calibrate / Tune Heuristics (60–90 min)
If any metric misses the target:
Script: /src/tune_v3_heuristics.py
* Sweep thresholds for ambiguity and uncertainty (e.g., 0.4–0.7)
* Re-bucket skin tone boundaries (quantiles vs fixed)
* Re-run Step 4 after tuning; keep a tuning_log.json
Deliverable: before/after table ? reports/tables/v3_tuning_summary.csv

6) Train & Evaluate Baselines (90–120 min)
Goal: show what V3 enables vs V1/V2.
Tasks
1. Model task: simple 7-way race classifier (consistent with V1/V2).
2. Datasets: use same 900 images across V1 and V3 to be fair.
3. Features:
o V1: single race label only (status quo).
o V3: keep same prediction target, but analyze results by V3 metadata (multi-label is for analysis, not the training target, to compare apples-to-apples).
Script: /src/train_eval_v1_v3.py
Outputs:
* reports/tables/cls_metrics_v1.csv, cls_metrics_v3.csv
* reports/figures/overall_acc_f1.png
Metrics
* Accuracy, Macro-F1
* Reliability: Brier score & reliability diagram (optional)

7) Bias Detection Using V3 (the showcase) (60–90 min)
Goal: produce the figures that prove “V3 reveals what V1 hides.”
Script: /src/analyze_bias_v3.py
Analyses (produce plots & tables)
* Within-race by skin tone: accuracy per bin (1–7) ? line/bar plot
* Ambiguity impact: accuracy for ambiguous_mixed=1 vs 0
* Uncertainty impact: accuracy for unknown_uncertain=1 vs 0
* Cultural markers: accuracy delta per marker type
* Confidence correlation: Spearman ? between conf_* and correctness
Outputs
* reports/figures/within_race_skin_tone_gaps.png
* reports/figures/ambiguity_uncertainty_gaps.png
* reports/figures/cultural_marker_penalties.png
* reports/figures/confidence_vs_error.png
* reports/tables/v3_bias_breakdown.csv
What to highlight
* Largest within-race gap (e.g., darkest vs lightest bin)
* Accuracy drop for ambiguous/uncertain subsets
* Any marker-specific penalties >3–5%
* Positive ? (confidence tracks correctness)

8) Selective Deferral / Risk-Aware Use (45–60 min)
Goal: show an actionable fairness win using V3 confidence.
Script: /src/selective_deferral.py
Method:
* Sort predictions by low conf_race; defer the bottom x% to “human review.”
* Plot coverage vs accuracy curve (as deferral increases, accuracy should rise).
* Report accuracy gains at 5%, 10%, 20% deferral.
Outputs
* reports/figures/coverage_accuracy_tradeoff.png
* reports/tables/deferral_results.csv
Talking point: V3 enables risk controls unavailable in V1.

9) Package Results (60–90 min)
Goal: clean, citable artifacts for your paper.
Figures (PNG)
* v1_v2_v3_comparison.png (evolution)
* v3_validation_plots.png (human vs heuristic)
* within_race_skin_tone_gaps.png
* ambiguity_uncertainty_gaps.png
* cultural_marker_penalties.png
* confidence_vs_error.png
* coverage_accuracy_tradeoff.png
Tables (CSV + optional .tex)
* v3_validation_summary.csv
* v3_tuning_summary.csv (if used)
* cls_metrics_v1.csv, cls_metrics_v3.csv
* v3_bias_breakdown.csv
* deferral_results.csv
One-pager summary (/reports/RESULTS_SUMMARY.md)
* 5 bullets (headline numbers) + figure references

10) Drop-in Write-Up (2–3 hrs)
Section 4.3 – Problems with Categorical Labels (rewritten)
* Use your three problems (forced single label, no uncertainty, no within-group granularity)
* Point forward to V3 as solution
Section 5 – Validation & Results (new)
* 5.1 Dataset Stats: V1/V2/V3 table (counts, features)
* 5.2 Manual Validation: the metrics from Step 4 (+ IAA if Step 3)
* 5.3 Bias Detection: plots from Step 7 with 1–2 sentences each
* 5.4 Selective Deferral: trade-off curve; concrete gains at 10% deferral
* 5.5 Ablation (optional): with/without skin tone; with/without uncertainty
Discussion
* How V3 reframes “labelling” ? “representation + uncertainty”
* Ethical upside: preserves ambiguity, supports human-in-the-loop
* Limitations: small manual set; heuristic bias; single domain
* Future work: multi-annotator expansion; crowdsourcing; external dataset

