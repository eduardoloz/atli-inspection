# Response to Internal Review (senior-author / edge-computing perspective)

We thank the reviewer for an unusually thorough reading. Every finding was either applied or explicitly rebutted below. A governing constraint on this revision: **no number may be invented** — where a demanded change required data that does not exist in the archived run record (measured on-device fps, power draw, per-condition training hours, SOTA FLOPs), we did not fabricate it; we scoped the claim to what was measured and moved the rest into explicit limitations and future work.

---

## Must-fix findings

### 1. "Validation set doubles as the test set [6]" (Sec. V-A)

**Applied (option b).** The reviewer is right that the published record contradicts the sentence as cited: APET Table IV states a 70/15/15 train/validation/testing split and Section III-C reports testing-set results. Our internal reproduction notes (`results/cv_proper_results.md`) do record a val=test characterization from the server reproduction of the original training artifacts, but that is a reproduction finding, not a published fact, and it has not been discussed with the original authors. It has **no place in print in its previous form**. The paragraph now rests the "comparison requires care" caveat solely on the two verifiable differences — single split vs. five disjoint val≠test folds, and the 732-image ATLI vs. the 1,343-image merged pool — and cites [6] only for what [6] actually says. We will raise the reproduction observation with the APET authors separately; if it is confirmed and cleared, it can return as a documented reproduction finding with artifacts.

### 2. Projected Jetson fps presented as deployment result (Abstract, Contribution 5, Sec. V-G, Table IV, Fig. 4)

**Applied to the maximum extent possible without fabricating measurements.** The reviewer is correct that measurement is the right fix, and correct that the compiled Maxwell FP16 engine makes it a short task — but no measured latency/fps/power numbers exist in the run record today, and this revision cannot invent them. Changes made instead:

- "Clears a coverage-derived throughput requirement severalfold" **removed from the abstract and from Contribution 5**. The contribution is retitled "The result exports to the edge" and explicitly scoped to the export and its validated accuracy, with on-device measurement stated as outstanding.
- Sec. V-G now states plainly that **every fps value is a projection** (GFLOPs-scaled from a third-party YOLOv8n anchor, across a different architecture and resolution, ±20–30%), that end-to-end latency, peak memory, and power draw **have not been measured**, and that "no throughput number here should be read as verified."
- Table IV caption: "All fps values are *projections* … none is measured." Fig. 4 caption likewise says "projected (unmeasured)".
- On the variant question: the run record does not name the unit; the text no longer implies possession-based knowledge of the envelope and instead states the engine + CUDA context sit "inside the 2 GB memory budget we set to cover the smallest Nano variant" — a design-target statement, not a measurement claim.
- Limitation (ii) expanded accordingly; measured on-device benchmarking is the first-listed item of future work.

We agree measured numbers should be in the camera-ready and will run trtexec/timed-loop measurements on the device; they are simply not in this draft because they do not exist yet.

### 3. No training-cost accounting (contrast APET Table V hours column)

**Partially applied; remainder rebutted as unavailable data.** The criticism is fair — the lineage's original table reports hours per condition, and our recipe multiplies training compute. However, per-condition wall-clock hours were **not systematically logged** across the 250+ run campaign, and reconstructing GPU-hours post-hoc would be estimation, which the ground-truth rule forbids. What the revision does:

- Sec. IV-B now states the champ:base cost relationship **structurally**: 1280 px is 4× the per-image pixel work of 640 px and oversampling lengthens each epoch, so a champ run costs a structural multiple (≥4×) of a base run's GPU-time — an arithmetic fact, not an estimate.
- The same paragraph names the reporting gap explicitly ("a reporting gap relative to [6], which tabulates hours per configuration") and Limitation (iii) owns it.
- We did **not** add invented GPU-hour columns to Table I. Logging per-condition hours is added to future work; hardware (8× Quadro RTX 6000) was already stated in Sec. IV-A.

### 4. "+1.7 under stricter evaluation" over APET (Sec. V-A)

**Applied.** We cannot re-run on the original 732-image ATLI under 5-fold CV within this revision, so we took the reviewer's second option in full: the paragraph now opens "Comparison with the originating study is context rather than a controlled comparison," names the 84%-larger pool as a confound that alone can move mAP by several points, deletes the "+1.7 under stricter evaluation" framing, and states explicitly: "We draw no cross-study improvement claim from these figures; the improvement claims of this paper are the within-protocol base→champ deltas of Table I." The recall comparison is fixed exactly as demanded: APET's 0.538 is retained as protocol context only, and the matched claim is now the identical-fold 0.695→0.764 (base→champ, v11n). The honest APET Table V numbers (78.9/77.5/76.8) are retained as context per the earlier review round.

### 5. Tuned-YOLO vs. stock-SOTA (Sec. IV-D, V-B, Table II, Contribution 3)

**Applied via rewording; the tuned-heavyweight control is future work.** Applying the resolution+oversampling levers to DINO/RTMDet on the same folds is the right control and has not been run; we will not synthesize its result. Changes:

- Contribution 3 retitled "Nano models **match** a 47M-parameter transformer" and now ends: "The heavyweights are trained off-the-shelf, so the claim is scoped accordingly: the tuned nano recipe matches *stock* heavyweight detectors on this benchmark."
- Sec. V-B opens with the scoping paragraph the reviewer asked for: default MMDetection configurations, standard epoch budgets, none of the data levers, "not the heavyweights' ceiling," levers acknowledged as model-agnostic, control flagged as unrun future work (also Limitation iv).
- Table II gains an **Epochs column** (36 / 150 / 40 vs. 150+100), values traceable to `sota_cv_results.md`. Input resolutions for the MMDetection defaults are not recorded in the ground-truth files, so the table says "default configurations" rather than asserting unrecorded numbers.
- "Removing the accuracy argument for heavyweight detectors" is gone; the claim is now "stock heavyweight detectors offer no accuracy premium over the tuned nano recipe."

---

## Should-fix findings

### 6. "Beats DINO on the rare class"

**Applied.** Abstract, Contribution 3, Sec. V-B, and the Conclusion now say **comparable, not a win**. Sec. V-B reports the paired per-fold differences computed from the archived per-fold values (−0.031, +0.012, +0.005, +0.026, +0.105; mean +0.023 ± 0.050) and states the margin "hinges on a single fold — we read this as comparable, not a win." The parameter-efficiency claim now rests on the mAP tie alone, exactly as the reviewer suggests. The fold-consistent claim that *is* defensible is added: the RTMDet mAP deficit is positive on all five folds.

### 7. Table III protocol mixing

**Applied.** The two offending rows are out of the table: pretrain-then-finetune (pooled run-log group shared with two architecture ablations, not disaggregable — reported in text as a qualitative negative with the pooled range labeled as such) and srcTL (different protocol — reported in text with its protocol named). The remaining table rows share one protocol and all carry n (8, 3, 4, 1, 11); the caption states that ladder rows report the range across doses and why the two conditions moved to text. Re-running enough clean seeds to tabulate PF/srcTL properly is left for the camera-ready.

### 8. Fig. 2 single-run confusion matrices vs. the paper's own protocol argument

**Applied via explicit demotion; aggregation is outstanding.** Seed-aggregated confusion matrices do not exist in the archived record (only the two rendered run artifacts do), so per the no-fabrication rule we took the demotion option: the caption now opens "Illustrative error structure," cites the protocol section for why cell values carry single-draw noise, and says they are read qualitatively. The V-D text now (a) leads with the caveat, (b) distinguishes the one *structural* observation that holds in both matrices (insulator defects are missed, not mislabeled as normal) from cell-level numerics, (c) ties it to the seed-averaged recall data in Table I, and (d) downgrades the detect-then-classify dismissal to an explicit *conjecture*. The Discussion's dependent claim is hedged the same way ("single-run, read qualitatively … appears to be"). Seed/fold-aggregated matrices are named as outstanding analysis.

### 9. Deployment gate adjudicated on 12 DD test instances

**Applied via the gate restatement option.** A 5-fold CV validation of the 768 px recipe has not been run (and is now explicitly named in the text as not yet run), so we took the reviewer's stated alternative: the deployment gate is now **mAP-only**, and the DD floor is labeled **provisional** in Sec. V-G, in the Table IV caption, in the Fig. 4 caption, and in the regenerated figure itself ("DD floor 0.59 (provisional)"). Limitation (i) now says outright that the deployment recipe's DD floor is unverified until the test pool grows.

### 10. Parameters-only efficiency comparison (Table II, Fig. 1)

**Partially applied; numeric demand rebutted as unavailable.** GFLOPs and latency for the MMDetection baselines were never collected in the campaign, and GFLOPs "at the evaluated input resolution" cannot be stated when the evaluated resolutions themselves are not in the run record — inventing either would violate the ground-truth rule. What the revision does: Table II/Fig. 1 captions and Sec. V-B now state explicitly that parameters are the only efficiency axis collected and are an imperfect proxy across architecture families (Limitation iii); the honest concession the reviewer asked for is made in text — RTMDet-Tiny at 4.8M "is itself a credible edge candidate, though we did not export it." Compute/latency profiling of the SOTA baselines joins the measurement work in future tasks.

### 11. Missing Sensors 2026 journal citation

**Applied.** The archival journal version (Vazquez, Zhai, Yang, "Edge-Friendly UAV Wildfire Smoke and Flame Detection Using Transfer Learning-Enhanced Lightweight Deep Learning Models," *Sensors* 26(10):3197, 2026, DOI 10.3390/s26103197) is added to refs.bib and is now the primary citation for both load-bearing uses: the FASDD-vs-COCO transfer result (Sec. II-C, with SmartNets 2024 retained as the conference version) and the 31.9-fps Raspberry Pi 5 pipeline (Sec. II-E), plus the Discussion and Sec. V-G mentions. The M.S. thesis citation is retained only in the repeated-split-protocol survey list.

---

## Questions

### 12. Coverage-derived 5–10 Hz requirement

**Applied (hybrid of both demanded options).** The full derivation parameters (speed, altitude, FOV, sighting geometry) live in a deployment memo that is not part of the paper's ground-truth record, so we could not reproduce the derivation without importing unverified numbers. Sec. V-G now: (a) reinstates the project's original 30 fps target as the primary yardstick and states plainly that projected ~18 fps at 768 px **falls short of it**; (b) presents the 5–10 Hz figure as an argument from "a separate coverage memo" whose flight-profile parameters "are not reproduced here"; and (c) acknowledges the regimes the reviewer names — faster flight or video-based tracking restores the higher requirement. The rescoped requirement no longer appears in the abstract or contributions.

### 13. Two per-context "best" configurations (CPLID restore)

**Applied.** Sec. III-C now declares the primary configurations up front (merged-pool headline = champ v11n under CV; clean benchmark and deployment = decontaminated-train-only). Sec. V-F reframes the CPLID restore as "a labeled ablation outside the declared primary configurations," adds the requested mechanism hypothesis (resolution-dependent interaction: at 1280 px the added Self-Exploded instances outweigh the CPLID domain shift; at 768 px they do not — testing it is future work), and states that because adopting the restore only where it helps would be post-hoc selection, **no headline or deployment number uses the restored variant**. The abstract and conclusion quote only primary-configuration numbers (they already did; this is now guaranteed by the declaration).

### 14. "ATLI-style" protocol deviations

**Applied.** Sec. IV-A now enumerates the three deviations from [6] in one sentence: (1) the published COCO checkpoint replaces stage-one source training (two stages rather than three); (2) the frozen-backbone feature-extraction arm is dropped from the main protocol, retained only as an ablation where it loses (Sec. V-D), consistent with [6]'s own unfrozen-preferred finding; (3) batch 32 vs. 8 per GPU. The sentence closes by confirming learning rates, image size, and epoch budgets otherwise match.

---

## Build status

`latexmk -pdf` exit 0; zero error lines in main.log; no undefined citations or references; no new overfull hbox above 4pt (two pre-existing table boxes at 1.6pt/3.0pt remain). Figures regenerated from `figures_src/gen_paper_figures.py` (only change: DD floor relabeled "provisional" in Fig. 4). Final PDF: 10 pages, body ending on page 9 with references spilling onto page 10 — the review's five must-fix scoping paragraphs added roughly two-thirds of a column; we tightened elsewhere and judge the added caveats worth the spill, but can compress further if the venue requires a hard 9.
