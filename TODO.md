# ClinOpDet-Bench: Q1 Readiness TODO

**Target Venues:** IEEE TMI / Medical Image Analysis / Nature Scientific Reports
**Authors:** Mohammad Amin Haji Alirezaei & Pouyan Delivandani
**Hardware Constraint:** RTX 4060 Laptop GPU, 8 GB VRAM, 16 GB RAM, i7-13650HX
**Date:** 2026-08-21

---

## 🔒 Scope Boundary: Frozen vs. New

### FROZEN (unchanged, no re-execution):
- Phase 5 clean evaluation (10 checkpoints, n=5)
- Phase 6 robustness (seed-17 only, 300 images, 72 bundles)
- Phase 7 explainability (seed-17 only, 111 boxes)
- Phase 8 statistics (patient-cluster bootstrap, n=5/n=4)
- n=3 threshold selection (0.69/0.05), FROC, Pareto → archived as `*_n3_archive.*`
- All existing `results/` artifacts and their SHA-256 hashes

### NEW (to be implemented):
- Phase 9: Patient-Cluster-Aware Threshold Calibration (n=5, CPU-only)
- Phase 10: Decision Curve Analysis (CPU-only)
- Phase 11: XAI Sanity Checks (GPU, seed-17 subset)
- Phase 12: Physics-Aware Acquisition Shifts (GPU, seed-17 subset)
- Phase 13: Reporting Standards Compliance (documentation only)

---

## 🔹 0. Pre-requisites: Hypotheses, Limitations, Decision Log & Literature Updates

**Rationale:** All new contributions require formal hypothesis registration, limitation disclosure, decision logging, and literature grounding BEFORE implementation begins. This prevents post-hoc narrative construction.

### 0a. Update `docs/HYPOTHESES.md`
Add three new hypotheses:
- **H6 (Threshold Calibration):** Patient-cluster-aware cost-weighted threshold calibration will produce detector-specific operating points that differ from both the fixed 0.25 threshold and the n=3 validation-max-F1 thresholds, and will change the relative ranking of detectors on at least one clinical utility metric.
- **H7 (DCA):** Decision Curve Analysis will reveal threshold ranges where one detector dominates clinical net benefit despite similar mAP, demonstrating that algorithmic metrics alone are insufficient for deployment decisions.
- **H8 (XAI Sanity):** At least one Grad-CAM sanity check (parameter randomization or data shuffling) will produce $C_{\text{sanity}} > 0.3$ for at least one detector, indicating that standard Grad-CAM maps are unreliable as clinical reasoning evidence.

### 0b. Update `docs/LIMITATIONS.md`
Add new subsections:
- **Threshold calibration scope:** The calibration optimizes on the same validation split used for checkpoint selection; it does not perform probabilistic calibration or guarantee transportability to a new clinical population.
- **DCA prevalence assumption:** Net benefit curves assume a fixed disease prevalence; actual clinical prevalence may differ.
- **XAI sanity check scope:** Sanity checks are run on a subset (≤50 images) of the seed-17 sample; they do not constitute a full validation study.
- **Acquisition shift scope:** Physics-based shifts are calibrated approximations, not measured scanner outputs. They do not replace multi-site external validation.

### 0c. Update `docs/DECISION_LOG.md`
Append:
- **D-004:** Adopt patient-cluster-aware cost-weighted threshold calibration as the primary operating-point protocol, superseding the fixed 0.25 threshold for clinical interpretation while retaining it as a protocol-sensitivity record.
- **D-005:** Retain n=3 threshold/FROC/Pareto as frozen historical analyses; do NOT regenerate at n=5. The new Phase 9 calibration is the authoritative n=5 operating-point result.
- **D-006:** Add XAI sanity checks and physics-aware acquisition shifts as supplementary analyses scoped to seed-17, without expanding the five-seed clean comparison.

### 0d. Update `docs/LITERATURE_REVIEW.md`
Add subsections covering:
- Patient-cluster bootstrap methods in medical imaging evaluation
- Decision Curve Analysis (Vickers & Elkin, 2006; Vickers et al., 2016)
- XAI sanity checks (Adebayo et al., 2018; Arun et al., 2021)
- Physics-aware domain shift in radiology (VOI/LUT, dose variation)
- CLAIM, TRIPOD-AI, STARD-AI reporting standards

👤 **Task Assignment:**
- **Mohammad Amin:** Draft H6-H8 mathematical formulations. Write D-004/D-005/D-006 entries.
- **Pouyan:** Draft LIMITATIONS additions. Update LITERATURE_REVIEW with new citations.
- **Joint:** Cross-review for consistency with existing H1-H5 and D-001/D-002/D-003.

---

## 🔹 1. Patient-Cluster-Aware Threshold Calibration (Primary Methodological Contribution)

**Current State**: Batch 14 performed detector-specific threshold selection (0.69/0.05) by maximizing mean validation F1 over n=3 seeds. This is documented in THRESHOLD_ANALYSIS.md and used by FROC/Pareto analyses.
Remaining Gap for Q1: The existing selection (a) uses symmetric F1 without encoding clinical FN≫FP cost asymmetry, (b) was performed on n=3 seeds and not updated after seeds 271/314, (c) optimizes a point estimate without accounting for patient-cluster sampling uncertainty, and (d) produces a threshold (0.05) at which YOLO seed 271 emits zero detections. The proposed contribution upgrades this to a patient-cluster-aware, cost-weighted, CI-optimized protocol at n=5.

**Critical Constraint from `THRESHOLD_ANALYSIS.md`:** The existing n=3 threshold selection (0.69/0.05) is FROZEN and must NOT be regenerated. The new calibration is a SEPARATE, ADDITIONAL analysis at n=5.

**Critical Constraint from `YOLO_BASELINE.md`:** YOLO seed 271 has maximum test confidence = 0.0412735. It will emit zero detections at ANY threshold > 0.0412735. The calibration must handle this identically to the existing conditional IoU/Dice policy: seed 271 contributes valid zeros to AP/F1 but is excluded from threshold optimization if its maximum score falls below the candidate threshold grid minimum (0.01). Since 0.0412735 > 0.01, seed 271 DOES participate in the calibration but will always contribute zero TP/FP at thresholds > 0.0412735.

**Scientific Gap:** Standard threshold optimization ignores (a) patient-level correlation in false predictions, (b) asymmetric clinical cost structures, and (c) uncertainty in metric estimation.

**Formal Solution:**
$$
F1_\beta(\tau) = \frac{(1+\beta^2) \cdot P(\tau) \cdot R(\tau)}{\beta^2 P(\tau) + R(\tau)}, \quad \beta = \sqrt{\frac{C_{FN}}{C_{FP}}}
$$
$$
\tau^* = \arg\max_{\tau \in [0.01, 0.99]} \left\{ \text{Quantile}_{\alpha}\left( \{F1_\beta(\tau, b)\}_{b=1}^{B} \right) \right\}
$$
where $B=2000$ bootstrap resamples, $\alpha=0.05$, patient-cluster resampling.

**Seed 271 Handling Protocol:**
- For thresholds $\tau \leq 0.0412735$: seed 271 participates normally (may produce detections).
- For thresholds $\tau > 0.0412735$: seed 271 contributes $TP=0, FP=0, FN=\text{total targets}$ (valid zeros, not excluded).
- This parallels the existing Phase 8 policy: "All predeclared attempted seeds are retained, including operational failures."

**Implementation Protocol:**
1. Create `src/stats/threshold_calibration.py` (~250 LOC).
2. Load ALL 10 frozen prediction bundles from Phase 5 (not just n=3).
3. Aggregate predictions per NIH patient ID before computing $P, R, F1_\beta$.
4. Run nested bootstrap: outer loop over $\tau \in [0.01, 0.99]$ (99 points), inner loop over 2000 patient-cluster resamples.
5. Output $\tau^*$ per detector, CI bounds, and sensitivity curves for $\beta \in \{1, 3, 5, 10\}$.
6. Add `--mode recalibrate` flag to `run_statistics.py` (PARALLEL to existing `--mode run`, not replacing it).
7. Create `configs/threshold_calibration.yaml` with strict Pydantic validation.
8. Write results to `results/tables/threshold_calibration_summary.csv` and `results/logs/phase9_threshold_calibration/summary.json`.

**Computational Budget:** CPU-only. Reads frozen `.json.gz` bundles. No GPU required. Estimated runtime: ~15-30 minutes on i7-13650HX.

**Expected Q1 Impact:** Transforms heuristic threshold selection into a statistically principled, clinically aligned operating-point protocol.

👤 **Task Assignment:**
- **Mohammad Amin:** Core implementation of `threshold_calibration.py` (nested bootstrap, patient aggregation, $\tau^*$ optimization, seed-271 handling). Hook into `run_statistics.py`. Create `configs/threshold_calibration.yaml`. Write Methodology Section (mathematical derivation & bootstrap protocol).
- **Pouyan:** Implement clinical cost mapping ($C_{FN}/C_{FP}$ justification from radiology guidelines, cite USPSTF/ACR). Generate sensitivity analysis plots for $\beta \in \{1,3,5,10\}$. Update `docs/THRESHOLD_ANALYSIS.md` with new n=5 calibration section (clearly separated from the frozen n=3 section).
- **Joint:** Cross-validate $\tau^*$ stability across seeds. Verify seed-271 handling matches Phase 8 policy. Co-author Results Section.

---

## 🔹 2. Clinical Utility & Decision Curve Analysis (DCA)

**Current Limitation:** Evaluation relies on algorithmic metrics (mAP, F1, Precision, Recall) that lack direct clinical interpretation.

**Scientific Gap:** Q1 medical journals require demonstration of *clinical net benefit* rather than pure predictive accuracy.

**Formal Solution:**
$$
\text{Net Benefit}(\tau) = \frac{TP}{N} - \frac{FP}{N} \cdot \frac{p_t}{1-p_t}, \quad p_t = \frac{\tau}{1-\tau}
$$

**Critical Constraint:** DCA uses the SAME frozen prediction bundles as Phase 9. No retraining or new inference required. Patient-cluster bootstrap CIs must use the same 323 patient groups as Phase 8.

**Implementation Protocol:**
1. Create `src/clinical/decision_curve.py` (~150 LOC).
2. Reuse frozen prediction bundles from Phase 5.
3. Compute $TP, FP$ across $\tau \in [0.01, 0.99]$ for each seed.
4. Calculate NB curves with 95% bootstrap CIs (patient-cluster resampling, 2000 draws).
5. Plot NB curves for both detectors against "treat-all" and "treat-none" baselines.
6. Create `configs/dca.yaml`.
7. Write to `results/figures/dca_curves.png` and `results/tables/dca_summary.csv`.

**Computational Budget:** CPU-only. Estimated runtime: ~5-10 minutes.

👤 **Task Assignment:**
- **Mohammad Amin:** Integrate DCA computation into the evaluation pipeline. Ensure patient-cluster bootstrapping aligns with Phase 8 methodology. Write Discussion Section (clinical utility translation).
- **Pouyan:** Implement `src/clinical/decision_curve.py`. Generate publication-ready DCA plots with 95% CI ribbons. Draft supplementary clinical prevalence assumptions (cite RSNA cohort prevalence: 6012/26684 = 22.5% opacity).
- **Joint:** Validate NB curves against reference DCA implementations (Vickers' `dcurves` R package or `pydca`). Co-author Results Section.

---

## 🔹 3. Statistical Rigor & Conditional Metric Handling

**Current Limitation:** Conditional IoU/Dice are excluded when $TP=0$ (YOLO seed 271). The handling is methodologically sound but lacks formal reporting structure and sensitivity analysis.

**Critical Constraint from `STATISTICAL_ANALYSIS.md`:** The existing Phase 8 already implements patient-cluster bootstrap with 2000 draws, 5000 permutations, and Holm correction. This Phase does NOT modify Phase 8. It ADDS supplementary visualization and sensitivity analysis.

**Formal Solution:**
- Pre-registered complete-case analysis framework with explicit MCAR characterization.
- Endpoint-specific $N$ prominently reported.
- Seed-specific distribution visualizations (raincloud plots).
- Sensitivity analysis: median-imputation of seed 271 conditional metrics.

**Implementation Protocol:**
1. Create `docs/CONDITIONAL_METRIC_POLICY.md` detailing:
   - Missingness mechanism (MCAR relative to random initialization)
   - Why seed 271 is MCAR (converged normally, score degeneracy is seed-specific)
   - Complete-case vs. imputation sensitivity comparison
2. Modify `aggregate_rows()` in `src/evaluate.py` to additionally output per-seed arrays (supplementary CSV).
3. Generate raincloud plots for all 7 metrics using `ptitprince` or `seaborn`.
4. Include sensitivity analysis table: results if seed 271 IoU/Dice were imputed via median of seeds 17/42/137/314.

**Computational Budget:** CPU-only. Plotting only.

👤 **Task Assignment:**
- **Mohammad Amin:** Refactor `aggregate_rows()` in `src/evaluate.py` to return per-seed arrays. Implement MCAR verification test. Draft `docs/CONDITIONAL_METRIC_POLICY.md`.
- **Pouyan:** Generate raincloud plots. Run median-imputation sensitivity analysis. Format supplementary table. Ensure all CSVs explicitly log `n_attempted` vs `n_defined`.
- **Joint:** Cross-check table formatting against journal guidelines. Co-author Limitations Section.

---

## 🔹 4. XAI Validity & Sanity Protocols

**Current Limitation:** Grad-CAM is applied without sanity checks. `EXPLAINABILITY.md` explicitly states: "parameter-randomization test and target-label sanity test were not implemented."

**Critical Constraint from `EXPLAINABILITY.md`:** Analysis remains scoped to seed-17 checkpoints and the 300-image Phase 6 sample. Sanity checks run on a SUBSET of this sample.

**Critical Constraint from `PROJECT_PLAN.md`:** "The draft's poor-localization true-positive category, corrupted-input CAM comparison, parameter-randomization test, and target-label sanity test were not implemented." → This Phase implements the parameter-randomization test.

**Formal Solution:**
Implement Adebayo et al. (2018) sanity tests:
1. **Model Parameter Randomization Test:** Compare Grad-CAM maps from trained vs. randomly initialized weights.
2. **Data Randomization Test:** Compare maps from real vs. pixel-shuffled inputs.

$$
C_{\text{sanity}} = \frac{1}{K}\sum_{k=1}^K \text{Corr}\left(M_{\text{trained}}^{(k)}, M_{\text{random}}^{(k)}\right)
$$

**Implementation Protocol:**
1. Create `src/explainability/sanity_checks.py` (~200 LOC).
2. Run on a stratified subset (50 images from the Phase 6 sample, ~55 targets).
3. For parameter randomization: reinitialize model weights with `torch.nn.init.xavier_normal_`, run Grad-CAM, compare.
4. For data randomization: shuffle pixel values within each image, run Grad-CAM, compare.
5. Report failure rates per detector and per failure mode.
6. Add qualitative panel: `Trained vs. Randomized Grad-CAM`.
7. Create `configs/sanity_checks.yaml`.
8. Write to `results/tables/gradcam_sanity_summary.csv` and `results/figures/gradcam_sanity_panel.png`.

**Computational Budget:** GPU required. 50 images × 2 detectors × 2 tests × 2 (trained + random) = 400 forward+backward passes. With 8GB VRAM and batch-1, estimated runtime: ~20-40 minutes.

**VRAM Note:** Each forward+backward pass at 640×640 with Grad-CAM hook uses ~1.5-2 GB. Sequential execution (no batching) stays within 8GB.

👤 **Task Assignment:**
- **Pouyan:** Implement `src/explainability/sanity_checks.py`. Run on stratified subset. Generate qualitative comparison panels. Log failure-mode frequencies.
- **Mohammad Amin:** Integrate sanity checks into `run_explainability.py`. Design failure-mode taxonomy table. Write Methodology Section (XAI validation protocol) and Discussion Section (clinical interpretability limits).
- **Joint:** Review failure-mode classifications. Co-author Results Section. Update `docs/EXPLAINABILITY.md` with sanity check results.

---

## 🔹 5. Physics-Aware Domain Shift Robustness

**Current Limitation:** Robustness is evaluated only on digital corruptions. `ROBUSTNESS.md` explicitly states: "Brightness and JPEG changes do not simulate scanner physics, DICOM window/VOI processing, reconstruction, population shift, or a new site."

**Critical Constraint from `DATASHEET.md`:** "DICOM conversion uses deterministic MONOCHROME1 inversion and per-image min-max scaling to 8-bit PNG. It does not reproduce vendor-specific window/VOI processing." → The existing preprocessing ALREADY introduces a VOI-related limitation. The acquisition shift analysis should be framed as quantifying the impact of this known limitation.

**Critical Constraint from `ROBUSTNESS.md`:** Analysis remains seed-17 only, 300-image sample. New shifts use the SAME sample.

**Formal Solution:**
$$
\text{DSI} = 1 - \frac{\text{Performance}_{\text{shifted}}}{\text{Performance}_{\text{clean}}}
$$

Three physics-based shifts:
- **VOI Windowing:** Apply window/level transforms to simulate radiologist adjustments (window width: 1500-2500 HU equivalent, level: -500 to 0 HU equivalent). Since images are already 8-bit PNG, approximate via contrast/brightness curves calibrated to typical lung window settings.
- **Poisson Noise:** Model dose reduction by applying signal-dependent Poisson noise (scale factors: 0.25, 0.5, 0.75 of original signal).
- **Reconstruction Kernel:** Apply realistic downsampling + Gaussian blur mimicking scanner reconstruction (kernel sizes: 3, 5, 7 pixels).

**Implementation Protocol:**
1. Extend `src/robustness/corruptions.py` with `AcquisitionShift` class (~150 LOC).
2. Add `configs/acquisition_shifts.yaml` with physically calibrated parameters.
3. Run inference on shifted subset (same 300-image sample, seed-17 checkpoints).
4. Compute DSI curves and compare against existing digital corruption retention.
5. Write to `results/tables/acquisition_shift_results.csv` and `results/figures/acquisition_shift_dsi.png`.

**Computational Budget:** GPU required. 300 images × 2 detectors × 3 shift types × 3 severities = 5400 inference passes. With batch-1 and 8GB VRAM, estimated runtime: ~45-90 minutes.

**VRAM Note:** Same as Phase 6 robustness (which already runs 72 conditions successfully on 8GB). The acquisition shift adds ~36 new conditions (3 types × 3 severities × 2 detectors × 300 images). Total inference passes: 5400 additional. This is within the demonstrated capacity of the existing robustness pipeline.

👤 **Task Assignment:**
- **Pouyan:** Implement `AcquisitionShift` class. Configure `configs/acquisition_shifts.yaml`. Run inference pipeline. Validate transform parameters against radiology physics literature.
- **Mohammad Amin:** Compute DSI curves. Integrate retention analysis into statistical pipeline. Author Robustness Section. Frame results in context of the known DICOM→PNG preprocessing limitation from `DATASHEET.md`.
- **Joint:** Co-author Discussion Section (deployment realism). Update `docs/ROBUSTNESS.md` with acquisition shift results.

---

## 🔹 6. Benchmark Reporting & Standards Compliance

**Current Limitation:** Reproducibility is excellent but lacks formal alignment with medical AI reporting guidelines.

**Implementation Protocol:**
1. Create `docs/REPORTING_CHECKLIST.md` mapping project features to:
   - CLAIM checklist (data provenance, split strategy, threshold selection, evaluation metrics, uncertainty quantification)
   - TRIPOD-AI checklist (model development, validation, performance reporting)
   - STARD-AI checklist (diagnostic accuracy study reporting)
2. Explicitly document hardware, software, seeds, and dependency hashes in formal supplementary table format.
3. Prepare LaTeX manuscript structure:
   - Abstract, Introduction, Related Work, Methodology, Results, Discussion, Limitations, Conclusion
   - Map existing 12-section report to new structure
4. Create supplementary materials:
   - Supplementary Table S1: CLAIM checklist mapping
   - Supplementary Table S2: TRIPOD-AI checklist mapping
   - Supplementary Table S3: Per-seed raw results (all 10 checkpoints)
   - Supplementary Table S4: Conditional metric sensitivity analysis
   - Supplementary Figure S1: Raincloud plots
   - Supplementary Figure S2: DCA curves
   - Supplementary Figure S3: XAI sanity check panels
   - Supplementary Figure S4: Acquisition shift DSI curves

👤 **Task Assignment:**
- **Pouyan:** Map all project artifacts to CLAIM/TRIPOD-AI/STARD-AI checklists. Format supplementary tables. Prepare `docs/REPORTING_CHECKLIST.md`. Handle LaTeX formatting, journal template conversion, and figure/table styling.
- **Mohammad Amin:** Draft the formal "Reproducibility & Reporting Statement". Write Cover Letter. Finalize manuscript structure, reference formatting, and submission metadata. Update `docs/REPRODUCIBILITY.md` with new module contracts.
- **Joint:** Cross-validate checklist completeness. Run final pre-submission proofread. Coordinate Zenodo DOI archiving.

---

## 🔹 7. Repository Infrastructure & CI Updates

**Rationale:** New modules require test coverage, CI integration, and config management consistent with the existing codebase patterns.

**Implementation Protocol:**
1. Add unit tests for each new module:
   - `tests/test_threshold_calibration.py`
   - `tests/test_decision_curve.py`
   - `tests/test_sanity_checks.py`
   - `tests/test_acquisition_shift.py`
2. Update `.github/workflows/ci.yml` to include new test modules.
3. Ensure all new configs use strict Pydantic validation (`extra="forbid"`, `frozen=True`).
4. Ensure all new outputs use atomic writes and SHA-256 hash verification.
5. Update `README.md` reproduction commands with new phases.

👤 **Task Assignment:**
- **Mohammad Amin:** Write tests for `threshold_calibration.py` and `decision_curve.py`. Update CI workflow.
- **Pouyan:** Write tests for `sanity_checks.py` and `acquisition_shift.py`. Update `README.md`.
- **Joint:** Review all PRs for consistency with existing code patterns.

---

## 🤝 Coordination & Fairness Protocol

| Dimension | Balance Mechanism |
|-----------|------------------|
| **Code Volume** | Amin: ~500 LOC (threshold calibration, DCA integration, aggregate_rows refactor, CI) \| Pouyan: ~500 LOC (sanity checks, acquisition shifts, DCA module, plots, checklists) |
| **Math/Stats** | Amin leads bootstrap/calibration/DCA math; Pouyan validates outputs & generates clinical mapping |
| **Visualization** | Pouyan leads plot generation (raincloud, DCA, XAI panels, DSI); Amin leads table formatting & manuscript integration |
| **Writing** | Split by section: Methodology (Amin), Clinical/Robustness (Pouyan), Results/Discussion (Co-authored 50/50) |
| **Documentation** | Pouyan leads LIMITATIONS/LITERATURE/REPORTING updates; Amin leads HYPOTHESES/DECISION_LOG/REPRODUCIBILITY updates |
| **Review Cycle** | Every PR requires mutual code review + 1-day manuscript cross-read before merge |
| **Risk Mitigation** | If Phase 5 (Acquisition Shifts) exceeds timeline, it moves to `Future Work` while Phases 1–4 & 6 remain intact for Q1 submission |

---

## 📅 Execution Timeline (7 Weeks)

| Week | Focus | Deliverables |
|------|-------|--------------|
| 0 | Pre-requisites (Phase 0) | HYPOTHESES.md updated, LIMITATIONS.md updated, DECISION_LOG.md updated, LITERATURE_REVIEW.md updated |
| 1 | Phase 1 & 2 Core | `threshold_calibration.py`, DCA module, initial sensitivity plots |
| 2 | Phase 3 & 4 Core | Conditional metric refactor, sanity checks, raincloud plots, failure taxonomy |
| 3 | Phase 5 & Integration | `AcquisitionShift`, DSI curves, pipeline hooks, cross-seed validation |
| 4 | Phase 7 & Manuscript Draft | Tests, CI, README, Sections 4–6 complete, all figures/tables generated |
| 5 | Phase 6 & Polish | Checklist mapped, LaTeX formatting, statistical cross-checks, cover letter |
| 6 | Review & Revision | Internal review, response-to-reviewers template, Zenodo prep |
| 7 | Submission | Final proofread, journal portal submission |

---

## ⚠️ Critical Constraints Checklist

Before marking any phase complete, verify:
- [ ] All new outputs use atomic writes with SHA-256 hashes
- [ ] All new configs use strict Pydantic validation
- [ ] Seed 271 handling matches Phase 8 policy (valid zeros, not coerced)
- [ ] n=3 frozen analyses are NOT modified or regenerated
- [ ] All GPU operations fit within 8GB VRAM (batch-1, sequential execution)
- [ ] All CPU operations complete within reasonable time (<60 minutes)
- [ ] New tests pass: `pytest -q tests/test_<module>.py`
- [ ] Ruff lint passes: `ruff check src tests`
- [ ] Documentation updated in corresponding `docs/` file
- [ ] Decision Log entry appended for any scope change

---
