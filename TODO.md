# ClinOpDet-Bench: Nature Sci Rep / Q1 Readiness TODO (v3)

**Target Venues:** Nature Scientific Reports / IEEE TMI / Medical Image Analysis
**Authors:** Mohammad Amin Haji Alirezaei & Pouyan Delivandani
**Hardware Constraint:** RTX 4060 Laptop GPU, 8 GB VRAM, 16 GB RAM, i7-13650HX
**Date:** 2026-08-22
**Revision:** v3 (Pruned for Scientific Validity, Scope Control, and Radiologic Physics Accuracy)

---

## 🔒 Scope Boundary: Frozen vs. Expanded

### FROZEN (unchanged, no re-execution):
- Existing Phase 5 clean evaluation for Faster R-CNN and YOLO11s (10 checkpoints, n=5)
- Existing Phase 6 robustness digital corruptions (seed-17 only, 300 images)
- Existing Phase 8 statistics (patient-cluster bootstrap, n=5/n=4) for the first two detectors
- n=3 threshold selection, FROC, Pareto → archived as `*_n3_archive.*`

### EXPANDED / NEW (to be implemented):
- **Phase 9:** Detection-Specific Probability Calibration (Multivariate/Küppers)
- **Phase 10:** Patient-Cluster-Aware Threshold Calibration ($\beta$ sweep)
- **Phase 11:** Decision Curve Analysis (DCA)
- **Phase 12:** XAI Sanity Checks (GPU, seed-17 subset)
- **Phase 13:** Radiography-Physics-Aware Acquisition Shifts (DICOM VOI LUT)
- **Phase 14:** Reporting Standards Compliance (CLAIM/TRIPOD-AI)

---

## 🔹 0. Pre-requisites: Hypotheses, Limitations, Decision Log

### 0a. Update `docs/HYPOTHESES.md`
- **H6 (Calibration):** Object detectors will exhibit significant miscalibration, requiring Multivariate Detection Calibration (Küppers et al.) rather than naive classifier ECE.
- **H7 (Threshold Calibration):** Cost-weighted threshold calibration ($\beta \in \{1,3,5,10\}$) will shift the optimal operating points significantly compared to the fixed 0.25 threshold.
- **H8 (DCA):** Decision Curve Analysis will reveal distinct threshold ranges where one detector dominates net benefit.
- **H9 (XAI Sanity):** Adebayo parameter randomization will yield $C_{\text{sanity}} > 0.3$, invalidating raw Grad-CAM for clinical reasoning.

### 0b. Update `docs/LIMITATIONS.md`
- **Seed 271 Degeneracy:** Explicitly define Seed 271's behavior as an "operational confidence-score degeneracy" at the frozen threshold, not as missing data.
- **Calibration Scope:** Optimized on validation split; no external transportability guaranteed.
- **DCA Assumption:** Assumes RSNA cohort disease prevalence (22.5%).

### 0c. Update `docs/DECISION_LOG.md`
- **D-004:** Reject architectural expansion (Transformer/RT-DETR) to focus on deep methodological evaluation of existing baselines and bound the Computational Overhead.
- **D-005:** Refine physics-aware perturbations to target DICOM VOI LUT/Window Center-Width, explicitly rejecting CT-specific Hounsfield Unit (HU) transformations.
- **D-006:** Reject MCAR imputation for Seed 271 to preserve statistical integrity of observed operational failures.

---

## 🔹 1. Detection-Specific Probability Calibration [Phase 9]

**Scientific Gap:** Medical decisions require calibrated probabilities. Naive classifier ECE is invalid for Object Detection due to the variable number of predicted bounding boxes and spatial alignment.
**Implementation Protocol:**
1. Implement multivariate detection calibration (Küppers et al.) to compute the Calibration Error.
2. Generate Reliability Diagrams specifically adapted for Object Detection.
3. Quantify the overconfidence pathology observed in YOLO Seed 271 without applying statistical imputation.

---

## 🔹 2. Patient-Cluster-Aware Threshold Calibration [Phase 10]

**Scientific Gap:** Previous thresholds assumed i.i.d. observations and symmetric costs, ignoring patient-level dependence and clinical reality ($C_{FN} \gg C_{FP}$).
**Implementation Protocol:**
1. Optimize threshold $\tau^*$ via nested bootstrap (2000 resamples grouped by `nih_patient_id`).
2. Perform parameter sensitivity sweep for clinical cost asymmetry: $\beta \in \{1, 3, 5, 10\}$.
3. Output $\tau^*$ per detector with 95% CIs.

---

## 🔹 3. Clinical Utility & Decision Curve Analysis (DCA) [Phase 11]

**Implementation Protocol:**
1. Compute True Positives ($TP$) and False Positives ($FP$) across $\tau \in [0.01, 0.99]$ from frozen prediction bundles.
2. Calculate Net Benefit (NB) curves with 95% bootstrap CIs (patient-cluster resampling).
3. Plot NB curves against "treat-all" and "treat-none" baselines.

---

## 🔹 4. XAI Validity & Sanity Protocols [Phase 12]

**Implementation Protocol:**
1. Run Adebayo sanity tests on a stratified subset (50 images).
2. **Parameter Randomization:** Compare Grad-CAM from trained vs. randomly initialized weights (Xavier).
3. **Data Randomization:** Compare maps from real vs. pixel-shuffled inputs.
4. Compute Pearson correlation $C_{\text{sanity}}$ and extract failure rates.

---

## 🔹 5. Radiography-Physics-Aware Acquisition Shifts [Phase 13]

**Implementation Protocol:**
1. Create `src/robustness/radiography_shifts.py`.
2. **DICOM VOI LUT:** Apply Window Center/Width variations appropriate for planar radiography (chest X-ray pixel values), strictly avoiding CT HU terminology.
3. **Poisson Noise:** Apply signal-dependent Poisson noise.
4. **Reconstruction Kernel:** Apply Gaussian blur matrices.
5. Run inference on the 300-image subset and compute Domain Shift Index (DSI).

---

## 🔹 6. Benchmark Reporting & Standards Compliance [Phase 14]

**Implementation Protocol:**
1. Create `docs/REPORTING_CHECKLIST.md` mapped to CLAIM, TRIPOD-AI, and STARD-AI.
2. Prepare Supplementary Materials, including Raincloud plots for metric distributions.

---

## 🤝 Coordination & Task Assignment

| Developer | Tasks |
| :--- | :--- |
| **Mohammad Amin** | Multivariate Detection Calibration (Phase 9), Threshold Calibration & Bootstrap (Phase 10), DCA Logic (Phase 11), Methodology & Statistical Rigor Writing. |
| **Pouyan** | Radiography-Physics Shifts (Phase 13), XAI Sanity Checks (Phase 12), Visualizations (DCA plots, Raincloud plots, Reliability Diagrams), Reporting Checklists, Clinical Relevance Writing. |

---