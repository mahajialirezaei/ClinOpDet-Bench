# ClinOpDet-Bench: Q1 Readiness TODO

**Target Venues:** IEEE TMI / Medical Image Analysis / Nature Scientific Reports

**Authors:** Mohammad Amin Haji Alirezaei & Pouyan Delivandani

**Hardware Constraint:** RTX 4060 Laptop GPU, 8 GB VRAM, 16 GB RAM, i7-13650HX

**Date:** 2026-08-21

**Revision:** v2.1 (Expanded with explicit task assignments and technical execution details)

---

## 🔒 Scope Boundary: Frozen vs. Expanded

### FROZEN (unchanged, no re-execution):

* Existing Phase 5 clean evaluation for Faster R-CNN and YOLO11s (10 checkpoints, n=5)
* Existing Phase 6 robustness digital corruptions (seed-17 only, 300 images)
* Existing Phase 8 statistics (patient-cluster bootstrap, n=5/n=4) for the first two detectors
* n=3 threshold selection, FROC, Pareto → archived as `*_n3_archive.*`

### EXPANDED / NEW (to be implemented):

* **Phase 9:** Transformer Architecture Baseline (RT-DETR, n=5 seeds)
* **Phase 10:** Probability Calibration & Reliability (ECE & Brier Score)
* **Phase 11:** Patient-Cluster-Aware Threshold Calibration (CPU-only)
* **Phase 12:** Decision Curve Analysis (DCA, CPU-only)
* **Phase 13:** Inference Latency Variance & MCAR Handling (Statistical Validation)
* **Phase 14:** XAI Sanity Checks (GPU, seed-17 subset)
* **Phase 15:** RAW-DICOM Physics-Aware Acquisition Shifts (GPU, seed-17 subset)
* **Phase 16:** Reporting Standards Compliance (CLAIM/TRIPOD-AI)

---

## 🔹 0. Pre-requisites: Hypotheses, Limitations, Decision Log

**Assignee:Mohammad Amin**

* **0a. Update `docs/HYPOTHESES.md`:** Add the mathematical and logical formulations for H6 (RT-DETR Robustness), H7 (ECE & Overconfidence), H8 (Threshold Calibration Shift), H9 (DCA Net Benefit), and H10 (XAI Sanity Degradation).
* **0c. Update `docs/DECISION_LOG.md`:** Log D-004 (RT-DETR inclusion), D-005 (DICOM-level shifts overriding 8-bit shifts), and D-006 (MCAR handling for seed 271).

**Assignee: Pouyan**

* **0b. Update `docs/LIMITATIONS.md`:** Document validation-split optimization constraints for calibration, the fixed 22.5% RSNA cohort disease prevalence assumption for DCA, and the subset limitations for XAI/Acquisition shifts.
* **0d. Update `docs/LITERATURE_REVIEW.md`:** Add sections covering patient-cluster bootstrap in medical imaging, DCA (Vickers et al., 2016), Adebayo sanity checks, and radiologic physics domain shifts (HU windowing, Poisson noise).

---

## 🔹 1. Transformer Architecture Baseline (RT-DETR) [Phase 9]

**Assignee:Mohammad Amin**

**Detailed Protocol:**

1. **Configuration:** Create `configs/rt_detr.yaml` utilizing Ultralytics RT-DETR implementation. Ensure `batch_size`, `amp_dtype`, and `optimizer` settings are hardware-compliant.
2. **Data Consistency:** Bind training strictly to the existing Phase 5 frozen data splits (`data/splits/`).
3. **Execution:** Train `rtdetr-l.pt` (or size-appropriate variant) over the required 5 seeds (17, 42, 137, 271, 314).
4. **Integration:** Execute unified inference on the test split. Ensure output structures (`.json.gz` prediction bundles) strictly match the Faster R-CNN/YOLO format to seamlessly feed downstream scripts.

---

## 🔹 2. Probability Calibration & Reliability (ECE) [Phase 10]

**Assignee:Mohammad Amin**

**Detailed Protocol:**

1. **Implementation:** Create `src/stats/calibration.py`.
2. **Metric Calculation:** Implement Expected Calibration Error (ECE) using 10 uniform bins and compute the Brier Score for all three detectors (Faster R-CNN, YOLO11s, RT-DETR).
3. **Diagnosis:** Generate Reliability Diagrams (Calibration Curves) mapping mean predicted probability vs. fraction of true positives.
4. **Correction:** If ECE > 5% for any model (specifically YOLO seed 271), implement Post-hoc Temperature Scaling by minimizing NLL on the validation set. Save the optimal temperature parameters $T$ and evaluate the scaled probabilities on the test set.

---

## 🔹 3. Patient-Cluster-Aware Threshold Calibration [Phase 11]

**Assignee:Mohammad Amin**

**Detailed Protocol:**

1. **Implementation:** Create `src/stats/threshold_calibration.py`.
2. **Logic:** Calculate $F1_\beta(\tau)$ using the formula:

$$F1_\beta(\tau) = \frac{(1+\beta^2) \cdot P(\tau) \cdot R(\tau)}{\beta^2 P(\tau) + R(\tau)}, \quad \beta = \sqrt{\frac{C_{FN}}{C_{FP}}}$$


3. **Optimization:** Implement an outer loop scanning $\tau \in [0.01, 0.99]$. Implement an inner loop executing 2000 bootstrap resamples grouped by `nih_patient_id`.
4. **Outputs:** Select $\tau^*$ optimizing the lower 95% CI bound. Write outputs to `results/tables/threshold_calibration_summary.csv`. Generate parameter sensitivity plots for $\beta \in \{1, 3, 5, 10\}$.

---

## 🔹 4. Clinical Utility & Decision Curve Analysis (DCA) [Phase 12]

**Assignee: Mohammad Amin (Calculation Logic) & Pouyan (Visualization)**

**Detailed Protocol:**

1. **Logic (Mohammad Amin):** Create `src/clinical/decision_curve.py`. Compute True Positives ($TP$) and False Positives ($FP$) across $\tau \in [0.01, 0.99]$ directly from frozen prediction bundles.
2. **Metrics (Mohammad Amin):** Implement the Net Benefit formula:

$$\text{Net Benefit}(\tau) = \frac{TP}{N} - \frac{FP}{N} \cdot \frac{p_t}{1-p_t}, \quad p_t = \frac{\tau}{1-\tau}$$



Apply the 2000 patient-cluster resampling to compute 95% CIs.
3. **Visualization (Pouyan):** Plot Net Benefit curves for all detectors, including standard "treat-all" and "treat-none" reference lines. Save to `results/figures/dca_curves.png`.

---

## 🔹 5. Statistical Rigor, Latency Variance & MCAR Handling [Phase 13]

**Assignee: Mohammad Amin (Metrics & Policy) & Pouyan (Visualization)**

**Detailed Protocol:**

1. **Hardware Metrics (Mohammad Amin):** Extract Inference Latency (ms/image) and Peak VRAM utilization from the unified evaluation logs. Compute 95% CIs across the 5 seeds for all architectures.
2. **MCAR Formalization (Mohammad Amin):** Draft `docs/CONDITIONAL_METRIC_POLICY.md` mathematically justifying why YOLO seed 271 is Missing Completely At Random (MCAR) and run a median-imputation sensitivity analysis.
3. **Visualization (Pouyan):** Create `src/plot_rainclouds.py` and generate raincloud plots (using `ptitprince` or `seaborn`) for the 7 primary predictive metrics and hardware operational metrics. Save to `results/figures/raincloud_metrics.png`.

---

## 🔹 6. XAI Validity & Sanity Protocols [Phase 14]

**Assignee: Pouyan**

**Detailed Protocol:**

1. **Implementation:** Create `src/explainability/sanity_checks.py`. Define a subset of 50 images from the Phase 6 test pool.
2. **Parameter Randomization:** Clone the trained model, reinitialize all weights using `torch.nn.init.xavier_normal_`, and extract Grad-CAM heatmaps.
3. **Data Randomization:** Shuffle pixel matrices of the input images and extract Grad-CAM heatmaps from the trained model.
4. **Metric Calculation:** Compute Pearson correlation $C_{\text{sanity}}$:

$$C_{\text{sanity}} = \frac{1}{K}\sum_{k=1}^K \text{Corr}\left(M_{\text{trained}}^{(k)}, M_{\text{random}}^{(k)}\right)$$


5. **Outputs:** Output failure rates to `results/tables/gradcam_sanity_summary.csv` and stitch qualitative "Trained vs. Randomized" visual panels to `results/figures/gradcam_sanity_panel.png`.

---

## 🔹 7. RAW-DICOM Physics-Aware Acquisition Shifts [Phase 15 - REVISED]

**Assignee: Pouyan**

**Detailed Protocol:**

1. **Implementation:** Create `src/robustness/dicom_shifts.py`. This script *must* bypass the processed 8-bit PNGs and read directly from `data/raw/dicoms/*.dcm`.
2. **VOI Windowing:** Apply Hounsfield Unit (HU) window/level variations directly to the 12/16-bit raw pixel arrays.
3. **Poisson Noise:** Apply signal-dependent Poisson noise directly to the raw sensor data arrays to accurately model dose reduction.
4. **Reconstruction Kernel:** Apply Gaussian blur matrices mimicking variation in CT/X-ray reconstruction filters.
5. **Post-Processing:** Run the modified raw arrays through the existing `per_image_minmax` scaling to produce valid 8-bit inputs for the models.
6. **Inference & Metrics:** Run inference over the shifted 300-image subset. Calculate the Domain Shift Index (DSI). Save outputs to `results/tables/acquisition_shift_results.csv`.

---

## 🔹 8. Benchmark Reporting & Standards Compliance [Phase 16]

**Assignee: Pouyan (Checklists & LaTeX) & Mohammad Amin (Methodology Review)**

**Detailed Protocol:**

1. **Checklists (Pouyan):** Create `docs/REPORTING_CHECKLIST.md`. Systematically map project artifacts, dataset splits, threshold selections, and code files to CLAIM, TRIPOD-AI, and STARD-AI checklist items.
2. **Supplementary Data (Pouyan):** Compile Supplementary Tables S1-S4 and Figures S1-S4 as defined in the master plan.
3. **LaTeX Integration (Pouyan):** Format the manuscript according to the target journal's LaTeX template.
4. **Review (Mohammad Amin):** Verify that statistical methodologies and MCAR handling in the manuscript perfectly reflect the codebase logic. Compile the final "Reproducibility & Reporting Statement".