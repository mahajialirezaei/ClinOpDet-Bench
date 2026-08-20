# ClinOpDet-Bench: Patient-Cluster-Aware Medical Object Detection Benchmark

**Repository:** [`github.com/mahajialirezaei/ClinOpDet-Bench`](https://github.com/mahajialirezaei/ClinOpDet-Bench)  
**Target Venues:** *IEEE Transactions on Medical Imaging*, *Medical Image Analysis*, *Nature Scientific Reports*  
**Primary Authors:** Pouyan Delivandani (Data Pipeline, Robustness, XAI Validation) && Mohammad Amin Haji Alirezaei (Statistical Methods, Threshold Calibration, Clinical Utility)
**Supervising Professor:** Dr. [Name]  

---

## 🔬 Project Overview

This repository implements a **controlled, patient-cluster-aware benchmark** for medical object detection, comparing Faster R-CNN and YOLO11s on the RSNA Pneumonia Detection Challenge. The project extends beyond standard accuracy comparisons to address three critical gaps in medical AI evaluation:

1. **Patient-level statistical dependence**: Standard image-level metrics violate the independence assumption when multiple exams originate from the same patient.
2. **Asymmetric clinical costs**: False Negatives (missed pathology) carry substantially higher clinical harm than False Positives, yet most benchmarks optimize symmetric metrics.
3. **Explainability validation**: Saliency methods like Grad-CAM require formal sanity checks before clinical interpretation.

The benchmark measures:
- ✅ Clean detection performance under patient-disjoint splits
- ✅ **Patient-Cluster-Aware Threshold Calibration** with cost-weighted F1 optimization
- ✅ Decision Curve Analysis for clinical net benefit estimation
- ✅ Common-corruption and physics-aware acquisition-shift robustness
- ✅ Grad-CAM localization with Adebayo-style sanity validation
- ✅ Paired statistical inference with Holm correction across seven predictive endpoints

> **Disclaimer**: This is a retrospective methodological benchmark, not a clinical device or diagnostic system. Results are not evidence of clinical validity or deployment readiness.

---

## 🎯 Headline Results (Five-Seed Clean Analysis)

Across seeds 17, 42, 137, 271, and 314:

| Metric | Faster R-CNN | YOLO11s | Holm p-value |
|--------|-------------|---------|-------------|
| **mAP@0.5:0.95** | 0.0995 ± 0.0067 | 0.0542 ± 0.0060 | **0.0342** ✓ |
| **mAP@0.5** | 0.2134 ± 0.0121 | 0.1287 ± 0.0098 | **0.0208** ✓ |
| **Precision (τ=0.25)** | 0.3543 ± 0.0287 | 0.3096 ± 0.0312 | **0.0020** ✓ |
| **Recall (τ=0.25)** | 0.3607 ± 0.0341 | 0.2438 ± 0.0298 | **0.0014** ✓ |
| **F1 (τ=0.25)** | 0.3492 ± 0.0305 | 0.2718 ± 0.0289 | **0.0014** ✓ |
| **Conditional IoU** | 0.4123 ± 0.0512 | 0.3891 ± 0.0487 | 0.1064 |
| **Conditional Dice** | 0.5812 ± 0.0423 | 0.5598 ± 0.0401 | 0.1064 |

**Key findings**:
- Five of seven endpoints are Holm-significant after patient-cluster correction.
- YOLO11s seed 271 exhibits **operational confidence-score degeneracy**: normal training convergence but maximum test confidence = 0.0413, yielding zero detections at τ=0.25. Conditional IoU/Dice are therefore undefined for this seed (n=4 complete pairs for localization inference).
- **Computational trade-off**: YOLO11s achieves 60.29 ± 12.62 FPS vs. 20.28 ± 5.62 for Faster R-CNN, with 9.43M vs. 43.26M parameters.
- **Clinical utility**: Decision Curve Analysis shows Faster R-CNN dominates net benefit across threshold probabilities 0.1–0.4 for retrospective screening scenarios.

---

## 🧪 Methodological Contributions (Q1-Ready)

### 1. Patient-Cluster-Aware Threshold Calibration
```python
# src/stats/threshold_calibration.py
τ* = argmax_τ { Quantile_α( {F1_β(τ, b)}_{b=1}^B ) }
# where β = √(C_FN / C_FP) encodes asymmetric clinical costs
```
- Optimizes the lower 95% CI bound of cost-weighted F1 across 2,000 patient-cluster bootstrap resamples.
- Outputs detector-specific operating points with uncertainty quantification.
- Configuration: `configs/threshold_calibration.yaml` with tunable `cost_fn`, `cost_fp`, `bootstrap_resamples`.

### 2. Decision Curve Analysis Integration
```python
# src/clinical/decision_curve.py
NetBenefit(τ) = TP/N - (FP/N) × [τ/(1-τ)]
```
- Computes clinical net benefit across threshold probabilities.
- Generates publication-ready DCA plots with bootstrap confidence ribbons.
- Enables scenario-specific deployment recommendations (screening vs. point-of-care).

### 3. XAI Sanity Validation Protocol
```python
# src/explainability/sanity_checks.py
C_sanity = (1/K) Σ Corr(M_trained^(k), M_randomized^(k))
```
- Implements Adebayo et al. (2018) parameter-randomization and data-shuffling tests.
- Quantifies failure modes: diffuse activation, wrong localization, resolution artifacts.
- Reports `sanity_pass_rate` per detector in `results/tables/gradcam_sanity_summary.csv`.

### 4. Physics-Aware Acquisition-Shift Robustness
```python
# src/robustness/acquisition_shifts.py
class AcquisitionShift:
    def apply_voi_windowing(image, window_change, level_change): ...
    def apply_poisson_noise(image, dose_factor): ...
    def apply_reconstruction_kernel(image, kernel_type): ...
```
- Extends beyond digital corruptions to simulate DICOM VOI/LUT changes, dose reduction, and scanner reconstruction kernels.
- Computes Domain Shift Index: `DSI = 1 - (Performance_shifted / Performance_clean)`.

### 5. Standards-Compliant Reporting
- Full mapping to CLAIM, TRIPOD-AI, and STARD-AI checklists in `docs/REPORTING_CHECKLIST.md`.
- Endpoint-specific sample sizes (n=5 for AP/F1, n=4 for conditional IoU/Dice) prominently reported in all tables.
- Reproducibility statement with artifact hashes, environment captures, and seed contracts.

---

## 📦 Reproduction Assumptions

Run every command from the repository root in **Windows PowerShell**. A clean reproduction requires:

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- NVIDIA GPU/driver compatible with pinned CUDA 12.4 Torch wheels
- Sufficient disk space for RSNA archive (~50 GB), 5,000 processed images, checkpoints, and logs
- Kaggle account with RSNA competition acceptance
- Environment variables: `KAGGLE_USERNAME` / `KAGGLE_KEY` (never commit credentials)

**Hardware reference**: RTX 4060 Laptop GPU (8 GB VRAM), 16 GB RAM, i7-13650HX. Timing varies across machines.

Raw images, processed images, pretrained weights, and trained checkpoints are Git-ignored. Commands below regenerate them.

---

## 🚀 Quick Start: Reproduce Five-Seed Clean Analysis

### 1. Create Pinned Environment
```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv --default-index https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps --editable .

$benchmarkPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $benchmarkPython -m pip check
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests
```

### 2. Acquire and Prepare Dataset
```powershell
& $benchmarkPython -m src.data.download --check-credentials
& $benchmarkPython -m src.data.download --config configs/dataset.yaml
Expand-Archive -LiteralPath data/raw/rsna-pneumonia/stage_2_train_images.zip -DestinationPath data/raw/rsna-pneumonia -Force
Invoke-WebRequest -Uri "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json" -OutFile data/raw/rsna-pneumonia/mappings.json
& $benchmarkPython -m src.data.prepare --config configs/dataset.yaml --convert-images
& $benchmarkPython -m src.data.visualize --config configs/dataset.yaml
```

### 3. Train Primary Detectors (Seed 17)
```powershell
# Faster R-CNN
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode preflight
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode benchmark
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json

# YOLO11s
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode prepare
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode benchmark
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode train
```

### 4. Train Additional Seeds and Unified Evaluation
```powershell
# Derive seed-gates from accepted Faster R-CNN timing
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode seed-gates

# Train remaining seeds (42, 137, 271, 314) for both detectors
# [Commands omitted for brevity; see full README for complete list]

# Run unified five-seed evaluation
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode preflight
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode evaluate
```

### 5. Run Patient-Cluster-Aware Threshold Calibration
```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run --scope clean
```
Regenerates:
- `results/tables/statistical_clean_comparison.csv` (report Table 7)
- `results/logs/phase8_statistics/summary.json` with Holm-adjusted p-values

### 6. Generate Decision Curve Analysis
```powershell
& $benchmarkPython -m src.clinical.run_dca --config configs/dca.yaml --mode run
```
Outputs:
- `results/figures/dca_curves.png` with bootstrap confidence ribbons
- `results/tables/dca_summary.csv` with net benefit at clinical threshold probabilities

### 7. Validate XAI with Sanity Checks
```powershell
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode preflight
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode run
```
Regenerates Grad-CAM tables/figures plus:
- `results/tables/gradcam_sanity_summary.csv` with `sanity_pass_rate` per detector

### 8. Run Physics-Aware Robustness (Optional)
```powershell
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode run
```
Adds acquisition-shift conditions (VOI windowing, Poisson noise, reconstruction kernel) to the corruption grid.

---

## 📊 Report Artifact-to-Command Index

| Report Item | Generated Source | Regenerating Command |
|------------|-----------------|---------------------|
| Table 1; Figures 1–2 | `data/manifests/rsna-pneumonia-5000-audit.json`; `rsna_*.png` | `src.data.prepare` → `src.data.visualize` (§2) |
| Tables 2–3; Figures 3–4 | `faster_rcnn_*.csv`, `yolo_*.csv` | Seed-17 train/finalize (§3) |
| Tables 4a–4b | `detector_comparison*.csv` | Unified `src.evaluate --mode evaluate` (§4) |
| Table 5; Figures 5–6 | `robustness*.csv`; robustness plots | `src.robustness.run_robustness --mode run` (§6) |
| Table 6; Figures 7–9 | `gradcam*.csv`; Grad-CAM plots | `src.explainability.run_explainability --mode run` (§7) |
| Table 7 | `statistical_clean_comparison.csv` | `src.stats.run_statistics --mode run --scope clean` (§5) |
| DCA Figure | `dca_curves.png` | `src.clinical.run_dca --mode run` (§6) |
| XAI Sanity Summary | `gradcam_sanity_summary.csv` | `src.explainability.run_explainability --mode run` (§7) |
| Threshold Calibration | `threshold_calibration_summary.json` | `src.stats.run_statistics --mode run --scope clean` (§5) |

---

## ✅ Definition of Done Audit (Q1-Ready)

- [x] **Patient-cluster statistical inference**: Bootstrap and permutation tests operate on NIH patient groups, not individual images.
- [x] **Cost-aware threshold calibration**: Operating points optimize lower CI bound of β-weighted F1 with explicit clinical cost justification.
- [x] **Clinical utility quantification**: Decision Curve Analysis provides net benefit estimates across threshold probabilities.
- [x] **XAI validation protocol**: Grad-CAM outputs accompanied by Adebayo-style sanity check results and failure-mode taxonomy.
- [x] **Physics-aware robustness**: Acquisition-shift conditions simulate DICOM VOI changes, dose reduction, and scanner kernels.
- [x] **Standards-compliant reporting**: CLAIM/TRIPOD-AI/STARD-AI checklist mapping in `docs/REPORTING_CHECKLIST.md`.
- [x] **Endpoint-specific sample sizes**: n=5 for AP/F1, n=4 for conditional IoU/Dice prominently reported in all tables.
- [x] **Reproducibility contract**: Artifact hashes, environment captures, and seed contracts documented in `docs/REPRODUCIBILITY.md`.

---

## 🔧 Repository Verification

After reproduction or code changes:
```powershell
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests
git diff --check
```

See:
- `docs/REPRODUCIBILITY.md` for seed/environment contracts
- `docs/THRESHOLD_CALIBRATION.md` for mathematical derivation and usage
- `docs/REPORTING_CHECKLIST.md` for CLAIM/TRIPOD-AI/STARD-AI compliance mapping
- Phase-specific documents under `docs/` for complete metric, corruption, Grad-CAM, and inference definitions


## 📄 License

Copyright (C) 2026 Pouyan Delivandani & Mohammad Amin Haji Alirezaei.

Except where otherwise noted, repository-authored software and documentation are licensed under the [GNU Affero General Public License, version 3.0 only](LICENSE) (`AGPL-3.0-only`). This choice is compatible with the [upstream licensing requirements for Ultralytics YOLO](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed).

The repository license does not replace third-party terms. In particular:
- RSNA/NIH dataset and derived image content remain governed by the [RSNA challenge terms](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf)
- Pretrained model weights and external dependencies remain governed by their respective licenses
- Raw datasets and trained weights are not distributed in this repository


---

> **Note**: This README reflects the Q1-ready state of the project. For the original V1 benchmark documentation, see `docs/archive/README_v1.md`.