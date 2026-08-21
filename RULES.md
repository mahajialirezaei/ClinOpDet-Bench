# RULES.md

## Repository Rules & Research-Grade GitFlow Workflow

### 1. Core Branching Strategy
- `main`: **Publication-ready branch.** Contains the frozen manuscript, final LaTeX source, validated result artifacts, reproducibility scripts, and final checkpoint/config hashes. Direct commits are strictly forbidden.
- `develop`: Primary integration branch. All completed experimental modules (threshold calibration, DCA, XAI sanity, robustness shifts) are merged here for cross-module validation before manuscript integration.
- `feature/*`: Short-lived branches used for developing specific Q1 upgrades, statistical modules, or reporting utilities.

### 2. Branch Naming Conventions
All feature branches must explicitly indicate the research domain and use kebab-case for the module name.
- Statistical/Calibration: `feat/stats-[module-name]` (e.g., `feat/stats-patient-cluster-calibration`)
- Explainability/XAI: `feat/xai-[module-name]` (e.g., `feat/xai-sanity-checks`)
- Robustness/Shift: `feat/robustness-[module-name]` (e.g., `feat/robustness-acquisition-shift`)
- Clinical/DCA: `feat/clinical-[module-name]` (e.g., `feat/clinical-decision-curve`)
- Documentation: `docs/[section]` (e.g., `docs/methodology-update`)
- Bugfixes: `fix/[bug-name]` (e.g., `fix/coco-evaluator-precision-bug`)

### 3. Development & Merge Workflow
The research lifecycle must strictly follow these steps:
1. **Initialize Feature Branch:** Always branch from the latest `develop`.
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feat/stats-patient-cluster-calibration
   ```
2. **Commit Changes:** Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`). Every commit must be tied to a specific methodological or reproducibility improvement.
   ```bash
   git commit -m "feat: implement patient-cluster bootstrap for threshold calibration"
   ```
3. **Push and Open Pull Request (PR):** Target `develop`. Include a summary of mathematical changes, affected configs, and validation outputs.
4. **CI/CD Execution:** GitHub Actions automatically trigger on PR creation.
5. **Merge to `develop`:** Allowed only if:
   - CI pipeline completes successfully (green status).
   - Unit/integration tests pass with 100% coverage on new logic.
   - Artifact hashes match frozen baselines.
   - Co-author/methodology reviewer approves.
6. **Merge to `main` (Release/Manuscript Freeze):** Triggered when the full experimental pipeline is validated, tables/figures are regenerated from code, and the LaTeX manuscript aligns with the final artifact state.

### 4. GitHub Actions (.yml) Requirements
The `.github/workflows/` directory must contain the CI configuration.
- **Backend/Research Workflow:** Must execute `pytest` (unit + integration), `ruff check src tests`, artifact hash verification (config/checkpoint/prediction bundles), and LaTeX compilation check.
- **Hard-Block Rule:** Merging is strictly blocked if any workflow fails, if new code lacks tests, or if generated tables differ from committed CSVs without an explicit regeneration commit.

### 5. Documentation Constraint
You **MUST** create a `.md` document file in `docs/` for each new module, config, or statistical procedure implemented. Each document must contain:
- Mathematical formulation & algorithmic pseudocode.
- Input/output schema & config parameters.
- Validation protocol (unit tests, bootstrap convergence, sanity check results).
- Explicit linkage to the corresponding section in `report/report.md`.


## Agent Constraints & Research Integrity Guidelines
**Project:** `ClinOpDet-Bench` – Patient-Cluster-Aware Medical Object Detection Benchmark  
**Primary Authors:** Pouyan Delivandani & Mohammad Amin Haji Alirezaei    
**Target Venues:** *IEEE TMI*, *Medical Image Analysis*, *Nature Scientific Reports*  
**Domain:** Clinical AI Benchmarking / Safe RL / XAI Validation / Reproducible ML  

### 🔒 Core Scientific & Integrity Constraints
#### Data & Evaluation Protocol
- **NEVER** access, leak, or optimize against the held-out test split outside the frozen `src/evaluate.py` pipeline.
- Patient-disjoint splits (`nih_patient_id`) are immutable. Re-sampling, bootstrapping, or permutation must strictly operate at the **patient-cluster level**, never image-level.
- Thresholds, operating points, and DCA curves must be computed programmatically. Manual table editing, cherry-picking seeds, or post-hoc result filtering are strictly prohibited.

#### Artifact & Reproducibility Integrity
- Raw DICOMs, trained checkpoints, and `.pkl` normalization stats must remain `.gitignore`d.
- All configs, prediction bundles, result CSVs, and summary JSONs must be version-controlled with embedded SHA-256 hashes.
- Any change to a config, evaluator, or statistical module requires re-running the pipeline and committing the regenerated artifacts with updated hashes.

#### Methodological Rigor
- Threshold calibration must optimize a **cost-weighted patient-cluster F1** with explicit `β` sensitivity analysis. Fixed thresholds or validation-max-F1 alone are unacceptable for Q1.
- XAI claims (Grad-CAM) are invalid without **Adebayo-style sanity checks** (parameter randomization, data shuffling). Failure rates must be quantified and reported.
- Robustness claims must include at least one **physics-aware acquisition shift** (VOI windowing, Poisson noise, or reconstruction kernel) alongside digital corruptions.

### 📁 File Restrictions
```gitignore
# Critical ignores (DO NOT REMOVE)
data/raw/
data/processed/
results/checkpoints/
results/logs/
*.pt
*.pth
*.pkl
.env
__pycache__/
*.pyc
venv/
```

### 🔀 Branch Management (Research GitFlow)
| Type | Pattern | Example |
|------|---------|---------|
| Statistical/Calibration | `feat/stats-<description>` | `feat/stats-patient-cluster-threshold` |
| Explainability/XAI | `feat/xai-<description>` | `feat/xai-sanity-checks` |
| Robustness/Shift | `feat/robustness-<description>` | `feat/robustness-acquisition-voi` |
| Clinical/DCA | `feat/clinical-<description>` | `feat/clinical-net-benefit-curves` |
| Manuscript/Docs | `docs/<section>` | `docs/methodology-calibration` |
| Bugfix/Reproducibility | `fix/<issue>` | `fix/coco-evaluator-precision-drift` |

**Workflow Rules:**
- All branches fork from `develop`.
- PRs require methodological review + CI green status.
- `main` contains only manuscript-ready code, frozen artifacts, and publication LaTeX.

### 📝 Documentation & Verification Requirements
#### File-Centric Documentation Principle
Every modified or newly created `.py` module, config, or statistical routine must have a corresponding `.md` file in `docs/`.

| File/Folder | Purpose | Update Trigger |
|-------------|---------|----------------|
| `docs/<module>.md` | Mathematical formulation, config schema, validation protocol, statistical assumptions | New module or methodological change |
| `docs/REPORTING_CHECKLIST.md` | CLAIM/TRIPOD-AI compliance mapping, reproducibility statement, missingness policy | Manuscript prep / final freeze |
| `README.md` | Reproduction chain, artifact index, environment contract | Pipeline or config changes |
| `RULES.md` / `AGENT_CONSTRAINTS.md` | Workflow & integrity constraints | (Static, unless protocol evolves) |

#### Task Verification & Proof of Work
When a methodological upgrade is marked complete, it must include:
- **Expected Result:** Clear description of the mathematical or statistical outcome (e.g., "`τ*` selected at lower 95% CI bound of `F1_β` across 2000 patient-cluster resamples").
- **Verification Artifact:** Evidence of successful implementation:
  - **Code/Stats:** Unit tests + integration tests passing (`pytest -v`), hash-matched CSV/JSON outputs, convergence plots.
  - **Manuscript:** Regenerated tables/figures from code, DCA curves with confidence bands, sanity check failure tables.
- If unable to verify programmatically, the task MUST be tagged `⚠️ Requires Methodological Review`.

#### Checklist for Task Completion:
- [ ] Mathematical formulation & config schema documented in `docs/`
- [ ] Unit + integration tests added & passing
- [ ] Pipeline regeneration produces hash-matched artifacts
- [ ] Corresponding `report/report.md` section updated with results & limitations
- [ ] CLAIM/TRIPOD-AI checklist updated if applicable

### 🗣️ Communication & Collaboration Protocol
#### Progress Reporting Format
When reporting methodological completion:
```
[PHASE-X] <Module Name> - ✅ Complete / 🔄 In Progress / ⚠️ Blocked
📁 Documentation:
- docs/<module>.md updated.
- report/report.md section <#> aligned.
📊 Verification (Proof of Work):
- pytest -q: 100% pass | artifact hashes: verified | LaTeX: compiled
- [Table/Figure/Plot path] OR [⚠️ Requires Methodological Review]
🔗 References:
- Branch: feat/stats-<name>
- Config: configs/<name>.yaml
⚠️ Blockers / Decisions Needed:
- [If any: describe statistical assumption, dataset limitation, or reviewer concern]
```

#### Team Communication
- All statistical choices (β values, bootstrap resamples, Holm families, conditional metric policies) must be pre-declared or explicitly justified in `docs/`.
- PRs must include a diff of regenerated CSV/JSON artifacts to prove computational reproducibility.
- Manual result adjustment, seed hunting, or post-hoc threshold tuning without re-running the full pipeline is strictly prohibited.

### 🚨 Critical Constraints Summary
| Constraint | Priority | Enforcement Point |
|------------|----------|-------------------|
| Patient-disjoint & cluster-aware inference only | 🔴 CRITICAL | `src/stats/`, `src/evaluate.py`, CI checks |
| No test-set leakage or post-hoc cherry-picking | 🔴 CRITICAL | `src/evaluate.py`, artifact hash validation |
| XAI sanity checks mandatory before claims | 🔴 CRITICAL | `src/explainability/`, `docs/xai_validation.md` |
| Threshold calibration uses cost-weighted CI optimization | 🔴 CRITICAL | `src/stats/threshold_calibration.py` |
| DCA & clinical net benefit computed programmatically | 🟠 HIGH | `src/clinical/dca.py`, manuscript Section 7 |
| Physics-aware robustness shift included | 🟠 HIGH | `src/robustness/acquisition_shifts.py` |
| CLAIM/TRIPOD-AI checklist & reproducibility statement | 🟡 MEDIUM | `docs/REPORTING_CHECKLIST.md`, PR review |

---

These files are now fully aligned with your Q1 research pipeline, preserving the strict, professional structure of your original templates while embedding the exact methodological, statistical, and reproducibility constraints required for top-tier medical AI journals.