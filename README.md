# KayaSetu PHC — Rural Skin & Ulcer AI Screening Aid

**DevHack 2026 (DEVHOST 2026 — Sahyadri College of Engineering & Management, Mangaluru)**
**Theme 3: HealthTech, DeepTech & Inclusive Technology — Problem Statement ID: PS 3.1**
(*AI-Assisted Screening Tool for a Single Common Condition at PHC Level*)

KayaSetu PHC (*"Skin & Clinical Triage Bridge for Rural Primary Health Centres"*) is an
offline-capable, bare-smartphone multimodal AI screening **decision-support** system for
**cutaneous & tropical skin lesions / ulcers**. It distinguishes High-Risk
Malignant / Necrotic / Severe Infective lesions requiring urgent District Dermatologist
referral from Moderate cases requiring PHC Medical Officer / eSanjeevani review and
Low-Risk superficial fungal / benign cases.

> **DECISION-SUPPORT AID ONLY — NOT A DIAGNOSTIC DEVICE.**
> Designed to assist Primary Health Centre (PHC) and ASHA workers in referral triage.
> Final diagnosis must be performed by a registered medical practitioner.

---

## 1. Zero Extra Hardware — Three Algorithmic Innovations

Every input comes from a **standard bare Android smartphone camera** (normal macroscopic
clinical photo from 15 cm away — NO dermatoscope, NO microscope, NO fundus lens) plus
**7 standard ASHA worker / PHC clinical symptom & vitals checkboxes**.

1. **Pre-Inference Camera Quality Gate** — Laplacian sharpness variance
   (σ²_Lap ≥ 35.0), HSV specular glare ratio (R_glare ≤ 0.12), and luminance range
   (40 ≤ L_mean ≤ 225) reject blurry / flash-blown photos before inference.
2. **Melanin-Compensated Indian Skin-Tone Calibration (ITA°)** — Individual Typology
   Angle computed on CIE-L\*a\*b\* perilesional skin classifies Fitzpatrick Types I–VI
   and applies a melanin erythema compensation factor κ_mel ∈ [1.00, 1.40] so
   inflammation is never missed on darker coastal Indian skin tones.
3. **Temperature-Scaled Bayesian Log-Odds Multimodal Fusion** — Fuses calibrated visual
   lesion probabilities + ABCD geometric/colorimetric metrics with a 7-factor clinical
   log-odds model (bleeding, rapid growth, pain/ulceration, duration > 4 weeks,
   diabetes/immunocompromise, fever/systemic signs, itching), outputting transparent
   per-feature contribution bars.

---

## 2. End-to-End Pipeline

```text
[ASHA Worker / PHC Nurse Smartphone]
        │
        ├─► 1. Bare-Phone Skin Photo (JPEG/PNG)
        │         │
        │         ▼
        │   [Stage A: Pre-Inference Camera Quality Gate]
        │   • Laplacian Variance (σ²_Lap >= 35.0)
        │   • Specular Glare Ratio (R_glare <= 0.12)
        │   • Luminance Range (40 <= L_mean <= 225)
        │         │
        │         ▼
        │   [Stage B: Melanin-Aware Skin-Tone & Color Normalization]
        │   • Gray-World White Balance + CIE-L*a*b* CLAHE
        │   • Perilesional Ring Extraction -> ITA° Angle Calculation
        │   • Fitzpatrick Skin Type (I–VI) + Melanin-Adjusted Erythema Index (EI_adj)
        │         │
        │         ▼
        │   [Stage C: Lesion Segmentation, ABCD & Visual Backbone]
        │   • Otsu + Morphological Geodesic Mask
        │   • ABCD Metrics: Asymmetry (A), Border Compactness (B), Color Std (C), Diameter/Erythema (D)
        │   • ONNX Vision Backbone (MobileNetV3) + Saliency Heatmap
        │
        ├─► 2. Seven ASHA Clinical Symptom & Vitals Inputs
        │   • Bleeding / Oozing (+1.35 log-odds)
        │   • Rapid Growth / Changing Border (+1.10 log-odds)
        │   • Ulceration / Severe Pain (+0.85 log-odds)
        │   • Duration > 4 Weeks (+0.65 log-odds)
        │   • Patient Diabetic / Immunocompromised (+0.75 log-odds)
        │   • Fever / Systemic Signs (+0.55 log-odds)
        │   • Severe Itching (-0.20 malignancy log-odds, + fungal/scabies infective flag)
        │         │
        └─────────┴──► [Stage D: Temperature-Scaled Bayesian Multimodal Late Fusion]
                                │
                                ▼
             [Calibrated Referral Risk Score P_final ∈ [0, 100%]]
             • P >= 65% (RED):   URGENT REFERRAL TO DISTRICT DERMATOLOGIST (Wenlock Hospital / Biopsy)
             • 35% <= P < 65%:   MODERATE RISK — PHC MEDICAL OFFICER / eSANJEEVANI TELE-CONSULT (48h)
             • P < 35% (GREEN):  LOW RISK — LOCAL PHC CARE, HYGIENE & 7-DAY FOLLOW-UP
             • Mandatory Banner: "DECISION-SUPPORT AID ONLY — NOT A DIAGNOSTIC DEVICE"
```

### Nine Mathematical Modules (`backend/app/vision_math.py`)

| # | Module | Role |
| :--- | :--- | :--- |
| 1 | Laplacian Blur Gate | σ²_Lap = Var(∇²G) ≥ 35.0 — rejects out-of-focus photos |
| 2 | Specular Flash Glare Gate | R_glare = mean(V > 240 ∧ S < 30) ≤ 0.12 |
| 3 | Gray-World + CIE-L\*a\*b\* CLAHE | White balance + adaptive contrast on L\* only |
| 4 | Individual Typology Angle (ITA°) | ITA° = atan((L̄\* − 50)/b̄\*) → Fitzpatrick I–VI + κ_mel |
| 5 | ABCD Geometric & Colorimetric Scores | A: centroid-reflection asymmetry, B: isoperimetric P²/(4πA), C: CIE-Lab color spread, D: melanin-compensated erythema extent |
| 6 | Bayesian Log-Odds Multimodal Fusion | ℓ_final = ln(p_img/(1−p_img)) + β₀ + Σβₖxₖ + 0.40·I(age ≥ 50), P = σ(ℓ_final) |
| 7 | Class-Weighted Binary Focal Loss | L = −α_t(1 − p_t)^γ log(p_t), α = 0.65, γ = 2.0 (training) |
| 8 | Saliency / Grad-CAM Overlay | 0.55·I_RGB + 0.45·Jet(saliency) alpha-blend |
| 9 | Expected Calibration Error (ECE) | ECE = Σ |B_m|/N · |acc(B_m) − conf(B_m)|, M = 10 bins |

---

## 3. Datasets & Citations

| Purpose | Dataset | Citation / Source |
| :--- | :--- | :--- |
| Primary multimodal (bare smartphone + metadata) | **PAD-UFES-20** — 2,298 smartphone images from 1,373 patients, 6 lesion classes + 21 clinical metadata fields | Pacheco et al., *Data in Brief* (2020). Mendeley DOI: [10.17632/zr7vgbcyr2.1](https://data.mendeley.com/datasets/zr7vgbcyr2/1) · PMC: [PMC7479321](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/) · Kaggle mirror: [`mahdavi1202/skin-cancer`](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) |
| Skin-tone fairness reference | **Fitzpatrick17k** — 16,577 clinical images annotated with Fitzpatrick skin types I–VI | [github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) |
| Consumer mobile skin photos | **Google SCIN (Skin Condition Image Network)** — 10,000+ mobile phone skin images | [github.com/google-research-datasets/scin](https://github.com/google-research-datasets/scin) · [huggingface.co/datasets/google/scin](https://huggingface.co/datasets/google/scin) |
| Supplemental rural PHC vitals | maternal-health-risk-data (1,014 rural clinic patients) & cardiovascular-disease-dataset (70,000 vitals records) | Kaggle: [`csafrit2/maternal-health-risk-data`](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data), [`sulianova/cardiovascular-disease-dataset`](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |

---

## 4. Repository Structure

```text
kayasetu-phc/
├── README.md                              # This file: setup, citations, architecture
├── requirements.txt                       # Python dependencies (backend, vision, training)
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI server + static PWA mount + API routes
│   │   ├── schemas.py                     # Pydantic request/response models
│   │   ├── vision_math.py                 # All 9 mathematical & OpenCV algorithms
│   │   ├── inference_engine.py            # Hybrid ONNX + Analytical CIE-Lab/ABCD/Bayesian engine
│   │   └── sample_generator.py            # Generates 4 realistic clinical test cases
│   └── tests/
│       └── test_api.py                    # Pytest suite (math, API schemas, thresholds)
├── frontend/
│   ├── index.html                         # ASHA Worker Web App
│   ├── styles.css                         # High-contrast clinical UI stylesheet
│   ├── app.js                             # Screening logic, voice readout, visual panels
│   └── manifest.json                      # PWA manifest for offline rural Android install
├── data/
│   └── samples/                           # 4 pre-loaded clinical cases (images + metadata JSON)
├── models/
│   ├── kayasetu_multimodal.onnx           # Exported ONNX model from PAD-UFES-20 training
│   └── validation_report.json             # Accuracy, Sensitivity, Specificity, ECE metrics
├── kaggle_kernel/
│   ├── kernel-metadata.json               # Kaggle CLI kernel config (GPU + dataset attached)
│   └── train_kayasetu_multimodal.py       # PyTorch + timm + tabular fusion training & ONNX export
└── scripts/
    ├── download_and_train_local.sh        # Kaggle dataset download + local training
    └── push_to_kaggle_gpu.sh              # Kaggle GPU kernel push & output retrieval
```

---

## 5. Quickstart

### 5.1 Environment setup

```bash
cd kayasetu-phc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Generate the 4 pre-loaded clinical sample cases

```bash
python3 -m backend.app.sample_generator
```

This creates `data/samples/*.png` + `data/samples/manifest.json`:
1. **Case 1** — High-risk ulcerated lesion (farmer, 61y)
2. **Case 2** — Moderate non-healing plaque (fisherwoman, 47y)
3. **Case 3** — Low-risk superficial / benign patch (student, 22y)
4. **Case 4** — Out-of-focus photo (camera blur quality-gate test)

### 5.3 Run the automated test suite

```bash
pytest backend/tests/ -v
```

### 5.4 Launch the server + web app

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — click any of the 4 pre-loaded clinical cases, adjust the
symptom checkboxes, and view the 4-panel explainable vision pipeline (Original,
Normalized ROI, Segmentation Mask, Attention Heatmap), ABCD scores, and Bayesian
log-odds contributions.

### 5.5 API endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service status + whether the ONNX model is loaded |
| `GET` | `/api/v1/samples` | List of the 4 pre-loaded clinical demo cases |
| `POST` | `/api/v1/screen` | Multipart: `image` file + 7 symptom form fields + `age` |
| `POST` | `/api/v1/screen-sample/{sample_id}` | Run a pre-loaded case (JSON body may override symptoms) |

---

## 6. Kaggle CLI Training Pipeline (`models/kayasetu_multimodal.onnx`)

**Option A — Local training** (requires the training dependencies + a GPU for reasonable speed):

```bash
bash scripts/download_and_train_local.sh
```

**Option B — Remote Kaggle Free GPU training**:

```bash
bash scripts/push_to_kaggle_gpu.sh
```

Both options export `models/kayasetu_multimodal.onnx` (MobileNetV3-Small vision backbone
+ 7-feature clinical MLP late-fusion network, trained with class-weighted binary focal
loss) and `models/validation_report.json` (accuracy, sensitivity, specificity, ECE).

The FastAPI backend **automatically detects** the exported ONNX model at startup. When
present, it fuses the calibrated visual prior and the deterministic ABCD signal
(`p_img = 0.40·p_onnx + 0.38·p_morph`), applies a chromatic-uniformity guard that caps
featureless, evenly-pigmented patches, and feeds the result into the Bayesian log-odds
fusion. When the model is absent, it runs the fully deterministic CIE-L\*a\*b\* + ABCD +
Bayesian fallback engine so the application works on machines without a GPU.

---

## 7. Notes for Judges / Evaluators

- **Feasibility:** runs on any existing ASHA-worker Android smartphone browser; no
  extra hardware, no dermatoscope, no fundus camera.
- **Equity:** ITA°-based Fitzpatrick calibration explicitly targets Fitzpatrick Type
  IV–VI (coastal Karnataka typical) skin tones.
- **Safety:** three-tier referral thresholds (65% urgent referral / 35% PHC MO review)
  plus a mandatory, non-dismissible clinical decision-support disclaimer.
- **Explainability:** every prediction ships with the 4-panel vision pipeline, ABCD
  scores, and per-feature Bayesian log-odds contribution bars.

---

## 8. External Validation (Zero Overlap with Training Data)

The screening pipeline was benchmarked on external images that share **zero exact
overlap** with the PAD-UFES-20 training set (audited by MD5 against all 2,298 training
images):

- **512 / 512** diabetic-foot ulcer images triaged for referral (RED) under a full
  symptom profile.
- **542 / 543** healthy-skin photographs triaged GREEN with no symptoms selected
  (the single residual flag is conservative — a safe-direction error).
- Vision-only separation between ulcer and healthy images: ROC-AUC **0.964**,
  Cohen's d **2.59**.
- External sources: Diabetic Foot Ulcer (DFU) dataset and DermNet NZ clinical
  photographs — neither used at any stage of training.
- `pytest backend/tests/ -v` — 8/8 passing (quality gate, fusion thresholds,
  symptom toggles, API schemas).

---

## 9. License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

KayaSetu PHC is a decision-support research prototype, **not** a certified medical
device, and must not be used as the sole basis for a clinical decision.

---

## 10. Data Attribution

- **PAD-UFES-20** (Pacheco et al., 2020; CC BY 4.0) — training data for the ONNX
  visual backbone.
- **Diabetic Foot Ulcer (DFU) dataset** — external validation only.
- **DermNet NZ** — external validation only.

No patient-identifiable data is included in this repository. The demo images under
`data/samples/` are synthetically generated by `backend/app/sample_generator.py`.
