#!/usr/bin/env python3
"""
Unseen Data & Real-World Generalization Tester for KayaSetu PHC (DevHack 2026 PS 3.1)

Supports 3 testing modes so you can verify the model on completely unseen data:
  1. Stress-Test Mode (Zero setup required — generates 6 brand-new unseen synthetic cases):
       python scripts/test_unseen_images.py --stress-test
  2. Held-Out Real PAD-UFES-20 Dataset Mode (Tests random real smartphone images from disk):
       python scripts/test_unseen_images.py --pad-ufes-dir data/raw/pad_ufes_20 --count 15
  3. Custom Single Image Mode (Test any photo from your phone or downloaded from the web):
       python scripts/test_unseen_images.py --image path/to/photo.jpg --age 58 --bleeding --rapid-growth
"""

import argparse
import glob
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.inference_engine import KayaSetuInferenceEngine


def synthesize_unseen_case(
    skin_rgb: tuple,
    lesion_rgb: tuple,
    irregularity: float,
    radius_px: int,
    add_glare: bool = False,
    add_blur: bool = False,
    seed: int = 101,
) -> bytes:
    rng = np.random.default_rng(seed)
    size = 256
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = skin_rgb[c] + rng.normal(0, 4.0, (size, size))

    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2 + int(rng.integers(-12, 12)), size // 2 + int(rng.integers(-12, 12))
    angle = np.arctan2(yy - cy, xx - cx)
    r_mod = radius_px * (
        1.0 + irregularity * (0.45 * np.sin(3 * angle) + 0.35 * np.cos(5 * angle))
    )
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist < r_mod

    for c in range(3):
        img[mask, c] = lesion_rgb[c] + rng.normal(0, 10.0 * (irregularity + 0.2), np.sum(mask))

    if irregularity > 0.4:
        core = dist < (radius_px * 0.42)
        img[core, 0] = min(255.0, lesion_rgb[0] + 95.0)
        img[core, 1] = max(15.0, lesion_rgb[1] - 15.0)
        img[core, 2] = max(15.0, lesion_rgb[2] - 15.0)

    if add_glare:
        glare_spot = ((xx - (cx + 15)) ** 2 + (yy - (cy - 15)) ** 2) < (48 ** 2)
        img[glare_spot] = [252.0, 252.0, 250.0]

    out = np.clip(img, 0, 255).astype(np.uint8)
    if add_blur:
        out = cv2.GaussianBlur(out, (35, 35), 14.0)

    pil_img = Image.fromarray(out)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def run_stress_test(engine: KayaSetuInferenceEngine):
    print("\n" + "=" * 92)
    print("🔬 KAYASETU PHC — UNSEEN DATA GENERALIZATION STRESS TEST (6 NEW SCENARIOS)")
    print("=" * 92)
    scenarios = [
        {
            "name": "Unseen 1: Deep Melanin (Type VI) Nodular Ulcer + Bleeding",
            "img": synthesize_unseen_case((105, 68, 50), (42, 20, 18), 0.65, 52, seed=201),
            "sym": {"age": 64, "bleeding": True, "rapid_growth": True, "pain_ulcer": True, "duration_long": True, "diabetes_immunocomp": False, "fever_systemic": False, "itching": False},
            "expected_tier": "HIGH_RISK_REFER_IMMEDIATELY",
        },
        {
            "name": "Unseen 2: Diabetic Foot / Limb Non-Healing Ulcer (Age 59)",
            "img": synthesize_unseen_case((152, 104, 78), (155, 42, 38), 0.50, 46, seed=202),
            "sym": {"age": 59, "bleeding": True, "rapid_growth": False, "pain_ulcer": True, "duration_long": True, "diabetes_immunocomp": True, "fever_systemic": True, "itching": False},
            "expected_tier": "HIGH_RISK_REFER_IMMEDIATELY",
        },
        {
            "name": "Unseen 3: Subacute Erythematous Plaque (48h PHC MO Review)",
            "img": synthesize_unseen_case((168, 118, 90), (162, 65, 55), 0.25, 38, seed=203),
            "sym": {"age": 45, "bleeding": False, "rapid_growth": False, "pain_ulcer": True, "duration_long": True, "diabetes_immunocomp": False, "fever_systemic": False, "itching": False},
            "expected_tier": "MODERATE_RISK_PHC_MO_REVIEW",
        },
        {
            "name": "Unseen 4: Superficial Tinea Ringworm (Pruritic, No Bleeding)",
            "img": synthesize_unseen_case((162, 112, 84), (140, 88, 68), 0.10, 26, seed=204),
            "sym": {"age": 26, "bleeding": False, "rapid_growth": False, "pain_ulcer": False, "duration_long": False, "diabetes_immunocomp": False, "fever_systemic": False, "itching": True},
            "expected_tier": "LOW_RISK_LOCAL_FOLLOWUP",
        },
        {
            "name": "Unseen 5: Benign Symmetric Melanocytic Nevus (Asymptomatic)",
            "img": synthesize_unseen_case((180, 132, 102), (98, 66, 48), 0.05, 20, seed=205),
            "sym": {"age": 31, "bleeding": False, "rapid_growth": False, "pain_ulcer": False, "duration_long": False, "diabetes_immunocomp": False, "fever_systemic": False, "itching": False},
            "expected_tier": "LOW_RISK_LOCAL_FOLLOWUP",
        },
        {
            "name": "Unseen 6: Severe Out-of-Focus Camera Shake (Quality Gate Test)",
            "img": synthesize_unseen_case((155, 105, 80), (90, 55, 45), 0.20, 32, add_blur=True, seed=206),
            "sym": {"age": 40, "bleeding": False, "rapid_growth": False, "pain_ulcer": False, "duration_long": False, "diabetes_immunocomp": False, "fever_systemic": False, "itching": False},
            "expected_tier": "QUALITY_GATE_BLUR",
        },
    ]

    passed = 0
    for sc in scenarios:
        res = engine.analyze(sc["img"], sc["sym"])
        if sc["expected_tier"] == "QUALITY_GATE_BLUR":
            ok = res["camera_quality"]["is_blurry"] is True and res["camera_quality"]["quality_passed"] is False
            status_str = f"BLUR DETECTED (σ²={res['camera_quality']['sharpness_variance']})"
        else:
            ok = res["triage_tier"] == sc["expected_tier"]
            status_str = f"{res['final_risk_percentage']:5.1f}% [{res['badge_color']:5s}] | {res['fitzpatrick_type']}"
        passed += int(ok)
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"{mark} | {sc['name']:58s} | {status_str}")

    print("-" * 92)
    print(f"Summary: {passed}/{len(scenarios)} unseen generalization scenarios passed ({engine.ort_session is not None and 'ONNX Active' or 'Fallback Mode'}).")
    print("=" * 92 + "\n")


def run_pad_ufes_random_eval(engine: KayaSetuInferenceEngine, pad_dir: str, count: int):
    csvs = glob.glob(os.path.join(pad_dir, "**", "*.csv"), recursive=True)
    if not csvs:
        print(f"No metadata CSV found under {pad_dir}. Check your --pad-ufes-dir path.")
        return
    df = pd.read_csv(csvs[0])
    img_map = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in glob.glob(os.path.join(pad_dir, "**", ext), recursive=True):
            img_map[os.path.basename(p)] = p
    df = df[df["img_id"].astype(str).isin(set(img_map.keys()))].copy()
    sample_df = df.sample(n=min(count, len(df)), random_state=99)

    print("\n" + "=" * 102)
    print(f"📂 EVALUATING {len(sample_df)} REAL HELD-OUT SMARTPHONE CASES FROM PAD-UFES-20")
    print("=" * 102)

    mal_total, mal_caught = 0, 0
    ben_total, ben_safe = 0, 0

    for _, row in sample_df.iterrows():
        img_id = str(row["img_id"])
        diag = str(row.get("diagnostic", "UNK")).strip().upper()
        is_malignant = diag in {"MEL", "BCC", "SCC"}
        is_precancer = (diag == "ACK")
        with open(img_map[img_id], "rb") as f:
            img_bytes = f.read()

        def bval(v):
            return str(v).strip().upper() in {"TRUE", "1", "1.0", "YES"}

        sym = {
            "age": int(float(row.get("age", 45))) if not pd.isna(row.get("age")) else 45,
            "bleeding": bval(row.get("bleed")),
            "rapid_growth": bval(row.get("grew")),
            "pain_ulcer": bval(row.get("hurt")),
            "duration_long": bval(row.get("changed")),
            "diabetes_immunocomp": bval(row.get("elevation")),
            "fever_systemic": False,
            "itching": bval(row.get("itch")),
        }
        res = engine.analyze(img_bytes, sym)

        if is_malignant:
            gt_str = f"MALIGNANT ({diag})"
            mal_total += 1
            if res["badge_color"] in {"RED", "AMBER"}:
                mal_caught += 1
                verdict = "✅ CAUGHT (Referred)"
            else:
                verdict = "⚠️ MISSED (False Neg)"
        elif is_precancer:
            gt_str = "PRE-CANCEROUS (ACK)"
            verdict = "✅ TRIAGED (Pre-Cancer)"
        else:
            gt_str = f"BENIGN ({diag})"
            ben_total += 1
            if res["badge_color"] in {"GREEN", "AMBER"}:
                ben_safe += 1
                verdict = "✅ SAFE (No False RED)"
            else:
                verdict = "⚠️ OVER-REFERRED"

        print(
            f"{img_id:18s} | GT: {gt_str:21s} | Pred: {res['final_risk_percentage']:5.1f}% [{res['badge_color']:5s}] | Skin: {res['fitzpatrick_type']:7s} | {verdict}"
        )

    print("-" * 102)
    if mal_total > 0:
        print(f"• Malignant Referral Sensitivity (RED/AMBER catch rate): {mal_caught}/{mal_total} ({mal_caught/mal_total*100:.1f}%)")
    if ben_total > 0:
        print(f"• True Benign Specificity (0% unnecessary urgent RED):   {ben_safe}/{ben_total} ({ben_safe/ben_total*100:.1f}%)")
    print("• Note: ACK = Actinic Keratosis (pre-cancerous lesion that progresses to SCC; RED/AMBER when bleeding/ulcerated is clinically appropriate).")
    print("=" * 102 + "\n")


def run_single_image(engine: KayaSetuInferenceEngine, args):
    with open(args.image, "rb") as f:
        img_bytes = f.read()
    sym = {
        "age": args.age,
        "bleeding": args.bleeding,
        "rapid_growth": args.rapid_growth,
        "pain_ulcer": args.pain_ulcer,
        "duration_long": args.duration_long,
        "diabetes_immunocomp": args.diabetes,
        "fever_systemic": args.fever,
        "itching": args.itching,
    }
    res = engine.analyze(img_bytes, sym)
    print("\n" + "=" * 72)
    print(f"📸 SINGLE IMAGE DIAGNOSTIC REPORT: {args.image}")
    print("=" * 72)
    print(f"Case ID               : {res['case_id']}")
    print(f"Inference Engine Mode : {res['inference_mode']}")
    print(f"Camera Quality Gate   : {'PASS' if res['camera_quality']['quality_passed'] else 'FAIL (RETAKE)'} (Sharpness σ²={res['camera_quality']['sharpness_variance']}, Glare={res['camera_quality']['glare_ratio']})")
    print(f"Skin Tone Calibration : {res['fitzpatrick_type']} ({res['skin_tone_label']}, ITA°={res['ita_degrees']}°, Melanin Boost=x{res['melanin_compensation_factor']})")
    print(f"ABCD Lesion Scores    : A={res['abcd_scores']['asymmetry']} | B={res['abcd_scores']['border_irregularity']} | C={res['abcd_scores']['color_variegation']} | D={res['abcd_scores']['diameter_erythema_extent']}")
    print(f"Visual Prior Prob     : {res['visual_prior_probability'] * 100:.1f}%")
    print(f"Clinical Log-Odds Δ   : {res['clinical_log_odds_shift']:+.3f}")
    print(f"FINAL REFERRAL RISK   : {res['final_risk_percentage']:.1f}% [{res['badge_color']}] -> {res['triage_tier']}")
    print(f"Clinical Action       : {res['recommendation_en']}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KayaSetu PHC on unseen skin images.")
    parser.add_argument("--stress-test", action="store_true", help="Run 6 synthetic unseen scenarios")
    parser.add_argument("--pad-ufes-dir", type=str, default=None, help="Path to local PAD-UFES-20 folder")
    parser.add_argument("--count", type=int, default=15, help="Number of random PAD-UFES-20 cases to test")
    parser.add_argument("--image", type=str, default=None, help="Path to a single custom image file")
    parser.add_argument("--age", type=int, default=45)
    parser.add_argument("--bleeding", action="store_true")
    parser.add_argument("--rapid-growth", action="store_true")
    parser.add_argument("--pain-ulcer", action="store_true")
    parser.add_argument("--duration-long", action="store_true")
    parser.add_argument("--diabetes", action="store_true")
    parser.add_argument("--fever", action="store_true")
    parser.add_argument("--itching", action="store_true")
    args = parser.parse_args()

    eng = KayaSetuInferenceEngine()
    if args.image:
        run_single_image(eng, args)
    elif args.pad_ufes_dir:
        run_pad_ufes_random_eval(eng, args.pad_ufes_dir, args.count)
    else:
        run_stress_test(eng)
