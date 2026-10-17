import base64
import io
import math
from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image


def decode_image_to_rgb(image_bytes: bytes, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Decodes raw image bytes into a resized RGB uint8 numpy array (H, W, 3)."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def encode_rgb_to_base64_png(rgb_arr: np.ndarray) -> str:
    """Encodes an RGB uint8 numpy array to a data:image/png;base64 URI."""
    pil_img = Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def evaluate_camera_quality_gate(rgb_arr: np.ndarray) -> Dict[str, Any]:
    """
    Formulas 1 & 2: Computes Laplacian sharpness variance, HSV specular glare ratio,
    and ambient luminance status.
    """
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_var = float(np.var(laplacian))

    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)

    glare_mask = (v_channel > 240.0) & (s_channel < 30.0)
    glare_ratio = float(np.mean(glare_mask))
    mean_luminance = float(np.mean(v_channel))

    is_blurry = sharpness_var < 35.0
    has_glare = glare_ratio > 0.12
    if mean_luminance < 45.0:
        lighting_status = "TOO_DARK"
    elif mean_luminance > 235.0:
        lighting_status = "OVEREXPOSED"
    else:
        lighting_status = "OPTIMAL"

    quality_passed = (not is_blurry) and (not has_glare) and (lighting_status == "OPTIMAL")
    return {
        "quality_passed": bool(quality_passed),
        "sharpness_variance": round(sharpness_var, 2),
        "is_blurry": bool(is_blurry),
        "glare_ratio": round(glare_ratio, 4),
        "has_glare": bool(has_glare),
        "mean_luminance": round(mean_luminance, 2),
        "lighting_status": lighting_status,
        "_glare_mask": glare_mask,
    }


def normalize_illumination_and_clahe(rgb_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Formula 3: Gray-World illuminant white balance + CIE-L*a*b* CLAHE on L* channel.
    Note: Also preserves un-bleached CIE-L*a*b* channels in lab_cie[:, :, 3:6] if needed,
    or computes mild illuminant scaling so natural skin melanin chromaticity (b*) is not erased.
    """
    img_f = rgb_arr.astype(np.float32)
    means = np.mean(img_f, axis=(0, 1))
    global_mean = float(np.mean(means))
    # Damped Gray-World (35% correction) removes extreme color casts without bleaching skin melanin to gray
    raw_scale = global_mean / (means + 1e-6)
    damped_scale = 0.65 + 0.35 * raw_scale
    gw_rgb = np.clip(img_f * damped_scale, 0, 255).astype(np.uint8)

    lab_cv = cv2.cvtColor(gw_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab_cv)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan)
    lab_enhanced_cv = cv2.merge([l_clahe, a_chan, b_chan])
    norm_rgb = cv2.cvtColor(lab_enhanced_cv, cv2.COLOR_LAB2RGB)

    # Use raw un-equalized L* and b* from original RGB for authentic ITA° skin-tone photometry,
    # while keeping CLAHE-enhanced contrast for lesion segmentation
    raw_lab_cv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    l_cie = raw_lab_cv[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    a_cie = raw_lab_cv[:, :, 1].astype(np.float32) - 128.0
    b_cie = raw_lab_cv[:, :, 2].astype(np.float32) - 128.0
    lab_cie = np.stack([l_cie, a_cie, b_cie], axis=-1)
    return norm_rgb, lab_cie


def segment_lesion_and_skin_tone(
    norm_rgb: np.ndarray, lab_cie: np.ndarray, glare_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Formulas 4 & 5: Segments lesion mask M, computes perilesional Individual Typology Angle (ITA°),
    maps to Fitzpatrick Skin Type (I-VI), and extracts ABCD geometric & colorimetric scores.
    """
    h, w, _ = lab_cie.shape
    l_cie = lab_cie[:, :, 0]
    a_cie = lab_cie[:, :, 1]
    b_cie = lab_cie[:, :, 2]

    # Spatial center prior (Gaussian centered on image)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    sigma_spatial = 0.38 * min(h, w)
    center_prior = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_spatial ** 2))

    # Multi-cue lesion salience (darkening + erythema)
    darkness = np.clip(100.0 - l_cie, 0.0, 100.0)
    erythema_raw = np.clip(a_cie, 0.0, 60.0)
    salience = center_prior * (0.65 * darkness + 0.35 * erythema_raw * 1.5)
    salience_u8 = np.clip((salience / (np.max(salience) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)

    _, otsu_mask = cv2.threshold(salience_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        lesion_mask = (labels == largest_idx)
    else:
        lesion_mask = (salience_u8 > np.percentile(salience_u8, 70))

    # Ensure minimum reasonable mask coverage
    area_px = int(np.sum(lesion_mask))
    if area_px < int(0.02 * h * w) or area_px > int(0.80 * h * w):
        lesion_mask = (center_prior > 0.55) & (salience_u8 > np.median(salience_u8))
        area_px = max(1, int(np.sum(lesion_mask)))

    # Perilesional healthy skin mask (exclude lesion & glare)
    skin_mask = (~lesion_mask) & (~glare_mask)
    if np.sum(skin_mask) < 100:
        skin_mask = ~lesion_mask

    # Formula 4: Authentic ITA° on un-bleached surrounding healthy skin
    l_skin = float(np.mean(l_cie[skin_mask]))
    a_skin = float(np.mean(a_cie[skin_mask]))
    b_skin = float(np.mean(b_cie[skin_mask]))
    ita_deg = float(math.atan2(l_skin - 50.0, max(1.0, abs(b_skin))) * (180.0 / math.pi))

    if ita_deg > 55.0:
        fitzpatrick, skin_label, kappa_mel = "Type I", "Very Light", 1.00
    elif ita_deg > 41.0:
        fitzpatrick, skin_label, kappa_mel = "Type II", "Light", 1.02
    elif ita_deg > 28.0:
        fitzpatrick, skin_label, kappa_mel = "Type III", "Intermediate", 1.08
    elif ita_deg > 10.0:
        fitzpatrick, skin_label, kappa_mel = "Type IV", "Indian Olive / Medium", 1.16
    elif ita_deg > -30.0:
        fitzpatrick, skin_label, kappa_mel = "Type V", "Indian Brown (Coastal Typical)", 1.28
    else:
        fitzpatrick, skin_label, kappa_mel = "Type VI", "Deep Melanin / Dark Brown", 1.40

    # Melanin-compensated Erythema Index
    a_lesion = float(np.mean(a_cie[lesion_mask]))
    delta_a = max(0.0, a_lesion - a_skin)
    ei_adj = float(np.clip((delta_a * kappa_mel) / 35.0, 0.0, 1.0))

    # Formula 5: ABCD Scores
    ys, xs = np.where(lesion_mask)
    m_cy, m_cx = int(np.mean(ys)), int(np.mean(xs))
    shift_y, shift_x = (h // 2) - m_cy, (w // 2) - m_cx
    M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_mask = cv2.warpAffine(
        lesion_mask.astype(np.uint8), M_trans, (w, h), flags=cv2.INTER_NEAREST
    )
    flip_h = np.fliplr(centered_mask)
    flip_v = np.flipud(centered_mask)
    asym_raw = (
        float(np.sum(np.abs(centered_mask - flip_h)) + np.sum(np.abs(centered_mask - flip_v)))
        / (2.0 * area_px + 1e-6)
    )
    a_score = float(np.clip(asym_raw / 0.85, 0.0, 1.0))

    contours, _ = cv2.findContours(
        lesion_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 1.0
    circ_idx = (perimeter ** 2) / (4.0 * math.pi * area_px + 1e-6)
    b_score = float(np.clip((circ_idx - 1.0) / 3.5, 0.0, 1.0))

    std_l = float(np.std(l_cie[lesion_mask]))
    std_a = float(np.std(a_cie[lesion_mask]))
    std_b = float(np.std(b_cie[lesion_mask]))
    sigma_color = 0.45 * std_l + 0.30 * std_a + 0.25 * std_b
    c_score = float(np.clip(sigma_color / 22.0, 0.0, 1.0))

    area_ratio = float(area_px) / float(h * w)
    l_contrast = max(0.0, (l_skin - float(np.mean(l_cie[lesion_mask]))) / 45.0)
    d_score = float(
        np.clip(0.45 * ei_adj + 0.35 * l_contrast + 0.20 * min(1.0, area_ratio / 0.25), 0.0, 1.0)
    )

    p_morph = float(
        np.clip(0.28 * a_score + 0.24 * b_score + 0.28 * c_score + 0.20 * d_score, 0.02, 0.98)
    )

    seg_vis = norm_rgb.copy()
    cv2.drawContours(seg_vis, contours, -1, (0, 255, 180), 2)

    cam_raw = cv2.GaussianBlur(salience_u8, (21, 21), 0)
    cam_norm = cv2.normalize(cam_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cam_bgr = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
    cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
    heatmap_overlay = cv2.addWeighted(norm_rgb, 0.55, cam_rgb, 0.45, 0)

    return {
        "ita_degrees": round(ita_deg, 2),
        "fitzpatrick_type": fitzpatrick,
        "skin_tone_label": skin_label,
        "melanin_compensation_factor": round(kappa_mel, 2),
        "adjusted_erythema_index": round(ei_adj, 3),
        "abcd_scores": {
            "asymmetry": round(a_score, 3),
            "border_irregularity": round(b_score, 3),
            "color_variegation": round(c_score, 3),
            "diameter_erythema_extent": round(d_score, 3),
        },
        "morphological_probability": round(p_morph, 4),
        "visuals": {
            "normalized_rgb": norm_rgb,
            "segmentation_rgb": seg_vis,
            "heatmap_rgb": heatmap_overlay,
        },
    }


def fuse_multimodal_bayesian_risk(
    p_img: float, symptoms: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Formula 6: Bayesian Log-Odds Multimodal Late Fusion combining visual probability p_img
    with 7 clinical symptom indicators and patient age.
    """
    p_clamped = float(np.clip(p_img, 0.02, 0.98))
    log_odds_img = math.log(p_clamped / (1.0 - p_clamped))

    weights = {
        "bleeding": (1.35, "Spontaneous Bleeding / Oozing"),
        "rapid_growth": (1.10, "Rapid Growth / Changing Shape"),
        "pain_ulcer": (0.85, "Painful / Tender / Open Ulcer"),
        "diabetes_immunocomp": (0.75, "Diabetes / High Blood Sugar History"),
        "duration_long": (0.65, "Non-Healing Duration > 4 Weeks"),
        "fever_systemic": (0.55, "Fever / Regional Lymph Swelling"),
        "itching": (-0.20, "Prominent Itching (Pruritus)"),
    }

    # Visual Safety Floor with Clinical Gating: only when the user provided NO symptom
    # flags at all (itching included) may a high visual prior prevent a visually severe
    # lesion from being pulled down into GREEN (< 35%) by missing checkboxes. Itching-only
    # fungal photos (tinea) keep the standard -0.85 baseline.
    # Main floor: -0.10 at p_clamped >= 0.385: ln(0.385/0.615) - 0.10 = -0.568 -> 36.2% AMBER.
    # Age >= 50 sub-band: -0.42 at p_clamped >= 0.355 (with the +0.40 age bonus, net -0.02),
    # recovering elderly zero-checkbox ulcer catch without touching age-35 specificity.
    age = int(symptoms.get("age", 35))
    any_symptom_active = any(bool(symptoms.get(k, False)) for k in weights)
    if not any_symptom_active and p_clamped >= 0.385:
        beta_0 = -0.10
    elif not any_symptom_active and age >= 50 and p_clamped >= 0.355:
        beta_0 = -0.42
    else:
        beta_0 = -0.85
    delta_clin = beta_0
    contributions = [
        {
            "feature": "Smartphone Visual Lesion Analysis (ABCD + CIE-Lab + CNN)",
            "active": True,
            "log_odds_delta": round(log_odds_img, 3),
            "impact_direction": "risk_increase" if log_odds_img >= 0 else "risk_decrease",
        }
    ]

    for key, (w_val, label) in weights.items():
        is_active = bool(symptoms.get(key, False))
        if is_active:
            delta_clin += w_val
            contributions.append({
                "feature": label,
                "active": True,
                "log_odds_delta": round(w_val, 3),
                "impact_direction": "risk_increase" if w_val >= 0 else "infective_fungal_indicator",
            })

    age = int(symptoms.get("age", 35))
    if age >= 50:
        delta_clin += 0.40
        contributions.append({
            "feature": f"Patient Age >= 50 yrs ({age} yrs)",
            "active": True,
            "log_odds_delta": 0.40,
            "impact_direction": "risk_increase",
        })

    final_log_odds = log_odds_img + delta_clin
    p_final = 1.0 / (1.0 + math.exp(-final_log_odds))
    risk_pct = round(p_final * 100.0, 1)

    is_fungal_pattern = (
        bool(symptoms.get("itching", False))
        and not bool(symptoms.get("bleeding", False))
        and p_final < 0.55
    )

    if p_final >= 0.65:
        tier = "HIGH_RISK_REFER_IMMEDIATELY"
        badge_color = "RED"
        en_action = (
            "REFER TO SPECIALIST IMMEDIATELY: High risk of malignant/necrotic or severe deep "
            "ulcerative lesion. Refer to District Dermatologist / Wenlock Hospital Mangaluru "
            "for biopsy and specialist management."
        )
        kn_action = (
            "ತಕ್ಷಣ ತಜ್ಞ ವೈದ್ಯರಿಗೆ ಕಳುಹಿಸಿ (ತುರ್ತು ಶಿಫಾರಸು): ಗಂಭೀರ ಚರ್ಮದ ಗಾಯ/ಹುಣ್ಣು ಕಂಡುಬಂದಿದೆ. "
            "ದಯವಿಟ್ಟು ತಕ್ಷಣ ಜಿಲ್ಲಾ ಆಸ್ಪತ್ರೆ (ವೆನ್ಲಾಕ್ ಆಸ್ಪತ್ರೆ, ಮಂಗಳೂರು) ಚರ್ಮರೋಗ ತಜ್ಞರಿಗೆ ಕಳುಹಿಸಿ."
        )
    elif p_final >= 0.35:
        tier = "MODERATE_RISK_PHC_MO_REVIEW"
        badge_color = "AMBER"
        en_action = (
            "REFER TO PHC MEDICAL OFFICER / eSANJEEVANI (WITHIN 48 HOURS): Moderate clinical risk. "
            "Schedule PHC Medical Officer evaluation or upload to eSanjeevani Tele-Dermatology."
        )
        kn_action = (
            "ಪಿಎಚ್‌ಸಿ ವೈದ್ಯಾಧಿಕಾರಿಗಳ ಪರೀಕ್ಷೆ ಅಗತ್ಯ (48 ಗಂಟೆಗಳೊಳಗೆ): ಮಧ್ಯಮ ಅಪಾಯದ ಲಕ್ಷಣಗಳು. "
            "ಪಿಎಚ್‌ಸಿ ವೈದ್ಯಾಧಿಕಾರಿಗಳಿಗೆ ಅಥವಾ ಇ-ಸಂಜೀವನಿ ಟೆಲಿ-ಕನ್ಸಲ್ಟೇಶನ್‌ಗೆ ಕಳುಹಿಸಿ."
        )
    else:
        tier = "LOW_RISK_LOCAL_FOLLOWUP"
        badge_color = "GREEN"
        sub_note = " (Suspected superficial fungal tinea/scabies pattern — check hygiene/topical care)" if is_fungal_pattern else ""
        en_action = (
            f"LOW RISK — ROUTINE PHC CARE & 7-DAY FOLLOW-UP{sub_note}: Below specialist referral "
            "threshold. Provide skin hygiene counseling and re-screen in 7 days if unhealed."
        )
        kn_action = (
            "ಕಡಿಮೆ ಅಪಾಯ — ಸಾಮಾನ್ಯ ಪಿಎಚ್‌ಸಿ ಆರೈಕೆ ಮತ್ತು 7 ದಿನಗಳ ನಂತರ ಮರು-ಪರೀಕ್ಷೆ: "
            "ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ, 7 ದಿನಗಳಲ್ಲಿ ವಾಸಿಯಾಗದಿದ್ದರೆ ಮತ್ತೆ ಪರೀಕ್ಷಿಸಿ."
        )

    return {
        "visual_prior_probability": round(p_clamped, 4),
        "clinical_log_odds_shift": round(delta_clin, 3),
        "final_risk_probability": round(p_final, 4),
        "final_risk_percentage": risk_pct,
        "referral_threshold_percentage": 65.0,
        "review_threshold_percentage": 35.0,
        "refer_to_specialist": bool(p_final >= 0.65),
        "triage_tier": tier,
        "badge_color": badge_color,
        "suspected_superficial_fungal_or_scabies": is_fungal_pattern,
        "recommendation_en": en_action,
        "recommendation_kn": kn_action,
        "feature_contributions": contributions,
        "disclaimer": (
            "DECISION-SUPPORT AID ONLY — NOT A DIAGNOSTIC DEVICE. Designed to assist rural PHC "
            "and ASHA workers in referral triage. Final diagnosis must be performed by a "
            "registered medical practitioner."
        ),
    }
