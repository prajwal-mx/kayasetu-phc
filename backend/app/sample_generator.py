import json
import os
import cv2
import numpy as np


def create_synthetic_skin_lesion(
    kind: str, size: int = 256
) -> np.ndarray:
    """Generates realistic macroscopic skin + lesion images for the 4 clinical benchmark scenarios."""
    rng = np.random.default_rng(42 if kind == "high" else (7 if kind == "moderate" else 19))
    # Base coastal Indian skin tone (Fitzpatrick Type V brown: RGB ~ [158, 108, 82])
    base_rgb = np.zeros((size, size, 3), dtype=np.float32)
    base_rgb[:, :, 0] = 158.0 + rng.normal(0, 3.5, (size, size))
    base_rgb[:, :, 1] = 108.0 + rng.normal(0, 3.0, (size, size))
    base_rgb[:, :, 2] = 82.0 + rng.normal(0, 3.0, (size, size))

    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2

    if kind == "high":
        # Irregular asymmetric multi-colored ulcerated melanoma/SCC-like lesion
        angle = np.arctan2(yy - cy, xx - cx)
        radius = 48.0 + 14.0 * np.sin(3 * angle) + 9.0 * np.cos(5 * angle)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist < radius
        base_rgb[mask, 0] = 65.0 + 55.0 * (np.sin(xx[mask] / 6.0) ** 2)
        base_rgb[mask, 1] = 32.0 + 20.0 * (np.cos(yy[mask] / 7.0) ** 2)
        base_rgb[mask, 2] = 28.0
        # Ulcerated erythematous core
        core = dist < 20.0
        base_rgb[core, 0] = 175.0
        base_rgb[core, 1] = 38.0
        base_rgb[core, 2] = 35.0
    elif kind == "moderate":
        # Indurated erythematous plaque / chronic non-healing sore
        dist = np.sqrt(((xx - cx) * 1.15) ** 2 + ((yy - cy) * 0.9) ** 2)
        mask = dist < 42.0
        base_rgb[mask, 0] = 168.0 + rng.normal(0, 8.0, np.sum(mask))
        base_rgb[mask, 1] = 62.0 + rng.normal(0, 6.0, np.sum(mask))
        base_rgb[mask, 2] = 52.0 + rng.normal(0, 6.0, np.sum(mask))
    elif kind == "low":
        # Symmetric benign nevus / mild superficial patch
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist < 24.0
        base_rgb[mask, 0] = 112.0
        base_rgb[mask, 1] = 74.0
        base_rgb[mask, 2] = 54.0
    elif kind == "blurry":
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist < 35.0
        base_rgb[mask] = [95.0, 60.0, 48.0]
        img_u8 = np.clip(base_rgb, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(img_u8, (31, 31), 12.0)

    # Add fine skin pore micro-texture so Laplacian sharpness passes naturally
    texture = rng.normal(0, 4.5, (size, size, 3))
    out = np.clip(base_rgb + texture, 0, 255).astype(np.uint8)
    return out


def generate_samples(out_dir: str = "data/samples") -> None:
    os.makedirs(out_dir, exist_ok=True)
    cases = [
        {
            "id": "case_high_scc_ulcer",
            "title_en": "Case 1: High-Risk Ulcerated Lesion (Farmer, 61y — Bantwal PHC)",
            "title_kn": "ಪ್ರಕರಣ 1: ತೀವ್ರ ಅಪಾಯದ ಒಡೆಯುತ್ತಿರುವ ಹುಣ್ಣು (ರೈತ, 61 ವರ್ಷ)",
            "kind": "high",
            "symptoms": {
                "age": 61,
                "bleeding": True,
                "rapid_growth": True,
                "pain_ulcer": True,
                "duration_long": True,
                "diabetes_immunocomp": True,
                "fever_systemic": False,
                "itching": False,
            },
        },
        {
            "id": "case_mod_infective",
            "title_en": "Case 2: Moderate Non-Healing Plaque (Fisherwoman, 47y — Ullal PHC)",
            "title_kn": "ಪ್ರಕರಣ 2: ಮಧ್ಯಮ ಅಪಾಯದ ವಾಸಿಯಾಗದ ಚರ್ಮದ ಕಲೆ (47 ವರ್ಷ)",
            "kind": "moderate",
            "symptoms": {
                "age": 47,
                "bleeding": False,
                "rapid_growth": False,
                "pain_ulcer": True,
                "duration_long": True,
                "diabetes_immunocomp": False,
                "fever_systemic": False,
                "itching": True,
            },
        },
        {
            "id": "case_low_benign_tinea",
            "title_en": "Case 3: Low-Risk Superficial / Benign Patch (Student, 22y — Puttur PHC)",
            "title_kn": "ಪ್ರಕರಣ 3: ಕಡಿಮೆ ಅಪಾಯದ ಸಾಮಾನ್ಯ ಚರ್ಮದ ತುರಿಕೆ (22 ವರ್ಷ)",
            "kind": "low",
            "symptoms": {
                "age": 22,
                "bleeding": False,
                "rapid_growth": False,
                "pain_ulcer": False,
                "duration_long": False,
                "diabetes_immunocomp": False,
                "fever_systemic": False,
                "itching": True,
            },
        },
        {
            "id": "case_blurry_reject",
            "title_en": "Case 4: Camera Blur Quality Gate Test (Out-of-Focus Handshake Photo)",
            "title_kn": "ಪ್ರಕರಣ 4: ಮಂದವಾದ ಫೋಟೋ ಗುಣಮಟ್ಟ ಪರೀಕ್ಷೆ (ಮರು-ಫೋಟೋ ಎಚ್ಚರಿಕೆ)",
            "kind": "blurry",
            "symptoms": {
                "age": 38,
                "bleeding": False,
                "rapid_growth": False,
                "pain_ulcer": False,
                "duration_long": False,
                "diabetes_immunocomp": False,
                "fever_systemic": False,
                "itching": False,
            },
        },
    ]

    manifest = []
    for c in cases:
        rgb = create_synthetic_skin_lesion(c["kind"])
        img_filename = f"{c['id']}.png"
        img_path = os.path.join(out_dir, img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        manifest.append({
            "id": c["id"],
            "title_en": c["title_en"],
            "title_kn": c["title_kn"],
            "image_url": f"/samples/{img_filename}",
            "symptoms": c["symptoms"],
        })

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(manifest)} clinical sample cases in {out_dir}/")


if __name__ == "__main__":
    generate_samples()
