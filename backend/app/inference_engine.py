import math
import os
import uuid
from pathlib import Path
from typing import Dict, Any
import numpy as np

from backend.app.vision_math import (
    decode_image_to_rgb,
    encode_rgb_to_base64_png,
    evaluate_camera_quality_gate,
    normalize_illumination_and_clahe,
    segment_lesion_and_skin_tone,
    fuse_multimodal_bayesian_risk,
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class KayaSetuInferenceEngine:
    """
    Runs the full 4-stage pipeline:
      1. Laplacian Blur + HSV Specular Glare Camera Quality Gate
      2. Gray-World + CIE-L*a*b* CLAHE + Fitzpatrick IV-VI ITA° Skin-Tone Calibration
      3. Lesion Segmentation + ABCD Metrics + Temperature-Calibrated ONNX Visual Prior
      4. Single-Pass Bayesian Log-Odds Multimodal Fusion + Referral Triage
    """

    def __init__(self, onnx_path: str = None):
        if onnx_path is None:
            onnx_path = str(PROJECT_ROOT / "models" / "kayasetu_multimodal.onnx")
        self.onnx_path = onnx_path
        self.ort_session = None
        if ort is not None and os.path.exists(self.onnx_path):
            try:
                self.ort_session = ort.InferenceSession(
                    self.onnx_path, providers=["CPUExecutionProvider"]
                )
            except Exception:
                self.ort_session = None

    def _run_onnx_visual_prior(self, norm_rgb: np.ndarray) -> float:
        """
        Extracts the pure visual prior probability from the MobileNetV3 ONNX backbone
        by passing a neutral marginal clinical vector, preventing double-counting of
        clinical symptoms when fuse_multimodal_bayesian_risk() is applied downstream.
        """
        if self.ort_session is None:
            return -1.0
        try:
            import cv2
            img_224 = cv2.resize(norm_rgb, (224, 224)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_224 - mean) / std
            img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

            # Neutral baseline clinical expectation so ONNX evaluates visual appearance only
            clin_neutral = np.array(
                [[0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.50]], dtype=np.float32
            )

            outputs = self.ort_session.run(
                None, {"image": img_chw, "clinical_features": clin_neutral}
            )
            logit_vis = float(outputs[0].squeeze())
            # Temperature-calibrated visual probability (removes marginal intercept shift)
            return 1.0 / (1.0 + math.exp(-(logit_vis + 2.30) / 1.40))
        except Exception:
            return -1.0

    def analyze(self, image_bytes: bytes, symptoms: Dict[str, Any]) -> Dict[str, Any]:
        rgb_arr = decode_image_to_rgb(image_bytes, target_size=(256, 256))
        cam_gate = evaluate_camera_quality_gate(rgb_arr)
        glare_mask = cam_gate.pop("_glare_mask")

        norm_rgb, lab_cie = normalize_illumination_and_clahe(rgb_arr)
        seg_data = segment_lesion_and_skin_tone(norm_rgb, lab_cie, glare_mask)

        p_morph = seg_data["morphological_probability"]
        p_onnx_vis = self._run_onnx_visual_prior(norm_rgb)
        if p_onnx_vis >= 0.0:
            p_img = 0.40 * p_onnx_vis + 0.38 * p_morph
            # Chromatic Uniformity Guard: uniform healthy skin patches (featureless low-ABCD
            # color/border) must not be inflated above 0.34 by CNN shadow artifacts.
            c_score = seg_data["abcd_scores"]["color_variegation"]
            b_score = seg_data["abcd_scores"]["border_irregularity"]
            if c_score < 0.24 and b_score < 0.25:
                p_img = min(p_img, 0.34)
            mode = "ONNX_Deep_Multimodal_Plus_CIE_Lab_ABCD"
        else:
            p_img = p_morph
            mode = "Deterministic_CIE_Lab_ABCD_Bayesian_Engine"

        fusion = fuse_multimodal_bayesian_risk(p_img, symptoms)
        visuals = seg_data["visuals"]

        return {
            "case_id": f"PHC-{uuid.uuid4().hex[:8].upper()}",
            "inference_mode": mode,
            "camera_quality": cam_gate,
            "ita_degrees": seg_data["ita_degrees"],
            "fitzpatrick_type": seg_data["fitzpatrick_type"],
            "skin_tone_label": seg_data["skin_tone_label"],
            "melanin_compensation_factor": seg_data["melanin_compensation_factor"],
            "adjusted_erythema_index": seg_data["adjusted_erythema_index"],
            "abcd_scores": seg_data["abcd_scores"],
            **fusion,
            "images_base64": {
                "original": encode_rgb_to_base64_png(rgb_arr),
                "normalized": encode_rgb_to_base64_png(visuals["normalized_rgb"]),
                "segmentation": encode_rgb_to_base64_png(visuals["segmentation_rgb"]),
                "heatmap": encode_rgb_to_base64_png(visuals["heatmap_rgb"]),
            },
        }
