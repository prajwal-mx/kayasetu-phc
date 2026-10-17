from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SymptomInput(BaseModel):
    age: int = Field(default=42, ge=1, le=110, description="Patient age in years")
    bleeding: bool = Field(default=False, description="Spontaneous bleeding or oozing from lesion")
    rapid_growth: bool = Field(default=False, description="Lesion growing rapidly or changing border")
    pain_ulcer: bool = Field(default=False, description="Pain, tenderness, or open ulceration")
    duration_long: bool = Field(default=False, description="Present for more than 4 weeks without healing")
    diabetes_immunocomp: bool = Field(default=False, description="Known diabetes or high blood sugar")
    fever_systemic: bool = Field(default=False, description="Fever or swollen regional lymph nodes")
    itching: bool = Field(default=False, description="Significant itching / pruritus")


class CameraQualityOutput(BaseModel):
    quality_passed: bool
    sharpness_variance: float
    is_blurry: bool
    glare_ratio: float
    has_glare: bool
    mean_luminance: float
    lighting_status: str


class AbcdScoresOutput(BaseModel):
    asymmetry: float
    border_irregularity: float
    color_variegation: float
    diameter_erythema_extent: float


class FeatureContribution(BaseModel):
    feature: str
    active: bool
    log_odds_delta: float
    impact_direction: str


class ScreeningResponse(BaseModel):
    case_id: str
    inference_mode: str
    camera_quality: CameraQualityOutput
    ita_degrees: float
    fitzpatrick_type: str
    skin_tone_label: str
    melanin_compensation_factor: float
    adjusted_erythema_index: float
    abcd_scores: AbcdScoresOutput
    visual_prior_probability: float
    clinical_log_odds_shift: float
    final_risk_probability: float
    final_risk_percentage: float
    referral_threshold_percentage: float
    review_threshold_percentage: float
    refer_to_specialist: bool
    triage_tier: str
    badge_color: str
    suspected_superficial_fungal_or_scabies: bool
    recommendation_en: str
    recommendation_kn: str
    feature_contributions: List[FeatureContribution]
    images_base64: Dict[str, str]
    disclaimer: str
