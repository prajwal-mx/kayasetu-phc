from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["problem_statement"] == "PS 3.1"
    assert body["onnx_loaded"] is True


def test_model_metrics_endpoint():
    res = client.get("/api/v1/model-metrics")
    assert res.status_code == 200
    body = res.json()
    assert body["total_samples_evaluated"] == 2298
    assert body["validation_accuracy"] >= 0.80


def test_high_risk_sample_triggers_specialist_referral():
    res = client.post("/api/v1/screen-sample/case_high_scc_ulcer")
    assert res.status_code == 200
    data = res.json()
    assert data["refer_to_specialist"] is True
    assert data["final_risk_percentage"] >= 65.0
    assert data["badge_color"] == "RED"
    assert data["triage_tier"] == "HIGH_RISK_REFER_IMMEDIATELY"
    assert "NOT A DIAGNOSTIC DEVICE" in data["disclaimer"]
    assert "heatmap" in data["images_base64"]


def test_moderate_risk_sample_triggers_amber_phc_review():
    res = client.post("/api/v1/screen-sample/case_mod_infective")
    assert res.status_code == 200
    data = res.json()
    assert data["refer_to_specialist"] is False
    assert 35.0 <= data["final_risk_percentage"] < 65.0
    assert data["badge_color"] == "AMBER"
    assert data["triage_tier"] == "MODERATE_RISK_PHC_MO_REVIEW"


def test_low_risk_sample_stays_below_referral_threshold():
    res = client.post("/api/v1/screen-sample/case_low_benign_tinea")
    assert res.status_code == 200
    data = res.json()
    assert data["refer_to_specialist"] is False
    assert data["final_risk_percentage"] < 35.0
    assert data["badge_color"] == "GREEN"


def test_symptom_toggle_drops_high_risk_to_low():
    # Itching-only override keeps the standard -0.85 baseline (the Visual Safety Floor is
    # gated off whenever any symptom flag is active), so this visual 44.6% sample correctly
    # drops to the low-risk fungal tinea/scabies pathway instead of a referral.
    res = client.post(
        "/api/v1/screen-sample/case_high_scc_ulcer",
        json={
            "age": 35,
            "bleeding": False,
            "rapid_growth": False,
            "pain_ulcer": False,
            "duration_long": False,
            "diabetes_immunocomp": False,
            "fever_systemic": False,
            "itching": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["final_risk_percentage"] < 35.0
    assert data["badge_color"] == "GREEN"
    assert data["suspected_superficial_fungal_or_scabies"] is True


def test_zero_checkbox_ulcer_triggers_visual_safety_floor_amber():
    # Zero-checkbox mode: the Visual Safety Floor must hold a visually severe sample
    # (Visual Prior 44.6% >= 0.385) at AMBER for PHC MO review instead of GREEN.
    res = client.post(
        "/api/v1/screen-sample/case_high_scc_ulcer",
        json={
            "age": 35,
            "bleeding": False,
            "rapid_growth": False,
            "pain_ulcer": False,
            "duration_long": False,
            "diabetes_immunocomp": False,
            "fever_systemic": False,
            "itching": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert 35.0 <= data["final_risk_percentage"] < 65.0
    assert data["badge_color"] == "AMBER"


def test_blurry_sample_triggers_camera_quality_gate():
    res = client.post("/api/v1/screen-sample/case_blurry_reject")
    assert res.status_code == 200
    data = res.json()
    assert data["camera_quality"]["is_blurry"] is True
    assert data["camera_quality"]["quality_passed"] is False
