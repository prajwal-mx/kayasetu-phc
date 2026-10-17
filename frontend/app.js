let samplesData = [];
let activeSampleId = "case_high_scc_ulcer";
let lastScreeningResult = null;
let customUploadedBlob = null;

async function initApp() {
  try {
    const mRes = await fetch("/api/v1/model-metrics");
    if (mRes.ok) {
      const m = await mRes.json();
      const acc = (m.validation_accuracy * 100).toFixed(1);
      const sens = (m.validation_sensitivity * 100).toFixed(1);
      document.getElementById("topModelMetric").textContent =
        `PAD-UFES-20 ONNX (${m.total_samples_evaluated.toLocaleString()} samples · ${acc}% Acc · ${sens}% Sens · ECE ${m.expected_calibration_error_ece})`;
    }
  } catch (_) {}

  const res = await fetch("/api/v1/samples");
  samplesData = await res.json();
  renderSampleButtons();
  bindReactiveListeners();
  selectSample(samplesData[0].id);
}

function bindReactiveListeners() {
  const ids = [
    "ageInput",
    "chk_bleeding",
    "chk_rapid_growth",
    "chk_pain_ulcer",
    "chk_duration_long",
    "chk_diabetes",
    "chk_fever",
    "chk_itching",
  ];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => runCurrentScreening());
  });

  const fileInput = document.getElementById("imageInput");
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      customUploadedBlob = fileInput.files[0];
      document.getElementById("uploadFileName").textContent = `Uploaded: ${customUploadedBlob.name}`;
      activeSampleId = "custom_upload";
      renderSampleButtons();
      runCurrentScreening();
    }
  });
}

function renderSampleButtons() {
  const container = document.getElementById("sampleButtons");
  container.innerHTML = "";
  const dotMap = {
    case_high_scc_ulcer: { cls: "high", sub: "High-Risk Ulcer · Urgent Specialist Transfer" },
    case_mod_infective: { cls: "mod", sub: "Moderate Plaque · 48h PHC MO / eSanjeevani" },
    case_low_benign_tinea: { cls: "low", sub: "Low-Risk Benign / Tinea · Routine PHC Care" },
    case_blurry_reject: { cls: "blur", sub: "Out-of-Focus Shake · Camera Quality Gate Test" },
  };

  samplesData.forEach((s) => {
    const btn = document.createElement("button");
    btn.className = "case-tab-btn" + (s.id === activeSampleId ? " active" : "");
    const meta = dotMap[s.id] || { cls: "low", sub: "Clinical Case" };
    btn.innerHTML = `
      <span class="case-tab-dot ${meta.cls}"></span>
      <div>
        <span class="case-tab-title">${s.title_en.replace(/^Case \d+:\s*/, "")}</span>
        <span class="case-tab-sub">${meta.sub}</span>
      </div>
    `;
    btn.onclick = () => selectSample(s.id);
    container.appendChild(btn);
  });
}

function selectSample(sampleId) {
  activeSampleId = sampleId;
  customUploadedBlob = null;
  document.getElementById("imageInput").value = "";
  document.getElementById("uploadFileName").textContent = "Upload Unseen Skin Photo";
  const sample = samplesData.find((s) => s.id === sampleId);
  if (sample) {
    document.getElementById("ageInput").value = sample.symptoms.age;
    document.getElementById("chk_bleeding").checked = sample.symptoms.bleeding;
    document.getElementById("chk_rapid_growth").checked = sample.symptoms.rapid_growth;
    document.getElementById("chk_pain_ulcer").checked = sample.symptoms.pain_ulcer;
    document.getElementById("chk_duration_long").checked = sample.symptoms.duration_long;
    document.getElementById("chk_diabetes").checked = sample.symptoms.diabetes_immunocomp;
    document.getElementById("chk_fever").checked = sample.symptoms.fever_systemic;
    document.getElementById("chk_itching").checked = sample.symptoms.itching;
  }
  renderSampleButtons();
  runCurrentScreening();
}

function clearAllSymptoms() {
  [
    "chk_bleeding",
    "chk_rapid_growth",
    "chk_pain_ulcer",
    "chk_duration_long",
    "chk_diabetes",
    "chk_fever",
    "chk_itching",
  ].forEach((id) => {
    document.getElementById(id).checked = false;
  });
  runCurrentScreening();
}

function getSymptomPayload() {
  return {
    age: parseInt(document.getElementById("ageInput").value || "40", 10),
    bleeding: document.getElementById("chk_bleeding").checked,
    rapid_growth: document.getElementById("chk_rapid_growth").checked,
    pain_ulcer: document.getElementById("chk_pain_ulcer").checked,
    duration_long: document.getElementById("chk_duration_long").checked,
    diabetes_immunocomp: document.getElementById("chk_diabetes").checked,
    fever_systemic: document.getElementById("chk_fever").checked,
    itching: document.getElementById("chk_itching").checked,
  };
}

async function runCurrentScreening() {
  const symptoms = getSymptomPayload();
  let response;

  if (customUploadedBlob) {
    const formData = new FormData();
    formData.append("image", customUploadedBlob, "unseen_capture.png");
    Object.entries(symptoms).forEach(([k, v]) => formData.append(k, v));
    response = await fetch("/api/v1/screen", { method: "POST", body: formData });
  } else {
    const targetId = activeSampleId === "custom_upload" ? samplesData[0].id : activeSampleId;
    response = await fetch(`/api/v1/screen-sample/${targetId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(symptoms),
    });
  }

  lastScreeningResult = await response.json();
  renderResult(lastScreeningResult);
}

/* 1-Click Unseen Perturbation Generator right in Browser Canvas */
function applyUnseenPerturbation(mode) {
  const origImg = document.getElementById("imgOrig");
  if (!origImg || !origImg.src) return;
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  if (mode === "camera_shake") {
    ctx.filter = "blur(9px)";
    ctx.drawImage(origImg, 0, 0, 256, 256);
  } else {
    ctx.drawImage(origImg, 0, 0, 256, 256);
    const imgData = ctx.getImageData(0, 0, 256, 256);
    const d = imgData.data;
    if (mode === "darker_skin") {
      for (let i = 0; i < d.length; i += 4) {
        d[i] = Math.max(20, d[i] * 0.72);
        d[i + 1] = Math.max(15, d[i + 1] * 0.68);
        d[i + 2] = Math.max(12, d[i + 2] * 0.65);
      }
      ctx.putImageData(imgData, 0, 0);
    } else if (mode === "glare_flash") {
      ctx.putImageData(imgData, 0, 0);
      const grad = ctx.createRadialGradient(135, 115, 5, 135, 115, 62);
      grad.addColorStop(0, "rgba(255, 255, 255, 0.98)");
      grad.addColorStop(0.7, "rgba(255, 255, 255, 0.85)");
      grad.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(135, 115, 62, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  canvas.toBlob((blob) => {
    customUploadedBlob = blob;
    document.getElementById("uploadFileName").textContent = `Unseen Perturbation: ${mode.replace("_", " ")}`;
    runCurrentScreening();
  }, "image/png");
}

function renderResult(data) {
  document.getElementById("caseIdDisplay").textContent =
    `CASE ID: ${data.case_id} · ${data.inference_mode.replace(/_/g, " ")}`;
  document.getElementById("inferenceModeBadge").textContent =
    data.inference_mode.includes("ONNX") ? "ONNX Multimodal + CIE-L*a*b* Active" : "CIE-L*a*b* Analytical Mode";

  const q = data.camera_quality;
  const chipQ = document.getElementById("chipQuality");
  const dotQ = document.getElementById("dotQuality");
  if (q.quality_passed) {
    chipQ.textContent = `Camera Gate: PASS (Sharpness σ²=${q.sharpness_variance})`;
    dotQ.textContent = "✓";
    dotQ.style.background = "#4D6A36";
  } else {
    const reason = q.is_blurry
      ? `Blur σ²=${q.sharpness_variance} (<35)`
      : q.has_glare
      ? `Glare ${(q.glare_ratio * 100).toFixed(1)}% (>12%)`
      : q.lighting_status;
    chipQ.textContent = `RETAKE PHOTO: ${reason}`;
    dotQ.textContent = "!";
    dotQ.style.background = "#B45309";
  }

  document.getElementById("chipSkinTone").textContent =
    `${data.fitzpatrick_type} (${data.skin_tone_label} · ITA° ${data.ita_degrees}°)`;
  document.getElementById("chipErythema").textContent =
    `Melanin Boost ×${data.melanin_compensation_factor} · Erythema ${data.adjusted_erythema_index}`;

  // Update Sarvam Workflow Source Node
  document.getElementById("wfVisualPrior").textContent =
    `${(data.visual_prior_probability * 100).toFixed(1)}%`;
  document.getElementById("wfClinShift").textContent =
    `${data.clinical_log_odds_shift >= 0 ? "+" : ""}${data.clinical_log_odds_shift} log-odds`;
  document.getElementById("wfSkinTone").textContent =
    `${data.fitzpatrick_type} (${data.ita_degrees}°)`;

  // Update Hero Score & Progress Bar
  const card = document.getElementById("referralCard");
  const fill = document.getElementById("riskProgressFill");

  if (!q.quality_passed) {
    card.className = "score-hero-box SLATE";
    document.getElementById("tierBadge").textContent = "QUALITY GATE FAILED · RETAKE PHOTO";
    document.getElementById("riskHeadline").textContent = `${data.final_risk_percentage}%*`;
    document.getElementById("recommendationText").textContent =
      `⚠️ CAMERA QUALITY GATE WARNING: Photo failed pre-inference quality check (Sharpness σ²=${q.sharpness_variance}, Glare=${(q.glare_ratio * 100).toFixed(1)}%, Lighting=${q.lighting_status}). Please hold the smartphone steady 15 cm from the skin in indirect daylight and retake.`;
  } else {
    card.className = `score-hero-box ${data.badge_color}`;
    document.getElementById("tierBadge").textContent = data.triage_tier.replace(/_/g, " ");
    document.getElementById("riskHeadline").textContent = `${data.final_risk_percentage}%`;
    document.getElementById("recommendationText").textContent = data.recommendation_en;
  }

  fill.style.width = `${Math.min(100, data.final_risk_percentage)}%`;
  fill.style.background =
    data.badge_color === "RED" ? "#B91C1C" : data.badge_color === "AMBER" ? "#B45309" : "#15803D";

  // Highlight Active Branch in Sarvam Workflow Graph
  const nodeHigh = document.getElementById("nodeHigh");
  const nodeMod = document.getElementById("nodeMod");
  const nodeLow = document.getElementById("nodeLow");
  const pathHigh = document.getElementById("pathHigh");
  const pathMod = document.getElementById("pathMod");
  const pathLow = document.getElementById("pathLow");

  nodeHigh.classList.remove("active");
  nodeMod.classList.remove("active");
  nodeLow.classList.remove("active");
  pathHigh.className.baseVal = "wf-path";
  pathMod.className.baseVal = "wf-path";
  pathLow.className.baseVal = "wf-path";

  if (data.badge_color === "RED") {
    nodeHigh.classList.add("active");
    pathHigh.className.baseVal = "wf-path active-high";
  } else if (data.badge_color === "AMBER") {
    nodeMod.classList.add("active");
    pathMod.className.baseVal = "wf-path active-mod";
  } else {
    nodeLow.classList.add("active");
    pathLow.className.baseVal = "wf-path active-low";
  }

  document.getElementById("nodeLowSub").textContent =
    data.suspected_superficial_fungal_or_scabies
      ? "Suspected superficial fungal tinea/scabies pattern — topical hygiene & 7-day check"
      : "Provide routine PHC hygiene counseling & re-screen in 7 days if unhealed";

  // 4-Stage Vision Images
  document.getElementById("imgOrig").src = data.images_base64.original;
  document.getElementById("imgNorm").src = data.images_base64.normalized;
  document.getElementById("imgSeg").src = data.images_base64.segmentation;
  document.getElementById("imgHeat").src = data.images_base64.heatmap;

  // ABCD Morphological Bars
  const abcd = data.abcd_scores;
  const abcdItems = [
    ["A — Asymmetry Index (Centroid Reflection)", abcd.asymmetry],
    ["B — Border Irregularity (P² / 4π·Area)", abcd.border_irregularity],
    ["C — Color Variegation (CIE-L*a*b* Std)", abcd.color_variegation],
    ["D — Melanin-Adjusted Erythema & Extent", abcd.diameter_erythema_extent],
  ];
  document.getElementById("abcdList").innerHTML = abcdItems
    .map(
      ([label, val]) => `
      <li>
        <div class="metric-row-top">
          <span>${label}</span>
          <strong>${(val * 100).toFixed(1)}%</strong>
        </div>
        <div class="mini-bar-track">
          <div class="mini-bar-fill" style="width:${Math.min(100, val * 100)}%"></div>
        </div>
      </li>
    `
    )
    .join("");

  // Bayesian Log-Odds Contributions
  document.getElementById("contribList").innerHTML = data.feature_contributions
    .map((c) => {
      const pctWidth = Math.min(100, Math.abs(c.log_odds_delta) * 55);
      const barColor = c.log_odds_delta >= 0 ? "#18181B" : "#15803D";
      return `
      <li>
        <div class="metric-row-top">
          <span>${c.feature}</span>
          <strong>${c.log_odds_delta >= 0 ? "+" : ""}${c.log_odds_delta} log-odds</strong>
        </div>
        <div class="mini-bar-track">
          <div class="mini-bar-fill" style="width:${pctWidth}%; background:${barColor}"></div>
        </div>
      </li>
    `;
    })
    .join("");
}

function speakRecommendation() {
  if (!lastScreeningResult || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(lastScreeningResult.recommendation_en);
  utter.lang = "en-IN";
  window.speechSynthesis.speak(utter);
}

function openReferralSlipModal() {
  if (!lastScreeningResult) return;
  const d = lastScreeningResult;
  document.getElementById("slipCaseId").textContent = d.case_id;
  document.getElementById("slipDate").textContent = new Date().toLocaleString("en-IN");
  document.getElementById("slipAge").textContent = document.getElementById("ageInput").value;
  document.getElementById("slipTier").textContent = d.triage_tier.replace(/_/g, " ");
  document.getElementById("slipRecommendation").textContent = d.recommendation_en;
  document.getElementById("slipScore").textContent = `${d.final_risk_percentage}%`;
  document.getElementById("slipSkinTelemetry").textContent =
    `${d.fitzpatrick_type} (${d.skin_tone_label}) · ITA° ${d.ita_degrees}° · Melanin Boost ×${d.melanin_compensation_factor} · Sharpness σ²=${d.camera_quality.sharpness_variance}`;
  const abcd = d.abcd_scores;
  document.getElementById("slipAbcdSummary").textContent =
    `Asymmetry: ${(abcd.asymmetry * 100).toFixed(1)}% | Border: ${(abcd.border_irregularity * 100).toFixed(1)}% | Color: ${(abcd.color_variegation * 100).toFixed(1)}% | Extent: ${(abcd.diameter_erythema_extent * 100).toFixed(1)}%`;
  document.getElementById("slipSymptomsList").innerHTML = d.feature_contributions
    .map((c) => `<li>${c.feature}: <strong>${c.log_odds_delta >= 0 ? "+" : ""}${c.log_odds_delta}</strong></li>`)
    .join("");
  document.getElementById("slipImgOrig").src = d.images_base64.normalized;
  document.getElementById("slipImgHeat").src = d.images_base64.heatmap;
  document.getElementById("slipDisclaimer").textContent = d.disclaimer;
  document.getElementById("referralModal").classList.remove("hidden");
}

function closeReferralSlipModal() {
  document.getElementById("referralModal").classList.add("hidden");
}

window.addEventListener("DOMContentLoaded", initApp);
