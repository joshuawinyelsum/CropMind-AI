'use strict';

// ============================================================================
// APPLICATION STATE — single source of truth
// ============================================================================
const state = {
  view:            'scan',     // 'scan' | 'history' | 'settings'
  phase:           'upload',   // 'upload' | 'preview' | 'analyzing' | 'results'
  file:            null,       // File object from input / drop
  previewUrl:      null,       // Object URL for local preview
  result:          null,       // API response data
  history:         [],         // Local scan history (survives view switches)
  cameraStream:    null,       // Media stream for webcam
};

const PREDICT_API_URL = 'http://127.0.0.1:8000/predict';

// ============================================================================
// DOM CACHE — resolved once on DOMContentLoaded
// ============================================================================
const dom = {};

function cacheDom() {
  // Navigation
  dom.navScan     = document.getElementById('nav-scan');
  dom.navHistory  = document.getElementById('nav-history');
  dom.navSettings = document.getElementById('nav-settings');
  dom.navItems    = [dom.navScan, dom.navHistory, dom.navSettings];

  // Views
  dom.scanLeft          = document.getElementById('scan-left');
  dom.treatmentsSection = document.getElementById('treatments-section');
  dom.historyView       = document.getElementById('history-view');
  dom.settingsView      = document.getElementById('settings-view');

  // Upload state
  dom.uploadState      = document.getElementById('upload-state');
  dom.resultsState     = document.getElementById('results-state');
  dom.uploadZone       = document.getElementById('upload-zone');
  dom.uploadPlaceholder = document.getElementById('upload-placeholder');
  dom.uploadPreview    = document.getElementById('upload-preview');
  dom.previewImage     = document.getElementById('preview-image');
  dom.previewFilename  = document.getElementById('preview-filename');
  dom.fileInput        = document.getElementById('file-input');
  
  // Camera
  dom.cameraView       = document.getElementById('camera-view');
  dom.cameraVideo      = document.getElementById('camera-video');
  dom.cameraCanvas     = document.getElementById('camera-canvas');
  dom.btnSnapPhoto     = document.getElementById('btn-snap-photo');
  dom.btnCancelCamera  = document.getElementById('btn-cancel-camera');

  dom.btnTakePhoto     = document.getElementById('btn-take-photo');
  dom.btnUploadImage   = document.getElementById('btn-upload-image');
  dom.btnAnalyze       = document.getElementById('btn-analyze');
  dom.btnAnalyzeText   = document.getElementById('btn-analyze-text');

  // Results
  dom.resultImage      = document.getElementById('result-image');
  dom.resultDisease    = document.getElementById('result-disease');
  dom.resultConfidence = document.getElementById('result-confidence');
  dom.resultSeverity   = document.getElementById('result-severity');
  dom.resultSeverityDot = document.getElementById('result-severity-dot');
  dom.resultCategory   = document.getElementById('result-category');
  dom.btnCloseResult   = document.getElementById('btn-close-result');
  dom.btnNewScan       = document.getElementById('btn-new-scan');

  // Treatments
  dom.treatmentsContainer = document.getElementById('treatments-container');
  dom.treatmentList = document.getElementById('treatment-list');

  // History
  dom.historyList = document.getElementById('history-list');

  // Toast
  dom.toastContainer = document.getElementById('toast-container');
}

// ============================================================================
// RENDER — state → DOM  (unidirectional, always safe to call)
// ============================================================================

function render() {
  renderNavigation();
  renderViewVisibility();
  if (state.view === 'scan')    renderScanPhase();
  if (state.view === 'history') renderHistory();
}

// --- Navigation highlights ---
function renderNavigation() {
  const viewMap = { scan: dom.navScan, history: dom.navHistory, settings: dom.navSettings };
  dom.navItems.forEach((el) => el.classList.remove('active'));
  if (viewMap[state.view]) viewMap[state.view].classList.add('active');
}

// --- Show / hide top-level views ---
function renderViewVisibility() {
  const isScan     = state.view === 'scan';
  const isHistory  = state.view === 'history';
  const isSettings = state.view === 'settings';

  toggle(dom.scanLeft,          isScan);
  toggle(dom.treatmentsSection, isScan && state.phase === 'results');
  toggle(dom.historyView,       isHistory);
  toggle(dom.settingsView,      isSettings);
}

// --- Scan view phases ---
function renderScanPhase() {
  const { phase } = state;
  const showUploadCard = phase !== 'results';

  toggle(dom.uploadState,  showUploadCard);
  toggle(dom.resultsState, phase === 'results');

  // Upload zone: placeholder vs preview vs camera
  const hasFile = phase === 'preview' || phase === 'analyzing';
  const isCamera = phase === 'camera';
  
  toggle(dom.uploadPlaceholder, !hasFile && !isCamera);
  toggle(dom.uploadPreview,      hasFile);
  toggle(dom.cameraView,         isCamera);

  if (hasFile && state.previewUrl) {
    dom.previewImage.src = state.previewUrl;
    dom.previewFilename.textContent = state.file ? state.file.name : '';
  }

  // Analyze button
  const isAnalyzing = phase === 'analyzing';
  dom.btnAnalyze.disabled = phase === 'upload' || isAnalyzing;
  dom.btnAnalyze.classList.toggle('loading', isAnalyzing);
  dom.btnAnalyzeText.textContent = isAnalyzing ? 'Analyzing…' : 'Analyze Crop';

  // Results content
  if (phase === 'results' && state.result) {
    populateResults(state.result);
  }
}

function populateResults(r) {
  dom.resultImage.src            = state.previewUrl || '';
  dom.resultDisease.textContent  = r.disease;
  dom.resultConfidence.textContent = r.confidence + '%';
  dom.resultSeverity.textContent = r.severity || 'Not provided';
  dom.resultCategory.textContent = r.category || 'Not provided';

  // ── OOD Warning Banner ──────────────────────────────────────────────────
  // Show an amber warning banner when the model rejects the image as OOD.
  // The banner element is optional — gracefully skip if the HTML doesn't have it yet.
  const oodBanner = document.getElementById('ood-warning-banner');
  if (oodBanner) {
    toggle(oodBanner, !!r.isOOD);
    if (r.isOOD && r.oodMessage) {
      const oodText = document.getElementById('ood-warning-text');
      if (oodText) oodText.textContent = r.oodMessage;
    }
  } else if (r.isOOD) {
    // Fallback: show a toast if the banner element isn't in the HTML
    showToast('⚠️ ' + (r.oodMessage || 'Unsupported crop detected. Results are not reliable.'), 'error');
  }
  // ── End OOD Banner ───────────────────────────────────────────────────────

  // Severity dot color — suppress for OOD results
  if (!r.isOOD) {
    const dotColor = r.severity === 'High'   ? 'var(--danger)'
                   : r.severity === 'Medium' ? '#f59e0b'
                   :                           'var(--primary-green)';
    dom.resultSeverityDot.style.backgroundColor = dotColor;
  } else {
    dom.resultSeverityDot.style.backgroundColor = 'var(--text-muted, #888)';
  }

  renderTreatments(r.treatments);
}

// --- Treatments ---
function renderTreatments(treatments) {
  const listEl = dom.treatmentList || dom.treatmentsContainer;

  if (!treatments || !treatments.length) {
    listEl.innerHTML = '<li class="empty-state">No treatments available.</li>';
    return;
  }

  listEl.innerHTML = treatments.map(function (treatment) {
    return '<li>' + formatTreatmentRecommendation(treatment) + '</li>';
  }).join('');
}

// --- History ---
function renderHistory() {
  if (!state.history.length) {
    dom.historyList.innerHTML =
      '<div class="empty-state">' +
        '<svg class="icon-lg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:0.5rem;color:var(--text-muted)">' +
          '<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>' +
        '</svg>' +
        '<p>No scans yet. Upload a leaf image to get started.</p>' +
      '</div>';
    return;
  }

  dom.historyList.innerHTML = state.history.map((h) =>
    '<div class="history-item">' +
      '<div class="history-thumb">' +
        (h.previewUrl
          ? '<img src="' + h.previewUrl + '" alt="Scan">'
          : '<div class="thumb-placeholder"><svg class="icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--text-light);margin-top:14px;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg></div>') +
      '</div>' +
      '<div class="history-info">' +
        '<div class="history-disease">' + esc(h.disease) + '</div>' +
        '<div class="history-meta">' + esc(h.crop) + ' · ' + h.confidence + '% · ' + formatDate(h.timestamp) + '</div>' +
      '</div>' +
      '<div class="history-severity severity-' + h.severity.toLowerCase() + '">' + esc(h.severity) + '</div>' +
      '<button class="history-delete" data-delete-id="' + h.id + '" title="Delete Scan">' +
        '<svg class="icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"></path><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>' +
      '</button>' +
    '</div>'
  ).join('');
}

// ============================================================================
// ACTIONS — mutate state, then render()
// ============================================================================

/**
 * Validate and load a selected file into state.
 */
function selectFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('Please select a valid image file (JPEG, PNG, WebP).', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('Image must be under 10 MB.', 'error');
    return;
  }

  // Release previous blob URL
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);

  state.file       = file;
  state.previewUrl = URL.createObjectURL(file);
  state.phase      = 'preview';
  state.result     = null;
  render();
}

/**
 * Clear the selected file and return to upload phase.
 * Also aborts any in-flight request.
 */
function removeFile() {
  stopCamera();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);

  state.file       = null;
  state.previewUrl = null;
  state.result     = null;
  state.phase      = 'upload';
  dom.fileInput.value = '';
  render();
}

/**
 * Analyze the selected image through the FastAPI backend.
 */
async function analyzeImage() {
  if (state.phase !== 'preview' || !state.file) return;

  state.phase = 'analyzing';
  render();

  try {
    const payload = await predictImage(state.file);
    state.result = normalizePredictionResponse(payload);
    state.phase = 'results';

    const historyItem = {
      id: state.result.imageId || Date.now(),
      disease: state.result.disease,
      crop: state.result.crop || 'Unknown',
      confidence: state.result.confidence,
      severity: state.result.severity || 'Not provided',
      category: state.result.category || 'Not provided',
      previewUrl: state.previewUrl,
      timestamp: new Date().toISOString(),
    };
    state.history.unshift(historyItem);
    if (state.history.length > 20) state.history.pop();
    saveHistoryToStorage();
  } catch (err) {
    console.error('[AnalyzeImage] Error:', err);
    if (err.responseStatus) {
      console.error('[AnalyzeImage] Server Response:', err.responseStatus, err.responseText);
    }
    
    const errMsg = err.message || 'Analysis failed. Please try again.';
    showToast(errMsg, 'error');
    state.phase = 'preview';
  } finally {
    render();
  }
}

async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });
    const data = await response.json();
    console.log("API RESPONSE:", data);
    document.getElementById("result").innerText = `Prediction: ${data.predicted_class} (${data.confidence})`;
    return data;
  } catch (error) {
    console.error("ERROR:", error);
    alert("Network error: Cannot reach backend.");
    throw error;
  }
}

function normalizePredictionResponse(payload) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('Response is not a valid JSON object.');
  }

  // 1. Defensively locate the disease/class name
  const diseaseRaw = payload.predicted_class 
                  || payload.class 
                  || payload.disease 
                  || payload.prediction;
                  
  if (!diseaseRaw || typeof diseaseRaw !== 'string') {
    console.error('[Normalize] Missing disease field. Payload:', payload);
    throw new Error('Response is missing the predicted class name.');
  }

  // ── OOD / Unknown class detection ───────────────────────────────────────
  // The backend sets predicted_class = "unknown" and disease_key = "unsupported_crop"
  // when the OOD gate fires (low confidence OR high entropy).
  const isOOD = (diseaseRaw === 'unknown') ||
                (payload.disease_key === 'unsupported_crop');

  if (isOOD) {
    let rawConfidence = payload.confidence ?? 0;
    if (typeof rawConfidence === 'string') rawConfidence = parseFloat(rawConfidence);
    if (!Number.isFinite(rawConfidence)) rawConfidence = 0;
    const confidencePercent = rawConfidence <= 1
      ? Math.round(rawConfidence * 100)
      : Math.round(rawConfidence);

    console.warn('[Normalize] OOD response detected. message:', payload.message);
    return {
      imageId:      payload.image_id     || `img_${Date.now()}`,
      predictionId: payload.prediction_id || `pred_${Date.now()}`,
      disease:      'Unsupported Crop',
      confidence:   Math.min(Math.max(confidencePercent, 0), 100),
      treatments:   [],
      crop:         'Unknown',
      category:     'Out of Distribution',
      severity:     'Unknown',
      isOOD:        true,
      oodMessage:   payload.message || 'Unsupported crop or unclear image. Please upload a tomato leaf.',
      entropy:      payload.entropy  || null,
    };
  }
  // ── End OOD branch ───────────────────────────────────────────────────────

  // Format "Tomato___Late_blight" -> "Tomato - Late blight"
  const formattedDisease = diseaseRaw.replace(/___/g, ' - ').replace(/_/g, ' ');

  // 2. Defensively parse confidence
  let rawConfidence = payload.confidence ?? payload.probability ?? payload.score ?? 0;
  
  if (typeof rawConfidence === 'string') {
    rawConfidence = parseFloat(rawConfidence.replace(/[^0-9.]/g, ''));
  }
  
  if (!Number.isFinite(rawConfidence) || isNaN(rawConfidence)) {
    console.warn('[Normalize] Confidence missing or invalid, defaulting to 0');
    rawConfidence = 0;
  }

  const confidencePercent = rawConfidence <= 1 
    ? Math.round(rawConfidence * 100) 
    : Math.round(rawConfidence);

  // 3. Extract other optional fields
  const treatments = Array.isArray(payload.treatments) ? payload.treatments : [];

  return {
    imageId: payload.image_id || `img_${Date.now()}`,
    predictionId: payload.prediction_id || `pred_${Date.now()}`,
    disease: formattedDisease,
    confidence: Math.min(Math.max(confidencePercent, 0), 100),
    treatments: treatments,
    crop: payload.crop || formattedDisease.split(' - ')[0] || 'Unknown',
    category: payload.category || 'Uncategorized',
    severity: payload.severity || 'Not provided',
    isOOD: false,
    oodMessage: null,
    entropy: payload.entropy || null,
  };
}

/**
 * Save history to localStorage
 */
function saveHistoryToStorage() {
  try {
    // Avoid saving previewUrls directly to localStorage as they are transient Object URLs
    const serializedHistory = state.history.map(h => ({ ...h, previewUrl: null }));
    localStorage.setItem('cropmind_history', JSON.stringify(serializedHistory));
  } catch (err) {
    console.error('Failed to save history to storage', err);
  }
}

/**
 * Fetch history from localStorage.
 */
function fetchHistory() {
  try {
    const stored = localStorage.getItem('cropmind_history');
    if (stored) {
      state.history = JSON.parse(stored);
    }
  } catch (err) {
    console.error('Failed to load history from storage:', err);
    state.history = [];
  }
  if (state.view === 'history') render();
}

/**
 * Delete a history item.
 */
function deleteHistoryItem(id) {
  state.history = state.history.filter(h => h.id !== id);
  saveHistoryToStorage();
  renderHistory();
  showToast('Scan deleted successfully.', 'success');
}

/**
 * Switch the active navigation view.
 */
function navigate(view) {
  if (state.view === view) return;
  state.view = view;
  if (view === 'history') fetchHistory();
  render();
}

// ============================================================================
// TOAST NOTIFICATIONS
// ============================================================================

function showToast(message, type) {
  type = type || 'info';
  const el = document.createElement('div');
  el.className = 'toast toast-' + type;
  el.textContent = message;
  dom.toastContainer.appendChild(el);

  // Trigger CSS transition
  requestAnimationFrame(function () { el.classList.add('show'); });

  setTimeout(function () {
    el.classList.remove('show');
    el.addEventListener('transitionend', function () { el.remove(); });
  }, 4000);
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Basic heuristic validation for image quality (blur and brightness).
 * Runs on the canvas context before allowing submission.
 * @returns {boolean} true if acceptable, false if quality is bad.
 */
function validateImageQuality(canvas) {
  const ctx = canvas.getContext('2d');
  // Downsample to check a smaller central region for performance on low-end Androids
  const w = Math.min(canvas.width, 200);
  const h = Math.min(canvas.height, 200);
  const startX = Math.max(0, Math.floor((canvas.width - w) / 2));
  const startY = Math.max(0, Math.floor((canvas.height - h) / 2));
  
  let imageData;
  try {
    imageData = ctx.getImageData(startX, startY, w, h);
  } catch (e) {
    return true; // CORS or other canvas error, fallback to allowing it
  }
  
  const data = imageData.data;
  let totalBrightness = 0;
  const pixelCount = w * h;
  
  // Calculate average brightness
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i+1], b = data[i+2];
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b);
    totalBrightness += luminance;
  }
  const avgBrightness = totalBrightness / pixelCount;
  
  // Check for very dark or empty images (corrupted/black)
  if (avgBrightness < 20) {
    return false;
  }
  
  // Check for Blur using Variance of Horizontal Differences
  let diffSum = 0;
  let diffSqSum = 0;
  let diffCount = 0;
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w - 1; x++) {
      const idx1 = (y * w + x) * 4;
      const idx2 = (y * w + (x + 1)) * 4;
      
      const lum1 = (0.299 * data[idx1] + 0.587 * data[idx1+1] + 0.114 * data[idx1+2]);
      const lum2 = (0.299 * data[idx2] + 0.587 * data[idx2+1] + 0.114 * data[idx2+2]);
      
      const diff = Math.abs(lum1 - lum2);
      diffSum += diff;
      diffSqSum += (diff * diff);
      diffCount++;
    }
  }
  
  const meanDiff = diffSum / diffCount;
  const variance = (diffSqSum / diffCount) - (meanDiff * meanDiff);
  
  // Low variance = blurry, High variance = sharp
  if (variance < 15) {
    return false;
  }
  
  return true;
}

/** Minimal HTML escaping for controlled data inserted via innerHTML. */
function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

/** Toggle visibility via the .hidden class. */
function toggle(el, show) {
  if (!el) return;
  el.classList.toggle('hidden', !show);
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(track => track.stop());
    state.cameraStream = null;
  }
  if (dom.cameraVideo) {
    dom.cameraVideo.srcObject = null;
  }
}

function formatTreatmentRecommendation(treatment) {
  if (typeof treatment === 'string') {
    return esc(treatment);
  }

  if (!treatment || typeof treatment !== 'object') {
    return 'Treatment details not available.';
  }

  const title = treatment.title
    || treatment.name
    || treatment.product_name
    || treatment.recommendation
    || treatment.treatment
    || 'Treatment recommendation';

  const details = [];
  if (treatment.active_ingredient) details.push('Active ingredient: ' + treatment.active_ingredient);
  if (treatment.dosage) details.push('Dosage: ' + treatment.dosage);
  if (treatment.description) details.push(treatment.description);
  if (treatment.usage) details.push(treatment.usage);

  const steps = Array.isArray(treatment.application_steps)
    ? treatment.application_steps
    : Array.isArray(treatment.steps)
      ? treatment.steps
      : [];

  const stepText = steps.length ? ' Steps: ' + steps.join(' ') : '';
  const detailText = details.length ? ' - ' + details.join(' | ') : '';

  return '<strong>' + esc(title) + '</strong>' + esc(detailText + stepText);
}

// ============================================================================
// EVENT BINDING
// ============================================================================

function bindEvents() {
  // --- File selection ---
  dom.btnUploadImage.addEventListener('click', function () {
    dom.fileInput.removeAttribute('capture');
    dom.fileInput.click();
  });

  dom.btnTakePhoto.addEventListener('click', async function () {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('Camera API not supported in this browser.', 'error');
      // Fallback to file input
      dom.fileInput.setAttribute('capture', 'environment');
      dom.fileInput.click();
      return;
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      state.cameraStream = stream;
      dom.cameraVideo.srcObject = stream;
      state.phase = 'camera';
      render();
    } catch (err) {
      showToast('Camera access denied or unavailable.', 'error');
    }
  });

  if (dom.btnCancelCamera) {
    dom.btnCancelCamera.addEventListener('click', function() {
      stopCamera();
      state.phase = 'upload';
      render();
    });
  }

  if (dom.btnSnapPhoto) {
    dom.btnSnapPhoto.addEventListener('click', function() {
      if (!state.cameraStream) return;
      const video = dom.cameraVideo;
      const canvas = dom.cameraCanvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Basic image validation check
      if (!validateImageQuality(canvas)) {
        showToast('Image not clear. Please retake with better lighting and closer view.', 'error');
        return; // Don't stop camera, let user retake
      }
      
      canvas.toBlob(function(blob) {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        stopCamera();
        selectFile(file);
      }, 'image/jpeg', 0.9);
    });
  }

  dom.fileInput.addEventListener('change', function (e) {
    if (e.target.files && e.target.files[0]) selectFile(e.target.files[0]);
  });

  // --- Drag & drop ---
  dom.uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dom.uploadZone.classList.add('drag-over');
  });
  dom.uploadZone.addEventListener('dragleave', function (e) {
    e.preventDefault();
    dom.uploadZone.classList.remove('drag-over');
  });
  dom.uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    dom.uploadZone.classList.remove('drag-over');
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files[0]) selectFile(files[0]);
  });

  // --- Analyze ---
  dom.btnAnalyze.addEventListener('click', analyzeImage);

  // --- Reset / close ---
  dom.btnCloseResult.addEventListener('click', removeFile);
  dom.btnNewScan.addEventListener('click', removeFile);
  document.getElementById('btn-remove-preview').addEventListener('click', removeFile);

  // --- Navigation ---
  dom.navScan.addEventListener('click', function (e) { e.preventDefault(); navigate('scan'); });
  dom.navHistory.addEventListener('click', function (e) { e.preventDefault(); navigate('history'); });
  dom.navSettings.addEventListener('click', function (e) { e.preventDefault(); navigate('settings'); });

  document.addEventListener('click', function (e) {
    var delBtn = e.target.closest('[data-delete-id]');
    if (delBtn) {
      var delId = parseInt(delBtn.dataset.deleteId, 10);
      if (confirm('Are you sure you want to delete this scan?')) {
        deleteHistoryItem(delId);
      }
    }
  });

  // Settings dark mode toggle
  const themeToggle = document.getElementById('setting-theme');
  if (themeToggle) {
    themeToggle.addEventListener('change', function(e) {
      if (e.target.checked) {
        document.documentElement.style.setProperty('--bg-color', '#111827');
        document.documentElement.style.setProperty('--white', '#1f2937');
        document.documentElement.style.setProperty('--text-dark', '#f9fafb');
        document.documentElement.style.setProperty('--text-muted', '#9ca3af');
        document.documentElement.style.setProperty('--border-color', '#374151');
        document.documentElement.style.setProperty('--border-light', '#374151');
        document.documentElement.style.setProperty('--surface-elevated', 'rgba(31, 41, 55, 0.85)');
      } else {
        document.documentElement.style.cssText = '';
      }
    });
  }
}

// ============================================================================
// INIT
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
  cacheDom();
  bindEvents();
  fetchHistory(); // initial fetch
  render();
});
