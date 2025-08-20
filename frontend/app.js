// adjust if your backend runs elsewhere
const BASE_URL = "http://127.0.0.1:8000";
document.getElementById("backendUrl").textContent = BASE_URL;

const anemiaFields = ["Age", "Hemoglobin", "MCV", "MCH", "MCHC"];
const heartFields  = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"];

const qs = (sel) => document.querySelector(sel);
const $ = (id) => document.getElementById(id);

function showToast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2500);
}

async function apiHealthCheck() {
  try {
    const r = await fetch(`${BASE_URL}/`);
    if (r.ok) {
      $("apiStatus").classList.remove("offline");
      $("apiStatus").classList.add("online");
    }
  } catch (_) {
    $("apiStatus").classList.remove("online");
    $("apiStatus").classList.add("offline");
  }
}
apiHealthCheck();

function parseValues(formEl, fields) {
  const obj = {};
  for (const f of fields) {
    const el = formEl.querySelector(`[name="${f}"]`);
    if (!el) continue;
    const val = el.value.trim();
    obj[f] = val === "" ? null : Number(val);
  }
  return obj;
}

function renderResult(el, value) {
  if (!value) { el.innerHTML = ""; return; }
  const positive = String(value).toLowerCase() === "positive";
  el.innerHTML = `
    <span class="pill ${positive ? "negative" : "positive"}">
      ${positive ? "⚠️ Positive" : "✅ Negative"}
    </span>
  `;
}

async function handlePredict({ formId, btnId, endpoint, fields, resultId }) {
  const formEl = $(formId);
  const btnEl = $(btnId);
  const resEl = $(resultId);

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = parseValues(formEl, fields);

    // quick front-end validation
    if (Object.values(payload).some(v => v === null || Number.isNaN(v))) {
      showToast("Please fill all fields with valid numbers.");
      return;
    }

    btnEl.disabled = true;
    btnEl.textContent = "Predicting...";

    try {
      const r = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!r.ok) {
        const txt = await r.text();
        throw new Error(`API ${r.status}: ${txt}`);
      }

      const data = await r.json();
      renderResult(resEl, data.prediction);
    } catch (err) {
      console.error(err);
      resEl.innerHTML = "";
      showToast("Request failed. Check backend or CORS.");
    } finally {
      btnEl.disabled = false;
      btnEl.textContent = "Predict";
    }
  });
}

handlePredict({
  formId: "anemiaForm",
  btnId: "anemiaBtn",
  endpoint: "/predict/anemia",
  fields: anemiaFields,
  resultId: "anemiaResult",
});

handlePredict({
  formId: "heartForm",
  btnId: "heartBtn",
  endpoint: "/predict/heart",
  fields: heartFields,
  resultId: "heartResult",
});
