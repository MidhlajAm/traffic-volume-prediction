"use strict";
const data = JSON.parse(document.getElementById("project-data").textContent);
const page = document.body.dataset.page;
const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const measurement = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });
const menuButton = document.getElementById("menu-button");
const sidebar = document.getElementById("sidebar");
menuButton.addEventListener("click", () => {
  const expanded = menuButton.getAttribute("aria-expanded") !== "true";
  menuButton.setAttribute("aria-expanded", String(expanded));
  sidebar.classList.toggle("open", expanded);
});
document.getElementById("close-menu").addEventListener("click", () => { closeMenu(); menuButton.focus(); });
document.addEventListener("click", (event) => {
  if (!sidebar.contains(event.target) && !menuButton.contains(event.target)) closeMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sidebar.classList.contains("open")) { closeMenu(); menuButton.focus(); }
});
function closeMenu() {
  sidebar.classList.remove("open");
  menuButton.setAttribute("aria-expanded", "false");
}

if (typeof Chart !== "undefined") {
  Chart.defaults.font.family = '"Segoe UI", Arial, sans-serif';
  Chart.defaults.font.size = 10;
  Chart.defaults.color = "#8a98ad";
  Chart.defaults.plugins.legend.display = false;
  // Render immediately, including when the demo opens in a background tab.
  Chart.defaults.animation.duration = 0;
  Chart.defaults.plugins.tooltip.backgroundColor = "#1c3354";
  Chart.defaults.plugins.tooltip.padding = 11;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.plugins.tooltip.displayColors = false;
  const scales = () => ({
    x: { grid: { display: false }, border: { display: false }, ticks: { maxRotation: 0, padding: 9 } },
    y: { beginAtZero: true, border: { display: false, dash: [3, 4] }, grid: { color: "#eef2f8", drawTicks: false }, ticks: { padding: 10, maxTicksLimit: 5, callback: (v) => v >= 1000 ? `${v / 1000}k` : v } },
  });
  const options = () => ({ responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, scales: scales() });
  if (page === "dashboard") {
    const hourlyOptions = options();
    hourlyOptions.scales.x.ticks.callback = function(value, index) { return index % 3 === 0 ? `${String(index).padStart(2, "0")}:00` : ""; };
    hourlyOptions.plugins = { tooltip: { callbacks: { title: (items) => `${String(items[0].dataIndex).padStart(2, "0")}:00 local time`, label: (item) => `${integer.format(item.raw)} vehicles/hour` } } };
    const hourlyChart = new Chart(document.getElementById("hourly-chart"), {
      type: "line", data: { labels: Array.from({ length: 24 }, (_, i) => i), datasets: [{ data: data.charts.hourly.all, borderColor: "#3975ed", borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 5, fill: true, tension: 0.32, backgroundColor: (context) => {
        const { ctx, chartArea } = context.chart;
        if (!chartArea) return "#3975ed12";
        const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
        gradient.addColorStop(0, "#3975ed25"); gradient.addColorStop(1, "#3975ed01"); return gradient;
      } }] }, options: hourlyOptions,
    });
    document.querySelectorAll("[data-profile]").forEach((button) => button.addEventListener("click", () => {
      document.querySelectorAll("[data-profile]").forEach((other) => { other.classList.toggle("selected", other === button); other.setAttribute("aria-pressed", String(other === button)); });
      hourlyChart.data.datasets[0].data = data.charts.hourly[button.dataset.profile]; hourlyChart.update();
    }));
    const weekdayOptions = options();
    weekdayOptions.plugins = { tooltip: { callbacks: { label: (item) => `${integer.format(item.raw)} vehicles/hour` } } };
    new Chart(document.getElementById("weekday-chart"), { type: "bar", data: { labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], datasets: [{ data: data.charts.weekday, backgroundColor: ["#83aaf4", "#83aaf4", "#83aaf4", "#83aaf4", "#3975ed", "#d7e3fb", "#d7e3fb"], borderRadius: 4, maxBarThickness: 28 }] }, options: weekdayOptions });
  }
  if (page === "performance") {
    const scatterOptions = options();
    scatterOptions.interaction = { mode: "nearest", intersect: true };
    scatterOptions.scales.x = { ...scales().y, title: { display: true, text: "Observed vehicles/hour", color: "#7b8da6", font: { size: 10 } } };
    scatterOptions.scales.y.title = { display: true, text: "Predicted vehicles/hour", color: "#7b8da6", font: { size: 10 } };
    const maximum = Math.ceil(Math.max(...data.charts.scatter.flatMap((point) => [point.x, point.y])) / 1000) * 1000;
    scatterOptions.scales.x.max = maximum; scatterOptions.scales.y.max = maximum;
    scatterOptions.plugins = { tooltip: { callbacks: { label: (item) => `Observed: ${integer.format(item.parsed.x)} · Predicted: ${integer.format(item.parsed.y)}` }, filter: (item) => item.datasetIndex === 0 } };
    new Chart(document.getElementById("scatter-chart"), { type: "scatter", data: { datasets: [{ data: data.charts.scatter, backgroundColor: "#3975ed66", pointRadius: 2.5, pointHoverRadius: 5 }, { type: "line", data: [{ x: 0, y: 0 }, { x: maximum, y: maximum }], borderColor: "#a0aec0", borderWidth: 1.5, borderDash: [5, 5], pointRadius: 0 }] }, options: scatterOptions });
    const timelineOptions = options();
    timelineOptions.scales.x.ticks = { maxRotation: 0, maxTicksLimit: 4, font: { size: 9 } };
    timelineOptions.plugins = { legend: { display: true, position: "bottom", labels: { usePointStyle: true, pointStyle: "line", padding: 16, boxWidth: 18, font: { size: 10 } } }, tooltip: { callbacks: { label: (item) => `${item.dataset.label}: ${integer.format(item.raw)} veh/hr` } } };
    new Chart(document.getElementById("timeline-chart"), { type: "line", data: { labels: data.charts.timeline.labels, datasets: [{ label: "Observed", data: data.charts.timeline.actual, borderColor: "#3975ed", borderWidth: 2, pointRadius: 0, tension: 0.18 }, { label: "Predicted", data: data.charts.timeline.predicted, borderColor: "#26a58e", borderDash: [4, 3], borderWidth: 2, pointRadius: 0, tension: 0.18 }] }, options: timelineOptions });
  }
} else if (document.querySelector("canvas")) {
  document.querySelectorAll(".chart-container").forEach((container) => {
    const notice = document.createElement("p"); notice.textContent = "Charts could not load. View the data tables or download the test results."; notice.className = "small-note"; container.replaceChildren(notice);
  });
}

if (page === "predict") {
  const form = document.getElementById("prediction-form");
  const submitButton = document.getElementById("predict-button");
  const message = document.getElementById("form-message");
  const comparison = document.getElementById("sample-comparison");
  const sampleButton = document.getElementById("sample-button");
  const activeFeatures = data.feature_columns;
  for (const key of ["temperature", "rainfall", "snowfall", "cloud_cover", "weather", "holiday"]) {
    const input = document.getElementById(key);
    const available = activeFeatures.includes(key);
    input.disabled = !available;
    const field = input.closest(".field") || input.closest(".checkbox-field");
    if (field) field.hidden = !available;
  }
  // A local input date is intentionally independent of the user's browser timezone.
  document.getElementById("date").value = data.sample.date;
  let requestVersion = 0;
  let pending = false;
  const bodyForForm = () => ({ dataset_id: data.dataset.id, ...Object.fromEntries(
    ["date", ...activeFeatures.filter((key) => !["weekday", "month"].includes(key))].map((key) =>
      [key, key === "holiday" ? document.getElementById(key).checked : document.getElementById(key).value])) });
  function invalidateResult() {
    requestVersion++;
    document.getElementById("result-content").hidden = true;
    document.getElementById("result-empty").hidden = false;
    document.getElementById("result-heading").textContent = "A look at the next hour.";
  }
  form.addEventListener("input", invalidateResult);
  form.addEventListener("change", invalidateResult);
  sampleButton.addEventListener("click", () => {
    for (const [key, value] of Object.entries(data.sample)) {
      const input = document.getElementById(key);
      if (key === "holiday") input.checked = value;
      else input.value = typeof value === "number" ? Number(value.toFixed(8)) : value;
    }
    invalidateResult(); clearErrors();
    document.getElementById("date").focus();
  });
  function clearErrors() {
    document.querySelectorAll(".field-error").forEach((element) => element.textContent = "");
    form.querySelectorAll("[aria-invalid]").forEach((element) => element.removeAttribute("aria-invalid"));
    message.hidden = true;
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pending) return;
    clearErrors();
    const payload = bodyForForm();
    const version = ++requestVersion;
    pending = true; submitButton.disabled = true; sampleButton.disabled = true; form.setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const result = await response.json();
      if (version !== requestVersion) return;
      if (!response.ok) {
        for (const [key, error] of Object.entries(result.errors || {})) {
          const target = document.getElementById(`${key}-error`);
          if (target) { target.textContent = error; document.getElementById(key)?.setAttribute("aria-invalid", "true"); }
        }
        message.textContent = result.message || "Check the inputs and try again."; message.hidden = false;
        form.querySelector("[aria-invalid=true]")?.focus(); return;
      }
      const conditions = result.conditions;
      document.getElementById("result-empty").hidden = true;
      document.getElementById("result-content").hidden = false;
      document.getElementById("result-heading").textContent = "Your traffic estimate.";
      document.getElementById("prediction-value").textContent = integer.format(result.predicted_volume);
      const date = new Date(`${conditions.date}T12:00:00`);
      const dateLabel = date.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" });
      document.getElementById("result-date").textContent = `${dateLabel} · ${String(conditions.hour).padStart(2, "0")}:00–${String(conditions.hour).padStart(2, "0")}:59 local time`;
      const chipLabels = { temperature: (value) => `${measurement.format(value)} °C`, weather: (value) => value,
        cloud_cover: (value) => `${measurement.format(value)}% cloud`, rainfall: (value) => `${measurement.format(value)} mm rain`,
        snowfall: (value) => `${measurement.format(value)} mm snow`, holiday: (value) => value ? "Holiday indicator" : "No holiday indicator" };
      const chips = Object.entries(chipLabels).filter(([key]) => key in conditions).map(([key, format]) => format(conditions[key]));
      document.getElementById("result-conditions").replaceChildren(...chips.map((text) => { const span = document.createElement("span"); span.textContent = text; return span; }));
      const matchesSample = Object.entries(data.sample).every(([key, value]) => ["date", "weather", "holiday"].includes(key) ? conditions[key] === value : Math.abs(Number(conditions[key]) - value) < 1e-7);
      comparison.hidden = !matchesSample;
      if (matchesSample) comparison.textContent = `${data.sample_is_test === false ? "This training example" : "This held-out example"} recorded ${integer.format(data.sample_actual)} vehicles/hour. Prediction difference: ${integer.format(Math.abs(result.predicted_volume - data.sample_actual))} vehicles/hour.${data.sample_imputed ? " Missing weather readings use training-set medians." : ""}`;
      const outside = conditions.date < data.start.slice(0, 10) || conditions.date > data.end.slice(0, 10);
      const note = document.getElementById("extrapolation-note"); note.hidden = !outside;
      note.textContent = "This date is outside the historical data range. The estimate uses learned calendar and weather patterns, not current road conditions.";
    } catch (_error) {
      if (version === requestVersion) { message.textContent = "Could not reach the prediction service. Check that the local server is running, then try again."; message.hidden = false; }
    } finally {
      pending = false; submitButton.disabled = false; sampleButton.disabled = false; form.removeAttribute("aria-busy");
    }
  });
}

if (page === "datasets") {
  const uploadForm = document.getElementById("dataset-upload-form");
  const preview = document.getElementById("dataset-preview");
  const notice = document.getElementById("dataset-message");
  const trainButton = document.getElementById("train-dataset-button");
  const unitField = document.getElementById("temperature-unit-field");
  const unitSelect = document.getElementById("temperature-unit");
  const locationField = document.getElementById("location-choice-field");
  const locationSelect = document.getElementById("location-choice");
  let inspection = null;
  let busy = false;
  function notify(text, isError = false) {
    notice.textContent = text; notice.hidden = false;
    notice.classList.toggle("error", isError);
    notice.setAttribute("role", isError ? "alert" : "status");
  }
  function setBusy(value) {
    busy = value;
    document.querySelectorAll(".activate-dataset, #inspect-button, #train-dataset-button, #dataset-upload-form input, #temperature-unit, #location-choice").forEach(element => { element.disabled = value; });
    uploadForm.setAttribute("aria-busy", String(value));
  }
  async function readResponse(response) {
    let body;
    try { body = await response.json(); }
    catch (_error) { throw new Error("The server could not complete this request. Your current dataset remains available."); }
    if (!response.ok) throw new Error(body.message || "Could not process this dataset.");
    return body;
  }
  uploadForm.addEventListener("input", () => { if (!busy) { inspection = null; preview.hidden = true; notice.hidden = true; } });
  uploadForm.addEventListener("submit", async event => {
    event.preventDefault(); if (busy) return;
    const file = document.getElementById("dataset-file").files[0];
    if (!file) { notify("Choose a traffic CSV file first.", true); return; }
    if (file.size > 10 * 1024 * 1024) { notify("Choose a CSV file no larger than 10 MB.", true); return; }
    const body = new FormData(uploadForm);
    setBusy(true); inspection = null; preview.hidden = true; notify("Reading the CSV and detecting traffic columns…");
    try {
      inspection = await readResponse(await fetch("/api/datasets/inspect", { method: "POST", body }));
      document.getElementById("preview-summary").textContent = `${integer.format(inspection.rows)} observations · ${integer.format(inspection.unique_hours)} unique location-hours · ${inspection.start.slice(0,10)} to ${inspection.end.slice(0,10)}. ${integer.format(inspection.duplicates)} repeated location-hours will be removed.`;
      document.getElementById("detected-columns").replaceChildren(...Object.entries(inspection.mapping).map(([field, column]) => {
        const row = document.createElement("div"); const source = document.createElement("strong"); const target = document.createElement("span");
        source.textContent = column; target.textContent = field.replaceAll("_", " "); row.append(source, target); return row;
      }));
      unitField.hidden = !("temperature" in inspection.mapping); unitSelect.value = inspection.temperature_unit || "";
      locationField.hidden = !inspection.locations.length;
      locationSelect.replaceChildren();
      if (inspection.locations.length > 1) { const option = new Option("Choose one location", ""); locationSelect.append(option); }
      for (const location of inspection.locations) {
        const option = new Option(`${location.value} (${integer.format(location.hours)} hours)`, location.value);
        option.disabled = location.hours < 100; locationSelect.append(option);
      }
      if (inspection.locations.length === 1) locationSelect.value = inspection.locations[0].value;
      document.getElementById("preview-rows").replaceChildren(...inspection.preview.map(row => {
        const tr = document.createElement("tr");
        for (const value of [row.date_time.replace("T", " "), measurement.format(row.traffic_volume)]) {
          const td = document.createElement("td"); td.textContent = value; tr.append(td);
        }
        return tr;
      }));
      preview.hidden = false; notify("Traffic columns detected. Review the preview, then train and use this dataset.");
    } catch (error) { notify(error.message || "Could not reach the local server. Try again.", true); }
    finally { setBusy(false); }
  });
  trainButton.addEventListener("click", async () => {
    if (busy || !inspection) return;
    if (!unitField.hidden && !unitSelect.value) { notify("Choose Celsius or Kelvin for the source temperature column.", true); unitSelect.focus(); return; }
    if (!locationField.hidden && !locationSelect.value) { notify("Choose one detected road or station before training.", true); locationSelect.focus(); return; }
    setBusy(true); notify("Training Linear Regression and evaluating the held-out hours. Your current model remains available until training succeeds…");
    try {
      await readResponse(await fetch("/api/datasets/train", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: inspection.id, temperature_unit: unitSelect.value || null, location: locationSelect.value || null }) }));
      window.location.assign("/");
    } catch (error) { notify(error.message || "Training could not finish. Your current dataset remains active.", true); setBusy(false); }
  });
  document.querySelectorAll(".activate-dataset").forEach(button => button.addEventListener("click", async () => {
    if (busy) return; setBusy(true); notify("Loading the selected dataset…");
    try {
      await readResponse(await fetch("/api/datasets/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: button.dataset.datasetId }) }));
      window.location.reload();
    } catch (error) { notify(error.message || "Could not switch datasets. Try again.", true); setBusy(false); }
  }));
}
