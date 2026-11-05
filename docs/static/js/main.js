/* RI-Prune project page: interactive results tables, theme toggle, BibTeX copy. */

// Numbers transcribed from Tables 1-3 of the paper (arXiv:2611.06047).
// Each cell: [accuracy, relative performance vs. vanilla (%)].
const BENCHES = ["Mantis-Eval", "MIBench", "MIRB", "MUIRBench", "Avg."];

const VENUES = {
  "FastV": "ECCV'24",
  "VisionZip": "CVPR'25",
  "VTW": "AAAI'25",
  "DivPrune": "CVPR'25",
  "SCOPE": "NeurIPS'25",
};

const RESULTS = {
  "qwen25": {
    label: "Qwen2.5-VL-7B",
    budgetLabel: "Retained tokens",
    vanilla: [[73.3, 100.0], [54.6, 100.0], [63.6, 100.0], [60.9, 100.0], [63.1, 100.0]],
    budgets: {
      "45.3%": {
        "FastV":     [[71.9, 98.1], [54.0, 98.9], [56.0, 88.1], [51.5, 84.6], [58.3, 92.5]],
        "VisionZip": [[71.9, 98.1], [52.1, 95.4], [49.5, 77.8], [55.9, 91.8], [57.3, 90.9]],
        "VTW":       [[55.8, 76.1], [45.6, 83.5], [42.1, 66.3], [44.4, 72.9], [47.0, 74.4]],
        "DivPrune":  [[71.9, 98.1], [54.4, 99.6], [60.0, 94.3], [59.5, 97.7], [61.4, 97.4]],
        "SCOPE":     [[72.4, 98.7], [54.4, 99.6], [58.3, 91.7], [57.4, 94.3], [60.6, 96.1]],
        "RI-Prune":  [[72.8, 99.4], [54.5, 99.8], [58.0, 91.2], [60.9, 100.0], [61.5, 97.5]],
      },
      "23.5%": {
        "FastV":     [[70.5, 96.2], [52.3, 95.8], [55.1, 86.7], [49.4, 81.1], [56.8, 90.1]],
        "VisionZip": [[68.7, 93.7], [48.5, 88.8], [47.4, 74.5], [54.2, 89.0], [54.7, 86.7]],
        "VTW":       [[38.3, 52.2], [26.1, 47.8], [43.3, 68.2], [34.1, 56.0], [35.4, 56.2]],
        "DivPrune":  [[71.0, 96.9], [53.2, 97.4], [57.5, 90.4], [56.4, 92.6], [59.5, 94.3]],
        "SCOPE":     [[71.9, 98.1], [52.2, 95.6], [49.4, 77.6], [55.7, 91.5], [57.3, 90.8]],
        "RI-Prune":  [[72.8, 99.4], [53.2, 97.4], [57.5, 90.4], [59.4, 97.5], [60.7, 96.2]],
      },
      "4.6%": {
        "FastV":     [[61.8, 84.3], [33.5, 61.3], [44.0, 69.2], [40.5, 66.5], [44.9, 71.2]],
        "VisionZip": [[59.5, 81.2], [39.0, 71.4], [41.3, 65.0], [47.4, 77.8], [46.8, 74.2]],
        "VTW":       [[14.3, 19.5], [22.5, 41.2], [23.0, 36.2], [31.8, 52.2], [22.9, 36.3]],
        "DivPrune":  [[62.7, 85.6], [37.5, 68.7], [44.5, 70.1], [49.1, 80.6], [48.4, 76.8]],
        "SCOPE":     [[64.5, 88.0], [39.5, 72.3], [43.2, 68.0], [47.5, 78.0], [48.7, 77.1]],
        "RI-Prune":  [[63.6, 86.8], [36.7, 67.2], [50.3, 79.1], [52.3, 85.9], [50.7, 80.4]],
      },
    },
    defaultBudget: "4.6%",
  },
  "qwen3": {
    label: "Qwen3-VL-8B",
    budgetLabel: "Retained tokens",
    vanilla: [[78.7, 100.0], [60.85, 100.0], [68.75, 100.0], [64.4, 100.0], [68.2, 100.0]],
    budgets: {
      "23.5%": {
        "FastV":     [[69.4, 88.2], [55.1, 90.6], [59.3, 86.3], [60.3, 93.6], [61.0, 89.5]],
        "VisionZip": [[74.6, 94.8], [53.8, 88.4], [62.4, 90.8], [60.9, 94.6], [62.9, 92.3]],
        "VTW":       [[63.7, 80.9], [48.0, 78.9], [40.1, 58.3], [58.5, 90.8], [52.6, 77.1]],
        "DivPrune":  [[64.7, 82.2], [51.2, 84.1], [60.1, 87.4], [61.0, 94.7], [59.3, 86.9]],
        "SCOPE":     [[75.1, 95.4], [58.7, 96.5], [63.0, 91.6], [62.7, 97.4], [64.9, 95.2]],
        "RI-Prune":  [[75.5, 95.9], [57.2, 94.0], [66.0, 96.0], [64.1, 99.5], [65.7, 96.4]],
      },
      "4.6%": {
        "FastV":     [[60.4, 76.7], [45.5, 74.8], [52.1, 75.8], [49.6, 77.0], [51.9, 76.1]],
        "VisionZip": [[68.2, 86.7], [51.2, 84.1], [52.9, 76.9], [57.3, 89.0], [57.4, 84.2]],
        "VTW":       [[54.2, 68.9], [43.5, 71.5], [62.1, 90.3], [42.3, 65.7], [50.5, 74.1]],
        "DivPrune":  [[54.2, 68.9], [47.6, 78.2], [52.2, 75.9], [59.9, 93.0], [53.5, 78.4]],
        "SCOPE":     [[74.7, 94.9], [47.5, 78.1], [52.5, 76.4], [58.3, 90.5], [58.3, 85.4]],
        "RI-Prune":  [[73.2, 93.0], [53.2, 87.4], [65.8, 95.7], [61.6, 95.7], [63.5, 93.1]],
      },
    },
    defaultBudget: "4.6%",
  },
  "llavaov": {
    label: "LLaVA-OneVision-7B",
    budgetLabel: "Token budget",
    vanilla: [[61.29, 100.0], [51.37, 100.0], [52.27, 100.0], [36.10, 100.0], [50.3, 100.0]],
    budgets: {
      "5000": {
        "FastV":     [[43.9, 71.6], [42.67, 83.1], [35.7, 68.3], [29.6, 82.0], [38.0, 75.5]],
        "VisionZip": [[59.0, 96.3], [49.87, 97.1], [50.0, 95.7], [34.9, 96.7], [48.4, 96.4]],
        "VTW":       [[33.6, 54.8], [38.00, 74.0], [39.9, 76.3], [26.1, 72.3], [34.4, 68.5]],
        "DivPrune":  [[60.4, 98.5], [48.57, 94.5], [48.8, 93.4], [35.7, 98.9], [48.4, 96.2]],
        "SCOPE":     [[60.4, 98.5], [48.17, 93.8], [49.9, 95.5], [35.0, 97.0], [48.4, 96.2]],
        "RI-Prune":  [[60.7, 99.0], [46.77, 91.0], [51.2, 98.0], [36.0, 99.7], [48.7, 96.8]],
      },
      "3300": {
        "FastV":     [[42.9, 70.0], [41.67, 81.1], [34.7, 66.4], [26.0, 72.0], [36.3, 72.3]],
        "VisionZip": [[56.2, 91.7], [48.57, 94.5], [48.6, 93.0], [34.5, 95.6], [47.0, 93.5]],
        "VTW":       [[28.6, 46.7], [28.27, 55.0], [36.8, 70.4], [22.5, 62.3], [29.0, 57.8]],
        "DivPrune":  [[56.7, 92.5], [46.47, 90.5], [48.7, 93.2], [33.9, 93.9], [46.4, 92.4]],
        "SCOPE":     [[56.7, 92.5], [47.67, 92.8], [48.8, 93.4], [34.7, 96.1], [47.0, 93.5]],
        "RI-Prune":  [[56.4, 92.0], [46.07, 89.7], [49.9, 95.5], [34.8, 96.4], [46.8, 93.1]],
      },
      "1700": {
        "FastV":     [[39.9, 65.1], [40.07, 78.0], [30.1, 57.6], [25.1, 69.5], [33.8, 67.2]],
        "VisionZip": [[53.5, 87.3], [47.87, 93.2], [48.3, 92.4], [33.0, 91.4], [45.7, 90.9]],
        "VTW":       [[21.2, 34.6], [18.00, 35.0], [25.0, 47.8], [15.7, 43.5], [20.0, 39.8]],
        "DivPrune":  [[55.3, 90.2], [46.07, 89.7], [48.1, 92.0], [33.2, 92.0], [45.7, 90.9]],
        "SCOPE":     [[52.1, 85.0], [47.07, 91.6], [48.5, 92.8], [33.1, 91.7], [45.2, 89.9]],
        "RI-Prune":  [[54.2, 88.4], [44.87, 87.3], [49.2, 94.1], [33.5, 92.8], [45.4, 90.4]],
      },
    },
    defaultBudget: "5000",
    order: ["5000", "3300", "1700"],   // numeric-looking keys would otherwise iterate ascending
  },
};

const state = { model: "qwen25", budget: RESULTS.qwen25.defaultBudget };

function fmt(v) {
  // keep the paper's precision (e.g. 46.77 vs 60.7)
  return Math.abs(v * 10 - Math.round(v * 10)) < 1e-6 ? v.toFixed(1) : v.toFixed(2);
}

// Best (bold) / second-best (underline) among pruning methods, per column.
// Competition ranking: if two methods tie for best, nothing is marked second.
function ranks(rows) {
  return BENCHES.map((_, j) => {
    const vals = rows.map(([, cells]) => cells[j][0]).sort((a, b) => b - a);
    const best = vals[0];
    const second = vals[1] === best ? null : vals[1];
    return { best, second };
  });
}

function cell(acc, rel, cls = "") {
  return `<td class="${cls}">${fmt(acc)}<span class="rel">${rel.toFixed(1)}%</span></td>`;
}

function renderTable() {
  const m = RESULTS[state.model];
  const rows = Object.entries(m.budgets[state.budget]);
  const rk = ranks(rows);

  const head = `<thead><tr><th scope="col">Method</th>${BENCHES.map(b => `<th scope="col">${b}</th>`).join("")}</tr></thead>`;
  const vanilla = `<tr class="vanilla"><td>Vanilla (all tokens)</td>${m.vanilla.map(([a, r]) => cell(a, r)).join("")}</tr>`;
  const body = rows.map(([name, cells]) => {
    const ours = name === "RI-Prune";
    const label = ours ? "RI-Prune (Ours)" : `${name}<span class="venue-tag">${VENUES[name]}</span>`;
    const tds = cells.map(([a, r], j) => {
      const cls = a === rk[j].best ? "best" : a === rk[j].second ? "second" : "";
      return cell(a, r, cls);
    }).join("");
    return `<tr class="${ours ? "ours" : ""}"><td>${label}</td>${tds}</tr>`;
  }).join("");

  document.getElementById("results-table").innerHTML = `${head}<tbody>${vanilla}${body}</tbody>`;
  document.getElementById("results-caption").textContent =
    `${m.label} · ${m.budgetLabel.toLowerCase()}: ${state.budget}. Each cell shows accuracy (%) and, below it, performance relative to the unpruned model.`;
}

function renderBudgetButtons() {
  const m = RESULTS[state.model];
  document.getElementById("budget-label").textContent = m.budgetLabel;
  const seg = document.getElementById("budget-seg");
  seg.innerHTML = (m.order || Object.keys(m.budgets)).map(b =>
    `<button type="button" data-budget="${b}" aria-pressed="${b === state.budget}">${b}</button>`).join("");
  seg.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    state.budget = btn.dataset.budget;
    seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b === btn));
    renderTable();
  }));
}

function initResults() {
  const seg = document.getElementById("model-seg");
  seg.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    state.model = btn.dataset.model;
    state.budget = RESULTS[state.model].defaultBudget;
    seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b === btn));
    renderBudgetButtons();
    renderTable();
  }));
  renderBudgetButtons();
  renderTable();
}

/* ---------- theme ---------- */
function initTheme() {
  const root = document.documentElement;
  const btn = document.getElementById("theme-btn");
  try {
    const saved = localStorage.getItem("riprune-theme");
    if (saved) root.dataset.theme = saved;
  } catch (e) { /* storage unavailable */ }
  btn.addEventListener("click", () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("riprune-theme", root.dataset.theme); } catch (e) { /* ignore */ }
  });
}

/* ---------- BibTeX copy ---------- */
function initCopy() {
  const btn = document.getElementById("copy-bib");
  btn.addEventListener("click", async () => {
    const text = document.getElementById("bibtex").innerText;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
    } catch (e) {
      const r = document.createRange();
      r.selectNodeContents(document.getElementById("bibtex"));
      const s = getSelection(); s.removeAllRanges(); s.addRange(r);
      btn.textContent = "Press Ctrl+C";
    }
    setTimeout(() => (btn.textContent = "Copy"), 1600);
  });
}

/* ---------- math ---------- */
function initMath() {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initResults();
  initCopy();
  initMath();
});
