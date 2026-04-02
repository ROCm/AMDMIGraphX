(function () {
  const tbody = document.getElementById("tbody");
  const statusEl = document.getElementById("status");
  const stateFilter = document.getElementById("stateFilter");
  const refreshBtn = document.getElementById("refreshBtn");
  const errorBanner = document.getElementById("errorBanner");
  const rateBanner = document.getElementById("rateBanner");
  const emptyEl = document.getElementById("empty");

  function fmtDate(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
    } catch {
      return iso;
    }
  }

  function setLoading(on) {
    statusEl.textContent = on ? "Loading…" : "";
    refreshBtn.disabled = on;
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function clearError() {
    errorBanner.classList.add("hidden");
    errorBanner.textContent = "";
  }

  async function load() {
    clearError();
    rateBanner.classList.add("hidden");
    emptyEl.classList.add("hidden");
    setLoading(true);
    tbody.innerHTML = "";

    const state = stateFilter.value;
    try {
      const res = await fetch(`/api/prs?state=${encodeURIComponent(state)}`);
      const data = await res.json();

      if (data.error) {
        const e = data.error;
        const msg =
          e.message ||
          (typeof e === "string" ? e : "Request failed");
        showError(
          e.status
            ? `GitHub API error (${e.status}): ${msg}`
            : String(msg)
        );
        setLoading(false);
        return;
      }

      const meta = data.meta || {};
      const parts = [];
      if (meta.rate_limit && meta.rate_limit.remaining !== "") {
        parts.push(`API rate limit remaining: ${meta.rate_limit.remaining}`);
      }
      if (meta.truncated) {
        parts.push(
          "GitHub Search returns at most 1000 results; list may be incomplete."
        );
      }
      if (parts.length) {
        rateBanner.textContent = parts.join(" — ");
        rateBanner.classList.remove("hidden");
      }

      const prs = data.pull_requests || [];
      if (prs.length === 0) {
        emptyEl.classList.remove("hidden");
        setLoading(false);
        return;
      }

      const frag = document.createDocumentFragment();
      for (const pr of prs) {
        const tr = document.createElement("tr");
        const num = pr.number;
        const url = pr.html_url || `https://github.com/ROCm/AMDMIGraphX/pull/${num}`;
        tr.innerHTML = `
          <td class="num"><a href="${url}" target="_blank" rel="noopener">#${num}</a></td>
          <td>
            <a href="${url}" target="_blank" rel="noopener">${escapeHtml(pr.title || "")}</a>
            ${pr.draft ? '<span class="badge draft">Draft</span>' : ""}
          </td>
          <td>${escapeHtml(pr.user || "—")}</td>
          <td><span class="badge ${pr.state === "open" ? "open" : "closed"}">${escapeHtml(pr.state || "—")}</span></td>
          <td class="num">${escapeHtml(fmtDate(pr.updated_at))}</td>
        `;
        frag.appendChild(tr);
      }
      tbody.appendChild(frag);
      statusEl.textContent = `${prs.length} PR${prs.length === 1 ? "" : "s"}`;
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  refreshBtn.addEventListener("click", load);
  stateFilter.addEventListener("change", load);
  load();
})();
