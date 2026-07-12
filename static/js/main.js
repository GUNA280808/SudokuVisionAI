// Dark mode toggle (persists for the session via a JS variable, since
// artifacts/browsers here avoid localStorage; for a real deployment you
// can safely use localStorage in your own hosted app).
(function () {
  const root = document.documentElement;
  const toggleBtn = document.getElementById("darkModeToggle");

  function applyTheme(theme) {
    root.setAttribute("data-bs-theme", theme);
    if (toggleBtn) {
      const icon = toggleBtn.querySelector("i");
      icon.className = theme === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
    }
  }

  // Respect OS preference on first load
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(prefersDark ? "dark" : "light");

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = root.getAttribute("data-bs-theme");
      applyTheme(current === "dark" ? "light" : "dark");
    });
  }
})();

// Upload form loading overlay + button state
(function () {
  const form = document.getElementById("uploadForm");
  const overlay = document.getElementById("loadingOverlay");
  const uploadBtn = document.getElementById("uploadBtn");

  if (form) {
    form.addEventListener("submit", () => {
      const fileInput = document.getElementById("fileInput");
      if (!fileInput.files || fileInput.files.length === 0) return;
      if (overlay) overlay.classList.remove("d-none");
      if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.querySelector(".btn-text").innerHTML =
          '<span class="spinner-border spinner-border-sm me-1"></span>Solving...';
      }
    });
  }
})();

// Drag-and-drop dropzone visuals
(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
    }
  });
})();
