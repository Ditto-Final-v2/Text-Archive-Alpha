(function () {
  if (window.__taImageViewerInit) return;
  window.__taImageViewerInit = true;

  function ensureViewer() {
    let root = document.getElementById("ta-image-viewer");
    if (root) return root;

    root = document.createElement("div");
    root.id = "ta-image-viewer";
    root.className = "ta-image-viewer";
    root.style.display = "none";
    root.innerHTML = [
      '<div class="ta-image-viewer-backdrop" data-iv-close="1"></div>',
      '<div class="ta-image-viewer-panel" role="dialog" aria-modal="true" aria-label="Image preview">',
      '  <button type="button" class="ta-image-viewer-close" data-iv-close="1" aria-label="Close preview">x</button>',
      '  <img class="ta-image-viewer-img" alt="">',
      '  <div class="ta-image-viewer-actions">',
      '    <a class="ta-image-viewer-download" href="#" download>Download</a>',
      "  </div>",
      "</div>",
    ].join("");
    document.body.appendChild(root);

    root.addEventListener("click", function (event) {
      if (event.target && event.target.getAttribute("data-iv-close") === "1") {
        closeViewer();
      }
    });

    return root;
  }

  function openViewer(previewUrl, downloadUrl, altText) {
    const root = ensureViewer();
    const img = root.querySelector(".ta-image-viewer-img");
    const download = root.querySelector(".ta-image-viewer-download");
    if (!img || !download) return;

    img.src = previewUrl || "";
    img.alt = altText || "Attachment preview";
    download.href = downloadUrl || previewUrl || "#";
    root.style.display = "block";
    document.body.classList.add("ta-image-viewer-open");
  }
  window.taOpenImageViewer = openViewer;

  function closeViewer() {
    const root = document.getElementById("ta-image-viewer");
    if (!root) return;
    const img = root.querySelector(".ta-image-viewer-img");
    if (img) img.src = "";
    root.style.display = "none";
    document.body.classList.remove("ta-image-viewer-open");
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeViewer();
    }
  });

  function findTrigger(target) {
    let el = (target && target.nodeType === 1) ? target : (target && target.parentElement ? target.parentElement : null);
    while (el) {
      if (el.getAttribute && el.getAttribute("data-image-viewer") === "1") return el;
      el = el.parentElement;
    }
    return null;
  }

  document.addEventListener("click", function (event) {
    const trigger = findTrigger(event.target);
    if (!trigger) return;
    event.preventDefault();
    event.stopPropagation();
    openViewer(
      trigger.getAttribute("data-preview-url"),
      trigger.getAttribute("data-download-url"),
      trigger.getAttribute("data-alt")
    );
  }, true);
})();
