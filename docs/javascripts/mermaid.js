// Mermaid diagram initialization for zensical modern theme.
// Loads mermaid from CDN and renders all .mermaid elements on page load.

(function () {
  function initMermaid() {
    if (typeof mermaid === "undefined") return;
    var dark = document.querySelector("[data-md-color-scheme=\"slate\"]") !== null;
    mermaid.initialize({
      startOnLoad: true,
      theme: dark ? "dark" : "default",
      securityLevel: "loose",
      fontFamily: getComputedStyle(document.body).fontFamily,
    });
  }

  // Load mermaid from CDN and initialize.
  var script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
  script.defer = true;
  script.onload = initMermaid;
  document.head.appendChild(script);

  // Re-render on theme toggle (palette switch).
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      if (m.attributeName === "data-md-color-scheme") {
        // Reload page to re-render mermaid with new theme.
        // Mermaid doesn't support re-theming without re-render.
        var blocks = document.querySelectorAll(".mermaid");
        blocks.forEach(function (el) { el.innerHTML = el.dataset.source || el.textContent; });
        if (typeof mermaid !== "undefined") {
          var dark = document.querySelector("[data-md-color-scheme=\"slate\"]") !== null;
          mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "default" });
          mermaid.run({ nodes: blocks });
        }
      }
    });
  });
  observer.observe(document.documentElement, { attributes: true });
})();
