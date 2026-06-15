window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  if (typeof MathJax === "undefined" || !MathJax.typesetPromise) {
    return;
  }
  if (MathJax.typesetClear) {
    MathJax.typesetClear();
  }
  MathJax.typesetPromise();
});
