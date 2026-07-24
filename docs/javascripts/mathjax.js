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
  if (typeof MathJax !== "undefined" && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(() => {
      if (MathJax.typesetClear) {
        MathJax.typesetClear();
      }
      return MathJax.typesetPromise();
    }).catch((err) => console.error("MathJax error:", err));
  } else if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    if (MathJax.typesetClear) {
      MathJax.typesetClear();
    }
    MathJax.typesetPromise();
  }
});
