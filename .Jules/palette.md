## 2024-05-23 - Consistent Reference Lines
**Learning:** Users process visual information faster when reference elements (like identity lines or zero lines) share a consistent visual language. Using a subtle, neutral color (gray) instead of varying or aggressive colors (red, black) allows the actual data to stand out and reduces cognitive load.
**Action:** When designing a suite of plots, standardize all reference/baseline elements to use the same subtle style (e.g., `color="gray", alpha=0.5, linestyle="--"`) across all visualizations.

## 2024-05-24 - Invisible UX & Documentation Integrity
**Learning:** UX isn't just what users see; it's also about security (removing polyfill.io) and trust (accurate documentation). Broken docs or insecure scripts erode user confidence even if the UI looks "nice".
**Action:** Always verify that documentation dependencies and API references match the actual codebase state. "Invisible" cleanups often have the highest ROI for project health.
