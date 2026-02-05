## 2024-05-23 - Consistent Reference Lines
**Learning:** Users process visual information faster when reference elements (like identity lines or zero lines) share a consistent visual language. Using a subtle, neutral color (gray) instead of varying or aggressive colors (red, black) allows the actual data to stand out and reduces cognitive load.
**Action:** When designing a suite of plots, standardize all reference/baseline elements to use the same subtle style (e.g., `color="gray", alpha=0.5, linestyle="--"`) across all visualizations.

## 2024-05-24 - Invisible UX & Documentation Integrity
**Learning:** UX isn't just what users see; it's also about security (removing polyfill.io) and trust (accurate documentation). Broken docs or insecure scripts erode user confidence even if the UI looks "nice".
**Action:** Always verify that documentation dependencies and API references match the actual codebase state. "Invisible" cleanups often have the highest ROI for project health.

## 2024-10-27 - Histogram Bar Alignment
**Learning:** In histograms plotted over fixed bins, using the data mean within each bin as the bar's x-coordinate causes visual misalignment and clutter, especially with fixed-width bars. It misrepresents the bin structure to the user.
**Action:** Always center histogram bars at the geometric center of the bins, regardless of the data distribution within the bin, to clearly communicate the binning structure.

## 2025-02-17 - Reference Line Labels
**Learning:** Users rely on legends to interpret reference lines (e.g., identity lines, zero lines). Inconsistent or missing labels (e.g., "Perfect calibration" vs "Perfectly Calibrated", or no label for zero lines) increase cognitive load.
**Action:** Explicitly label all reference lines in diagnostic plots (e.g., "Perfectly Calibrated", "Perfect Prediction") and ensure they appear in the legend.

## 2025-02-18 - Visual Separation in Composite Plots
**Learning:** When combining two distinct plot types (e.g., a line plot and a histogram) in the same axes, using a visual separator (like a zero line) significantly improves readability by grounding the secondary plot and preventing visual bleeding.
**Action:** Always include a subtle separator line (e.g., at y=0) when appending secondary visualizations like histograms to the bottom of a main plot.
