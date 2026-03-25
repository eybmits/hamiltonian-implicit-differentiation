# Plot Style Standard

All publication-facing figures must use the shared helper in [experiments/plot_style.py](../experiments/plot_style.py).

Exp 2 defines the canonical visual standard for the repository. All other experiment renderers are aligned to that same mechanism instead of carrying local figure sizing rules.

## Fixed Thesis Width

The final LaTeX document uses:

- `\textwidth = 422.52348 pt`
- publication plot width `= 0.98\textwidth`
- resulting physical plot width `= 5.7295 in`

Every final PDF is rendered directly at that physical size. The repository does not rely on LaTeX downscaling to make fonts “look right”.

## Shared Typography

`use_thesis_style()` applies the canonical paper settings:

- `text.usetex = True`
- `font.family = serif`
- `font.size = 10 pt`
- `axes.labelsize = 11 pt`
- `axes.titlesize = 11 pt`
- `xtick.labelsize = 10 pt`
- `ytick.labelsize = 10 pt`
- `legend.fontsize = 10 pt`
- white figure and axes backgrounds

The LaTeX preamble uses `fontenc` and `lmodern` so the PDF typography matches the thesis body text.

## Figure Geometry

The helper exposes the canonical figure sizes:

- `FIG_W = 5.7295 in`
- `FIG_H = 2.45 in` for standard single-panel figures
- `dual_panel_size()` for the exact Exp 2 one-row/two-panel layout
- `grid_size(n_rows)` for Exp 2 style stacked collages with a footer legend row

Final PDFs must be written with `savepdf()` / `save_figure()` and must not use `bbox_inches='tight'`, because that changes the physical output size and shrinks text in the final thesis.

## Layout Rules

- Use footer legends instead of squeezing legends into the data region when the panel would otherwise compress.
- Heatmaps that represent matrix-like comparisons must use `aspect="equal"` so cells stay square.
- Sixpack collages use square panel boxes via `set_box_aspect(1.0)`.
- Exp 1 story grids and the spectrum-compare collage use the same thesis typography and fixed-width layout as the rest of the suite.

## Repository Policy

- Do not add ad-hoc per-script font stacks or local savefig policies.
- Reuse the shared helper for every publication plot.
- Keep exploratory or probe plots out of the canonical checked-in artifact tree under `output/exp01` to `output/exp08`.
