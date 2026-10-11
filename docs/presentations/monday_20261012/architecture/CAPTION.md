# Suggested figure caption

Direct, lead-conditioned U-Net with hydraulic halo (implemented V5-B design).
Central terrain, current hydraulic state and one preceding state form spatial
features. A separate CNN encodes a 96 × 96 observed hydraulic neighborhood;
its central 80 × 80 feature crop is added to the tile features before U-Net
downsampling. Causal forcing through the requested target, tile attributes,
origin/target times and lead condition the 10 × 10 bottleneck. Decoder skip
features support central predictions of depth, signed unit discharge and wet
probability. Gated residuals update the observed origin directly at the requested
lead; intermediate predicted states are not fed back. The activity gate applies
to depth and discharge residuals, while the wet head is separate.

The figure describes the implemented method, not evidence of improved forecasting.
Corrected-data comparisons remain pending. Graph context, local forcing and the
optional extra discharge-correction gate are separate variants and are omitted.
Masking, normalizations and nonlinearities are simplified for readability.

Sources: stage2/model.py, stage2/data.py, triton_core/model.py,
docs/stage2/V5B_EXPERIMENT_REPORT.md, docs/stage2/TARGET_DATA_AUDIT.md.

Use the PNG for easy slide insertion, PDF for print, or SVG for vector editing
in Inkscape/Illustrator (text remains editable). regenerate with:
MPLCONFIGDIR=/tmp/triton-architecture-mpl /ccs/home/haoranniu/miniconda3/envs/triton/bin/python docs/presentations/monday_20261012/draw_architecture.py
