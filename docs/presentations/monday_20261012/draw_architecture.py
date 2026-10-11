"""Export the implemented direct-plus-halo architecture as SVG, PDF and PNG."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent / 'architecture'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 13})
NAVY, TEAL, INK, MUTED = '#123147', '#007F83', '#243C4B', '#58707F'
fig, ax = plt.subplots(figsize=(20, 10.3))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 20), ylim=(10.3, 0))
ax.axis('off')
editable_items = []


def text(x, y, words, size=13, color=INK, weight='normal', ha='left'):
    editable_items.append(dict(kind='text', x=x, y=y, text=words, size=size,
                               color=color, bold=weight=='bold'))
    ax.text(x, y, words, fontsize=size, color=color, fontweight=weight,
            ha=ha, va='top', linespacing=1.45, zorder=5)


def box(x, y, w, h, title, body='', fill='#EEF4F6', edge='#D1E0E6', accent=NAVY):
    editable_items.append(dict(kind='box', x=x, y=y, w=w, h=h, fill=fill, edge=edge))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.10',
                              linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=3))
    text(x+.18, y+.18, title, 14, accent, 'bold')
    if body:
        text(x+.18, y+.59, body, 12)


def arrow(points, color=TEAL, dashed=False):
    editable_items.append(dict(kind='arrow', points=points, color=color, dashed=dashed))
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                lw=1.8, ls='--' if dashed else '-', zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=17, lw=1.8, color=color,
                                linestyle='--' if dashed else '-', zorder=2))


text(.55, .35, 'Direct U-Net with hydraulic halo', 27, NAVY, 'bold')
text(.57, .96, 'Observed-state, lead-conditioned flood forecasting  |  Implemented V5-B design', 14, MUTED)

box(7.7, 1.55, 5.15, 1.15, 'Forcing and forecast conditioning',
    'Causal forcing through t + L\nTile, origin, target and lead embeddings',
    fill='#E9EEF5', edge='#BDCDD9')
# Keep the long conditioning caption on two explicit lines.
ax.texts[-1].set_text('Causal forcing through t + L\nTile, origin, target and lead embeddings')
ax.texts[-1].set_fontsize(11.5)

box(.60, 3.02, 3.1, 1.12, 'Central terrain + mask', '6 terrain layers; 80 × 80')
box(.60, 4.64, 3.1, 1.42, 'Observed hydraulic history',
    'S(t) and S(t − 1)\nS = [h, HU, HV]')
box(4.28, 3.61, 2.57, 1.60, 'Central feature fusion',
    'Static and state CNNs\nHistory-difference adapter\n80 × 80 features')
arrow([(3.7, 3.58), (3.96, 3.58), (3.96, 4.20), (4.28, 4.20)])
arrow([(3.7, 5.33), (3.96, 5.33), (3.96, 4.72), (4.28, 4.72)])

# Separate hydraulic context branch, precisely matching model.py.
ax.add_patch(FancyBboxPatch((.38, 6.83), 6.72, 2.16,
                          boxstyle='round,pad=0.02,rounding_size=0.10',
                          facecolor='#F1FAF9', edgecolor='#8CC7C6', linewidth=1.2, zorder=1))
editable_items.append(dict(kind='box',x=.38,y=6.83,w=6.72,h=2.16,
                           fill='#F1FAF9',edge='#8CC7C6'))
box(.60, 7.06, 3.1, 1.63, 'Observed hydraulic halo',
    '96 × 96 at t and t − 1\n8-cell border around center\nCurrent state + history change',
    fill='#DCF0EE', edge='#86BEBC', accent=TEAL)
box(4.28, 7.06, 2.57, 1.63, 'Separate halo CNN',
    'Encode 96 × 96 context\nCrop central features\n80 × 80 feature crop',
    fill='#DCF0EE', edge='#86BEBC', accent=TEAL)
arrow([(3.70, 7.87), (4.28, 7.87)])

ax.add_patch(Circle((7.45, 4.40), .23, facecolor=TEAL, edgecolor=TEAL, zorder=4))
editable_items.append(dict(kind='circle',x=7.22,y=4.17,w=.46,h=.46,fill=TEAL,edge=TEAL))
editable_items.append(dict(kind='plus',x=7.45,y=4.40,text='+',color='#FFFFFF',size=19))
ax.text(7.45, 4.40, '+', va='center', ha='center', fontsize=19, color='white', zorder=5)
arrow([(6.85, 4.40), (7.20, 4.40)])
arrow([(6.85, 7.87), (7.45, 7.87), (7.45, 4.65)])

box(8.04, 3.61, 2.05, 1.60, 'Spatial encoder', '80 → 40 → 20\nCentral features')
arrow([(7.70, 4.40), (8.04, 4.40)])
box(10.65, 5.20, 2.15, 1.57, 'Bottleneck', '10 × 10\nConcatenate\nconditioning')
arrow([(10.09, 4.40), (10.36, 4.40), (10.36, 5.98), (10.65, 5.98)])
arrow([(11.72, 2.70), (11.72, 5.20)], NAVY)
text(11.88, 3.39, 'Conditioning', 11, NAVY)
box(13.48, 3.61, 2.45, 1.60, 'U-Net decoder', '20 → 40 → 80\nSkip-feature fusion')
arrow([(12.80, 5.98), (13.12, 5.98), (13.12, 4.40), (13.48, 4.40)])
arrow([(9.06, 3.61), (9.06, 3.05), (14.69, 3.05), (14.69, 3.61)], MUTED, dashed=True)
text(12.94, 2.81, 'U-Net skip features', 10, MUTED)

box(16.57, 3.26, 2.82, 2.05, 'Forecast heads',
    'Δh, ΔHU, ΔHV\nSpatial activity gate\nWet/dry logit head')
arrow([(15.93, 4.40), (16.57, 4.40)])
box(14.18, 7.06, 5.21, 1.65, 'Central forecast at t + L',
    'Observed S(t) + gated residual change\nDepth h ≥ 0; signed HU, HV; wet probability\n80 × 80 output',
    fill='#DCF0EE', edge='#86BEBC', accent=TEAL)
arrow([(17.98, 5.31), (17.98, 7.06)])

text(8.02, 7.36, 'One requested lead.\nOne model application.\nNo recursive hydraulic-state feedback.', 13, TEAL, 'bold')
ax.plot([.57, 19.4], [9.35, 9.35], color='#D1E0E6', linewidth=1)
editable_items.append(dict(kind='line',points=[(.57,9.35),(19.4,9.35)],color='#D1E0E6'))
text(.60, 9.55, 'Lead L: 0.5–24 h  •  HU/HV: signed unit discharge (m²/s)  •  Halo features are added before spatial downsampling', 12, INK)
text(.60, 9.95, 'Method schematic, not validated performance evidence. Corrected-data baseline and halo comparisons are pending.', 11, MUTED)

for extension in ('svg', 'pdf', 'png'):
    fig.savefig(OUT/f'direct_unet_halo_architecture.{extension}', dpi=200, facecolor='white')
plt.close(fig)
(OUT/'editable_diagram.json').write_text(json.dumps(editable_items, indent=2)+'\n')
(OUT/'CAPTION.md').write_text('''# Suggested figure caption

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
''')
print(OUT)
