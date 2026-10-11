"""Replace slide 8 with native editable architecture shapes; preserve other slides."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'TRITON_Methods_and_Experiment_History.pptx'
OUTPUT = ROOT/'TRITON_Methods_with_Editable_Architecture.pptx'
ITEMS = json.loads((ROOT/'architecture/editable_diagram.json').read_text())

NOTES = '''Direct U-Net with hydraulic halo — detailed presenter walkthrough

Opening explanation
This figure describes our implemented V5-B architecture. It combines the observed hydraulic state, recent hydraulic history, terrain, forcing and a requested forecast lead. The central prediction remains an 80 × 80 tile. The hydraulic halo adds immediate spatial context around that tile. This is a methods explanation, not a claim that halo has already improved accuracy on the corrected data.

1. Define the forecast origin and outputs
Let t be the forecast origin and L the requested lead. S(t) contains depth h and two signed unit-discharge components HU and HV. HU/HV have units of m²/s, not velocity units. The older state is one half-hour before the origin. We request L from 1 to 48 half-hour steps, corresponding to 0.5–24 hours. The model predicts depth, wet probability and signed discharge at t+L. Velocity can be derived by dividing discharge by depth, with the declared wetness/denominator policy; shallow water requires separate diagnostics.

2. Central terrain and hydraulic history
The upper-left inputs contain six static raster layers and the valid-cell mask. Terrain layers include DEM, flow accumulation, stream mask, distance to stream, slope and relative elevation. Static features use their own CNN. The current hydraulic state is also encoded. In the two-state adapter configuration, the difference between current and older states goes through a learned history adapter before state encoding. Internally the depth state uses log1p depth, while the discharge components remain signed. The static and state features are added to form the central 80 × 80 spatial feature map. The schematic groups these operations for readability.

3. Why we introduced halo
An independent tile can hide water conditions just beyond its edge. The halo branch supplies observed hydraulic conditions around the central tile so the network can use nearby water levels and signed discharge when forming local spatial features. This is a hypothesis about useful information, rather than proof of a corrected-data improvement. It does not by itself provide watershed-wide upstream routing.

4. Exact halo implementation
With an eight-cell border on each side, an 80 × 80 center becomes a 96 × 96 hydraulic input. At the actual approximately 9.506 m cell spacing, this border is about 76 m wide. Both the current and older hydraulic neighborhoods are read. The halo CNN receives current hydraulic channels and their changes from the older neighborhood, with log-transformed depth. A separate convolutional encoder processes the 96 × 96 patch. Its central 80 × 80 feature crop is added at the plus sign to the central tile features BEFORE spatial downsampling. This is not an enlarged static-terrain U-Net. The supervised forecast output is still central-only. Halo influence depends on the encoder's receptive field; supplying a border does not imply that every border cell affects every output cell.

5. Forcing and forecast conditioning
The causal temporal encoder provides forcing information through the requested target time t+L, never after it. Tile attributes, target-time features, origin-time features and lead features are also encoded, concatenated and passed through a conditioning MLP. The resulting conditioning vector is expanded spatially and concatenated with the pooled raster features at the 10 × 10 bottleneck. Known future forcing is an explicit input assumption. This model does not predict weather or generate its own future hydrographs. Future hydraulic states are labels only and never enter these inputs.

6. The U-Net spatial pathway
Spatial features are downsampled from 80 to 40 to 20 cells, then pooled to the 10 × 10 bottleneck. The conditioned bottleneck is decoded back through 20, 40 and 80 cells using interpolation and convolutional blocks. Skip features from the matching encoder scales are concatenated into the decoder. The dashed connection is a schematic representation of these multiple scale-specific skip connections, not a single shared tensor. They help preserve fine spatial information while the bottleneck supplies broader context and forecast conditioning.

7. Heads and residual output construction
The decoder produces raw depth-change and signed-discharge-change maps, a wet/dry logit map and, in this configuration, a spatial activity gate. The sigmoid activity gate multiplies the depth and discharge residuals. The model adds those gated residuals to the observed origin state. Depth is clamped nonnegative; HU/HV can be positive or negative. The valid-cell mask restricts the hydraulic outputs. Wet probability is obtained from the separate wet head; it is not the same quantity as the activity gate. A validation-selected extent rule converts wet scores into a displayed binary mask.

8. Direct forecast versus recursive transition
For a requested lead, this model makes one application from the observed origin to the target. It does not feed an intermediate predicted hydraulic state back into itself 48 times. That removes recursive state feedback from the formulation. It does not guarantee lower error, physical conservation or accurate long-lead forecasting. A separate model application is needed for each requested lead if producing all 48 forecasts.

9. What is omitted from this main figure
Local forcing, an additional discharge-correction gate and graph communication are separate experimental variants. The implemented local graph adds learned tile-message context at the bottleneck; it is not represented as a water-flux operator here. The hierarchical HUC12 graph is future work. Algebraically coupling h and velocity to discharge is output consistency, not mass conservation. The figure also simplifies normalization, masking, nonlinearities and channel dimensions.

10. Evidence and how to present this honestly
The previous forecasting experiments used targets with scrambled chronology and lost negative discharge. Their metric rankings do not establish valid forecasting skill or prove a halo benefit. The repaired converter uses numeric counters and preserves signed components; corrected hydraulic events are being validated against the raw archive. The next scientific comparison is a freshly trained direct baseline versus persistence and halo on identical corrected event splits, origins, leads and losses. Train from scratch, fit normalization using training events only, and choose checkpoints and extent rules on validation events only. Test events remain held out. Report depth, extent and signed flow separately, and check shallow-water velocity amplification.

Suggested spoken explanation (about one minute)
“We start with the observed depth and signed flow at the forecast origin, one preceding state, and the terrain of the central tile. The turquoise lower branch is the halo: we read an eight-cell hydraulic border, encode that neighborhood separately, and add its central features to the tile representation. The U-Net then combines those spatial features with forcing and the requested lead at its bottleneck. Its decoder predicts gated changes in depth and discharge plus a wet probability. Adding the changes to the observed origin gives the requested future state directly, without a recursive sequence of predicted hydraulic states. This figure explains the implemented method; we are rebuilding the evidence for its performance on corrected targets.”

Sources
stage2/model.py — history adapter, halo feature crop, conditioning, decoder, heads and residual outputs.
stage2/data.py — direct origin/target pairing and observed neighborhood loading.
triton_core/model.py — causal event encoder and convolutional blocks.
docs/stage2/V5B_EXPERIMENT_REPORT.md — historical comparison designs; numerical claims superseded by the audit.
docs/stage2/TARGET_DATA_AUDIT.md — chronology/sign defects, repairs and validation scope.
'''

def color(value):
    return RGBColor.from_string(value.lstrip('#'))


def diagram(slide, slide_width, items=None, notes=NOTES):
    sx = slide_width/20
    sy = sx
    ox, oy = 0, .28
    for item in (ITEMS if items is None else items):
        kind=item['kind']
        if kind in ('box','circle'):
            shape=slide.shapes.add_shape(MSO_SHAPE.OVAL if kind=='circle' else MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(ox+item['x']*sx), Inches(oy+item['y']*sy),
                Inches(item['w']*sx), Inches(item['h']*sy))
            shape.fill.solid();shape.fill.fore_color.rgb=color(item['fill'])
            shape.line.color.rgb=color(item['edge']);shape.line.width=Pt(.85)
            if kind=='box':shape.adjustments[0]=.06
        elif kind in ('text','plus'):
            x,y=item['x'],item['y']
            if kind=='plus':x-=.23;y-=.23
            # Room for explicit lines; text stays native and individually editable.
            width=min(20-x, max(1.0, max(map(len,item['text'].splitlines()))*.125))*sx
            height=(len(item['text'].splitlines())*item['size']/72*1.5+.12)*sy
            if kind=='plus':width=.46*sx;height=.46*sy
            shape=slide.shapes.add_textbox(Inches(ox+x*sx), Inches(oy+y*sy), Inches(width), Inches(height))
            tf=shape.text_frame;tf.word_wrap=False
            if kind=='plus':tf.vertical_anchor=MSO_ANCHOR.MIDDLE
            tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
            for i,line in enumerate(item['text'].splitlines()):
                p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
                p.text=line;p.font.name='DejaVu Sans'
                if kind=='plus':p.alignment=PP_ALIGN.CENTER
                p.font.size=Pt(item['size']*sx)
                p.font.color.rgb=color(item['color']);p.font.bold=item.get('bold',False)
                p.line_spacing=Pt(item['size']*sx*1.45)
                p.space_before=p.space_after=Pt(0)
        else:
            points=item['points']
            for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
                shape=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                    Inches(ox+a[0]*sx), Inches(oy+a[1]*sy),
                    Inches(ox+b[0]*sx), Inches(oy+b[1]*sy))
                shape.line.color.rgb=color(item['color']);shape.line.width=Pt(1.15)
                if item.get('dashed'):shape.line.dash_style=MSO_LINE_DASH_STYLE.DASH
                if kind=='arrow' and i==len(points)-2:
                    end=OxmlElement('a:tailEnd');end.set('type','triangle')
                    shape.line._get_or_add_ln().append(end)
    slide.notes_slide.notes_text_frame.text=notes


def build():
    prs=Presentation(SOURCE)
    slide=prs.slides[7]
    for shape in list(slide.shapes):
        element=shape._element;element.getparent().remove(element)
    diagram(slide, prs.slide_width/914400)
    prs.save(OUTPUT)
    standalone=Presentation()
    standalone.slide_width=Inches(20);standalone.slide_height=Inches(11)
    single=standalone.slides.add_slide(standalone.slide_layouts[6])
    diagram(single,20)
    standalone.save(ROOT/'architecture/Editable_Architecture.pptx')
    (ROOT/'architecture/DETAILED_SPEAKER_NOTES.md').write_text(NOTES)
    reopened=Presentation(OUTPUT)
    assert len(reopened.slides)==len(prs.slides)==21
    architecture=reopened.slides[7]
    assert not any(s.shape_type==13 for s in architecture.shapes), 'Diagram contains a raster picture'
    assert sum(s.has_text_frame for s in architecture.shapes)>20
    assert 'Exact halo implementation' in architecture.notes_slide.notes_text_frame.text
    assert 'S(t) and S(t − 1)' in '\n'.join(s.text for s in architecture.shapes if s.has_text_frame)
    print(f'Updated deck: {OUTPUT}\nSlide 8: {len(architecture.shapes)} native editable shapes; detailed notes verified.')


if __name__=='__main__':
    build()
