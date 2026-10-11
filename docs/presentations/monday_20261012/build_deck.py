"""Build an editable methods deck, SVG preview, notes, and ChatGPT source bundle.

Run with Python 3.11 and python-pptx/Pillow available (temporary dependencies
used for initial creation: PYTHONPATH=/tmp/triton-slides-deps).
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAVY, TEAL, GOLD = '#123147', '#007F83', '#B86627'
INK, MUTED, PALE, WHITE = '#243C4B', '#58707F', '#EEF4F6', '#FFFFFF'
FONT = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'
BOLD_FONT = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf'
W, H = 13.333333, 7.5
SRC = {
    'peak': 'docs/project/MODEL_DESCRIPTION.md',
    'early': 'docs/project/MODEL_STATUS_AND_CHALLENGES_SUMMARY.md',
    'timestamp': 'archive/stage1/docs/STAGE1_TIMESTAMP_SURROGATE.md',
    'max': 'archive/stage1/docs/STAGE1_MAX_PERFORMANCE.md',
    'v4': 'archive/stage1/docs/STAGE1_V4_FINE_TUNING.md',
    'progress': 'archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md',
    'v1': 'archive/stage1/docs/STAGE1_TRANSITION_V1.md',
    'v2': 'archive/stage1/docs/STAGE1_TRANSITION_V2.md',
    'v3': 'archive/stage1/docs/STAGE1_TRANSITION_V3.md',
    'v4a': 'archive/stage1/docs/STAGE1_TRANSITION_V4A.md',
    'v4b': 'archive/stage1/docs/STAGE1_TRANSITION_V4B.md',
    'semantics': 'archive/stage1/docs/TRITON_COMPONENT_SEMANTICS_AUDIT.md',
    'plan': 'docs/stage2/STAGE2_V5_PLAN.md',
    'report': 'docs/stage2/V5B_EXPERIMENT_REPORT.md',
    'graph': 'docs/stage2/V5D_GRAPH_COMMUNICATION.md',
    'audit': 'docs/stage2/TARGET_DATA_AUDIT.md',
    'briefing': 'docs/stage2/LEADERSHIP_BRIEFING.md',
    'model': 'stage2/model.py',
    'readme': 'README.md',
}
STAMP = datetime.now(ZoneInfo('America/New_York')).strftime('%B %d, %Y, %I:%M %p Eastern')

# A method catalog, rather than a tally of jobs, seeds, or checkpoints.
# Older apparent successes/failures are not treated as corrected-data findings.
HISTORY = [
 ('Early maps', 'Blockwise peak V2', 'Depth and wet/dry heads; event + terrain inputs', 'Historical method', 'early'),
 ('Early maps', 'Blockwise peak V3', 'Raster features, depth-weighted Huber, wet classification', 'Historical method', 'peak'),
 ('Timestamp', 'Timestamp baseline', 'Predict requested state from forcing, time and terrain', 'Legacy-data run', 'timestamp'),
 ('Timestamp', 'Label-aware / stratified sampling', 'Represent dry, boundary, wet and deep patches', 'Legacy-data run', 'progress'),
 ('Timestamp', 'Stage-1 Max', 'Dense index, whole-batch balancing, hybrid depth loss', 'Legacy-data run', 'max'),
 ('Timestamp', 'Calibrated wet gating', 'Select an extent rule and gate displayed depth', 'Legacy-data diagnostic', 'progress'),
 ('Timestamp', 'Timestamp V4 fine-tuning', 'Strict quotas, depth/wet coupling, flow-aware losses', 'Legacy-data run', 'v4'),
 ('Transitions', 'Transition V1', 'Observed state + forcing -> residual next-state update', 'Legacy-data run', 'v1'),
 ('Transitions', 'Continued V1', 'Lower-rate training of the one-step reference', 'Legacy-data run', 'v1'),
 ('Transitions', 'Multi-step V1', 'Six-step scheduled exposure to predicted states', 'Legacy-data run', 'v1'),
 ('Transitions', 'Regime-aware V2', 'Filling/draining/stable sampling + delta/storage objectives', 'Legacy-data run', 'v2'),
 ('Transitions', 'Exact-local V3', 'Paired M6 labels, 12-step rollout, velocity MSE', 'Legacy-data run', 'v3'),
 ('Transitions', 'History + gate V4a', 'Two hydraulic states; spatial gate on residual changes', 'Legacy-data run', 'v4a'),
 ('Transitions', 'Adapter + gate V4b', 'Separate history adapter; faster training of new modules', 'Legacy-data run', 'v4b'),
 ('Transitions', 'Component-policy diagnostics', 'Compare learned flow, persistent HU/HV and persistent u/v', 'Legacy-data diagnostic', 'plan'),
 ('Direct / context', 'Direct V5-A', 'Requested lead in one pass; no recursive state rollout', 'Legacy-data run', 'plan'),
 ('Direct / context', 'Hydraulic halo V5-B', '8-cell neighboring state/history; central output', 'Legacy-data run', 'report'),
 ('Direct / context', 'Direction / boundary losses', 'Auxiliary velocity-direction and boundary-discharge terms', 'Legacy-data ablation', 'report'),
 ('Direct / context', 'Local forcing V5-C', 'Inverse-distance signal from 8 nearby forcing locations', 'Tested with coupling', 'plan'),
 ('Direct / context', 'Discharge-correction gate', 'Low-initialized gate on learned HU/HV corrections', 'Legacy-data ablation', 'report'),
 ('Direct / context', '3-seed / 5-fold protocol', 'Optimization variability and complete-event holdouts', 'Legacy-data evaluation', 'report'),
 ('Graph / next', 'Local graph V5-D', 'Boundary summaries + adjacency messages + bottleneck context', 'Prototype + GPU smoke', 'graph'),
 ('Graph / next', 'Graph learning checks', 'Tiny training cohort; baseline / self-edge / neighbor arms', 'Diagnostic; skill unproven', 'graph'),
 ('Graph / next', 'HUC12 hierarchy', 'Coarse watershed nodes for wider routing context', 'Proposed; not implemented', 'graph'),
 ('Graph / next', 'Conservative physics V5-E', 'Storage and shared-boundary flux constraints', 'Proposed; not established', 'graph'),
 ('Graph / next', 'Transformer / operator V5-F', 'Controlled backbone comparison after fixing inputs', 'Proposed; not implemented', 'plan'),
]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
pages = []


def rgb(color):
    return RGBColor.from_string(color.lstrip('#'))


class Page:
    def __init__(self, title, subtitle='', sources=(), notes='', appendix=False):
        self.title, self.subtitle, self.sources, self.notes = title, subtitle, sources, notes
        self.slide = prs.slides.add_slide(prs.slide_layouts[6])
        self.items = []
        pages.append(self)
        self.rect(0, 0, W, H, WHITE)
        self.rect(0, 0, W, .13, TEAL)
        self.text(.55, .33, 12.1, .58, title, 27, NAVY, True)
        self.text(.57, 1.00, 12.1, .48, subtitle, 14, MUTED)
        self.rect(.55, 6.94, 12.2, .015, '#D5E2E7')
        self.text(.57, 7.03, 10.6, .23, 'TRITON LITE  |  METHODS & DATA REPAIR  |  '+('APPENDIX' if appendix else '12 OCT 2026'), 9, MUTED)
        self.text(12.17, 7.03, .55, .23, f'{len(pages):02d}', 10, TEAL, True)

    def rect(self, x,y,w,h,color=PALE,line=None):
        s=self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        s.fill.solid();s.fill.fore_color.rgb=rgb(color)
        if line:s.line.color.rgb=rgb(line)
        else:s.line.fill.background()
        self.items.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color,line=line))
        return s

    def text(self,x,y,w,h,text,size=19,color=INK,bold=False):
        # Wrap text explicitly so PowerPoint and the standalone preview agree.
        font=ImageFont.truetype(BOLD_FONT if bold else FONT, round(size*96/72))
        max_width=w*96-6
        lines=[]
        for paragraph in str(text).split('\n'):
            current=''
            for word in paragraph.split():
                proposed=(current+' '+word).strip()
                if current and font.getlength(proposed)>max_width:
                    lines.append(current);current=word
                else:current=proposed
            lines.append(current)
        pitch=size*1.18/72
        if len(lines)*pitch > h+.02:
            raise ValueError(f'Text overflows on slide {len(pages)}: {text[:60]} ({len(lines)} lines / {h} in)')
        s=self.slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        tf=s.text_frame;tf.word_wrap=False
        tf.margin_left=tf.margin_right=Inches(.015)
        tf.margin_top=tf.margin_bottom=0
        for i,line in enumerate(lines):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
            p.text=line;p.font.name='DejaVu Sans';p.font.size=Pt(size)
            p.font.bold=bold;p.font.color.rgb=rgb(color)
            p.line_spacing=Pt(size*1.18);p.space_before=p.space_after=Pt(0)
        self.items.append(dict(kind='text',x=x,y=y,w=w,h=h,lines=lines,size=size,color=color,bold=bold))
        return s

    def arrow(self,x,y,w=.45,h=.20,color=TEAL):
        s=self.slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
        s.fill.solid();s.fill.fore_color.rgb=rgb(color);s.line.fill.background()
        self.items.append(dict(kind='arrow',x=x,y=y,w=w,h=h,color=color))

    def box(self,x,y,w,h,title,body='',color=PALE,title_color=TEAL,size=18):
        self.rect(x,y,w,h,color)
        for title_size in (size,size-1,size-2,size-3):
            try:
                self.text(x+.18,y+.16,w-.36,.58,title,title_size,title_color,True)
                break
            except ValueError:
                if title_size==size-3:raise
        if body:
            for body_size in (17,16,15,14):
                try:
                    self.text(x+.18,y+.88,w-.36,h-1.0,body,body_size,INK)
                    break
                except ValueError:
                    if body_size==14:raise

    def callout(self,text,color=TEAL,y=6.12):
        self.rect(.58,y,12.13,.60,color)
        size=16
        while ImageFont.truetype(BOLD_FONT,round(size*96/72)).getlength(text)>11.77*96-6:
            size-=.5
        self.text(.76,y+.12,11.77,.39,text,size,WHITE,True)

    def finish(self):
        source_text='\n'.join(SRC.get(s,s) for s in self.sources)
        self.slide.notes_slide.notes_text_frame.text=(self.notes+'\n\nSources:\n'+source_text+
            '\n\nEvidence rule: legacy timestamp/transition/direct metrics do not establish corrected-data forecasting skill.')


def table(page, headers, rows, widths, top=1.72, rowh=.62, fontsize=15):
    left=.58
    for j,(head,width) in enumerate(zip(headers,widths)):
        page.rect(left,top,width,rowh,NAVY)
        page.text(left+.12,top+.13,width-.24,rowh-.15,head,fontsize,WHITE,True)
        left+=width
    for i,row in enumerate(rows):
        left=.58;y=top+rowh*(i+1)
        for val,width in zip(row,widths):
            page.rect(left,y,width,rowh,PALE if i%2==0 else '#F8FAFB')
            page.text(left+.12,y+.11,width-.24,rowh-.13,str(val),fontsize,INK)
            left+=width


p=Page('Flood forecasting with hydraulic context', 'TRITON Lite | Methods, experiments, and the corrected-data restart', ('audit','report','graph'),
       'Prepared for Monday, October 12, 2026. This deck explains the work already implemented and the experimental sequence. It does not claim validated forecasting gains from the legacy dataset. Replace the status snapshot and result placeholders before the meeting.')
p.text(.65,1.85,10.8,1.3,'From independent tiles to\nstate-aware, direct forecasts',34,NAVY,True)
p.box(.65,3.6,3.85,1.85,'01  Hydraulic history','Use the observed state and its recent change.')
p.box(4.75,3.6,3.85,1.85,'02  Spatial context','Add a halo; test graph communication.')
p.box(8.85,3.6,3.85,1.85,'03  Trustworthy targets','Repair chronology and signed discharge.')
p.callout('Methods are ready to explain. Corrected-data model comparisons are still pending.')

p=Page('What the model is asked to predict', 'A requested future hydraulic map, given an observed state and known forcing', ('plan','model'),
       'The direct model uses observed hydraulic states at the origin and one preceding half-hour state. Forcing is supplied through the requested target time, but never after it. Known future forcing is an input assumption; this is not a weather forecasting model.')
p.box(.65,1.8,3.7,3.75,'Inputs at forecast origin','Current h, HU, HV\nPrevious hydraulic state\nTerrain, mask and tile attributes\nForcing through the target time')
p.arrow(4.48,3.36,.48,.3)
p.box(5.08,2.2,2.7,2.8,'Direct model','Requested lead:\n0.5–24 hours\nOne model application')
p.arrow(7.96,3.36,.48,.3)
p.box(8.57,1.8,4.1,3.75,'Outputs at target time','Depth h\nWet probability / flood extent\nSigned unit discharges HU, HV\nDerived velocity as a diagnostic')
p.callout('Prediction unit: one central 80 × 80 tile. Forecast lead: 1–48 half-hour steps.')

p=Page('The available simulation archive', 'One Conasauga domain supports controlled development experiments', ('readme','report','audit'),
       'The raster spacing is approximately 9.506 m; the project calls this 10 m resolution. An 80-cell tile is roughly 0.76 km across, conventionally described as approximately 0.8 km. Hydrograph comments use cfs; physical forcing-unit handling remains to be audited before conservation losses. Do not describe these 300 locations as observed gauges without checking provenance.')
for x,title,body in [( .65,'40 events','D001–D040\n240 simulated hours each'),(4.8,'480 states / event','Half-hour snapshots\nDepth plus signed discharge'),(8.95,'~6,900 tiles','80 × 80 cells per tile\n34 HUC12 subwatersheds')]:p.box(x,1.85,3.75,2.5,title,body)
p.text(.78,4.67,11.9,1.10,'Forcing tensor: 480 time steps × 300 source locations\nTerrain inputs: DEM, flow accumulation, stream proximity, slope and relative elevation',20,INK)
p.callout('Current scope: in-domain development. New events and regions are needed for final validation.')

p=Page('Our method evolved through several questions', 'The project includes many controlled attempts, not one architecture switch', ('peak','progress','plan','graph'),
       'This is a conceptual sequence, not a date timeline. Version names are reused: timestamp V4 and transition V4a/V4b are different families. The appendix lists 26 method/protocol entries, including proposed work; that count is not a count of completed runs. Historical diagnoses motivated experiments but must be revisited after target repair.')
stages=[('Peak maps','Can event + terrain\npredict maximum depth?'),('Timestamp states','Can forcing + time\npredict each state?'),('State transitions','Does observed history\nhelp forecast change?'),('History + gates','Can updates respect\nstable conditions?'),('Direct + halo','Can we predict a lead\nwith nearby context?'),('Graph + audit','Can communication help?\nAre targets correct?')]
for i,(title,body) in enumerate(stages):
 x=.65+(i%3)*4.15;y=1.82+(i//3)*2.08
 p.box(x,y,3.75,1.83,title,body,size=19)
p.callout('The audit changed the immediate priority: repair targets, then repeat the model comparisons.')

p=Page('Early attempts: maps, sampling and supervision', 'We explored how to represent rare wet cells and flood boundaries', ('early','peak','timestamp','max','v4','progress'),
       'The earliest peak-depth model uses an event embedding and block features with a spatial decoder. Timestamp models add a requested time and causal forcing. Subsequent attempts included label-aware batches, a denser dynamic index, strict category quotas, physical/log-depth losses, wet BCE/Dice, and calibrated wet gating. The target chronology issue directly invalidates timestamp forecast interpretation; this deck makes no new claim about whether peak-depth products are affected.')
p.box(.65,1.85,3.75,3.65,'Blockwise peak V2 / V3','Event + tile + static rasters\nDepth and wet/dry heads\nDepth-weighted Huber\nExtent-threshold diagnostics')
p.box(4.8,1.85,3.75,3.65,'Timestamp / Stage-1 Max','Forcing + requested time\nLabel-aware sampling\nDense candidate index\nWhole-batch wet/deep balance')
p.box(8.95,1.85,3.75,3.65,'Gated / timestamp V4','Validation-calibrated extent\nStrict category quotas\nDepth × wet-probability coupling\nFlow-aware supervision')
p.callout('These remain reusable methods; their old metric rankings do not establish forecasting skill.')

p=Page('Transition attempts: learn the state change', 'We explored temporal history, rollout exposure, and stable-update safeguards', ('v1','v2','v3','v4a','v4b'),
       'Transition V1 starts with one-step observed-state training, then continuation and six-step scheduled predicted-state exposure. V2 changes regimes and objectives; V3 uses exact paired local statistics and twelve-step exposure. V4a adds a second state and activity gate; V4b separates the history adapter and trains new modules faster. Those were tests on legacy targets, not proof that any feature improves the corrected task.')
table(p,['Attempt','What changed','Question'],[
 ('V1 + continuation','Residual update from observed state','Can history help predict change?'),
 ('Multi-step V1','Six-step predicted-state exposure','Can training cover rollout behavior?'),
 ('Regime-aware V2','Filling, draining and stable objectives','Can rare changes be represented?'),
 ('Exact-local V3','Paired labels; 12-step training','Are regimes and lead exposure reliable?'),
 ('History V4a / adapter V4b','Two states, activity gate, separate adapter','Can stable updates and history be learned?')], [2.6,4.65,4.88],rowh=.70,fontsize=15)
p.callout('Retain the implementations. Reassess the historical diagnoses on corrected data.')

p=Page('Direct forecasting removes recursive state feedback', 'V5-A predicts the requested lead from the observed origin', ('plan','model','v4b'),
       'The direct model still predicts a residual from the observed origin. Direct prediction avoids the mechanism of feeding intermediate predicted states back into the network. It does not guarantee lower forecast error. Both models must be evaluated on the same corrected origins and targets.')
p.text(.7,1.82,11.9,.45,'Recursive transition forecast',22,NAVY,True)
for i,label in enumerate(['Observed S(t)','Predicted S(t+1)','Predicted S(t+2)','… S(t+48)']):
 x=.75+i*3.1;p.rect(x,2.46,2.67,.85,PALE);p.text(x+.12,2.67,2.43,.42,label,17,INK,True)
 if i<3:p.arrow(x+2.74,2.78,.3,.18,GOLD)
p.text(.75,3.48,11.8,.5,'Each predicted state becomes part of the next model input.',18,MUTED)
p.text(.7,4.12,11.9,.45,'Direct, lead-conditioned forecast',22,TEAL,True)
p.rect(.75,4.80,4.02,.84,PALE);p.text(.9,5.02,3.72,.41,'Observed S(t) + history + lead',15,INK,True)
p.arrow(4.96,5.12,.48,.2)
p.rect(5.64,4.80,2.5,.84,TEAL);p.text(5.8,5.02,2.18,.41,'One pass',17,WHITE,True)
p.arrow(8.35,5.12,.48,.2)
p.rect(9.02,4.80,3.62,.84,PALE);p.text(9.15,5.02,3.36,.41,'Predicted S(t + lead)',17,INK,True)
p.callout('A design change that must be tested, rather than a claim of improved accuracy.')

p=Page('The direct U-Net combines space, history and forcing', 'Terrain and hydraulic features meet lead-conditioned information', ('model','readme'),
       'The active implementation uses a causal forcing encoder, tile/time/lead embeddings, state/history fusion, a conditioned U-Net bottleneck, skip connections, and residual hydraulic heads. The diagram groups modules for readability. Halo features are added to the first central spatial feature map; graph context enters at the bottleneck. New corrected-data runs should train from scratch, not initialize from contaminated checkpoints.')
p.box(.65,1.76,3.4,1.8,'Spatial inputs','Terrain + mask\nCurrent state + history',size=17)
p.box(.65,3.86,3.4,1.8,'Conditioning inputs','Causal forcing encoder\nTile, time and lead embeddings',size=17)
p.arrow(4.18,2.62,.38,.22)
p.box(4.7,1.76,3.5,1.8,'Tile CNN encoder','Central spatial features\nDownsample 80 → 40 → 20',size=17)
p.arrow(4.18,4.7,.38,.22)
p.box(4.7,3.86,3.5,1.8,'Conditioned bottleneck','Spatial features + conditioning\nU-Net decoder + skip features',size=17)
p.arrow(8.37,4.7,.38,.22)
p.box(8.9,3.86,3.8,1.8,'Forecast heads','Δh, ΔHU, ΔHV; activity gate\nWet probability',size=17)
p.box(8.9,1.76,3.8,1.8,'Optional context','Halo: early spatial features\nGraph: bottleneck messages',size=17)
p.callout('Future hydraulic states are targets only. They never enter forecast inputs.')

p=Page('Halo: neighboring hydraulics around a central tile', 'V5-B adds an 8-cell border while keeping an 80 × 80 output', ('report','model'),
       'Implementation detail matters: this is a separate hydraulic halo encoder, not an enlarged terrain U-Net. A 96x96 current/history hydraulic patch is encoded; its central 80x80 feature crop is added to the central tile feature map. Only central outputs are supervised. Two history states produce current channels plus changes from the older state. Eight cells correspond to roughly 76 m at the actual grid spacing. Whether halo improves corrected-data accuracy is pending.')
p.rect(.9,1.82,4.08,4.08,'#D2ECEB',TEAL)
p.rect(1.24,2.16,3.4,3.4,WHITE,TEAL)
for j in range(1,8):
 p.rect(1.24+j*.425,2.16,.012,3.4,'#E0E9ED')
 p.rect(1.24,2.16+j*.425,3.4,.012,'#E0E9ED')
p.rect(1.61,3.37,2.7,.77,TEAL)
p.text(1.77,3.54,2.38,.44,'Central 80 × 80',18,WHITE,True)
p.text(1.08,5.99,4.05,.44,'96 × 96 hydraulic input',17,TEAL,True)
p.box(5.62,1.85,7.0,1.9,'Observe a wider neighborhood','The border supplies nearby depth and signed discharge at the forecast origin.',size=20)
p.box(5.62,4.0,7.0,1.9,'Score only the central prediction','Halo encoder → central feature crop → central U-Net features. Output remains 80 × 80.',size=20)

p=Page('The Stage-2 comparisons we already attempted', 'Historical experiment designs are reusable; performance conclusions need reruns', ('report','plan'),
       'The first nested ablation compared direct, halo, halo+direction/boundary losses, and halo+those losses+local forcing. Local forcing was tested together with coupling, so it was not an isolated causal estimate of local forcing. The later fold-0 study compared baseline, halo, and halo+discharge gate at three seeds. Halo alone received five-fold evaluation at one seed. None of the numerical rankings establishes corrected-data forecasting skill.')
table(p,['Experiment family','Variants / protocol','Evidence status'],[
 ('Initial four-way ablation','Direct; halo; + auxiliary losses; + local forcing','Ran on legacy targets'),
 ('Three-seed fold-0 study','Baseline; halo; halo + discharge gate','Ran on legacy targets'),
 ('Five-fold event evaluation','Halo candidate; each event tested once','Ran on legacy targets'),
 ('Graph learning diagnostic','U-Net; local capacity control; adjacency graph','Tiny training cohort only')], [2.8,6.1,3.23],rowh=.80,fontsize=15)
p.callout('Do not use the old “11.8% depth improvement” as a corrected-data claim.',GOLD)

p=Page('Depth, discharge and velocity need separate checks', 'Archived U/V outputs represent signed unit discharge, HU/HV', ('semantics','audit','plan'),
       'The source-code semantics audit establishes HU/HV as unit discharge with m²/s units. Derived velocity divides by depth, so shallow-depth errors can be amplified. The three policies shown are diagnostic alternatives, not a selected corrected-data forecast policy. Reconstructing HU=h*u and HV=h*v imposes output consistency, not conservation or validated forecast skill. The old semantics document predates discovery of negative-value clipping; its claim that values were copied unchanged is superseded by the target audit.')
p.text(.77,1.82,11.9,.75,'HU = h · u     HV = h · v     u = HU / h     v = HV / h',25,NAVY,True)
p.box(.65,2.97,3.75,2.73,'Learned HU/HV','Predict signed flow corrections with depth. Inspect derived velocity in shallow and deep water.',size=18)
p.box(4.8,2.97,3.75,2.73,'Component persistence','Copy origin HU/HV while forecasting depth. Declare this policy explicitly.',size=18)
p.box(8.95,2.97,3.75,2.73,'Velocity persistence','Copy origin u/v; reconstruct HU/HV using forecast depth. Evaluate outputs jointly.',size=18)
p.callout('Coupled outputs do not establish mass conservation. Signed flow direction must be preserved.')

p=Page('Graph prototype: communicate learned tile features', 'V5-D retains the raster U-Net and adds adjacency context', ('graph',),
       'Each node is one tile. Edges are actual shared raster boundaries in both directions: adjacency is geometry, not verified hydraulic passability. Node features combine local features, four boundary summaries, forcing and lead. Connected batches share event, origin and lead; losses score centers. Local graph communication is implemented. HUC12 hierarchy is a future extension. Messages are learned features, not fluxes.')
for x,y,lab in [(2.15,2.03,'North'),(.65,3.38,'West'),(3.65,3.38,'East'),(2.15,4.73,'South')]:
 p.rect(x,y,1.15,.71,PALE,TEAL);p.text(x+.07,y+.19,1.01,.35,lab,14,TEAL,True)
p.rect(2.15,3.38,1.15,.71,TEAL);p.text(2.22,3.57,1.01,.35,'Center',14,WHITE,True)
p.rect(2.7,2.75,.05,.62,TEAL);p.rect(2.7,4.10,.05,.62,TEAL)
p.rect(1.81,3.71,.33,.05,TEAL);p.rect(3.31,3.71,.33,.05,TEAL)
p.box(5.1,1.85,7.58,1.82,'Message-passing pathway','Tile features + boundary summaries → graph messages → residual bottleneck context',size=20)
p.box(5.1,3.96,7.58,1.82,'What has been checked','Topology, connected batches, GPU execution, and tiny-cohort learning diagnostics.',size=20)
p.callout('Forecasting benefit is unproven. Graph messages are not conservative water exchange.')

p=Page('A fair comparison holds out complete events', 'Use the same origins, leads, sampling and metrics for every candidate', ('report','plan','graph'),
       'The existing five-fold contract uses 25 training events, 7 validation events, and 8 test events per run. One historical seed covers all five folds, while three-seed robustness was fold 0 only. Preserve event membership after repair; fit normalization and sampling using training data. Select checkpoints and extent thresholds on validation only. Recompute persistence on corrected targets; do not reuse old persistence metrics. Original dense tests sampled 80 batches per fold, not every pixel in the domain.')
for x,w,num,lab,color in [(.65,7.4,'25','TRAIN EVENTS',TEAL),(8.25,1.92,'7','VALIDATION',NAVY),(10.38,2.3,'8','TEST EVENTS',GOLD)]:
 p.rect(x,1.86,w,1.6,color);p.text(x+.18,2.03,w-.36,.65,num,35,WHITE,True);p.text(x+.18,2.86,w-.36,.32,lab,12,WHITE,True)
p.box(.65,3.87,5.83,1.97,'Baseline: persistence','Copy the origin state. Compare against the same corrected targets and forecast origins.',size=19)
p.box(6.83,3.87,5.83,1.97,'Evaluation across leads','All 48 half-hour leads; depth, extent, signed flow, velocity and wetting/drying transitions.',size=19)
p.callout('Train from scratch. No legacy checkpoint or test-event normalization enters the new comparison.')

p=Page('The audit found two target-conversion defects', 'These affect both learning and the interpretation of forecast leads', ('audit',),
       'Raw evidence: D029 origin index239 labeled120h matches H_30_00.dat, physically15h; the next labeled120.5h matches H_310_00.dat, physically155h. Alphabetic order, followed by assigned regular timestamps, created wrong pairs. Negative signed discharge values were discarded and later loader filling made them zero. Numeric counter sorting and signed-component preservation are implemented. Forcing interpolation independently passed across all40 events. Do not infer that source simulations are wrong; original archives remain intact.')
p.box(.65,1.85,5.84,3.42,'01  Wrong snapshot chronology','Alphabetic sorting scrambled numeric counters.\n\nD029 label: 120 hours\nActual source state: 15 hours',title_color=GOLD,size=20)
p.box(6.84,1.85,5.84,3.42,'02  Lost signed discharge','Negative HU/HV values were discarded.\n\nThe model lost flow direction in affected cells.',title_color=GOLD,size=20)
p.callout('Forcing audit: all 40 events passed. Hydraulic targets required rebuilding.',y=5.65)

# Derive a clearly dated snapshot from existing reports, without altering jobs.
validated=[]
for job in (186578,187031):
    for report in (ROOT/f'processed_data/stage2_corrected_rebuild/job_{job}').glob('*/HYDRAULIC_VALIDATION.json'):
        d=json.loads(report.read_text());event=d['event_id']
        if d.get('status')=='hydraulic_validation_passed' and (report.parent/f'{event}_ACC_future.nc').exists():
            validated.append(event)
validated=sorted(set(validated))
p=Page('Corrected targets are being rebuilt and checked', f'Status snapshot: {STAMP} | update before presenting', ('audit',),
       f'Snapshot from successful validation reports: {len(validated)}/40 events: '+', '.join(validated)+'. Original array186578 produced eight validated outputs. Recovery array187031 uses the intact triton environment; two tasks were requeued after optional training imports caused an OpenMP conflict. Conversion now avoids those imports. Validation is sampled,108 component/tile/time comparisons per event, not all-pixel certification. This does not mean corrected model training is complete.')
p.box(.65,1.85,3.75,2.97,'Numeric time counters','Require all 480 sequential H/U/V snapshots. Preserve the relative half-hour time axis.',size=19)
p.box(4.8,1.85,3.75,2.97,'Signed-flow validation','Compare sampled values and masks directly with raw archives. Check units and coordinates.',size=19)
p.box(8.95,1.85,3.75,2.97,'Per-event completion','Final NetCDF + successful validation report required. Keep failures as partial files.',size=19)
p.text(.85,5.24,11.6,.62,f'{len(validated)} / 40 events validated in this snapshot',27,TEAL,True)
p.callout('Rebuild → sampling and normalization → fresh training → corrected persistence comparison.')

p=Page('Corrected-data results: fill after evaluation', 'Editable placeholders — no forecasting result is claimed on this slide', ('plan','audit'),
       'Replace these boxes only after a fresh corrected-data run and held-out evaluation. Report model versus persistence on the same cohort. Suggested leads0.5,3,6,12,24h for a compact table, while keeping all48 leads in an appendix. Use representative held-out maps with consistent depth color scales. Show depth/extent and derived velocity separately. If results are unavailable, keep this as the evaluation plan or remove the slide.')
p.box(.65,1.85,3.75,3.9,'A  Lead curves','[INSERT CORRECTED RESULTS]\n\nWet-depth RMSE\nExtent F1 / CSI\nModel vs persistence',title_color=GOLD,size=19)
p.box(4.8,1.85,3.75,3.9,'B  Held-out maps','[INSERT CORRECTED MAPS]\n\nTRITON target\nModel prediction\nError / missed flooding',title_color=GOLD,size=19)
p.box(8.95,1.85,3.75,3.9,'C  Runtime and flow','[INSERT MEASUREMENTS]\n\nTraining / inference time\nHU/HV error\nShallow / deep velocity',title_color=GOLD,size=19)
p.callout('Report event splits, evaluated samples, seed, checkpoint rule and output policy.')

p=Page('What we can responsibly bring to Monday', 'Explain the methods now; add fresh evidence as it becomes available', ('audit','plan','graph'),
       'Priority is one trustworthy baseline, then a halo comparison if time permits. Training throughput on Riker has not been established by a production corrected-data benchmark; previous eight-hour job limits are not runtime evidence. Sunday should be reserved for evaluation and slide assembly. If training slips, present the repair and methods with labeled diagnostic evidence, not old performance claims. No new training is submitted by creating this presentation.')
p.box(.65,1.85,3.75,3.66,'Ready to explain','Architecture and halo\nSequence of experiments\nGraph prototype\nData bugs and repair checks',size=19)
p.box(4.8,1.85,3.75,3.66,'Priority evidence','Corrected-data baseline\nPersistence comparison\nLead curves and maps\nMeasured runtime',size=19)
p.box(8.95,1.85,3.75,3.66,'Next scientific questions','Does halo help after repair?\nCan discharge stay stable?\nDoes graph add useful context?\nWhat generalizes to new events?',size=19)
p.callout('Retain the method work. Rebuild the evidence for model selection on trustworthy targets.')

# Detailed catalogs intentionally live in the appendix, not in a crowded timeline.
catalogs=[('Appendix: early maps and timestamp attempts',HISTORY[:7]),
          ('Appendix: transition and history attempts',HISTORY[7:15]),
          ('Appendix: direct, halo and graph attempts',HISTORY[15:23]),
          ('Appendix: proposed extensions and source documents',HISTORY[23:])]
for title,rows in catalogs[:3]:
 p=Page(title,'Method catalog; execution status is separate from scientific validation',
        tuple(dict.fromkeys(row[4] for row in rows)),
        '\n'.join(f'{r[1]}: {r[2]}. Status: {r[3]}. Source: {SRC[r[4]]}' for r in rows),appendix=True)
 table(p,['Attempt','Change / hypothesis','Status'],[(r[1],r[2],r[3]) for r in rows],
       [3.02,6.03,3.08],top=1.65,rowh=.54,fontsize=12.5)
 p.text(.78,6.64,11.9,.24,'Version labels are reused across families. These entries are not a count of jobs or proven gains.',10,MUTED)

p=Page('Appendix: future work and source documents','Separate implemented prototypes from the proposed roadmap', ('plan','graph','audit','report'),
       'For a ChatGPT redesign, upload CHATGPT_SOURCE_BRIEF.md and EXPERIMENT_HISTORY.md from this folder first. Optional full documents are included in CHATGPT_SOURCE_BUNDLE.zip, preserving repository paths. Legacy source documents contain obsolete numeric claims; the source brief explicitly supersedes them. The current deck and preview use native editable PowerPoint shapes; they do not reproduce legacy quantitative charts.',appendix=True)
table(p,['Proposed extension','Purpose','Status'],[(r[1],r[2],r[3]) for r in HISTORY[23:]],
      [3.1,5.9,3.13],top=1.70,rowh=.66,fontsize=14)
p.text(.78,4.58,11.85,.53,'Upload first: CHATGPT_SOURCE_BRIEF.md + EXPERIMENT_HISTORY.md',19,TEAL,True)
p.text(.78,5.28,11.85,1.15,'Optional full sources: target audit; V5-B report; V5-D graph plan; Stage-1 transition documents.\nHistorical numerical claims in those files are superseded by the audit.',17,INK)

for p in pages:p.finish()
prs.core_properties.title='TRITON Lite — Methods, experiment history and corrected-data restart'
prs.core_properties.subject='Editable presentation for October 12, 2026'
prs.core_properties.author='TRITON Lite project'
prs.core_properties.comments='Legacy forecast metrics excluded; corrected-data results pending.'
deck=HERE/'TRITON_Methods_and_Experiment_History.pptx'
prs.save(deck)


def svg(page):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W*96} {H*96}" role="img" aria-label="{html.escape(page.title)}">']
    for v in page.items:
        x,y,w,h=(v[k]*96 for k in ('x','y','w','h'))
        if v['kind']=='rect':
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{v["color"]}" stroke="{v.get("line") or "none"}"/>')
        elif v['kind']=='arrow':
            points=f'{x},{y+h*.28} {x+w*.62},{y+h*.28} {x+w*.62},{y} {x+w},{y+h/2} {x+w*.62},{y+h} {x+w*.62},{y+h*.72} {x},{y+h*.72}'
            parts.append(f'<polygon points="{points}" fill="{v["color"]}"/>')
        else:
            size=v['size']*96/72
            for j,line in enumerate(v['lines']):
                parts.append(f'<text x="{x+1.44}" y="{y+size*.9+j*size*1.18}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{700 if v["bold"] else 400}" fill="{v["color"]}">{html.escape(line)}</text>')
    return ''.join(parts)+'</svg>'


preview=['<!doctype html><html lang="en"><meta charset="utf-8"><title>TRITON editable deck preview</title>',
 '<style>body{background:#dbe5ea;margin:24px;font-family:Arial,sans-serif;color:#123147}main{max-width:1120px;margin:auto}section{margin:24px 0 40px;break-after:page}svg{width:100%;background:white;box-shadow:0 4px 18px #12314722}details{padding:12px;background:white}pre{white-space:pre-wrap;font:14px/1.55 Arial}header{background:white;padding:20px}@media print{body{margin:0;background:white}details,header{display:none}section{margin:0}svg{box-shadow:none}}</style><main>',
 f'<header><h1>TRITON Lite — slide preview</h1><p>{len(pages)} slides; editable PowerPoint alongside this file. Status snapshot: {STAMP}.</p><p>Expand speaker notes for source paths and interpretation. Print this page to PDF for a review copy.</p></header>']
notes=['# TRITON Lite — slide text and speaker notes',f'\nPrepared: {STAMP}. Meeting: October 12, 2026.\n',
 'Legacy timestamp, transition and direct-forecast numbers do not establish corrected-data forecasting skill.\n']
for i,p in enumerate(pages,1):
    preview.append(f'<section id="slide-{i}">{svg(p)}<details><summary>{i:02d}. Speaker notes and sources</summary><pre>{html.escape(p.slide.notes_slide.notes_text_frame.text)}</pre></details></section>')
    notes.append(f'## {i:02d}. {p.title}\n\n{p.subtitle}\n')
    for item in p.items:
        if item['kind']=='text' and 1.5<item['y']<6.9:notes.append(' / '.join(item['lines'])+'\n')
    notes.append('\nSpeaker notes: '+p.notes+'\n\nSources: '+', '.join(SRC.get(s,s) for s in p.sources)+'\n')
preview.append('</main></html>')
(HERE/'preview.html').write_text('\n'.join(preview))
(HERE/'SLIDE_TEXT_AND_NOTES.md').write_text('\n'.join(notes))

history=['# TRITON Lite experiment history', '\nThis catalog groups methods and protocols, not individual jobs. Proposed items did not run.\n',
 '> Evidence rule: timestamp/transition/direct forecast metrics on legacy hydraulic targets cannot support forecasting or architecture conclusions. Earlier observations motivated the trials but need corrected-data reassessment. Peak-map methods are included as historical methods without making new claims about their data validity.\n',
 '| Family | Attempt | Change / hypothesis | Status | Source |', '|---|---|---|---|---|']
for family,name,change,status,key in HISTORY:history.append(f'| {family} | {name} | {change} | {status} | `{SRC[key]}` |')
history.append('\nVersion labels are reused: peak V3, timestamp V4, transition V3/V4a/V4b, and Stage-2 V5 name distinct model families. Local forcing was tested with auxiliary coupling losses; a clean isolated local-forcing comparison remains pending. Multi-seed testing covered fold0 only; five-fold halo testing used one seed. Local graph implementation is distinct from the unimplemented HUC12 hierarchy. Coupling outputs is distinct from establishing mass conservation.\n')
(HERE/'EXPERIMENT_HISTORY.md').write_text('\n'.join(history))

brief=f'''# ChatGPT source brief: TRITON Lite Monday presentation

Meeting: Monday, October 12, 2026. Prepared: {STAMP}.

## Prompt to paste with this file

Build an editable scientific presentation for collaborators from this source brief and EXPERIMENT_HISTORY.md. Use clear architecture diagrams and a readable explanation of halo. Include the many earlier attempts, distinguishing completed legacy experiments, diagnostic prototypes, and proposed work. Use 17 main slides plus the detailed history appendix, or shorten to 12 main slides while retaining the appendix. Include speaker notes and source attribution. Treat the evidence rules below as authoritative over obsolete performance claims in older documents. Do not invent corrected-data results. Leave editable placeholders for new lead curves, held-out maps, and runtime. Explain the data repair candidly without making the whole deck a debugging report.

## Current scientific status — authoritative

Two hydraulic conversion defects were confirmed on October 6: alphabetic snapshot ordering assigned wrong regular half-hour timestamps; negative signed discharge values were discarded. Consequently old V4/V5 and other legacy timestamp/transition results do not validate the intended future-forecast task. Do not present old improvement percentages, metric tables or architecture rankings as valid forecast evidence. Specifically exclude the old 11.8% halo depth-gain claim. Historical diagnoses motivated experiments; revisit them on repaired targets. Earlier peak-depth modeling is historical context, with no new assertion here about whether peak-depth products are affected.

Original simulation archives are intact. The converter now sorts numeric counters and preserves signed HU/HV. Independent forcing alignment passed for all40 events, exactly matching original interpolated hydrographs on0.5–240h. Calendar epoch interpretation is not verified. Archive forcing comments use cfs; physical forcing-unit handling needs separate auditing before conservation losses.

Hydraulic rebuild status is a dated snapshot, not live telemetry: {len(validated)}/40 validated at {STAMP}. Completed events: {', '.join(validated)}. Each event has108 sampled raw tile/time/component comparisons, covering values, masks, signed components, timing, coordinates and units. This is not all-pixel verification. Initial array186578 produced eight complete events. Recovery187031 handles the remainder after environment/import repairs. Training on corrected data and fresh performance comparisons are pending.

## Task and data

One Conasauga domain,40 TRITON events,240hours/event,480half-hour hydraulic states/event. Forcing is a480×300tensor of source locations, not necessarily observed gauges. Approximately6900tiles and34HUC12units. Central tile80×80cells at approximately10m resolution (actual spacing about9.506m). Six terrain layers plus mask and tile attributes.

At forecast origin t, provide h,HU,HV and one older hydraulic state; terrain; tile/time attributes; requested lead; forcing supplied through t+lead. Predict h,wet probability,HU,HV at t+lead. Leads1–48half-hour steps (0.5–24h). No future hydraulic state enters inputs. Known future forcing is an assumption, not a weather prediction performed by this model.

## Architecture and halo — precise implementation

Direct V5-A uses a causal forcing encoder plus tile/time/lead embeddings, central terrain/state CNNs, history adapter, conditioned U-Net bottleneck, skip connections, residual depth/discharge heads and activity gate. It predicts the requested target in one pass rather than recursively feeding predicted states through48updates. That design avoids recursive feedback; it does not prove reduced error.

V5-B adds an8-cell hydraulic border:96×96observed current/history input around an80×80central tile. A separate halo CNN encodes hydraulic features; its central80×80crop is added to the first central U-Net feature map. It is not an enlarged terrain U-Net, not a watershed graph, and not future-target information. Only central output is scored. Explain with a colored outer border and central grid. Neighborhood benefit must be reestablished on corrected data. Approximately76m border at actual spacing.

HU/HV are signed unit discharge in m²/s. Velocity is derived as HU/h,HV/h; shallow depth can amplify error. Previous alternatives tested learned discharge, component persistence, and velocity persistence with discharge reconstructed from predicted depth. These are distinct declared policies. HU=h*u and HV=h*v coupling imposes output consistency, not mass conservation.

## Many earlier attempts

Use the companion EXPERIMENT_HISTORY.md as the authoritative catalog. Sequence: peak V2/V3; timestamp baseline; label-aware/stratified sampling; Stage-1Max; wet gating; timestampV4; transitionV1; continuedV1; six-step multi-stepV1; regime-awareV2; exact-localV3; history/gateV4a; adapter/gateV4b; persistence-policy diagnostics; directV5-A; haloV5-B; direction/boundary auxiliary losses; local forcingV5-C; discharge-correction gate; three-seed/five-fold protocols; local graphV5-D; graph learning diagnostics. Hierarchical HUC12 graphs, conservative physics and Transformer/operator variants are proposed extensions, not completed results.

Initial Stage-2 four-way comparison: baseline,halo,halo+direction/boundary losses,halo+those losses+local forcing. Local forcing was confounded with coupling in that initial comparison. Later three-seed fold0 comparison: baseline,halo,halo+discharge gate. One-seed five-fold halo evaluation also ran. All used legacy targets; do not infer corrected-data rankings.

Local graph prototype: one tile/node; actual shared-boundary adjacency in both directions;6900nodes and27262directed edges; boundary summaries and conditioning feed message passing; residual context enters the U-Net bottleneck. Connected samples share event,origin,lead; score centers. Messages are learned features, not water fluxes. Smoke and tiny training-cohort diagnostics establish software execution only. Full graph benefit and hydraulic passability of edges are unproven. HUC12hierarchy is not implemented.

## Corrected evaluation and Monday deliverables

Preserve complete-event splits:25training,7validation,8test per fold. Training-only normalization; validation-only checkpoint and extent threshold selection. Retrain from scratch; no contaminated checkpoint initialization. Compare persistence and learned forecasts on the same corrected origins at all48leads. Report wet/all-cell depth,extent precision/recall/F1/CSI,HU/HV,derived velocity,wetting/drying transitions and measured runtime. Sampling limits and future-forcing assumptions belong in notes.

Monday priority: methods and audit evidence; one fresh baseline versus persistence; halo comparison if time permits. Placeholders: lead curves, representative held-out maps, and actual training/inference time. If new results are unavailable, label the deck as methods plus repair progress. No operational, mass-conservation, unseen-region or graph-performance claim is established.

## Source documents to upload if more detail is needed

1. docs/stage2/TARGET_DATA_AUDIT.md — authoritative data defects, fixes and rebuild.
2. docs/stage2/V5B_EXPERIMENT_REPORT.md — halo, ablations and evaluation design. Numeric performance is obsolete evidence.
3. docs/stage2/V5D_GRAPH_COMMUNICATION.md — implemented local graph vs future hierarchy.
4. docs/stage2/STAGE2_V5_PLAN.md — forecast contract and roadmap. Earlier warm-start guidance is superseded by corrected-data training from scratch.
5. archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md and transitionV1,V2,V3,V4a,V4b documents — earlier methods. Numeric claims are historical.
6. docs/project/MODEL_DESCRIPTION.md — earliest peak-map architecture, not the active direct model.

The ZIP bundle preserves these source paths. Old LEADERSHIP_BRIEFING.md is included only as historical context: use this brief for the current story.
'''
# Keep prose spacing readable while preserving code identifiers and file paths.
for old,new in {
    'all40':'all 40', 'on0.5':'on 0.5', '40 TRITON events,240hours/event,480half-hour':'40 TRITON events, 240 hours/event, 480 half-hour',
    'a480×300tensor':'a 480 × 300 tensor', 'Approximately6900tiles and34HUC12units':'Approximately 6,900 tiles and 34 HUC12 units',
    'tile80×80cells':'tile 80 × 80 cells', 'approximately10m':'approximately 10 m', 'about9.506m':'about 9.506 m',
    'Leads1–48half-hour':'Leads 1–48 half-hour', 'through48updates':'through 48 updates',
    'an8-cell':'an 8-cell', ':96×96observed':': 96 × 96 observed', 'an80×80central':'an 80 × 80 central',
    'central80×80crop':'central 80 × 80 crop', 'Approximately76m':'Approximately 76 m',
    'HU/HV':'HU/HV', 'HU=h*u':'HU = h*u', 'HV=h*v':'HV = h*v',
    'peak V2/V3; timestamp':'peak V2/V3; timestamp', 'Stage-1Max':'Stage-1 Max',
    'timestampV4':'timestamp V4', 'transitionV1':'transition V1', 'continuedV1':'continued V1',
    'multi-stepV1':'multi-step V1', 'regime-awareV2':'regime-aware V2', 'exact-localV3':'exact-local V3',
    'history/gateV4a':'history/gate V4a', 'adapter/gateV4b':'adapter/gate V4b',
    'directV5-A':'direct V5-A', 'haloV5-B':'halo V5-B', 'forcingV5-C':'forcing V5-C',
    'graphV5-D':'graph V5-D', 'three-seed fold0':'three-seed fold 0',
    'baseline,halo':'baseline, halo', 'gate. One-seed':'gate. One-seed',
    ';6900nodes and27262directed':' ; 6,900 nodes and 27,262 directed',
    'share event,origin,lead':'share event, origin, lead', 'HUC12hierarchy':'HUC12 hierarchy',
    ':25training,7validation,8test':': 25 training, 7 validation, 8 test',
    'all48leads':'all 48 leads', 'each event has108':'each event has 108',
    'Each event has108':'Each event has 108', 'Initial array186578':'Initial array 186578',
    'Recovery187031':'Recovery 187031', 'transitionV1,V2,V3,V4a,V4b':'transition V1, V2, V3, V4a, V4b',
}.items():
    brief=brief.replace(old,new)
(HERE/'CHATGPT_SOURCE_BRIEF.md').write_text(brief)

manifest=[]
with zipfile.ZipFile(HERE/'CHATGPT_SOURCE_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ('CHATGPT_SOURCE_BRIEF.md','EXPERIMENT_HISTORY.md','SLIDE_TEXT_AND_NOTES.md'):
        z.write(HERE/name,name)
    for path in dict.fromkeys(SRC.values()):
        file=ROOT/path
        z.write(file,'sources/'+path)
        manifest.append(dict(path=path,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

# A schematic contact sheet uses the same stored native-shape coordinates.
def thumbnail(page,scale=.42):
    image=Image.new('RGB',(round(W*96*scale),round(H*96*scale)),WHITE);draw=ImageDraw.Draw(image)
    for item in page.items:
        x,y,w,h=(item[k]*96*scale for k in ('x','y','w','h'))
        if item['kind']=='rect':draw.rectangle((x,y,x+w,y+h),fill=item['color'],outline=item.get('line'))
        elif item['kind']=='arrow':draw.polygon([(x,y+h*.28),(x+w*.62,y+h*.28),(x+w*.62,y),(x+w,y+h/2),(x+w*.62,y+h),(x+w*.62,y+h*.72),(x,y+h*.72)],fill=item['color'])
        else:
            size=item['size']*96/72*scale
            font=ImageFont.truetype(BOLD_FONT if item['bold'] else FONT,max(1,round(size)))
            for j,line in enumerate(item['lines']):draw.text((x+1.44*scale,y+j*size*1.18),line,font=font,fill=item['color'])
    return image

thumbs=[thumbnail(page) for page in pages]
tw,th=thumbs[0].size;cols=3;gap=18
sheet=Image.new('RGB',(cols*tw+(cols+1)*gap,math.ceil(len(thumbs)/cols)*(th+gap)+gap),'#D5E2E7')
for i,t in enumerate(thumbs):sheet.paste(t,(gap+(i%cols)*(tw+gap),gap+(i//cols)*(th+gap)))
sheet.save(HERE/'contact_sheet.png')

readme=f'''# Editable Monday presentation

- **TRITON_Methods_and_Experiment_History.pptx** — {len(pages)} slides, including 4 appendix slides; native editable text and diagrams, with speaker notes/source paths. Open in PowerPoint or LibreOffice Impress. The 26-entry history catalog includes proposed work and evaluation protocols; it is not a count of completed experiments.
- **preview.html** — standalone local browser preview with notes; print to PDF if wanted. The preview is a schematic rendering from the same layout data, not a PowerPoint renderer.
- **SLIDE_TEXT_AND_NOTES.md** — all slide wording and presenter notes.
- **CHATGPT_SOURCE_BRIEF.md** and **EXPERIMENT_HISTORY.md** — upload these two first if asking ChatGPT to redesign the deck. The brief includes a paste-ready prompt and evidence rules.
- **CHATGPT_SOURCE_BUNDLE.zip** — the brief, history, notes and full source documents with their paths and SHA256 manifest. Extract before uploading if your client does not accept ZIP archives.
- **contact_sheet.png** — compact visual review of all slides.
- **build_deck.py** — regeneration script; requires Python3.11+, python-pptx and Pillow. Initial temporary command: `PYTHONPATH=/tmp/triton-slides-deps /ccs/home/haoranniu/miniconda3/bin/python docs/presentations/monday_20261012/build_deck.py`.

Before Monday: update the dated rebuild snapshot on slide15 and replace slide16 with corrected-data evaluation. Old forecasting percentages and metric charts are intentionally omitted as evidence. The files do not launch training or modify rebuild jobs.

The source documents contain historical numeric claims; CHATGPT_SOURCE_BRIEF.md explicitly supersedes them. Details of the implemented halo are checked against stage2/model.py, rather than relying on the larger-input-halo shorthand in older plans.
'''
(HERE/'README.md').write_text(readme)

# Structural and editable-content verification.
loaded=Presentation(deck)
assert len(loaded.slides)==len(pages)
for i,slide in enumerate(loaded.slides,1):
    assert slide.has_notes_slide and 'Sources:' in slide.notes_slide.notes_text_frame.text
    assert sum(bool(s.has_text_frame) for s in slide.shapes)>4
    for s in slide.shapes:
        assert s.left>=0 and s.top>=0 and s.left+s.width<=loaded.slide_width+10
        assert s.top+s.height<=loaded.slide_height+10
with zipfile.ZipFile(deck) as z:
    assert z.testzip() is None
print(json.dumps(dict(slides=len(pages),catalog_entries=len(HISTORY),validated_snapshot=len(validated),deck=str(deck),status='reopened_and_checked'),indent=2))
