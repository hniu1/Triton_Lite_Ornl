"""Add editable graph overview and message-passing detail to the methods deck."""
import json
import inspect
import os
import subprocess
import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches
from insert_editable_architecture import diagram

HERE=Path(__file__).resolve().parent
OUT=HERE/'architecture'
NAVY,TEAL,INK,MUTED='#123147','#007F83','#243C4B','#58707F'


class Figure:
    def __init__(self,title,subtitle):
        self.items=[]
        self.text(.55,.35,title,27,NAVY,True)
        self.text(.57,.96,subtitle,14,MUTED)
    def text(self,x,y,text,size=13,color=INK,bold=False):
        self.items.append(dict(kind='text',x=x,y=y,text=text,size=size,color=color,bold=bold))
    def box(self,x,y,w,h,title,body='',teal=False):
        self.items.append(dict(kind='box',x=x,y=y,w=w,h=h,fill='#DCF0EE' if teal else '#EEF4F6',edge='#86BEBC' if teal else '#D1E0E6'))
        self.text(x+.18,y+.18,title,15,TEAL if teal else NAVY,True)
        if body:self.text(x+.18,y+.68,body,13)
    def arrow(self,points,color=TEAL,dashed=False):
        self.items.append(dict(kind='arrow',points=points,color=color,dashed=dashed))
    def line(self,points,color=TEAL):
        self.items.append(dict(kind='line',points=points,color=color))
    def footer(self,statement):
        self.line([(.57,9.35),(19.4,9.35)],'#D1E0E6')
        self.text(.60,9.55,statement,12)
        self.text(.60,9.95,'Implemented local graph prototype; corrected-data forecasting benefit remains unproven.',11,MUTED)


overview=Figure('Tile graph communication inside the direct U-Net',
                'V5-D prototype  |  One tile = one node; graph context conditions the raster forecast')
overview.box(.60,2.03,4.00,2.47,'1  Shared tile encoder',
             'Terrain + observed state/history\nForcing, times and requested lead\nOptional hydraulic halo\nEncode every context tile')
overview.box(5.25,2.03,4.00,2.47,'2  Node features',
             'Pooled 20 × 20 spatial features\nFour directional boundary summaries\nTile forcing/time/lead conditioning\nLinear node projection')
overview.box(9.90,2.03,3.72,2.47,'3  Message passing',
             'Learn messages on tile edges\nMean incoming messages\nResidual node update\nRepeat K spatial layers',True)
overview.box(14.30,2.03,5.08,2.47,'4  U-Net bottleneck context',
             'c′ = c + P(zᴷ − z⁰)\nConcatenate with pooled raster features\nDecoder + original skip features\nCentral depth, extent, HU/HV forecast')
for x in (4.60,9.25,13.62):overview.arrow([(x,3.28),(x+.65,3.28)])

overview.text(.78,5.00,'Shared-boundary tile adjacency',17,NAVY,True)
for x,y,label in [(2.56,5.82,'North'),(.81,7.13,'West'),(4.31,7.13,'East'),(2.56,8.44,'South')]:
    overview.box(x,y,1.23,.64,label,teal=True)
overview.box(2.56,7.13,1.23,.64,'Center',teal=True)
# Two offset arrows per boundary show both directions explicitly.
for x in (2.95,3.38):
    overview.arrow([(x,6.46),(x,7.13)] if x==2.95 else [(x,7.13),(x,6.46)])
    overview.arrow([(x,7.77),(x,8.44)] if x==2.95 else [(x,8.44),(x,7.77)])
for y in (7.32,7.57):
    overview.arrow([(2.04,y),(2.56,y)] if y==7.32 else [(2.56,y),(2.04,y)])
    overview.arrow([(3.79,y),(4.31,y)] if y==7.32 else [(4.31,y),(3.79,y)])
overview.box(7.06,5.23,5.68,3.20,'Graph sampling contract',
             'Connected tiles share one event, origin and lead.\nExpand context to cover the graph layers.\nUse observed inputs; score designated centers.\nLocal communication does not cover the entire basin.',True)
overview.box(13.33,5.23,6.05,3.20,'What the graph adds',
             'Halo: nearby pixel-level hydraulic context.\nGraph: learned tile-to-tile feature communication.\nZero-initialized projection starts with no added context.\nHUC12 hierarchy remains a proposed extension.')
overview.footer('Edges encode geometric contact. Learned messages are features, not conserved water fluxes.')

detail=Figure('What happens in one message-passing layer?',
              'Source tile j → destination tile i  |  The same learned functions are shared across edges and nodes')
detail.box(.60,2.03,4.06,2.09,'1  Build each edge message',
           'm(j→i) = Mₖ([zⱼ, zᵢ, aⱼᵢ])\nUse source and destination features.\nEdge attributes a are optional.',True)
detail.box(5.29,2.03,4.06,2.09,'2  Aggregate at destination',
           'm̄ᵢ = mean of incoming messages\nAverage over the tile’s neighbors.\nDegree can vary between tiles.',True)
detail.box(9.98,2.03,4.06,2.09,'3  Update the destination',
           'zᵢ ← zᵢ + Uₖ([zᵢ, m̄ᵢ])\nResidual update retains local features.\nNo incoming edges:\nnode stays unchanged.',True)
detail.box(14.67,2.03,4.71,2.09,'4  Repeat and condition',
           'Repeat for K graph layers.\nc′ᵢ = cᵢ + P(zᵢᴷ − zᵢ⁰)\nUse c′ᵢ in the U-Net bottleneck.',True)
for x in (4.66,9.35,14.04):detail.arrow([(x,3.05),(x+.63,3.05)])
detail.box(.60,4.90,5.80,3.05,'Reading the symbols',
           'zᵢ: encoded features for tile i\nMₖ: learned message MLP\nUₖ: learned update MLP\nP: learned linear projection\n[ , ]: concatenation; k: graph layer')
detail.box(7.10,4.90,5.80,3.05,'Two layers = two spatial hops',
           'Layer 1 uses adjacent-tile features.\nLayer 2 can carry features from two hops away.\nThese are spatial exchanges, not forecast steps.\nAll nodes still share the same origin and lead.')
detail.box(13.60,4.90,5.78,3.05,'How we test its value',
           'Same U-Net and connected context batches:\n• U-Net without graph messages\n• Local capacity control with self edges\n• Graph with actual adjacency edges\nMatch data, losses and evaluation origins.')
detail.footer('Messages have no water-volume interpretation; physical conservation requires a separate formulation.')

OVERVIEW_NOTES='''Tile graph architecture — presenter notes

Main explanation
The graph is an extension inside our direct, lead-conditioned U-Net. It does not replace the raster encoder and decoder. Each tile remains an 80 × 80 raster patch for prediction, but it is also represented as one node in a connected neighborhood. We encode local terrain and observed hydraulic state/history, summarize the tile and its boundaries, pass learned features across tile edges, and feed the additional context into the U-Net bottleneck. The decoder still predicts a central raster state at the requested future lead.

1. Shared tile encoding
Every tile in a sampled neighborhood is encoded with the same model weights. Inputs include terrain, current and older observed hydraulics, forcing through the requested target time, tile attributes and forecast times/lead. An optional halo can be retained, as in the prototype configurations. Graph and halo are complementary methods: halo provides spatial raster context near a tile border, while the graph communicates compressed learned tile features. No target hydraulic state is an input.

2. Construct one feature vector per tile
The implementation globally averages the 20 × 20 x3 encoder features. It also computes four directional boundary summaries by averaging the top, bottom, left and right boundaries of the early x1 feature map. Those summaries are learned-feature summaries; they are not measured boundary fluxes. Concatenate pooled features, boundary summaries and the existing forcing/time/lead conditioning. A linear node encoder maps this concatenation to the graph node dimension, normally the conditioning dimension. The diagram calls this initial vector z⁰.

3. Establish adjacency
The implemented topology comes from actual shared contacts in the tile-label raster. It includes both directed edges for a touching pair. The topology has 6,900 tile nodes and 27,262 directed edges in the audited configuration. The small cross diagram is illustrative, not a claim that every tile has exactly four neighbors. Geometric contact does not establish channel connectivity, hydraulic passability or permanently one-way flow. Verified hydraulic edge features are a later extension. The current base graph can run with no edge attributes.

4. Synchronize the neighborhood
All connected nodes in a group must share an event, forecast origin and requested lead. Distinct groups have distinct IDs; validation rejects edges between groups. Connected-neighborhood loading expands around sampled central tiles by at least as many spatial hops as graph layers. We record truncation at the outer context boundary and score only designated central tiles. The loading pipeline may read context labels for its own batch structure, but future hydraulic labels must enter neither graph features nor scored center inputs.

5. Pass and aggregate messages
For each directed source-to-destination edge, a learned message MLP uses both tile feature vectors and optional edge attributes. Incoming messages are averaged by destination degree. An update MLP combines a node’s existing features and its aggregated message, and adds the resulting change back to the node. Two graph layers can move information across two spatial hops. A graph layer is not a half-hour forecast step and does not simulate hydraulic evolution.

6. Return context to the raster U-Net
The graph returns zᴷ. The implementation computes c′ = c + P(zᴷ − z⁰), where c is the preexisting conditioning vector and P is a learned linear projection. P is initialized to zero, so the added graph branch initially contributes no extra conditioning. Broadcast c′ over the pooled 10 × 10 spatial grid and concatenate it into the bottleneck. The original decoder, skip connections, activity gate, depth/discharge residual heads and wet head remain. Forecast outputs retain the direct origin-to-target formulation. Zero projection is an initialization property, not a guarantee that the learned graph will help.

7. What is implemented and what is not
Local adjacency message passing, directional summaries, connected samplers, central-only scoring, topology checks and GPU smoke execution are implemented. A hierarchical graph with HUC12 coarse watershed nodes is proposed, not implemented. Learned graph messages are not water exchange, and this module does not impose mass or momentum conservation. A conservative update needs explicit storage, source/sink and shared-boundary-flux accounting.

8. Interpretation of the previous checks
The graph prototype has software and tiny training-cohort diagnostics, not a completed corrected-data held-out forecasting comparison. The tiny learning checks used legacy hydraulic targets and do not establish the intended forecast skill. Their apparent depth/extent and velocity behavior cannot establish a physical advantage or diagnose the correct-data model. We should first establish the corrected-data direct baseline and halo comparison, then rerun graph learning and a controlled comparison.

Suggested one-minute explanation
“We keep the tile U-Net and give it a communication branch. Every neighboring tile is encoded, then compressed into a vector containing its internal spatial features, boundary summaries and forecast conditioning. A message network exchanges those vectors along shared tile edges, and an update network combines the incoming information with each tile’s own representation. After a few spatial layers, the new graph context modifies the U-Net bottleneck. The raster decoder then predicts the central future hydraulic map. These are learned feature messages, not water fluxes, so the graph’s forecasting benefit and physical consistency still require corrected-data tests.”

Sources
stage2/graph.py — message/update MLPs, degree-normalized incoming aggregation, residual update, group checks.
stage2/model.py — pooled spatial features, four boundary summaries, node projection and c + P(context − nodes) conditioning.
stage2/graph_data.py — adjacency, neighborhood expansion and synchronized center/context batches.
docs/stage2/V5D_GRAPH_COMMUNICATION.md — implementation scope, diagnostics and controlled comparison design.
docs/stage2/TARGET_DATA_AUDIT.md — legacy target contamination and corrected-data restart.
'''

DETAIL_NOTES='''Message-passing equations — detailed presenter notes

Walk through the four operations from left to right.

1. Edge messages
At graph layer k, take the source node zⱼ and destination node zᵢ and concatenate them. If configured, append an edge-attribute vector aⱼᵢ. A learned MLP Mₖ produces the message mⱼ→ᵢ. The implementation uses Linear → GELU → Linear with its output in the graph node dimension. It has separate message MLP parameters at each layer. Because source and destination features both enter, a message is conditioned on the sending tile and the receiving tile. Messages for the two directions of a touching pair may differ. Edge attributes are optional; do not present hydraulic routing attributes as already verified.

2. Destination aggregation
Sum messages at each destination and divide by the number of incoming edges, using a degree floor of one to avoid division by zero. This is mean aggregation, allowing nodes with different neighbor counts. It is not attention: there is no learned normalized attention coefficient in this implementation. It is also not summing a conserved hydraulic flux. Mean aggregation acts on learned feature vectors.

3. Residual node update
Concatenate the current node vector and the aggregated incoming message. Pass that through a separate update MLP Uₖ (Linear → GELU → Linear), then add the change to the original node vector. The implementation masks the update for nodes with no incoming edges. If the whole graph has no edges, the module immediately returns its input nodes. Self edges are not required in the normal adjacency graph because the residual update and update input preserve the destination’s own representation. The local capacity control deliberately supplies only self edges instead of neighbor edges.

4. Repeat and inject graph context
Repeat these operations K times with separate per-layer MLP parameters. Each layer propagates influence one additional spatial hop, assuming the sampled context includes those neighbors. Two layers are not two forecast time steps. Finally the graph-conditioned vector is c′ᵢ = cᵢ + P(zᵢᴷ − zᵢ⁰). P starts at zero, preserving the preexisting conditioning initially. P is learned during optimization. The graph-disabled default has graph_layers=0 and does not instantiate the graph modules. An enabled zero-projection branch is a separate configuration whose initial output can agree with the reference when the other settings and inputs are identical.

How this differs from halo
Halo supplies neighboring observed hydraulic pixels to a convolutional branch and adds central cropped spatial features. The graph supplies compressed features across tile edges. Their receptive fields and parameterizations differ. Neither establishes full-domain upstream routing. A HUC12 hierarchy would require a further graph design; it is not part of the equations on this slide.

The fair comparison
Use three arms with the same central examples, contextual nodes, event splits, initial U-Net weights, losses, origins and budget: U-Net without graph messages; local capacity control with the same extra node/message/update/projection capacity but self edges; and the graph with true adjacency edges. Context batching can itself affect batch normalization during training, so use the same contextual batch structure across arms. Report cost, depth, extent, HU/HV, derived velocity and wetting/drying behavior. A selected tiny training-cohort diagnostic is a learning check, not generalization evidence. No legacy-data accuracy number establishes corrected-data graph value.

Common questions
Are messages physical flows? No. They are latent feature vectors without water-volume units or antisymmetric flux constraints.
Is this a Transformer or attention graph? No. The current implementation is MLP messaging with mean aggregation.
Do nodes use future hydraulic labels? No. They use observed origin/history information, terrain and supplied forcing through the requested valid time.
Is one tile one watershed? No. A tile is an 80 × 80 raster patch. HUC12 coarse units are separate proposed nodes.
Do two graph layers produce a one-hour forecast? No. Graph layers are spatial hops; the requested forecast lead is a separate conditioning input.
Does successful GPU execution prove the graph is better? No. It verifies a software path. Corrected-data held-out comparisons are required.

Sources: stage2/graph.py; stage2/model.py; stage2/graph_data.py; docs/stage2/V5D_GRAPH_COMMUNICATION.md; docs/stage2/TARGET_DATA_AUDIT.md.
'''


def render_assets(items,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(20,10.3));fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax.set(xlim=(0,20),ylim=(10.3,0));ax.axis('off')
    for item in items:
        kind=item['kind']
        if kind=='box':
            ax.add_patch(FancyBboxPatch((item['x'],item['y']),item['w'],item['h'],
                boxstyle='round,pad=0.02,rounding_size=0.1',facecolor=item['fill'],edgecolor=item['edge'],lw=1.3))
        elif kind=='circle':
            ax.add_patch(Circle((item['x']+item['w']/2,item['y']+item['h']/2),
                item['w']/2,facecolor=item['fill'],edgecolor=item['edge']))
        elif kind in ('text','plus'):
            ax.text(item['x'],item['y'],item['text'],fontsize=item['size'],color=item['color'],
                    fontweight='bold' if item.get('bold') else 'normal',
                    va='center' if kind=='plus' else 'top',
                    ha='center' if kind=='plus' else 'left',linespacing=1.45)
        else:
            points=item['points']
            for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
                if kind=='arrow' and i==len(points)-2:
                    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,lw=1.8,
                        color=item['color'],linestyle='--' if item.get('dashed') else '-'))
                else:ax.plot([a[0],b[0]],[a[1],b[1]],lw=1.8,color=item['color'])
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    (OUT/f'{name}.json').write_text(json.dumps(items,indent=2)+'\n')


def render(items,name):
    path=OUT/f'{name}.json'
    path.write_text(json.dumps(items,indent=2)+'\n')
    code=('from pathlib import Path\nimport json\nOUT=Path('+repr(str(OUT))+')\n'+
          inspect.getsource(render_assets)+'\nrender_assets(json.loads(Path('+repr(str(path))+
          ').read_text()),'+repr(name)+')\n')
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    subprocess.run(['/ccs/home/haoranniu/miniconda3/envs/triton/bin/python','-c',code],env=env,check=True)


def build():
    render(overview.items,'graph_architecture')
    render(detail.items,'message_passing_detail')
    original=HERE/'TRITON_Methods_with_Editable_Architecture.pptx'
    prs=Presentation(original)
    slide=prs.slides[11]
    for shape in list(slide.shapes):
        el=shape._element;el.getparent().remove(el)
    diagram(slide,prs.slide_width/914400,overview.items,OVERVIEW_NOTES)
    appendix=prs.slides.add_slide(prs.slide_layouts[6])
    diagram(appendix,prs.slide_width/914400,detail.items,DETAIL_NOTES)
    output=HERE/'TRITON_Methods_with_Halo_and_Graph.pptx'
    prs.save(output)
    standalone=Presentation();standalone.slide_width=Inches(20);standalone.slide_height=Inches(11)
    for figure,notes in ((overview,OVERVIEW_NOTES),(detail,DETAIL_NOTES)):
        s=standalone.slides.add_slide(standalone.slide_layouts[6]);diagram(s,20,figure.items,notes)
    standalone.save(OUT/'Editable_Graph_Architecture.pptx')
    (OUT/'GRAPH_SPEAKER_NOTES.md').write_text(OVERVIEW_NOTES+'\n\n'+DETAIL_NOTES)
    checked=Presentation(output)
    assert len(checked.slides)==22
    for index in (11,21):
        assert not any(s.shape_type==13 for s in checked.slides[index].shapes)
        assert len(checked.slides[index].notes_slide.notes_text_frame.text)>2000
    baseline=Presentation(original)
    for i in range(21):
        if i==11:continue
        assert baseline.slides[i]._element.xml==checked.slides[i]._element.xml,i
        assert baseline.slides[i].notes_slide.notes_text_frame.text==checked.slides[i].notes_slide.notes_text_frame.text,i
    for path in (output,OUT/'Editable_Graph_Architecture.pptx'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
    print(f'{output}\nSlide 12: graph overview; slide 22: message-passing detail. All other slides preserved.')


if __name__=='__main__':build()
