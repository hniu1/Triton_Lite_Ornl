"""Make slide 8 a unified, editable halo + optional graph architecture."""
import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches
from add_graph_architecture import Figure, render, OVERVIEW_NOTES
from insert_editable_architecture import diagram, NOTES

HERE=Path(__file__).resolve().parent
PURPLE='#7050A3'
f=Figure('Direct U-Net with halo and tile graph context',
         'Unified architecture  |  Halo enters early spatial features; optional graph messages modify bottleneck conditioning')


def gbox(x,y,w,h,title,body):
    f.box(x,y,w,h,title,body)
    f.items[-3]['fill']='#F0EAF8';f.items[-3]['edge']='#BAA8D4'
    f.items[-2]['color']=PURPLE


gbox(4.48,1.82,2.64,1.46,'Tile node encoder',
     'Pooled encoder features\nFour boundary summaries\nConditioning c → z⁰')
gbox(7.68,1.82,2.64,1.46,'Message passing',
     'Neighbor tile vectors\nMean messages + updates\nK layers → zᴷ')
f.arrow([(7.12,2.56),(7.68,2.56)],PURPLE)
f.box(10.90,1.82,3.09,1.46,'Forecast conditioning',
      'Causal forcing through t + L\nTile, origin, target and lead\nConditioning vector c')
gbox(10.90,3.63,3.09,1.02,'Graph conditioning',
     'c′ = c + P(zᴷ − z⁰)')
f.arrow([(10.32,2.56),(10.59,2.56),(10.59,4.16),(10.90,4.16)],PURPLE)
f.arrow([(12.445,3.28),(12.445,3.63)])
# Conditioning is part of the initial node features, as in the implementation.
f.arrow([(11.66,1.82),(11.66,1.48),(5.80,1.48),(5.80,1.82)],'#58707F')
f.text(7.38,1.29,'base conditioning c',10,'#58707F')

f.box(.60,4.45,3.10,1.16,'Central terrain + mask','6 terrain layers; 80 × 80')
f.box(.60,6.01,3.10,1.16,'Observed hydraulic history','S(t), S(t − 1); h, HU, HV')
f.box(4.48,5.15,2.64,1.53,'Central feature fusion',
      'Static and state CNNs\nHistory adapter\n80 × 80 features')
f.arrow([(3.70,5.03),(4.08,5.03),(4.08,5.53),(4.48,5.53)])
f.arrow([(3.70,6.59),(4.08,6.59),(4.08,6.13),(4.48,6.13)])
f.box(7.68,5.15,2.64,1.53,'Spatial encoder','80 → 40 → 20\nCentral raster features')
# Addition junction for cropped halo features.
f.items.append(dict(kind='circle',x=7.23,y=5.73,w=.30,h=.30,fill='#007F83',edge='#007F83'))
f.items.append(dict(kind='plus',x=7.38,y=5.88,text='+',size=13,color='#FFFFFF'))
f.arrow([(7.12,5.88),(7.23,5.88)])
f.arrow([(7.53,5.88),(7.68,5.88)])
f.box(10.90,5.30,3.09,1.23,'U-Net bottleneck','Pooled spatial grid: 10 × 10\nConcatenate conditioning c′')
f.arrow([(10.32,5.88),(10.90,5.88)])
f.arrow([(12.445,4.65),(12.445,5.30)],PURPLE)
f.box(14.57,5.15,2.15,1.53,'U-Net decoder','20 → 40 → 80\nSkip-feature fusion')
f.arrow([(13.99,5.88),(14.57,5.88)])
f.box(17.30,4.80,2.10,1.98,'Forecast heads','Δh, ΔHU, ΔHV\nActivity gate\nWet/dry logits')
f.arrow([(16.72,5.88),(17.30,5.88)])

# Two encoder-derived inputs to the graph: x3 pooled features and x1 boundaries.
f.arrow([(9.00,5.15),(9.00,4.92),(5.80,4.92),(5.80,3.28)],PURPLE)
f.arrow([(7.38,5.73),(7.38,4.92),(5.80,4.92)],PURPLE)
f.text(6.02,4.60,'tile features + boundaries',10,PURPLE)
f.arrow([(9.58,5.15),(9.58,4.82),(15.645,4.82),(15.645,5.15)],'#58707F',True)
f.text(14.54,4.53,'U-Net skip features',10,'#58707F')

f.box(.60,7.66,3.10,1.40,'Observed hydraulic halo',
      '96 × 96 at t and t − 1\n8-cell border around center',True)
f.box(4.48,7.66,2.64,1.40,'Separate halo CNN',
      'Encode 96 × 96\nCrop: 80 × 80 features',True)
f.arrow([(3.70,8.33),(4.48,8.33)])
f.arrow([(7.12,8.33),(7.38,8.33),(7.38,6.04)])
f.text(8.06,7.63,'Raster pathway stays intact.\nGraph returns a conditioning correction.\nOne direct forecast for the requested lead.',13,'#007F83',True)
f.box(14.57,7.66,4.83,1.40,'Central forecast at t + L',
      'Observed S(t) + gated residual change\n80 × 80 depth, signed HU/HV and wet probability',True)
f.arrow([(18.35,6.78),(18.35,7.66)])
f.footer('Purple: graph context branch. Learned messages are features, not hydraulic fluxes; hierarchy is separate future work.')

updated_notes=NOTES.replace('our implemented V5-B architecture',
                          'our implemented V5-B halo architecture with the optional V5-D local graph branch')
start=updated_notes.index('9. What is omitted from this main figure')
end=updated_notes.index('10. Evidence and how to present this honestly')
updated_notes=(updated_notes[:start]+'''9. Optional graph branch now included on this unified slide
Read the purple branch from left to right. The graph is not an encoder before the whole U-Net. It sits after the raster encoder and changes the bottleneck conditioning. Pool the 20 × 20 encoder features and collect four boundaries from the early 80 × 80 feature map. Combine those summaries with the same base conditioning c to form one node vector z⁰ per tile. Neighboring context tiles follow the same shared encoding path. The diagram shows one representative tile, not all neighbors in parallel.

Message passing produces zᴷ by computing source/destination messages, averaging incoming messages and applying residual node updates. The graph returns the projected feature change P(zᴷ − z⁰). Add this to the base conditioning to get c′. This graph-conditioned vector is what is concatenated into the 10 × 10 bottleneck. The raster features still feed the bottleneck directly, and the U-Net skip connections and decoder remain. The graph does not replace the halo or decode a raster on its own. P starts at zero, so the additional context initially contributes nothing.

The upper gray connection carries base conditioning into initial node features. The purple upward connection combines encoder pooled features and early boundary summaries. The purple output reaches the graph residual conditioning box, not the early halo addition junction. The output of that box, c′, enters the bottleneck. These are the three graph connections to emphasize when presenting the diagram.

In graph-disabled mode, the original c conditions the bottleneck directly. In graph-enabled mode, c′ takes its place. Local forcing and the optional extra discharge-correction gate are omitted. The local graph is implemented, but hydraulic routing edges, HUC12 hierarchy and conservative flux updates remain separate extensions.

'''+updated_notes[end:])
updated_notes=updated_notes.replace('The resulting conditioning vector is expanded spatially and concatenated',
    'The base vector c also enters the graph node features. In graph-enabled mode the updated vector c′ is expanded spatially and concatenated')
updated_notes+='\n\nAdditional graph walkthrough and sources\n'+OVERVIEW_NOTES


def build():
    render(f.items,'integrated_halo_graph_architecture')
    source=HERE/'TRITON_Methods_with_Halo_and_Graph.pptx'
    output=HERE/'TRITON_Methods_Integrated_Halo_Graph.pptx'
    prs=Presentation(source);slide=prs.slides[7]
    for shape in list(slide.shapes):
        el=shape._element;el.getparent().remove(el)
    diagram(slide,prs.slide_width/914400,f.items,updated_notes)
    prs.save(output)
    one=Presentation();one.slide_width=Inches(20);one.slide_height=Inches(11)
    single=one.slides.add_slide(one.slide_layouts[6]);diagram(single,20,f.items,updated_notes)
    one.save(HERE/'architecture/Editable_Integrated_Halo_Graph.pptx')
    (HERE/'architecture/INTEGRATED_SPEAKER_NOTES.md').write_text(updated_notes)
    check=Presentation(output);old=Presentation(source)
    assert len(check.slides)==len(old.slides)==22
    for i in range(22):
        if i==7:continue
        assert old.slides[i]._element.xml==check.slides[i]._element.xml,i
        assert old.slides[i].notes_slide.notes_text_frame.text==check.slides[i].notes_slide.notes_text_frame.text,i
    assert not any(s.shape_type==13 for s in check.slides[7].shapes)
    assert 'Optional graph branch now included' in check.slides[7].notes_slide.notes_text_frame.text
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    print(f'{output}\nSlide 8: integrated native-shape architecture and expanded notes. Other slides preserved.')


if __name__=='__main__':build()
