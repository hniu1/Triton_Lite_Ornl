Direct U-Net with hydraulic halo — detailed presenter walkthrough

Opening explanation
This figure describes our implemented V5-B halo architecture with the optional V5-D local graph branch. It combines the observed hydraulic state, recent hydraulic history, terrain, forcing and a requested forecast lead. The central prediction remains an 80 × 80 tile. The hydraulic halo adds immediate spatial context around that tile. This is a methods explanation, not a claim that halo has already improved accuracy on the corrected data.

1. Define the forecast origin and outputs
Let t be the forecast origin and L the requested lead. S(t) contains depth h and two signed unit-discharge components HU and HV. HU/HV have units of m²/s, not velocity units. The older state is one half-hour before the origin. We request L from 1 to 48 half-hour steps, corresponding to 0.5–24 hours. The model predicts depth, wet probability and signed discharge at t+L. Velocity can be derived by dividing discharge by depth, with the declared wetness/denominator policy; shallow water requires separate diagnostics.

2. Central terrain and hydraulic history
The upper-left inputs contain six static raster layers and the valid-cell mask. Terrain layers include DEM, flow accumulation, stream mask, distance to stream, slope and relative elevation. Static features use their own CNN. The current hydraulic state is also encoded. In the two-state adapter configuration, the difference between current and older states goes through a learned history adapter before state encoding. Internally the depth state uses log1p depth, while the discharge components remain signed. The static and state features are added to form the central 80 × 80 spatial feature map. The schematic groups these operations for readability.

3. Why we introduced halo
An independent tile can hide water conditions just beyond its edge. The halo branch supplies observed hydraulic conditions around the central tile so the network can use nearby water levels and signed discharge when forming local spatial features. This is a hypothesis about useful information, rather than proof of a corrected-data improvement. It does not by itself provide watershed-wide upstream routing.

4. Exact halo implementation
With an eight-cell border on each side, an 80 × 80 center becomes a 96 × 96 hydraulic input. At the actual approximately 9.506 m cell spacing, this border is about 76 m wide. Both the current and older hydraulic neighborhoods are read. The halo CNN receives current hydraulic channels and their changes from the older neighborhood, with log-transformed depth. A separate convolutional encoder processes the 96 × 96 patch. Its central 80 × 80 feature crop is added at the plus sign to the central tile features BEFORE spatial downsampling. This is not an enlarged static-terrain U-Net. The supervised forecast output is still central-only. Halo influence depends on the encoder's receptive field; supplying a border does not imply that every border cell affects every output cell.

5. Forcing and forecast conditioning
The causal temporal encoder provides forcing information through the requested target time t+L, never after it. Tile attributes, target-time features, origin-time features and lead features are also encoded, concatenated and passed through a conditioning MLP. The base vector c also enters the graph node features. In graph-enabled mode the updated vector c′ is expanded spatially and concatenated with the pooled raster features at the 10 × 10 bottleneck. Known future forcing is an explicit input assumption. This model does not predict weather or generate its own future hydrographs. Future hydraulic states are labels only and never enter these inputs.

6. The U-Net spatial pathway
Spatial features are downsampled from 80 to 40 to 20 cells, then pooled to the 10 × 10 bottleneck. The conditioned bottleneck is decoded back through 20, 40 and 80 cells using interpolation and convolutional blocks. Skip features from the matching encoder scales are concatenated into the decoder. The dashed connection is a schematic representation of these multiple scale-specific skip connections, not a single shared tensor. They help preserve fine spatial information while the bottleneck supplies broader context and forecast conditioning.

7. Heads and residual output construction
The decoder produces raw depth-change and signed-discharge-change maps, a wet/dry logit map and, in this configuration, a spatial activity gate. The sigmoid activity gate multiplies the depth and discharge residuals. The model adds those gated residuals to the observed origin state. Depth is clamped nonnegative; HU/HV can be positive or negative. The valid-cell mask restricts the hydraulic outputs. Wet probability is obtained from the separate wet head; it is not the same quantity as the activity gate. A validation-selected extent rule converts wet scores into a displayed binary mask.

8. Direct forecast versus recursive transition
For a requested lead, this model makes one application from the observed origin to the target. It does not feed an intermediate predicted hydraulic state back into itself 48 times. That removes recursive state feedback from the formulation. It does not guarantee lower error, physical conservation or accurate long-lead forecasting. A separate model application is needed for each requested lead if producing all 48 forecasts.

9. Optional graph branch now included on this unified slide
Read the purple branch from left to right. The graph is not an encoder before the whole U-Net. It sits after the raster encoder and changes the bottleneck conditioning. Pool the 20 × 20 encoder features and collect four boundaries from the early 80 × 80 feature map. Combine those summaries with the same base conditioning c to form one node vector z⁰ per tile. Neighboring context tiles follow the same shared encoding path. The diagram shows one representative tile, not all neighbors in parallel.

Message passing produces zᴷ by computing source/destination messages, averaging incoming messages and applying residual node updates. The graph returns the projected feature change P(zᴷ − z⁰). Add this to the base conditioning to get c′. This graph-conditioned vector is what is concatenated into the 10 × 10 bottleneck. The raster features still feed the bottleneck directly, and the U-Net skip connections and decoder remain. The graph does not replace the halo or decode a raster on its own. P starts at zero, so the additional context initially contributes nothing.

The upper gray connection carries base conditioning into initial node features. The purple upward connection combines encoder pooled features and early boundary summaries. The purple output reaches the graph residual conditioning box, not the early halo addition junction. The output of that box, c′, enters the bottleneck. These are the three graph connections to emphasize when presenting the diagram.

In graph-disabled mode, the original c conditions the bottleneck directly. In graph-enabled mode, c′ takes its place. Local forcing and the optional extra discharge-correction gate are omitted. The local graph is implemented, but hydraulic routing edges, HUC12 hierarchy and conservative flux updates remain separate extensions.

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


Additional graph walkthrough and sources
Tile graph architecture — presenter notes

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
