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


Message-passing equations — detailed presenter notes

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
