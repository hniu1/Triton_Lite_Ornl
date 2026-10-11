# ChatGPT source brief: TRITON Lite Monday presentation

Meeting: Monday, October 12, 2026. Prepared: October 07, 2026, 04:36 PM Eastern.

## Prompt to paste with this file

Build an editable scientific presentation for collaborators from this source brief and EXPERIMENT_HISTORY.md. Use clear architecture diagrams and a readable explanation of halo. Include the many earlier attempts, distinguishing completed legacy experiments, diagnostic prototypes, and proposed work. Use 17 main slides plus the detailed history appendix, or shorten to 12 main slides while retaining the appendix. Include speaker notes and source attribution. Treat the evidence rules below as authoritative over obsolete performance claims in older documents. Do not invent corrected-data results. Leave editable placeholders for new lead curves, held-out maps, and runtime. Explain the data repair candidly without making the whole deck a debugging report.

## Current scientific status — authoritative

Two hydraulic conversion defects were confirmed on October 6: alphabetic snapshot ordering assigned wrong regular half-hour timestamps; negative signed discharge values were discarded. Consequently old V4/V5 and other legacy timestamp/transition results do not validate the intended future-forecast task. Do not present old improvement percentages, metric tables or architecture rankings as valid forecast evidence. Specifically exclude the old 11.8% halo depth-gain claim. Historical diagnoses motivated experiments; revisit them on repaired targets. Earlier peak-depth modeling is historical context, with no new assertion here about whether peak-depth products are affected.

Original simulation archives are intact. The converter now sorts numeric counters and preserves signed HU/HV. Independent forcing alignment passed for all 40 events, exactly matching original interpolated hydrographs on 0.5–240h. Calendar epoch interpretation is not verified. Archive forcing comments use cfs; physical forcing-unit handling needs separate auditing before conservation losses.

Hydraulic rebuild status is a dated snapshot, not live telemetry: 12/40 validated at October 07, 2026, 04:36 PM Eastern. Completed events: D001, D002, D003, D004, D005, D006, D007, D008, D010, D012, D013, D014. Each event has 108 sampled raw tile/time/component comparisons, covering values, masks, signed components, timing, coordinates and units. This is not all-pixel verification. Initial array 186578 produced eight complete events. Recovery 187031 handles the remainder after environment/import repairs. Training on corrected data and fresh performance comparisons are pending.

## Task and data

One Conasauga domain,40 TRITON events, 240 hours/event, 480 half-hour hydraulic states/event. Forcing is a 480 × 300 tensor of source locations, not necessarily observed gauges. Approximately 6,900 tiles and 34 HUC12 units. Central tile 80 × 80 cells at approximately 10 m resolution (actual spacing about 9.506 m). Six terrain layers plus mask and tile attributes.

At forecast origin t, provide h,HU,HV and one older hydraulic state; terrain; tile/time attributes; requested lead; forcing supplied through t+lead. Predict h,wet probability,HU,HV at t+lead. Leads 1–48 half-hour steps (0.5–24h). No future hydraulic state enters inputs. Known future forcing is an assumption, not a weather prediction performed by this model.

## Architecture and halo — precise implementation

Direct V5-A uses a causal forcing encoder plus tile/time/lead embeddings, central terrain/state CNNs, history adapter, conditioned U-Net bottleneck, skip connections, residual depth/discharge heads and activity gate. It predicts the requested target in one pass rather than recursively feeding predicted states through 48 updates. That design avoids recursive feedback; it does not prove reduced error.

V5-B adds an 8-cell hydraulic border: 96 × 96 observed current/history input around an 80 × 80 central tile. A separate halo CNN encodes hydraulic features; its central 80 × 80 crop is added to the first central U-Net feature map. It is not an enlarged terrain U-Net, not a watershed graph, and not future-target information. Only central output is scored. Explain with a colored outer border and central grid. Neighborhood benefit must be reestablished on corrected data. Approximately 76 m border at actual spacing.

HU/HV are signed unit discharge in m²/s. Velocity is derived as HU/h,HV/h; shallow depth can amplify error. Previous alternatives tested learned discharge, component persistence, and velocity persistence with discharge reconstructed from predicted depth. These are distinct declared policies. HU = h*u and HV = h*v coupling imposes output consistency, not mass conservation.

## Many earlier attempts

Use the companion EXPERIMENT_HISTORY.md as the authoritative catalog. Sequence: peak V2/V3; timestamp baseline; label-aware/stratified sampling; Stage-1 Max; wet gating; timestamp V4; transition V1; continued V1; six-step multi-step V1; regime-aware V2; exact-local V3; history/gate V4a; adapter/gate V4b; persistence-policy diagnostics; direct V5-A; halo V5-B; direction/boundary auxiliary losses; local forcing V5-C; discharge-correction gate; three-seed/five-fold protocols; local graph V5-D; graph learning diagnostics. Hierarchical HUC12 graphs, conservative physics and Transformer/operator variants are proposed extensions, not completed results.

Initial Stage-2 four-way comparison: baseline, halo,halo+direction/boundary losses,halo+those losses+local forcing. Local forcing was confounded with coupling in that initial comparison. Later three-seed fold 0 comparison: baseline, halo,halo+discharge gate. One-seed five-fold halo evaluation also ran. All used legacy targets; do not infer corrected-data rankings.

Local graph prototype: one tile/node; actual shared-boundary adjacency in both directions ; 6,900 nodes and 27,262 directed edges; boundary summaries and conditioning feed message passing; residual context enters the U-Net bottleneck. Connected samples share event, origin, lead; score centers. Messages are learned features, not water fluxes. Smoke and tiny training-cohort diagnostics establish software execution only. Full graph benefit and hydraulic passability of edges are unproven. HUC12 hierarchy is not implemented.

## Corrected evaluation and Monday deliverables

Preserve complete-event splits: 25 training, 7 validation, 8 test per fold. Training-only normalization; validation-only checkpoint and extent threshold selection. Retrain from scratch; no contaminated checkpoint initialization. Compare persistence and learned forecasts on the same corrected origins at all 48 leads. Report wet/all-cell depth,extent precision/recall/F1/CSI,HU/HV,derived velocity,wetting/drying transitions and measured runtime. Sampling limits and future-forcing assumptions belong in notes.

Monday priority: methods and audit evidence; one fresh baseline versus persistence; halo comparison if time permits. Placeholders: lead curves, representative held-out maps, and actual training/inference time. If new results are unavailable, label the deck as methods plus repair progress. No operational, mass-conservation, unseen-region or graph-performance claim is established.

## Source documents to upload if more detail is needed

1. docs/stage2/TARGET_DATA_AUDIT.md — authoritative data defects, fixes and rebuild.
2. docs/stage2/V5B_EXPERIMENT_REPORT.md — halo, ablations and evaluation design. Numeric performance is obsolete evidence.
3. docs/stage2/V5D_GRAPH_COMMUNICATION.md — implemented local graph vs future hierarchy.
4. docs/stage2/STAGE2_V5_PLAN.md — forecast contract and roadmap. Earlier warm-start guidance is superseded by corrected-data training from scratch.
5. archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md and transition V1,V2,V3,V4a,V4b documents — earlier methods. Numeric claims are historical.
6. docs/project/MODEL_DESCRIPTION.md — earliest peak-map architecture, not the active direct model.

The ZIP bundle preserves these source paths. Old LEADERSHIP_BRIEFING.md is included only as historical context: use this brief for the current story.
