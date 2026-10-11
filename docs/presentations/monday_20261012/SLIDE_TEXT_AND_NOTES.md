# TRITON Lite — slide text and speaker notes

Prepared: October 07, 2026, 04:36 PM Eastern. Meeting: October 12, 2026.

Legacy timestamp, transition and direct-forecast numbers do not establish corrected-data forecasting skill.

## 01. Flood forecasting with hydraulic context

TRITON Lite | Methods, experiments, and the corrected-data restart

From independent tiles to / state-aware, direct forecasts

01 Hydraulic history

Use the observed state and / its recent change.

02 Spatial context

Add a halo; test graph / communication.

03 Trustworthy targets

Repair chronology and / signed discharge.

Methods are ready to explain. Corrected-data model comparisons are still pending.


Speaker notes: Prepared for Monday, October 12, 2026. This deck explains the work already implemented and the experimental sequence. It does not claim validated forecasting gains from the legacy dataset. Replace the status snapshot and result placeholders before the meeting.

Sources: docs/stage2/TARGET_DATA_AUDIT.md, docs/stage2/V5B_EXPERIMENT_REPORT.md, docs/stage2/V5D_GRAPH_COMMUNICATION.md

## 02. What the model is asked to predict

A requested future hydraulic map, given an observed state and known forcing

Inputs at forecast / origin

Current h, HU, HV / Previous hydraulic state / Terrain, mask and tile / attributes / Forcing through the target / time

Direct model

Requested lead: / 0.5–24 hours / One model / application

Outputs at target time

Depth h / Wet probability / flood extent / Signed unit discharges HU, HV / Derived velocity as a / diagnostic

Prediction unit: one central 80 × 80 tile. Forecast lead: 1–48 half-hour steps.


Speaker notes: The direct model uses observed hydraulic states at the origin and one preceding half-hour state. Forcing is supplied through the requested target time, but never after it. Known future forcing is an input assumption; this is not a weather forecasting model.

Sources: docs/stage2/STAGE2_V5_PLAN.md, stage2/model.py

## 03. The available simulation archive

One Conasauga domain supports controlled development experiments

40 events

D001–D040 / 240 simulated hours each

480 states / event

Half-hour snapshots / Depth plus signed / discharge

~6,900 tiles

80 × 80 cells per tile / 34 HUC12 subwatersheds

Forcing tensor: 480 time steps × 300 source locations / Terrain inputs: DEM, flow accumulation, stream proximity, slope and relative / elevation

Current scope: in-domain development. New events and regions are needed for final validation.


Speaker notes: The raster spacing is approximately 9.506 m; the project calls this 10 m resolution. An 80-cell tile is roughly 0.76 km across, conventionally described as approximately 0.8 km. Hydrograph comments use cfs; physical forcing-unit handling remains to be audited before conservation losses. Do not describe these 300 locations as observed gauges without checking provenance.

Sources: README.md, docs/stage2/V5B_EXPERIMENT_REPORT.md, docs/stage2/TARGET_DATA_AUDIT.md

## 04. Our method evolved through several questions

The project includes many controlled attempts, not one architecture switch

Peak maps

Can event + terrain / predict maximum depth?

Timestamp states

Can forcing + time / predict each state?

State transitions

Does observed history / help forecast change?

History + gates

Can updates respect / stable conditions?

Direct + halo

Can we predict a lead / with nearby context?

Graph + audit

Can communication help? / Are targets correct?

The audit changed the immediate priority: repair targets, then repeat the model comparisons.


Speaker notes: This is a conceptual sequence, not a date timeline. Version names are reused: timestamp V4 and transition V4a/V4b are different families. The appendix lists 26 method/protocol entries, including proposed work; that count is not a count of completed runs. Historical diagnoses motivated experiments but must be revisited after target repair.

Sources: docs/project/MODEL_DESCRIPTION.md, archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md, docs/stage2/STAGE2_V5_PLAN.md, docs/stage2/V5D_GRAPH_COMMUNICATION.md

## 05. Early attempts: maps, sampling and supervision

We explored how to represent rare wet cells and flood boundaries

Blockwise peak V2 / V3

Event + tile + static / rasters / Depth and wet/dry heads / Depth-weighted Huber / Extent-threshold / diagnostics

Timestamp / Stage-1 / Max

Forcing + requested time / Label-aware sampling / Dense candidate index / Whole-batch wet/deep / balance

Gated / timestamp V4

Validation-calibrated extent / Strict category quotas / Depth × wet-probability / coupling / Flow-aware supervision

These remain reusable methods; their old metric rankings do not establish forecasting skill.


Speaker notes: The earliest peak-depth model uses an event embedding and block features with a spatial decoder. Timestamp models add a requested time and causal forcing. Subsequent attempts included label-aware batches, a denser dynamic index, strict category quotas, physical/log-depth losses, wet BCE/Dice, and calibrated wet gating. The target chronology issue directly invalidates timestamp forecast interpretation; this deck makes no new claim about whether peak-depth products are affected.

Sources: docs/project/MODEL_STATUS_AND_CHALLENGES_SUMMARY.md, docs/project/MODEL_DESCRIPTION.md, archive/stage1/docs/STAGE1_TIMESTAMP_SURROGATE.md, archive/stage1/docs/STAGE1_MAX_PERFORMANCE.md, archive/stage1/docs/STAGE1_V4_FINE_TUNING.md, archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md

## 06. Transition attempts: learn the state change

We explored temporal history, rollout exposure, and stable-update safeguards

Attempt

What changed

Question

V1 + continuation

Residual update from observed state

Can history help predict change?

Multi-step V1

Six-step predicted-state exposure

Can training cover rollout behavior?

Regime-aware V2

Filling, draining and stable objectives

Can rare changes be represented?

Exact-local V3

Paired labels; 12-step training

Are regimes and lead exposure reliable?

History V4a / adapter / V4b

Two states, activity gate, separate / adapter

Can stable updates and history be learned?

Retain the implementations. Reassess the historical diagnoses on corrected data.


Speaker notes: Transition V1 starts with one-step observed-state training, then continuation and six-step scheduled predicted-state exposure. V2 changes regimes and objectives; V3 uses exact paired local statistics and twelve-step exposure. V4a adds a second state and activity gate; V4b separates the history adapter and trains new modules faster. Those were tests on legacy targets, not proof that any feature improves the corrected task.

Sources: archive/stage1/docs/STAGE1_TRANSITION_V1.md, archive/stage1/docs/STAGE1_TRANSITION_V2.md, archive/stage1/docs/STAGE1_TRANSITION_V3.md, archive/stage1/docs/STAGE1_TRANSITION_V4A.md, archive/stage1/docs/STAGE1_TRANSITION_V4B.md

## 07. Direct forecasting removes recursive state feedback

V5-A predicts the requested lead from the observed origin

Recursive transition forecast

Observed S(t)

Predicted S(t+1)

Predicted S(t+2)

… S(t+48)

Each predicted state becomes part of the next model input.

Direct, lead-conditioned forecast

Observed S(t) + history + lead

One pass

Predicted S(t + lead)

A design change that must be tested, rather than a claim of improved accuracy.


Speaker notes: The direct model still predicts a residual from the observed origin. Direct prediction avoids the mechanism of feeding intermediate predicted states back into the network. It does not guarantee lower forecast error. Both models must be evaluated on the same corrected origins and targets.

Sources: docs/stage2/STAGE2_V5_PLAN.md, stage2/model.py, archive/stage1/docs/STAGE1_TRANSITION_V4B.md

## 08. The direct U-Net combines space, history and forcing

Terrain and hydraulic features meet lead-conditioned information

Spatial inputs

Terrain + mask / Current state + history

Conditioning inputs

Causal forcing encoder / Tile, time and lead / embeddings

Tile CNN encoder

Central spatial features / Downsample 80 → 40 → 20

Conditioned / bottleneck

Spatial features + conditioning / U-Net decoder + skip features

Forecast heads

Δh, ΔHU, ΔHV; activity gate / Wet probability

Optional context

Halo: early spatial features / Graph: bottleneck messages

Future hydraulic states are targets only. They never enter forecast inputs.


Speaker notes: The active implementation uses a causal forcing encoder, tile/time/lead embeddings, state/history fusion, a conditioned U-Net bottleneck, skip connections, and residual hydraulic heads. The diagram groups modules for readability. Halo features are added to the first central spatial feature map; graph context enters at the bottleneck. New corrected-data runs should train from scratch, not initialize from contaminated checkpoints.

Sources: stage2/model.py, README.md

## 09. Halo: neighboring hydraulics around a central tile

V5-B adds an 8-cell border while keeping an 80 × 80 output

Central 80 × 80

96 × 96 hydraulic input

Observe a wider neighborhood

The border supplies nearby depth and signed / discharge at the forecast origin.

Score only the central prediction

Halo encoder → central feature crop → central U-Net / features. Output remains 80 × 80.


Speaker notes: Implementation detail matters: this is a separate hydraulic halo encoder, not an enlarged terrain U-Net. A 96x96 current/history hydraulic patch is encoded; its central 80x80 feature crop is added to the central tile feature map. Only central outputs are supervised. Two history states produce current channels plus changes from the older state. Eight cells correspond to roughly 76 m at the actual grid spacing. Whether halo improves corrected-data accuracy is pending.

Sources: docs/stage2/V5B_EXPERIMENT_REPORT.md, stage2/model.py

## 10. The Stage-2 comparisons we already attempted

Historical experiment designs are reusable; performance conclusions need reruns

Experiment family

Variants / protocol

Evidence status

Initial four-way ablation

Direct; halo; + auxiliary losses; + local forcing

Ran on legacy targets

Three-seed fold-0 study

Baseline; halo; halo + discharge gate

Ran on legacy targets

Five-fold event / evaluation

Halo candidate; each event tested once

Ran on legacy targets

Graph learning / diagnostic

U-Net; local capacity control; adjacency graph

Tiny training cohort only

Do not use the old “11.8% depth improvement” as a corrected-data claim.


Speaker notes: The first nested ablation compared direct, halo, halo+direction/boundary losses, and halo+those losses+local forcing. Local forcing was tested together with coupling, so it was not an isolated causal estimate of local forcing. The later fold-0 study compared baseline, halo, and halo+discharge gate at three seeds. Halo alone received five-fold evaluation at one seed. None of the numerical rankings establishes corrected-data forecasting skill.

Sources: docs/stage2/V5B_EXPERIMENT_REPORT.md, docs/stage2/STAGE2_V5_PLAN.md

## 11. Depth, discharge and velocity need separate checks

Archived U/V outputs represent signed unit discharge, HU/HV

HU = h · u HV = h · v u = HU / h v = HV / h

Learned HU/HV

Predict signed flow / corrections with depth. / Inspect derived velocity in / shallow and deep water.

Component / persistence

Copy origin HU/HV while / forecasting depth. Declare / this policy explicitly.

Velocity persistence

Copy origin u/v; / reconstruct HU/HV using / forecast depth. Evaluate / outputs jointly.

Coupled outputs do not establish mass conservation. Signed flow direction must be preserved.


Speaker notes: The source-code semantics audit establishes HU/HV as unit discharge with m²/s units. Derived velocity divides by depth, so shallow-depth errors can be amplified. The three policies shown are diagnostic alternatives, not a selected corrected-data forecast policy. Reconstructing HU=h*u and HV=h*v imposes output consistency, not conservation or validated forecast skill. The old semantics document predates discovery of negative-value clipping; its claim that values were copied unchanged is superseded by the target audit.

Sources: archive/stage1/docs/TRITON_COMPONENT_SEMANTICS_AUDIT.md, docs/stage2/TARGET_DATA_AUDIT.md, docs/stage2/STAGE2_V5_PLAN.md

## 12. Graph prototype: communicate learned tile features

V5-D retains the raster U-Net and adds adjacency context

North

West

East

South

Center

Message-passing pathway

Tile features + boundary summaries → graph messages → / residual bottleneck context

What has been checked

Topology, connected batches, GPU execution, and / tiny-cohort learning diagnostics.

Forecasting benefit is unproven. Graph messages are not conservative water exchange.


Speaker notes: Each node is one tile. Edges are actual shared raster boundaries in both directions: adjacency is geometry, not verified hydraulic passability. Node features combine local features, four boundary summaries, forcing and lead. Connected batches share event, origin and lead; losses score centers. Local graph communication is implemented. HUC12 hierarchy is a future extension. Messages are learned features, not fluxes.

Sources: docs/stage2/V5D_GRAPH_COMMUNICATION.md

## 13. A fair comparison holds out complete events

Use the same origins, leads, sampling and metrics for every candidate

25

TRAIN EVENTS

7

VALIDATION

8

TEST EVENTS

Baseline: persistence

Copy the origin state. Compare against the / same corrected targets and forecast origins.

Evaluation across leads

All 48 half-hour leads; depth, extent, signed / flow, velocity and wetting/drying transitions.

Train from scratch. No legacy checkpoint or test-event normalization enters the new comparison.


Speaker notes: The existing five-fold contract uses 25 training events, 7 validation events, and 8 test events per run. One historical seed covers all five folds, while three-seed robustness was fold 0 only. Preserve event membership after repair; fit normalization and sampling using training data. Select checkpoints and extent thresholds on validation only. Recompute persistence on corrected targets; do not reuse old persistence metrics. Original dense tests sampled 80 batches per fold, not every pixel in the domain.

Sources: docs/stage2/V5B_EXPERIMENT_REPORT.md, docs/stage2/STAGE2_V5_PLAN.md, docs/stage2/V5D_GRAPH_COMMUNICATION.md

## 14. The audit found two target-conversion defects

These affect both learning and the interpretation of forecast leads

01 Wrong snapshot chronology

Alphabetic sorting scrambled numeric / counters. /  / D029 label: 120 hours / Actual source state: 15 hours

02 Lost signed discharge

Negative HU/HV values were discarded. /  / The model lost flow direction in affected / cells.

Forcing audit: all 40 events passed. Hydraulic targets required rebuilding.


Speaker notes: Raw evidence: D029 origin index239 labeled120h matches H_30_00.dat, physically15h; the next labeled120.5h matches H_310_00.dat, physically155h. Alphabetic order, followed by assigned regular timestamps, created wrong pairs. Negative signed discharge values were discarded and later loader filling made them zero. Numeric counter sorting and signed-component preservation are implemented. Forcing interpolation independently passed across all40 events. Do not infer that source simulations are wrong; original archives remain intact.

Sources: docs/stage2/TARGET_DATA_AUDIT.md

## 15. Corrected targets are being rebuilt and checked

Status snapshot: October 07, 2026, 04:36 PM Eastern | update before presenting

Numeric time counters

Require all 480 sequential / H/U/V snapshots. Preserve / the relative half-hour time / axis.

Signed-flow validation

Compare sampled values / and masks directly with / raw archives. Check units / and coordinates.

Per-event completion

Final NetCDF + successful / validation report required. / Keep failures as partial / files.

12 / 40 events validated in this snapshot

Rebuild → sampling and normalization → fresh training → corrected persistence comparison.


Speaker notes: Snapshot from successful validation reports: 12/40 events: D001, D002, D003, D004, D005, D006, D007, D008, D010, D012, D013, D014. Original array186578 produced eight validated outputs. Recovery array187031 uses the intact triton environment; two tasks were requeued after optional training imports caused an OpenMP conflict. Conversion now avoids those imports. Validation is sampled,108 component/tile/time comparisons per event, not all-pixel certification. This does not mean corrected model training is complete.

Sources: docs/stage2/TARGET_DATA_AUDIT.md

## 16. Corrected-data results: fill after evaluation

Editable placeholders — no forecasting result is claimed on this slide

A Lead curves

[INSERT CORRECTED / RESULTS] /  / Wet-depth RMSE / Extent F1 / CSI / Model vs persistence

B Held-out maps

[INSERT CORRECTED / MAPS] /  / TRITON target / Model prediction / Error / missed flooding

C Runtime and flow

[INSERT MEASUREMENTS] /  / Training / inference time / HU/HV error / Shallow / deep velocity

Report event splits, evaluated samples, seed, checkpoint rule and output policy.


Speaker notes: Replace these boxes only after a fresh corrected-data run and held-out evaluation. Report model versus persistence on the same cohort. Suggested leads0.5,3,6,12,24h for a compact table, while keeping all48 leads in an appendix. Use representative held-out maps with consistent depth color scales. Show depth/extent and derived velocity separately. If results are unavailable, keep this as the evaluation plan or remove the slide.

Sources: docs/stage2/STAGE2_V5_PLAN.md, docs/stage2/TARGET_DATA_AUDIT.md

## 17. What we can responsibly bring to Monday

Explain the methods now; add fresh evidence as it becomes available

Ready to explain

Architecture and halo / Sequence of experiments / Graph prototype / Data bugs and repair / checks

Priority evidence

Corrected-data baseline / Persistence comparison / Lead curves and maps / Measured runtime

Next scientific / questions

Does halo help after / repair? / Can discharge stay stable? / Does graph add useful / context? / What generalizes to new / events?

Retain the method work. Rebuild the evidence for model selection on trustworthy targets.


Speaker notes: Priority is one trustworthy baseline, then a halo comparison if time permits. Training throughput on Riker has not been established by a production corrected-data benchmark; previous eight-hour job limits are not runtime evidence. Sunday should be reserved for evaluation and slide assembly. If training slips, present the repair and methods with labeled diagnostic evidence, not old performance claims. No new training is submitted by creating this presentation.

Sources: docs/stage2/TARGET_DATA_AUDIT.md, docs/stage2/STAGE2_V5_PLAN.md, docs/stage2/V5D_GRAPH_COMMUNICATION.md

## 18. Appendix: early maps and timestamp attempts

Method catalog; execution status is separate from scientific validation

Attempt

Change / hypothesis

Status

Blockwise peak V2

Depth and wet/dry heads; event + terrain inputs

Historical method

Blockwise peak V3

Raster features, depth-weighted Huber, wet classification

Historical method

Timestamp baseline

Predict requested state from forcing, time and terrain

Legacy-data run

Label-aware / stratified / sampling

Represent dry, boundary, wet and deep patches

Legacy-data run

Stage-1 Max

Dense index, whole-batch balancing, hybrid depth loss

Legacy-data run

Calibrated wet gating

Select an extent rule and gate displayed depth

Legacy-data diagnostic

Timestamp V4 fine-tuning

Strict quotas, depth/wet coupling, flow-aware losses

Legacy-data run

Version labels are reused across families. These entries are not a count of jobs or proven gains.


Speaker notes: Blockwise peak V2: Depth and wet/dry heads; event + terrain inputs. Status: Historical method. Source: docs/project/MODEL_STATUS_AND_CHALLENGES_SUMMARY.md
Blockwise peak V3: Raster features, depth-weighted Huber, wet classification. Status: Historical method. Source: docs/project/MODEL_DESCRIPTION.md
Timestamp baseline: Predict requested state from forcing, time and terrain. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TIMESTAMP_SURROGATE.md
Label-aware / stratified sampling: Represent dry, boundary, wet and deep patches. Status: Legacy-data run. Source: archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md
Stage-1 Max: Dense index, whole-batch balancing, hybrid depth loss. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_MAX_PERFORMANCE.md
Calibrated wet gating: Select an extent rule and gate displayed depth. Status: Legacy-data diagnostic. Source: archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md
Timestamp V4 fine-tuning: Strict quotas, depth/wet coupling, flow-aware losses. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_V4_FINE_TUNING.md

Sources: docs/project/MODEL_STATUS_AND_CHALLENGES_SUMMARY.md, docs/project/MODEL_DESCRIPTION.md, archive/stage1/docs/STAGE1_TIMESTAMP_SURROGATE.md, archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md, archive/stage1/docs/STAGE1_MAX_PERFORMANCE.md, archive/stage1/docs/STAGE1_V4_FINE_TUNING.md

## 19. Appendix: transition and history attempts

Method catalog; execution status is separate from scientific validation

Attempt

Change / hypothesis

Status

Transition V1

Observed state + forcing -> residual next-state update

Legacy-data run

Continued V1

Lower-rate training of the one-step reference

Legacy-data run

Multi-step V1

Six-step scheduled exposure to predicted states

Legacy-data run

Regime-aware V2

Filling/draining/stable sampling + delta/storage objectives

Legacy-data run

Exact-local V3

Paired M6 labels, 12-step rollout, velocity MSE

Legacy-data run

History + gate V4a

Two hydraulic states; spatial gate on residual changes

Legacy-data run

Adapter + gate V4b

Separate history adapter; faster training of new modules

Legacy-data run

Component-policy diagnostics

Compare learned flow, persistent HU/HV and persistent u/v

Legacy-data diagnostic

Version labels are reused across families. These entries are not a count of jobs or proven gains.


Speaker notes: Transition V1: Observed state + forcing -> residual next-state update. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V1.md
Continued V1: Lower-rate training of the one-step reference. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V1.md
Multi-step V1: Six-step scheduled exposure to predicted states. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V1.md
Regime-aware V2: Filling/draining/stable sampling + delta/storage objectives. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V2.md
Exact-local V3: Paired M6 labels, 12-step rollout, velocity MSE. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V3.md
History + gate V4a: Two hydraulic states; spatial gate on residual changes. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V4A.md
Adapter + gate V4b: Separate history adapter; faster training of new modules. Status: Legacy-data run. Source: archive/stage1/docs/STAGE1_TRANSITION_V4B.md
Component-policy diagnostics: Compare learned flow, persistent HU/HV and persistent u/v. Status: Legacy-data diagnostic. Source: docs/stage2/STAGE2_V5_PLAN.md

Sources: archive/stage1/docs/STAGE1_TRANSITION_V1.md, archive/stage1/docs/STAGE1_TRANSITION_V2.md, archive/stage1/docs/STAGE1_TRANSITION_V3.md, archive/stage1/docs/STAGE1_TRANSITION_V4A.md, archive/stage1/docs/STAGE1_TRANSITION_V4B.md, docs/stage2/STAGE2_V5_PLAN.md

## 20. Appendix: direct, halo and graph attempts

Method catalog; execution status is separate from scientific validation

Attempt

Change / hypothesis

Status

Direct V5-A

Requested lead in one pass; no recursive state rollout

Legacy-data run

Hydraulic halo V5-B

8-cell neighboring state/history; central output

Legacy-data run

Direction / boundary losses

Auxiliary velocity-direction and boundary-discharge terms

Legacy-data ablation

Local forcing V5-C

Inverse-distance signal from 8 nearby forcing locations

Tested with coupling

Discharge-correction gate

Low-initialized gate on learned HU/HV corrections

Legacy-data ablation

3-seed / 5-fold protocol

Optimization variability and complete-event holdouts

Legacy-data evaluation

Local graph V5-D

Boundary summaries + adjacency messages + bottleneck / context

Prototype + GPU smoke

Graph learning checks

Tiny training cohort; baseline / self-edge / neighbor arms

Diagnostic; skill unproven

Version labels are reused across families. These entries are not a count of jobs or proven gains.


Speaker notes: Direct V5-A: Requested lead in one pass; no recursive state rollout. Status: Legacy-data run. Source: docs/stage2/STAGE2_V5_PLAN.md
Hydraulic halo V5-B: 8-cell neighboring state/history; central output. Status: Legacy-data run. Source: docs/stage2/V5B_EXPERIMENT_REPORT.md
Direction / boundary losses: Auxiliary velocity-direction and boundary-discharge terms. Status: Legacy-data ablation. Source: docs/stage2/V5B_EXPERIMENT_REPORT.md
Local forcing V5-C: Inverse-distance signal from 8 nearby forcing locations. Status: Tested with coupling. Source: docs/stage2/STAGE2_V5_PLAN.md
Discharge-correction gate: Low-initialized gate on learned HU/HV corrections. Status: Legacy-data ablation. Source: docs/stage2/V5B_EXPERIMENT_REPORT.md
3-seed / 5-fold protocol: Optimization variability and complete-event holdouts. Status: Legacy-data evaluation. Source: docs/stage2/V5B_EXPERIMENT_REPORT.md
Local graph V5-D: Boundary summaries + adjacency messages + bottleneck context. Status: Prototype + GPU smoke. Source: docs/stage2/V5D_GRAPH_COMMUNICATION.md
Graph learning checks: Tiny training cohort; baseline / self-edge / neighbor arms. Status: Diagnostic; skill unproven. Source: docs/stage2/V5D_GRAPH_COMMUNICATION.md

Sources: docs/stage2/STAGE2_V5_PLAN.md, docs/stage2/V5B_EXPERIMENT_REPORT.md, docs/stage2/V5D_GRAPH_COMMUNICATION.md

## 21. Appendix: future work and source documents

Separate implemented prototypes from the proposed roadmap

Proposed extension

Purpose

Status

HUC12 hierarchy

Coarse watershed nodes for wider routing context

Proposed; not implemented

Conservative physics V5-E

Storage and shared-boundary flux constraints

Proposed; not established

Transformer / operator V5-F

Controlled backbone comparison after fixing inputs

Proposed; not implemented

Upload first: CHATGPT_SOURCE_BRIEF.md + EXPERIMENT_HISTORY.md

Optional full sources: target audit; V5-B report; V5-D graph plan; Stage-1 transition documents. / Historical numerical claims in those files are superseded by the audit.


Speaker notes: For a ChatGPT redesign, upload CHATGPT_SOURCE_BRIEF.md and EXPERIMENT_HISTORY.md from this folder first. Optional full documents are included in CHATGPT_SOURCE_BUNDLE.zip, preserving repository paths. Legacy source documents contain obsolete numeric claims; the source brief explicitly supersedes them. The current deck and preview use native editable PowerPoint shapes; they do not reproduce legacy quantitative charts.

Sources: docs/stage2/STAGE2_V5_PLAN.md, docs/stage2/V5D_GRAPH_COMMUNICATION.md, docs/stage2/TARGET_DATA_AUDIT.md, docs/stage2/V5B_EXPERIMENT_REPORT.md
