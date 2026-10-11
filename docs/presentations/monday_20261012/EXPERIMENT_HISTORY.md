# TRITON Lite experiment history

This catalog groups methods and protocols, not individual jobs. Proposed items did not run.

> Evidence rule: timestamp/transition/direct forecast metrics on legacy hydraulic targets cannot support forecasting or architecture conclusions. Earlier observations motivated the trials but need corrected-data reassessment. Peak-map methods are included as historical methods without making new claims about their data validity.

| Family | Attempt | Change / hypothesis | Status | Source |
|---|---|---|---|---|
| Early maps | Blockwise peak V2 | Depth and wet/dry heads; event + terrain inputs | Historical method | `docs/project/MODEL_STATUS_AND_CHALLENGES_SUMMARY.md` |
| Early maps | Blockwise peak V3 | Raster features, depth-weighted Huber, wet classification | Historical method | `docs/project/MODEL_DESCRIPTION.md` |
| Timestamp | Timestamp baseline | Predict requested state from forcing, time and terrain | Legacy-data run | `archive/stage1/docs/STAGE1_TIMESTAMP_SURROGATE.md` |
| Timestamp | Label-aware / stratified sampling | Represent dry, boundary, wet and deep patches | Legacy-data run | `archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md` |
| Timestamp | Stage-1 Max | Dense index, whole-batch balancing, hybrid depth loss | Legacy-data run | `archive/stage1/docs/STAGE1_MAX_PERFORMANCE.md` |
| Timestamp | Calibrated wet gating | Select an extent rule and gate displayed depth | Legacy-data diagnostic | `archive/stage1/docs/PROGRESS_SINCE_GATED_BASELINE.md` |
| Timestamp | Timestamp V4 fine-tuning | Strict quotas, depth/wet coupling, flow-aware losses | Legacy-data run | `archive/stage1/docs/STAGE1_V4_FINE_TUNING.md` |
| Transitions | Transition V1 | Observed state + forcing -> residual next-state update | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V1.md` |
| Transitions | Continued V1 | Lower-rate training of the one-step reference | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V1.md` |
| Transitions | Multi-step V1 | Six-step scheduled exposure to predicted states | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V1.md` |
| Transitions | Regime-aware V2 | Filling/draining/stable sampling + delta/storage objectives | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V2.md` |
| Transitions | Exact-local V3 | Paired M6 labels, 12-step rollout, velocity MSE | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V3.md` |
| Transitions | History + gate V4a | Two hydraulic states; spatial gate on residual changes | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V4A.md` |
| Transitions | Adapter + gate V4b | Separate history adapter; faster training of new modules | Legacy-data run | `archive/stage1/docs/STAGE1_TRANSITION_V4B.md` |
| Transitions | Component-policy diagnostics | Compare learned flow, persistent HU/HV and persistent u/v | Legacy-data diagnostic | `docs/stage2/STAGE2_V5_PLAN.md` |
| Direct / context | Direct V5-A | Requested lead in one pass; no recursive state rollout | Legacy-data run | `docs/stage2/STAGE2_V5_PLAN.md` |
| Direct / context | Hydraulic halo V5-B | 8-cell neighboring state/history; central output | Legacy-data run | `docs/stage2/V5B_EXPERIMENT_REPORT.md` |
| Direct / context | Direction / boundary losses | Auxiliary velocity-direction and boundary-discharge terms | Legacy-data ablation | `docs/stage2/V5B_EXPERIMENT_REPORT.md` |
| Direct / context | Local forcing V5-C | Inverse-distance signal from 8 nearby forcing locations | Tested with coupling | `docs/stage2/STAGE2_V5_PLAN.md` |
| Direct / context | Discharge-correction gate | Low-initialized gate on learned HU/HV corrections | Legacy-data ablation | `docs/stage2/V5B_EXPERIMENT_REPORT.md` |
| Direct / context | 3-seed / 5-fold protocol | Optimization variability and complete-event holdouts | Legacy-data evaluation | `docs/stage2/V5B_EXPERIMENT_REPORT.md` |
| Graph / next | Local graph V5-D | Boundary summaries + adjacency messages + bottleneck context | Prototype + GPU smoke | `docs/stage2/V5D_GRAPH_COMMUNICATION.md` |
| Graph / next | Graph learning checks | Tiny training cohort; baseline / self-edge / neighbor arms | Diagnostic; skill unproven | `docs/stage2/V5D_GRAPH_COMMUNICATION.md` |
| Graph / next | HUC12 hierarchy | Coarse watershed nodes for wider routing context | Proposed; not implemented | `docs/stage2/V5D_GRAPH_COMMUNICATION.md` |
| Graph / next | Conservative physics V5-E | Storage and shared-boundary flux constraints | Proposed; not established | `docs/stage2/V5D_GRAPH_COMMUNICATION.md` |
| Graph / next | Transformer / operator V5-F | Controlled backbone comparison after fixing inputs | Proposed; not implemented | `docs/stage2/STAGE2_V5_PLAN.md` |

Version labels are reused: peak V3, timestamp V4, transition V3/V4a/V4b, and Stage-2 V5 name distinct model families. Local forcing was tested with auxiliary coupling losses; a clean isolated local-forcing comparison remains pending. Multi-seed testing covered fold0 only; five-fold halo testing used one seed. Local graph implementation is distinct from the unimplemented HUC12 hierarchy. Coupling outputs is distinct from establishing mass conservation.
