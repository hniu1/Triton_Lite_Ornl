# TRITON U/V Component Semantics Audit

## Conclusion

The archived D030 TRITON executable source proves that the binary `U_*.dat`
and `V_*.dat` outputs used by this project are the conserved `HU` and `HV`
fields. They are depth-integrated momentum, commonly called unit discharge,
with units of `m²/s`. They are not velocity components in `m/s`.

No numerical target conversion is required. The preprocessing pipeline copied
the binary values without scaling; it only assigned incorrect velocity names
and unit attributes.

## Authoritative source evidence

The original simulation archive is:

```text
/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/
4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/
ACCESS_RegCM_baseline_flood_3hr/D030.zip
```

Its archived `D030/src/cppflood.cu` contains three decisive operations:

1. The active matrices are assigned as `g_HUa = HU` and `g_HVa = HV`.
2. Binary output `U` selects `g_HUa` directly, while `V` selects `g_HVa`.
3. The flux solver obtains velocity by dividing the stored fields by depth:
   `u = hu/h` and `v = hv/h`.

Therefore:

```text
stored U = h × u
stored V = h × v
units = m²/s
```

## Corrections made

- Future raw-to-netCDF conversion retains the legacy variable names for file
  compatibility but assigns unit-discharge long names, `m2 s-1` units, and an
  explicit `triton_component_semantics=unit_discharge` attribute.
- Future dynamic manifests default to `unit_discharge` and preserve the legacy
  source-unit string separately.
- The active manifest metadata now identifies the correct semantics. Existing
  numerical netCDF arrays and completed checkpoints were not modified.
- New transition whole-domain plots label the fields as unit discharge and use
  `m²/s`.

## Interpretation of existing results

All reported component RMSE values remain numerically valid, but their units
must be read as `m²/s`, not `m/s`. Terms previously described as speed loss or
speed RMSE are more precisely component-magnitude losses or unit-discharge
magnitude metrics. Directional comparisons remain meaningful where magnitude
is nonzero.

Velocity can be derived for wet cells as `u=HU/h` and `v=HV/h`, with an
explicit minimum-depth safeguard. That derived diagnostic should be added
separately; the model should continue predicting the conserved HU/HV state for
better compatibility with shallow-water dynamics and conservation losses.

