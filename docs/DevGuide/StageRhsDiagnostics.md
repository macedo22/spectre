# Targeted BBH timestep diagnostics

`EvolveGhBinaryBlackHole` has a top-level `StageRhsDiagnostics` option for
tracing the Dormand--Prince embedded error into the stage derivatives. It is
independent of the existing `Diagnostics` option under `ErrorControl`.

## Two runs with one executable

Build only the BBH executable in your existing build directory:

```sh
cmake --build /path/to/build --target EvolveGhBinaryBlackHole -j 16
```

For the first run, add this at the top level of the executable-options YAML
document (after `---`, alongside `Evolution`, not inside `ControlSystems`):

```yaml
StageRhsDiagnostics: None
```

Leave `Diagnostics: true` in your existing `ErrorControl` options. This writes
the old `TimeStepperErrorDiagnostics.txt` and performs no targeted stage
sampling. The new top-level key is required, including when disabled.

After locating the limiting element, component, and point, use the same
executable for a second run with, for example:

```yaml
StageRhsDiagnostics:
  StartTime: 50.0
  EndTime: 51.0
  Targets:
    - Element: "[B6,(L1I0,L0I0,L1I1)]"
      Components: [Pi_zt]
      GridPoints: [1539, 1619]
```

The indices above came from the older `[10,17,10]` mesh. Select the actual
indices and element IDs from the first run. Component spelling must match the
old diagnostics exactly (`Pi_zt`, for example). Each target samples every
listed component at every listed point; add another target for another element.
Use one target entry per element. An element ID absent from the evolving grid
will produce no samples. Unknown components and out-of-range point indices in
matched elements cause an error when sampling begins.

This implementation requires `LocalTimeStepping: Off`, `DormandPrince5`, and
an `ErrorControl` requesting embedded estimates. The old `Diagnostics` can be
either `true` or `false` during the targeted run. Leave tolerances and all
evolution settings the same between comparisons.

The interval selects **whole steps by their start time**:
`StartTime <= step_start < EndTime`. All seven stages of each selected step
are captured, even if its last stage lies beyond `EndTime`. The interval does
not change the evolution timestep or stop the simulation. One or a few M of
output during the timestep plateau should be enough for the first inspection.

Use separate run directories: both text outputs append to existing files.
The diagnostic state and options are serialized with each element. Treat these
as two fresh runs with different input options; editing YAML does not change
the settings already stored in a checkpoint. Checkpoints from a binary without
the new state are not compatible with this binary.

## Output

### Optional volume visualization of pointwise and element errors

`EvolveGhBinaryBlackHole` also exposes these fields through `ObserveFields`:

- **`PointwiseNormalizedTimeStepperError`**: the normalized embedded error,
  maximized over independent evolved tensor components **at each grid point**.
  Use this to look for angular structure, boundary concentrations, or other
  patterns within elements.
- `NormalizedTimeStepperError`: the latest stored normalized error used by
  error control, maximized over all grid points and all independent evolved
  tensor components in the element.
- `TimeStepperErrorStepStart`: start time of the step that produced that error.
- `TimeStepperErrorStepSize`: size of that step.

The last three fields are element-level scalars repeated at every point of the
element's full mesh. `PointwiseNormalizedTimeStepperError` varies within each
element. For a component `c` and point `p`, it reconstructs the signed embedded
difference from the completed Dormand--Prince stage derivatives:

```text
delta[c,p] = dt * sum_i((b5[i] - b4[i]) * k[i,c,p])
E[p] = max_c abs(delta[c,p]) /
    (atol + rtol * max(abs(u_start[c,p]), abs(u_start[c,p] + delta[c,p])))
```

The normalization and summation ordering match the production calculation.
The maximum of `E[p]` within an element should reproduce
`NormalizedTimeStepperError` up to floating-point roundoff. The component
providing the maximum can differ from point to point.

Pointwise output currently requires **DormandPrince5** and a completed-step
observation under `EventsAndTriggersAtSlabs` or `EventsAndTriggersAtSteps`.
It uses the retained RK history, so it does work **only when requested for an
observation**, with no new per-step capture or checkpoint state. Do not use it
with dense-output/substep observations or after a mesh-changing event at the
same observation time. Missing completed history produces a descriptive error.
Use a fixed mesh for comparisons to the stored production estimate.

All four fields work with
`ErrorControl: Diagnostics: false` and `StageRhsDiagnostics: None`, as long as
error control is enabled and computing estimates. Both existing text outputs
remain independently available.

Add the field names to an existing `ObserveFields: VariablesToObserve` list,
or add this entry under your existing `EventsAndTriggersAtSlabs`:

```yaml
  - Trigger:
      Slabs:
        EvenlySpaced:
          Interval: 500
          Offset: 0
    Events:
      - ObserveFields:
          SubfileName: TimeStepperError
          VariablesToObserve:
            - PointwiseNormalizedTimeStepperError
            - NormalizedTimeStepperError
            - TimeStepperErrorStepStart
            - TimeStepperErrorStepSize
          InterpolateToMesh: None
          ProjectToMesh: None
          CoordinatesFloatingPointType: Double
          FloatingPointTypes: [Double]
          BlocksToObserve: All
```

The interval is in slabs, not simulation time. With `VolumeFileName: BbhVolume`,
this writes `TimeStepperError.vol` in `BbhVolume*.h5`. Generate visualization
metadata with:

```sh
spectre generate-xdmf --subfile-name TimeStepperError \
  --output TimeStepperError BbhVolume*.h5
```

Open the XDMF file in ParaView and color by
`PointwiseNormalizedTimeStepperError` for the requested spatial patterns, or
`NormalizedTimeStepperError` for the element maxima.
Use a logarithmic color scale to distinguish small errors, and a fixed color
range when comparing times. Avoid merging points across element boundaries,
since the errors can be discontinuous there. Keep `InterpolateToMesh: None`
and `ProjectToMesh: None` to save the values on the original grid points;
ParaView still interpolates colors between points when displaying surfaces.

At a slab boundary in a forward GTS run, the estimate normally belongs to the
step just completed, rather than the step about to begin. Match text records
using `TimeStepperErrorStepStart` (the text's `error_step_start`), not the
volume observation time. No estimate exists initially, so these fields are
NaN at the initial observation; NaN means unavailable, not zero error. The
pointwise field requires the completed-step history described above; the
element-level fields simply report the latest stored estimate. An error near
one is near the specified tolerance scale, but the
PI controller's safety factor and previous error also affect its request.

Nothing is written unless the fields are requested. Pointwise output adds an
RK-error reconstruction over all evolved components at each observation, and
writes one scalar per grid point. The other fields only repeat stored scalars.
It does not enable the expensive per-step text diagnostics.

### Targeted stage/RHS text output

`TimeStepperStageDiagnostics.txt` contains one complete key/value record per
line, with two record types. Output from elements may be interleaved. Group by
`element`, `step_start`, `component`, and `point`, and sort stages by `stage`,
not by stage time (the last two stages have the same time).

### `StageRhs`: one record per stage and selected scalar

Stages are numbered **0 through 6**. The `k_*` entries are derivatives, not
derivatives multiplied by the timestep.

| Column | Meaning |
| --- | --- |
| `dt`, `stage_time` | Actual step size and stage evaluation time |
| `component_index`, `point` | Independent component in the full Variables and flattened point index |
| `nr`, `ntheta`, `nz` | Logical mesh extents, in the cylindrical runs' logical order |
| `value` | Stage field value before adding boundary corrections and updating the fields |
| `x`, `y`, `z` | Inertial coordinates of the sampled point |
| `vx`, `vy`, `vz` | Inertial components of the grid velocity (zero for a static map) |
| `k_total` | Complete derivative from the history actually used by the stepper |
| `k_volume` | Volume derivative captured before external and internal boundary corrections |
| `k_mesh` | Explicit moving-grid term, `v_grid^j partial_j u`, using the same spatial derivatives as the volume calculation |
| `k_inertial_volume` | `k_volume - k_mesh`: GH volume remainder |
| `k_boundary` | `k_total - k_volume`: combined boundary contribution |
| `error_weight` | Fifth-order weight minus fourth-order weight, read from the stepper's tableau |
| `error_contribution` | `error_weight * dt * k_total` |

Boundary contributions include all faces, mortars, and any external boundary
condition contribution. They are **not separated by face**. The remainder
called `k_inertial_volume` still depends on the maps through the spatial
derivatives and other inputs; a small explicit `k_mesh` does not exclude an
indirect map/control-system effect. No evolution term, filter cadence, or
stepper coefficient is changed by this diagnostic.

### `StageError`: one record per completed step and selected scalar

For each contribution, apply the same signed embedded-error weights:

```text
delta_X = dt * sum_i((b5_i - b4_i) * k_X_i)
```

| Column | Meaning |
| --- | --- |
| `delta_history` | Signed raw embedded error reconstructed directly from the complete RK history, with the stepper's arithmetic ordering |
| `delta_sampled` | The same sum using the separately captured stage totals |
| `delta_inertial_volume`, `delta_mesh`, `delta_boundary` | Signed contributions to that raw error |
| `history_residual` | `delta_sampled - delta_history` |
| `split_residual` | Sum of the three contributions minus `delta_history` |
| `start_value`, `end_value` | Step-start and fifth-order endpoint fields, with endpoint recorded before the subsequent volume filter |
| `normalization` | The actual absolute/relative tolerance formula at this point |
| `normalized_error` | `abs(delta_history) / normalization` at this selected scalar |
| `element_normalized_error` | The production stepper's maximum normalized error over the entire element |

Check the two residuals before interpreting the split. `history_residual`
should vanish up to roundoff; `split_residual` can reflect additional rounding
from subtracting and separately summing the contributions. Compare it with the
raw embedded error itself. If comparable, the attribution is not resolved.
The `delta_*` values can cancel strongly: do not sum their absolute values or
identify a cause merely from the largest unweighted stage derivative.

The scalar `normalized_error` should reproduce `element_normalized_error` when
the selected point/component is the element's maximum. Otherwise it should be
no larger, apart from roundoff. Match the old diagnostic using
`error_step_start`, not its later `evaluation_time`.

## Implementation and scope

The volume hook uses the existing spatial derivatives and stores only selected
scalar entries. The output action runs after `UpdateU`, before history cleanup
and volume filtering. It reads the same history used for the embedded estimate.
With `None`, the hooks return without scans or output. With settings enabled,
nonmatching elements do no scalar sampling. Samples migrate/checkpoint with the
element; AMR preserves the settings and clears samples. Use fixed meshes for
point-by-point comparisons, since a numerical point index does not describe a
fixed physical point after a resolution change.

This is an attribution along the *actual coupled trajectory*. A boundary
contribution also changes later stage fields and hence later volume terms.
The split helps select the next dependency to investigate; it does not isolate
an independent causal response as a controlled rerun would.
