# BandTrace v0.1 model–sensor conformance preflight

> **Status:** iteratively specified and red-teamed during implementation on 2026-07-27; revision 29
> is frozen before any coherent release-validation run. Earlier live test snapshots are development
> evidence only. This is a software-conformance experiment, not a certificate, biological
> validation, deployment approval or safety claim. Trademark clearance is separate. The machine contract is
> `configs/product/bandtrace_v1.yaml`; the runtime policy ID is `bandtrace-0.1-r29`.

## 1. Product question

Given a numeric model executable, its declared training spectral contract, a target-sensor
contract and unlabeled target-sensor probes, can a local tool deterministically report:

1. whether the declared route reproduces the adapter-reported tap on deterministic challenges;
2. which target bands measurably affect the selected numeric output on those probes; and
3. whether the routed target response functions lie inside the model supplier's declared training
   support?

The three answers are orthogonal. Executing a model does not establish spectral support, and both
together do not establish biological transport.

## 2. Exact v0.1 scope

The bundle root contains a `bandtrace.yaml` that SHA-256 pins every manifest-declared file. It
declares:

- a model contract: model ID/version/artifact hash, required model-channel IDs/order, center
  wavelength and units, mandatory FWHM and optional full SRF, radiometric quantity, valid range, affine
  normalization, declared validated spectral support, an explicit non-empty
  `required_dependence_target_band_ids` set for which output dependence is required, and one
  numeric pre-decision output. It must explicitly declare whether wavelength and FWHM are each
  executable conditioning inputs;
- a target-sensor contract: sensor/model/serial, target-band IDs/order, center wavelengths,
  mandatory FWHM and optional SRFs, radiometric quantity, valid range, one per-band target-domain neutral value,
  calibration state and preprocessing version;
- at least 16 unlabeled finite probes in a non-pickled NPZ as `[N,B]` or `[N,B,H,W]`, with target
  band IDs and a manifest hash; labels and endpoints are forbidden;
- an explicit route matrix `W` of shape `[model_channels,target_bands]`; and
- an executable adapter declaration.

Version 0.1 accepts only `unitless_reflectance_factor`; radiance and raw-DN models are outside this
release. Any other quantity or a model/sensor quantity mismatch is an invalid bundle (exit 2).
Exact equality is required between model and sensor, but remains a supplier assertion:
BandTrace does not validate calibration
traceability, illumination/view geometry or reference-panel provenance.

The only route eligible for tap-agreement state X2 is selection/permutation or non-negative
row-normalized linear resampling, followed by declared per-channel affine scaling and a fixed
spatial operation. For rank-2 `none`, the exact equation is
`z_i=(sum_j W[i,j]*x_j-offset_i)/scale_i`, with finite strictly positive `scale_i`. For rank-4
`mean`, the semantic definition is the same per-cell route and affine transform followed by the
mean over `H,W`, but the **canonical binary64 evaluation order** first computes each target band's
float64 mean over `H,W`, then applies `W` and the affine transform. Linearity makes the two
expressions mathematically equivalent while this order prevents an `N*model_channels*H*W`
intermediate. An adapter's returned `[N,model_channels]` `pre_core` must match that canonical
arithmetic. `none` is valid only for `[N,B]`, and `mean` only for `[N,B,H,W]`. Opaque or
nonlinear spectral mixing may execute but cannot reach X2. Missing units, ambiguous ordering,
duplicate IDs or undeclared transforms fail closed.
For `selection_or_permutation`, each row has exactly one weight strictly greater than zero; no
small positive contribution is ignored as numerical zero. Because eligible `W` is non-negative,
the complete routed raw interval is exact for each model channel:
`lo_i=sum_j W[i,j]*sensor_lo_j`, `hi_i=sum_j W[i,j]*sensor_hi_j`. These bounds are evaluated as exact
rationals of the parsed binary64 inputs, with no permissive absolute tolerance. Both must lie inside
the exact parsed model `valid_range`; otherwise
`routed_domain_outside_model_valid_range` blocks X2/X3 without erasing independently valid S.
IDs are unique within the model-channel axis and within the target-band axis. Those two namespaces
are separate: the same text may legally name one model channel and one target band, and every
lookup and report field must remain keyed by role plus axis index rather than by text alone.
Route-matrix axes and probe columns may be submitted in any permutation when complete unique IDs
move with them; BandTrace canonicalizes these explicitly keyed orders and records the submitted
order diagnostically. An ID-tied permutation is not a `reordered_bands` fault. Positional adapters
that ignore those IDs are tested by C4 and fail.

Version 0.1's subprocess adapter is POSIX-only, and **all completed-audit output publication is
Linux-only** because the no-clobber commit requires `renameat2(RENAME_NOREPLACE)` with no unsafe
fallback. Windows process-group, pipe-limit and publication behavior is unsupported, and no
cross-platform byte identity is claimed. Version 0.1 accepts at most 512 bands, 256 probes, 262,144 spatial cells per
probe, a 256-MiB probe file and 256 MiB after preflighted float64 expansion so every whole-probe
invocation remains inside the same 256-MiB canary-input ceiling. A manifest may declare at most 32
files totaling at most 536,870,912 stat bytes; both budgets are checked before any declared
payload is loaded or hashed. Every YAML/JSON
manifest is size-bounded, parsed without object constructors, aliases or
duplicate keys, and limited to 32 nested mapping/array levels. YAML is restricted to the JSON data
model with string mapping keys; a parser recursion/depth failure is an invalid bundle, never an
uncaught crash. Paths are restricted to bundle-relative regular files whose SHA-256 is pinned;
traversal and symlinks are rejected. The loader opens the bundle root once with
`O_DIRECTORY|O_NOFOLLOW`, records its identity, and reads `bandtrace.yaml` from a descriptor opened
relative to that root. The manifest descriptor is held and revalidated through the completed parse.
Every declared path component is then opened descriptor-relative: intermediate
components require `O_DIRECTORY|O_NOFOLLOW`, and the final component requires a regular
`O_NOFOLLOW` file. The at-most-32 final descriptors are held from the aggregate stat preflight
through hashing and parsing, so no pathname is reopened across that boundary. The root path must
still resolve to the pinned directory after loading; a rename or replacement is an invalid bundle.
This prevents a concurrently swapped intermediate symlink from redirecting reads outside the
bundle. NPZ inputs are inspected before array loading for member count,
uncompressed size, compression ratio, names and object dtype; only ZIP `STORED` and `DEFLATED`
members are accepted. It never loads pickle/joblib or
arbitrary eager-model files.
Before execution, exact C2 shift-selection work must satisfy
`(probe_count-1)*probes.size <= 536,870,912` float-cell comparisons. A larger bundle is invalid
rather than silently sampling shifts. The complete frozen canary schedule must also present at most
4,294,967,296 cumulative float64 probe-value bytes to the adapter. The exact preflight sum includes
three C0 baselines, C5 neutral, every C1 basis chunk and optional spatial request, one C2 request per
target band, six C3 requests, and two C4 requests when `B>1`; it is checked before adapter
construction. This is a cumulative work/I/O ledger, not a peak-RSS guarantee. Catchable allocation
failures become invalid-bundle or execution failures, but an operating-system OOM kill cannot be
converted into a BandTrace exit code. Every declared numeric scalar and probe value, and every
numeric array value in the NumPy reference artifact, must have absolute value at most `1e12`.
Reflectance valid-range endpoints and probes are additionally confined to `[-0.1,2.0]`, every valid
range has width at least `0.1`, each normalization offset lies inside the model raw range, and each
normalization scale lies in `[0.01,2.0]`. Under the maximum 512 bands and 262,144 spatial cells,
these bounds keep the worst binary64 route/affine/mean accumulation uncertainty below the frozen
`1e-6` tap and recovered-route tolerance; float32/internal approximate arithmetic receives no such
allowance. Every center, support endpoint and SRF wavelength coordinate, after explicit nm/µm
conversion to nm, must lie in `[100,100000] nm`; this is the frozen domain used by C3 clipping.
Every derived static quantity must remain finite. An
adapter response value with magnitude above `1e150` is outside the protocol. These ceilings are
resource/numerical safety bounds, not biological thresholds. FWHM must be in `[1,50,000] nm`, and
each route weight is either exactly zero or at least `1e-12` (a negative declared weight is an
invalid bundle, exit 2); these rules prevent incompatible
resolution floors and subnormal weights from changing categorical logic through underflow.

## 3. Executable adapters and trust boundary

Two adapters are in scope:

- `numpy-linear-v1`: a safe, non-pickled NPZ fixture/reference model used to prove the instrument;
- `subprocess-npz-v1`: a user-supplied argv list receives a temporary non-pickled NPZ and must
  return a bounded NPZ containing `output` and the tapped `pre_core` tensor.

The adapter input contains exactly `probes`, `target_band_ids`, `wavelength_nm` and `fwhm_nm`.
It never contains the declared route matrix, normalization parameters or an expected tap. Those
must be owned independently by the pinned adapter/artifact; otherwise C1 would merely ask the
executable to repeat BandTrace's asserted answer. The adapter must bind columns by
`target_band_ids` and accept a tied arbitrary ordering for C4. A fixed positional wrapper that
ignores those IDs is a conformance fault, even if its baseline happens to use the expected order.
The NumPy artifact must contain its own route, affine parameters and spatial operation; none may
fall back to the bundle declaration. A subprocess working directory contains only its pinned
artifact, explicitly argv-referenced pinned adapter assets, and the invocation input/output paths;
BandTrace does not stage `model.json`, `sensor.json`, `route.json` or the original probes beside it.
The only argv tokens that BandTrace itself interprets, stages and verifies as files are exact
`{artifact}`, `{input_npz}`, `{output_npz}` and `{asset:<manifest_key>}` placeholders. Every other
token is passed verbatim. Such an ordinary token may name an absolute or relative ambient path to
the trusted subprocess; BandTrace neither detects nor pins that dependency, and the report's
`SUBPROCESS_DEPENDENCIES_UNATTESTED` boundary applies. Referenced asset keys must be pinned manifest
extras. The staged artifact is mode 0700 so
`{artifact}` may be the direct executable. If argv token 0 is `{asset:<key>}`, that runner is also
mode 0700; every other staged asset is mode 0600 regardless of source-tree permission bits.

For every invocation, `output` must be finite float64 with exact shape `[N]`; `pre_core` must be
finite float64 with exact shape `[N,model_channels]` after the declared fixed spatial reduction.
No implicit reduction, class selection, squeezing or broadcasting is allowed.

The subprocess and its tap placement are explicitly reported as `USER_CODE_TRUSTED` and
`SUPPLIER_REPORTED_TAP`. BandTrace cannot prove that a returned tap feeds the selected output; a
dishonest or incorrectly instrumented process can return a decoy. The NumPy reference adapter is
`INSTRUMENT_CONTROLLED_REFERENCE`, but that assurance does not transfer to customer subprocesses.
BandTrace does not claim to sandbox user code; production users must run it in their own
network-disabled container or equivalent boundary.
Nor can BandTrace observe every file, package, environment variable, device or network resource a
subprocess reads. Its report must therefore state `SUBPROCESS_DEPENDENCIES_UNATTESTED`; manifest
hashes cover only declared bytes, and the runtime fingerprint is explicitly non-exhaustive.
Remote APIs, network inference, stochastic/generative outputs, labels-only/argmax-only outputs and
opaque in-model preprocessing are outside v0.1. ONNX, TorchScript or any other runtime can be
wrapped behind the subprocess protocol without BandTrace deserializing its artifact.

## 4. Deterministic checks

All mutations derive from a length-framed seed. `model`, `sensor`, `probe` and `route` are the
lowercase SHA-256 hex strings of the exact manifest-pinned files, not hashes of parsed/canonicalized
objects. Start SHA-256 with ASCII `bandtrace-mutation-seed-v1` followed by exactly one zero byte; for
`(model,sensor,probe,route,policy)` in that order, append the label
length as unsigned 2-byte big-endian, label ASCII bytes, value length as unsigned 8-byte
big-endian, then the value bytes. Hashes are lowercase ASCII hex and `policy_id` is UTF-8. A
canary sub-seed starts with ASCII `bandtrace-canary-v1` followed by exactly one zero byte, then
hashes `base_seed_bytes || u16be(len(canary_id_utf8)) || canary_id_utf8`.
For a length-`B>1` C4 permutation, `shift=1+u64be(subseed[0:8]) mod (B-1)` and
`p[i]=(i+shift) mod B`; `B=1` is not applicable. The known-answer vector with model/sensor/probe/
route hashes equal to 64 repetitions of `0`/`1`/`2`/`3`, policy `bandtrace-0.1-r29` and `B=5` is
base seed `d9d2f0dc8aefdc56e2b3d91ca7112b698b7463f4f0a38747f7bd1b6c5e3257bf`, C4 shift `1`
and permutation `[1,2,3,4,0]`. No ambient RNG is used.

- **C0 replay:** run baseline three times. `Pq` is frozen linear interpolation on the sorted finite
  flattened vector: `h=(n-1)q`, then `v[floor(h)]+(h-floor(h))*(v[ceil(h)]-v[floor(h)])`;
  `median` is the same rule at `q=0.5`. With the translation-invariant
  `S=max(1,P99(abs(output-median(output))))`, output replay jitter is the
  maximum across probes of `(max_replay(output)-min_replay(output))/S`; `j` must be at most `1e-7`.
  Apply the same elementwise max-minus-min statistic to `pre_core` using
  `S_tap=max(1,P99(abs(first_replay_pre_core)))`; both jitters must be at most `1e-7`. Passing
  establishes output and reported-tap replay stability on these probes,
  not global determinism.
- **C1 declared-tap agreement:** compare the adapter-reported `pre_core` tensor with declared `W`, affine
  normalization and spatial transform on all three baseline replay taps and an exact band-basis
  challenge; baseline error is the maximum across all three, never replay 0 alone.
  It contains one row per target band: all bands start at their declared neutral and exactly that
  row's band is set to the valid-range endpoint farthest from neutral (equal-distance tie -> lower
  endpoint). Rank-4 rows are constant over the original `H,W`. Rows are kept in target-contract
  order and chunked to at most 256 rows and 256 MiB of float64 probe values per invocation. Every
  model channel's expected maximum absolute basis-to-neutral tap span must exceed `1e-6`; otherwise
  C1 is `INCONCLUSIVE_INSUFFICIENT_ROUTE_EXCITATION`, an overall fault that blocks X2/X3 rather than
  a false agreement. Every required model channel must have provenance. Missing, duplicate,
  reordered or many-to-one-when-undeclared executable routes are completed faults; a negative
  declared weight is rejected earlier as an invalid bundle. Basis-minus-neutral taps are also
  inverted through the declared scale and raw active delta. A declaration-only recovery is first
  evaluated through exactly the same arithmetic; its recovered route must match `W` and its
  recovered offset must match the declared offset within `1e-6`. Otherwise C1 is
  `INCONCLUSIVE_ILL_CONDITIONED_RAW_RECOVERY`, an overall route fault that blocks X2/X3 and makes
  C6 reported-tap collision evidence inconclusive. When conditioned, the observed recovered route
  is compared **directly** with `W`, and the offset recovered as
  `sum_j W[i,j]*neutral_j - neutral_tap_i*scale_i` is compared **directly** with the declared
  offset; both maximum absolute errors must be at most `1e-6`. The two comparisons may not be
  replaced by separate observed-to-reference and reference-to-declaration tolerances, which would
  silently admit twice the frozen error. This prevents affine scaling from hiding a materially wrong raw
  route while its normalized tap remains within the absolute tap tolerance. An exactly declared route may still
  be spectrally unsupported; that is reported on S without erasing valid X evidence.
  For rank-4 `mean`, C1 invokes the complementary even/odd rows plus two asymmetric three-level rows:
  flattened cell classes cycle through neutral, neutral plus one quarter of the far-endpoint delta,
  and the far endpoint, with the second row reversing cell order. The declared per-cell
  route/affine transform followed by the exact H/W mean must match the reported tap within `1e-6`.
  When mathematically distinguishable on the submitted spatial shape, a first-pixel, max, midrange,
  median, crop or other observed undeclared reduction is `undeclared_spatial_reduction`, blocks
  X2/X3 and does not erase valid S. With one cell all singleton-preserving reductions collapse to
  identity; with exactly
  two cells the usual even-sample median, midrange and mean are identical for every input. Those
  mathematical equivalences are explicitly reported rather than claimed distinguishable.
- **C2 value dependence:** for each complete target band, evaluate input-only cyclic shifts
  `1..N-1` and choose one maximizing the fraction of probes whose band tensor changes by more than
  `1e-6` in max absolute cell difference. Break sorted-shift ties with canary ID
  `C2_value_dependence:<target_band_id>` and the frozen sub-seed rule, then invoke only that mutation
  while holding all other values and metadata fixed. A band is observably dependent when normalized output
  delta exceeds `tau=max(1e-6,10*j)` on at least 20% of probes. This is probe-local evidence, never
  proof that a band is unused everywhere. Before interpreting output invariance, the mutation itself
  must change that band above `1e-6` on at least 20% of probes; otherwise the result is
  `INCONCLUSIVE_INSUFFICIENT_EXCITATION`, not evidence of non-use. X3 requires adequate excitation
  and observed dependence for every target ID in the supplier-declared required-dependence set.
  Each such ID must have aggregate absolute route weight at least `1e-4`; a static declaration below
  that floor is an invalid bundle (exit 2), not a completed dependence result. All other bands remain in
  the report but do not silently become X3 conjuncts through tiny numerical weights. Inadequate
  excitation of a required ID is an overall conformance fault and blocks X3; it is never exit-0
  evidence of non-use. Observed dependence outside the required set is diagnostic, not a fault,
  unless that target's declared route column is exactly zero, in which case it is
  `hidden_resampling_or_extrapolation`. Every C2 returned tap must also match the declared transform
  for that mutated request within `1e-6`; a context-dependent tap is a route fault even when output
  dependence is real.
- **C3 metadata dependence:** always run separate wavelength-only and FWHM-only rotations while
  holding values, IDs and the other metadata field fixed. Each field also receives two seed-keyed,
  non-uniform magnitude challenges. For the field's canary sub-seed, each target ID receives a
  `key_j` formed by hashing the sub-seed, one zero byte, ASCII `rank`, one zero byte,
  `u16be(len(id_utf8))`, then `id_utf8`; ascending `(key_j,id)`
  order assigns zero-based `rank_j`, and `a_j=(rank_j+1)/(B+1)`. Wavelength challenges are
  `clip(lambda_j+a_j,[100,100000])` and `clip(lambda_j-a_j,[100,100000])`. For FWHM,
  `g_j=1+0.01*a_j`; challenges are `clip(fwhm_j*g_j,[1,50000])` and
  `clip(fwhm_j/g_j,[1,50000])`. These non-uniform vectors change multisets and expose demonstrated
  range/ratio-invariant bypasses that rotations and uniform affine mutations preserve. Rotation ties use the
  `C3_wavelength_dependence` or `C3_fwhm_dependence` sub-seed. A field is observed when any adequately
  excited rotation or magnitude challenge changes output above `tau` on at least 20% of probes.
  If fewer than 20% of tuples can be changed, emit
  `INCONCLUSIVE_INSUFFICIENT_METADATA_EXCITATION`; do not infer metadata non-use. When that field is
  declared true, also emit the overall conformance fault `claimed_<field>_input_inconclusive` and
  block X3; an asserted required conditioning path cannot pass untested. With adequate
  excitation, a declared conditioning input must change output above `tau` on at least 20% of
  probes or it receives `NO_<FIELD>_DEPENDENCE_OBSERVED_ON_PROBES` and blocks X3. Conversely,
  observed dependence for a field explicitly declared false is
  `UNDECLARED_<FIELD>_DEPENDENCE_OBSERVED_ON_PROBES`, an overall conformance fault that blocks X3.
  Non-observation or insufficient excitation for a false declaration does not prove global
  non-use and is retained as a limitation rather than promoted. No finite challenge set proves
  non-use outside these exact rotations and keyed perturbations. X3 requires every claimed metadata
  field, as well as C2's required value paths, to pass. Every C3 tap must equal the declared static
  pre-core transform for the unchanged values within `1e-6`; otherwise
  `context_dependent_undeclared_tap` blocks X2/X3.
- **C4 order and ID binding:** run two distinct requests using the exact C4 permutation above.
  First, reorder columns, IDs, wavelengths and FWHM as tied tuples. The declared tap must match
  within `1e-6`, and output must remain invariant within
  `max(abs(delta_output))/S <= 1e-6`. Second, hold values, wavelengths and FWHM at their submitted
  positions and permute **only the IDs**. Its declaration-derived expected tap must change from the
  baseline by strictly more than `1e-6`; otherwise C4 is
  `INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION`, an overall route fault rather than a pass. When
  adequately excited, the returned ID-only tap must match its declaration-derived tap within
  `1e-6`. C4 is `PASS` with finding `ID_BOUND` only when both subtests pass. A failure emits the
  `reordered_bands` route fault and blocks X2/X3. This catches an adapter that ignores IDs and merely
  sorts tied tuples by wavelength. For one target band C4 is `NOT_APPLICABLE_SINGLE_BAND`.
- **C5 declared target neutral:** replace every target band by its sensor-contract neutral value,
  which must be finite and inside that target band's valid raw range. If no output delta exceeds
  `tau` on at least 20% of probes, emit `NO_JOINT_TARGET_NEUTRAL_EFFECT_OBSERVED`, unless the baseline is itself insufficiently different from
  the target-neutral input, in which case emit `INCONCLUSIVE_INSUFFICIENT_EXCITATION`.
  Neutral-mutation cancellation cannot prove a prior-only executable. Only when C2, both C3 fields
  and C5 are adequately excited may the report add the bounded diagnostic
  `NO_TARGET_EFFECT_OBSERVED_ABOVE_FROZEN_THRESHOLD_ON_CHALLENGES`; this is not a global or causal
  prior-only claim. Required-band nondependence already blocks X3. Otherwise C5 is diagnostic and
  cannot lower otherwise valid X evidence.
- **C6 edge alias:** report every target band whose raw unit SRF (or analytic FWHM Gaussian) has less
  than 0.99 mass inside support. Pair it with the in-support target whose center is nearest the
  crossed support endpoint, where in-support also means raw response mass at least 0.99. Choose the
  endpoint with greater outside mass (tie -> lower endpoint)
  and break band-distance ties by lexical target ID. Reuse each band's selected C2 output-delta
  vector rather than invoking a new mutation. Declared columns collide only when they are equal
  within `1e-6` and both have aggregate absolute weight at least `1e-4`; tapped columns use the same
  rule after C1's validated raw-route recovery from basis and C5 neutral taps. Output effects are indistinguishable
  only when `max(abs(delta_outside/S-delta_edge/S))<=tau`. A declared or tapped collision is
  `CLAMP_ALIAS_CONFIRMED`; indistinguishable output effects without one are only `ALIAS_SUSPECTED`.
  Missing an in-support partner or adequate C2 excitation is explicitly inconclusive. No pair is
  silently dropped. A malformed SRF makes only that target ID support-unresolved; valid outside
  targets are still paired and reported, while the aggregate C6 finding remains support-unresolved.
  A collision already present in the declared route is diagnostic and cannot
  invalidate X; only an undeclared collision in the reported tap is an executable-axis fault.

Changed output beyond declared support never establishes valid extrapolation.

## 5. Spectral-support states

Full-SRF comparison is performed **per model channel**, never on raw target bands. For channel `i`,
first area-normalize every target-band SRF independently, then form the routed effective response
`e_i(λ)=sum_j W[i,j] normalized_target_srf_j(λ)`, area-normalize `e_i`, and compare it with the
independently area-normalized paired training-channel SRF. Every strictly positive declared
`W[i,j]` participates; neither the row-sum tolerance nor the positive-weight floor licenses
discarding a contribution. A finite non-negative route with positive mass in every row is enough
to define this spectral-shape comparison. A positive row-mass defect, including a uniform 0.9 or
1.1 row scaling, remains an executable-route fault but cannot erase independently resolvable S
evidence; the final area normalization makes those two scalings spectrally equivalent. A zero-mass
row makes the corresponding response mixture unresolved and yields S0; negative declared weights
cannot reach a completed S state because they are invalid bundles.

SRFs must be finite, non-negative, strictly increasing in nm, have at least four knots and end at
no more than `1e-4` of peak response. An optional supplied SRF that fails those comparison-validity
rules yields S0. Missing mandatory center/FWHM/unit metadata is
instead an invalid bundle (exit 2), while a valid center+FWHM band without an SRF is eligible only
for S2. Every present SRF is validated even when another band lacks one; malformed partial SRFs
cannot be hidden by the Gaussian fallback. S3 requires valid SRFs for every model and target band.
`declared_validated_support.supplier_assertion` must be the literal JSON boolean `true` to enable
S2/S3; string lookalikes such as `"false"`, `"no"` or `"0"` are invalid rather than truthy.

Each supplied SRF is the piecewise-linear curve through its knots. Linear half-maximum crossings
must form one connected component. Its outer crossings define `derived_FWHM`, which must differ
from declared FWHM by no more than `max(0.1 nm,0.10*declared_FWHM)`. The segment-exact centroid must
differ from declared center by no more than `max(0.25 nm,0.25*declared_FWHM)`. A violation is S0 with
`srf_metadata_inconsistent`. These are internal-consistency checks, not calibration truth.

For full SRFs, the sorted union contains every native knot and support endpoint. On each union
interval, a curve is the native piecewise-linear segment only when the **whole interval** is inside
its supplied domain; otherwise both interval-side values are exactly zero. The left and right limits
at a native endpoint are therefore intentionally allowed to differ. This interval-side convention
prevents a non-zero endpoint from being spuriously connected by a trapezoid to a distant zero-outside
point. Available mass sums only whole intervals contained in support. Normalized L1 is the exact
integral of the absolute linear difference on every interval; a sign-changing interval is split
analytically at its one difference root. That analytic split is charged to the work ledger as a
root-augmented logical grid even though no interpolation array is materialized.
First moments are segment-exact: for `(x0,y0),(x1,y1)`, use
`dx/6*((2*x0+x1)*y0+(x0+2*x1)*y1)`. The S3 center tolerance uses derived training FWHM.

The S2 path uses analytic unit Gaussians. Available mass uses the normal CDF and centers use exact
mixture moments. L1 uses deterministic component-relative grids at `center+sigma*q`, initially
`q=-12..12` in `1/8` steps, then `1/16`; it must change by at most `1e-6`, otherwise one `1/32`
refinement is allowed. Bracketed difference roots use exactly 60 bisections. More than 250,000
unique knots, more than 50,000,000 component-point evaluations over the audit, or failed convergence
returns S0 `numerical_support_unresolved`, never a pass. The full-SRF path has the same 250,000
unique-knot ceiling per channel and a separate 50,000,000 interpolation component-point budget per
audit. It charges every training or positive-weight target interval evaluation on the union grid and
a conservative two-curve equivalent charge on the logical root-augmented difference grid; the
initial and augmented logical grids must each satisfy the knot ceiling. Exceeding either budget also returns S0
`numerical_support_unresolved`. Without any SRF, BandTrace cannot detect a
supplier-false but numerically well-formed center/FWHM declaration.

`S3_SRF_WITHIN_DECLARED_SUPPORT` requires at least 0.99 available target SRF mass, normalized L1
distance at most 0.05 and center shift at most `max(0.25 nm,0.25*derived_training_FWHM)`.
Center+FWHM-only Gaussian approximation can reach only
`S2_APPROX_WITHIN_SUPPORT`. Raw available mass below 0.99 is `target_srf_outside_support`; an
in-support effective response that fails L1 or center criteria is `routed_response_mismatch`; both
produce `S1_OUTSIDE_DECLARED_SUPPORT`. An absent supplier support assertion, invalid present SRF or
bounded numerical failure is `S0_SUPPORT_UNRESOLVED`. A non-reflectance or mismatched radiometric
quantity and other missing mandatory schema fields remain invalid-bundle exit 2 rather than a
completed S state.

The declared support is a supplier assertion pinned into the report. BandTrace does not infer a
validated envelope from weights.

## 6. Orthogonal claim states

Executable axis:

- `X0_NO_EXECUTABLE_OBSERVATION`
- `X1_REPLAY_STABLE_ON_PROBES`
- `X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES`
- `X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES`

Spectral axis:

- `S0_SUPPORT_UNRESOLVED`
- `S1_OUTSIDE_DECLARED_SUPPORT`
- `S2_APPROX_WITHIN_SUPPORT`
- `S3_SRF_WITHIN_DECLARED_SUPPORT`

Every report also states route assurance as `INSTRUMENT_CONTROLLED_REFERENCE` or
`SUPPLIER_REPORTED_TAP`; X2 is never independent attestation for the latter. Biological axis is
always `T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED` in v0.1. Even `X3+S3+T0` means only
`ELIGIBLE_FOR_EXTERNAL_TRANSPORT_TEST`. It says nothing about disease,
accuracy, calibration, field performance, spatial geometry, utility or safety.

## 7. Outputs and exit semantics

A completed run writes canonical `report.json`, deterministic summary `report.html`, `route.csv`,
deterministic `canary_outputs.npz` and `manifest.sha256`. The report retains every manifest-declared
input/artifact hash, while `manifest.sha256` hashes the other four output files in lexical filename
order; it does not make the circular claim of hashing itself. A non-exhaustive runtime fingerprint,
raw thresholds, mutation results, per-band route/SRF/dependence results, orthogonal states and
mandatory limitations are retained. To avoid
duplicating hundreds of thousands of verbose objects, `route_audit` stores ordered model/target ID
axes plus parallel two-dimensional matrices for rounded weight, exact float64 hex and positivity,
and one target-axis exact-zero vector. `route.csv` remains the per-cell tabular view. The HTML
contains states, faults, limitations, canary statuses and provenance plus SHA-256 values for
`report.json`, `route.csv` and `canary_outputs.npz`; it does not duplicate the full route matrices.
Floats are rounded to eight decimal places; non-finite values are forbidden. Every categorical
comparison uses the unrounded binary64 value. Near a threshold, a failed metric and its threshold
can therefore display as the same rounded decimal; the categorical status/pass boolean is
authoritative. Because exact zero versus positive
route weight is categorical, the exact arrays ensure rounding can never hide the basis of a
route/dependence decision. The report also records the installed BandTrace source-
tree digest. Its prefix is ASCII `bandtrace-source-tree-v1` followed by exactly one zero byte (the
complete prefix hex is `62616e6474726163652d736f757263652d747265652d763100`), followed by the
records below and then SHA-256, where
every recursively discovered regular `*.py` file under the installed `bandtrace` package
contributes, in sorted relative POSIX-path order,
`u32be(path_utf8_length) || path_utf8 || u64be(file_length) || file_bytes`. It also records the
installed distribution version and exact normative document/config SHA-256 values. Missing, empty,
unreadable or malformed installed distribution metadata fails with exit 3 before bundle loading or
adapter execution rather than emitting ambiguous provenance. This is Python
source provenance: it distinguishes different installed regular `*.py` and normative bytes, but it
is **not execution attestation**. It does not hash imported bytecode, the interpreter, native or
other dependency bytes, environment state, or in-memory mutation; the runtime fingerprint remains
explicitly non-exhaustive. The wheel and source
distribution ship exact copies of both normative resources under `bandtrace/normative`; every audit
verifies those bytes against compiled constants before loading the bundle and fails with exit 3 if
either resource is missing, unreadable or different. This is a build-internal consistency gate, not
external authentication: coordinated replacement of code, constants and resources is not detected.
Authenticity requires an independently trusted wheel/release digest or signature. Three Linux runs of the instrument-controlled
NumPy reference in the same pinned CPU environment must produce identical output bytes. A subprocess can earn
only C0 probe-local replay stability; its transitive reproducibility remains unattested even when
three reports happen to match.
Measured wall-clock durations are excluded from all required deterministic artifacts; reports
retain invocation counts and configured budgets only. Timing may be printed as a non-artifact CLI
diagnostic.
The requested output directory must not already exist, and its parent must already exist as a
directory. BandTrace opens and pins that parent directory identity and verifies Linux/libc
`renameat2` availability before bundle loading or adapter execution. All
staging creation, file writes, cleanup, publication and fsync operations are then relative to the
pinned parent descriptor. All five files are written and file-fsynced in a fresh sibling staging
directory, `manifest.sha256` is written and fsynced last as the completion marker, and the staging
directory is fsynced. Immediately before publication, the staging pathname must still identify the
opened staging directory. Linux `renameat2(RENAME_NOREPLACE)` then atomically publishes by name
without replacing an entry created concurrently; there is no `rename` fallback. The parent is then fsynced and both
descriptor-relative and path-reachable identities are checked. Invalid, execution and publication
failures **before** the successful no-replace rename leave no BandTrace destination. A failure after
successful publication—such as parent fsync failure, parent path replacement or final identity
check failure—returns exit 3 and deliberately does not roll back or delete the destination path.
The filesystem threat model treats every process running as BandTrace's Unix UID as trusted; use a
dedicated account and a private output parent when that assumption would otherwise be false. Linux
does not provide a `renameat2` operation conditioned on the already-open source-directory inode, so
a same-UID process can swap the random staging name in the final check-to-rename window. If an exit-3
path exists, it is untrusted and is not a BandTrace report; it may be a concurrently substituted
entry. In the trusted-parent case, the published entry contained all five file-fsynced artifacts and
its completion manifest at the instant of the atomic rename, though a concurrent actor may later
move, replace, mutate or remove it. The originally requested pathname may therefore no longer resolve
to it. No post-rename cleanup is attempted, preventing BandTrace from deleting a concurrently swapped
path. A partial or stale prior report can never be silently reused as the current run. Each non-manifest output is at
most 268,435,456 bytes and all five outputs total at most 536,870,912 bytes. Exceeding either budget
is an execution/output failure (exit 3) that leaves no destination.

Prepublication cleanup is deliberately non-recursive: it unlinks only files found through the
opened staging descriptor, refuses unexpected subdirectories, and rechecks the staging name against
that descriptor immediately before `rmdir`. Like publication, `rmdir` is still name-based and cannot
be conditioned on an inode; the private trusted-parent requirement is therefore also part of the
cleanup boundary. A malicious same-UID swap inside the last check-to-`rmdir` window is outside the
guarantee.

Exit `0` means a report completed without a conformance fault, not that deployment is safe. Exit
`2` is an invalid bundle, `3` an execution failure and `4` a completed report with a conformance
fault.

## 8. Planted-fault release gate

The release suite contains clean cases plus 22 planted faults: dropped band, edge clamp, reorder,
an ID-blind adapter that sorts tied metadata tuples by wavelength,
nm/µm mismatch, missing mandatory FWHM, invalid present SRF, in-support routed-response mismatch,
target SRF outside support, ignored or undeclared wavelength/FWHM conditioning, routed raw domain
outside model range, target-invariant output on adequately excited challenges, stochastic output,
radiometric mismatch, hidden normalization, undeclared spatial reduction, context-dependent
undeclared taps, hidden resampling and duplicate band IDs.
The planted `edge_clamp` is specifically an **undeclared executable** mix of an outside-support band
into the exposed tap while the declared route remains on its supported edge. If the mix were
declared and matched, X2/X3 evidence could survive while S correctly fell to S1.

Release requires fail-closed detection of all 22 applicable planted faults. Invalid-bundle fixtures
must exit 2 without a report; every completed faulty fixture must exit 4 and set the overall
conformance-fault flag. The planted list is a release corpus, not an exhaustive runtime fault-code
or severity enumeration; consumers identify a fault using code, axis and report/exit context.
Orthogonality is preserved: route/execution faults cannot reach X2
or X3; dependence faults cannot reach X3; spectral-support faults cannot reach S2 or S3. A fault on
one axis does not erase valid evidence on another. Every clean fixture must reach its frozen
expected states, no clean fixture may be falsely outside support, and all report bytes must match
across three runs. A hidden transform after a supplier-reported tap is explicitly not observable;
that planted fault is applicable only where the fixture exposes the actual execution tap.
Thresholds may not be tuned after fixture results.

Security fixtures additionally require fail-closed handling of manifest traversal/symlinks,
duplicate YAML/JSON keys, aliases, unpinned `{asset:<key>}` references, object arrays, oversized/decompression-
bomb NPZ members, non-finite/wrong-shape arrays, subprocess timeout/nonzero exit and oversized
adapter output. Subprocesses run without a shell in a new process group; every success, failure or
timeout attempts one `SIGKILL` of that process group while its leader PID is still reserved. When
the OS delivers that signal, affected same-group members terminate. BandTrace observes leader exit with
Linux `waitid(...,WNOWAIT)`, attempts the group signal before reaping the leader, and never signals
the numeric process-group ID after reap, so its cleanup attempt cannot target a newly reused ID.
An OS error from group signalling is a stable execution failure: BandTrace still attempts a direct
leader kill and reap, but cannot guarantee cleanup of descendants when the kernel rejects the group
signal. This is cleanup, not containment:
trusted code can create a new session/process group, and BandTrace does not prevent or attest that
escape. Combined subprocess stdout/stderr is capped at 1 MiB. A cumulative measured adapter-wall
failure threshold of 600 seconds applies to both adapters; for subprocesses its ledger covers
parent-side request serialization, child execution, response decoding and invocation cleanup rather
than only child runtime. The child loop actively polls the threshold, but synchronous NumPy work,
parent serialization/decoding and cleanup are checked only after they return and are not
preemptible. It is therefore not a hard end-to-end deadline; deployments that require one need an
external supervisor. Total
invocations at `2*bands+12`. Temporary directories are mode 0700. Each per-invocation directory and
its input/output NPZ files are removed in an invocation-level `finally` before the next invocation,
after validated response arrays have been copied into owned memory. Thus transient staging is
non-cumulative: at most one invocation's input/output plus the fixed staged artifact/assets are
retained. Cleanup on success, adapter error and timeout is verified by release fixtures, but is not
a containment claim against hostile trusted code. HTML is escaped,
CSV is RFC 4180 encoded, and formula-leading cells are prefixed to prevent spreadsheet execution.

## 9. Relationship to EXP-0121

EXP-0121 supplied the motivating failure mode, but not product validation. Its fixed-grid query
coordinate can alias to an envelope-edge slot while returning a plausible number. A 2026-07-27
audit also found that its synthetic `COORDINATE_SHIFT` control was not paired, its old
`consumed_bands` name overstated a query-representation check, and other arm/gate defects require a
partial retraction. BandTrace therefore imports no EXP-0121 causal or adapter-correctness claim. It
must earn every v0.1 state through the planted-fault matrix above.
