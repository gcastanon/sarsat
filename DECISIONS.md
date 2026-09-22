# SarSat design decisions

Each entry records one design issue: the question, the options weighed, the decision
and its consequences. Anything rejected as "beyond a basic model" is cross-referenced to
[IMPROVEMENTS.md](IMPROVEMENTS.md). Status is **Decided** unless noted.

| # | Issue | Decision in one line |
|---|-------|----------------------|
| 1 | Package boundary | Standalone Jumanji environment + a thin MAPX wrapper |
| 2 | Orbit model | Circular two-body orbits, evaluated in closed form |
| 3 | Earth rotation | Included, folded into the node angle |
| 4 | Pointing convention | SAR terms: squint along track, look angle across it |
| 5 | Rectangular beam test | Box in az/el space, evaluated in tangent space |
| 6 | Order of events in a step | Advance time, then sense at the new geometry |
| 7 | Sensing within a 60 s step | Instantaneous snapshot at the step boundary |
| 8 | Action encoding | `[incidence, squint, side, sense]`; hybrid 2 continuous + 2 binary |
| 9 | Land / water targets | 1-degree embedded land mask, exact split, area-uniform |
| 10 | Number of targets at reset | `reset(key, num_targets)` with static padded capacity |
| 11 | Reward and priority | Shared team reward, priorities sum to 1 |
| 12 | Battery / duty cycle | Linear charge model, duty cycle = recharge / cost |
| 13 | Target observation | Fixed slots, nearest-to-nadir, tracked across look-ahead |
| 14 | Neighbour observation | Exact K-nearest by dense N x N search |
| 15 | Episode end | Terminate when all imaged; truncate at time limit |
| 16 | Numerical stability | No integrator, no division by small numbers, float32 km |
| 17 | Visibility limits | Two-sided 20-45 degree incidence band + squint limit + horizon |
| 18 | Agent IDs and global state in MAPX | No agent IDs; joint-observation global state |
| 19 | Satellite count cap | Hard limit of 1000 |
| 20 | Evaluation logging and viewer | Host-side loop over the jitted step; one self-contained HTML |
| 21 | Two-sided access and the action encoding | Binary side switch (default); signed encoding kept behind a flag |
| 22 | Cooperation pressure | Hotspot targets + battery-binding + overlapping planes; measured, not assumed |

---

## Issue 1 - Package boundary

**Question.** Should the environment live inside MAPX or stand alone?

**Options.** (a) A module inside `mapx/`. (b) A standalone package that depends only on
JAX and Jumanji, plus a small adapter in MAPX, which is how MAPX treats CAMAR, AeroJAX
and every other environment family.

**Decision.** (b). `sarsat` is a plain `jumanji.env.Environment`. Its observation tuple
uses MAPX's field names (`agents_view`, `action_mask`, `step_count`) and it exposes
`num_agents`, `time_limit` and `action_dim`, so the MAPX wrapper
(`mapx_integration/mapx/wrappers/sarsat.py`, ~60 lines of logic) only re-types the
observation, optionally adds a global state, declares the hybrid action spec and adds
the `env_metrics` extras entry.

**Consequences.** The environment can be tested and benchmarked without MAPX. The state
carries a `key` field solely because MAPX's `AutoResetWrapper` requires one.

## Issue 2 - Orbit model

**Question.** How simple can the physics be while still behaving like LEO?

**Options.** (a) Numerical integration of two-body dynamics. (b) Keplerian elements with
eccentricity. (c) Circular orbits evaluated in closed form.

**Decision.** (c). Each satellite has five constants (radius, mean motion, inclination,
RAAN, initial phase). Its position at time `t` is
`r * (cos(u) * p + sin(u) * q)` with `u = phase + n * t`, where `p`, `q` span the orbital
plane. Satellites do not manoeuvre, so nothing else is needed. Orbits are drawn
independently and uniformly: altitude in `altitude_km`, inclination in
`inclination_deg`, RAAN and phase in `[0, 2 pi)`. Set `min == max` to pin a value.

**Consequences.** No integrator state, no drift, and any future time can be evaluated
directly, which is what makes the look-ahead observation cheap. Eccentricity, J2
precession and drag are in IMPROVEMENTS.md.

## Issue 3 - Earth rotation

**Question.** Is a rotating Earth "beyond basic"?

**Decision.** It is kept, because without it every ground track repeats the same great
circle forever and coverage becomes degenerate. It costs nothing: in the Earth-fixed
frame the orbital plane simply drifts west, so the node angle becomes
`raan - omega_earth * t` and no rotation matrices appear anywhere. Targets are therefore
constant vectors and all geometry is done in the Earth-fixed frame.

**Consequences.** The Greenwich angle at `t = 0` is taken as zero; this loses nothing
because RAAN is uniformly random. The "along" body axis is the inertial flight direction;
the ~0.5 km/s contribution of Earth rotation to ground-relative velocity is ignored.

## Issue 4 - Pointing convention

**Question.** What do "azimuth" and "elevation" mean for a nadir-looking sensor?

**Options.** (a) Polar: off-nadir angle plus clock angle. It is singular exactly at
nadir, where most targets are, so observations and actions would be discontinuous there.
(b) Antenna-style az/el about a nadir boresight.

**Decision.** (b), named in SAR terms. Body axes are `along` (flight direction), `cross`,
`nadir`; a direction is `(sin s, cos s sin a, cos s cos a)` where `s` is the **squint**,
the along-track angle off broadside, and `a` is the **cross-track look angle** measured
at the satellite from nadir. The only singular directions are straight ahead and
straight behind, which are above the horizon and can never be a ground target.

The two combine into the total off-nadir angle through `cos(off_nadir) = cos(s) cos(a)`,
and the off-nadir angle maps to the **incidence angle** at the target by the sine rule on
the satellite-centre-target triangle, `sin(incidence) = (r / R) sin(off_nadir)`
(`look_angle_band`). Incidence always exceeds the look angle and the gap grows with
range; at 550 km, 20 degrees of incidence is 18.4 degrees of look angle and 45 is 40.6.
Incidence is the angle the sensor model is written in because it is the one that governs
backscatter and ground-range resolution; the look angle is what the geometry is actually
tested in, because that is where the satellite sits.

**Consequences.** Observed target angles and commanded angles share one encoding, so
copying an observation slot into the action images that target -- a property a test
pins down across the whole action square. Pointing is instantaneous, including switching
sides; slew-rate limits are in IMPROVEMENTS.md.

## Issue 5 - Rectangular beam test

**Question.** How is "target inside the +-4 x +-4 degree beam" evaluated for up to
1000 x 1000 satellite-target pairs?

**Options.** (a) Rotate every line of sight into a per-satellite beam frame. (b) Compare
az/el angles: `|az_t - az_cmd| <= 4` and `|el_t - el_cmd| <= 4`. (c) The same comparison
as (b) done in tangent space.

**Decision.** (c). The beam is a box in az/el space. Rather than calling `arctan2` on
every pair, `in_az_el_box` evaluates `tan` on the box edges (one pair per satellite) and
compares against line-of-sight components. The same function implements the pointing
limits. A unit test proves it equals the explicit-angle comparison.

**Consequences.** The per-pair work is one 3x3 rotation, one `hypot` and comparisons. This
single change made the step 10x (8 satellites) to 60x (1000 satellites) faster on CPU.
Requirement: pointing limit + beam half-width < 90 degrees (checked in the constructor).
The azimuth extent of the box in true angle shrinks by `cos(el)` away from nadir; an
exactly rectangular footprint is in IMPROVEMENTS.md.

## Issue 6 - Order of events in a step

**Question.** Is the action applied before or after time advances?

**Decision.** `step` advances time by 60 s and then applies the action at the *new*
positions. The observation describes the next `lookahead_steps` steps, so its first
look-ahead slot is exactly the geometry under which the chosen action executes; with the
default `N = 2` the second slot is pure planning information (for example, saving
battery for a better pass).

**Consequences.** Nothing in the observation is stale or redundant, and "next N steps"
means what it says. Neighbour features are also evaluated at the next step for the same
reason.

## Issue 7 - Sensing within a 60 s step

**Question.** A satellite travels ~450 km in one step while the beam footprint is ~80 km
wide. What does "in the beam" mean over a step?

**Decision.** Sensing is an instantaneous snapshot at the step boundary.

**Consequences.** Simple, cheap and deterministic. With the default +-25 degree squint
the snapshots join up along track: the accessible region reaches +-317 km along track at
550 km altitude while the satellite travels 447 km per step, so consecutive steps overlap
and no ground is skipped between them. (A narrower squint limit reopens those gaps --
at +-15 degrees of *pointing* the reach was only +-150 km against the same 447 km, and
about a third of each ground track was unreachable.) Swept (strip-map) footprints and
dwell time are in IMPROVEMENTS.md.

## Issue 8 - Action encoding

**Decision.** One row per satellite, float32: `[incidence, squint, side, sense]`. The
two pointing slots are in `[-1, 1]` (clipped); `side` and `sense` are category indices
`{0, 1}` stored as floats, the convention MAPX uses for hybrid actions. `sense > 0.5`
requests imaging. In MAPX the wrapper declares a `HybridArray` with layout
`[2 continuous, 1 binary, 1 binary]`, so `HybridActionHead` produces a 2-D tanh-Gaussian
plus two Bernoullis. The `action_mask` closes the `sense` slot when the battery cannot
pay for a look, which the hybrid head maps to "do not sense"; the `side` slot is always
open today and is the natural place to express a roll constraint later.

**Consequences.** A sense request with an empty battery is silently ignored by the
environment, so an unmasked or purely continuous policy is also safe. Issue 21 records
why the side is a switch rather than a sign.

## Issue 9 - Land / water targets

**Question.** How are targets placed "80% on land, 20% on water" with no heavy GIS
dependency and no work at step time?

**Options.** (a) Runtime dependency on a land-mask package (2.5 MB, NumPy only, not
jittable). (b) Rejection sampling inside `reset` (unbounded loop). (c) A small embedded
mask with inverse-CDF sampling.

**Decision.** (c). `land_mask_1deg.npy` is a 1-degree majority-vote mask (8 kB bit-packed)
derived once from the public-domain NOAA GLOBE data by `scripts/build_land_mask.py`. At
construction the land cells and the water cells each get an area-weighted (`cos(lat)`)
CDF. `reset` draws a cell by `searchsorted` and jitters uniformly inside it. The split is
exact, not random: the first `round(land_fraction * num_targets)` slots are land.

**Consequences.** Jittable, O(M log cells), no rejection loop. The mask's land area is
28.8% of the globe (true value ~29%). At 1-degree resolution a "land" point near a coast
can be up to ~50 km offshore; irrelevant to the task. Targets are point masses fixed for
the whole episode; `State` keeps both lat/lon (the definition) and the derived
Earth-fixed position (what the geometry uses).

## Issue 10 - Number of targets at reset

**Question.** The request is for `reset` to take the number of targets, but JAX needs
static array shapes and Jumanji/MAPX call `reset(key)`.

**Decision.** The constructor takes `max_targets`, the static capacity. `reset(key,
num_targets=None)` accepts a Python or traced integer up to that capacity (default: the
capacity). Unused slots are inactive padding with zero priority and are never visible.

**Consequences.** One compiled program serves any target count, including a different
count per vmapped environment. Compute scales with `max_targets`, not `num_targets`, so
set the capacity close to what you use. Through MAPX (which only calls `reset(key)`) the
count equals `max_targets`.

## Issue 11 - Reward and priority

**Decision.** Every target has priority `1 / num_targets`. When an active target is
inside any sensing beam, its priority is added to a single team reward that every agent
receives, and the target is deactivated. `any` over satellites means simultaneous hits
by several beams pay once.

**Consequences.** Priorities sum to 1, so the episode return is the (priority-weighted)
fraction of targets imaged, in `[0, 1]`, comparable across scenario sizes. MAPX's
`RecordEpisodeMetrics` averages rewards over agents, which leaves this value unchanged.
Non-uniform priorities only require changing one line in `reset`; they are not in the
observation yet (IMPROVEMENTS.md).

## Issue 12 - Battery / duty cycle

**Decision.** Charge is a fraction in `[0, 1]`, full at reset:
`battery <- clip(battery + recharge_rate - sense_cost * sensing, 0, 1)`. Sensing is only
honoured when `battery >= sense_cost`. Defaults: `sense_cost = 0.10`,
`recharge_rate = 0.02`.

**Consequences.** A full battery buys ~12 consecutive sensing steps; the sustainable
long-run duty cycle is `recharge_rate / sense_cost = 20%`. The comparison uses a `1e-6`
tolerance so float32 round-off cannot make the duty cycle irregular; the same helper
drives the action mask, so mask and dynamics can never disagree. Eclipse-dependent
charging and idle loads are in IMPROVEMENTS.md.

## Issue 13 - Target observation

**Question.** "All targets visible in the next N steps" is variable-length; networks
need a fixed width.

**Decision.** `max_observed_targets` slots (default 10). A target's score is its best
(smallest) off-nadir angle over the look-ahead steps in which it is visible; the top
slots are kept, best first. Each slot holds, for every look-ahead step,
`(visible, az / max_az, el / max_el)`, zeroed when not visible. The same target occupies
the same slot across the look-ahead steps, so the agent sees a short track rather than
unrelated detections.

**Consequences.** Width is `3 * slots * N`. With the defaults the accessible region is
about 404,000 km^2, or 0.079% of the globe, so 10 slots only saturate for dense target
fields (roughly 10,000 targets and up), where the dropped ones are the ones furthest
from the inner edge of the band. Ranking by off-nadir angle means near ties go to the
steeper, shorter-range look. Ranking uses the
cosine of the off-nadir angle, and `arctan2` is evaluated only for the selected slots.

## Issue 14 - Neighbour observation

**Decision.** For each satellite the `num_neighbors` (default 5) nearest satellites at
the next step, nearest first, as `(az / pi, el / (pi/2), distance / max_distance)` in
the same body frame and az/el convention as targets. Found by an exact dense search:
N x N difference vectors, squared distance, `top_k`. If there are fewer than K other
satellites, the block shrinks (static) rather than being padded.

**Consequences.** At N = 1000 this is a 1M-entry matrix per step - small next to the
target geometry - and it is exact. Differences are taken before squaring to avoid the
catastrophic cancellation of the `|a|^2 + |b|^2 - 2ab` form in float32. Line-of-sight
blockage by the Earth is ignored; this block is coordination context, not a
communication model. Neighbour azimuth wraps at +-pi (a satellite directly overhead),
which is rare; a unit-vector encoding is noted in IMPROVEMENTS.md.

## Issue 15 - Episode end

**Decision.** `StepType.LAST` with discount 0 when no active target remains
(termination); `StepType.LAST` with discount 1 at `time_limit` (truncation), matching
the convention MAPX's `bootstrap_truncation` option relies on. Default `time_limit = 180`
(three hours, about two orbits).

## Issue 16 - Numerical stability

**Decisions.**
* Positions are closed-form functions of the integer step count; no state is integrated,
  so error does not accumulate. float32 error against a float64 reference is < 50 m after
  one simulated day and < 400 m after a week (beam footprint: ~80 km).
* Units are kilometres; line-of-sight components are formed as "target rotated into body
  axes, plus radius on the nadir axis", which avoids building N x M x 3 differences and
  has ~1 m round-off.
* Angles use `arctan2` and `hypot` only - no `arcsin`/`arccos` (NaN at |x| > 1 by
  round-off) and no division by a quantity that can approach zero. The one division
  (`/ |line of sight|`) is bounded below by the orbit altitude.
* Masked `top_k` scores use `-inf`, never NaN; the dense neighbour search puts `inf` on
  the diagonal.
* A test steps 1000 satellites x 1000 targets under `vmap` and asserts finite output.

## Issue 17 - Visibility limits

**Decision.** A target is accessible (observable and imageable) when it is active, on the
near side of the horizon, within `+-max_squint_deg` of broadside, and at an incidence
angle inside `incidence_deg` -- by default 20 to 45 degrees, the band a spaceborne SAR
actually works in. The region is therefore a pair of annular sectors, one either side of
the ground track, with a hole over nadir where a side-looking sensor cannot image.

Both edges are real constraints, not conveniences: inside about 20 degrees the nadir
return dominates and backscatter contrast collapses, and beyond about 45 the slant range
and shadowing make the imagery poor. The horizon test remains a closed form on the nadir
component of the line of sight, and the band test is written so every comparison is
squared -- no square roots, no inverse trigonometry on satellite-target pairs.

**Consequences.** The accessible area per satellite is about 404,000 km^2, 0.079% of the
globe, reaching 183 to 488 km either side of the ground track at zero squint. A target
in the nadir hole is invisible even when the satellite passes directly overhead, which is
the single biggest behavioural difference from a naive "point anywhere below you" model:
the greedy baseline images 0.54 of the targets rather than the 0.76 it managed when
nadir was allowed. Backscatter and resolution still do not vary with incidence inside the
band; that, and the fact that switching sides is instantaneous, are in IMPROVEMENTS.md.

## Issue 18 - Agent IDs and global state in MAPX

**Decisions.** `sarsat.yaml` sets `implicit_agent_id: True` so MAPX does not append a
one-hot ID: agents are homogeneous with egocentric observations, and an ID would add
1000 features at full scale. For centralised-critic systems the wrapper's global state
is the joint observation, as in the CAMAR wrapper.

**Consequences.** The joint observation is `N * obs_dim` wide per agent, which is fine
for tens of satellites and unusable for hundreds. Use the independent-critic systems
(`rec_ippo`, `rec_imaspo`, `rec_imars`) for large constellations; a pooled global state
is in IMPROVEMENTS.md. This mirrors the limitation MAPX's README already documents for
AeroJAX.

## Issue 19 - Satellite count cap

**Decision.** `num_satellites` must be in `[1, 1000]`, as specified. The cap is a guard,
not a technical limit: memory per environment is dominated by
`(lookahead_steps + 1) * N * M * 3` floats (~36 MB at 1000 x 1000) and the N x N
neighbour search (~16 MB).

## Issue 20 - Evaluation logging and viewer

**Question.** How should a seeded episode be logged for inspection without slowing or
complicating the environment?

**Options.** (a) Return per-step diagnostics (which beam hit which target) in
`TimeStep.extras`, which every MAPX rollout would then carry through `lax.scan`. (b) Keep
the environment as is and let an evaluator recompute what it needs from the state.

**Decision.** (b). `step` now delegates action decoding to `_apply_action`, and
`sarsat.evaluate.run_episode` calls that same function inside one jitted
`act_and_step` to obtain pointing, honoured sensing flags and the beam-hit matrix,
alongside the real transition. The loop over steps runs on the host (an episode is a
few hundred steps; logging is I/O-bound anyway), positions and beam footprints are
derived in float64 NumPy by intersecting each look ray with the sphere, and the logs go
to four plain CSVs. The viewer is a single HTML file with the data and the 1-degree land
mask embedded and drawn on a canvas - no map tiles, libraries or server - so it opens
from disk on any machine and can be attached to a report.

**Consequences.** The environment's hot path is unchanged (tests and benchmark
unaffected). Because a target can sit in several beams in the same step, `captures.csv`
holds one row per (target, satellite) pair while `targets.csv` records the first capture
by the lowest satellite index; the viewer counts unique targets. Footprints straddling
the dateline are drawn on both sides; segments and footprints that jump more than 60
degrees of longitude in one step (pole crossings) are not drawn as lines. The HTML size
scales with steps x satellites, which is fine up to a few hundred satellites for a full
episode and about 1 MB per 8 steps at 1000.

## Issue 21 - Two-sided access and the action encoding

**Question.** The accessible region is two disjoint sectors with a hole between them. How
does a bounded action address it?

**Options.** (a) A binary categorical slot for the side plus a continuous slot sweeping
the band. (b) Signed encoding: one continuous slot whose sign picks the side and whose
magnitude sweeps the band. (c) One continuous slot across the whole cross-track ground
line, with the nadir hole clipped to its nearest edge.

**Decision.** (a), with (b) retained behind `side_switch=False` so the two can be
compared in training. The first implementation was (b); it was replaced after weighing
what each parametrisation can represent:

* Under (b) the seam sits at zero, exactly where a tanh-Gaussian head initialises, and
  the two decisions share one number. "Image whatever is at the inner edge on whichever
  side it is on" -- a natural strategy, since the inner edge is the shortest range --
  needs a mean near zero *and* a decisive sign, which a unimodal distribution cannot
  give; a Gaussian at mean 0.02, std 0.05 picks the right side about 65% of the time.
  Worse, once the policy sharpens to, say, mean 0.3, std 0.1, it never samples the other
  side again: the side decision is frozen by the magnitude decision. Concretely, (b)
  cannot even express "left, inner edge", because -0.0 == +0.0 (a test skips that case).
* Under (a) the Bernoulli over side carries its own entropy, so the policy can be sharp
  about incidence and undecided about side or the reverse; exploration noise on the
  continuous slot never flips sides by accident; and initialisation gives p = 0.5 on side
  with mid-band incidence, a sensible prior rather than a seam. It also matches the
  physics -- on a real SAR the look side is a discrete mode changed by a roll manoeuvre --
  and charging for a switch, or forbidding it on consecutive steps, becomes a mask on a
  categorical slot, which `HybridActionHead` already honours.
* (c) looks smoother but is not: the outcome still jumps at zero, and 37% of the range
  becomes a dead band that clipping turns into a point mass at each inner edge.

Under both encodings the cross-track range is computed *after* the squint command,
because squint and cross-track look combine into the off-nadir angle the band
constrains (`cross_track_band`). At 25 degrees of squint the beam is already past the
inner edge and the available range starts at zero; at zero squint it runs the full 18.4
to 40.6 degrees. The observation encodes each target with the same `_encode_pointing`
the action decodes with, which is what keeps "copy the slot into the action" exact, and
`greedy_policy` is literally that copy for either layout.

**Consequences.** Action width 4 instead of 3 and 96 observation features instead of 76
by default. The greedy baseline's looks and captures are byte-identical under the two
encodings (only the idle pointing logged on non-sensing steps differs), so the change is
to the interface, not the geometry. No training evidence yet distinguishes them -- the
PPO smoke tests predate the nadir hole -- and that side-by-side is the open question the
flag exists for.

## Issue 22 - Cooperation pressure

**Question.** This is a multi-agent benchmark, so the scenario has to reward
coordination: a cooperating team should beat a team of independent greedy satellites by a
margin worth training for. For a given constellation size (N = 100 as the working
example), what target distribution makes that true?

**Method.** Rather than reason about it, measure it. `scripts/cooperation_gap.py` runs
four reference policies through the real environment -- same beam, battery and access
rules, only the *choice* of target differs:

* `greedy_beam` -- independent and beam-aware: aim where the beam captures the most priority,
  sense whenever the battery allows. This, not "nearest target", is the fair baseline:
  the naive nearest-to-nadir greedy aims at the inner edge of the band, where half the
  beam falls in the nadir hole, and loses 30-45% to *aiming* alone. Attributing that to
  cooperation would be a mistake.
* `solo_plan` -- independent and far-sighted: rations looks against its own budget and
  weighs a target by how rarely *it* will see it again. Temporal planning with no
  knowledge of teammates.
* `coop_dedup` -- centralised: no target is taken twice in one step.
* `coop_plan` -- centralised: dedup, plus scarcity weighted by *every* satellite's future
  access ("leave what a teammate will cover, spend on what only you reach"), plus the
  budget rule.

The gap is `coop_plan` over the better independent policy. The centralised policies see
the whole constellation's future, so they are a yardstick for what coordination is worth,
not an upper bound a learner must reach.

**Findings** (100 satellites, 180 steps, `max_targets` 8000, one seed unless noted):

| Constellation | Targets | greedy_beam | solo_plan | coop_dedup | coop_plan | gap |
|---|---|---|---|---|---|---|
| random orbits | uniform 8000 | 0.666 | 0.661 | 0.673 | 0.698 | +5% |
| random orbits | hotspots | 0.822 | 0.846 | 0.826 | 0.923 | +9% |
| 10 Walker planes | hotspots (3 seeds) | 0.79-0.82 | 0.82-0.83 | 0.80-0.82 | 0.90-0.93 | +9-12% |
| 10 trains, 60 s spacing | hotspots | 0.731 | 0.738 | 0.746 | 0.787 | +7% |
| 50 pairs, 3 s spacing | hotspots | 0.571 | 0.655 | 0.790 | 0.922 | +41% |
| random orbits | 40 clusters, no priority contrast | 0.924 | -- | 0.926 | 0.922 | 0% |

"Hotspots" = 6000 uniform background targets plus 40 clusters of 50 targets (100 km
sigma) on land, each hotspot target worth 10 background targets.

Three things the table says. First, *what* the targets are matters less than *whether the
battery binds and priorities differ*. Dense uniform targets give +5%; clustered targets
with equal priority give nothing, because once every satellite aims well there is nothing
left to coordinate over -- each satellite images whatever it passes and leftovers flow to
the next. Hotspots give ~10% with any constellation, because a satellite that spends its
charge on background targets is not there for the hotspot it alone will pass later, and
knowing which hotspots teammates will cover is what tells it which to skip. Second, that
gain is cooperative, not merely far-sighted: `solo_plan`, which plans just as far ahead
using only its own future, recovers almost none of it (0.824 vs 0.899 on Walker). Third,
same-step overlap is the other lever, and it is a constellation property: formation pairs
sharing one view make an independent policy waste looks on duplicates, and dedup alone
is worth +40%. Trains one step apart do *not* create it -- the follower simply sees the
target already gone.

**Decision.** Two things are built in. The target field gains `hotspots`,
`hotspot_targets`, `hotspot_radius_km` and `hotspot_weight` (`_sample_target_field`);
zero hotspots reproduces the previous uniform field exactly. The constellation gains
`planes` and `train_spacing_s` (`sample_planes`): Walker planes for overlapping coverage,
trains, or a few seconds' spacing for formation flyers; zero planes reproduces the
previous independent random orbits. The recommended training scenario is
`sarsat-100sat-hotspots` (10 Walker planes, hotspots), where cooperation is worth about
10%, and `sarsat-100sat-pairs` is the coordination stress test at +40%. The cooperation
gap script stays in the repository so any future change to the sensor or battery model
can be checked against the same yardstick.

**Consequences.** Hotspot slots come first in the target arrays so `reset(num_targets)`
trims background, never hotspots. A 10% margin is enough to train for but not enormous;
the levers to widen it without formation flying are a tighter battery (more binding), a
larger hotspot weight, or fewer, more scattered planes. Time-windowed targets would widen
it a great deal more, at the cost of the "targets persist" simplification (IMPROVEMENTS.md).
The reference policies are heuristics: `coop_plan`'s budget threshold uses optimistic
future values (it ignores what teammates take first) and is damped by 0.5 to compensate;
a proper assignment solver would raise the yardstick somewhat.

## Issue 23 - An observation and architecture a MARL learner can use

**Question.** Issue 22 built a benchmark where cooperation is worth something
(`sarsat-100sat-hotspots`: 100 satellites, 10 Walker planes, 6000 background targets plus
40 hotspots of 50 targets each) but never trained anything on it. Can IPPO or MAPPO
(MAPX's `rec_ippo` / `rec_mappo`) actually close that gap against `greedy_beam`, with the
reward left alone -- shared team reward, no shaping beyond a constant `reward_scale=10`
so returns sit at O(1-10) for the value loss? (`SarSatCoopWrapper` also carries a
`credit_mix` knob that blends in each satellite's own share of the team reward; it
defaults to 0. The findings below report runs with and without it, and say which is
which -- it turned out to matter a great deal.)

**Method.** First check whether the base `SarSat` observation gives a learner enough to
work with. It has 10 target slots -- the accessible targets nearest to nadir, tracked two
steps ahead -- with no notion of priority or how much a beam actually captures, and no
information about teammates beyond bearing and range. That is not what separates the
reference policies: `greedy_beam` wins over a naive nearest-target greedy by being
beam-aware (aiming for the beam that captures the most priority, not the nearest point),
and `coop_plan` wins over every independent policy by rationing battery toward hotspots
using *every* satellite's future access to them, not just its own. Neither quantity is in
the base observation, and a satellite cannot approximate "how many looks will the team
still get at this cluster" from nearby bearings alone.

So `sarsat/coop.py` (`CoopSarSat`) replaces the observation only -- the dynamics, action
and reward are exactly `SarSat`'s. Its `agents_view` (see the module docstring for the
exact layout) has `num_slots` ranked beams found by non-maximum suppression over the
`num_candidates` most valuable accessible targets: slot 0 is the beam `greedy_beam` would
aim at, slot 1 the best over what slot 0 leaves, and so on. Each slot carries the beam's
raw priority, the same priority discounted by a scarcity factor
`kappa / (kappa + team_future_accesses)` for its hotspot cluster (so a beam over a cluster
the team will see many more times is worth less to take now), the team's remaining access
count itself, and contention -- how many other charged satellites can see the same target
right now. A `horizon`-step look-ahead tracks what is left of every hotspot cluster this
satellite will still reach. Every one of these is something a satellite could compute
on board from the public ephemeris and the shared target catalogue (which the base
observation already assumes); nothing here is a hidden oracle. Values are normalised with
a log squash (`log1p(x) / log1p(scale)`) so a raw count of, say, 50 future team accesses
does not swamp a `[-1, 1]`-scaled pointing feature. For the critic, `global_state` is the
agent's own view followed by a fixed ~64-number team summary (battery quantiles, value
remaining in hotspots vs background, the sorted per-cluster remaining fractions, the
team's averaged look-ahead) rather than the joint observation every other MAPX wrapper
uses (issue 18): on `sarsat-100sat-hotspots` that is 133 numbers per agent instead of
9,600 (100 satellites x the 96-wide base `agents_view`), which is what makes `rec_mappo`
usable at this scale at all.

As a sanity check, a policy that always acts on slot 0 alone (`action = concat(view[:,
1:4], view[:, 0:1])`, the same "copy the slot" rule the base environment's greedy baseline
uses) should reproduce `greedy_beam` on the same seed, since slot 0 is defined as the
beam-maximising choice. It does, up to the sampling noise of building the candidate set
from `num_candidates=64` targets rather than every accessible one
(`scripts/coop_check.py`):

| Seed | Slot-0 policy | `greedy_beam` |
|---|---|---|
| 1000 | 0.8008 | 0.7989 |
| 1001 | 0.8193 | 0.8202 |
| 1002 | 0.7772 | 0.7734 |
| 1003 | 0.7056 | 0.7091 |

The action space is unchanged. The network (`mapx.networks.sarsat`) is a shared-parameter
recurrent actor built to exploit the structure of the new observation rather than treat
it as a flat vector: `SarSatTorso` runs one shared encoder over the ranked slots and
another over the look-ahead sequence before the rest of the MLP, `AnchoredRecurrentActor`
keeps a skip connection from the pre-torso embedding around the GRU (128 units) into the
post-torso, and `AnchoredHybridHead` anchors the Gaussian mean to `atanh(slot 0's
pointing)` plus a learned correction, and leans the side logit toward slot 0's side by a
fixed amount -- so the policy starts out imitating the beam-aware greedy baseline and
learns a correction and a sensing/side policy on top, rather than discovering "aim
somewhere reasonable" from a reward that only pays off after committing to a target. The
initial Gaussian std is 0.15: tight enough that early rollouts stay close to the anchor,
loose enough to explore departures from it. Whether to sense at all remains entirely
learned.

Throughput needed separate work before any of this was trainable at 100 satellites x 8000
targets. MAPX's `AutoResetWrapper` runs a full `reset` on every step under `vmap`
(cheap at small scale, not here: this scenario's `reset` is as much geometry as
`observe`), so `SarSatCoopWrapper` fuses the reset into `step` itself
(`fused_auto_reset=True`, paired with `PassthroughAutoReset` in place of MAPX's wrapper)
and builds the next episode's cluster-access schedule three rows per step over the
running episode so it is always ready before the hand-over. Geometry (`body`, `visible`)
that both the observation and the transition need is computed once per step and cached on
`CoopState` instead of twice. Selecting the `num_candidates` best beams used to be a
`top_k` over all 8000 targets; because hotspot targets occupy the first slots by
construction (issue 22), "the first `num_candidates` visible targets in slot order" is
already the most valuable set, so it is now a cumulative-sum plus `searchsorted`, and
`TargetSampler`'s own `searchsorted` was switched to `method="sort"`
(`base_state`: 30 ms -> 7 ms for 32 environments, `scripts/coop_profile.py`). Measured
together, with 32 environments while two other trainings shared the same GPU, the wrapped
environment went from about 440 to about 3,000 env steps/s. WSL2's allocator caps a
single GPU allocation near 4 GB regardless (`XLA_PYTHON_CLIENT_ALLOCATOR=platform` reaches
it; the default BFC allocator cannot), which is what limits a run to 32 environments x
180-step rollouts rather than raw compute.

**Findings.** Reference-policy means over seeds 1000-1015 on `sarsat-100sat-hotspots`
(`runs/baselines/*.jsonl`, `scripts/baseline_seeds.py`):

| Policy | Mean return |
|---|---|
| `greedy_beam` | 0.786 |
| `solo_plan` | 0.807 |
| `coop_plan` | 0.904 |

Trained policies, greedy (mode) actions, same 16 seeds (`eval_checkpoint.py`; every run
is 32 envs x 180-step rollouts, shared parameters, one training seed, so differences of
about 0.005 are within noise). Each row is the run's best checkpoint by MAPX's own
32-episode evaluation, re-scored on the paired seeds:

| Run | System | Head | Reward seen by the learner | Updates | Mean return | Beats `greedy_beam` |
|---|---|---|---|---|---|---|
| `mappo_v1` | `rec_mappo` | slot-0 anchor | team | ~350 | 0.847 | 16 / 16 |
| `mappo_v2` | `rec_mappo` (gamma 0.999, LR decay) | slot-0 anchor | team | 630 | ~0.80 (MAPX eval; plateaued) | -- |
| `ippo_v2` | `rec_ippo` | slot-0 anchor | team | 500 | ~0.74 (MAPX eval; plateaued) | -- |
| `mappo_v3` | `rec_mappo`, continued from `mappo_v1` | slot-0 anchor | `credit_mix=0.5` | +360 | 0.890 | 16 / 16 |
| `ippo_v3/v4` | `rec_ippo`, from scratch | slot-0 anchor | `credit_mix=0.5` | 360 + 360 | 0.894 | 16 / 16 |
| `mappo_v5` | `rec_mappo`, from scratch | slot mixture, ranked contention | `credit_mix=0.5` | 360 of 1200 | **0.898** | 16 / 16 |
| `ippo_v5` | `rec_ippo`, from scratch, LR decay | slot mixture, ranked contention | `credit_mix=0.5` | 600 | 0.895 | 16 / 16 |

Four things the runs say.

*The observation and the anchored head are enough to beat `greedy_beam` with the pure team
reward.* `mappo_v1` starts at `greedy_beam`'s level (its mode action *is* slot 0) and
learns battery rationing on top: 0.847, above `solo_plan` too. With the pure team reward
the centralised critic matters: `rec_ippo` with a local critic stalled below
`greedy_beam` (0.74), because one satellite's value function cannot explain a reward that
is 99% other satellites' doing, and its advantages are mostly noise.

*What the team reward cannot teach at N = 100 is the value of a cheap look.*
`diagnose_policy.py` on `mappo_v1`: it images 95% of hotspot priority (`coop_plan`: 96%)
but only 49% of the background (`coop_plan` and `greedy_beam`: 70%), ending episodes with
half its battery unspent. One background target is 1/26000 of the return; with 100 agents
sharing one reward, that signal is far below the noise, so the policy learns "sense on
hotspots" and nothing finer. Blending in each satellite's own share of the team reward
(`credit_mix=0.5`: `0.5 * team + 0.5 * N * own_share`, whose mean over agents is still the
team reward, and with simultaneous captures split evenly so duplicates earn less) fixes
exactly that: background coverage rises to 66%, `rec_ippo` reaches 0.895 on MAPX's
evaluation within 30 updates from scratch, and both systems settle at `coop_plan`'s level
on MAPX's random evaluation episodes (0.90-0.91) and within 0.006-0.014 of it on the
paired seeds. This is credit assignment, not shaping in the potential-based sense -- no
new term rewards any behaviour the team reward does not -- but it is a departure from the
literally shared reward, so both are reported. With it, the centralised critic no longer
helps: IPPO and MAPPO are indistinguishable.

*The remaining gap is same-step duplication.* On `ippo_v3` hotspot coverage equals
`coop_plan`'s, background is 66% vs 70%, and the priority-weighted duplicate captures are
about 1.4% of the return where `coop_plan` has none; the slot-0 head never aims anywhere
but slot 0 (0% of looks match slots 1-3), so two satellites over one hotspot take the same
beam. `SlotMixtureHead` makes the slot a latent categorical choice marginalised in the
log-probability (the action vector is unchanged), and `ranked_contention=True` counts only
lower-indexed charged teammates, a convention under which exactly one contender sees no
competition. `mappo_v5` with both reached the best paired-seed score (0.898) at 360
updates and then held 0.895-0.91 on MAPX's evaluation for its remaining 840 updates: a
small gain over the slot-0 head, inside what one training seed can resolve.

*Constant learning rates drift after the plateau.* `ippo_v4` fell from 0.907 to 0.87 on
MAPX's evaluation over its last 150 updates; MAPX keeps the best checkpoint, which is what
is scored above, but `system.decay_learning_rates=True` is the better default.

**Decision.** Ship `CoopSarSat`, `SarSatCoopWrapper` and the anchored network as the
recommended path for training on `sarsat-100sat-hotspots`, alongside `train_sarsat.py` /
`run.sh` (mapx_integration/README.md documents the override set used for the runs above)
and `eval_checkpoint.py` for scoring a checkpoint against the same seeded baselines the
reference policies use (`SARSAT_HEAD=mixture` selects the slot-mixture head in both). The
environment's reward stays the shared team reward and `credit_mix` stays off by default,
so the pure-team-reward result (0.847) is what the defaults reproduce; `+env.credit_mix=0.5`
is the documented, opt-in way to reach `coop_plan`'s level.

**Consequences.** `CoopSarSat` requires `side_switch=True` (it raises otherwise) since
the anchor needs a well-defined side feature; the signed encoding is not supported here.
The global-state saving (133 vs 9,600 numbers) is specific to this observation design and
does not generalise to `SarSatWrapper`'s plain joint-observation global state. The
fused-reset throughput work assumes episodes end on the time limit in practice (an
environment metric, `fraction_imaged`, still reports the true unscaled return), matching
`bootstrap_truncation=False`. The 32-environment ceiling is a WSL2/single-GPU artefact of
this development machine, not a property of the environment or the wrapper; a dedicated
GPU or a Linux host without the allocator cap should train with far larger batches -- and
larger batches are the principled alternative to `credit_mix` for pulling a 1/26000 signal
out of the team reward. WSL2 host memory is as tight as the GPU: three CPU evaluation
workers beside two trainings exhausted its 15 GB and took the VM down, so side jobs run
one process at a time. All results are one training seed per configuration; "beats
`greedy_beam`" is solid (16 / 16 seeds, +0.06 to +0.11), "matches `coop_plan`" holds to
within about 0.006 and wants more seeds before anyone leans on it.

## Issue 24 - A 200-satellite scenario where cooperation is worth something

**Question.** `sarsat-100sat-hotspots` gives a centralised planner about +12% over the
best independent policy (issues 22, 23). Is there a 200-satellite scenario with a
*significant* gap, and which of the environment's levers produces it honestly?

**Method.** `scripts/scenario_sweep.py` runs `greedy_beam`, `solo_plan` and `coop_plan`
over a table of named configurations (one JSON line per result in `runs/sweep/`). Two gaps
matter: `coop_plan` over `greedy_beam` (what the benchmark advertises) and `coop_plan`
over `solo_plan` (the part only reachable by *coordinating*: `solo_plan` plans just as far
ahead using its own future, so anything it recovers is temporal planning, not
cooperation). Everything below is 8000 targets, 180 steps, seed 0 unless noted.

**Findings.**

*Doubling the constellation alone removes the gap.* 200 satellites over the 100-satellite
target field image 0.92-1.00 of the priority under `greedy_beam` (20 or 10 Walker planes,
40 or 80 hotspots, weight 10 or 30): nothing is left to coordinate over. A tight battery
(10% duty cycle) restores +4%; 90-step episodes +8%.

*Formation pairs give a large gap (+42% at 90 steps, +96% with a tight battery) but were
rejected*: two satellites 3 s apart share one view, so the entire gain is same-step
deduplication forced by geometry rather than by any decision the scenario poses. The two
scenario files written for them were removed; the sweep rows remain in `runs/sweep/`.

*Time windows are the honest lever, but only in one form.* `sarsat/windows.py` adds
`WindowedSarSat` / `WindowedCoopSarSat`, built on the existing classes without changing
them: every target carries a `(first, last)` step window and `_target_geometry` gates on
it, so the base observation, `step` and all four reference policies respect windows with
no further change (and `window_steps=(0, 0)` is bit-identical to `SarSat`; the windows
are sampled from a key folded off the reset key, so orbits and targets are unchanged).
Windowing *everything* fails: at 100 satellites only 35% of the priority is reachable at
all, satellites idle between windows, the battery never binds and the gap is +2-5%. What
works is *events over a persistent background*: hotspot clusters become events, each open
for one 10-30 step window shared by all of its targets, while the 6000 background targets
stay available all episode. The background tempts a greedy satellite to spend charge it
will need for the event that only it can reach in time, and knowing that it is the only
one is exactly what the centralised planner has and an independent policy does not.

| Scenario (200 satellites, 20 planes, 40 event clusters, background persistent) | `greedy_beam` | `solo_plan` | `coop_plan` | gap |
|---|---|---|---|---|
| events, default battery | 0.72 | 0.72 | 0.79 | +9% |
| events, 100-target clusters | 0.67 | 0.67 | 0.71 | +6% |
| events, 10 planes | 0.57 | 0.57 | 0.60 | +6% |
| events, 90 steps (3 seeds) | 0.49-0.51 | 0.50-0.52 | 0.65-0.68 | +28-32% |
| **events, 10% duty cycle (3 seeds)** | **0.43-0.50** | **0.43-0.50** | **0.67-0.75** | **+49-56%** |
| events, 90 steps and 10% duty (3 seeds) | 0.28-0.31 | 0.37-0.39 | 0.60-0.61 | +96-115% |

In every event scenario `solo_plan` equals `greedy_beam`: the gap is cooperative through
and through. The 100-satellite version (`ev100`) shows the same shape at +32%.

**Decision.** `sarsat-200sat-events` is the 200-satellite benchmark: 20 Walker planes,
6000 persistent background targets, 40 event clusters with 10-30 step windows, and
`recharge_rate=0.01` (10% duty cycle). Three seeds give `greedy_beam` 0.43-0.50 and
`coop_plan` 0.67-0.75, none of it recoverable by solo planning. The 90-step variant is
the cheaper choice if compute binds (+28-32%). The combined variant was not adopted: a
third of its gap is temporal planning and `greedy_beam` at 0.3 is too weak a baseline to
mean much. For training, `make_sarsat_coop_envs` picks `WindowedCoopSarSat` whenever the
scenario sets `window_steps`; its beam slots carry a ninth feature (fraction of the
target's window still open) and the cluster look-ahead is gated on the cluster windows,
so the networks' `slot_dim` is set from the environment. `CoopSarSat`'s own eight-feature
layout is untouched and the issue-23 checkpoints still load.

**Consequences.** The state gains a field only in the windowed subclasses
(`WindowedState`, `WindowedCoopState`), so nothing that consumes `State` changes.
`SarSat.reset` was split into `_initial_state` plus the observation, a behaviour-neutral
hook the subclasses extend. At 200 satellites a training run fits 16 environments x
180 steps on this GPU (the WSL2 cap of issue 23), and the untrained slot-0 policy
evaluates at 0.45-0.48 on the new scenario, i.e. at `greedy_beam`'s level, as it should.
No learner has been trained on it yet.

## Issue 25 - Overnight results: what the learners reach on both benchmarks

**Question.** With a full GPU for a night: can a learner match or beat `coop_plan` on
`sarsat-200sat-events` (issue 24), and can it beat `coop_plan` on
`sarsat-100sat-hotspots` (issue 23)?

**Method.** Every run uses the issue-23 recipe (`mapx_integration/launch.sh`: slot-mixture
head, `credit_mix`, `ranked_contention`, LR decay) unless a variant says otherwise. Every
score below is the run's best checkpoint by MAPX's own evaluation, replayed with greedy
actions on the paired test seeds 1000-1015 (`eval_checkpoint.py`, tables in
`reports/results.json`, deck in `reports/sarsat_marl_results.html`). The 100-satellite
runs were also scored on held-out seeds 2000-2015 (`runs/baselines/hotspots100_val.jsonl`
holds the references) so that picking the best of several runs does not lean on the
test seeds. Reference rows come from `scripts/baseline_seeds.py`
(`runs/baselines/summary_all.md`).

**Findings.**

| `sarsat-100sat-hotspots`, seeds 1000-1015 | mean | std |
|---|---|---|
| Random / naive Greedy / `greedy_beam` / `solo_plan` / `coop_dedup` | 0.067 / 0.452 / 0.786 / 0.807 / 0.789 | |
| `coop_plan` | 0.904 | 0.015 |
| MAPPO, team reward (issue 23) | 0.847 | 0.021 |
| MAPPO, credit_mix 0.5, slot mixture (`mappo_v5`, seed 42) | **0.898** | 0.017 |
| same recipe, seeds 11 / 23 | 0.894 / 0.896 | |
| 6 slots, 24-step look-ahead (`h100_wide`) | 0.898 | |
| `mappo_v5` fine-tuned at credit_mix 0.25 | 0.896 | |

Five independent runs land at 0.894-0.898. On the validation seeds `mappo_v5` scores
0.909 against `coop_plan`'s 0.914 and beats it on 2 of 16: the same 0.005-0.006 margin.
The learners are level with `coop_plan` within noise and do not beat it; a wider
observation, more seeds and a lower-`credit_mix` fine-tune all return the same number,
so this is the ceiling of the recipe, not seed luck.

| `sarsat-200sat-events`, seeds 1000-1015 | mean | std |
|---|---|---|
| Random / naive Greedy / `greedy_beam` / `solo_plan` / `coop_dedup` | 0.025 / 0.279 / 0.440 / 0.450 / 0.453 | |
| `coop_plan` | 0.670 | 0.047 |
| IPPO, credit_mix 0.5 (`ev_ippo_a`) | 0.646 | 0.045 |
| MAPPO, credit_mix 0.5 (`ev_mappo_a`) | 0.645 | |
| MAPPO, credit_mix 0.1, look-ahead 48 (`ev_mappo_c`) | 0.641 | |
| MAPPO, credit_mix 0.1 (`ev_mappo_d`) | 0.648 | |
| MAPPO, credit_mix 0.25, gamma 0.998, lambda 0.98 (`ev_mappo_e`) | **0.649** | 0.048 |

Every learner beats `greedy_beam` on 16 of 16 seeds by +0.20 (+47%) and reaches 96-97% of
`coop_plan`; none matches it. MAPX's evaluations are paired across runs (the same episode
keys at each evaluation index), and at matching indices the five runs agree to within
0.005, so the plateau is robust to the critic (IPPO = MAPPO), the credit mix (0.1-0.5),
the look-ahead (16 or 48 steps) and the GAE horizon (18 or 67 steps). The diagnosis
(`diagnose_policy.py` on `ev_ippo_a`, seeds 1000 and 1009) says where the 0.02 is:
`coop_plan` images 68% of event priority to the learner's 64% and *less* background (84%
vs 87%), because it arrives at event windows with battery 0.77 where the learner has 0.37,
and it wastes 3.5 priority units on same-step duplicates where the learner wastes 38.
The learner has all the information (`ahead_value` gated on the windows, contention), so
what it lacks is either the exploration to discover "idle for 30 steps now" or a way to
resolve same-step conflicts that a shared convention on indices does not give it.

**Decision.** Report both benchmarks as "beats `greedy_beam` decisively, ties or trails
`coop_plan` slightly": 0.898 vs 0.904 and 0.649 vs 0.670. `mappo_v5` and `ev_mappo_e`
are the reference checkpoints (`~/marl/snapshots/`, copies under `runs/marl/`). The
credit-mixed reward is disclosed on the deck's shaping slides; the pure-team-reward
result (0.847) stays in the table.

**Consequences.** Two GPU casualties taught a rule now in `mapx_integration/README.md`:
on this WSL2 machine a second training or evaluation process next to a run that holds
more than ~8 GB kills the first with `cudaErrorUnknown`, so runs at 200 satellites go
alone or in pairs of 12 environments. MAPX keeps the best checkpoint *by its own noisy
evaluation* (12-24 episodes), which at 200 satellites picked step-195 checkpoints from
runs that trained ten times longer; a validation-seed selection over saved checkpoints
would be the next improvement to the tooling. For the environment, the honest next levers
on the windowed benchmark are an explicit conflict-resolution channel (an intent broadcast
or a turn-taking token) and larger batches on a Linux host without the allocation cap.
