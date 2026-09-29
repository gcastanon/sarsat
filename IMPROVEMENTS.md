# Possible improvements (deliberately not implemented)

The brief was a basic, fast model. Everything below was considered and left out to keep
the environment small; each item notes where it would plug in. Issue numbers refer to
[DECISIONS.md](DECISIONS.md). When an item gets built, move it out of this file and into
the DECISIONS issue that builds it.

Already built, for reference: Walker-style planes (`sample_planes`, issue 22),
time-windowed event targets (`sarsat/windows.py`, issue 24), a compact global state for
centralised critics (`CoopSarSat`, issue 23), per-satellite credit assignment
(`credit_mix`, issue 23), trained-policy evaluation (`mapx_integration/eval_checkpoint.py`)
and GPU throughput measurements (issues 23 and 27).

## Orbital mechanics (issue 2)
* **J2 nodal precession.** Adds a constant rate to the node angle - a one-line change in
  `satellite_frames` (`node = raan + (raan_rate - omega_earth) * t`). Needed for
  sun-synchronous behaviour over multi-day episodes.
* **Eccentric orbits, drag, manoeuvres.** Require solving Kepler's equation or
  integrating; would break the closed-form look-ahead.
* **Sun-synchronous planes.** `planes=` gives Walker-style planes with random
  inclinations; a sun-synchronous family (inclination tied to altitude, J2 above) would
  change only `sample_planes`.
* **Ground-relative flight direction.** The along-track axis ignores the ~0.5 km/s
  contribution of Earth rotation (issue 3).

## Sensor and imaging (issues 4, 5, 7, 17)
* **Look-side switching costs nothing.** The sensor is two-sided and a satellite may look
  left on one step and right on the next. A real spacecraft rolls to change side, which
  takes minutes and rules out alternating every step; fixing the side per pass, or
  charging a slew penalty, would be more honest. The machinery is a per-satellite sign
  held in the state.
* **Incidence does not affect image quality.** Access is a hard band with uniform reward
  inside it, but backscatter, ground-range resolution (slant resolution over
  `sin(incidence)`) and shadowing all vary strongly across 20 to 45 degrees. Scaling the
  reward by an incidence-dependent quality factor is a one-line change in `step`.
* **No range or azimuth ambiguities, no NESZ.** The PRF constraints that in practice
  couple swath width to along-track sampling are absent.
* **Swept footprint / dwell.** Integrate the beam over the 60 s step (strip-map) instead
  of a snapshot, or require a minimum dwell for a valid image.
* **Slew-rate limits and settling time.** Pointing is instantaneous; a limit would add
  the previous pointing to the state and the observation.
* **True rectangular footprint.** The beam is a box in az/el space, whose azimuth extent
  shrinks by `cos(el)` away from nadir.
* **Image quality.** Reward scaled by range, incidence angle or resolution; probabilistic
  detection; imaging modes with different swath / energy trade-offs.
* **Oblate Earth and terrain.** The Earth is a sphere of radius 6371 km and targets sit
  at sea level.

## Power (issue 12)
* **Eclipse-aware charging.** Recharge only in sunlight (needs a sun vector and a
  cylindrical shadow test).
* **Idle and slewing loads, battery depth-of-discharge limits, thermal duty limits.**

## Targets and reward (issues 9, 10, 11)
* **Priorities in the plain `SarSat` observation.** Hotspots and events make priorities
  non-uniform, but only `CoopSarSat` observes them (through beam value). The plain
  observation could add `priority * num_targets` as a fourth feature per target slot and
  rank slots by priority.
* **Moving and revisit targets.** Time-windowed events exist (issue 24); targets that
  move (ships) or need repeated looks do not.
* **A real assignment solver as the cooperation yardstick.** `scripts/cooperation_gap.py`
  uses greedy heuristics for the centralised reference; a rolling-horizon ILP over
  (satellite, step, target) would give a tighter upper bound.
* **Finer land mask / regional target distributions.** The mask is 1 degree; targets are
  uniform by area within land and within water.
* **Reward shaping.** The environment pays only the shared team reward (the MAPX wrapper's
  `credit_mix` blends in each satellite's own share, issue 23); there is no energy penalty
  for sensing empty ground beyond the battery itself.

## Observation and scaling (issues 13, 14, 18, 19)
* **Spatial hashing.** Target geometry is dense O(N x M) and the neighbour search dense
  O(N^2). Bucketing targets and satellites on a lat/lon grid would cut both for
  N, M >> 1000, at a real cost in code complexity.
* **Caching look-ahead geometry between steps.** Each step recomputes `lookahead_steps + 1` geometry
  evaluations; a rolling buffer in the state would need only one but multiplies state
  memory by `lookahead_steps * N * M`.
* **Unit-vector neighbour encoding.** Replace neighbour (az, el) by a 3-D unit vector to
  remove the wrap at az = +-pi.
* **Earth-blocked neighbours / communication model.** Neighbours are purely geometric;
  there are no inter-satellite links, ground stations, downlink or data storage.
* **Scalable global state for the plain wrapper.** `CoopSarSat` has a compact global
  state (issue 23), but the plain `SarSat` MAPX wrapper still concatenates every agent's
  observation, which limits its centralised critics to tens of satellites.
* **Observation of absolute position / time of day.** Omitted to keep the view strictly
  egocentric.

## Tooling (issue 20)
* **Viewer at very large scale.** The HTML embeds the full log as JSON; a binary
  (typed-array) encoding or loading the CSVs from a local server would cut a 1000
  satellite x 180 step file from tens of MB to a few.
* **Viewer projections and detail.** Equirectangular only, 1-degree coastlines, no zoom.
  A globe or a zoomable tile map would need an external library.
* **Parameter snapshot tool.** The live demo (`sarsat-live`) loads a 5 MB msgpack snapshot
  of the actor, which was extracted from a MAPX checkpoint on the training host by a short
  orbax script that is not in this repository.
