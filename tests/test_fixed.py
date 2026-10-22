import jax
import numpy as np
import pytest

from sarsat.fixed import FixedCoopSarSat, FixedSarSat, load_scenario
from sarsat.reference import ReferenceController

SCENARIO = "scenarios/adversarial_pair.json"


def _return(env, policy):
    state, _ = env.reset(jax.random.PRNGKey(0))
    controller, total = ReferenceController(env, state, policy), 0.0
    step = jax.jit(env.step)
    for _ in range(env.time_limit):
        state, ts = step(state, controller(state))
        total += float(ts.reward[0])
    return total


def test_every_reset_is_the_file():
    env = FixedSarSat(scenario_file=SCENARIO)
    a, _ = env.reset(jax.random.PRNGKey(0))
    b, _ = env.reset(jax.random.PRNGKey(7))
    doc = load_scenario(SCENARIO)
    assert env.num_agents == len(doc["satellites"]) and env.max_targets == len(doc["targets"])
    for x, y in zip(jax.tree.leaves(a[1:]), jax.tree.leaves(b[1:])):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(a.battery, [s["initial_battery"] for s in doc["satellites"]])
    np.testing.assert_allclose(a.target_priority, [t["priority"] for t in doc["targets"]], rtol=1e-6)


def test_file_parameters_are_enforced():
    with pytest.raises(ValueError):
        FixedSarSat(scenario_file=SCENARIO, time_limit=180)


@pytest.mark.parametrize(
    "policy, expected",
    [("greedy_beam", 0.125), ("coop_dedup", 0.25), ("solo_plan", 0.375), ("coop_plan", 0.75)],
)
def test_reference_gaps(policy, expected):
    assert _return(FixedSarSat(scenario_file=SCENARIO), policy) == pytest.approx(expected, abs=1e-6)


def test_coop_variant_resets_and_steps():
    env = FixedCoopSarSat(scenario_file=SCENARIO, hotspots=2, hotspot_targets=1)
    state, ts = env.reset(jax.random.PRNGKey(3))
    assert ts.observation.agents_view.shape == (2, env.obs_dim)
    np.testing.assert_allclose(state.battery, 0.55)
