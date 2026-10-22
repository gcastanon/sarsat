"""``sarsat.recommend``: variants of a request the policy collects at no cost to the rest."""

import jax
import pytest
from test_live import SEED, _actions, _env, _feasible_point

from sarsat.live import LiveWorld
from sarsat.recommend import Recommender, Variant, _offset, acceptable


def _world(steps=2):
    env = _env()
    world = LiveWorld(env, seed=SEED)
    for i in range(steps):
        world.step(_actions(env, jax.random.PRNGKey(700 + i), sense=False))
    return world


def test_offset_moves_the_right_distance() -> None:
    lat, lon = _offset(10.0, 20.0, 50.0, 90.0)
    assert abs(lat - 10.0) < 0.01 and 0.44 < lon - 20.0 < 0.47  # 50 km east at 10 deg lat
    lat, lon = _offset(10.0, 20.0, 50.0, 0.0)
    assert 0.44 < lat - 10.0 < 0.46 and abs(lon - 20.0) < 1e-6


def test_variants_change_one_or_two_parameters_each() -> None:
    world = _world()
    preview = world.preview_request(10.0, 20.0, "normal", 10)
    rec = Recommender(world, "greedy_beam", radii_km=(25.0,), bearings=4)
    vs = rec.variants(preview)
    assert vs and all(1 <= len(v.changes) <= 2 for v in vs)
    assert all(v.deadline_steps <= min(30, world.T - int(world.state.step)) for v in vs)
    singles = [v for v in vs if len(v.changes) == 1]
    kinds = {c.split()[0] for v in singles for c in v.changes}
    assert kinds == {"deadline", "priority", "location"}  # no quality axis without modes
    assert all(v.priority != "normal" for v in singles if v.changes[0].startswith("priority"))
    params = [tuple(sorted(v.params.items())) for v in vs]
    assert len(params) == len(set(params))  # no duplicates
    # The variants come in change-count order.
    counts = [len(v.changes) for v in vs]
    assert counts == sorted(counts)


def test_rank_keeps_reachable_variants_and_orders_them() -> None:
    world = _world()
    latlon, _ = _feasible_point(world, None, 10)
    preview = world.preview_request(*latlon, "normal", 10)
    rec = Recommender(world, "greedy_beam", radii_km=(25.0,), bearings=4, max_changes=1)
    base = world.forecast("greedy_beam")
    ranked = rec.rank(rec.variants(preview), base)
    assert ranked
    for v in ranked:
        assert v.passes > 0 or all(c.startswith("priority") for c in v.changes)
        if v.first_pass is not None:
            assert v.first_pass > world.global_step
    keys = [
        (len(v.changes), v.first_pass if v.first_pass is not None else 10**9, -v.passes)
        for v in ranked
    ]
    assert keys == sorted(keys)


def test_recommend_confirms_by_forecast_within_budget() -> None:
    world = _world()
    latlon, _ = _feasible_point(world, None, 6)
    preview = world.preview_request(*latlon, "low", 6)
    before = world.state
    rec = Recommender(
        world, "greedy_beam", radii_km=(25.0,), bearings=4, max_forecasts=3, max_results=2
    )
    base = world.forecast("greedy_beam")
    out = rec.recommend(preview, base)
    assert world.state is before and world.requests == []  # nothing committed
    assert set(out) >= {
        "request",
        "original",
        "alternatives",
        "rejected",
        "untried",
        "forecasts",
        "verdict",
    }
    assert out["request"]["id"] == preview.request.id
    if acceptable(
        {"request_fate": out["original"]["request_fate"], "same_set": out["original"]["same_set"]}
    ):
        assert out["forecasts"] == 0 and out["alternatives"] == []
    else:
        assert out["forecasts"] <= 3
        assert len(out["alternatives"]) <= 2
        for alt in out["alternatives"]:
            assert (
                alt["confirmed"]
                and alt["whatif"]["request_fate"]["collected"]
                and alt["whatif"]["same_set"]
            )
            assert alt["changes"]
        for rej in out["rejected"]:
            assert rej["confirmed"] is False and rej["reason"]
        assert out["candidates"] == len(out["alternatives"]) + len(out["rejected"]) + out["untried"]


def test_variant_to_dict_round_trips_params() -> None:
    v = Variant(1.0, 2.0, "high", 12, None, changes=["deadline 10 -> 12 steps"])
    d = v.to_dict()
    assert d["lat"] == 1.0 and d["deadline_steps"] == 12 and d["confirmed"] is None
    with pytest.raises(ValueError):
        Recommender(LiveWorld(_env(), seed=SEED), "greedy_beam", allow=("split",))
