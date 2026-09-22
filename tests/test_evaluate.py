"""Tests for the seeded evaluator, its CSV logs and the HTML viewer."""

import csv
from pathlib import Path

import numpy as np
import pytest

from sarsat import SarSat
from sarsat.evaluate import POLICIES, greedy_policy, run_episode, write_csv, write_html


def _read(path: Path) -> list:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_logs_are_consistent_and_reproducible(tmp_path: Path) -> None:
    env = SarSat(num_satellites=4, max_targets=30, time_limit=40)
    log = run_episode(env, greedy_policy, seed=3)
    again = run_episode(env, greedy_policy, seed=3)
    np.testing.assert_array_equal(log.captures, again.captures)
    np.testing.assert_array_equal(log.sat_latlon, again.sat_latlon)

    assert log.num_steps <= 40 and log.sat_latlon.shape == (log.num_steps + 1, 4, 2)
    assert np.all(np.abs(log.sat_latlon[..., 0]) <= 90) and np.all(
        np.abs(log.sat_latlon[..., 1]) <= 180
    )
    np.testing.assert_allclose(log.reward.sum(), len({int(t) for _, t, _ in log.captures}) / 30)

    paths = write_csv(log, tmp_path)
    satellites, looks, targets, captures = (_read(p) for p in paths)
    assert len(satellites) == (log.num_steps + 1) * 4
    assert len(looks) == int(log.sensing.sum())
    assert sum(int(row["targets_hit"]) for row in looks) == len(captures) == len(log.captures)
    imaged = {row["target"]: row for row in targets if int(row["captured_step"]) >= 0}
    assert len(imaged) == len({row["target"] for row in captures})
    for row in captures:
        assert int(row["step"]) >= int(imaged[row["target"]]["captured_step"])
        look = next(
            r for r in looks if r["step"] == row["step"] and r["satellite"] == row["satellite"]
        )
        # A captured target lies within a beam half-width of that look's boresight (roughly:
        # 4 degrees of beam at ~550 km is under 1 degree of latitude).
        assert abs(float(look["boresight_lat_deg"]) - float(row["lat_deg"])) < 1.5
    assert sum(int(row["on_land"]) for row in targets) == 24

    html = write_html(log, tmp_path / "episode.html")
    text = html.read_text()
    assert "/*EPISODE_DATA*/null" not in text and '"captures": ' in text and "LAND_MASK" in text


def test_random_policy_runs() -> None:
    env = SarSat(num_satellites=2, max_targets=5, time_limit=3)
    log = run_episode(env, POLICIES["random"], seed=0)
    assert log.sensing.shape == (3, 2) and log.sensing.all()


def test_shipped_example_csvs_are_reproducible(tmp_path: Path) -> None:
    """The examples must match a fresh run; if this fails, regenerate them."""
    example = Path(__file__).parents[1] / "examples" / "8sat-100tg-seed0"
    env = SarSat(num_satellites=8, max_targets=100, time_limit=180)
    for path in write_csv(run_episode(env, greedy_policy, seed=0), tmp_path):
        assert path.read_text() == (example / path.name).read_text(), (
            f"{path.name} differs from examples/8sat-100tg-seed0 — regenerate it "
            "(see examples/README.md)"
        )


def test_reference_controller_logs_through_the_evaluator() -> None:
    """A state-based reference policy runs through run_episode like any other policy."""
    env = SarSat(num_satellites=4, max_targets=300, time_limit=25, hotspots=2, planes=2)
    logs = {p: run_episode(env, p, seed=1) for p in ("greedy_beam", "coop_plan")}
    for log in logs.values():
        assert log.sensing.shape == (25, 4) and np.isfinite(log.reward).all()
    # Same scenario, same seed: the target field is identical across policies.
    np.testing.assert_array_equal(
        logs["greedy_beam"].target_latlon, logs["coop_plan"].target_latlon
    )
    with pytest.raises(ValueError):
        run_episode(env, "no_such_policy", seed=1)
