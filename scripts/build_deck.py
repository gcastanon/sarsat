# ruff: noqa: RUF001, E501  (presentation strings: typographic characters, long lines)
"""Build reports/sarsat_marl_results.html: a self-contained HTML deck.

Reads reports/results.json (produced by scripts/collect_results.py) at BUILD time and
inlines the numbers directly into the generated markup -- the resulting HTML file has no
runtime dependencies (no fetch, no external CSS/JS/fonts). 12 slides (100-agent +
200-agent + 500-agent summary/gap) plus a 13th 500-agent "best approach" slide once
events500 has a trained RL row.

Run: python scripts/build_deck.py
"""

import base64
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "reports", "results.json")
OUT_PATH = os.path.join(REPO_ROOT, "reports", "sarsat_marl_results.html")
# Four viewer panels of one 500-satellite episode (seed 1000, step 120), captured from
# sarsat/viewer.html; the slide is skipped when the image is absent.
VIEWER_500_PATH = os.path.join(REPO_ROOT, "docs", "viewer-500sat-seed1000-step120.jpg")
FOOTER_TEXT = "SarSat MARL — 2026-09-23"
TOTAL_SLIDES = 10  # overwritten in main() once the final slide count is known


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def best_rl_row(rows):
    rl = [r for r in rows if r["kind"] == "rl"]
    return max(rl, key=lambda r: r["mean"]) if rl else None


def find_row(rows, label):
    return next(r for r in rows if r["label"] == label)


def credit_mix_of(label):
    m = re.search(r"credit_mix\s*([0-9.]+)", label)
    return m.group(1) if m else "?"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Reusable fragments
# --------------------------------------------------------------------------- #


def results_table(rows, table_id):
    best = best_rl_row(rows)
    head = (
        "<tr><th>Policy</th><th>Mean ± std</th><th>Min</th><th>Max</th>"
        "<th>Beats Greedy-Beam</th></tr>"
    )
    body = []
    for r in rows:
        classes = []
        if best is not None and r is best:
            classes.append("best-rl")
        if r["label"] == "Coop-Plan":
            classes.append("coop-plan")
        cls_attr = f' class="{" ".join(classes)}"' if classes else ""
        beats = r["beats_greedy_beam"] if r["beats_greedy_beam"] is not None else "—"
        kind_badge = (
            '<span class="badge badge-rl">RL</span>'
            if r["kind"] == "rl"
            else '<span class="badge badge-ref">ref</span>'
        )
        body.append(
            f"<tr{cls_attr}><td>{kind_badge}{esc(r['label'])}</td>"
            f"<td>{fmt(r['mean'])} ± {fmt(r['std'])}</td>"
            f"<td>{fmt(r['min'])}</td><td>{fmt(r['max'])}</td>"
            f"<td>{esc(beats)}</td></tr>"
        )
    return (
        f'<div class="table-scroll"><table id="{table_id}">'
        f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def sweep_table(rows, table_id):
    """``rows`` is a list of (config, greedy_beam, solo_plan, coop_plan)."""
    head = "<tr><th>Config</th><th>Greedy-Beam</th><th>Solo-Plan</th><th>Coop-Plan</th></tr>"
    body = []
    for config, gb, sp, cp in rows:
        body.append(
            f"<tr><td>{esc(config)}</td><td>{fmt(gb)}</td><td>{fmt(sp)}</td><td>{fmt(cp)}</td></tr>"
        )
    return (
        f'<div class="table-scroll"><table id="{table_id}">'
        f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def slide_wrap(number, title, body_html, kicker=""):
    kicker_html = f'<div class="kicker">{esc(kicker)}</div>' if kicker else ""
    return f"""
<section class="slide" id="slide-{number}" data-index="{number}">
  <div class="slide-inner">
    {kicker_html}
    <h1>{esc(title)}</h1>
    <div class="slide-body">
      {body_html}
    </div>
  </div>
  <div class="slide-footer">
    <span class="footer-text">{esc(FOOTER_TEXT)}</span>
    <span class="footer-page">{number} / {TOTAL_SLIDES}</span>
  </div>
</section>
"""


def bullets(items):
    lis = "\n".join(f"<li>{item}</li>" for item in items)
    return f'<ul class="bullets">{lis}</ul>'


def columns(cols):
    """``cols`` is a list of (heading, [bullet, ...])."""
    parts = []
    for heading, items in cols:
        li = "\n".join(f"<li>{item}</li>" for item in items)
        parts.append(
            f'<div class="col"><h3>{esc(heading)}</h3><ul class="bullets tight">{li}</ul></div>'
        )
    return f'<div class="col-wrap cols-{len(cols)}">{"".join(parts)}</div>'


# --------------------------------------------------------------------------- #
# Architecture diagram (slide 3 / slide 8) -- pure CSS boxes + arrows
# --------------------------------------------------------------------------- #


def arch_diagram(obs_dim, slot_note, critic_dim, extra_note=""):
    extra = f'<div class="arch-note">{extra_note}</div>' if extra_note else ""
    return f"""
<div class="arch-wrap">
  <div class="arch-flow">
    <div class="arch-box small">obs {obs_dim}</div>
    <div class="arch-arrow">&#9660;</div>
    <div class="arch-box wide">
      <div class="arch-title">SarSatTorso</div>
      <div class="arch-sub">{slot_note}</div>
      <div class="arch-sub">shared Dense(64) over look-ahead</div>
      <div class="arch-sub">+ self features</div>
    </div>
    <div class="arch-arrow">&#9660;</div>
    <div class="arch-box">MLP 256 &rarr; 128 + LayerNorm</div>
    <div class="arch-arrow">&#9660;</div>
    <div class="arch-skip">
      <div class="arch-box gru">GRU 128</div>
      <div class="arch-skip-rail" aria-hidden="true"></div>
      <div class="arch-skip-label">skip connection around GRU</div>
    </div>
    <div class="arch-arrow">&#9660;</div>
    <div class="arch-box">MLP 128</div>
    <div class="arch-arrow">&#9660;</div>
    <div class="arch-box wide head">
      <div class="arch-title">SlotMixtureHead</div>
      <div class="arch-sub">categorical over 4 slots (latent, marginalised in log-prob)</div>
      <div class="arch-sub">&times; tanh-Gaussian pointing around chosen slot (mean =
        atanh(slot pointing) + learned correction, init std 0.15)</div>
      <div class="arch-sub">&times; side Bernoulli (leans to slot side)</div>
      <div class="arch-sub">&times; sense Bernoulli (masked when battery &lt; sense cost)</div>
    </div>
  </div>
  <div class="arch-critic">
    <div class="arch-title">Critic</div>
    <div class="arch-sub">same torso / GRU / MLP &rarr; Dense(1)</div>
    <div class="arch-sub">input {critic_dim}</div>
    {extra}
  </div>
</div>
"""


# --------------------------------------------------------------------------- #
# Slide content builders
# --------------------------------------------------------------------------- #


def slide_summary(number, scenario_key, title, one_liner, rows, table_id, verdict=""):
    body = f'<p class="one-liner">{one_liner}</p>' + results_table(rows, table_id)
    body += (
        '<p class="legend"><span class="swatch best-rl"></span> best RL row '
        '&nbsp;&nbsp; <span class="swatch coop-plan"></span> Coop-Plan (centralised '
        "reference; sees the full state and every satellite's future access) "
        "&nbsp;&nbsp; mean &plusmn; std over test seeds 1000&ndash;1015, greedy actions"
        + (f"<br><strong>{verdict}</strong>" if verdict else "")
        + "</p>"
    )
    return slide_wrap(number, title, body, kicker=scenario_key)


def slide_best_approach(number, kicker, best_sentence, hyper_items, learned_items, remains_items):
    body = f'<p class="best-sentence">{best_sentence}</p>'
    body += columns(
        [
            ("System", hyper_items),
            ("What it learned", learned_items),
            ("What remains", remains_items),
        ]
    )
    return slide_wrap(number, kicker + ": Best RL Approach", body, kicker=kicker)


def slide_architecture(number, kicker, obs_dim, slot_note, critic_dim, extra_note=""):
    body = arch_diagram(obs_dim, slot_note, critic_dim, extra_note)
    return slide_wrap(number, kicker + ": Architecture", body, kicker=kicker)


def slide_obs_action_critic(number, kicker, obs_items, action_items, critic_items):
    body = columns(
        [
            ("Observation", obs_items),
            ("Action", action_items),
            ("Critic input", critic_items),
        ]
    )
    return slide_wrap(number, kicker + ": Observation, Action and Critic", body, kicker=kicker)


def slide_reward_shaping(number, kicker, items):
    body = bullets(items)
    return slide_wrap(number, kicker + ": Reward Shaping", body, kicker=kicker)


def slide_viewer_500(number):
    """One episode of each policy at the same moment, straight from the map viewer."""
    with open(VIEWER_500_PATH, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    stats = [
        ("Greedy-Beam", "0.280", "2,564", "14%", "7,310"),
        ("Solo-Plan", "0.415", "1,772", "14%", "7,275"),
        ("MAPPO (trained)", "0.605", "439", "67%", "4,463"),
        ("Coop-Plan", "0.598", "336", "79%", "4,039"),
    ]
    rows = "".join(
        f"<tr><td>{esc(a)}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>"
        for a, b, c, d, e in stats
    )
    body = f"""
<div style="display:flex; gap:18px; align-items:flex-start;">
  <img src="data:image/jpeg;base64,{data}" alt="Viewer: four policies at step 120"
       style="width:66%; border:1px solid #ccd; border-radius:6px;">
  <div style="flex:1; font-size:0.8em;">
    <p>Seed 1000, step 120 of 180. Top: Greedy-Beam, Solo-Plan. Bottom: MAPPO, Coop-Plan.
    Magenta clusters are events open and not yet imaged.</p>
    <div class="table-scroll"><table id="table-ev500-viewer">
      <thead><tr><th>Policy</th><th>Return so far</th><th>Event targets missed
      (of 5,000)</th><th>Mean battery</th><th>Looks</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p>Both independent policies have spent their charge on background targets when
    events open; the trained policy, like Coop-Plan, takes about 40% fewer looks and keeps
    the battery for the events.</p>
  </div>
</div>"""
    return slide_wrap(number, "500-Agent: One Episode in the Viewer", body, kicker="500-Agent")


def slide_500_gap(number, kicker, title, bullet_items, sweep_rows, table_id):
    body = bullets(bullet_items)
    body += (
        '<p class="legend">Scenario sweep, seed 0, 180 steps '
        "(config, greedy_beam, solo_plan, coop_plan):</p>"
    )
    body += sweep_table(sweep_rows, table_id)
    return slide_wrap(number, title, body, kicker=kicker)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    global TOTAL_SLIDES

    data = load_results()
    hot_rows = data["hotspots100"]
    ev_rows = data["events200"]
    ev500_rows = data.get("events500", [])

    hot_best = best_rl_row(hot_rows)
    ev_best = best_rl_row(ev_rows)
    ev500_best = best_rl_row(ev500_rows) if ev500_rows else None
    hot_coop = find_row(hot_rows, "Coop-Plan")
    ev_coop = find_row(ev_rows, "Coop-Plan")
    ev500_coop = find_row(ev500_rows, "Coop-Plan") if ev500_rows else None
    ev500_solo = find_row(ev500_rows, "Solo-Plan") if ev500_rows else None

    # slides 1-10 (100-agent + 200-agent) + slide 11 (500-agent summary) + slide 12
    # (500-agent gap) + slide 13 (500-agent best approach, only once trained) + the
    # viewer slide (only once trained and the image exists)
    has_viewer_500 = ev500_best is not None and os.path.exists(VIEWER_500_PATH)
    TOTAL_SLIDES = 12 + (1 if ev500_best is not None else 0) + (1 if has_viewer_500 else 0)

    best100_sentence = (
        f"<strong>{esc(hot_best['label'])}</strong> is the best RL row: mean "
        f"<strong>{fmt(hot_best['mean'])}</strong> over {hot_best['n']} seeds "
        f"({esc(hot_best['beats_greedy_beam'])} beat Greedy-Beam), within "
        f"{fmt(abs(hot_coop['mean'] - hot_best['mean']))} of Coop-Plan's "
        f"{fmt(hot_coop['mean'])}. Five independent runs (three seeds, a wider observation, "
        f"a fine-tune) all land at 0.894–0.898: level with Coop-Plan within noise, not above "
        f"it. Held-out validation seeds 2000–2015 agree: 0.909 vs Coop-Plan 0.914, beating it "
        f"on 2 of 16."
    )
    best200_sentence = (
        f"<strong>{esc(ev_best['label'])}</strong> is the best RL row: mean "
        f"<strong>{fmt(ev_best['mean'])}</strong> over {ev_best['n']} seeds "
        f"({esc(ev_best['beats_greedy_beam'])} beat Greedy-Beam), still "
        f"{fmt(ev_coop['mean'] - ev_best['mean'])} below Coop-Plan's {fmt(ev_coop['mean'])} "
        f"(97% of it) and +{fmt(ev_best['mean'] - find_row(ev_rows, 'Greedy-Beam')['mean'])} "
        f"over Greedy-Beam; Solo-Plan, which plans as far ahead alone, gains nothing over "
        f"Greedy-Beam here."
    )

    slides = []

    # ---------------- Slide 1: 100-agent summary ----------------
    one_liner_100 = (
        "<strong>sarsat-100sat-hotspots</strong>: 100 satellites, 10 Walker planes, "
        "6000 background targets + 40 hotspots × 50 targets at 10× priority, "
        "180 steps, 20% duty cycle."
    )
    slides.append(
        slide_summary(
            1,
            "100-Agent Non-Windowed",
            "100-Agent Non-Windowed: Summary",
            one_liner_100,
            hot_rows,
            "table-hot",
            verdict="Every RL agent beats Greedy-Beam on 16/16 seeds; the best is within "
            "0.006 of Coop-Plan (a statistical tie), not above it.",
        )
    )

    # ---------------- Slide 2: 100-agent best approach ----------------
    slides.append(
        slide_best_approach(
            2,
            "100-Agent",
            best100_sentence,
            [
                "MAPPO (<code>rec_mappo</code>, MAPX) with shared parameters over 100 agents",
                "<code>CoopSarSat</code> observation (ranked beam-value slots + cluster look-ahead)",
                "Slot-mixture anchored head; <code>credit_mix</code> 0.5",
                "32 envs × 180-step rollouts, 4 epochs × 4 minibatches",
                "Clip 0.1, gamma 0.995, lambda 0.95",
                "Actor 3e-4 / critic 5e-4, linear LR decay",
                "GRU 128, recurrent chunks of 30",
            ],
            [
                "Rations battery toward hotspots rather than spending it all on background",
                "Learns beam-aware pointing and side/sense choices, not just “aim somewhere”",
            ],
            [
                "Same-step duplicate captures ≈ 1.4% of the return (two agents take one beam)",
                "Battery still ends episodes partly unspent",
            ],
        )
    )

    # ---------------- Slide 3: 100-agent architecture ----------------
    slides.append(
        slide_architecture(
            3,
            "100-Agent",
            obs_dim=69,
            slot_note="shared Dense(32) per slot × 4",
            critic_dim="133 = own view 69 + team summary 64",
        )
    )

    # ---------------- Slide 4: 100-agent obs/action/critic ----------------
    slides.append(
        slide_obs_action_critic(
            4,
            "100-Agent",
            [
                "4 slots × 8: valid, incidence, squint, side (action encoding), "
                "beam value, scarcity-weighted value, team future accesses, contention",
                "16 × 2 look-ahead: remaining hotspot value reachable, raw and scarcity-weighted",
                "5 self: battery, can-sense, time remaining, visible value, visible count",
                "All features log-squashed to roughly [-1, 1] / [0, 1]",
            ],
            [
                "4 slots: incidence [-1, 1], squint [-1, 1], side {0, 1}, sense {0, 1}",
                "Unchanged from the base environment's action space",
            ],
            [
                "Own view (69) + team summary (64): time remaining; battery mean & "
                "10/50/90 percentiles; fraction able to sense",
                "Remaining hotspot & background value",
                "40 sorted per-cluster remaining fractions",
                "16-step team-mean look-ahead",
                "Size independent of N — 133 vs 9,600 for a joint-observation critic",
            ],
        )
    )

    # ---------------- Slide 5: 100-agent reward shaping ----------------
    slides.append(
        slide_reward_shaping(
            5,
            "100-Agent",
            [
                "The environment reward is untouched: shared team reward = priority newly imaged",
                "<code>reward_scale</code> × 10 for value-loss conditioning "
                "(returns O(1–10); <code>fraction_imaged</code> still logs the true return)",
                "<code>credit_mix</code> 0.5: r<sub>i</sub> = 0.5 · team + "
                "0.5 · N · own_share<sub>i</sub>; a target hit by k beams "
                "splits its value evenly; mean over agents is still the team reward",
                "Why: a background target is 1/26,000 of the return — invisible in "
                "a 100-agent shared reward. With pure team reward MAPPO reaches 0.847 "
                "and IPPO only ≈ 0.74",
                "This is credit assignment, not potential-based shaping: no new term "
                "rewards anything the team reward does not",
                "No other reward terms",
            ],
        )
    )

    # ---------------- Slide 6: 200-agent summary ----------------
    one_liner_200 = (
        "<strong>sarsat-200sat-events</strong>: 200 satellites, 20 Walker planes, "
        "6000 persistent background targets, 40 hotspot clusters that are events open "
        "for one 10–30 step window each, 10% duty cycle, 180 steps."
    )
    slides.append(
        slide_summary(
            6,
            "200-Agent Windowed",
            "200-Agent Windowed: Summary",
            one_liner_200,
            ev_rows,
            "table-ev",
            verdict="Every RL agent beats Greedy-Beam on 16/16 seeds (+0.21, i.e. +47%); "
            "the best reaches 97% of Coop-Plan and plateaus 0.02 below it.",
        )
    )

    # ---------------- Slide 7: 200-agent best approach ----------------
    cm200 = credit_mix_of(ev_best["label"])
    slides.append(
        slide_best_approach(
            7,
            "200-Agent",
            best200_sentence,
            [
                f"{esc(ev_best['label'])} — same shared-parameter recurrent "
                f"system as the 100-agent recipe",
                "<code>WindowedCoopSarSat</code> observation (9-wide slots; "
                "cluster look-ahead gated on window)",
                f"<code>credit_mix</code> {cm200}",
                "12 envs × 180-step rollouts (16 for the first two variants); GAE horizon lengthened to gamma 0.998 / lambda 0.98",
                "Otherwise the 100-agent PPO recipe (clip 0.1, 4 × 4, actor 3e-4 / critic 5e-4 with decay, GRU 128)",
            ],
            [
                "Coordinated timing: which satellite must spend charge on the event "
                "only it can reach before its window closes",
            ],
            [
                "Plateau: five variants (IPPO/MAPPO, credit_mix 0.1–0.5, look-ahead 16/48, "
                "GAE horizon 18/67 steps) all land at 0.641–0.649 vs Coop-Plan's 0.670",
                "Battery at window-open averages 0.37 vs Coop-Plan's 0.77 — the "
                "policy is not rationing far enough ahead of the event",
                "Duplicate captures: 38 priority units vs Coop-Plan's 3.5",
            ],
        )
    )

    # ---------------- Slide 8: 200-agent architecture ----------------
    slides.append(
        slide_architecture(
            8,
            "200-Agent",
            obs_dim=73,
            slot_note="shared Dense(32) per slot × 4 (9 features/slot: + fraction of "
            "window still open)",
            critic_dim="own view 73 + team summary 64",
            extra_note="Cluster look-ahead gated on the cluster windows; otherwise "
            "identical to the 100-agent network.",
        )
    )

    # ---------------- Slide 9: 200-agent obs/action/critic ----------------
    slides.append(
        slide_obs_action_critic(
            9,
            "200-Agent",
            [
                "Same 8 base features per slot plus a 9th: fraction of the target's "
                "window still open",
                "Cluster look-ahead (16 × 2) gated on the cluster windows — "
                "only counts value the satellite can still reach before the window "
                "closes",
                "Same 5 self features as the 100-agent observation",
            ],
            [
                "Identical to the 100-agent action space: incidence, squint, side, sense",
                "Unchanged from the base environment",
            ],
            [
                "Own view (73) + the same fixed ~64-number team summary",
                "Still independent of N — the same saving that makes MAPPO usable "
                "at 200 satellites",
            ],
        )
    )

    # ---------------- Slide 10: 200-agent reward shaping ----------------
    slides.append(
        slide_reward_shaping(
            10,
            "200-Agent",
            [
                "Same shaping recipe as the 100-agent scenario: environment reward "
                "untouched, shared team reward = priority newly imaged",
                "<code>reward_scale</code> × 10 for value-loss conditioning",
                f"Best row uses <code>credit_mix</code> {cm200}: r<sub>i</sub> = "
                f"(1 − {cm200}) · team + {cm200} · N · own_share"
                f"<sub>i</sub>; simultaneous captures split evenly; mean over agents "
                f"is still the team reward",
                "Same rationale as the 100-agent case: a background target is a tiny "
                "fraction of the shared reward and gets lost in 200-way averaging",
                "Still credit assignment, not potential-based shaping — no new "
                "behaviour is rewarded that the team reward does not already reward",
                "No other reward terms",
            ],
        )
    )

    # ---------------- Slide 11: 500-agent summary ----------------
    one_liner_500 = (
        "<strong>sarsat-500sat-events</strong>: 500 satellites, 50 Walker planes of 10, "
        "15,000 persistent background targets + 100 event clusters × 50 targets "
        "(10× priority, open 10–30 steps), 180 steps, 5% duty cycle."
    )
    if ev500_best is not None:
        verdict_500 = (
            f"{esc(ev500_best['label'])} reaches mean {fmt(ev500_best['mean'])} over "
            f"{ev500_best['n']} seeds ({esc(ev500_best['beats_greedy_beam'])} beat "
            f"Greedy-Beam), {fmt(ev500_coop['mean'] - ev500_best['mean'])} below "
            f"Coop-Plan's {fmt(ev500_coop['mean'])} and "
            f"{fmt(ev500_best['mean'] - ev500_solo['mean'])} above Solo-Plan's "
            f"{fmt(ev500_solo['mean'])}."
        )
    else:
        verdict_500 = "Training in progress; reference policies only."
    slides.append(
        slide_summary(
            11,
            "500-Agent Windowed Events",
            "500-Agent Windowed Events: Summary",
            one_liner_500,
            ev500_rows,
            "table-ev500",
            verdict=verdict_500,
        )
    )

    # ---------------- Slide 12: 500-agent gap ----------------
    bullets_500 = [
        "On seeds 1000–1015: Greedy-Beam 0.394, Solo-Plan 0.658 (+67%, beats "
        "Greedy-Beam on 16/16 seeds), Coop-Plan 0.883 (+34% over Solo-Plan, 16/16 seeds)",
        "Coop-Dedup reaches only 0.432 — same-step deduplication is worth about 10%, so "
        "almost none of the cooperative gain is satellites sharing a view",
        "No formation flying: Walker planes only — 25 planes, 50 planes and random "
        "orbits all give the same three numbers",
        "Key message: the 5% duty cycle makes rationing pay (opens the Solo-Plan gap); "
        "keeping 100 clusters with 10–30 step windows keeps cooperation worth a third "
        "on top — more events or longer windows hand the gain to Solo-Plan",
    ]
    sweep_rows_500 = [
        ("200-sat field, 10% duty", 0.885, 0.886, 0.932),
        ("scaled field, 10% duty", 0.636, 0.652, 0.889),
        ("7.5% duty", 0.540, 0.592, 0.876),
        ("5% duty (chosen), 50 planes of 10", 0.407, 0.684, 0.891),
        ("5% duty, 200 clusters", 0.456, 0.769, 0.878),
        ("5% duty, 20–60 step windows", 0.490, 0.841, 0.911),
    ]
    slides.append(
        slide_500_gap(
            12,
            "500-Agent",
            "500-Agent: Why This Benchmark",
            bullets_500,
            sweep_rows_500,
            "table-ev500-sweep",
        )
    )

    # ---------------- Slide 13: 500-agent best approach (once trained) ----------------
    if ev500_best is not None:
        best500_sentence = (
            f"<strong>{esc(ev500_best['label'])}</strong> is the best RL row: mean "
            f"<strong>{fmt(ev500_best['mean'])}</strong> over {ev500_best['n']} seeds "
            f"({esc(ev500_best['beats_greedy_beam'])} beat Greedy-Beam), "
            f"{fmt(ev500_coop['mean'] - ev500_best['mean'])} below Coop-Plan's "
            f"{fmt(ev500_coop['mean'])} and "
            f"{fmt(ev500_best['mean'] - ev500_solo['mean'])} above Solo-Plan's "
            f"{fmt(ev500_solo['mean'])}."
        )
        slides.append(
            slide_best_approach(
                13,
                "500-Agent",
                best500_sentence,
                [
                    "MAPPO (<code>rec_mappo</code>, MAPX) with shared parameters over 500 agents",
                    "<code>WindowedCoopSarSat</code> observation, 73 numbers per agent "
                    "(unchanged from 200 satellites)",
                    "Slot-mixture anchored head; <code>credit_mix</code> 0.5; ranked contention",
                    "16 envs × 180-step rollouts on one A40 (Runpod), 4 epochs × 4 minibatches",
                    "Clip 0.1, gamma 0.995",
                    "Actor 3e-4 / critic 5e-4, linear LR decay",
                ],
                [
                    "Fast: 0.836 after 30 updates, 0.870 by update 210; stopped at a plateau "
                    "at update ~660 (best checkpoint: update 600), 2.4 A40-hours, $1.19",
                    "Beats Solo-Plan on 16/16 seeds by about 0.21: it rations the 5% duty "
                    "cycle far better than the independent heuristic",
                    "Trails Coop-Plan on 16/16 seeds by 0.012–0.020: 98% of the centralised "
                    "planner, the same pattern as at 100 and 200 satellites",
                ],
                [
                    "Not diagnosed: how much of the gain over Solo-Plan is coordination (the "
                    "observation carries teammate contention and team access counts) and how "
                    "much is better rationing",
                    "The 200-satellite limits (same-step duplicates, battery timing at event "
                    "windows) are likely here too but were not measured",
                    "One training seed; no pure team-reward or IPPO run at 500 satellites yet "
                    "(<code>credit_mix</code> 0.5 is reward shaping, disclosed)",
                ],
            )
        )

    if has_viewer_500:
        slides.append(slide_viewer_500(len(slides) + 1))

    html = render_document(slides, hot_rows, ev_rows)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"Wrote {OUT_PATH} ({len(html):,} bytes, {len(slides)} slides)")
    print(f"Best hotspots100 RL row: {hot_best['label']} mean={hot_best['mean']:.3f}")
    print(f"Best events200 RL row:   {ev_best['label']} mean={ev_best['mean']:.3f}")
    if ev500_best is not None:
        print(f"Best events500 RL row:   {ev500_best['label']} mean={ev500_best['mean']:.3f}")
    else:
        print("events500: reference policies only (no trained run yet)")


# --------------------------------------------------------------------------- #
# Document shell: CSS + JS + slide assembly
# --------------------------------------------------------------------------- #

CSS = """
:root {
  --accent: #2b6cb0;
  --accent-dark: #1a4971;
  --bg: #f7f8fa;
  --stage-bg: #ffffff;
  --fg: #1a202c;
  --fg-muted: #4a5568;
  --border: #d8dee6;
  --row-alt: #f0f4f9;
  --best-bg: #eaf3ff;
  --coop-bg: #fff6e5;
  --good: #1a7a3c;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
body {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.stage-wrap {
  flex: 1 1 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  min-height: 0;
}
.stage {
  position: relative;
  width: 1280px;
  height: 720px;
  background: var(--stage-bg);
  box-shadow: 0 10px 40px rgba(20, 30, 50, 0.18);
  border-radius: 6px;
  flex: 0 0 auto;
  transform-origin: center center;
}
.slide {
  position: absolute;
  inset: 0;
  display: none;
  flex-direction: column;
  padding: 0;
}
.slide.active { display: flex; }
.slide-inner {
  flex: 1 1 auto;
  padding: 40px 68px 14px 68px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.kicker {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 4px;
}
.slide h1 {
  font-size: 33px;
  margin: 0 0 14px 0;
  color: var(--fg);
  font-weight: 700;
  border-bottom: 3px solid var(--accent);
  padding-bottom: 10px;
}
.slide-body {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.one-liner {
  font-size: 19px;
  color: var(--fg-muted);
  margin: 0 0 16px 0;
  line-height: 1.4;
}
.best-sentence {
  font-size: 21px;
  background: var(--best-bg);
  border-left: 5px solid var(--accent);
  padding: 14px 18px;
  border-radius: 4px;
  margin: 0 0 20px 0;
  line-height: 1.45;
}
.slide-footer {
  flex: 0 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 68px 18px 68px;
  font-size: 13px;
  color: var(--fg-muted);
  border-top: 1px solid var(--border);
}

/* Tables */
.table-scroll {
  max-height: 470px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
thead th {
  position: sticky;
  top: 0;
  background: var(--accent);
  color: #fff;
  text-align: left;
  padding: 4px 14px;
  font-weight: 600;
  z-index: 1;
}
tbody td {
  padding: 2px 14px;
  line-height: 1.25;
  border-bottom: 1px solid var(--border);
}
tbody tr:nth-child(even) { background: var(--row-alt); }
tbody tr.best-rl { background: var(--best-bg); font-weight: 700; }
tbody tr.coop-plan { background: var(--coop-bg); font-weight: 700; }
.badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  padding: 1px 6px;
  border-radius: 3px;
  margin-right: 8px;
  vertical-align: middle;
}
.badge-rl { background: var(--accent); color: #fff; }
.badge-ref { background: #d8dee6; color: var(--fg-muted); }
.legend {
  font-size: 14px;
  color: var(--fg-muted);
  margin: 10px 0 0 0;
}
.swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 3px;
  vertical-align: middle;
  margin-right: 4px;
}
.swatch.best-rl { background: var(--best-bg); border: 1px solid var(--accent); }
.swatch.coop-plan { background: var(--coop-bg); border: 1px solid #c8961e; }

/* Bullets */
ul.bullets {
  font-size: 19px;
  line-height: 1.5;
  margin: 0;
  padding-left: 24px;
}
ul.bullets.tight { font-size: 17px; line-height: 1.42; }
ul.bullets li { margin-bottom: 10px; }
code {
  background: #eef1f5;
  border-radius: 3px;
  padding: 1px 5px;
  font-size: 0.92em;
  font-family: "SF Mono", Consolas, monospace;
}

/* Columns */
.col-wrap {
  display: flex;
  gap: 28px;
  flex: 1 1 auto;
  min-height: 0;
}
.col-wrap .col {
  flex: 1 1 0;
  background: #fbfcfd;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  overflow-y: auto;
}
.col h3 {
  margin: 0 0 10px 0;
  font-size: 19px;
  color: var(--accent-dark);
  border-bottom: 2px solid var(--border);
  padding-bottom: 6px;
}

/* Architecture diagram */
.arch-wrap {
  display: flex;
  gap: 22px;
  flex: 1 1 auto;
  align-items: center;
  justify-content: center;
  min-height: 0;
  overflow: hidden;
}
.arch-flow {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  flex: 0 1 auto;
}
.arch-box {
  border: 2px solid var(--accent);
  border-radius: 7px;
  background: #fbfdff;
  padding: 4px 14px;
  font-size: 13.5px;
  text-align: center;
  color: var(--fg);
  min-width: 200px;
}
.arch-box.small { min-width: 140px; font-weight: 700; }
.arch-box.wide { min-width: 660px; }
.arch-box.head { border-color: var(--accent-dark); background: var(--best-bg); }
.arch-box .arch-title {
  font-weight: 700;
  color: var(--accent-dark);
  font-size: 14.5px;
  margin-bottom: 2px;
}
.arch-box .arch-sub {
  font-size: 12px;
  color: var(--fg-muted);
  line-height: 1.22;
}
.arch-arrow {
  color: var(--accent);
  font-size: 13px;
  line-height: 1;
}
.arch-skip {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.arch-skip .arch-box.gru { min-width: 220px; }
.arch-skip-rail {
  position: absolute;
  right: -34px;
  top: -6px;
  bottom: -6px;
  width: 20px;
  border: 3px solid var(--accent);
  border-left: none;
  border-radius: 0 10px 10px 0;
}
.arch-skip-label {
  position: absolute;
  right: -142px;
  top: 50%;
  transform: translateY(-50%) rotate(0deg);
  font-size: 11.5px;
  color: var(--fg-muted);
  width: 110px;
  line-height: 1.25;
}
.arch-critic {
  border: 2px dashed var(--accent-dark);
  border-radius: 8px;
  padding: 16px 20px;
  background: #fbfcfd;
  min-width: 220px;
  max-width: 260px;
}
.arch-critic .arch-title {
  font-weight: 700;
  color: var(--accent-dark);
  font-size: 17px;
  margin-bottom: 8px;
}
.arch-critic .arch-sub {
  font-size: 14px;
  color: var(--fg-muted);
  margin-bottom: 8px;
  line-height: 1.4;
}
.arch-note {
  font-size: 12.5px;
  color: var(--fg-muted);
  border-top: 1px solid var(--border);
  padding-top: 8px;
  margin-top: 4px;
  line-height: 1.35;
}

/* Controls (outside the scaled stage) */
.controls {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 10px 0;
  background: var(--bg);
}
.controls button {
  font-size: 15px;
  padding: 8px 18px;
  border-radius: 20px;
  border: 1px solid var(--accent);
  background: #fff;
  color: var(--accent);
  cursor: pointer;
  font-weight: 600;
}
.controls button:hover { background: var(--accent); color: #fff; }
.controls button:disabled {
  opacity: 0.4;
  cursor: default;
  background: #fff;
  color: var(--accent);
}
.controls .counter {
  font-size: 15px;
  color: var(--fg-muted);
  min-width: 64px;
  text-align: center;
}

/* Print: one slide per page, no chrome, no scaling, all rows visible */
@media print {
  html, body { height: auto; overflow: visible; background: #fff; }
  .controls { display: none; }
  .stage-wrap { display: block; overflow: visible; }
  .stage {
    position: static;
    width: auto;
    height: auto;
    box-shadow: none;
    transform: none !important;
    border-radius: 0;
  }
  .slide {
    position: static;
    display: flex !important;
    width: 100%;
    height: 100vh;
    page-break-after: always;
  }
  .table-scroll { max-height: none !important; overflow: visible !important; }
}
"""

JS = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var total = slides.length;
  var index = 0;
  var counter = document.getElementById('counter');
  var prevBtn = document.getElementById('prevBtn');
  var nextBtn = document.getElementById('nextBtn');
  var stage = document.getElementById('stage');
  var stageWrap = document.getElementById('stageWrap');
  var STAGE_W = 1280, STAGE_H = 720;

  function render() {
    slides.forEach(function (s, i) {
      s.classList.toggle('active', i === index);
    });
    counter.textContent = (index + 1) + ' / ' + total;
    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === total - 1;
  }

  function go(delta) {
    index = Math.max(0, Math.min(total - 1, index + delta));
    render();
  }

  function rescale() {
    var w = stageWrap.clientWidth;
    var h = stageWrap.clientHeight;
    var scale = Math.min(w / STAGE_W, h / STAGE_H);
    stage.style.transform = 'scale(' + scale + ')';
  }

  prevBtn.addEventListener('click', function () { go(-1); });
  nextBtn.addEventListener('click', function () { go(1); });
  window.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') { go(1); e.preventDefault(); }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { go(-1); e.preventDefault(); }
  });
  window.addEventListener('resize', rescale);

  render();
  rescale();
  window.addEventListener('load', rescale);
  requestAnimationFrame(rescale);
})();
"""


def render_document(slides, hot_rows, ev_rows):
    slides_html = "\n".join(slides)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SarSat MARL Results</title>
<style>
{CSS}
</style>
</head>
<body>
  <div class="stage-wrap" id="stageWrap">
    <div class="stage" id="stage">
      {slides_html}
    </div>
  </div>
  <div class="controls">
    <button id="prevBtn" aria-label="Previous slide">&larr; Prev</button>
    <span class="counter" id="counter">1 / {TOTAL_SLIDES}</span>
    <button id="nextBtn" aria-label="Next slide">Next &rarr;</button>
  </div>
<script>
{JS}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
