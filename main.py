"""pumpfun-top5-forecast-01: seeded Monte Carlo forecaster (Python 3.13 stdlib only).

Reads /input/task.json + /input/history.ndjson (10-minute snapshots of pump.fun
coins), simulates each candidate coin's log10 market cap forward to each answer
read as a drifted random walk with per-coin drift/volatility shrunk toward
global estimates, plus a simulated field of newcomer coins born after the cut.
Tallies place probabilities and place-size bins over the simulations, averaged
over the seven reads, and writes /output/forecast.json.

Deterministic: one fixed-seed RNG, fixed iteration order, no wall clock, no
network, no model calls. Times and horizons are read from task.json.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_SIMS = 8000
MAX_CANDIDATES = 250
N_NEWCOMERS = 120
NEWCOMER_SD = 0.70
SIZE_UNIFORM_MIX = 0.06
MIN_ROW_P = 0.0005
FILL_P = 0.0005
MAX_ROWS_PER_PLACE = 100
MICRO = 1_000_000
# Plausible log10-cap range in SOL. A random walk's unbounded upside would
# otherwise put large mass on absurd caps (over 1M SOL) at long horizons;
# the biggest pump.fun launches reach roughly 1e5 SOL, so 5.5 is generous.
CAP_LO = -1.0
CAP_HI = 5.5


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf8") as f:
        task = json.load(f)
    edges = task["size"]["edges"]
    places = int(task["places"])
    reads = [parse_time(r) for r in task["reads"]]

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise RuntimeError("the history holds no usable mark")
    marks.sort(key=lambda m: m["at"])
    last_at = parse_time(marks[-1]["at"])
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    floor_cap = logcap_sol(last_rows[-1]["market_cap_lamports"])

    # Trajectories: mint -> list of (hours since first mark, logcap), time order.
    t0 = parse_time(marks[0]["at"])
    traj = {}
    created = {}
    for m in marks:
        t = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            mint = r["mint"]
            traj.setdefault(mint, []).append((t, logcap_sol(r["market_cap_lamports"])))
            created.setdefault(mint, r.get("created_at"))

    # Global drift/volatility from consecutive-observation changes.
    velocities = []
    sq_changes = []
    for mint, obs in traj.items():
        if len(obs) >= 2:
            (t1, x1), (t2, x2) = obs[0], obs[-1]
            if t2 > t1:
                velocities.append((x2 - x1) / (t2 - t1))
        for (t1, x1), (t2, x2) in zip(obs, obs[1:]):
            dt = t2 - t1
            if dt > 0:
                sq_changes.append(((x2 - x1) ** 2) / dt)
    velocities.sort()
    mu_global = velocities[len(velocities) // 2] if velocities else -0.05
    mu_global = max(-0.6, min(0.6, mu_global))
    sig_global = math.sqrt(sum(sq_changes) / len(sq_changes)) if sq_changes else 0.3
    sig_global = max(0.05, min(1.2, sig_global))

    # Candidate coins: last mark's rows plus any coin top-30 at any mark.
    cand = {r["mint"] for r in last_rows}
    for m in marks:
        for r in sorted(m["rows"], key=lambda r: r["rank"])[:30]:
            cand.add(r["mint"])

    def coin_params(mint):
        obs = traj[mint]
        # Last-seen cap; coins absent from the last mark fell below its chart.
        x0 = obs[-1][1]
        if mint not in {r["mint"] for r in last_rows}:
            x0 = min(x0, floor_cap)
        # Velocity from the most recent observations (least squares slope).
        recent = obs[-7:]
        if len(recent) >= 2:
            n = len(recent)
            mt = sum(t for t, _ in recent) / n
            mx = sum(x for _, x in recent) / n
            den = sum((t - mt) ** 2 for t, _ in recent)
            v = sum((t - mt) * (x - mx) for t, x in recent) / den if den > 0 else 0.0
            w = n / (n + 2.0)
            mu = w * v + (1 - w) * mu_global
        else:
            mu = mu_global
        mu = max(-0.6, min(0.6, mu))
        # Volatility from this coin's step changes, shrunk to the global value.
        sq = []
        for (t1, x1), (t2, x2) in zip(obs, obs[1:]):
            dt = t2 - t1
            if dt > 0:
                sq.append(((x2 - x1 - mu * dt) ** 2) / dt)
        if sq:
            sig = math.sqrt((sum(sq) + 4.0 * sig_global ** 2) / (len(sq) + 4.0))
        else:
            sig = sig_global
        sig = max(0.05, min(1.2, sig))
        return x0, mu, sig

    params = {mint: coin_params(mint) for mint in cand}
    # Keep the most plausible candidates by last-seen cap, deterministic order.
    ordered = sorted(params, key=lambda m: (-params[m][0], m))[:MAX_CANDIDATES]
    params = {m: params[m] for m in ordered}

    # Horizons in hours from the last mark to each answer read.
    horizons = [max(0.05, (r - last_at).total_seconds() / 3600.0) for r in reads]
    mean_horizon = sum(horizons) / len(horizons)
    max_horizon = max(horizons)
    dts = [horizons[0]] + [b - a for a, b in zip(horizons, horizons[1:])]
    dts = [max(0.01, d) for d in dts]

    # Newcomer field: distribution of fast risers' early peaks, from the data.
    # A coin's peak within its first hour of life, over coins whose peak
    # reached the chart floor, is what a future winner's early cap looks like.
    peaks = []
    for mint, obs in traj.items():
        c0 = created.get(mint)
        if not c0:
            continue
        ct = parse_time(c0)
        early = [x for (t, x) in obs if (t0.timestamp() + t * 3600) - ct.timestamp() <= 3600]
        if early and max(early) >= floor_cap:
            peaks.append(max(early))
    peaks.sort()
    if len(peaks) >= 10:
        # Upper quartile, not the median: a newcomer that takes a top-5
        # place is a right-tail riser, not a typical new coin.
        newcomer_mean = peaks[(3 * len(peaks)) // 4]
    else:
        top10 = sorted((logcap_sol(r["market_cap_lamports"]) for r in last_rows[:10]))
        newcomer_mean = top10[len(top10) // 2] - 0.4
    # The unnamed floor grows with the horizon: the longer the wait, the more
    # likely a coin born after the cut takes a place. With very little
    # history the model is untrustworthy, so the floor rises there too.
    unnamed_floor = max(0.25, min(0.45, 0.25 + 0.015 * max_horizon))
    if len(marks) < 4:
        unnamed_floor = max(unnamed_floor, 0.35)
    unnamed_cap = 0.65

    # Place-level series: the log-cap holding each place at each mark. Sizes
    # are forecast from this series directly, not from the coin simulations:
    # 10-minute jump volatility does not scale as sqrt(t) over 13 hours, and
    # the extremes of 275 random walks would pile up at any cap ceiling.
    levels = []
    for m in marks:
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        lv = [logcap_sol(r["market_cap_lamports"]) for r in rows[:places]]
        while len(lv) < places:
            lv.append(lv[-1] if lv else floor_cap)
        levels.append(lv)
    mark_times = [(parse_time(m["at"]) - t0).total_seconds() / 3600.0 for m in marks]

    def level_params(k):
        changes = []
        for i in range(1, len(levels)):
            dt = mark_times[i] - mark_times[i - 1]
            if dt > 0:
                changes.append((levels[i][k] - levels[i - 1][k], dt))
        if not changes:
            return 0.0, 0.15
        n = len(changes)
        mu = sum(d / dt for d, dt in changes) / n
        mu = mu * n / (n + 4.0)
        mu = max(-0.24, min(0.24, mu))
        var = sum(((d - mu * dt) ** 2) / dt for d, dt in changes) / n
        sig = math.sqrt(var) if var > 0 else 0.15
        return mu, sig

    level_mu = []
    sq_all = []
    for k in range(places):
        mu, _ = level_params(k)
        level_mu.append(mu)
        for i in range(1, len(levels)):
            dt = mark_times[i] - mark_times[i - 1]
            if dt > 0:
                d = levels[i][k] - levels[i - 1][k]
                sq_all.append(((d - mu * dt) ** 2) / dt)
    level_sig = math.sqrt(sum(sq_all) / len(sq_all)) if sq_all else 0.15
    level_sig = max(0.05, min(0.30, level_sig))
    last_levels = levels[-1]
    # Total level drift over the forecast window is bounded: the early
    # ramp cannot compound for 13 hours. Cap |mu| * mean_horizon at 1.5.
    level_mu = [max(-1.5, min(1.5, mu * mean_horizon)) / mean_horizon
                for mu in level_mu]

    rng = random.Random(SEED)
    mints = list(params)
    x0s = [params[m][0] for m in mints]
    mus = [params[m][1] for m in mints]
    sigs = [params[m][2] for m in mints]
    n_reads = len(reads)
    place_counts = {m: [0] * places for m in mints}
    newcomer_counts = [0] * places
    size_counts = [[0] * (len(edges) + 1) for _ in range(places)]

    def bin_of(logcap):
        if logcap < edges[0]:
            return 0
        for i in range(1, len(edges)):
            if logcap < edges[i]:
                return i
        return len(edges)

    for _ in range(N_SIMS):
        xs = list(x0s)
        for ri in range(n_reads):
            dt = dts[ri]
            sqdt = math.sqrt(dt)
            for i in range(len(mints)):
                xs[i] += mus[i] * dt + sigs[i] * sqdt * rng.gauss(0.0, 1.0)
                if xs[i] < CAP_LO:
                    xs[i] = CAP_LO
                elif xs[i] > CAP_HI:
                    xs[i] = CAP_HI
            # Rank: coins plus simulated newcomers.
            scored = [(xs[i], mints[i]) for i in range(len(mints))]
            for _ in range(N_NEWCOMERS):
                nc = rng.gauss(newcomer_mean, NEWCOMER_SD)
                if nc > CAP_HI:
                    nc = CAP_HI
                elif nc < CAP_LO:
                    nc = CAP_LO
                scored.append((nc, None))
            scored.sort(key=lambda p: -p[0])
            for k in range(places):
                cap, mint = scored[k]
                if mint is None:
                    newcomer_counts[k] += 1
                else:
                    place_counts[mint][k] += 1

    # Size simulation: each place's cap level as its own random walk, with
    # drift/volatility estimated from that place's level series. Independent
    # of the coin ranking; the score treats order and sizes independently.
    for _ in range(N_SIMS):
        lv = list(last_levels)
        for ri in range(n_reads):
            dt = dts[ri]
            sqdt = math.sqrt(dt)
            for k in range(places):
                lv[k] += level_mu[k] * dt + level_sig * sqdt * rng.gauss(0.0, 1.0)
                if lv[k] < CAP_LO:
                    lv[k] = CAP_LO
                elif lv[k] > CAP_HI:
                    lv[k] = CAP_HI
                size_counts[k][bin_of(lv[k])] += 1

    total = N_SIMS * n_reads
    probabilities = []
    for k in range(places):
        unnamed = newcomer_counts[k] / total
        unnamed = max(unnamed_floor, min(unnamed_cap, unnamed))
        target = 1.0 - unnamed
        named_total = sum(place_counts[m][k] for m in mints)
        rows = []
        for m in mints:
            c = place_counts[m][k]
            if c > 0:
                p = (c / named_total) * target if named_total > 0 else 0.0
                if p >= MIN_ROW_P:
                    rows.append((m, p))
        rows.sort(key=lambda r: (-r[1], r[0]))
        rows = rows[:MAX_ROWS_PER_PLACE]
        # Chart coverage: name the last mark's largest coins not already
        # named, at a small floor probability. The unnamed share is then
        # shared only by coins absent from the history (newcomers) instead
        # of being diluted over the whole future chart, and a fill row
        # still scores if its coin holds the place.
        named_mints = {m for m, _ in rows}
        fill = [r["mint"] for r in last_rows if r["mint"] not in named_mints]
        fill = fill[:MAX_ROWS_PER_PLACE - len(rows)]
        if fill:
            fill_mass = len(fill) * FILL_P
            scale = max(0.0, (target - fill_mass) / target) if target > 0 else 0.0
            rows = [(m, p * scale) for m, p in rows]
            rows += [(m, FILL_P) for m in fill]
        # Floor-round to 6 decimals; leftover mass joins the unnamed share.
        for m, p in rows:
            u = math.floor(p * MICRO)
            if u > 0:
                probabilities.append({"mint": m, "place": k + 1, "p": u / MICRO})

    sizes = []
    n_bins = len(edges) + 1
    # With very little history the size model is untrustworthy: flatten it
    # toward uniform instead of betting confidently on the wrong bins.
    size_mix = 0.30 if len(marks) < 4 else SIZE_UNIFORM_MIX
    for k in range(places):
        raw = [size_counts[k][b] / total for b in range(n_bins)]
        # Kernel smoothing over neighbouring bins: the exact bin is uncertain,
        # and a near miss should still score something.
        smooth = []
        for b in range(n_bins):
            v = 0.5 * raw[b]
            if b > 0:
                v += 0.25 * raw[b - 1]
            if b < n_bins - 1:
                v += 0.25 * raw[b + 1]
            smooth.append(v)
        for b in range(n_bins):
            p = (1.0 - size_mix) * smooth[b] + size_mix / n_bins
            u = math.floor(p * MICRO)
            if u > 0:
                sizes.append({"place": k + 1, "bin": b, "p": u / MICRO})

    # top5: highest total top-5 probability, ordered by expected place.
    totals = {}
    expected = {}
    for m in mints:
        tot = sum(place_counts[m])
        if tot > 0:
            totals[m] = tot
            expected[m] = sum((k + 1) * place_counts[m][k] for k in range(places)) / tot
    ranked = sorted(totals, key=lambda m: (-totals[m], expected[m], m))
    top5 = ranked[:places]
    if len(top5) < places:
        for r in last_rows:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) == places:
                break

    forecast = {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w", encoding="utf8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
