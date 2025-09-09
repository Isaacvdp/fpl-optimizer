#!/usr/bin/env python3
"""
fpl_optimize.py — stateful, fixture-aware, spend-savvy FPL optimizer

Features
- Pulls live data from the official FPL API (bootstrap + fixtures)
- Weighted scoring with per-position normalization (toggleable)
- Fixture-aware scoring (home/away multipliers + difficulty, over N GWs)
- Optimal 15 under FPL rules (budget, positions, ≤3 per club)
- Best XI with Captain + Vice-captain (NEVER a GK)
- Saves/loads your baseline squad (current_team.json) to respect free transfers
- Suggests taking extra transfers (hits) if net XI gain (after −4/transfer) looks worthwhile
- NEW: spend controls (min spend, spend reward), GK spend cap, position bias (favor DEF/MID, cool GK)

Quick start
  pip install requests pulp
  python fpl_optimize.py --pick-xi --write squad.csv --save-current current_team.json
Weekly
  python fpl_optimize.py --use-current current_team.json --free-transfers 1 --suggest-extra 2 --pick-xi
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Set

import requests
import pulp

# ---------------- API endpoints ----------------
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

# FPL position mapping
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL squad rules
SQUAD_SIZE = 15
POSITION_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_SIZE = 11


# ---------------- Utilities ----------------
def fetch_json(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def normalize(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(x - lo) / (hi - lo) for x in values]


def chance_of_playing(e) -> float:
    """0..1 proxy for next-GW availability based on FPL flags."""
    c = e.get("chance_of_playing_next_round")
    if c is not None:
        try:
            return max(0.0, min(1.0, float(c) / 100.0))
        except Exception:
            pass
    status = e.get("status", "a")
    if status == "a":
        return 1.0
    if status == "d":
        return 0.75
    return 0.0


def detect_next_event(events: List[Dict]) -> int:
    """Return next GW id (event id). Prefer 'is_next'; otherwise first not-finished."""
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    for ev in events:
        if not ev.get("finished", False):
            return ev["id"]
    return max(ev["id"] for ev in events)


# ---------------- Fixture modelling ----------------
def build_team_fixture_factors(
    fixtures: List[Dict],
    start_event: int,
    horizon: int,
    home_mult: float,
    away_mult: float,
    diff_floor: float,
) -> Dict[int, Dict[str, float]]:
    """
    For each team_id, compute:
      - fixture_factor: avg over next <horizon> events of (loc_multiplier * difficulty_factor)
      - home_next: 1 if start_event is a HOME fixture, else 0
    Difficulty (1..5) maps linearly to [diff_floor, 1.0]
    """
    by_event = defaultdict(list)
    for f in fixtures:
        ev = f.get("event")
        if ev is not None:
            by_event[ev].append(f)

    per_team_event = defaultdict(dict)  # team_id -> event -> (loc, diff)
    for ev, rows in by_event.items():
        for f in rows:
            th, ta = f["team_h"], f["team_a"]
            dh, da = f.get("team_h_difficulty", 3), f.get("team_a_difficulty", 3)
            per_team_event[th][ev] = ("home", dh)
            per_team_event[ta][ev] = ("away", da)

    def diff_to_factor(d: int) -> float:
        d = max(1, min(5, int(d)))
        span = 1.0 - diff_floor
        # d=1 -> 1.0, d=5 -> diff_floor
        return diff_floor + span * (5 - (d - 1)) / 5.0

    out = {}
    horizon_events = [start_event + k for k in range(max(1, horizon))]
    for team_id, ev_map in per_team_event.items():
        factors = []
        for ev in horizon_events:
            if ev not in ev_map:
                continue
            loc, diff = ev_map[ev]
            loc_mult = home_mult if loc == "home" else away_mult
            factors.append(loc_mult * diff_to_factor(diff))

        fixture_factor = sum(factors) / len(factors) if factors else 1.0
        home_next = 1.0 if (start_event in ev_map and ev_map[start_event][0] == "home") else 0.0
        out[team_id] = {"fixture_factor": fixture_factor, "home_next": home_next}
    return out


# ---------------- Player assembly ----------------
def build_players(api_bootstrap: Dict, team_fx: Dict[int, Dict[str, float]]) -> List[Dict]:
    teams_by_id = {t["id"]: t for t in api_bootstrap["teams"]}
    elements = api_bootstrap["elements"]

    players = []
    for e in elements:
        team_id = e["team"]
        fx = team_fx.get(team_id, {"fixture_factor": 1.0, "home_next": 0.0})
        players.append(
            {
                "id": e["id"],
                "name": f'{e["first_name"]} {e["second_name"]}'.strip(),
                "web_name": e["web_name"],
                "position": POS_MAP.get(e["element_type"], "UNK"),
                "team_id": team_id,
                "team": teams_by_id[team_id]["short_name"],
                "price": e["now_cost"] / 10.0,
                "ep_next": safe_float(e.get("ep_next")),
                "form": safe_float(e.get("form")),
                "ppg": safe_float(e.get("points_per_game")),
                "ict": safe_float(e.get("ict_index")),
                "bps": safe_float(e.get("bps")),
                "selected_by": safe_float(e.get("selected_by_percent")),
                "minutes": safe_float(e.get("minutes")),
                "status": e.get("status"),
                "chance_next": chance_of_playing(e),
                "fixture_factor": float(fx["fixture_factor"]),
                "home_next": float(fx["home_next"]),
            }
        )
    return players


# ---------------- Scoring ----------------
def min_max_scores(players: List[Dict], fields: List[str], per_position=True) -> Dict[str, List[float]]:
    n = len(players)
    out = {f: [0.0] * n for f in fields}
    if not per_position:
        for f in fields:
            vals = [players[i][f] for i in range(n)]
            out[f] = normalize(vals)
        return out

    buckets = defaultdict(list)
    for i, p in enumerate(players):
        buckets[p["position"]].append(i)
    for f in fields:
        for pos, idxs in buckets.items():
            vals = [players[i][f] for i in idxs]
            norms = normalize(vals)
            for k, gi in enumerate(idxs):
                out[f][gi] = norms[k]
    return out


def compose_weighted_score(players: List[Dict], weights: Dict[str, float], per_position=True) -> List[float]:
    fields = list(weights.keys())
    norms = min_max_scores(players, fields, per_position=per_position)
    scores = []
    for i, _ in enumerate(players):
        s = 0.0
        for f, w in weights.items():
            s += w * norms[f][i]
        scores.append(s)
    return scores


# ---------------- Optimizer (with transfers & spend controls) ----------------
def ilp_solve_squad(
    players,
    scores,
    budget,
    max_per_team,
    locks,
    excludes,
    owned=None,
    max_transfers=None,
    spend_weight: float = 0.0,
    min_spend_ratio: float = 0.0,
    gk_spend_cap: float = None,
):
    """
    Adds:
      - spend_weight: reward for spending; objective += spend_weight * (total_spend / budget)
      - min_spend_ratio: require total_spend >= min_spend_ratio * budget
      - gk_spend_cap: cap total GK spend (e.g., 9.5)
    """
    n = len(players)
    owned = owned or [0] * n

    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]
    y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]  # 1 if new-in

    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    total_spend = pulp.lpSum(players[i]["price"] * x[i] for i in range(n))
    gk_spend = pulp.lpSum(players[i]["price"] * x[i] for i in range(n) if players[i]["position"] == "GK")

    # Objective: score + reward for spending budget
    obj = pulp.lpSum(scores[i] * x[i] for i in range(n))
    if spend_weight and spend_weight != 0.0:
        obj += spend_weight * (total_spend / budget)
    prob += obj

    # Budget ≤
    prob += total_spend <= budget

    # Optional min spend
    if min_spend_ratio and min_spend_ratio > 0:
        prob += total_spend >= min_spend_ratio * budget

    # Squad size
    prob += pulp.lpSum(x) == SQUAD_SIZE

    # Positions
    for pos, need in POSITION_REQUIREMENTS.items():
        prob += pulp.lpSum(x[i] for i in range(n) if players[i]["position"] == pos) == need

    # Max per team
    teams = set(p["team_id"] for p in players)
    for t in teams:
        prob += pulp.lpSum(x[i] for i in range(n) if players[i]["team_id"] == t) <= max_per_team

    # GK spend cap
    if gk_spend_cap is not None:
        prob += gk_spend <= gk_spend_cap

    # Locks / excludes
    name_to_idxs = defaultdict(list)
    for i, p in enumerate(players):
        name_to_idxs[p["name"].lower()].append(i)
        name_to_idxs[p["web_name"].lower()].append(i)
    for nm in locks:
        nm_l = nm.strip().lower()
        if nm_l in name_to_idxs:
            prob += pulp.lpSum(x[i] for i in name_to_idxs[nm_l]) >= 1
    for nm in excludes:
        nm_l = nm.strip().lower()
        for i in name_to_idxs.get(nm_l, []):
            prob += x[i] == 0

    # Link transfers (y)
    for i in range(n):
        oi = owned[i]
        prob += y[i] >= x[i] - oi
        prob += y[i] <= x[i]
        prob += y[i] <= 1 - oi

    if max_transfers is not None:
        prob += pulp.lpSum(y) <= max_transfers

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [i for i in range(n) if pulp.value(x[i]) > 0.5]
    transfers = int(round(sum(float(pulp.value(v) or 0.0) for v in y)))
    return selected, transfers, pulp.LpStatus[status]


# ---------------- XI selection + captain/vice (no GK) ----------------
def choose_best_xi(players: List[Dict], selected_idx: List[int]) -> Tuple[List[int], int, int]:
    """Return (xi_global_indices, captain_global_index, vice_global_index). Capt/VC NEVER GK."""
    # Build XI metric (prefer ep_next)
    metric = []
    for gi in selected_idx:
        p = players[gi]
        m = p["ep_next"] if p["ep_next"] > 0 else (0.55 * p["ppg"] + 0.35 * p["form"] + 0.10 * p["fixture_factor"])
        metric.append(m)

    by_pos = defaultdict(list)
    for local, gi in enumerate(selected_idx):
        by_pos[players[gi]["position"]].append(local)

    def take_best(pos, k):
        idxs = by_pos[pos]
        return sorted(idxs, key=lambda loc: metric[loc], reverse=True)[:k]

    xi_loc = []
    xi_loc += take_best("GK", XI_MIN["GK"])
    xi_loc += take_best("DEF", XI_MIN["DEF"])
    xi_loc += take_best("MID", XI_MIN["MID"])
    xi_loc += take_best("FWD", XI_MIN["FWD"])

    used = set(xi_loc)
    remaining = XI_SIZE - len(xi_loc)
    rest = sorted([i for i in range(len(selected_idx)) if i not in used], key=lambda loc: metric[loc], reverse=True)
    xi_loc += rest[:remaining]

    # Captain + Vice: best two NON-GK in XI
    non_gk_locs = [loc for loc in xi_loc if players[selected_idx[loc]]["position"] != "GK"]
    if non_gk_locs:
        cap_loc = max(non_gk_locs, key=lambda loc: metric[loc])
        others = [loc for loc in non_gk_locs if loc != cap_loc]
        vc_loc = max(others, key=lambda loc: metric[loc]) if others else None
    else:
        cap_loc = max(xi_loc, key=lambda loc: metric[loc])
        vc_loc = None

    xi_global = [selected_idx[loc] for loc in xi_loc]
    cap_global = selected_idx[cap_loc]
    vc_global = selected_idx[vc_loc] if vc_loc is not None else None
    return xi_global, cap_global, vc_global


def project_xi_points(players: List[Dict], selected_idx: List[int]) -> Tuple[List[int], int, int, float]:
    """(xi, cap, vc, sum_ep_next_or_fallback) used for hit evaluation."""
    xi, cap, vc = choose_best_xi(players, selected_idx)
    pts = 0.0
    for i in xi:
        p = players[i]
        pts += p["ep_next"] if p["ep_next"] > 0 else (0.55 * p["ppg"] + 0.35 * p["form"] + 0.10 * p["fixture_factor"])
    return xi, cap, vc, pts


# ---------------- Save/Load current team ----------------
def save_current_team(path: str, players: List[Dict], selected_idx: List[int], meta=None):
    data = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "player_ids": [players[i]["id"] for i in selected_idx],
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] Current team -> {path}")


def load_current_team(path: str) -> Set[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set(data.get("player_ids", []))
    if not ids:
        raise ValueError(f"No player_ids found in {path}")
    print(f"[loaded] Current team from {path} ({len(ids)} players)")
    return ids


# ---------------- CLI + main ----------------
def parse_weight_kv_list(kvs: List[str]) -> Dict[str, float]:
    out = {}
    for kv in kvs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v)
    return out


def parse_pos_bias(kvs: List[str]) -> Dict[str, float]:
    out = {}
    for kv in kvs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip().upper()
        if k in {"GK", "DEF", "MID", "FWD"}:
            out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    # Core knobs
    ap.add_argument("--budget", type=float, default=100.0, help="Total bank (e.g., 100.0)")
    ap.add_argument("--max-per-team", type=int, default=3)
    ap.add_argument("--lock", type=str, default="", help="Comma-separated names/web_names to force in")
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated names/web_names to forbid")
    ap.add_argument("--min-price", type=float, default=3.5, help="Filter out players cheaper than this")
    ap.add_argument("--min-play-prob", type=float, default=0.0, help="Filter: minimum chance to play 0..1")
    ap.add_argument("--exclude-flagged", action="store_true", help="Drop injured/suspended (status in {i,s})")
    ap.add_argument("--no-pos-norm", action="store_true", help="Disable per-position normalization")
    ap.add_argument("--pick-xi", action="store_true", help="Pick a Best XI + (no-GK) Captain + Vice")
    ap.add_argument("--write", type=str, default="", help="Write chosen squad to CSV")

    # Fixture-aware
    ap.add_argument("--gw-horizon", type=int, default=1, help="How many upcoming GWs to average over (>=1)")
    ap.add_argument("--home-mult", type=float, default=1.05, help="Multiplier for home fixtures (>1.0 boosts)")
    ap.add_argument("--away-mult", type=float, default=0.97, help="Multiplier for away fixtures (<1.0 penalizes)")
    ap.add_argument("--diff-floor", type=float, default=0.70, help="Min factor for hardest diff=5 (0.6–0.8 sensible)")

    # Weights
    ap.add_argument("--weights", type=str, default="", help="JSON mapping metric->weight")
    ap.add_argument(
        "--w",
        nargs="*",
        default=[],
        help="Inline weights, e.g.: ep_next=1.0 form=0.4 ppg=0.4 ict=0.2 bps=0.1 avail=0.6 fixture=0.8 home_next=0.2",
    )

    # Stateful team management
    ap.add_argument("--save-current", type=str, default="", help="Save chosen 15 to a JSON file (e.g., current_team.json)")
    ap.add_argument("--use-current", type=str, default="", help="Load your saved 15 to limit transfers from")
    ap.add_argument("--free-transfers", type=int, default=1, help="Number of free transfers available this week")
    ap.add_argument("--suggest-extra", type=int, default=0, help="Evaluate up to this many extra transfers beyond free")
    ap.add_argument("--hit-cost", type=float, default=4.0, help="Point cost per extra transfer for suggestion math")

    # NEW: spend & position controls
    ap.add_argument("--spend-weight", type=float, default=0.25,
                    help="Objective reward for spending the budget (0 = off). Roughly adds +spend_weight when you spend 100% of budget.")
    ap.add_argument("--min-spend-ratio", type=float, default=0.0,
                    help="Force spending at least this fraction of budget (e.g., 0.97 = spend >= 97%% of budget). 0 = off")
    ap.add_argument("--gk-spend-cap", type=float, default=None,
                    help="Max combined spend on both GKs (e.g., 9.5). Leave empty to disable.")
    ap.add_argument("--pos-bias", nargs="*", default=[],
                    help='Position multipliers, e.g.: GK=-0.2 DEF=0.10 MID=0.15 FWD=0.0 (applied as (1+bias) to each player score)')

    args = ap.parse_args()

    # Default weights (good baseline)
    weights = {
        "ep_next": 1.0,
        "form": 0.4,
        "ppg": 0.4,
        "ict": 0.25,
        "bps": 0.15,
        "avail": 0.6,     # maps to chance_next
        "fixture": 0.6,   # maps to fixture_factor
        "home_next": 0.0, # small optional nudge
    }
    if args.weights:
        with open(args.weights, "r", encoding="utf-8") as f:
            weights.update(json.load(f))
    weights.update(parse_weight_kv_list(args.w))

    # Fetch API data
    boot = fetch_json(BOOTSTRAP)
    fixtures = fetch_json(FIXTURES)
    next_ev = detect_next_event(boot["events"])

    # Team-level fixture factors
    team_fx = build_team_fixture_factors(
        fixtures=fixtures,
        start_event=next_ev,
        horizon=args.gw_horizon,
        home_mult=args.home_mult,
        away_mult=args.away_mult,
        diff_floor=args.diff_floor,
    )

    # Players table
    players = build_players(boot, team_fx)

    # Filters
    filtered = []
    for p in players:
        if p["price"] < args.min_price:
            continue
        if args.exclude_flagged and p["status"] in {"i", "s"}:
            continue
        if p["chance_next"] < args.min_play_prob:
            continue
        filtered.append(p)
    players = filtered
    if not players:
        print("No players left after filters; relax constraints.")
        return

    # Map weight names to data fields
    effective_weights = {}
    for k, v in weights.items():
        if k == "avail":
            effective_weights["chance_next"] = v
        elif k == "fixture":
            effective_weights["fixture_factor"] = v
        else:
            effective_weights[k] = v

    # Scores (normalized per position by default)
    scores = compose_weighted_score(players, effective_weights, per_position=not args.no_pos_norm)

    # Apply position bias (if any)
    pos_bias = parse_pos_bias(args.pos_bias)
    if pos_bias:
        for i, p in enumerate(players):
            scores[i] *= (1.0 + pos_bias.get(p["position"], 0.0))

    # Owned mapping if using a saved current team
    owned = [0] * len(players)
    if args.use_current:
        owned_ids = load_current_team(args.use_current)
        id_to_idx = {p["id"]: i for i, p in enumerate(players)}
        for pid in owned_ids:
            i = id_to_idx.get(pid)
            if i is not None:
                owned[i] = 1

    # Locks / excludes
    locks = [s for s in args.lock.split(",") if s.strip()]
    excludes = [s for s in args.exclude.split(",") if s.strip()]

    # Primary plan: respect free transfers if we have a baseline
    max_tr = max(0, int(args.free_transfers)) if args.use_current else None
    selected, transfers, status = ilp_solve_squad(
        players, scores, args.budget, args.max_per_team, locks, excludes,
        owned=owned, max_transfers=max_tr,
        spend_weight=args.spend_weight,
        min_spend_ratio=args.min_spend_ratio,
        gk_spend_cap=args.gk_spend_cap
    )
    if not selected:
        print(f"Solver status: {status}. No feasible squad.")
        return

    # Output squad
    total_cost = sum(players[i]["price"] for i in selected)
    bank = args.budget - total_cost
    print(f"Solver: {status} | Picked {len(selected)} | Cost £{total_cost:.1f}m | Bank £{bank:.1f}m | Transfers used={transfers}")

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

    def row(i):
        p = players[i]
        return {
            "pos": p["position"],
            "name": p["name"],
            "web": p["web_name"],
            "team": p["team"],
            "price": p["price"],
            "ep_next": round(p["ep_next"], 2),
            "form": round(p["form"], 2),
            "ppg": round(p["ppg"], 2),
            "ict": round(p["ict"], 2),
            "bps": int(p["bps"]),
            "avail": round(p["chance_next"], 2),
            "fixture": round(p["fixture_factor"], 3),
            "home_next": int(p["home_next"]),
            "score": round(scores[i], 4),
        }

    chosen = sorted([row(i) for i in selected], key=lambda r: (pos_order[r["pos"]], -r["score"]))
    for r in chosen:
        print(
            f"{r['pos']:>3} | {r['name']:<22} {r['team']:<3} | £{r['price']:.1f}m "
            f"| ep_next {r['ep_next']:<4} form {r['form']:<4} ppg {r['ppg']:<4} ict {r['ict']:<5} "
            f"| avail {r['avail']:<4} | fixture {r['fixture']:<4} home {r['home_next']} | score {r['score']}"
        )

    if args.pick_xi:
        xi, cap, vc = choose_best_xi(players, selected)
        xi_rows = sorted([row(i) for i in xi], key=lambda r: (pos_order[r["pos"]], -r["ep_next"]))
        print("\nBest XI (primarily ep_next):")
        for r in xi_rows:
            print(f"{r['pos']:>3} | {r['web']:<18} {r['team']:<3} | ep_next {r['ep_next']:<4} | price £{r['price']:.1f}m")
        print(f"\nCaptain: {players[cap]['web_name']}")
        if vc is not None:
            print(f"Vice-captain: {players[vc]['web_name']}")

    # Save as current team if requested
    if args.save_current:
        save_current_team(args.save_current, players, selected, meta={"budget": args.budget})

    # Suggest extra transfers (hits) beyond free transfers
    if args.use_current and args.suggest_extra > 0:
        _, _, _, base_pts = project_xi_points(players, selected)
        print(f"\n[plan] With ≤{max_tr} transfer(s): projected XI pts ≈ {base_pts:.2f}  (transfers={transfers})")
        best_net_gain = 0.0
        best_plan = (base_pts, max_tr, selected)

        for extra in range(1, args.suggest_extra + 1):
            cap_k = max_tr + extra
            sel_k, tr_k, st_k = ilp_solve_squad(
                players, scores, args.budget, args.max_per_team, locks, excludes,
                owned=owned, max_transfers=cap_k,
                spend_weight=args.spend_weight,
                min_spend_ratio=args.min_spend_ratio,
                gk_spend_cap=args.gk_spend_cap
            )
            if not sel_k:
                continue
            _, _, _, pts_k = project_xi_points(players, sel_k)
            net_gain = (pts_k - base_pts) - extra * args.hit_cost
            print(
                f"[plan] With ≤{cap_k} transfer(s): XI pts ≈ {pts_k:.2f}  raw Δ={pts_k - base_pts:+.2f} "
                f"| hit cost={extra * args.hit_cost:.1f} → net {net_gain:+.2f}"
            )
            if net_gain > best_net_gain:
                best_net_gain = net_gain
                best_plan = (pts_k, cap_k, sel_k)

        if best_plan[1] > max_tr and best_net_gain > 0:
            print(
                f"\n👉 Suggestion: take up to **{best_plan[1] - max_tr} extra transfer(s)** (hits) — "
                f"net gain ≈ +{best_net_gain:.2f} XI pts after hits."
            )
        else:
            print("\n👍 Suggestion: stick to **free transfer(s) only** this week.")


if __name__ == "__main__":
    main()
