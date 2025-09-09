# fpl-optimizer

---

## Overview

The script fetches live data from the official FPL API (`bootstrap-static` and `fixtures`).
It builds player records including:

* Standard FPL stats (`ep_next`, `form`, `points_per_game`, `ict_index`, `bps`, etc.)
* Availability proxy (`chance_next` from FPL injury flags)
* Fixture-aware metrics (`fixture_factor`, `home_next`)

It uses integer linear programming to pick an optimal 15-man squad under FPL rules:

* 2 GK, 5 DEF, 5 MID, 3 FWD
* Budget (default £100.0m)
* ≤ 3 players per real club

Optionally, it chooses a Best XI and assigns a captain and vice-captain.

---

## Weightings

### How scoring works

* Each chosen metric is **min–max normalized** to the range \[0, 1].
* By default, normalization is **within each position** (GK/DEF/MID/FWD). Disable with `--no-pos-norm`.
* Each normalized metric is multiplied by your chosen weight:

```
score(player) = Σ (weight_m × normalized_metric_m)
```

The optimizer maximizes the total score across the squad.

### Weight keys available

* `ep_next` → FPL’s expected points for the next GW
* `form` → recent form
* `ppg` → points per game
* `ict` → ICT Index
* `bps` → Bonus Point System points (season)
* `avail` → chance of playing (0–1 proxy)
* `fixture` → fixture factor (home/away multipliers + difficulty, averaged across `--gw-horizon`)
* `home_next` → binary (1 if next fixture is home, else 0)

### Changing weights

Inline:

```bash
python fpl_optimize.py --w ep_next=1.0 form=0.4 ppg=0.4 fixture=0.8 avail=0.6
```

JSON file:

```json
{
  "ep_next": 1.2,
  "form": 0.4,
  "ppg": 0.4,
  "fixture": 0.8,
  "avail": 0.6
}
```

Run with:

```bash
python fpl_optimize.py --weights weights.json
```

### Fixture adjustments

* `--gw-horizon N` → number of future GWs to consider
* `--home-mult` / `--away-mult` → multipliers for home vs away
* `--diff-floor` → floor for hardest fixtures (difficulty=5)

---

## Captain & Vice-Captain Selection

* **XI selection**: 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD, fill rest with best XI metric.
* **XI metric**: If `ep_next` > 0, use that; otherwise `0.55*ppg + 0.35*form + 0.10*fixture_factor`.
* **Captain** = highest XI metric among outfield players (DEF/MID/FWD).
* **Vice-captain** = second-highest outfielder.
* Falls back to best overall only if no outfielders exist (should never happen).

---

## Quick Reference: Arguments & Weights

### Scoring & Weights

| Flag                  | What it controls        | Typical values            | Notes                                 |
| --------------------- | ----------------------- | ------------------------- | ------------------------------------- |
| `--weights FILE.json` | Load saved weights      | file path                 | Clean & repeatable                    |
| `--w k=v ...`         | Inline weight overrides | `ep_next=1.1 fixture=0.9` | Ratios matter; metrics normalized 0–1 |

**Weight keys:**

* `ep_next` (0.8–1.4) → short-term signal
* `form` (0.2–0.6) → hot streaks
* `ppg` (0.2–0.6) → consistency
* `ict` (0.2–0.4) → attacking involvement
* `bps` (0.1–0.4) → bonus potential
* `avail` (0.5–1.0) → avoid risks
* `fixture` (0.5–1.2) → fixture swings
* `home_next` (0.0–0.2) → small nudge

### Fixture Model

| Flag           | Controls              | Typical   | Notes             |
| -------------- | --------------------- | --------- | ----------------- |
| `--gw-horizon` | # GWs to average      | 1–3       | 2 is balanced     |
| `--home-mult`  | Home multiplier       | 1.05–1.12 | >1 boosts home    |
| `--away-mult`  | Away multiplier       | 0.92–0.98 | <1 penalizes away |
| `--diff-floor` | Hardest fixture floor | 0.65–0.75 | 0.70 common       |

### Spend Control

| Flag                | Controls               | Typical   | Notes               |
| ------------------- | ---------------------- | --------- | ------------------- |
| `--spend-weight`    | Reward for spending    | 0.2–0.4   | Gentle push         |
| `--min-spend-ratio` | Minimum spend fraction | 0.95–0.99 | Leave little ITB    |
| `--gk-spend-cap`    | Max spend on 2 GKs     | 9.0–9.5   | Avoid expensive GKs |

### Position Bias

| Flag         | Controls             | Example                      | Notes               |
| ------------ | -------------------- | ---------------------------- | ------------------- |
| `--pos-bias` | Position multipliers | `MID=0.15 DEF=0.10 GK=-0.20` | Applied as (1+bias) |

### Team & Risk Constraints

* `--budget` (100.0) → total budget
* `--max-per-team` (3) → club limit
* `--lock "Name"` → force include player(s)
* `--exclude "Name"` → force exclude player(s)
* `--min-price` (3.5–4.5) → drop cheap fodder
* `--min-play-prob` (0.5–0.7) → drop rotation risks
* `--exclude-flagged` → drop injured/suspended

### Transfers (Weekly Use)

| Flag               | Controls                  | Typical             | Notes                |
| ------------------ | ------------------------- | ------------------- | -------------------- |
| `--save-current`   | Save squad JSON           | `current_team.json` | Baseline             |
| `--use-current`    | Load baseline JSON        | `current_team.json` | Transfer limits      |
| `--free-transfers` | Free transfers            | 1–2                 | Hard cap             |
| `--suggest-extra`  | Try extra transfers       | 1–3                 | Shows net after hits |
| `--hit-cost`       | Points per extra transfer | 4                   | FPL default          |

### Output

* `--pick-xi` → print Best XI + Captain/Vice
* `--write FILE.csv` → export squad
* `--no-pos-norm` → disable per-position normalization

---

## Ready-Made Examples

**First time (build & save baseline):**

```bash
python fpl_optimize.py \
  --weights weights_first_time.json \
  --gw-horizon 2 --home-mult 1.10 --away-mult 0.95 --diff-floor 0.70 \
  --spend-weight 0.30 --min-spend-ratio 0.97 \
  --gk-spend-cap 9.5 \
  --pos-bias MID=0.15 DEF=0.10 GK=-0.20 \
  --min-play-prob 0.6 --exclude-flagged \
  --pick-xi --write squad_initial.csv \
  --save-current current_team.json
```

**Weekly (respect 1 FT, suggest hits):**

```bash
python fpl_optimize.py \
  --use-current current_team.json \
  --weights weights_fixture_chaser.json \
  --gw-horizon 2 --home-mult 1.10 --away-mult 0.95 --diff-floor 0.70 \
  --free-transfers 1 --suggest-extra 2 --hit-cost 4 \
  --spend-weight 0.30 --min-spend-ratio 0.97 --gk-spend-cap 9.5 \
  --pos-bias MID=0.15 DEF=0.10 GK=-0.20 \
  --pick-xi --write squad_weekN.csv
```

---
