"""
SP500 Option Trade-Flow Simulator
=================================

This program generates a synthetic stream of executed SP500
option trades from a day of real trades (Databento glbx-mdp3 TBBO export).  T

How the synthetic flow is built
--------------------------------
1. WHICH option         -- selected with probability proportional to that
   option's actual daily VOLUME
3. SIZE (contracts)      -- bootstrapped from the selected option's OWN observed
   trade sizes
2. BUY vs SELL           -- an even 50/50 split.
4. TIME OF DAY           -- a configurable intraday model (see --time-model):
       empirical : bootstrap the real, size-weighted trade timestamps.  This is
                   the actual market profile -- heavy at the US open and into the
                   close, lighter midday, plus ~30% overnight Globex flow.
       ushape    : a parametric open/close-weighted curve over regular hours
                   (RTH).  Use when you want a clean, controllable U-shape.
       uniform   : a flat distribution (provided for comparison only).

Usage
-----
    python trade_flow_simulator.py                         # defaults
    python trade_flow_simulator.py --n-trades 1000 --seed 7
    python trade_flow_simulator.py --time-model ushape --rth-only
    python trade_flow_simulator.py --time-model uniform

Input:
    data/raw/glbx-mdp3-20260601.tbbo.csv        the real day of trades

Output:
    data/simulated/simulated_executed_trades.csv   the generated executed option trades

Each row carries everything a downstream pricer/hedger needs: timestamp, symbol,
root, expiry code, expiry date (3rd Friday), call/put, strike, side, quantity and
a reference execution price (the line's volume-weighted traded price).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
# This script lives in <project>/notebooks/; data sits in <project>/data/.
# Inputs are read from data/raw and generated outputs are written to
# data/simulated, resolved from this file so the script runs from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SIMULATED_DIR = PROJECT_ROOT / "data" / "simulated"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Futures month codes -> calendar month.
MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

# "ESM6 C7700" -> root, month-code, year-digit, C/P, strike
_SYMBOL_RE = re.compile(r"^(ES)([FGHJKMNQUVXZ])(\d) ([CP])(\d+(?:\.\d+)?)$")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    csv_path: str = str(RAW_DIR / "glbx-mdp3-20260601.tbbo.csv")
    out_path: str = str(SIMULATED_DIR / "simulated_executed_trades.csv")
    n_trades: int = 5000            # ~ one real day of prints; tight volume match
    seed: int = 42

    # --- time-of-day model ---
    time_model: str = "empirical"        # empirical | ushape | uniform
    rth_only: bool = False               # restrict trades to regular hours
    et_utc_offset_hours: int = -4        # US Eastern offset (EDT in June 2026)
    rth_open_et: str = "09:30"           # regular-hours session (US Eastern)
    rth_close_et: str = "16:15"
    time_jitter_sec: int = 120           # +/- jitter on bootstrapped times

    # --- parametric U-shape parameters (time_model="ushape") ---
    ushape_base: float = 1.0             # midday baseline weight
    ushape_open_peak: float = 2.0        # extra weight at the open
    ushape_close_peak: float = 3.0       # extra weight at the close (heavier)
    ushape_width: float = 0.15           # peak width as fraction of session


# --------------------------------------------------------------------------- #
# Symbol parsing
# --------------------------------------------------------------------------- #
def third_friday(year: int, month: int) -> date:
    """Standard monthly expiry: the third Friday of the month."""
    d = date(year, month, 1)
    first_friday = 1 + (4 - d.weekday()) % 7      # Monday=0 ... Friday=4
    return date(year, month, first_friday + 14)


def parse_symbol(symbol: str):
    """Return dict for a vanilla ES outright option, or None for combos/other."""
    m = _SYMBOL_RE.match(symbol)
    if not m:
        return None
    root, mc, yd, cp, strike = m.groups()
    return {
        "root": root,
        "expiry_code": f"{root}{mc}{yd}",
        "opt_type": "C" if cp == "C" else "P",
        "strike": float(strike),
        "expiry_date": third_friday(2020 + int(yd), MONTH_CODES[mc]),
    }


def _et_hhmm_to_utc_sec(hhmm: str, offset_h: int) -> int:
    """'09:30' US-Eastern -> seconds-of-day in UTC."""
    h, m = (int(x) for x in hhmm.split(":"))
    return (h * 3600 + m * 60) - offset_h * 3600


# --------------------------------------------------------------------------- #
# Market data: load, filter to vanilla options, aggregate per-line volume
# --------------------------------------------------------------------------- #
class MarketData:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        raw = pd.read_csv(cfg.csv_path)
        raw = raw[raw["action"] == "T"].copy()        # keep trades only
        raw["ts"] = pd.to_datetime(raw["ts_event"], utc=True, format="ISO8601")

        parsed = raw["symbol"].map(parse_symbol)
        keep = parsed.notna()
        df = raw[keep].copy()
        meta = pd.json_normalize(parsed[keep].tolist())
        meta.index = df.index
        df = pd.concat([df, meta], axis=1)

        self.n_total_rows = len(raw)
        self.n_vanilla_rows = len(df)
        self.n_combo_rows = self.n_total_rows - self.n_vanilla_rows
        self.trade_date = df["ts"].dt.date.min()
        df["sec_of_day"] = (df["ts"].dt.hour * 3600
                            + df["ts"].dt.minute * 60
                            + df["ts"].dt.second).astype(int)
        self.trades = df

        # ---- per option line: volume, VWAP price, and the option's own size pool ----
        lines = []
        size_pools = {}
        for sym, g in df.groupby("symbol"):
            volume = int(g["size"].sum())
            lines.append({
                "symbol": sym, "root": g["root"].iloc[0],
                "expiry_code": g["expiry_code"].iloc[0],
                "expiry_date": g["expiry_date"].iloc[0],
                "opt_type": g["opt_type"].iloc[0], "strike": g["strike"].iloc[0],
                "volume": volume,
                "vwap_price": float((g["price"] * g["size"]).sum() / volume),
                "n_obs": len(g),
            })
            size_pools[sym] = g["size"].to_numpy()
        self.lines = pd.DataFrame(lines).set_index("symbol")

        # Each option's own observed trade sizes. A simulated trade's size is
        # bootstrapped from the SAME option it lands on, so a line only ever shows
        # block sizes it actually traded.
        self.size_pools = size_pools


# --------------------------------------------------------------------------- #
# Intraday time-of-day sampler
# --------------------------------------------------------------------------- #
class TimeSampler:
    """Draws trade times-of-day (seconds in UTC) under the chosen model."""

    def __init__(self, market: MarketData, cfg: Config):
        self.cfg = cfg
        self.open_sec = _et_hhmm_to_utc_sec(cfg.rth_open_et, cfg.et_utc_offset_hours)
        self.close_sec = _et_hhmm_to_utc_sec(cfg.rth_close_et, cfg.et_utc_offset_hours)

        if cfg.time_model == "empirical":
            t = market.trades
            if cfg.rth_only:
                t = t[(t["sec_of_day"] >= self.open_sec)
                      & (t["sec_of_day"] <= self.close_sec)]
            self._times = t["sec_of_day"].to_numpy()
            w = t["size"].to_numpy(dtype=float)       # weight timing by volume
            self._weights = w / w.sum()
        elif cfg.time_model == "ushape":
            mins = np.arange(self.open_sec, self.close_sec, 60)
            u = (mins - self.open_sec) / (self.close_sec - self.open_sec)
            w = (cfg.ushape_base
                 + cfg.ushape_open_peak * np.exp(-((u) / cfg.ushape_width) ** 2)
                 + cfg.ushape_close_peak * np.exp(-((1 - u) / cfg.ushape_width) ** 2))
            self._minutes = mins
            self._weights = w / w.sum()
        elif cfg.time_model != "uniform":
            raise ValueError(f"unknown time-model: {cfg.time_model}")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cfg = self.cfg
        if cfg.time_model == "empirical":
            base = rng.choice(self._times, size=n, p=self._weights)
            sec = base + rng.integers(-cfg.time_jitter_sec,
                                      cfg.time_jitter_sec + 1, size=n)
            lo, hi = (self.open_sec, self.close_sec) if cfg.rth_only else (0, 86399)
            return np.clip(sec, lo, hi)
        if cfg.time_model == "ushape":
            mins = rng.choice(self._minutes, size=n, p=self._weights)
            return mins + rng.integers(0, 60, size=n)
        # uniform
        lo, hi = (self.open_sec, self.close_sec) if cfg.rth_only else (0, 86399)
        return rng.integers(lo, hi + 1, size=n)


# --------------------------------------------------------------------------- #
# Trade simulator
# --------------------------------------------------------------------------- #
class TradeSimulator:
    def __init__(self, market: MarketData, cfg: Config):
        self.market = market
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.time_sampler = TimeSampler(market, cfg)

    def simulate(self) -> pd.DataFrame:
        cfg, mkt, rng = self.cfg, self.market, self.rng
        lines = mkt.lines

        # (1) WHICH option: probability proportional to each option's DAILY
        # VOLUME
        symbols = lines.index.to_numpy()
        vol = lines["volume"].to_numpy(dtype=float)
        chosen = rng.choice(len(symbols), size=cfg.n_trades, p=vol / vol.sum())

        # (4) WHEN: intraday time-of-day model (independent of which option).
        secs = self.time_sampler.sample(rng, cfg.n_trades)
        midnight = datetime.combine(mkt.trade_date, datetime.min.time(),
                                    tzinfo=timezone.utc)

        recs = []
        for k, idx in enumerate(chosen):
            sym = symbols[idx]
            # (3) SIZE: bootstrap from THIS option's own observed trade sizes, so
            # it only ever shows block sizes it actually traded.
            qty = int(rng.choice(mkt.size_pools[sym]))
            row = lines.loc[sym]
            side = "BUY" if rng.random() < 0.5 else "SELL"  # (2) even 50/50
            sec = int(secs[k])
            recs.append({
                "timestamp": midnight + timedelta(seconds=sec), "sec_of_day": sec,
                "symbol": sym, "root": row["root"],
                "expiry_code": row["expiry_code"], "expiry_date": row["expiry_date"],
                "opt_type": row["opt_type"], "strike": row["strike"],
                "side": side, "quantity": qty, "exec_price": row["vwap_price"],
            })

        out = (pd.DataFrame(recs).sort_values("timestamp").reset_index(drop=True))
        out.insert(0, "trade_id", np.arange(1, len(out) + 1))
        return out


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def _hourly_profile(df: pd.DataFrame, offset_h: int) -> pd.Series:
    """Volume share (%) by US-Eastern hour."""
    et_hour = ((df["sec_of_day"] // 3600) + offset_h) % 24
    vol = df.groupby(et_hour)["quantity"].sum() if "quantity" in df \
        else df.groupby(et_hour)["size"].sum()
    return (100 * vol / vol.sum()).round(1)


def main():
    p = argparse.ArgumentParser(description="SP500 option trade-flow simulator")
    p.add_argument("--csv", default=Config.csv_path,
                   help="input trades CSV (default: data/raw/...)")
    p.add_argument("--out", default=Config.out_path,
                   help="output CSV (default: data/simulated/...)")
    p.add_argument("--n-trades", type=int, default=Config.n_trades)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--time-model", choices=["empirical", "ushape", "uniform"],
                   default=Config.time_model)
    p.add_argument("--rth-only", action="store_true",
                   help="restrict trades to regular trading hours (US Eastern)")
    args = p.parse_args()

    cfg = Config(csv_path=args.csv, out_path=args.out, n_trades=args.n_trades,
                 seed=args.seed, time_model=args.time_model, rth_only=args.rth_only)

    print("=" * 70)
    print("  SP500 OPTION TRADE-FLOW SIMULATOR")
    print("=" * 70)

    mkt = MarketData(cfg)
    print(f"Source file   : {cfg.csv_path}")
    print(f"Trade date    : {mkt.trade_date}")
    print(f"Rows (T)      : {mkt.n_total_rows:,}  "
          f"(vanilla {mkt.n_vanilla_rows:,} / combos excluded {mkt.n_combo_rows:,})")
    print(f"Option lines  : {len(mkt.lines):,}")
    print(f"Time model    : {cfg.time_model}"
          f"{' (RTH only)' if cfg.rth_only else ''}")

    sim = TradeSimulator(mkt, cfg)
    trades = sim.simulate()
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"\nGenerated {len(trades):,} executed trades "
          f"-> {out_path}")
    print(f"  Buy/Sell split  : {(trades.side == 'BUY').mean():.1%} / "
          f"{(trades.side == 'SELL').mean():.1%}")
    print(f"  Total contracts : {int(trades['quantity'].sum()):,}")

    # Verify the distribution matches real-world volume (and isn't uniform).
    real_share = 100 * mkt.lines["volume"] / mkt.lines["volume"].sum()
    g = trades.groupby("symbol")
    sim_trade_share = (100 * g.size() / len(trades)).reindex(mkt.lines.index).fillna(0)
    sim_vol_share = (100 * g["quantity"].sum() / trades["quantity"].sum()
                     ).reindex(mkt.lines.index).fillna(0)
    print(f"\n  Distribution match vs real volume share "
          f"(uniform would give each line {100/len(mkt.lines):.3f}%):")
    print(f"    trade-frequency corr = {real_share.corr(sim_trade_share):.3f}")
    print(f"    contracts      corr = {real_share.corr(sim_vol_share):.3f}")
    print("\n  Top 6 options by real volume (% shares):")
    top = (pd.DataFrame({"real_vol_%": real_share,
                         "sim_trades_%": sim_trade_share,
                         "sim_vol_%": sim_vol_share})
             .fillna(0.0).sort_values("real_vol_%", ascending=False).head(6))
    print(top.round(2).to_string())

    print("\n  Intraday profile -- volume %% by US-Eastern hour "
          "(open 09:30, close 16:00):")
    sim_prof = _hourly_profile(trades, cfg.et_utc_offset_hours)
    real_prof = _hourly_profile(mkt.trades, cfg.et_utc_offset_hours)
    comp = (pd.DataFrame({"real_%": real_prof, "sim_%": sim_prof})
              .fillna(0.0).sort_index())
    print(comp.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
