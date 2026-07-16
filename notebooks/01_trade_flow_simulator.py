"""
SP500 Option Trade-Flow Simulator
=================================

This program generates a synthetic stream of executed SP500 (ES) option trades
from real trades pulled live from Databento (GLBX.MDP3, TBBO schema, ``ES.OPT``
parent symbology).  Instead of a single day, it now spans a configurable date
range (default 2026-01-01 .. 2026-05-29) and builds the synthetic flow one
trading day at a time, then concatenates the days into one stream.

How the synthetic flow is built (per trading day)
-------------------------------------------------
1. WHICH option         -- selected with probability proportional to that
   option's actual daily VOLUME
3. SIZE (contracts)      -- bootstrapped from the selected option's OWN observed
   trade sizes
2. BUY vs SELL           -- an even 50/50 split.
4. TIME OF DAY           -- bootstrapped from the real, size-weighted trade
   timestamps within regular trading hours (09:30-16:15 ET).  This reproduces
   the actual intraday profile -- heavy at the US open and into the close,
   lighter midday.

Only regular-trading-hours prints are used: the day's tape is filtered to the
RTH session before anything else, so the option universe, volumes, size pools
and timing are all RTH-only.

Data source
-----------
Each trading day in the range is fetched from Databento and cached to
``data/raw/glbx-mdp3-YYYYMMDD.tbbo.csv`` (same name/format as the bundled
sample).  A cached day is reused; only missing days hit the API.

Usage
-----
    python 04_trade_flow_simulator.py                          # full Jan-May 2026
    python 04_trade_flow_simulator.py --start 2026-01-01 --end 2026-01-31
    python 04_trade_flow_simulator.py --n-trades-per-day 5000  # fixed daily count
    python 04_trade_flow_simulator.py --start 2026-03-01 --seed 7

Input:
    Databento GLBX.MDP3 TBBO (ES.OPT), cached under data/raw/

Output:
    data/simulated/simulated_executed_trades.csv   the generated executed option trades

Each row carries everything a downstream pricer/hedger needs: timestamp, symbol,
root, expiry code, expiry date (3rd Friday), call/put, strike, side, quantity and
a reference execution price (the line's volume-weighted traded price).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
# This script lives in <project>/notebooks/; data sits in <project>/data/.
# Inputs are read from / cached to data/raw and generated outputs are written to
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

# US Eastern, used to map UTC trade times to the trading session (handles
# the EST/EDT change that falls inside the Jan-May window).
ET = ZoneInfo("America/New_York")

# 2026 U.S. equity/options market closures inside the study window.
MARKET_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25),
}


def et_offset_hours(d: date) -> int:
    """UTC offset (whole hours) for US Eastern on date ``d`` -- -5 (EST) or -4 (EDT)."""
    off = datetime(d.year, d.month, d.day, 12, tzinfo=ET).utcoffset()
    return int(off.total_seconds() // 3600)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- data source: Databento GLBX.MDP3, ES options, parent symbology ---
    start_date: str = "2026-01-01"          # inclusive
    end_date: str = "2026-05-29"            # inclusive
    dataset: str = "GLBX.MDP3"
    parent_symbol: str = "ES.OPT"
    schema: str = "tbbo"
    api_key_env: str = "DATABENTO_API_KEY"
    raw_dir: str = str(RAW_DIR)

    out_path: str = str(SIMULATED_DIR / "simulated_executed_trades.csv")
    # Simulated executed trades per trading day. None -> derive from mm_fraction.
    n_trades_per_day: int | None = None
    # Fraction of real daily vanilla volume we assume we market-make.
    # Used only when n_trades_per_day is None.
    mm_fraction: float = 0.30
    seed: int = 42

    # --- regular trading hours (US Eastern); the tape is filtered to this
    #     session and intraday timing is bootstrapped from it ---
    rth_open_et: str = "09:30"
    rth_close_et: str = "16:15"
    time_jitter_sec: int = 120           # +/- jitter on bootstrapped times


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
# Data acquisition: fetch a day of TBBO from Databento (cached under data/raw)
# --------------------------------------------------------------------------- #
def trading_days(cfg: Config) -> list[date]:
    """Weekdays in [start, end] excluding U.S. market holidays."""
    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in MARKET_HOLIDAYS:
            days.append(d)
        d += timedelta(days=1)
    return days


def raw_csv_path(cfg: Config, day: date) -> Path:
    return Path(cfg.raw_dir) / f"glbx-mdp3-{day:%Y%m%d}.tbbo.csv"


def make_databento_client(cfg: Config):
    """Construct a Databento Historical client from the env-var API key."""
    import databento as db  # imported lazily so cached-only runs need no install

    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise SystemExit(
            f"Set the {cfg.api_key_env} environment variable to your Databento "
            f"API key (no day in the requested range is cached under {cfg.raw_dir})."
        )
    return db.Historical(key)


def load_raw_day(cfg: Config, day: date, client=None) -> pd.DataFrame:
    """Return one day's raw TBBO rows, fetching from Databento if not cached."""
    path = raw_csv_path(cfg, day)
    if path.exists():
        return pd.read_csv(path)

    if client is None:
        client = make_databento_client(cfg)
    nxt = day + timedelta(days=1)
    store = client.timeseries.get_range(
        dataset=cfg.dataset, symbols=cfg.parent_symbol, stype_in="parent",
        schema=cfg.schema, start=day.isoformat(), end=nxt.isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # pretty_ts -> ISO8601 timestamps, pretty_px -> prices in dollars, map_symbols
    # -> human option symbol; this matches the bundled sample CSV exactly.
    store.to_csv(path, pretty_ts=True, pretty_px=True, map_symbols=True)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Market data: filter to vanilla options, aggregate per-line volume (one day)
# --------------------------------------------------------------------------- #
class MarketData:
    def __init__(self, raw: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        raw = raw[raw["action"] == "T"].copy()        # keep trades only
        raw["ts"] = pd.to_datetime(raw["ts_event"], utc=True, format="ISO8601")
        raw["sec_of_day"] = (raw["ts"].dt.hour * 3600
                             + raw["ts"].dt.minute * 60
                             + raw["ts"].dt.second).astype(int)
        self.trade_date = raw["ts"].dt.date.min() if len(raw) else None

        # Restrict to regular trading hours (US Eastern, DST-aware) up front, so
        # the option universe, volumes, size pools and timing are all RTH-only.
        if self.trade_date is not None:
            off = et_offset_hours(self.trade_date)
            open_sec = _et_hhmm_to_utc_sec(cfg.rth_open_et, off)
            close_sec = _et_hhmm_to_utc_sec(cfg.rth_close_et, off)
            raw = raw[(raw["sec_of_day"] >= open_sec)
                      & (raw["sec_of_day"] <= close_sec)]

        parsed = raw["symbol"].map(parse_symbol)
        keep = parsed.notna()
        df = raw[keep].copy()
        meta = pd.json_normalize(parsed[keep].tolist())
        meta.index = df.index
        df = pd.concat([df, meta], axis=1)

        self.n_total_rows = len(raw)
        self.n_vanilla_rows = len(df)
        self.n_combo_rows = self.n_total_rows - self.n_vanilla_rows
        self.trades = df

        # Detect bid/ask columns (present in TBBO exports).
        has_ba = "bid_px_00" in df.columns and "ask_px_00" in df.columns
        self.has_bid_ask = has_ba

        # ---- per option line: volume, VWAP price, and the option's own trade pools ----
        lines = []
        size_pools: dict[str, np.ndarray] = {}
        bid_pools: dict[str, np.ndarray] = {}
        ask_pools: dict[str, np.ndarray] = {}
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
            if has_ba:
                bid_pools[sym] = g["bid_px_00"].to_numpy()
                ask_pools[sym] = g["ask_px_00"].to_numpy()
        self.lines = pd.DataFrame(lines).set_index("symbol")

        # Pools aligned to positional order of ``lines``.  Size, bid, and ask are
        # sampled from the same index so they stay correlated (spread varies by event).
        self.size_pools = [size_pools[s] for s in self.lines.index]
        self.bid_pools = ([bid_pools[s] for s in self.lines.index] if has_ba else None)
        self.ask_pools = ([ask_pools[s] for s in self.lines.index] if has_ba else None)


# --------------------------------------------------------------------------- #
# Intraday time-of-day sampler
# --------------------------------------------------------------------------- #
class TimeSampler:
    """Bootstraps trade times-of-day (seconds in UTC) from the day's real,
    size-weighted tape."""

    def __init__(self, market: MarketData, cfg: Config, offset_h: int):
        self.cfg = cfg
        self.open_sec = _et_hhmm_to_utc_sec(cfg.rth_open_et, offset_h)
        self.close_sec = _et_hhmm_to_utc_sec(cfg.rth_close_et, offset_h)

        t = market.trades                             # already RTH-filtered
        self._times = t["sec_of_day"].to_numpy()
        w = t["size"].to_numpy(dtype=float)           # weight timing by volume
        self._weights = w / w.sum()

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cfg = self.cfg
        base = rng.choice(self._times, size=n, p=self._weights)
        sec = base + rng.integers(-cfg.time_jitter_sec,
                                  cfg.time_jitter_sec + 1, size=n)
        return np.clip(sec, self.open_sec, self.close_sec)


# --------------------------------------------------------------------------- #
# Volume-targeting helper
# --------------------------------------------------------------------------- #
def _target_trade_count(mkt: MarketData, mm_fraction: float) -> int:
    """Return N such that expected simulated contract volume ≈ mm_fraction × real vanilla volume."""
    total_real_vol = int(mkt.lines["volume"].sum())
    target_vol = mm_fraction * total_real_vol
    vol = mkt.lines["volume"].to_numpy(dtype=float)
    p = vol / vol.sum()
    mean_sizes = np.array([pool.mean() for pool in mkt.size_pools])
    expected_mean_size = float(np.dot(p, mean_sizes))
    if expected_mean_size <= 0:
        return 1
    return max(1, round(target_vol / expected_mean_size))


# --------------------------------------------------------------------------- #
# Trade simulator (one trading day; shares a single RNG across days)
# --------------------------------------------------------------------------- #
class TradeSimulator:
    def __init__(self, market: MarketData, cfg: Config, rng: np.random.Generator):
        self.market = market
        self.cfg = cfg
        self.rng = rng
        self.offset_h = et_offset_hours(market.trade_date)
        self.time_sampler = TimeSampler(market, cfg, self.offset_h)

    def simulate(self, n_trades: int) -> pd.DataFrame:
        mkt, rng = self.market, self.rng
        lines = mkt.lines

        # (1) WHICH option: probability proportional to each option's DAILY VOLUME.
        symbols = lines.index.to_numpy()
        vol = lines["volume"].to_numpy(dtype=float)
        chosen = rng.choice(len(symbols), size=n_trades, p=vol / vol.sum())

        # (3) SIZE / BID / ASK: bootstrap from THIS option's own observed trade
        # events.  A single random index into the pool gives (size, bid, ask) from
        # the same real event, preserving their natural correlation.
        qty = np.empty(n_trades, dtype=np.int64)
        bid = np.full(n_trades, np.nan)
        ask = np.full(n_trades, np.nan)
        order = np.argsort(chosen, kind="stable")
        uniq, starts = np.unique(chosen[order], return_index=True)
        bounds = list(starts) + [n_trades]
        for j, line_idx in enumerate(uniq):
            seg = order[bounds[j]:bounds[j + 1]]
            pool_sz = len(mkt.size_pools[line_idx])
            idxs = rng.integers(0, pool_sz, size=len(seg))
            qty[seg] = mkt.size_pools[line_idx][idxs]
            if mkt.has_bid_ask:
                bid[seg] = mkt.bid_pools[line_idx][idxs]
                ask[seg] = mkt.ask_pools[line_idx][idxs]

        # (2) BUY vs SELL: even 50/50.
        sides = np.where(rng.random(n_trades) < 0.5, "BUY", "SELL")

        # (4) WHEN: intraday timing bootstrapped from the real, size-weighted
        # tape (independent of which option).
        secs = self.time_sampler.sample(rng, n_trades).astype(int)
        midnight = pd.Timestamp(mkt.trade_date, tz="UTC")

        return pd.DataFrame({
            "timestamp": midnight + pd.to_timedelta(secs, unit="s"),
            "sec_of_day": secs,
            "symbol": symbols[chosen],
            "root": lines["root"].to_numpy()[chosen],
            "expiry_code": lines["expiry_code"].to_numpy()[chosen],
            "expiry_date": lines["expiry_date"].to_numpy()[chosen],
            "opt_type": lines["opt_type"].to_numpy()[chosen],
            "strike": lines["strike"].to_numpy()[chosen],
            "side": sides,
            "quantity": qty,
            "bid": bid,
            "ask": ask,
            "exec_price": lines["vwap_price"].to_numpy()[chosen],
        })


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> pd.DataFrame:
    days = trading_days(cfg)
    rng = np.random.default_rng(cfg.seed)
    client = None

    per_day = []
    real_vol_by_symbol: dict[str, int] = {}
    real_hour_vol = np.zeros(24)            # real size-volume by US-Eastern hour
    tot_rows = tot_vanilla = tot_combo = 0

    print(f"Trading days  : {len(days)}  ({cfg.start_date} .. {cfg.end_date})")
    print(f"Source        : Databento {cfg.dataset} / {cfg.parent_symbol} / "
          f"{cfg.schema}  (cache: {cfg.raw_dir})")
    print("-" * 70)

    for d in days:
        if not raw_csv_path(cfg, d).exists() and client is None:
            client = make_databento_client(cfg)
        raw = load_raw_day(cfg, d, client)
        mkt = MarketData(raw, cfg)
        if mkt.n_vanilla_rows == 0:
            print(f"  {d}  no vanilla trades -- skipped")
            continue

        n = cfg.n_trades_per_day or _target_trade_count(mkt, cfg.mm_fraction)
        per_day.append(TradeSimulator(mkt, cfg, rng).simulate(n))

        # Accumulate real-tape references for the end-of-run validation report.
        for sym, v in mkt.lines["volume"].items():
            real_vol_by_symbol[sym] = real_vol_by_symbol.get(sym, 0) + int(v)
        et_hour = ((mkt.trades["sec_of_day"].to_numpy() // 3600)
                   + et_offset_hours(d)) % 24
        np.add.at(real_hour_vol, et_hour, mkt.trades["size"].to_numpy())
        tot_rows += mkt.n_total_rows
        tot_vanilla += mkt.n_vanilla_rows
        tot_combo += mkt.n_combo_rows
        real_vol = int(mkt.lines["volume"].sum())
        print(f"  {d}  RTH(T) {mkt.n_total_rows:6,}  vanilla {mkt.n_vanilla_rows:6,}"
              f"  real_vol {real_vol:7,}  lines {len(mkt.lines):4,}  -> sim {n:6,}")

    if not per_day:
        raise SystemExit("No tradable days produced -- nothing to write.")

    trades = (pd.concat(per_day, ignore_index=True)
                .sort_values("timestamp").reset_index(drop=True))
    trades.insert(0, "trade_id", np.arange(1, len(trades) + 1))

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)

    _report(cfg, trades, real_vol_by_symbol, real_hour_vol,
            tot_rows, tot_vanilla, tot_combo, out_path)
    return trades


def _report(cfg, trades, real_vol_by_symbol, real_hour_vol,
            tot_rows, tot_vanilla, tot_combo, out_path):
    has_ba = not trades["bid"].isna().all()
    print("-" * 70)
    print(f"Generated {len(trades):,} executed trades -> {out_path}")
    print(f"  Span            : {trades['timestamp'].min().date()} .. "
          f"{trades['timestamp'].max().date()}")
    print(f"  Real RTH prints : {tot_rows:,}  "
          f"(vanilla {tot_vanilla:,} / combos excluded {tot_combo:,})")
    print(f"  MM fraction     : {cfg.mm_fraction:.0%} of real vanilla volume"
          + (" (overridden by --n-trades-per-day)" if cfg.n_trades_per_day else ""))
    print(f"  Buy/Sell split  : {(trades.side == 'BUY').mean():.1%} / "
          f"{(trades.side == 'SELL').mean():.1%}")
    print(f"  Total contracts : {int(trades['quantity'].sum()):,}")
    print(f"  Bid/Ask in CSV  : {'yes' if has_ba else 'no (bid_px_00/ask_px_00 absent from raw)'}")

    # Distribution match: simulated vs real per-symbol volume share over the period.
    real_vol = pd.Series(real_vol_by_symbol, dtype=float)
    real_share = 100 * real_vol / real_vol.sum()
    g = trades.groupby("symbol")
    sim_trade_share = (100 * g.size() / len(trades)).reindex(real_vol.index).fillna(0)
    sim_vol_share = (100 * g["quantity"].sum() / trades["quantity"].sum()
                     ).reindex(real_vol.index).fillna(0)
    print(f"\n  Distribution match vs real volume share "
          f"(uniform would give each line {100 / len(real_vol):.4f}%):")
    print(f"    trade-frequency corr = {real_share.corr(sim_trade_share):.3f}")
    print(f"    contracts      corr = {real_share.corr(sim_vol_share):.3f}")
    print("\n  Top 6 options by real volume (% shares):")
    top = (pd.DataFrame({"real_vol_%": real_share,
                         "sim_trades_%": sim_trade_share,
                         "sim_vol_%": sim_vol_share})
             .fillna(0.0).sort_values("real_vol_%", ascending=False).head(6))
    print(top.round(2).to_string())

    print("\n  Intraday profile -- volume %% by US-Eastern hour (RTH):")
    et_hour_sim = trades["timestamp"].dt.tz_convert(ET).dt.hour
    sim_hour = trades.groupby(et_hour_sim)["quantity"].sum()
    sim_share = 100 * sim_hour / sim_hour.sum()
    real_hour_share = pd.Series(100 * real_hour_vol / real_hour_vol.sum(),
                                index=range(24))
    comp = (pd.DataFrame({"real_%": real_hour_share, "sim_%": sim_share})
              .fillna(0.0).sort_index().round(1))
    comp = comp[(comp["real_%"] > 0) | (comp["sim_%"] > 0)]   # RTH hours only
    comp.index.name = "et_hour"
    print(comp.to_string())
    print("\nDone.")


def main():
    p = argparse.ArgumentParser(description="SP500 option trade-flow simulator")
    p.add_argument("--start", default=Config.start_date,
                   help="first date, inclusive (YYYY-MM-DD)")
    p.add_argument("--end", default=Config.end_date,
                   help="last date, inclusive (YYYY-MM-DD)")
    p.add_argument("--out", default=Config.out_path,
                   help="output CSV (default: data/simulated/...)")
    p.add_argument("--n-trades-per-day", type=int, default=Config.n_trades_per_day,
                   help="fixed simulated trades per day (overrides --mm-fraction)")
    p.add_argument("--mm-fraction", type=float, default=Config.mm_fraction,
                   help="fraction of real daily vanilla volume to market-make "
                        "(default: 0.20; ignored when --n-trades-per-day is set)")
    p.add_argument("--seed", type=int, default=Config.seed)
    args = p.parse_args()

    cfg = Config(start_date=args.start, end_date=args.end, out_path=args.out,
                 n_trades_per_day=args.n_trades_per_day,
                 mm_fraction=args.mm_fraction, seed=args.seed)

    print("=" * 70)
    print("  SP500 OPTION TRADE-FLOW SIMULATOR")
    print("=" * 70)
    run(cfg)


if __name__ == "__main__":
    main()
