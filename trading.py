#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import ccxt  # type: ignore

MAIN_ACCOUNT_ADDRESS = "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
MAX_CANDLES = 50  # enforce fetch_ohlcv limit per symbol

# -------------------------
# Helpers (env & logging)
# -------------------------

def log(level: str, symbol: str, msg: str):
    print(f"[{level}] [{symbol}] {msg}", flush=True)

def _get_env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default

LOG_SIGNALS = (os.getenv('LOG_SIGNALS', '1') != '0')
ENTRY_RULE = os.getenv('ENTRY_RULE', 'ema_and_vol')  # ema_and_vol | ema_only | always
MIN_TRADE_INTERVAL_SEC = _env_int('MIN_TRADE_INTERVAL_SEC', 60)

def norm_side(side: str) -> str:
    s = (side or "").lower()
    if s in ("buy", "long"): return "buy"
    if s in ("sell", "short"): return "sell"
    return s

def ema(a: np.ndarray, span: int) -> np.ndarray:
    if len(a) == 0:
        return a
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    tr = np.concatenate([[np.nan], tr])
    return pd.Series(tr).rolling(period).mean().to_numpy()

# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    SYMBOLS: List[str] = None
    TIMEFRAME: str = "15m"
    HISTORY_BARS: int = 50
    EMA_SHORT: int = 7
    EMA_LONG: int = 21
    ATR_PERIOD: int = 14
    VOL_MA_PERIOD: int = 20
    LEVERAGE: float = 10.0
    ROI_TRAIL_PPTS: float = 0.10   # 10 p.p. on leveraged ROI
    SLEEP_SEC: float = 10.0
    DEBUG: bool = True

    def __post_init__(self):
        if self.SYMBOLS is None:
            # Default to widely supported symbol for most exchanges
            self.SYMBOLS = _get_env_list("SYMBOLS", "BTC/USDT")
        self.TIMEFRAME = os.getenv("TIMEFRAME", self.TIMEFRAME)
        self.HISTORY_BARS = _env_int("HISTORY_BARS", self.HISTORY_BARS)
        self.LEVERAGE = _env_float("LEVERAGE", self.LEVERAGE)
        self.ROI_TRAIL_PPTS = _env_float("ROI_TRAIL_PPTS", self.ROI_TRAIL_PPTS)
        self.SLEEP_SEC = _env_float("SLEEP_SEC", self.SLEEP_SEC)
        self.DEBUG = (os.getenv("DEBUG", "1") != "0")

# -------------------------
# Trailing manager
# -------------------------

class TrailingManager:
    def __init__(self, exchange, leverage: float, debug: bool = True):
        self.dex = exchange
        self.leverage = max(1.0, float(leverage))
        self.debug = debug
        self._peak_lev_roi: Dict[str, float] = {}
        self._trail_order_id: Dict[str, Optional[str]] = {}
        self._trail_stop_px: Dict[str, Optional[float]] = {}

    def _key(self, symbol: str, side: str) -> str:
        return f"{symbol}:{norm_side(side)}"

    def update_and_upsert(self, symbol: str, side: str, entry_px: float, current_px: float, qty: float) -> Optional[Tuple[str, float]]:
        side = norm_side(side)
        if qty <= 0 or entry_px <= 0 or current_px <= 0:
            return None

        # ROI (unlevered)
        if side == "buy":
            roi = (current_px / entry_px) - 1.0
        else:
            roi = (entry_px / current_px) - 1.0

        lev_roi = roi * self.leverage
        key = self._key(symbol, side)
        peak = self._peak_lev_roi.get(key, lev_roi)
        if lev_roi > peak:
            peak = lev_roi
            self._peak_lev_roi[key] = peak

        stop_lev_roi = peak - 0.10  # 10 p.p. trailing
        stop_roi = stop_lev_roi / self.leverage
        if side == "buy":
            stop_px = entry_px * (1.0 + stop_roi)
        else:
            stop_px = entry_px * (1.0 - stop_roi)

        prev_px = self._trail_stop_px.get(key, None)
        improved = False
        if prev_px is None:
            improved = True
        else:
            improved = (stop_px > prev_px + 1e-12) if side == "buy" else (stop_px < prev_px - 1e-12)

        if not improved:
            if self.debug:
                log("DEBUG", symbol, f"TRAIL no improvement | prev={prev_px} new={stop_px:.6f} side={side}")
            return None

        # Upsert reduceOnly stop order
        cid = f"TRAILROI-{symbol.replace('/', '-')}"
        try:
            oid_prev = self._trail_order_id.get(key)
            if oid_prev:
                try:
                    self.dex.cancel_order(oid_prev, symbol)
                except Exception:
                    pass
            params = {
                "reduceOnly": True,
                "type": "stop_market",
                "stopPrice": float(stop_px),
                "clientOrderId": cid,
            }
            exit_side = "sell" if side == "buy" else "buy"
            qty_abs = abs(float(qty))
            o = self.dex.create_order(symbol, "market", exit_side, qty_abs, None, params)
            oid = (o.get("id") if isinstance(o, dict) else None) or ((o.get("info") or {}).get("oid") if isinstance(o, dict) else None)
            self._trail_order_id[key] = oid
            self._trail_stop_px[key] = float(stop_px)
            if self.debug:
                log("INFO", symbol, f"TRAIL upsert -> {stop_px:.6f} (qty={qty_abs})")
            return (cid, float(stop_px))
        except Exception as e:
            log("WARN", symbol, f"TRAIL upsert failed: {type(e).__name__}: {e}")
            return None

# -------------------------
# Strategy
# -------------------------

class EMAGradientStrategy:
    def __init__(self, dex, symbol: str, cfg: Config):
        self.dex = dex
        self.symbol = symbol
        self.cfg = cfg
        self.trailer = TrailingManager(dex, cfg.LEVERAGE, cfg.DEBUG)
        self._last_entry_ts: float = 0.0

    def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        lim = min(int(limit), int(MAX_CANDLES))
        log("INFO", symbol, f"Fetching OHLCV tf={timeframe} limit={lim}")
        data = self.dex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lim)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        a = df["close"].to_numpy(float)
        df["ema_short"] = ema(a, self.cfg.EMA_SHORT)
        df["ema_long"]  = ema(a, self.cfg.EMA_LONG)
        h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
        df["atr"] = atr(h, l, c, self.cfg.ATR_PERIOD)
        df["vol_ma"] = df["volume"].rolling(self.cfg.VOL_MA_PERIOD).mean()
        return df

    def _get_position(self) -> Optional[Dict[str, Any]]:
        try:
            pos = None
            positions = self.dex.fetch_positions([self.symbol])
            for p in positions:
                szi = float((p.get("info") or {}).get("position", {}).get("szi") or 0.0)
                if abs(szi) > 0:
                    pos = p
                    break
            return pos
        except Exception:
            return None

    def _qty_from_usd(self, usd: float, px: float) -> float:
        if px <= 0: return 0.0
        return float(usd) / float(px)

    def step(self):
        # Data + indicators
        df = self._fetch_ohlcv(self.symbol, self.cfg.TIMEFRAME, self.cfg.HISTORY_BARS)  # clamped to MAX_CANDLES
        df = self._indicators(df)
        row = df.iloc[-1]
        px = float(row["close"])

        # Signals
        ema_up = row['ema_short'] > row['ema_long']
        ema_down = row['ema_short'] < row['ema_long']
        vol_ok = row['volume'] >= (row['vol_ma'] or 0.0)
        if LOG_SIGNALS:
            log('DEBUG', self.symbol, (
                f"px={px:.6f} ema_s={row['ema_short']:.6f} ema_l={row['ema_long']:.6f} "
                f"ema_up={ema_up} ema_down={ema_down} vol={row['volume']} vol_ma={row['vol_ma']} vol_ok={vol_ok}"
            ))

        pos = self._get_position()
        if pos is None:
            now = time.time()
            if now - self._last_entry_ts < MIN_TRADE_INTERVAL_SEC:
                if self.cfg.DEBUG:
                    log('DEBUG', self.symbol, f'Skipping entry due to MIN_TRADE_INTERVAL_SEC={MIN_TRADE_INTERVAL_SEC}')
                return
            usd_risk = _env_float('USD_RISK_PER_TRADE', 10.0)
            qty = self._qty_from_usd(usd_risk * self.cfg.LEVERAGE, px)
            should_long = should_short = False
            if ENTRY_RULE == 'always':
                should_long = bool(ema_up)
                should_short = (not ema_up and ema_down)
            elif ENTRY_RULE == 'ema_only':
                should_long = ema_up
                should_short = ema_down
            else:
                should_long = ema_up and vol_ok
                should_short = ema_down and vol_ok
            if should_long and qty > 0:
                try:
                    if hasattr(self.dex, 'set_leverage'):
                        self.dex.set_leverage(int(self.cfg.LEVERAGE), {'symbol': self.symbol})
                except Exception as e:
                    log('WARN', self.symbol, f'set_leverage failed: {e}')
                self.dex.create_order(self.symbol, 'market', 'buy', qty)
                self._last_entry_ts = now
                log('INFO', self.symbol, f'Entered LONG qty={qty:.6f} ~{px:.6f}')
                return
            if should_short and qty > 0:
                try:
                    if hasattr(self.dex, 'set_leverage'):
                        self.dex.set_leverage(int(self.cfg.LEVERAGE), {'symbol': self.symbol})
                except Exception as e:
                    log('WARN', self.symbol, f'set_leverage failed: {e}')
                self.dex.create_order(self.symbol, 'market', 'sell', qty)
                self._last_entry_ts = now
                log('INFO', self.symbol, f'Entered SHORT qty={qty:.6f} ~{px:.6f}')
                return
            if self.cfg.DEBUG:
                log('DEBUG', self.symbol, f'No entry signal (ENTRY_RULE={ENTRY_RULE}).')
            return

        # Position open → manage exits
            log('DEBUG', self.symbol, (
                f"px={px:.6f} ema_s={row['ema_short']:.6f} ema_l={row['ema_long']:.6f} "
                f"ema_up={ema_up} ema_down={ema_down} vol={row['volume']} vol_ma={row['vol_ma']} vol_ok={vol_ok}"
            ))

        pos = self._get_position()
        if pos is None:
            # Throttle entries
            now = time.time()
            if now - self._last_entry_ts < MIN_TRADE_INTERVAL_SEC:
                return
            usd_risk = _env_float('USD_RISK_PER_TRADE', 10.0)
            qty = self._qty_from_usd(usd_risk * self.cfg.LEVERAGE, px)

            def _enter(side: str):
                try:
                    if hasattr(self.dex, 'set_leverage'):
                        self.dex.set_leverage(int(self.cfg.LEVERAGE), {'symbol': self.symbol})
                except Exception as e:
                    log('WARN', self.symbol, f'set_leverage failed: {e}')
                self.dex.create_order(self.symbol, 'market', side, qty)
                log('INFO', self.symbol, f'Entered {side.upper()} qty={qty:.6f} ~{px:.6f}')
                self._last_entry_ts = now

            if ENTRY_RULE == 'always':
                _enter('buy' if ema_up else 'sell')
                return
            elif ENTRY_RULE == 'ema_only':
                if ema_up:
                    _enter('buy'); return
                if ema_down:
                    _enter('sell'); return
            else:  # ema_and_vol
                if ema_up and vol_ok:
                    _enter('buy'); return
                if ema_down and vol_ok:
                    _enter('sell'); return
            return

        info = pos.get("info") or {}
        position = info.get("position") or {}
        szi = float(position.get("szi") or 0.0)
        side = "buy" if szi > 0 else "sell"
        entry_px = float(pos.get("entryPrice") or position.get("entryPx") or px)

        # Hard stop: unrealizedPnl <= -$0.10
        try:
            unreal = float(pos.get("unrealizedPnl") or info.get("unrealizedPnl") or 0.0)
            if unreal <= -0.10:
                qty_abs = abs(float(pos.get("contracts") or szi))
                exit_side = "sell" if side == "buy" else "buy"
                self.dex.create_order(self.symbol, "market", exit_side, qty_abs, None, {"reduceOnly": True})
                log("WARN", self.symbol, "Hard stop -$0.10 triggered. Closed.")
                return
        except Exception:
            pass

        # Trailing update (monotonic)
        qty_abs = abs(float(pos.get("contracts") or szi))
        self.trailer.update_and_upsert(self.symbol, side, entry_px, px, qty_abs)

# -------------------------
# Main
# -------------------------

def main():
    log("INFO", "WALLET", f"Operando SOMENTE na carteira principal: {MAIN_ACCOUNT_ADDRESS}")
    ex_id = os.getenv("EXCHANGE_ID", "binance")  # default to a widely available exchange
    api_key = os.getenv("HL_API_KEY") or os.getenv("API_KEY") or ""
    secret  = os.getenv("HL_API_SECRET") or os.getenv("API_SECRET") or ""
    password = os.getenv("HL_API_PASSWORD") or os.getenv("API_PASSWORD") or None

    # Exchange init with timeout and markets load
    if not hasattr(ccxt, ex_id):
        raise RuntimeError(f'Exchange id \"{ex_id}\" not found in ccxt.')
    dex = getattr(ccxt, ex_id)({
        "apiKey": api_key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "timeout": _env_int("CCXT_TIMEOUT_MS", 10000),
        "options": {},
    })
    log("INFO", "BOOT", f"ccxt.{ex_id} created. Loading markets...")
    dex.load_markets()
    log("INFO", "BOOT", f"Markets loaded: {len(dex.markets)}")

    cfg = Config()
    strats = [EMAGradientStrategy(dex, s, cfg) for s in cfg.SYMBOLS]

    while True:
        for st in strats:
            try:
                st.step()
            except Exception as e:
                log("ERROR", st.symbol, f"step failed: {type(e).__name__}: {e}")
        log("INFO", "HEARTBEAT", time.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(cfg.SLEEP_SEC)

if __name__ == "__main__":
    main()
