#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone trading bot (clean version) focusing on:
- MAIN ACCOUNT ONLY (no vaultAddress)
- Exit logic: Trailing ROI (10 p.p., monotonic) + hard stop of -$0.10 (unrealized PnL)
- Multiple entries allowed
- No TP/SL anywhere
- Minimal indicators & logging
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import ccxt  # type: ignore

MAIN_ACCOUNT_ADDRESS = "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"

# =========================
# Config
# =========================

@dataclass
class Config:
    SYMBOLS: List[str] = None
    TIMEFRAME: str = "15m"
    HISTORY_BARS: int = 200
    EMA_SHORT: int = 7
    EMA_LONG: int = 21
    ATR_PERIOD: int = 14
    VOL_MA_PERIOD: int = 20
    LEVERAGE: float = 10.0
    ROI_TRAIL_PPTS: float = 0.10   # 10 percentage points on *leveraged* ROI
    SLEEP_SEC: float = 10.0        # loop sleep
    DEBUG: bool = True

    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["BTC/USDC:USDC"]

# =========================
# Utilities
# =========================

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

def log(level: str, symbol: str, msg: str):
    print(f"[{level}] [{symbol}] {msg}", flush=True)

def norm_side(side: str) -> str:
    side = side.lower()
    if side in ("buy", "long"):
        return "buy"
    if side in ("sell", "short"):
        return "sell"
    return side

# =========================
# Trailing ROI Manager
# =========================

class TrailingManager:
    """
    Keeps peak (roi*lev) per symbol/side and updates a single reduceOnly stop order monotonically.
    """
    def __init__(self, exchange, leverage: float, debug: bool = True):
        self.dex = exchange
        self.leverage = max(1.0, float(leverage))
        self.debug = debug
        self._peak_lev_roi: Dict[str, float] = {}
        self._trail_order_id: Dict[str, Optional[str]] = {}
        self._trail_stop_px: Dict[str, Optional[float]] = {}

    def _key(self, symbol: str, side: str) -> str:
        return f"{symbol}:{norm_side(side)}"

    def update_and_upsert(
        self,
        symbol: str,
        side: str,
        entry_px: float,
        current_px: float,
        qty: float,
    ) -> Optional[Tuple[str, float]]:
        """
        Compute stop price from leveraged ROI and upsert reduceOnly stop order if improved.
        Returns (clientOrderId, stop_px) if an upsert happened, else None.
        """
        side = norm_side(side)
        if qty <= 0 or entry_px <= 0 or current_px <= 0:
            return None

        # ROI definition (unlevered):
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

        # Monotonic stop: LONG only increase, SHORT only decrease
        prev_px = self._trail_stop_px.get(key, None)
        improved = False
        if prev_px is None:
            improved = True
        else:
            improved = (stop_px > prev_px + 1e-12) if side == "buy" else (stop_px < prev_px - 1e-12)

        if not improved:
            if self.debug:
                log("DEBUG", symbol, f"TRAIL | no improvement. prev={prev_px}, new={stop_px:.6f}, side={side}")
            return None

        # Cancel previous if any, then create improved stop
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
            qty_abs = abs(float(qty))
            exit_side = "sell" if side == "buy" else "buy"
            o = self.dex.create_order(symbol, "market", exit_side, qty_abs, None, params)
            oid = o.get("id") or (o.get("info") or {}).get("oid")
            self._trail_order_id[key] = oid
            self._trail_stop_px[key] = float(stop_px)
            if self.debug:
                log("INFO", symbol, f"TRAIL | upsert stop to {stop_px:.6f} (qty={qty_abs})")
            return (cid, float(stop_px))
        except Exception as e:
            log("WARN", symbol, f"TRAIL | upsert failed: {type(e).__name__}: {e}")
            return None

# =========================
# Strategy (very compact)
# =========================

class EMAGradientStrategy:
    def __init__(self, dex, symbol: str, cfg: Config):
        self.dex = dex
        self.symbol = symbol
        self.cfg = cfg
        self.trailer = TrailingManager(dex, cfg.LEVERAGE, cfg.DEBUG)

    # --- Market data ---
    def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        ohlcv = self.dex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
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

    # --- Position helpers ---
    def _get_position(self) -> Optional[Dict[str, Any]]:
        try:
            pos = None
            for p in self.dex.fetch_positions([self.symbol]):
                szi = float((p.get("info") or {}).get("position", {}).get("szi") or 0.0)
                if abs(szi) > 0:
                    pos = p
                    break
            return pos
        except Exception:
            return None

    def _qty_from_usd(self, usd: float, px: float) -> float:
        if px <= 0: return 0.0
        # Cross in futures is typically quoted sizes; keep it simple:
        return float(usd) / float(px)

    # --- Entry/Exit core ---
    def step(self):
        df = self._fetch_ohlcv(self.symbol, self.cfg.TIMEFRAME, self.cfg.HISTORY_BARS)
        df = self._indicators(df)
        row = df.iloc[-1]
        px = float(row["close"])

        # ENTRY signal (simple and permissive; you can tailor later)
        ema_up = row["ema_short"] > row["ema_long"]
        ema_down = row["ema_short"] < row["ema_long"]
        vol_ok = row["volume"] >= (row["vol_ma"] or 0.0)

        pos = self._get_position()
        if pos is None:
            # Multiple entries allowed; just attempt a new one if signal hits
            usd_risk = 10.0  # example nominal exposure; adjust as needed
            qty = self._qty_from_usd(usd_risk * self.cfg.LEVERAGE, px)
            if ema_up and vol_ok:
                try:
                    self.dex.set_leverage(int(self.cfg.LEVERAGE), {"symbol": self.symbol})
                except Exception:
                    pass
                self.dex.create_order(self.symbol, "market", "buy", qty)
                log("INFO", self.symbol, f"Entered LONG qty={qty:.6f} @~{px:.6f}")
            elif ema_down and vol_ok:
                try:
                    self.dex.set_leverage(int(self.cfg.LEVERAGE), {"symbol": self.symbol})
                except Exception:
                    pass
                self.dex.create_order(self.symbol, "market", "sell", qty)
                log("INFO", self.symbol, f"Entered SHORT qty={qty:.6f} @~{px:.6f}")
            return

        # Position open → manage exits
        info = pos.get("info") or {}
        position = info.get("position") or {}
        szi = float(position.get("szi") or 0.0)
        side = "buy" if szi > 0 else "sell"
        entry_px = float(pos.get("entryPrice") or position.get("entryPx") or px)

        # Hard stop by unrealized PnL <= -$0.10
        try:
            unreal = float(pos.get("unrealizedPnl") or info.get("unrealizedPnl") or 0.0)
            if unreal <= -0.10:
                qty_abs = abs(float(pos.get("contracts") or szi))
                exit_side = "sell" if side == "buy" else "buy"
                self.dex.create_order(self.symbol, "market", exit_side, qty_abs, None, {"reduceOnly": True})
                log("WARN", self.symbol, f"Hard stop -$0.10 triggered. Closed position.")
                return
        except Exception:
            pass

        # Update trailing stop from ROI (10 p.p. on leveraged ROI, monotonic)
        qty_abs = abs(float(pos.get("contracts") or szi))
        self.trailer.update_and_upsert(self.symbol, side, entry_px, px, qty_abs)

# =========================
# Main
# =========================

def main():
    log("INFO", "WALLET", f"Operando SOMENTE na carteira principal: {MAIN_ACCOUNT_ADDRESS}")
    api_key = os.getenv("HL_API_KEY") or ""
    secret  = os.getenv("HL_API_SECRET") or ""
    password = os.getenv("HL_API_PASSWORD") or None

    # Exchange init (ccxt hyperliquid if available, else any ccxt-compatible)
    ex_id = os.getenv("EXCHANGE_ID", "hyperliquid")
    try:
        dex_class = getattr(ccxt, ex_id)
        dex = dex_class({
            "apiKey": api_key,
            "secret": secret,
            "password": password,
            "enableRateLimit": True,
            "options": {},
        })
    except Exception as e:
        raise RuntimeError(f"Failed to init exchange {ex_id}: {e}")

    cfg = Config()
    syms = cfg.SYMBOLS

    strats = [EMAGradientStrategy(dex, s, cfg) for s in syms]
    while True:
        for st in strats:
            try:
                st.step()
            except Exception as e:
                log("ERROR", st.symbol, f"step failed: {type(e).__name__}: {e}")
        time.sleep(cfg.SLEEP_SEC)

if __name__ == "__main__":
    main()
