# trailing_exit_patch.py
# -*- coding: utf-8 -*-
"""
Trailing-only exit for your strategy (no TP/SL).
- Trailing stop by leveraged ROI (10 percentage points), monotonic improvement.
- Hard stop at unrealizedPnl <= -$0.10.
- Hyperliquid fix: market orders require a price -> we set price accordingly and
  also support dex.options["defaultSlippage"] (env HL_DEFAULT_SLIPPAGE).
"""

from __future__ import annotations
from typing import Any, Optional, Tuple
import math, os

# === Trailing ROI State ===
_TRAIL_STATE = {
    "peak_lev_roi": {},   # key: f"{symbol}:{side}" -> float
    "stop_px": {},        # key -> float
    "order_id": {},       # key -> str|None
}

def _norm_side(side: str) -> str:
    s = (side or "").lower()
    if s in ("buy","long"): return "buy"
    if s in ("sell","short"): return "sell"
    return s

def _is_hl(dex) -> bool:
    try:
        return (getattr(dex, "id", "") or "").lower().startswith("hyperliquid")
    except Exception:
        return False

def _ensure_hl_slippage(dex) -> None:
    # set defaultSlippage if missing (percent as fraction, e.g. 0.05 = 5%)
    try:
        if not hasattr(dex, "options") or not isinstance(dex.options, dict):
            dex.options = {}
        dex.options.setdefault("defaultSlippage", float(os.getenv("HL_DEFAULT_SLIPPAGE", "0.05")))
    except Exception:
        pass

def _current_mark(dex, symbol: str, fallback: float) -> float:
    try:
        t = dex.fetch_ticker(symbol) or {}
        v = (t.get("last") or t.get("close") or (t.get("info") or {}).get("markPx") or fallback)
        return float(v)
    except Exception:
        return float(fallback)

def _price_for_market(dex, symbol: str, px: float) -> Optional[float]:
    # Hyperliquid requires price to compute max slippage
    if _is_hl(dex):
        _ensure_hl_slippage(dex)
        return float(px)
    return None

class TrailingROIManager:
    def __init__(self, dex):
        self.dex = dex

    def _k(self, symbol: str, side: str) -> str:
        return f"{symbol}:{_norm_side(side)}"

    def update(self, *, symbol: str, side: str, entry_px: float, current_px: float, lev: float, qty: float, trail_pp: float = 0.10):
        if entry_px <= 0 or current_px <= 0 or lev <= 0 or qty <= 0:
            return None
        side = _norm_side(side)
        if side not in ("buy","sell"): return None

        # ROI não alavancado
        if side == "buy":
            roi = (current_px/entry_px) - 1.0
        else:
            roi = (entry_px/current_px) - 1.0
        lev_roi = roi * float(lev)

        key = self._k(symbol, side)
        peak = _TRAIL_STATE["peak_lev_roi"].get(key, lev_roi)
        if lev_roi > peak:
            peak = lev_roi
            _TRAIL_STATE["peak_lev_roi"][key] = peak

        stop_lev_roi = peak - float(trail_pp)
        stop_roi = stop_lev_roi / float(lev)
        stop_px = entry_px * (1.0 + stop_roi) if side == "buy" else entry_px * (1.0 - stop_roi)

        prev = _TRAIL_STATE["stop_px"].get(key)
        improved = (prev is None) or ((stop_px > prev + 1e-12) if side=="buy" else (stop_px < prev - 1e-12))
        if not improved:
            return None

        # Upsert reduceOnly stop
        exit_side = "sell" if side == "buy" else "buy"
        qty_abs = float(abs(qty))

        # Cancela anterior (se houver)
        oid_prev = _TRAIL_STATE["order_id"].get(key)
        if oid_prev:
            try: self.dex.cancel_order(oid_prev, symbol)
            except Exception: pass

        # Build params
        price_arg = _price_for_market(self.dex, symbol, stop_px)
        params_variants = [
            {"reduceOnly": True, "type": "stop_market", "stopPrice": float(stop_px), "clientOrderId": f"TRAILROI-{symbol.replace('/', '-')}"},
            {"reduceOnly": True, "type": "stop", "triggerPrice": float(stop_px), "stopLossPrice": float(stop_px), "clientOrderId": f"TRAILROI-{symbol.replace('/', '-')}"},
        ]

        last_exc = None
        for params in params_variants:
            try:
                o = self.dex.create_order(symbol, "market", exit_side, qty_abs, price_arg, params)
                oid = None
                if isinstance(o, dict):
                    oid = o.get("id") or (o.get("info") or {}).get("oid")
                _TRAIL_STATE["order_id"][key] = oid
                _TRAIL_STATE["stop_px"][key] = float(stop_px)
                return float(stop_px)
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc

# --- Helpers ---

def _safe_entry_price_from_order(order: dict, fallback_px: float) -> float:
    try:
        info = order.get("info") or {}
        filled = info.get("filled") or {}
        if "avgPx" in filled and filled["avgPx"] is not None:
            return float(filled["avgPx"])
        if "avgPx" in info and info["avgPx"] is not None:
            return float(info["avgPx"])
    except Exception:
        pass
    for k in ("average", "price"):
        v = order.get(k)
        if v is not None:
            try: return float(v)
            except Exception: pass
    return float(fallback_px)

def _get_pos_size_and_leverage(dex, symbol: str, vault=None) -> Tuple[float, float, float, str]:
    try:
        positions = dex.fetch_positions([symbol])
        for p in positions:
            info = p.get("info") or {}
            posi = info.get("position") or {}
            szi = float(posi.get("szi") or p.get("contracts") or 0.0)
            if abs(szi) <= 0: 
                continue
            side = "buy" if szi > 0 else "sell"
            lev = float(posi.get("leverage", {}).get("value") if isinstance(posi.get("leverage"), dict) else posi.get("leverage") or p.get("leverage") or 10.0)
            entry = float(posi.get("entryPx") or p.get("entryPrice") or 0.0)
            return abs(szi), lev, entry, side
    except Exception:
        pass
    return 0.0, 1.0, 0.0, ""

def close_if_unrealized_pnl_breaches(dex, symbol: str, *, vault=None, threshold: float = -0.10) -> bool:
    """
    Fecha a posição se unrealizedPnl <= threshold (USD).
    Em Hyperliquid, define 'price' mesmo em ordens market.
    """
    try:
        positions = dex.fetch_positions([symbol])
        for p in positions:
            info = p.get("info") or {}
            unreal = float(p.get("unrealizedPnl") or info.get("unrealizedPnl") or 0.0)
            szi = float((info.get("position") or {}).get("szi") or p.get("contracts") or 0.0)
            if unreal <= float(threshold) and abs(szi) > 0:
                side = "sell" if szi > 0 else "buy"
                price_arg = _price_for_market(dex, symbol, _current_mark(dex, symbol, p.get("entryPrice") or 0.0))
                dex.create_order(symbol, "market", side, float(abs(szi)), price_arg, {"reduceOnly": True})
                return True
    except Exception:
        pass
    return False

# === Public API ===

def ensure_tpsl_for_position(dex, symbol, *, vault, retries: int = 2, price_tol_pct: float = 0.001):
    """
    Trailing-only + hard stop -$0.10. TP/SL fixos desativados.
    """
    # 1) Hard stop
    if close_if_unrealized_pnl_breaches(dex, symbol, vault=vault, threshold=-0.10):
        return {"ok": True, "reason": "hard_stop"}

    # 2) Estado da posição
    qty, lev, entry, side = _get_pos_size_and_leverage(dex, symbol, vault=vault)
    if qty <= 0 or entry <= 0 or not side:
        return {"ok": False, "reason": "no_position", "qty": qty, "lev": lev, "entry": entry}

    # 3) Preço atual
    current_px = _current_mark(dex, symbol, entry)

    # 4) Atualiza trailing (monotônico)
    mgr = TrailingROIManager(dex)
    stop_px = mgr.update(symbol=symbol, side=side, entry_px=float(entry), current_px=float(current_px), lev=float(lev or 1.0), qty=float(qty), trail_pp=0.10)
    return {"ok": True, "reason": "trail_updated", "stop_px": stop_px, "qty": qty, "lev": lev, "entry": entry, "side": side}

# === No-OP stub to disable legacy TP/SL ===
def _place_tp_sl_orders_idempotent(*args, **kwargs):
    return {"disabled": True}
