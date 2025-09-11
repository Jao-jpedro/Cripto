from __future__ import annotations

import os
import sys
import json
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import time


# =============================
# Exchange layer (Hyperliquid stub)
# =============================


@dataclass
class Position:
    side: Optional[str] = None  # 'LONG' | 'SHORT' | None
    qty: float = 0.0
    entry_px: float = 0.0
    entry_ts: Optional[pd.Timestamp] = None
    bars_held: int = 0
    mfe: float = 0.0
    mae: float = 0.0
    trail_px: Optional[float] = None

    def is_open(self) -> bool:
        return self.side in ("LONG", "SHORT") and self.qty > 0

    def update_excursions(self, close_px: float):
        if not self.is_open():
            return
        ret = (close_px - self.entry_px) / self.entry_px if self.side == "LONG" else (self.entry_px - close_px) / self.entry_px
        self.mfe = max(self.mfe, ret)
        self.mae = max(self.mae, -ret)


class ExchangeClient:
    def __init__(self, *, private_key: str, vault_address: str, owner: str, fee_bps: float, slippage_bps: float):
        self.private_key = private_key
        self.vault_address = vault_address
        self.owner = owner
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        # In live mode, instantiate SDK/CCXT here. This is a backtest-friendly stub.
        self.live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
        self._dex = None
        if self.live:
            try:
                import ccxt  # type: ignore
                self._dex = ccxt.hyperliquid({
                    "walletAddress": self.vault_address,
                    "privateKey": self.private_key,
                    # Ensure we operate on perps by default
                    "options": {"defaultType": "swap"},
                })
                # Allow overriding default slippage for market orders
                try:
                    sl_env = os.getenv("HL_MARKET_SLIPPAGE")
                    if sl_env is not None:
                        self._dex.options["defaultSlippage"] = float(sl_env)
                except Exception:
                    pass
                try:
                    # warm up markets to validate symbols early
                    self._dex.load_markets()
                except Exception:
                    pass
                # Log live USDC balance for visibility
                try:
                    bal = self._dex.fetch_balance()
                    usdc = bal.get("USDC", {}) if isinstance(bal, dict) else {}
                    total = float(usdc.get("total") or 0.0)
                    used = float(usdc.get("used") or 0.0)
                    free = usdc.get("free")
                    if free is None:
                        free = max(total - used, 0.0)
                    else:
                        free = float(free)
                    print(f"[LIVE] {self.owner} USDC balance total={total:.2f} free={free:.2f} used={used:.2f}")
                except Exception as e:
                    print(f"[LIVE] {self.owner} falha ao obter saldo: {type(e).__name__}: {e}")
                print(f"[LIVE] Hyperliquid client inicializado para owner={self.owner} vault=...{self.vault_address[-6:]}")
            except Exception as e:
                print(f"[LIVE] Falha ao inicializar ccxt.hyperliquid: {type(e).__name__}: {e}. Caindo para DRY-RUN.")
                self.live = False

    def get_price(self, symbol: str, mid_px: float) -> float:
        # In live: fetch best bid/ask or mark price. For backtest, use provided mid.
        return float(mid_px)

    def get_position(self, symbol: str) -> Position:
        # Live: fetch position on this vaultAddress. In backtest we manage state in StrategyRunner.
        return Position()

    def place_order(self, symbol: str, side: str, qty: float, price: Optional[float] = None,
                    reduce_only: bool = False, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        # Live: send order with vaultAddress routing and reduceOnly flag as required by Hyperliquid.
        # Backtest: this is a no-op stub; execution handled in StrategyRunner using slippage/fees.
        def _resolve_symbol(sym: str) -> str:
            # Accept aliases like "SOL-PERP" and map to CCXT unified "SOL/USDC:USDC"
            s = str(sym).upper().strip()
            if s.endswith("-PERP") and len(s) > 5:
                base = s[:-5]
                return f"{base}/USDC:USDC"
            return sym

        market_symbol = _resolve_symbol(symbol)
        payload = {
            "vaultAddress": self.vault_address,
            "symbol": market_symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "reduce_only": bool(reduce_only),
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "owner": self.owner,
        }
        print(f"[ORDER] owner={self.owner} vault={self.vault_address[-6:]} side={side} qty={qty:.6f} px={price if price is not None else 'MKT'} reduceOnly={reduce_only}")
        if self.live and self._dex is not None:
            try:
                params = {"reduceOnly": bool(reduce_only), "vaultAddress": self.vault_address}
                # Optional: override CCXT defaultSlippage via env var
                sl_env = os.getenv("HL_MARKET_SLIPPAGE")
                if sl_env is not None:
                    try:
                        params["slippage"] = float(sl_env)
                    except Exception:
                        pass
                ord_side = side.lower()
                # Prefer HL orderbook price to guarantee marketability
                px_for_order = price
                try:
                    ob = self._dex.fetch_order_book(market_symbol, limit=5)
                    best_ask = float(ob['asks'][0][0]) if ob.get('asks') else None
                    best_bid = float(ob['bids'][0][0]) if ob.get('bids') else None
                    if ord_side == 'buy' and best_ask is not None:
                        px_for_order = best_ask
                    elif ord_side == 'sell' and best_bid is not None:
                        px_for_order = best_bid
                except Exception:
                    pass
                res = self._dex.create_order(market_symbol, "market", ord_side, qty, px_for_order, params)
                payload["live_ack"] = True
                payload["live_resp"] = res
                return payload
            except Exception as e:
                payload["live_ack"] = False
                payload["error"] = f"{type(e).__name__}: {e}"
                print(f"[LIVE] Falha ao enviar ordem: {payload['error']}")
                return payload
        return payload

    def cancel_all(self, symbol: str) -> None:
        # Live: cancel all open orders on this vaultAddress for symbol
        print(f"[ORDER] cancel_all owner={self.owner} vault={self.vault_address[-6:]} symbol={symbol}")
        return None

    # ---------- Live balance helpers ----------
    def max_affordable_notional(self, leverage: float) -> Optional[float]:
        if not (self.live and self._dex is not None):
            return None
        try:
            bal = self._dex.fetch_balance()
            usdc = bal.get("USDC", {}) if isinstance(bal, dict) else {}
            total = float(usdc.get("total") or 0.0)
            free = usdc.get("free")
            used = usdc.get("used")
            if free is None:
                # fallback: free ~ total - used
                free = float(total - float(used or 0.0))
            else:
                free = float(free)
            free = max(free, 0.0)
            # apply small buffer to avoid edge rejections
            return float(free) * float(leverage)
        except Exception as e:
            print(f"[LIVE] Falha ao ler saldo: {type(e).__name__}: {e}")
            return None

    def live_position_size(self, symbol: str) -> Optional[float]:
        """Return current absolute position size on exchange for symbol, or None if unavailable."""
        if not (self.live and self._dex is not None):
            return None
        try:
            pos = self._dex.fetch_position(symbol)
            # ccxt unified position has 'contracts' or 'size' depending on market; try both
            size = pos.get('contracts') if isinstance(pos, dict) else None
            if size is None:
                size = pos.get('size') if isinstance(pos, dict) else None
            if size is None:
                # hyperliquid parse_position uses 'size' as number in some versions; fallback to info.szi
                info = pos.get('info', {}) if isinstance(pos, dict) else {}
                entry = info.get('position', {}) if isinstance(info, dict) else {}
                szi = entry.get('szi')
                if szi is not None:
                    try:
                        size = abs(float(szi))
                    except Exception:
                        size = None
            if size is None:
                return 0.0
            return float(size)
        except Exception as e:
            print(f"[LIVE] Falha ao consultar posição: {type(e).__name__}: {e}")
            return None


# =============================
# Discord notifications (optional)
# =============================

# Fallback padrão (se nenhuma env var for definida). Evite expor em logs.
DISCORD_WEBHOOK_FALLBACK = "https://discord.com/api/webhooks/1411808916316098571/m_qTenLaTMvyf2e1xNklxFP2PVIvrVD328TFyofY1ciCUlFdWetiC-y4OIGLV23sW9vM"

def _discord_webhook_for(owner: Optional[str] = None) -> Optional[str]:
    # Prioriza por owner: DISCORD_WEBHOOK_URL_BB / _VWAP; fallback DISCORD_WEBHOOK_URL
    if owner:
        key = f"DISCORD_WEBHOOK_URL_{owner.upper()}"
        url = os.getenv(key)
        if url:
            return url
    return os.getenv("DISCORD_WEBHOOK_URL") or DISCORD_WEBHOOK_FALLBACK


def discord_notify(owner: str, text: str) -> None:
    url = _discord_webhook_for(owner)
    if not url:
        return
    try:
        import requests
        payload = {"content": text}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[DISCORD] falha ao enviar: {type(e).__name__}: {e}")


# =============================
# Risk management
# =============================


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period), min_periods=1).mean()


@dataclass
class RiskParams:
    atr_period: int
    k_sl: float
    k_tp: float
    k_trail: Optional[float]
    time_stop_bars: Optional[int]
    break_even_mult: Optional[float] = None  # move stop to BE when MFE >= X*ATR


class RiskManager:
    def __init__(self, rp: RiskParams):
        self.rp = rp

    def evaluate(self, pos: Position, px: float, atr_now: float) -> Tuple[Optional[str], Optional[float]]:
        reason = None
        trail_px = pos.trail_px
        # trailing
        if self.rp.k_trail is not None and atr_now is not None:
            if pos.side == "LONG":
                t = px - self.rp.k_trail * atr_now
                trail_px = max(trail_px or -np.inf, t)
            elif pos.side == "SHORT":
                t = px + self.rp.k_trail * atr_now
                trail_px = min(trail_px or np.inf, t)

        # SL/TP levels
        if pos.side == "LONG":
            sl = pos.entry_px - self.rp.k_sl * atr_now
            tp = pos.entry_px + self.rp.k_tp * atr_now
            # break-even: raise sl to entry if hit target
            if self.rp.break_even_mult is not None and pos.mfe >= self.rp.break_even_mult * atr_now / max(pos.entry_px, 1e-9):
                sl = max(sl, pos.entry_px)
            if trail_px is not None:
                sl = max(sl, trail_px)
            if px <= sl:
                reason = "sl"
            elif px >= tp:
                reason = "tp"
        elif pos.side == "SHORT":
            sl = pos.entry_px + self.rp.k_sl * atr_now
            tp = pos.entry_px - self.rp.k_tp * atr_now
            if self.rp.break_even_mult is not None and pos.mfe >= self.rp.break_even_mult * atr_now / max(pos.entry_px, 1e-9):
                sl = min(sl, pos.entry_px)
            if trail_px is not None:
                sl = min(sl, trail_px)
            if px >= sl:
                reason = "sl"
            elif px <= tp:
                reason = "tp"

        # time stop
        if reason is None and self.rp.time_stop_bars is not None and pos.bars_held >= int(self.rp.time_stop_bars):
            reason = "time"
        return reason, trail_px


# =============================
# Strategy base and implementations
# =============================


@dataclass
class StrategyParams:
    # BB
    n: int = 20
    k: float = 2.0
    # VWAP
    win: int = 20
    ema_trend: int = 50
    # Entry/exit tolerances to create hysteresis around VWAP
    tol_entry: float = 0.002
    tol_exit: float = 0.004
    # Ichimoku
    tenkan: int = 9
    kijun: int = 26
    senkou: int = 52


@dataclass
class OrderIntent:
    side: str  # 'LONG' | 'SHORT'
    reason: str


class StrategyBase:
    owner: str
    params: StrategyParams

    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        raise NotImplementedError

    def entry_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Optional[OrderIntent]:
        raise NotImplementedError

    def exit_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series], pos: Position) -> bool:
        raise NotImplementedError

    def size_model(self, price: float, notional: float) -> float:
        return float(notional) / max(price, 1e-9)


class BBContrarian(StrategyBase):
    owner = "BB"

    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = pd.to_numeric(df["close"], errors="coerce")
        ma = close.rolling(self.params.n, min_periods=1).mean()
        sd = close.rolling(self.params.n, min_periods=1).std()
        up = ma + self.params.k * sd
        lo = ma - self.params.k * sd
        return {"ma": ma, "up": up, "lo": lo}

    def entry_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Optional[OrderIntent]:
        c = float(df["close"].iloc[i])
        if c < float(ind["lo"].iloc[i]):
            return OrderIntent("LONG", "bb_long_revert")
        if c > float(ind["up"].iloc[i]):
            return OrderIntent("SHORT", "bb_short_revert")
        return None

    def exit_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series], pos: Position) -> bool:
        c = float(df["close"].iloc[i])
        ma = float(ind["ma"].iloc[i])
        if pos.side == "LONG":
            return c > ma
        else:
            return c < ma


class VWAPPullback(StrategyBase):
    owner = "VWAP"

    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        close = pd.to_numeric(df["close"], errors="coerce")
        vol = None
        if "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            vol = pd.to_numeric(df.get("volume_compra", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("volume_venda", 0), errors="coerce").fillna(0)
        pv = (close * vol).rolling(self.params.win, min_periods=1).sum()
        vv = vol.rolling(self.params.win, min_periods=1).sum()
        vwap = pv / (vv + 1e-12)
        ema_tr = close.rolling(self.params.ema_trend, min_periods=1).mean()
        return {"vwap": vwap, "ema_trend": ema_tr}

    def entry_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Optional[OrderIntent]:
        c = float(df["close"].iloc[i])
        v = float(ind["vwap"].iloc[i])
        ema_tr = float(ind["ema_trend"].iloc[i])
        tol = float(getattr(self.params, 'tol_entry', 0.002))
        if abs(c - v) <= tol * c:
            if c > ema_tr:
                return OrderIntent("LONG", "vwap_pullback_long")
            if c < ema_tr:
                return OrderIntent("SHORT", "vwap_pullback_short")
        return None

    def exit_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series], pos: Position) -> bool:
        c = float(df["close"].iloc[i])
        v = float(ind["vwap"].iloc[i])
        # add hysteresis using tol_exit to reduce churn
        tol = float(getattr(self.params, 'tol_exit', 0.004))
        if pos.side == "LONG":
            return c < v * (1 - tol)
        else:
            return c > v * (1 + tol)


class IchimokuKumo(StrategyBase):
    owner = "ICHIMOKU"

    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        p_t = max(int(self.params.tenkan), 1)
        p_k = max(int(self.params.kijun), 1)
        p_s = max(int(self.params.senkou), 1)
        tenkan = (high.rolling(p_t, min_periods=1).max() + low.rolling(p_t, min_periods=1).min()) / 2.0
        kijun = (high.rolling(p_k, min_periods=1).max() + low.rolling(p_k, min_periods=1).min()) / 2.0
        ssa = (tenkan + kijun) / 2.0
        ssb = (high.rolling(p_s, min_periods=1).max() + low.rolling(p_s, min_periods=1).min()) / 2.0
        return {"tenkan": tenkan, "kijun": kijun, "ssa": ssa, "ssb": ssb}

    def entry_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Optional[OrderIntent]:
        c = float(df["close"].iloc[i])
        tenkan = float(ind["tenkan"].iloc[i])
        kijun = float(ind["kijun"].iloc[i])
        ssa = float(ind["ssa"].iloc[i])
        ssb = float(ind["ssb"].iloc[i])
        cloud_top = max(ssa, ssb)
        cloud_bot = min(ssa, ssb)
        # Cross conditions (use previous values to detect cross at i)
        if i > 0:
            t_prev = float(ind["tenkan"].iloc[i-1])
            k_prev = float(ind["kijun"].iloc[i-1])
        else:
            t_prev = tenkan
            k_prev = kijun
        bull_cross = (t_prev <= k_prev) and (tenkan > kijun)
        bear_cross = (t_prev >= k_prev) and (tenkan < kijun)
        if c > cloud_top and bull_cross:
            return OrderIntent("LONG", "kumo_bull_break")
        if c < cloud_bot and bear_cross:
            return OrderIntent("SHORT", "kumo_bear_break")
        return None

    def exit_signal(self, i: int, df: pd.DataFrame, ind: Dict[str, pd.Series], pos: Position) -> bool:
        # To prevent rapid flip and excessive fees, disable strategy exits.
        # Exits will be managed solely by RiskManager (SL/TP/trailing/time).
        return False


# =============================
# Strategy Runner (per owner / sub-account)
# =============================


class StrategyRunner:
    def __init__(self, *, owner: str, strategy: StrategyBase, exch: ExchangeClient, risk: RiskManager,
                 symbol: str, notional_per_trade: float, fee_bps: float, slippage_bps: float, leverage: float = 20.0,
                 cooldown_bars: int = 0, min_hold_bars: int = 1, gate_risk_until_min_hold: bool = False):
        self.owner = owner
        self.strategy = strategy
        self.exch = exch
        self.risk = risk
        self.symbol = symbol
        self.notional = float(notional_per_trade)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.leverage = float(leverage)

        self.pos = Position()
        self.trades: List[Dict[str, Any]] = []
        self.equity_rows: List[Dict[str, Any]] = []
        self.ind: Dict[str, pd.Series] = {}
        self.atr: Optional[pd.Series] = None
        self.balance = 0.0
        # controles de histórico
        self.cooldown_bars = int(cooldown_bars)
        self.entry_block_until: int = -1
        self.last_entry_idx: int = -1
        self.last_exit_idx: int = -1
        self.last_action_idx: int = -1
        self.min_hold_bars = int(min_hold_bars)
        self.gate_risk_until_min_hold = bool(gate_risk_until_min_hold)

    def _slip(self, px: float, side: str, is_entry: bool) -> float:
        mult = 1 + (self.slippage_bps / 1e4)
        if is_entry:
            return px * (mult if side == "LONG" else 1 / mult)
        else:
            return px * (1 / mult if side == "LONG" else mult)

    def _fee_cost(self) -> float:
        return (self.fee_bps / 1e4) * self.notional

    def _mark_equity(self, ts: pd.Timestamp):
        self.equity_rows.append({"ts": ts, "equity": self.balance})

    def compute_indicators(self, df: pd.DataFrame):
        self.ind = self.strategy.compute_indicators(df)
        self.atr = atr_series(df, self.risk.rp.atr_period)
        print(f"[IND] owner={self.owner} computed indicators | len={len(df)}")

    def rolling_perf(self, n: int = 30) -> Tuple[float, float]:
        if not self.trades:
            return 0.0, 0.0
        r = pd.Series([t.get("ret", 0.0) for t in self.trades])
        if len(r) > n:
            r = r.tail(n)
        wins = r[r > 0].sum()
        losses = -r[r < 0].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        return float(r.sum()), pf

    def wants_entry(self, i: int, df: pd.DataFrame) -> Optional[OrderIntent]:
        if self.pos.is_open():
            # Apenas para logging: verifica se haveria sinal e marca como ignorado
            try:
                intent = self.strategy.entry_signal(i, df, self.ind)
                if intent is not None:
                    print(f"[SKIP] owner={self.owner} sinal {intent.side} ignorado: já existe posição aberta")
            except Exception:
                pass
            return None
        # cooldown baseado em histórico de barras
        if i <= self.entry_block_until:
            print(f"[SKIP] owner={self.owner} cooldown active until i={self.entry_block_until} (now i={i})")
            return None
        if i == self.last_action_idx:
            return None
        intent = self.strategy.entry_signal(i, df, self.ind)
        if intent:
            print(f"[SIGNAL] owner={self.owner} {intent.side} @i={i}")
        return intent

    def process_bar(self, i: int, df: pd.DataFrame):
        ts = pd.to_datetime(df["ts"].iloc[i])
        px = float(df["close"].iloc[i])
        atr_now = float(self.atr.iloc[i]) if self.atr is not None else 0.0

        # Update open position
        if self.pos.is_open():
            self.pos.update_excursions(px)
            # Evita avaliar saída na mesma barra da entrada
            if i == self.last_entry_idx:
                # ainda na barra de entrada: não conta hold e não sai
                self._mark_equity(ts)
                return
            # evita múltiplos EXIT no mesmo índice (reprocesso)
            if i == self.last_exit_idx:
                self._mark_equity(ts)
                return
            # agora sim, já estamos numa barra subsequente à entrada
            self.pos.bars_held += 1
            reason, new_trail = self.risk.evaluate(self.pos, px, atr_now)
            # Afrouxar saídas apenas quando configurado (ex.: VWAP):
            # não permite SL/TP/trailing/time antes de atingir min_hold_bars
            if self.gate_risk_until_min_hold and self.pos.bars_held < self.min_hold_bars:
                reason = None
            if reason is None and self.pos.bars_held >= self.min_hold_bars and self.strategy.exit_signal(i, df, self.ind, self.pos):
                reason = "exit_signal"
            if reason is not None:
                # If live, ensure there's a real position before attempting reduceOnly close
                if getattr(self.exch, "live", False):
                    live_sz = self.exch.live_position_size(self.symbol)
                    if not live_sz or live_sz <= 0:
                        print(f"[LIVE] DESYNC detected on EXIT owner={self.owner}: no live position; dropping local position without PnL booking")
                        self.pos = Position()
                        self._mark_equity(ts)
                        return
                ex_px = self._slip(px, self.pos.side or "LONG", False)
                qty = self.pos.qty
                # First, if live, try to actually send the reduce-only order; only close upon ack
                if getattr(self.exch, "live", False):
                    res = self.exch.place_order(self.symbol, "SELL" if self.pos.side == "LONG" else "BUY", qty, price=ex_px, reduce_only=True)
                    if not res.get("live_ack"):
                        # keep position open; do not book PnL
                        print(f"[LIVE] EXIT REJECTED owner={self.owner} side={self.pos.side} reason={reason} px={ex_px:.6f} | keeping position open")
                        self._mark_equity(ts)
                        return
                # Book exit locally (backtest or live ack ok)
                pnl_price = (ex_px - self.pos.entry_px) * qty if self.pos.side == "LONG" else (self.pos.entry_px - ex_px) * qty
                fees = self._fee_cost()  # exit fee only (entry fee was at entry)
                pnl = pnl_price - fees
                margin = self.notional / self.leverage
                ret = pnl / margin
                self.balance += pnl
                self.trades.append({
                    "ts_entry": self.pos.entry_ts, "ts_exit": ts, "side": self.pos.side,
                    "entry": self.pos.entry_px, "exit": ex_px, "qty": qty,
                    "pnl": pnl, "roe": ret, "owner": self.owner, "reason_exit": reason,
                    "mfe": self.pos.mfe, "mae": self.pos.mae, "ret": ret,
                })
                print(f"[EXIT] owner={self.owner} side={self.pos.side} reason={reason} px={ex_px:.6f} pnl={pnl:.2f} roe={ret:.4f} bal={self.balance:.2f}")
                if getattr(self.exch, "live", False):
                    discord_notify(self.owner, f"[LIVE] EXIT | {self.symbol} | owner={self.owner} | side={self.pos.side} | reason={reason} | px={ex_px:.4f} | pnl={pnl:.2f} | roe={ret:.4f} | bal={self.balance:.2f}")
                else:
                    discord_notify(self.owner, f"[DRY-RUN] EXIT | {self.symbol} | owner={self.owner} | side={self.pos.side} | reason={reason} | px={ex_px:.4f} | pnl={pnl:.2f} | roe={ret:.4f} | bal={self.balance:.2f}")
                self.pos = Position()
                # aplica cooldown após saída
                if self.cooldown_bars > 0:
                    self.entry_block_until = i + self.cooldown_bars
                self.last_action_idx = i
                self.last_exit_idx = i
            else:
                self.pos.trail_px = new_trail

        # Equity mark
        self._mark_equity(ts)

    def try_enter(self, i: int, df: pd.DataFrame, intent: Optional[OrderIntent]):
        if intent is None or self.pos.is_open():
            return
        # evita duplicar entrada ao reprocessar mesmo índice
        if i == self.last_entry_idx:
            print(f"[SKIP] owner={self.owner} duplicate entry at i={i}")
            return
        # respeita cooldown
        if i <= self.entry_block_until:
            return
        ts = pd.to_datetime(df["ts"].iloc[i])
        px = float(df["close"].iloc[i])
        side = intent.side
        en_px = self._slip(px, side, True)
        desired_notional = self.notional
        # If live, cap notional by available margin * leverage
        if getattr(self.exch, "live", False):
            max_n = self.exch.max_affordable_notional(self.leverage)
            if max_n is None:
                print(f"[SKIP] owner={self.owner} sinal {side} ignorado: saldo não disponível (fetch_balance falhou)")
                return
            # keep 95% buffer
            desired_notional = min(desired_notional, 0.95 * max_n)
            if desired_notional < 10:  # HL min order cost is $10
                print(f"[SKIP] owner={self.owner} sinal {side} ignorado: margem insuficiente (max_n={max_n:.2f} USDC equiv.)")
                return
        qty = self.strategy.size_model(en_px, desired_notional)
        res = self.exch.place_order(self.symbol, "BUY" if side == "LONG" else "SELL", qty, price=en_px, reduce_only=False)
        # In live mode, only open position if exchange acknowledged AND position exists
        if getattr(self.exch, "live", False):
            if not res.get("live_ack"):
                print(f"[LIVE] ENTER REJECTED owner={self.owner} side={side} px={en_px:.6f} | not opening position")
                return
            # Confirm fill by checking position size
            live_sz = self.exch.live_position_size(self.symbol)
            if not live_sz or live_sz <= 0:
                print(f"[LIVE] ENTER NO FILL owner={self.owner} side={side} px={en_px:.6f} | position size=0 on exchange")
                return
        # entry fee only after confirming position (or always in backtest)
        self.balance -= self._fee_cost()
        self.pos = Position(side=side, qty=qty, entry_px=en_px, entry_ts=ts, bars_held=0)
        self.last_entry_idx = i
        self.last_action_idx = i
        # apply cooldown immediately after a successful entry
        if self.cooldown_bars > 0:
            self.entry_block_until = i + self.cooldown_bars
        print(f"[ENTER] owner={self.owner} side={side} px={en_px:.6f} qty={qty:.6f} i={i} bal={self.balance:.2f}")
        if getattr(self.exch, "live", False):
            discord_notify(self.owner, f"[LIVE] ENTER | {self.symbol} | owner={self.owner} | side={side} | px={en_px:.4f} | qty={qty:.6f} | i={i}")
        else:
            discord_notify(self.owner, f"[DRY-RUN] ENTER | {self.symbol} | owner={self.owner} | side={side} | px={en_px:.4f} | qty={qty:.6f} | i={i}")

    def finalize(self, out_dir: Path):
        trades_df = pd.DataFrame(self.trades)
        eq_df = pd.DataFrame(self.equity_rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(out_dir / f"trades_log_{self.owner}.csv", index=False)
        eq_df.to_csv(out_dir / f"equity_{self.owner}.csv", index=False)


# =============================
# Orchestrator for two owners (BB and VWAP)
# =============================


class Orchestrator:
    def __init__(self, df: pd.DataFrame, runners: Dict[str, StrategyRunner], priority_window: int = 30):
        self.df = df.reset_index(drop=True)
        self.runners = runners
        self.priority_window = int(priority_window)
        self.priority_rows: List[Dict[str, Any]] = []

    def loop(self, out_dir: str = "excel/live_top2"):
        outp = Path(out_dir)
        owners = list(self.runners.keys())
        # Precompute indicators per runner
        for r in self.runners.values():
            r.compute_indicators(self.df)

        for i in range(len(self.df)):
            ts = pd.to_datetime(self.df["ts"].iloc[i])
            # 1) Process exits and mark equity first for all runners
            for o in owners:
                self.runners[o].process_bar(i, self.df)

            # 2) Re-compute priority after exits
            perf = {o: self.runners[o].rolling_perf(self.priority_window) for o in owners}
            order = sorted(owners, key=lambda o: (perf[o][0], 1 if o == "BB" else 0), reverse=True)
            perf_str = ", ".join([f"{o}: pnl={perf[o][0]:.2f}/pf={perf[o][1]:.2f}" for o in owners])
            print(f"[BAR] ts={ts} order={order} perf={{{perf_str}}}")
            self.priority_rows.append({
                "ts": ts,
                **{f"{o}_rolling_pnl": perf[o][0] for o in owners},
                **{f"{o}_rolling_pf": perf[o][1] for o in owners},
                "priority_rank": ",".join(order),
            })

            # 3) Check entry signals and enter, respecting open positions and last_action_idx guard
            intents: Dict[str, Optional[OrderIntent]] = {o: self.runners[o].wants_entry(i, self.df) for o in owners}
            for o in order:
                r = self.runners[o]
                if r.pos.is_open():
                    continue
                if intents.get(o):
                    r.try_enter(i, self.df, intents[o])

        # Finalize
        for r in self.runners.values():
            r.finalize(outp)
        pd.DataFrame(self.priority_rows).to_csv(Path(out_dir) / "priority_score.csv", index=False)

    def process_new_bars(self, out_dir: str = "/tmp/live_top2", reprocess_last: bool = False):
        """Processa apenas as barras que foram adicionadas em self.df desde a última chamada.
        Recalcula indicadores e executa entradas/saídas para as novas barras.
        """
        owners = list(self.runners.keys())
        # Recompute indicators for new df
        for r in self.runners.values():
            r.compute_indicators(self.df)

        start = len(self.priority_rows)
        if reprocess_last and start > 0:
            start = start - 1
            # remover última marcação de equity (será substituída)
            for r in self.runners.values():
                if r.equity_rows:
                    r.equity_rows.pop()
        for i in range(start, len(self.df)):
            ts = pd.to_datetime(self.df["ts"].iloc[i])
            # Trade only on close: if reprocessing the last candle, skip entries/exits and only mark equity
            if reprocess_last and i == len(self.df) - 1:
                for r in self.runners.values():
                    r._mark_equity(ts)
                continue
            # 1) Process exits and mark equity first for all runners
            for o in owners:
                self.runners[o].process_bar(i, self.df)

            # 2) Compute priority after exits
            perf = {o: self.runners[o].rolling_perf(self.priority_window) for o in owners}
            order = sorted(owners, key=lambda o: (perf[o][0], 1 if o == "BB" else 0), reverse=True)
            perf_str = ", ".join([f"{o}: pnl={perf[o][0]:.2f}/pf={perf[o][1]:.2f}" for o in owners])
            print(f"[BAR] ts={ts} (incremental) order={order} perf={{{perf_str}}}")
            self.priority_rows.append({
                "ts": ts,
                **{f"{o}_rolling_pnl": perf[o][0] for o in owners},
                **{f"{o}_rolling_pf": perf[o][1] for o in owners},
                "priority_rank": ",".join(order),
            })

            # 3) Check entries in priority order
            intents: Dict[str, Optional[OrderIntent]] = {o: self.runners[o].wants_entry(i, self.df) for o in owners}
            for o in order:
                r = self.runners[o]
                if r.pos.is_open():
                    continue
                if intents.get(o):
                    r.try_enter(i, self.df, intents[o])
        # persist incremental logs
        outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
        for r in self.runners.values():
            trades_df = pd.DataFrame(r.trades)
            eq_df = pd.DataFrame(r.equity_rows)
            trades_df.to_csv(outp / f"trades_log_{r.owner}.csv", index=False)
            eq_df.to_csv(outp / f"equity_{r.owner}.csv", index=False)
        pd.DataFrame(self.priority_rows).to_csv(Path(out_dir) / "priority_score.csv", index=False)


# =============================
# Utilities
# =============================


def load_df(csv_path: str) -> pd.DataFrame:
    # Suporta caminho local ou URL http(s)
    if str(csv_path).lower().startswith(("http://", "https://")):
        df = pd.read_csv(csv_path)
    else:
        p = Path(csv_path)
        df = pd.read_csv(p)
    # Normalize columns to ts, open, high, low, close, volume
    cols = {c.lower(): c for c in df.columns}
    def pick(name: str, fallback: Optional[str] = None) -> Optional[str]:
        if name in cols:
            return cols[name]
        return cols.get(fallback) if fallback else None
    ts_col = pick("ts", "data") or "data"
    open_col = pick("open") or pick("valor_abertura") or pick("close")
    high_col = pick("high") or pick("valor_maximo") or pick("close")
    low_col = pick("low") or pick("valor_minimo") or pick("close")
    close_col = pick("close") or pick("valor_fechamento") or pick("open")
    vol_col = pick("volume")
    if vol_col is None:
        # compose from volume_compra+volume_venda or fallback 1.0
        vcomp = df.get("volume_compra")
        vvend = df.get("volume_venda")
        if vcomp is not None or vvend is not None:
            df["__volume__"] = pd.to_numeric(df.get("volume_compra", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("volume_venda", 0), errors="coerce").fillna(0)
            vol_col = "__volume__"
        else:
            df["__volume__"] = 1.0
            vol_col = "__volume__"
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col]),
        "open": pd.to_numeric(df[open_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "volume": pd.to_numeric(df[vol_col], errors="coerce").fillna(0),
    })
    return out.dropna().reset_index(drop=True)


def _tf_seconds(tf: str) -> int:
    s = str(tf).lower().strip()
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("d"):
        return int(s[:-1]) * 86400
    return 60


class BinanceFeed:
    BASE = "https://api.binance.com/api/v3/klines"

    @staticmethod
    def fetch(symbol: str, interval: str, limit: int = 1000, end_ms: Optional[int] = None) -> pd.DataFrame:
        import requests
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit),
        }
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        r = requests.get(BinanceFeed.BASE, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        for it in data:
            rows.append({
                "ts": pd.to_datetime(it[0], unit="ms"),
                "open": float(it[1]),
                "high": float(it[2]),
                "low": float(it[3]),
                "close": float(it[4]),
                "volume": float(it[5]),
            })
        return pd.DataFrame(rows)


# =============================
# main()
# =============================


def main():
    # Parâmetros 100% definidos no código (exceto chaves/vaults)
    # Use unified CCXT symbol for Hyperliquid perps (alias mapping also supported)
    symbol = "SOL/USDC:USDC"
    timeframe = "15m"
    feed_symbol = "SOLUSDT"  # símbolo para feed da Binance
    fee_bps = 2.5
    slippage_bps = 3.0
    notional = 100.0
    leverage = 20.0

    # Risco
    atr_period = 14
    # Parâmetros base de risco (BB)
    k_sl_bb = 1.2
    k_tp_bb = 1.6
    k_trail_bb: Optional[float] = 1.5
    # Parâmetros mais folgados apenas para VWAP
    k_sl_vw = 2.0
    k_tp_vw = 2.0
    k_trail_vw: Optional[float] = 3.0
    time_stop: Optional[int] = 80
    roll_n = 30

    # BB sub-account
    bb_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    bb_vault = os.getenv("WALLET_ADDRESS", "")
    # VWAP sub-account
    vw_key = os.getenv("VWAP_PULLBACK_PRIVATE_KEY", "")
    vw_vault = os.getenv("VWAP_PULLBACK_WALLET", "")

    if not bb_key or not bb_vault or not vw_key or not vw_vault:
        print("[WARN] Chaves/vaultAddress ausentes; rodando apenas em modo backtest.")

    rp_bb = RiskParams(atr_period=atr_period, k_sl=k_sl_bb, k_tp=k_tp_bb, k_trail=k_trail_bb, time_stop_bars=time_stop)
    rp_vw = RiskParams(atr_period=atr_period, k_sl=k_sl_vw, k_tp=k_tp_vw, k_trail=k_trail_vw, time_stop_bars=time_stop)

    # Build strategies
    bb_params = StrategyParams(n=20, k=2.0)
    # VWAP parameters with wider hysteresis bands
    vw_params = StrategyParams(win=20, ema_trend=50, tol_entry=0.003, tol_exit=0.006)
    bb = BBContrarian(); bb.params = bb_params
    # Replace VWAP with Ichimoku Kumo logic but keep the same wallet (owner label remains 'VWAP')
    ich = IchimokuKumo(); ich.params = vw_params

    # Exchange clients per owner (ensure vaultAddress is passed on all orders)
    exch_bb = ExchangeClient(private_key=bb_key, vault_address=bb_vault, owner="BB", fee_bps=fee_bps, slippage_bps=slippage_bps)
    exch_vw = ExchangeClient(private_key=vw_key, vault_address=vw_vault, owner="VWAP", fee_bps=fee_bps, slippage_bps=slippage_bps)

    # Strategy runners (independent state per owner)
    # BB runner: ativa bloqueio de risco e aumenta min_hold_bars para segurar posição
    r_bb = StrategyRunner(owner="BB", strategy=bb, exch=exch_bb, risk=RiskManager(rp_bb), symbol=symbol, notional_per_trade=notional, fee_bps=fee_bps, slippage_bps=slippage_bps, leverage=leverage, cooldown_bars=2, min_hold_bars=3, gate_risk_until_min_hold=True)
    # VWAP wallet running Ichimoku strategy
    r_vw = StrategyRunner(owner="VWAP", strategy=ich, exch=exch_vw, risk=RiskManager(rp_vw), symbol=symbol, notional_per_trade=notional, fee_bps=fee_bps, slippage_bps=slippage_bps, leverage=leverage, cooldown_bars=5, min_hold_bars=5, gate_risk_until_min_hold=True)

    # Sempre em modo live polling por padrão (independente de CSV local)
    out_dir = "/tmp/live_top2"
    print(f"[LIVE] Iniciando polling Binance para {feed_symbol} {timeframe}")
    df = BinanceFeed.fetch(feed_symbol, timeframe, limit=1000)
    orch = Orchestrator(df, runners={"BB": r_bb, "VWAP": r_vw}, priority_window=roll_n)
    orch.process_new_bars(out_dir=out_dir)

    import time
    tf_s = _tf_seconds(timeframe)
    last_ts = df["ts"].iloc[-1]
    while True:
        try:
            df_latest = BinanceFeed.fetch(feed_symbol, timeframe, limit=3)
            # Se veio a barra atual (mesmo ts), atualiza-a e reprocessa a última; se veio barra nova (ts>last_ts), anexa e processa.
            same_mask = df_latest["ts"] == last_ts
            if same_mask.any():
                # substitui última linha
                df.iloc[-1] = df_latest[same_mask].iloc[-1]
                orch.df = df
                orch.process_new_bars(out_dir=out_dir, reprocess_last=True)
                print(f"[LIVE] atualização parcial do candle {last_ts}")
            newer = df_latest[df_latest["ts"] > last_ts]
            if not newer.empty:
                df = pd.concat([df, newer], ignore_index=True)
                last_ts = df["ts"].iloc[-1]
                orch.df = df
                orch.process_new_bars(out_dir=out_dir)
                print(f"[LIVE] +{len(newer)} barras até {last_ts}")
        except Exception as e:
            print(f"[LIVE] erro no polling: {type(e).__name__}: {e}")
        time.sleep(10)


if __name__ == "__main__":
    main()
