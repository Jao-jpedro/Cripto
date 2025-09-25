#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trading bot com:
- Carteira principal fixa (sem vault)
- Lista completa de ASSET_SETUPS (alavancagem por ativo + env de sizing)
- Sem TP/SL: apenas trailing stop por ROI alavancado (10 p.p.), monotônico (só melhora)
- Hard stop: fecha posição se unrealizedPnl <= -$0,10
- Sem trava de 1 entrada por ativo
- OHLCV limitado a 50 candles por ciclo
- build_df() disponível para debug (Binance /klines), mas só roda sob __main__
"""

import os
import time
import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import ccxt  # type: ignore
import requests  # para build_df de debug

# Aliases/constantes globais exigidos pelo seu arquivo original
_time = time
UTC = timezone.utc
BASE_URL = os.getenv('BASE_URL', 'https://api.binance.com/api/v3/')

# ======================================================================
# Helpers
# ======================================================================

def log(level: str, tag: str, msg: str) -> None:
    print(f"[{level}] [{tag}] {msg}", flush=True)

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

def norm_side(side: str) -> str:
    s = (side or "").lower()
    if s in ("buy","long"): return "buy"
    if s in ("sell","short"): return "sell"
    return s

def ema(a: np.ndarray, span: int) -> np.ndarray:
    if len(a) == 0: return a
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    if len(close) < 2: return np.full_like(close, np.nan, dtype=float)
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    tr = np.concatenate([[np.nan], tr])
    return pd.Series(tr).rolling(period).mean().to_numpy()

# ======================================================================
# Config
# ======================================================================

@dataclass
class Config:
    TIMEFRAME: str = "15m"
    HISTORY_BARS: int = 50              # capado a 50
    EMA_SHORT: int = 7
    EMA_LONG: int = 21
    ATR_PERIOD: int = 14
    VOL_MA_PERIOD: int = 20
    LEVERAGE_DEFAULT: float = 10.0      # fallback (cada ativo tem o seu)
    ROI_TRAIL_PPTS: float = 0.10        # 10 pontos percentuais de ROI alavancado
    SLEEP_SEC: float = 10.0
    DEBUG: bool = True

    def __post_init__(self):
        self.TIMEFRAME = os.getenv("TIMEFRAME", self.TIMEFRAME)
        self.HISTORY_BARS = _env_int("HISTORY_BARS", self.HISTORY_BARS)
        self.LEVERAGE_DEFAULT = _env_float("LEVERAGE", self.LEVERAGE_DEFAULT)
        self.ROI_TRAIL_PPTS = _env_float("ROI_TRAIL_PPTS", self.ROI_TRAIL_PPTS)
        self.SLEEP_SEC = _env_float("SLEEP_SEC", self.SLEEP_SEC)
        self.DEBUG = (os.getenv("DEBUG", "1") != "0")

# ======================================================================
# Asset setup
# ======================================================================

@dataclass
class AssetSetup:
    name: str
    data_symbol: str
    hl_symbol: str
    leverage: int
    usd_env: Optional[str] = None

ASSET_SETUPS: List[AssetSetup] = [
    AssetSetup("BTC-USD", "BTCUSDT", "BTC/USDC:USDC", 40, usd_env="USD_PER_TRADE_BTC"),
    AssetSetup("SOL-USD", "SOLUSDT", "SOL/USDC:USDC", 20, usd_env="USD_PER_TRADE_SOL"),
    AssetSetup("ETH-USD", "ETHUSDT", "ETH/USDC:USDC", 25, usd_env="USD_PER_TRADE_ETH"),
    AssetSetup("HYPE-USD", "HYPEUSDT", "HYPE/USDC:USDC", 10, usd_env="USD_PER_TRADE_HYPE"),
    AssetSetup("XRP-USD", "XRPUSDT", "XRP/USDC:USDC", 20, usd_env="USD_PER_TRADE_XRP"),
    AssetSetup("DOGE-USD", "DOGEUSDT", "DOGE/USDC:USDC", 10, usd_env="USD_PER_TRADE_DOGE"),
    AssetSetup("AVAX-USD", "AVAXUSDT", "AVAX/USDC:USDC", 10, usd_env="USD_PER_TRADE_AVAX"),
    AssetSetup("ENA-USD", "ENAUSDT", "ENA/USDC:USDC", 10, usd_env="USD_PER_TRADE_ENA"),
    AssetSetup("BNB-USD", "BNBUSDT", "BNB/USDC:USDC", 10, usd_env="USD_PER_TRADE_BNB"),
    AssetSetup("SUI-USD", "SUIUSDT", "SUI/USDC:USDC", 10, usd_env="USD_PER_TRADE_SUI"),
    AssetSetup("ADA-USD", "ADAUSDT", "ADA/USDC:USDC", 10, usd_env="USD_PER_TRADE_ADA"),
    AssetSetup("PUMP-USD", "PUMPUSDT", "PUMP/USDC:USDC", 5, usd_env="USD_PER_TRADE_PUMP"),
    AssetSetup("AVNT-USD", "AVNTUSDT", "AVNT/USDC:USDC", 5, usd_env="USD_PER_TRADE_AVNT"),
    AssetSetup("XPL-USD", "XPLUSDT", "XPL/USDC:USDC", 3, usd_env="USD_PER_TRADE_XPL"),
    AssetSetup("KPEPE-USD", "KPEPEUSDT", "KPEPE/USDC:USDC", 10, usd_env="USD_PER_TRADE_KPEPE"),
    AssetSetup("LINK-USD", "LINKUSDT", "LINK/USDC:USDC", 10, usd_env="USD_PER_TRADE_LINK"),
    AssetSetup("WLD-USD", "WLDUSDT", "WLD/USDC:USDC", 10, usd_env="USD_PER_TRADE_WLD"),
    AssetSetup("AAVE-USD", "AAVEUSDT", "AAVE/USDC:USDC", 10, usd_env="USD_PER_TRADE_AAVE"),
    AssetSetup("CRV-USD", "CRVUSDT", "CRV/USDC:USDC", 10, usd_env="USD_PER_TRADE_CRV"),
    AssetSetup("LTC-USD", "LTCUSDT", "LTC/USDC:USDC", 10, usd_env="USD_PER_TRADE_LTC"),
    AssetSetup("NEAR-USD", "NEARUSDT", "NEAR/USDC:USDC", 10, usd_env="USD_PER_TRADE_NEAR"),
]

# ======================================================================
# Trailing Manager
# ======================================================================

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

    def update_and_upsert(self, symbol: str, side: str, entry_px: float, current_px: float, qty: float, trail_pp: float = 0.10) -> Optional[Tuple[str, float]]:
        side = norm_side(side)
        if qty <= 0 or entry_px <= 0 or current_px <= 0:
            return None

        # ROI (não alavancado)
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

        stop_lev_roi = peak - float(trail_pp)
        stop_roi = stop_lev_roi / self.leverage
        if side == "buy":
            stop_px = entry_px * (1.0 + stop_roi)
        else:
            stop_px = entry_px * (1.0 - stop_roi)

        prev_px = self._trail_stop_px.get(key, None)
        improved = (prev_px is None) or ((stop_px > prev_px + 1e-12) if side == "buy" else (stop_px < prev_px - 1e-12))
        if not improved:
            if self.debug:
                log("DEBUG", symbol, f"TRAIL sem melhora | prev={prev_px} novo={stop_px:.6f} side={side}")
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
            log("WARN", symbol, f"TRAIL upsert falhou: {type(e).__name__}: {e}")
            return None

# ======================================================================
# Estratégia
# ======================================================================

LOG_SIGNALS = (os.getenv('LOG_SIGNALS', '1') != '0')
ENTRY_RULE = os.getenv('ENTRY_RULE', 'ema_and_vol')  # ema_and_vol | ema_only | always
MIN_TRADE_INTERVAL_SEC = _env_int('MIN_TRADE_INTERVAL_SEC', 60)
MAX_CANDLES = 50
STAGGER_BETWEEN_SYMBOLS_SEC = _env_int('STAGGER_BETWEEN_SYMBOLS_SEC', 0)

class EMAGradientStrategy:
    def __init__(self, dex, symbol: str, cfg: Config, *, leverage: float, usd_env: str, name: str = ''):
        self.dex = dex
        self.symbol = symbol
        self.cfg = cfg
        self.name = name or symbol
        self.leverage = float(leverage)
        self.usd_env = usd_env
        self.trailer = TrailingManager(dex, self.leverage, cfg.DEBUG)
        self._last_entry_ts: float = 0.0

    def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        lim = min(int(limit), int(MAX_CANDLES))
        log("INFO", symbol, f"Fetching OHLCV tf={timeframe} limit={lim}")
        data = self.dex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lim)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        # ccxt retorna ms no campo timestamp como [0]; se vier no formato clássico, convertemos
        if not np.issubdtype(df["ts"].dtype, np.datetime64):
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        a = df["close"].astype(float).to_numpy()
        df["ema_short"] = ema(a, self.cfg.EMA_SHORT)
        df["ema_long"]  = ema(a, self.cfg.EMA_LONG)
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        df["atr"] = atr(h, l, c, self.cfg.ATR_PERIOD)
        df["vol_ma"] = df["volume"].rolling(self.cfg.VOL_MA_PERIOD).mean()
        return df

    def _get_position(self) -> Optional[Dict[str, Any]]:
        try:
            positions = self.dex.fetch_positions([self.symbol])
            for p in positions:
                # Heurística: posição aberta se tamanho != 0
                szi = float((p.get("info") or {}).get("position", {}).get("szi") or p.get("contracts") or 0.0)
                if abs(szi) > 0:
                    return p
            return None
        except Exception:
            return None

    def _qty_from_usd(self, usd: float, px: float) -> float:
        if px <= 0: return 0.0
        return float(usd) / float(px)

    def _current_price(self) -> float:
        try:
            ticker = self.dex.fetch_ticker(self.symbol)
            px = float(ticker.get("last") or ticker.get("close") or ticker["info"].get("markPx"))
            return px
        except Exception:
            # fallback: último candle
            df = self._fetch_ohlcv(self.symbol, self.cfg.TIMEFRAME, 2)
            return float(df["close"].iloc[-1])

    def step(self):
        # Data + indicadores
        df = self._fetch_ohlcv(self.symbol, self.cfg.TIMEFRAME, self.cfg.HISTORY_BARS)  # capado a 50
        df = self._indicators(df)
        row = df.iloc[-1]
        px = float(row["close"])

        # Sinais
        ema_up = row["ema_short"] > row["ema_long"]
        ema_down = row["ema_short"] < row["ema_long"]
        vol_ok = row["volume"] >= (row["vol_ma"] or 0.0)
        if LOG_SIGNALS:
            log("DEBUG", self.symbol,
                f"px={px:.6f} ema_s={row['ema_short']:.6f} ema_l={row['ema_long']:.6f} "
                f"ema_up={ema_up} ema_down={ema_down} vol={row['volume']} vol_ma={row['vol_ma']} vol_ok={vol_ok}"
            )

        pos = self._get_position()
        if pos is None:
            # throttle entradas
            now = time.time()
            if now - self._last_entry_ts < MIN_TRADE_INTERVAL_SEC:
                if self.cfg.DEBUG:
                    log("DEBUG", self.symbol, f"Skipping entry (MIN_TRADE_INTERVAL_SEC={MIN_TRADE_INTERVAL_SEC})")
                return

            usd_risk = _env_float(self.usd_env or "USD_RISK_PER_TRADE", _env_float("USD_RISK_PER_TRADE", 10.0))
            qty = self._qty_from_usd(usd_risk * self.leverage, px)

            should_long = should_short = False
            if ENTRY_RULE == "always":
                should_long = bool(ema_up)
                should_short = (not ema_up and ema_down)
            elif ENTRY_RULE == "ema_only":
                should_long = ema_up
                should_short = ema_down
            else:  # ema_and_vol
                should_long = ema_up and vol_ok
                should_short = ema_down and vol_ok

            if should_long and qty > 0:
                try:
                    if hasattr(self.dex, "set_leverage"):
                        self.dex.set_leverage(int(self.leverage), {"symbol": self.symbol})
                except Exception as e:
                    log("WARN", self.symbol, f"set_leverage falhou: {e}")
                self.dex.create_order(self.symbol, "market", "buy", qty)
                self._last_entry_ts = now
                log("INFO", self.symbol, f"Entered LONG qty={qty:.6f} ~{px:.6f}")
                return

            if should_short and qty > 0:
                try:
                    if hasattr(self.dex, "set_leverage"):
                        self.dex.set_leverage(int(self.leverage), {"symbol": self.symbol})
                except Exception as e:
                    log("WARN", self.symbol, f"set_leverage falhou: {e}")
                self.dex.create_order(self.symbol, "market", "sell", qty)
                self._last_entry_ts = now
                log("INFO", self.symbol, f"Entered SHORT qty={qty:.6f} ~{px:.6f}")
                return

            if self.cfg.DEBUG:
                log("DEBUG", self.symbol, f"Sem sinal de entrada (ENTRY_RULE={ENTRY_RULE}).")
            return

        # Posição aberta → gerenciar saídas
        info = pos.get("info") or {}
        position = info.get("position") or {}
        szi = float(position.get("szi") or pos.get("contracts") or 0.0)
        side = "buy" if szi > 0 else "sell"
        entry_px = float(pos.get("entryPrice") or position.get("entryPx") or px)

        # Hard stop: unrealizedPnl <= -$0,10
        try:
            unreal = float(pos.get("unrealizedPnl") or info.get("unrealizedPnl") or 0.0)
            if unreal <= -0.10:
                qty_abs = abs(float(pos.get("contracts") or szi))
                exit_side = "sell" if side == "buy" else "buy"
                self.dex.create_order(self.symbol, "market", exit_side, qty_abs, None, {"reduceOnly": True})
                log("WARN", self.symbol, "Hard stop -$0.10 acionado. Fechado.")
                return
        except Exception:
            pass

        # Trailing ROI (10 p.p.), monotônico
        qty_abs = abs(float(pos.get("contracts") or szi))
        self.trailer.update_and_upsert(self.symbol, side, entry_px, px, qty_abs, trail_pp=self.cfg.ROI_TRAIL_PPTS)

# ======================================================================
# build_df de debug (apenas se rodar como script)
# ======================================================================

def build_df(symbol: str, interval: str = "15m", start: Optional[datetime] = None, end: Optional[datetime] = None, debug: bool = False) -> pd.DataFrame:
    """Constrói um pequeno DataFrame OHLCV (limite 50) do endpoint /klines compatível com Binance."""
    valid_intervals = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}
    if interval not in valid_intervals:
        interval = "15m"
    params = {"symbol": symbol, "interval": interval, "limit": 50}
    if start is not None:
        params["startTime"] = int(start.replace(tzinfo=UTC).timestamp() * 1000)
    if end is not None:
        params["endTime"] = int(end.replace(tzinfo=UTC).timestamp() * 1000)
    url = BASE_URL.rstrip("/") + "/klines"
    if debug:
        log("DEBUG", "DATA", f"GET {url} {params}")
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    data = r.json()
    df = pd.DataFrame(data, columns=cols)
    df = df.assign(
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True),
        open = df["open"].astype(float),
        high = df["high"].astype(float),
        low  = df["low"].astype(float),
        close= df["close"].astype(float),
        volume = df["volume"].astype(float),
    )[["ts","open","high","low","close","volume"]]
    return df

# ======================================================================
# Main
# ======================================================================

MAIN_ACCOUNT_ADDRESS = "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"

def main():
    log("INFO", "WALLET", f"Operando SOMENTE na carteira principal: {MAIN_ACCOUNT_ADDRESS}")

    ex_id = os.getenv("EXCHANGE_ID", "binance")
    api_key = os.getenv("HL_API_KEY") or os.getenv("API_KEY") or ""
    secret  = os.getenv("HL_API_SECRET") or os.getenv("API_SECRET") or ""
    password = os.getenv("HL_API_PASSWORD") or os.getenv("API_PASSWORD") or None

    # Init exchange
    if not hasattr(ccxt, ex_id):
        raise RuntimeError(f'Exchange id "{ex_id}" não encontrado no ccxt.')
    dex = getattr(ccxt, ex_id)({
        "apiKey": api_key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "timeout": _env_int("CCXT_TIMEOUT_MS", 10000),
        "options": {},
    })
    log("INFO", "BOOT", f"ccxt.{ex_id} criado. Carregando mercados...")
    dex.load_markets()
    log("INFO", "BOOT", f"Mercados carregados: {len(dex.markets)}")

    cfg = Config()

    # Resolver símbolos por exchange
    use_hl = ('hyper' in ex_id.lower())
    resolved: List[Tuple[AssetSetup, str]] = []
    for a in ASSET_SETUPS:
        sym = None
        try:
            if use_hl:
                if a.hl_symbol in dex.markets:
                    sym = a.hl_symbol
                elif a.data_symbol in getattr(dex, 'markets_by_id', {}):
                    sym = dex.markets_by_id[a.data_symbol]['symbol']
            else:
                if a.data_symbol in getattr(dex, 'markets_by_id', {}):
                    sym = dex.markets_by_id[a.data_symbol]['symbol']
                elif a.hl_symbol in dex.markets:
                    sym = a.hl_symbol
        except Exception:
            sym = None
        if sym and sym in dex.markets:
            resolved.append((a, sym))
        else:
            log('WARN', 'BOOT', f'Skip {a.name}: não encontrado em {ex_id} (data={a.data_symbol}, hl={a.hl_symbol})')

    if not resolved:
        raise RuntimeError('Nenhum ativo resolvido para trading. Verifique exchange e símbolos.')

    log('INFO', 'BOOT', f'Resolvidos: {[sym for _, sym in resolved]}')

    # Cria estratégias
    strats: List[EMAGradientStrategy] = []
    for a, sym in resolved:
        st = EMAGradientStrategy(dex, sym, cfg, leverage=a.leverage, usd_env=a.usd_env or 'USD_RISK_PER_TRADE', name=a.name)
        strats.append(st)

    # Loop principal
    while True:
        for idx, st in enumerate(strats):
            try:
                st.step()
            except Exception as e:
                log("ERROR", st.symbol, f"step falhou: {type(e).__name__}: {e}")
            if STAGGER_BETWEEN_SYMBOLS_SEC and idx < len(strats) - 1:
                time.sleep(STAGGER_BETWEEN_SYMBOLS_SEC)
        log("INFO", "HEARTBEAT", time.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(cfg.SLEEP_SEC)

# Apenas para debug local (ex.: testar build_df sem rodar o bot)
if __name__ == "__main__":
    # Exemplo rápido de DF (opcional): define via env se quiser
    if os.getenv("DEBUG_BUILD_DF", "0") == "1":
        try:
            df_debug = build_df(os.getenv("SYMBOL_BINANCE", "BTCUSDT"), os.getenv("INTERVAL", "15m"), debug=True)
            log("INFO", "DATA", f"DF de exemplo criado: {len(df_debug)} linhas")
        except Exception as e:
            log("WARN", "DATA", f"build_df falhou: {e}")
    else:
        main()
