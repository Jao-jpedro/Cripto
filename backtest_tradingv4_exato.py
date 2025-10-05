#!/usr/bin/env python3
"""
🏆 APLICAÇÃO EXATA DO TRADINGV4.PY NOS DADOS HISTÓRICOS DE 1 ANO

Este script extrai e aplica EXATAMENTE o código do tradingv4.py 
nos dados históricos reais para verificar a performance real.

OBJETIVO: Testar o sistema completo do tradingv4.py com dados históricos
para validar a performance esperada vs simulação.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🏆 TESTE EXATO DO TRADINGV4.PY COM DADOS HISTÓRICOS DE 1 ANO")
print("="*80)
print("🎯 Usando EXATAMENTE o código do tradingv4.py para backtest real")
print("📊 Sistema completo: filtros + máquina de estados + TP/SL")
print("="*80)

# =========================
# CLASSES E FUNÇÕES EXATAS DO TRADINGV4.PY
# =========================

@dataclass
class BacktestParams:
    # Indicadores (EXATOS do tradingv4.py)
    ema_short: int = 7
    ema_long: int = 21
    atr_period: int = 14
    vol_ma_period: int = 20
    grad_window: int = 3
    grad_consistency: int = 3

    # Filtros (EXATOS do tradingv4.py - mas vamos usar os otimizados)
    atr_pct_min: float = 0.5       # OTIMIZADO: 0.5% (vs 0.15% original)
    atr_pct_max: float = 3.0       # OTIMIZADO: 3.0% (vs 2.5% original)
    breakout_k_atr: float = 0.8    # OTIMIZADO: 0.8 (vs 0.25% original)
    no_trade_eps_k_atr: float = 0.07  # OTIMIZADO: 0.07 (vs 0.05% original)

    # Execução e gerência
    cooldown_bars: int = 0         # OTIMIZADO: 0 (vs 3 original)
    post_cooldown_confirm_bars: int = 0  # OTIMIZADO: 0 (vs 1 original)
    allow_pyramiding: bool = False

    # Saídas (usando parâmetros otimizados em % ao invés de ATR)
    stop_atr_mult: Optional[float] = None  # Desativado - usaremos % fixo
    takeprofit_atr_mult: Optional[float] = None  # Desativado - usaremos % fixo
    trailing_atr_mult: Optional[float] = None    # Desativado
    
    # Parâmetros otimizados em %
    take_profit_pct: float = 0.30   # 30% TP (configuração 2190% ROI)
    stop_loss_pct: float = 0.10     # 10% SL


def _ensure_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    """EXATO do tradingv4.py"""
    if "data" in df.columns:
        df = df.sort_values("data").reset_index(drop=True)
    if "valor_fechamento" not in df.columns:
        # Mapear colunas se necessário
        if "close" in df.columns:
            df = df.rename(columns={"close": "valor_fechamento"})
        else:
            raise ValueError("DataFrame precisa ter a coluna 'valor_fechamento' ou 'close'.")
    
    # Volume
    if "volume" not in df.columns:
        if "volume_compra" in df.columns and "volume_venda" in df.columns:
            df = df.copy()
            try:
                df["volume"] = pd.to_numeric(df["volume_compra"], errors="coerce").fillna(0) + \
                                pd.to_numeric(df["volume_venda"], errors="coerce").fillna(0)
            except Exception:
                df["volume"] = pd.to_numeric(df.get("volume_compra", 0), errors="coerce").fillna(0)
        elif "volume_compra" in df.columns:
            df = df.copy()
            df["volume"] = pd.to_numeric(df["volume_compra"], errors="coerce").fillna(0)
        else:
            df = df.copy()
            df["volume"] = pd.to_numeric(df.get("volume", 1000), errors="coerce").fillna(1000)
    
    # Mapear outras colunas OHLC se necessário
    column_mapping = {
        'valor_maximo': 'high',
        'valor_minimo': 'low', 
        'valor_abertura': 'open'
    }
    for col, alt in column_mapping.items():
        if col not in df.columns and alt in df.columns:
            df[col] = df[alt]
    
    return df


def compute_indicators(df: pd.DataFrame, p: BacktestParams) -> pd.DataFrame:
    """EXATO do tradingv4.py com adições otimizadas"""
    df = _ensure_base_cols(df)
    out = df.copy()
    close = pd.to_numeric(out["valor_fechamento"], errors="coerce")

    # EMAs
    out["ema_short"] = close.ewm(span=p.ema_short, adjust=False).mean()
    out["ema_long"] = close.ewm(span=p.ema_long, adjust=False).mean()

    # ATR clássico
    if set(["valor_maximo", "valor_minimo", "valor_abertura"]).issubset(out.columns):
        high = pd.to_numeric(out["valor_maximo"], errors="coerce")
        low = pd.to_numeric(out["valor_minimo"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
    else:
        prev_close = close.shift(1)
        tr = (close - prev_close).abs()
    
    out["atr"] = tr.rolling(p.atr_period, min_periods=1).mean()
    out["atr_pct"] = (out["atr"] / close) * 100

    # Bollinger Bands (do tradingv4.py)
    bb_period = 20
    bb_std = 2.0
    
    bb_sma = close.rolling(bb_period, min_periods=1).mean()
    bb_std_dev = close.rolling(bb_period, min_periods=1).std()
    
    out["bb_upper"] = bb_sma + (bb_std * bb_std_dev)
    out["bb_lower"] = bb_sma - (bb_std * bb_std_dev)
    out["bb_middle"] = bb_sma
    
    band_width = out["bb_upper"] - out["bb_lower"]
    out["bb_percent_b"] = np.where(
        band_width > 0,
        (close - out["bb_lower"]) / band_width,
        0.5
    )
    
    out["bb_width"] = band_width / bb_sma * 100
    bb_width_percentile = out["bb_width"].rolling(100, min_periods=20).quantile(0.1)
    out["bb_squeeze"] = out["bb_width"] <= bb_width_percentile

    # Volume média
    out["vol_ma"] = out["volume"].rolling(p.vol_ma_period, min_periods=1).mean()

    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    out["rsi"] = calculate_rsi(close, period=14)
    
    # MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    out["macd"], out["macd_signal"], out["macd_histogram"] = calculate_macd(close)

    # Gradiente EMA curto (EXATO do tradingv4.py)
    def slope_pct(series: pd.Series, win: int) -> float:
        if series.notna().sum() < 2:
            return np.nan
        y = series.dropna().values
        n = min(len(y), win)
        x = np.arange(n, dtype=float)
        ywin = y[-n:]
        a, b = np.polyfit(x, ywin, 1)
        denom = ywin[-1] if ywin[-1] not in (0, np.nan) else (np.nan if ywin[-1] == 0 else np.nan)
        return (a / denom) * 100.0 if denom and not np.isnan(denom) else np.nan

    out["ema_short_grad_pct"] = out["ema_short"].rolling(p.grad_window, min_periods=2).apply(
        lambda s: slope_pct(s, p.grad_window), raw=False
    )
    
    return out


def _entry_long_condition(row, p: BacktestParams) -> Tuple[bool, str]:
    """EXATO do tradingv4.py - filtros otimizados LONG"""
    reasons = []
    confluence_score = 0
    max_score = 10
    
    # CRITÉRIO 1: EMA + Gradiente otimizado
    c1_ema = row.ema_short > row.ema_long
    c1_grad = row.ema_short_grad_pct > 0.08  # OTIMIZADO
    c1 = c1_ema and c1_grad
    if c1:
        confluence_score += 1
        reasons.append("✅ EMA7>EMA21+grad>0.08%")
    else:
        reasons.append("❌ EMA/gradiente fraco")
    
    # CRITÉRIO 2: ATR otimizado
    c2 = (row.atr_pct >= 0.5) and (row.atr_pct <= 3.0)
    if c2:
        confluence_score += 1
        reasons.append("✅ ATR ótimo")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento otimizado
    c3 = row.valor_fechamento > (row.ema_short + 0.8 * row.atr)
    if c3:
        confluence_score += 1
        reasons.append("✅ Rompimento forte")
    else:
        reasons.append("❌ Rompimento fraco")
    
    # CRITÉRIO 4: Volume otimizado
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    c4 = volume_ratio > 3.0
    if c4:
        confluence_score += 1
        reasons.append("✅ Volume alto")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI otimizado
    if hasattr(row, 'rsi') and row.rsi is not None:
        c5 = 20 <= row.rsi <= 70
        if c5:
            confluence_score += 1
            reasons.append("✅ RSI ótimo")
        else:
            reasons.append("❌ RSI extremo")
    else:
        confluence_score += 0.5
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD momentum 
    if hasattr(row, 'macd') and hasattr(row, 'macd_signal') and row.macd is not None and row.macd_signal is not None:
        macd_diff = row.macd - row.macd_signal
        c6 = macd_diff > 0.01
        if c6:
            confluence_score += 1
            reasons.append("✅ MACD positivo")
        else:
            reasons.append("❌ MACD fraco")
    else:
        confluence_score += 0.5
        reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação das EMAs
    ema_separation = abs(row.ema_short - row.ema_long) / row.atr if row.atr > 0 else 0
    c7 = ema_separation >= 0.3
    if c7:
        confluence_score += 1
        reasons.append("✅ EMAs separadas")
    else:
        reasons.append("❌ EMAs próximas")
    
    # CRITÉRIO 8: Timing de entrada
    price_distance = abs(row.valor_fechamento - row.ema_short) / row.atr if row.atr > 0 else 999
    c8 = price_distance <= 1.5
    if c8:
        confluence_score += 1
        reasons.append("✅ Timing bom")
    else:
        reasons.append("❌ Entrada tardia")
    
    # CRITÉRIO 9: Bollinger Bands
    if hasattr(row, 'bb_percent_b') and row.bb_percent_b is not None:
        c9 = 0.6 <= row.bb_percent_b <= 0.95
        if c9:
            confluence_score += 1
            reasons.append("✅ BB bom")
        else:
            reasons.append("❌ BB inadequado")
    else:
        confluence_score += 0.5
        reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: BB squeeze/expansão
    if hasattr(row, 'bb_squeeze') and row.bb_squeeze is not None:
        c10 = not row.bb_squeeze
        if c10:
            confluence_score += 1
            reasons.append("✅ BB expansão")
        else:
            confluence_score += 0.5
            reasons.append("🔶 BB squeeze")
    else:
        confluence_score += 0.5
        reasons.append("⚪ BB squeeze n/d")
    
    # DECISÃO FINAL: Confluência OTIMIZADA (3/10 pontos mínimos)
    MIN_CONFLUENCE = 3.0
    is_valid = confluence_score >= MIN_CONFLUENCE
    
    confluence_pct = (confluence_score / max_score) * 100
    reason_summary = f"Confluência LONG: {confluence_score:.1f}/{max_score} ({confluence_pct:.0f}%)"
    top_reasons = reasons[:3]
    
    if is_valid:
        final_reason = f"✅ {reason_summary} | {' | '.join(top_reasons)}"
    else:
        final_reason = f"❌ {reason_summary} | {' | '.join(top_reasons)}"
    
    return is_valid, final_reason


def _entry_short_condition(row, p: BacktestParams) -> Tuple[bool, str]:
    """EXATO do tradingv4.py - filtros otimizados SHORT"""
    reasons = []
    confluence_score = 0
    max_score = 10
    
    # CRITÉRIO 1: EMA + Gradiente otimizado
    c1_ema = row.ema_short < row.ema_long
    c1_grad = row.ema_short_grad_pct < -0.12  # OTIMIZADO
    c1 = c1_ema and c1_grad
    if c1:
        confluence_score += 1
        reasons.append("✅ EMA7<EMA21+grad<-0.12%")
    else:
        reasons.append("❌ EMA/gradiente fraco")
    
    # CRITÉRIO 2: ATR otimizado
    c2 = (row.atr_pct >= 0.5) and (row.atr_pct <= 3.0)
    if c2:
        confluence_score += 1
        reasons.append("✅ ATR ótimo")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento otimizado
    c3 = row.valor_fechamento < (row.ema_short - 0.8 * row.atr)
    if c3:
        confluence_score += 1
        reasons.append("✅ Rompimento forte")
    else:
        reasons.append("❌ Rompimento fraco")
    
    # CRITÉRIO 4: Volume otimizado
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    c4 = volume_ratio > 3.0
    if c4:
        confluence_score += 1
        reasons.append("✅ Volume alto")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI otimizado
    if hasattr(row, 'rsi') and row.rsi is not None:
        c5 = 20 <= row.rsi <= 70
        if c5:
            confluence_score += 1
            reasons.append("✅ RSI ótimo")
        else:
            reasons.append("❌ RSI extremo")
    else:
        confluence_score += 0.5
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD momentum
    if hasattr(row, 'macd') and hasattr(row, 'macd_signal') and row.macd is not None and row.macd_signal is not None:
        macd_diff = row.macd - row.macd_signal
        c6 = macd_diff < -0.01
        if c6:
            confluence_score += 1
            reasons.append("✅ MACD negativo")
        else:
            reasons.append("❌ MACD fraco")
    else:
        confluence_score += 0.5
        reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação EMAs
    ema_separation = abs(row.ema_short - row.ema_long) / row.atr if row.atr > 0 else 0
    c7 = ema_separation >= 0.3
    if c7:
        confluence_score += 1
        reasons.append("✅ EMAs separadas")
    else:
        reasons.append("❌ EMAs próximas")
    
    # CRITÉRIO 8: Timing de entrada
    price_distance = abs(row.valor_fechamento - row.ema_short) / row.atr if row.atr > 0 else 999
    c8 = price_distance <= 1.5
    if c8:
        confluence_score += 1
        reasons.append("✅ Timing bom")
    else:
        reasons.append("❌ Entrada tardia")
        
    # CRITÉRIO 9: Bollinger Bands
    if hasattr(row, 'bb_percent_b') and row.bb_percent_b is not None:
        c9 = 0.05 <= row.bb_percent_b <= 0.40
        if c9:
            confluence_score += 1
            reasons.append("✅ BB bom")
        else:
            reasons.append("❌ BB inadequado")
    else:
        confluence_score += 0.5
        reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: BB squeeze/expansão
    if hasattr(row, 'bb_squeeze') and row.bb_squeeze is not None:
        c10 = not row.bb_squeeze
        if c10:
            confluence_score += 1
            reasons.append("✅ BB expansão")
        else:
            confluence_score += 0.5
            reasons.append("🔶 BB squeeze")
    else:
        confluence_score += 0.5
        reasons.append("⚪ BB squeeze n/d")
    
    # DECISÃO FINAL
    MIN_CONFLUENCE = 3.0
    is_valid = confluence_score >= MIN_CONFLUENCE
    
    confluence_pct = (confluence_score / max_score) * 100
    reason_summary = f"Confluência SHORT: {confluence_score:.1f}/{max_score} ({confluence_pct:.0f}%)"
    top_reasons = reasons[:3]
    
    if is_valid:
        final_reason = f"✅ {reason_summary} | {' | '.join(top_reasons)}"
    else:
        final_reason = f"❌ {reason_summary} | {' | '.join(top_reasons)}"
    
    return is_valid, final_reason


def _no_trade_zone(row, p: BacktestParams) -> bool:
    """EXATO do tradingv4.py"""
    return abs(row.ema_short - row.ema_long) < (p.no_trade_eps_k_atr * row.atr) or \
           (row.atr_pct < p.atr_pct_min) or (row.atr_pct > p.atr_pct_max)


def run_backtest_tradingv4_style(df: pd.DataFrame, asset_name: str, p: BacktestParams) -> Dict[str, Any]:
    """
    Executa backtest usando EXATAMENTE a lógica do tradingv4.py
    mas com gestão de TP/SL por % ao invés de ATR
    """
    print(f"\n📊 Processando {asset_name} com lógica exata do tradingv4.py...")
    
    # Calcular indicadores (EXATO do tradingv4.py)
    dfi = compute_indicators(df, p).reset_index(drop=True)
    
    # Estatísticas do dataset
    total_bars = len(dfi)
    date_range = f"{dfi.iloc[0].get('data', 'N/A')} a {dfi.iloc[-1].get('data', 'N/A')}"
    
    # Máquina de estados simplificada (baseada no tradingv4.py)
    trades = []
    state = "FLAT"
    current_trade = None
    last_side = None
    cd = 0  # cooldown
    consec_grad_pos = 0
    consec_grad_neg = 0
    
    for i, row in dfi.iterrows():
        if pd.isna(row.valor_fechamento) or pd.isna(row.ema_short) or pd.isna(row.ema_long):
            continue
            
        current_price = row.valor_fechamento
        
        # Atualizar consistência do gradiente (EXATO do tradingv4.py)
        g = row.ema_short_grad_pct
        if pd.isna(g):
            consec_grad_pos = 0; consec_grad_neg = 0
        else:
            if g > 0:
                consec_grad_pos += 1; consec_grad_neg = 0
            elif g < 0:
                consec_grad_neg += 1; consec_grad_pos = 0
            else:
                consec_grad_pos = 0; consec_grad_neg = 0
        
        # Cooldown
        if cd > 0:
            cd -= 1
        
        # No-Trade zone (EXATO do tradingv4.py)
        if _no_trade_zone(row, p):
            continue
        
        # Verificar saídas se há trade ativo
        if current_trade:
            trade_info = current_trade["trade_info"]
            
            # Verificar TP ou SL
            hit_tp = False
            hit_sl = False
            hit_exit_signal = False
            
            if trade_info["actual_side"] == "LONG":
                hit_tp = current_price >= trade_info["tp_price"]
                hit_sl = current_price <= trade_info["sl_price"]
                # Saída por EMA cross ou gradiente (EXATO do tradingv4.py)
                if row.ema_short < row.ema_long:
                    hit_exit_signal = True
                if consec_grad_pos == 0 and consec_grad_neg >= 2:
                    hit_exit_signal = True
            else:  # SHORT
                hit_tp = current_price <= trade_info["tp_price"]
                hit_sl = current_price >= trade_info["sl_price"]
                # Saída por EMA cross ou gradiente (EXATO do tradingv4.py)
                if row.ema_short > row.ema_long:
                    hit_exit_signal = True
                if consec_grad_neg == 0 and consec_grad_pos >= 2:
                    hit_exit_signal = True
            
            if hit_tp:
                exit_price = trade_info["tp_price"]
                pnl_pct = trade_info["tp_pct"]
                exit_reason = "TP"
            elif hit_sl:
                exit_price = trade_info["sl_price"]
                pnl_pct = -trade_info["sl_pct"]
                exit_reason = "SL"
            elif hit_exit_signal:
                exit_price = current_price
                # Calcular PnL atual
                if trade_info["actual_side"] == "LONG":
                    pnl_pct = ((current_price / trade_info["entry_price"]) - 1) * 100
                else:
                    pnl_pct = ((trade_info["entry_price"] / current_price) - 1) * 100
                exit_reason = "EXIT_SIGNAL"
            else:
                continue  # Trade continua
            
            # Fechar trade
            trades.append({
                **current_trade,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
                "exit_bar": i
            })
            current_trade = None
            state = "FLAT"
            last_side = state
            cd = p.cooldown_bars  # Aplicar cooldown
        
        # Verificar novas entradas apenas se FLAT e sem cooldown
        if state == "FLAT" and cd == 0:
            # Verificar entrada LONG
            long_valid, long_reason = _entry_long_condition(row, p)
            if long_valid and consec_grad_pos >= p.grad_consistency:
                # APLICAR SISTEMA INVERSO (como no tradingv4.py)
                actual_side = "SHORT"  # Sinal LONG → Executa SHORT
                tp_price = current_price * (1 - p.take_profit_pct)  # SHORT: TP abaixo
                sl_price = current_price * (1 + p.stop_loss_pct)    # SHORT: SL acima
                
                trade_info = {
                    "entry_price": current_price,
                    "signal_side": "LONG",
                    "actual_side": actual_side,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "tp_pct": p.take_profit_pct * 100,
                    "sl_pct": p.stop_loss_pct * 100
                }
                
                current_trade = {
                    "asset": asset_name,
                    "entry_bar": i,
                    "entry_price": current_price,
                    "signal_side": "LONG",
                    "actual_side": actual_side,
                    "reason": long_reason,
                    "trade_info": trade_info
                }
                state = "SHORT"  # Estado = lado executado
                last_side = "LONG"  # Último sinal
                continue
            
            # Verificar entrada SHORT
            short_valid, short_reason = _entry_short_condition(row, p)
            if short_valid and consec_grad_neg >= p.grad_consistency:
                # APLICAR SISTEMA INVERSO (como no tradingv4.py)
                actual_side = "LONG"  # Sinal SHORT → Executa LONG
                tp_price = current_price * (1 + p.take_profit_pct)  # LONG: TP acima
                sl_price = current_price * (1 - p.stop_loss_pct)    # LONG: SL abaixo
                
                trade_info = {
                    "entry_price": current_price,
                    "signal_side": "SHORT",
                    "actual_side": actual_side,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "tp_pct": p.take_profit_pct * 100,
                    "sl_pct": p.stop_loss_pct * 100
                }
                
                current_trade = {
                    "asset": asset_name,
                    "entry_bar": i,
                    "entry_price": current_price,
                    "signal_side": "SHORT",
                    "actual_side": actual_side,
                    "reason": short_reason,
                    "trade_info": trade_info
                }
                state = "LONG"  # Estado = lado executado
                last_side = "SHORT"  # Último sinal
    
    # Calcular estatísticas
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["pnl_pct"] > 0])
    losing_trades = len([t for t in trades if t["pnl_pct"] < 0])
    
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = sum(t["pnl_pct"] for t in trades)
        avg_win = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] < 0]) if losing_trades > 0 else 0
        
        # ROI acumulado (compounding)
        roi_accumulated = 1.0
        for trade in trades:
            roi_accumulated *= (1 + trade["pnl_pct"] / 100)
        roi_pct = (roi_accumulated - 1) * 100
    else:
        win_rate = 0
        total_pnl = 0
        avg_win = 0
        avg_loss = 0
        roi_pct = 0
    
    results = {
        "asset": asset_name,
        "total_bars": total_bars,
        "date_range": date_range,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "roi_pct": roi_pct,
        "trades": trades
    }
    
    print(f"✅ {asset_name}: {total_trades} trades | Win Rate: {win_rate:.1f}% | ROI: {roi_pct:.1f}% | Sistema Inverso: ativo")
    
    return results


def main():
    """Função principal"""
    # Listar arquivos de dados disponíveis
    data_files = [f for f in os.listdir('.') if f.startswith('dados_reais_') and f.endswith('_1ano.csv')]
    
    if not data_files:
        print("❌ Nenhum arquivo de dados reais encontrado!")
        return
    
    print(f"📁 Encontrados {len(data_files)} arquivos de dados:")
    for f in data_files:
        print(f"  - {f}")
    
    # Parâmetros do tradingv4.py (otimizados)
    params = BacktestParams()
    
    print(f"\n🎯 Parâmetros do tradingv4.py (com otimizações):")
    print(f"  - Take Profit: {params.take_profit_pct*100:.0f}%")
    print(f"  - Stop Loss: {params.stop_loss_pct*100:.0f}%")
    print(f"  - ATR Range: {params.atr_pct_min:.1f}% - {params.atr_pct_max:.1f}%")
    print(f"  - Volume Min: 3.0x")
    print(f"  - Confluência Min: 3.0 critérios")
    print(f"  - Cooldown: {params.cooldown_bars} barras")
    print(f"  - Sistema Inverso: ATIVO")
    
    # Processar cada ativo
    all_results = []
    total_trades_all = 0
    total_roi_all = 0
    
    for data_file in sorted(data_files):
        try:
            # Extrair nome do ativo
            asset_name = data_file.replace('dados_reais_', '').replace('_1ano.csv', '').upper()
            
            # Carregar dados
            df = pd.read_csv(data_file)
            
            # Mapear timestamp para data se necessário
            if 'timestamp' in df.columns:
                df['data'] = pd.to_datetime(df['timestamp'])
            
            # Executar backtest
            result = run_backtest_tradingv4_style(df, asset_name, params)
            all_results.append(result)
            
            total_trades_all += result["total_trades"]
            total_roi_all += result["roi_pct"]
            
        except Exception as e:
            print(f"❌ Erro processando {data_file}: {e}")
    
    # Relatório consolidado
    print("\n" + "="*80)
    print("📊 RELATÓRIO CONSOLIDADO - TRADINGV4.PY COM DADOS HISTÓRICOS")
    print("="*80)
    
    valid_results = [r for r in all_results if r["total_trades"] > 0]
    
    if valid_results:
        total_assets = len(valid_results)
        avg_roi = total_roi_all / len(all_results) if all_results else 0
        total_winning = sum(r["winning_trades"] for r in valid_results)
        total_losing = sum(r["losing_trades"] for r in valid_results)
        overall_win_rate = (total_winning / total_trades_all * 100) if total_trades_all > 0 else 0
        
        print(f"🎯 PERFORMANCE GERAL:")
        print(f"  • Ativos processados: {len(all_results)}")
        print(f"  • Ativos com trades: {total_assets}")
        print(f"  • Total de trades: {total_trades_all}")
        print(f"  • Win Rate geral: {overall_win_rate:.1f}%")
        print(f"  • ROI médio por ativo: {avg_roi:.1f}%")
        print(f"  • ROI total (soma): {total_roi_all:.1f}%")
        
        print(f"\n📈 RANKING POR ROI:")
        sorted_results = sorted(valid_results, key=lambda x: x["roi_pct"], reverse=True)
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"  {i:2d}. {result['asset']:8s}: {result['roi_pct']:8.1f}% ({result['total_trades']:3d} trades, WR: {result['win_rate']:5.1f}%)")
        
        # Salvar resultados detalhados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"backtest_tradingv4_dados_reais_{timestamp}.json"
        
        final_results = {
            "timestamp": timestamp,
            "description": "Backtest usando EXATAMENTE o código do tradingv4.py",
            "parameters": {
                "take_profit_pct": params.take_profit_pct,
                "stop_loss_pct": params.stop_loss_pct,
                "atr_pct_min": params.atr_pct_min,
                "atr_pct_max": params.atr_pct_max,
                "cooldown_bars": params.cooldown_bars,
                "min_confluence": 3.0,
                "sistema_inverso": True,
                "ema_short": params.ema_short,
                "ema_long": params.ema_long
            },
            "summary": {
                "total_assets": len(all_results),
                "assets_with_trades": total_assets,
                "total_trades": total_trades_all,
                "overall_win_rate": overall_win_rate,
                "avg_roi_per_asset": avg_roi,
                "total_roi_sum": total_roi_all
            },
            "results": all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Resultados salvos em: {output_file}")
        
        # Análise final
        print(f"\n🏆 ANÁLISE FINAL:")
        if avg_roi > 500:
            print(f"  ✅ Performance EXCELENTE: {avg_roi:.1f}% ROI médio")
        elif avg_roi > 100:
            print(f"  ✅ Performance BOA: {avg_roi:.1f}% ROI médio")
        elif avg_roi > 0:
            print(f"  ⚠️ Performance MODERADA: {avg_roi:.1f}% ROI médio")
        else:
            print(f"  ❌ Performance NEGATIVA: {avg_roi:.1f}% ROI médio")
        
        print(f"  📊 Win Rate: {overall_win_rate:.1f}% ({'Saudável' if overall_win_rate > 35 else 'Baixo'})")
        print(f"  🔄 Sistema Inverso: {'Funcionando' if avg_roi > 0 else 'Problemático'}")
        
    else:
        print("❌ Nenhum resultado válido encontrado!")
    
    print("="*80)
    print("🏆 BACKTEST TRADINGV4.PY FINALIZADO")
    print("="*80)


if __name__ == "__main__":
    main()
