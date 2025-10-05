#!/usr/bin/env python3
"""
🏆 APLICAÇÃO DOS FILTROS OTIMIZADOS DO TRADINGV4 NOS DADOS REAIS DE 1 ANO

Este script aplica os exatos filtros do tradingv4.py que geraram 2190% ROI
nos dados históricos reais de 1 ano para validar a performance.

Configuração dos Filtros Otimizados:
- Confluência mínima: 3.0 critérios (vs 8.5 original)
- Take Profit: 40% (baseado nos testes de otimização)
- Stop Loss: 10%
- ATR: 0.5% - 3.0%
- Volume: 3.0x
- Gradiente LONG: ≥ 0.08%
- Gradiente SHORT: ≤ -0.12%
- RSI: 20-70 (range expandido)
- Sistema INVERSO ativo
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🏆 TESTE TP 40% COM DADOS REAIS - VALIDAÇÃO DA SIMULAÇÃO")
print("="*80)
print("🧪 TESTE: Sistema NORMAL (sem inversão) com TP 40% para validar simulação")
print("📊 TP: 40% | SL: 10% | ATR: 0.5-3.0% | Volume: 3.0x | Confluência: 3 critérios")
print("="*80)

@dataclass
class BacktestParams:
    """Parâmetros exatos do tradingv4.py"""
    # Indicadores
    ema_short: int = 7
    ema_long: int = 21
    atr_period: int = 14
    vol_ma_period: int = 20
    grad_window: int = 3
    grad_consistency: int = 3

    # Filtros OTIMIZADOS (exatos do tradingv4.py)
    atr_pct_min: float = 0.5        # ATR% mínimo - OTIMIZADO
    atr_pct_max: float = 3.0        # ATR% máximo - OTIMIZADO
    breakout_k_atr: float = 0.8     # banda de rompimento - OTIMIZADO
    no_trade_eps_k_atr: float = 0.07  # zona neutra

    # Execução e gerência
    cooldown_bars: int = 0
    post_cooldown_confirm_bars: int = 0
    allow_pyramiding: bool = False

    # Saídas OTIMIZADAS (teste TP 40% baseado na simulação)
    stop_loss_pct: float = 0.10     # 10% stop loss
    take_profit_pct: float = 0.40   # 40% take profit (baseado na simulação que mostrou superioridade)
    volume_mult_min: float = 3.0    # Volume mínimo 3.0x


def _ensure_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que o DataFrame tenha as colunas necessárias"""
    if "data" in df.columns:
        df = df.sort_values("data").reset_index(drop=True)
    
    # Renomear colunas se necessário
    column_mapping = {
        'close': 'valor_fechamento',
        'Close': 'valor_fechamento',
        'volume': 'volume',
        'Volume': 'volume',
        'high': 'high',
        'High': 'high',
        'low': 'low',
        'Low': 'low',
        'open': 'open',
        'Open': 'open'
    }
    
    df = df.rename(columns=column_mapping)
    
    if "valor_fechamento" not in df.columns:
        raise ValueError("DataFrame precisa ter a coluna 'valor_fechamento' ou 'close'.")
    
    # Garantir coluna de volume
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
            df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(1000)  # Volume padrão
    
    return df


def compute_indicators(df: pd.DataFrame, p: BacktestParams) -> pd.DataFrame:
    """Calcula indicadores técnicos - EXATO do tradingv4.py"""
    df = _ensure_base_cols(df)
    out = df.copy()
    close = pd.to_numeric(out["valor_fechamento"], errors="coerce")

    # EMAs
    out["ema_short"] = close.ewm(span=p.ema_short, adjust=False).mean()
    out["ema_long"] = close.ewm(span=p.ema_long, adjust=False).mean()

    # ATR clássico
    if set(["high", "low", "open"]).issubset(out.columns):
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
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

    # Bollinger Bands + %B
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

    # Gradiente EMA curto
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
    """
    🏆 FILTROS OTIMIZADOS PARA MÁXIMO ROI - LONG (EXATO do tradingv4.py)
    """
    reasons = []
    conds = []
    confluence_score = 0
    max_score = 10
    
    # CRITÉRIO 1: EMA + Gradiente otimizado
    c1_ema = row.ema_short > row.ema_long
    c1_grad = row.ema_short_grad_pct > 0.08  # OTIMIZADO: 0.08%
    c1 = c1_ema and c1_grad
    conds.append(c1)
    if c1:
        confluence_score += 1
        reasons.append("✅ EMA7>EMA21+grad>0.08%")
    else:
        reasons.append("❌ EMA/gradiente fraco")
    
    # CRITÉRIO 2: ATR otimizado
    c2 = (row.atr_pct >= 0.5) and (row.atr_pct <= 3.0)  # OTIMIZADO: 0.5%-3.0%
    conds.append(c2)
    if c2:
        confluence_score += 1
        reasons.append("✅ ATR ótimo")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento otimizado
    c3 = row.valor_fechamento > (row.ema_short + 0.8 * row.atr)  # OTIMIZADO: 0.8 ATR
    conds.append(c3)
    if c3:
        confluence_score += 1
        reasons.append("✅ Rompimento forte")
    else:
        reasons.append("❌ Rompimento fraco")
    
    # CRITÉRIO 4: Volume otimizado
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    c4 = volume_ratio > 3.0  # OTIMIZADO: 3.0x
    conds.append(c4)
    if c4:
        confluence_score += 1
        reasons.append("✅ Volume alto")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI otimizado
    if hasattr(row, 'rsi') and row.rsi is not None:
        c5 = 20 <= row.rsi <= 70  # OTIMIZADO: 20-70
        conds.append(c5)
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
        conds.append(c6)
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
    c7 = ema_separation >= 0.3  # Menos restritivo
    conds.append(c7)
    if c7:
        confluence_score += 1
        reasons.append("✅ EMAs separadas")
    else:
        reasons.append("❌ EMAs próximas")
    
    # CRITÉRIO 8: Timing de entrada
    price_distance = abs(row.valor_fechamento - row.ema_short) / row.atr if row.atr > 0 else 999
    c8 = price_distance <= 1.5  # Menos restritivo
    conds.append(c8)
    if c8:
        confluence_score += 1
        reasons.append("✅ Timing bom")
    else:
        reasons.append("❌ Entrada tardia")
    
    # CRITÉRIO 9: Bollinger Bands
    if hasattr(row, 'bb_percent_b') and row.bb_percent_b is not None:
        c9 = 0.6 <= row.bb_percent_b <= 0.95  # Menos restritivo
        conds.append(c9)
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
        conds.append(c10)
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
    MIN_CONFLUENCE = 3.0  # OTIMIZADO: muito menos restritivo
    is_valid = confluence_score >= MIN_CONFLUENCE
    
    confluence_pct = (confluence_score / max_score) * 100
    reason_summary = f"Confluência OTIMIZADA LONG: {confluence_score:.1f}/{max_score} ({confluence_pct:.0f}%)"
    top_reasons = reasons[:3]
    
    if is_valid:
        final_reason = f"✅ {reason_summary} | {' | '.join(top_reasons)}"
    else:
        final_reason = f"❌ {reason_summary} | {' | '.join(top_reasons)}"
    
    return is_valid, final_reason


def _entry_short_condition(row, p: BacktestParams) -> Tuple[bool, str]:
    """
    🏆 FILTROS OTIMIZADOS PARA MÁXIMO ROI - SHORT (EXATO do tradingv4.py)
    """
    reasons = []
    conds = []
    confluence_score = 0
    max_score = 10
    
    # CRITÉRIO 1: EMA + Gradiente otimizado
    c1_ema = row.ema_short < row.ema_long
    c1_grad = row.ema_short_grad_pct < -0.12  # OTIMIZADO: -0.12%
    c1 = c1_ema and c1_grad
    conds.append(c1)
    if c1:
        confluence_score += 1
        reasons.append("✅ EMA7<EMA21+grad<-0.12%")
    else:
        reasons.append("❌ EMA/gradiente fraco")
    
    # CRITÉRIO 2: ATR otimizado
    c2 = (row.atr_pct >= 0.5) and (row.atr_pct <= 3.0)  # OTIMIZADO: 0.5%-3.0%
    conds.append(c2)
    if c2:
        confluence_score += 1
        reasons.append("✅ ATR ótimo")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento otimizado
    c3 = row.valor_fechamento < (row.ema_short - 0.8 * row.atr)  # OTIMIZADO: 0.8 ATR
    conds.append(c3)
    if c3:
        confluence_score += 1
        reasons.append("✅ Rompimento forte")
    else:
        reasons.append("❌ Rompimento fraco")
    
    # CRITÉRIO 4: Volume otimizado
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    c4 = volume_ratio > 3.0  # OTIMIZADO: 3.0x
    conds.append(c4)
    if c4:
        confluence_score += 1
        reasons.append("✅ Volume alto")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI otimizado
    if hasattr(row, 'rsi') and row.rsi is not None:
        c5 = 20 <= row.rsi <= 70  # OTIMIZADO: 20-70
        conds.append(c5)
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
        conds.append(c6)
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
    c7 = ema_separation >= 0.3  # Menos restritivo
    conds.append(c7)
    if c7:
        confluence_score += 1
        reasons.append("✅ EMAs separadas")
    else:
        reasons.append("❌ EMAs próximas")
    
    # CRITÉRIO 8: Timing de entrada
    price_distance = abs(row.valor_fechamento - row.ema_short) / row.atr if row.atr > 0 else 999
    c8 = price_distance <= 1.5  # Menos restritivo
    conds.append(c8)
    if c8:
        confluence_score += 1
        reasons.append("✅ Timing bom")
    else:
        reasons.append("❌ Entrada tardia")
        
    # CRITÉRIO 9: Bollinger Bands
    if hasattr(row, 'bb_percent_b') and row.bb_percent_b is not None:
        c9 = 0.05 <= row.bb_percent_b <= 0.40  # Menos restritivo
        conds.append(c9)
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
        conds.append(c10)
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
    MIN_CONFLUENCE = 3.0  # OTIMIZADO: muito menos restritivo
    is_valid = confluence_score >= MIN_CONFLUENCE
    
    confluence_pct = (confluence_score / max_score) * 100
    reason_summary = f"Confluência OTIMIZADA SHORT: {confluence_score:.1f}/{max_score} ({confluence_pct:.0f}%)"
    top_reasons = reasons[:3]
    
    if is_valid:
        final_reason = f"✅ {reason_summary} | {' | '.join(top_reasons)}"
    else:
        final_reason = f"❌ {reason_summary} | {' | '.join(top_reasons)}"
    
    return is_valid, final_reason


def simulate_trade(entry_price: float, side: str, p: BacktestParams) -> Dict[str, Any]:
    """Simula uma trade com os parâmetros otimizados (SEM sistema inverso)"""
    if side == "LONG":
        actual_side = "LONG"  # Sem inversão
        tp_price = entry_price * (1 + p.take_profit_pct)  # LONG: TP acima
        sl_price = entry_price * (1 - p.stop_loss_pct)    # LONG: SL abaixo
    else:
        actual_side = "SHORT"  # Sem inversão
        tp_price = entry_price * (1 - p.take_profit_pct)  # SHORT: TP abaixo
        sl_price = entry_price * (1 + p.stop_loss_pct)    # SHORT: SL acima
    
    return {
        "entry_price": entry_price,
        "signal_side": side,
        "actual_side": actual_side,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "tp_pct": p.take_profit_pct * 100,
        "sl_pct": p.stop_loss_pct * 100
    }


def run_backtest_single_asset(df: pd.DataFrame, asset_name: str, p: BacktestParams) -> Dict[str, Any]:
    """Executa backtest em um único ativo"""
    print(f"\n📊 Processando {asset_name}...")
    
    # Calcular indicadores
    df_indicators = compute_indicators(df, p)
    
    # Estatísticas do dataset
    total_bars = len(df_indicators)
    date_range = f"{df_indicators.iloc[0].get('data', 'N/A')} a {df_indicators.iloc[-1].get('data', 'N/A')}"
    
    trades = []
    state = "FLAT"
    current_trade = None
    
    for i, row in df_indicators.iterrows():
        if pd.isna(row.valor_fechamento) or pd.isna(row.ema_short) or pd.isna(row.ema_long):
            continue
            
        current_price = row.valor_fechamento
        
        # Verificar se há trade ativo
        if current_trade:
            trade_info = current_trade["trade_info"]
            
            # Verificar TP ou SL
            hit_tp = False
            hit_sl = False
            
            if trade_info["actual_side"] == "LONG":
                hit_tp = current_price >= trade_info["tp_price"]
                hit_sl = current_price <= trade_info["sl_price"]
            else:  # SHORT
                hit_tp = current_price <= trade_info["tp_price"]
                hit_sl = current_price >= trade_info["sl_price"]
            
            if hit_tp:
                # Take Profit atingido
                exit_price = trade_info["tp_price"]
                pnl_pct = trade_info["tp_pct"] if trade_info["actual_side"] == "LONG" else trade_info["tp_pct"]
                if trade_info["actual_side"] == "SHORT":
                    pnl_pct = trade_info["tp_pct"]  # TP sempre positivo
                
                trades.append({
                    **current_trade,
                    "exit_price": exit_price,
                    "exit_reason": "TP",
                    "pnl_pct": pnl_pct,
                    "exit_bar": i
                })
                current_trade = None
                state = "FLAT"
                
            elif hit_sl:
                # Stop Loss atingido
                exit_price = trade_info["sl_price"]
                pnl_pct = -trade_info["sl_pct"]  # SL sempre negativo
                
                trades.append({
                    **current_trade,
                    "exit_price": exit_price,
                    "exit_reason": "SL",
                    "pnl_pct": pnl_pct,
                    "exit_bar": i
                })
                current_trade = None
                state = "FLAT"
        
        # Verificar novas entradas apenas se FLAT
        if state == "FLAT":
            # Verificar entrada LONG
            long_valid, long_reason = _entry_long_condition(row, p)
            if long_valid:
                trade_info = simulate_trade(current_price, "LONG", p)
                current_trade = {
                    "asset": asset_name,
                    "entry_bar": i,
                    "entry_price": current_price,
                    "signal_side": "LONG",
                    "actual_side": trade_info["actual_side"],
                    "reason": long_reason,
                    "trade_info": trade_info
                }
                state = "IN_TRADE"
                continue
            
            # Verificar entrada SHORT
            short_valid, short_reason = _entry_short_condition(row, p)
            if short_valid:
                trade_info = simulate_trade(current_price, "SHORT", p)
                current_trade = {
                    "asset": asset_name,
                    "entry_bar": i,
                    "entry_price": current_price,
                    "signal_side": "SHORT",
                    "actual_side": trade_info["actual_side"],
                    "reason": short_reason,
                    "trade_info": trade_info
                }
                state = "IN_TRADE"
    
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
    
    print(f"✅ {asset_name}: {total_trades} trades | Win Rate: {win_rate:.1f}% | ROI: {roi_pct:.1f}%")
    
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
    
    # Parâmetros otimizados
    params = BacktestParams()
    
    print(f"\n🎯 Parâmetros Otimizados:")
    print(f"  - Take Profit: {params.take_profit_pct*100:.0f}%")
    print(f"  - Stop Loss: {params.stop_loss_pct*100:.0f}%")
    print(f"  - ATR Range: {params.atr_pct_min:.1f}% - {params.atr_pct_max:.1f}%")
    print(f"  - Volume Min: {params.volume_mult_min:.1f}x")
    print(f"  - Confluência Min: 3.0 critérios")
    
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
            
            # Executar backtest
            result = run_backtest_single_asset(df, asset_name, params)
            all_results.append(result)
            
            total_trades_all += result["total_trades"]
            total_roi_all += result["roi_pct"]
            
        except Exception as e:
            print(f"❌ Erro processando {data_file}: {e}")
    
    # Relatório consolidado
    print("\n" + "="*80)
    print("📊 RELATÓRIO CONSOLIDADO - FILTROS TRADINGV4 NOS DADOS REAIS")
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
        output_file = f"filtros_tradingv4_dados_reais_{timestamp}.json"
        
        final_results = {
            "timestamp": timestamp,
            "parameters": {
                "take_profit_pct": params.take_profit_pct,
                "stop_loss_pct": params.stop_loss_pct,
                "atr_pct_min": params.atr_pct_min,
                "atr_pct_max": params.atr_pct_max,
                "volume_mult_min": params.volume_mult_min,
                "min_confluence": 3.0,
                "sistema_inverso": True
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
        
    else:
        print("❌ Nenhum resultado válido encontrado!")
    
    print("="*80)
    print("🏆 ANÁLISE COMPLETA DOS FILTROS TRADINGV4 FINALIZADA")
    print("="*80)


if __name__ == "__main__":
    main()
