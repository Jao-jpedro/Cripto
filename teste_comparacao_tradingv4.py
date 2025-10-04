#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 TESTE DE COMPARAÇÃO: Minha Análise vs tradingv4
Replica exatamente os cálculos do tradingv4 para identificar diferenças
"""

import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

def get_binance_data(symbol: str, interval: str = "15m", limit: int = 260) -> pd.DataFrame:
    """Busca dados da Binance com mesmo limite do tradingv4"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df.set_index('timestamp')
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()

def calculate_atr_tradingv4_style(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calcula ATR exatamente como o tradingv4"""
    if df.empty or len(df) < period:
        return df
    
    # True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR usando Wilder's smoothing (como tradingv4)
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    return df

def calculate_emas_tradingv4_style(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula EMAs como o tradingv4"""
    if df.empty:
        return df
    
    # EMAs 7 e 21
    df['ema7'] = df['close'].ewm(span=7).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    
    # Gradiente EMA7 (% change últimas 3 velas)
    df['ema7_grad_pct'] = df['ema7'].pct_change(periods=3) * 100
    
    return df

def calculate_volume_tradingv4_style(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Calcula volume como o tradingv4"""
    if df.empty:
        return df
    
    df['vol_ma'] = df['volume'].rolling(window=period).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    return df

def test_bnb_comparison():
    """Testa BNB com dados exatos do tradingv4"""
    print("🔍 TESTE DE COMPARAÇÃO: BNB - Minha Análise vs tradingv4")
    print("=" * 70)
    
    # Buscar dados com mesmo limite do tradingv4
    print("📊 Buscando dados BNBUSDT (limite=260, como tradingv4)...")
    df = get_binance_data("BNBUSDT", "15m", 260)
    
    if df.empty:
        print("❌ Falha ao obter dados")
        return
    
    print(f"✅ Obtidos {len(df)} candles")
    print(f"📅 Período: {df.index[0]} até {df.index[-1]}")
    
    # Calcular indicadores como tradingv4
    df = calculate_atr_tradingv4_style(df)
    df = calculate_emas_tradingv4_style(df)
    df = calculate_volume_tradingv4_style(df)
    
    # Remover NaNs
    df = df.dropna()
    
    if len(df) < 50:
        print("❌ Dados insuficientes após cálculos")
        return
    
    # Últimos dados
    last_row = df.iloc[-1]
    
    print(f"\n📊 DADOS CALCULADOS (último candle):")
    print(f"⏰ Timestamp: {last_row.name}")
    print(f"💰 Preço: ${last_row.close:.2f}")
    print(f"📈 EMA7: {last_row.ema7:.6f}")
    print(f"📉 EMA21: {last_row.ema21:.6f}")
    print(f"📊 ATR: {last_row.atr:.6f}")
    print(f"🎯 ATR%: {last_row.atr_pct:.3f}%")
    print(f"📊 Volume: {last_row.volume:,.0f}")
    print(f"📊 Vol MA: {last_row.vol_ma:,.0f}")
    print(f"📊 Vol Ratio: {last_row.vol_ratio:.2f}x")
    print(f"📈 Gradiente EMA7: {last_row.ema7_grad_pct:.3f}%")
    
    # Comparar com dados do log tradingv4
    print(f"\n🔍 COMPARAÇÃO COM LOG TRADINGV4:")
    print(f"{'MÉTRICA':<20} {'MEUS DADOS':<15} {'TRADINGV4':<15} {'DIFERENÇA':<15}")
    print("-" * 70)
    
    # Preço
    tradingv4_price = 1167.95
    price_diff = last_row.close - tradingv4_price
    print(f"{'Preço':<20} ${last_row.close:<14.2f} ${tradingv4_price:<14.2f} {price_diff:+.2f}")
    
    # ATR%
    tradingv4_atr_pct = 0.290
    atr_diff = last_row.atr_pct - tradingv4_atr_pct
    atr_diff_pct = (atr_diff / tradingv4_atr_pct) * 100
    print(f"{'ATR%':<20} {last_row.atr_pct:<14.3f} {tradingv4_atr_pct:<14.3f} {atr_diff:+.3f} ({atr_diff_pct:+.0f}%)")
    
    # Volume
    tradingv4_vol = 1587666.66
    tradingv4_vol_ma = 6336826.74
    tradingv4_vol_ratio = tradingv4_vol / tradingv4_vol_ma
    
    vol_diff = last_row.vol_ratio - tradingv4_vol_ratio
    print(f"{'Volume':<20} {last_row.volume:<14,.0f} {tradingv4_vol:<14,.0f} {last_row.volume - tradingv4_vol:+,.0f}")
    print(f"{'Vol Ratio':<20} {last_row.vol_ratio:<14.2f} {tradingv4_vol_ratio:<14.2f} {vol_diff:+.2f}")
    
    # Gradiente
    tradingv4_grad = -0.0863
    grad_diff = last_row.ema7_grad_pct - tradingv4_grad
    print(f"{'Gradiente%':<20} {last_row.ema7_grad_pct:<14.3f} {tradingv4_grad:<14.3f} {grad_diff:+.3f}")
    
    # Análise dos critérios
    print(f"\n🎯 ANÁLISE DOS CRITÉRIOS:")
    
    # No-trade zone
    eps = 0.05 * last_row.atr
    ema_diff = abs(last_row.ema7 - last_row.ema21)
    atr_healthy = 0.5 <= last_row.atr_pct <= 3.0
    no_trade = (ema_diff < eps) or (not atr_healthy)
    
    print(f"   • ATR saudável: {last_row.atr_pct:.3f}% >= 0.5% = {atr_healthy}")
    print(f"   • |EMA7-EMA21|: {ema_diff:.6f} vs eps: {eps:.6f} = {ema_diff >= eps}")
    print(f"   • No-Trade Zone: {no_trade}")
    
    # Critérios SHORT
    ema_short_trend = last_row.ema7 < last_row.ema21
    grad_ok = last_row.ema7_grad_pct < -0.12  # tradingv4 usa -0.12%
    atr_ok = atr_healthy
    breakout_level = last_row.ema7 - (0.8 * last_row.atr)
    breakout_ok = last_row.close < breakout_level
    vol_ok = last_row.vol_ratio > 3.0
    
    print(f"\n🟢 CRITÉRIOS SHORT:")
    print(f"   • EMA7 < EMA21: {ema_short_trend}")
    print(f"   • Gradiente < -0.12%: {grad_ok} ({last_row.ema7_grad_pct:.3f}%)")
    print(f"   • ATR OK: {atr_ok}")
    print(f"   • Breakout: {breakout_ok} (preço {last_row.close:.2f} vs {breakout_level:.2f})")
    print(f"   • Volume OK: {vol_ok} ({last_row.vol_ratio:.2f}x)")
    
    criterios_short = sum([ema_short_trend, grad_ok, atr_ok, breakout_ok, vol_ok])
    print(f"   • TOTAL: {criterios_short}/5 critérios")
    
    # Conclusão
    print(f"\n📋 CONCLUSÃO:")
    if no_trade:
        print("   🚫 NO-TRADE ZONE ativa (como tradingv4)")
        if not atr_healthy:
            print(f"      → ATR muito baixo: {last_row.atr_pct:.3f}% < 0.5%")
        if ema_diff < eps:
            print(f"      → EMAs muito próximas: {ema_diff:.6f} < {eps:.6f}")
    elif criterios_short >= 3:
        print("   ✅ Sinal SHORT detectado (sistema executaria LONG)")
    else:
        print(f"   ⚪ Critérios insuficientes: {criterios_short}/5")
    
    # Explicar diferenças
    print(f"\n💡 EXPLICAÇÃO DAS DIFERENÇAS:")
    if abs(atr_diff_pct) > 10:
        print(f"   • ATR muito diferente ({atr_diff_pct:+.0f}%) - possível:")
        print(f"     - Timeframes diferentes")
        print(f"     - Método de cálculo (Wilder vs SMA)")
        print(f"     - Dados de exchanges diferentes")
    
    if abs(vol_diff) > 0.5:
        print(f"   • Volume ratio diferente ({vol_diff:+.2f}x) - possível:")
        print(f"     - Janela de média móvel diferente")
        print(f"     - Dados de volumes diferentes entre APIs")
    
    print(f"\n✅ Teste concluído!")

if __name__ == "__main__":
    test_bnb_comparison()
