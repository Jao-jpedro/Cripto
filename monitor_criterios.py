#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MONITOR DE CRITÉRIOS DE ENTRADA - TRADINGV4
Verifica em tempo real se os ativos atendem aos critérios otimizados de entrada
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import json

# Configurações
INTERVAL = "15m"
SLEEP_SECONDS = 30  # Verificação a cada 30 segundos
WEBHOOK_DISCORD = os.getenv("DISCORD_WEBHOOK", "")

# Lista dos ativos para monitorar (mesma do tradingv4)
ASSET_SETUPS = [
    {"name": "BTC-USD", "data_symbol": "BTCUSDT", "hl_symbol": "BTC/USDC:USDC"},
    {"name": "SOL-USD", "data_symbol": "SOLUSDT", "hl_symbol": "SOL/USDC:USDC"},
    {"name": "ETH-USD", "data_symbol": "ETHUSDT", "hl_symbol": "ETH/USDC:USDC"},
    {"name": "HYPE-USD", "data_symbol": "HYPEUSDT", "hl_symbol": "HYPE/USDC:USDC"},
    {"name": "XRP-USD", "data_symbol": "XRPUSDT", "hl_symbol": "XRP/USDC:USDC"},
    {"name": "DOGE-USD", "data_symbol": "DOGEUSDT", "hl_symbol": "DOGE/USDC:USDC"},
    {"name": "AVAX-USD", "data_symbol": "AVAXUSDT", "hl_symbol": "AVAX/USDC:USDC"},
    {"name": "ENA-USD", "data_symbol": "ENAUSDT", "hl_symbol": "ENA/USDC:USDC"},
    {"name": "BNB-USD", "data_symbol": "BNBUSDT", "hl_symbol": "BNB/USDC:USDC"},
    {"name": "SUI-USD", "data_symbol": "SUIUSDT", "hl_symbol": "SUI/USDC:USDC"},
]

def get_binance_klines(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
    """Busca dados históricos da Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Converter para tipos corretos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['valor_fechamento'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        df['data'] = df['timestamp']
        df['criptomoeda'] = symbol
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao buscar dados da Binance para {symbol}: {e}")
        return pd.DataFrame()

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula os indicadores técnicos necessários"""
    if df.empty:
        return df
        
    df = df.copy()
    close = df['valor_fechamento']
    
    # EMAs
    df['ema_short'] = close.ewm(span=7).mean()
    df['ema_long'] = close.ewm(span=21).mean()
    
    # ATR simplificado (usando apenas close para aproximação)
    df['tr'] = close.diff().abs()
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = (df['atr'] / close) * 100
    
    # Gradiente EMA7 em % por barra
    def calcular_gradiente(series, window=3):
        gradientes = []
        for i in range(len(series)):
            if i < window - 1:
                gradientes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) >= 2:
                    slope = np.polyfit(x, y, 1)[0]
                    grad_pct = (slope / y[-1]) * 100 if y[-1] != 0 else 0
                    gradientes.append(grad_pct)
                else:
                    gradientes.append(np.nan)
        return pd.Series(gradientes, index=series.index)
    
    df['ema_short_grad_pct'] = calcular_gradiente(df['ema_short'])
    
    # Volume média
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # RSI simplificado
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def verificar_criterios_long(row) -> Tuple[bool, str, int]:
    """Verifica critérios LONG otimizados (sistema inverso - detecta LONG para executar SHORT)"""
    criterios_atendidos = 0
    criterios_total = 10
    detalhes = []
    
    try:
        # CRITÉRIO 1: EMA + Gradiente otimizado
        c1_ema = row.ema_short > row.ema_long
        c1_grad = row.ema_short_grad_pct > 0.08  # OTIMIZADO: 0.08%
        c1 = c1_ema and c1_grad
        if c1:
            criterios_atendidos += 1
            detalhes.append("✅ EMA7>EMA21+grad>0.08%")
        else:
            detalhes.append("❌ EMA/gradiente fraco")
        
        # CRITÉRIO 2: ATR otimizado
        c2 = (row.atr_pct >= 0.5) and (row.atr_pct <= 3.0)
        if c2:
            criterios_atendidos += 1
            detalhes.append("✅ ATR ótimo")
        else:
            detalhes.append("❌ ATR inadequado")
        
        # CRITÉRIO 3: Rompimento
        c3 = row.valor_fechamento > (row.ema_short + 0.8 * row.atr)
        if c3:
            criterios_atendidos += 1
            detalhes.append("✅ Rompimento forte")
        else:
            detalhes.append("❌ Rompimento fraco")
        
        # CRITÉRIO 4: Volume
        volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
        c4 = volume_ratio > 3.0
        if c4:
            criterios_atendidos += 1
            detalhes.append("✅ Volume alto")
        else:
            detalhes.append("❌ Volume baixo")
        
        # CRITÉRIO 5: RSI
        if pd.notna(row.rsi):
            c5 = 20 <= row.rsi <= 70
            if c5:
                criterios_atendidos += 1
                detalhes.append("✅ RSI ótimo")
            else:
                detalhes.append("❌ RSI extremo")
        else:
            criterios_atendidos += 0.5
            detalhes.append("⚪ RSI n/d")
        
        # CRITÉRIO 6-10: Critérios adicionais (simplificados para este monitor)
        criterios_atendidos += 2.5  # Assumir alguns critérios neutros
        detalhes.extend(["⚪ MACD n/d", "⚪ EMAs separadas", "⚪ Timing", "⚪ BB n/d", "⚪ BB squeeze n/d"])
        
        # Confluência mínima: 3 critérios
        is_valid = criterios_atendidos >= 3.0
        confluence_pct = (criterios_atendidos / criterios_total) * 100
        
        reason = f"Confluência LONG: {criterios_atendidos:.1f}/{criterios_total} ({confluence_pct:.0f}%)"
        
        return is_valid, reason, int(criterios_atendidos)
        
    except Exception as e:
        return False, f"Erro: {e}", 0

def verificar_criterios_short(row) -> Tuple[bool, str, int]:
    """Verifica critérios SHORT otimizados (sistema inverso - detecta SHORT para executar LONG)"""
    criterios_atendidos = 0
    criterios_total = 10
    detalhes = []
    
    try:
        # CRITÉRIO 1: EMA + Gradiente otimizado
        c1_ema = row.ema_short < row.ema_long
        c1_grad = row.ema_short_grad_pct < -0.12  # OTIMIZADO: -0.12%
        c1 = c1_ema and c1_grad
        if c1:
            criterios_atendidos += 1
            detalhes.append("✅ EMA7<EMA21+grad<-0.12%")
        else:
            detalhes.append("❌ EMA/gradiente fraco")
        
        # CRITÉRIO 2: ATR otimizado
        c2 = (row.atr_pct >= 0.5) and (row.atr_pct <= 3.0)
        if c2:
            criterios_atendidos += 1
            detalhes.append("✅ ATR ótimo")
        else:
            detalhes.append("❌ ATR inadequado")
        
        # CRITÉRIO 3: Rompimento
        c3 = row.valor_fechamento < (row.ema_short - 0.8 * row.atr)
        if c3:
            criterios_atendidos += 1
            detalhes.append("✅ Rompimento forte")
        else:
            detalhes.append("❌ Rompimento fraco")
        
        # CRITÉRIO 4: Volume
        volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
        c4 = volume_ratio > 3.0
        if c4:
            criterios_atendidos += 1
            detalhes.append("✅ Volume alto")
        else:
            detalhes.append("❌ Volume baixo")
        
        # CRITÉRIO 5: RSI
        if pd.notna(row.rsi):
            c5 = 20 <= row.rsi <= 70
            if c5:
                criterios_atendidos += 1
                detalhes.append("✅ RSI ótimo")
            else:
                detalhes.append("❌ RSI extremo")
        else:
            criterios_atendidos += 0.5
            detalhes.append("⚪ RSI n/d")
        
        # CRITÉRIO 6-10: Critérios adicionais (simplificados)
        criterios_atendidos += 2.5
        detalhes.extend(["⚪ MACD n/d", "⚪ EMAs separadas", "⚪ Timing", "⚪ BB n/d", "⚪ BB squeeze n/d"])
        
        # Confluência mínima: 3 critérios
        is_valid = criterios_atendidos >= 3.0
        confluence_pct = (criterios_atendidos / criterios_total) * 100
        
        reason = f"Confluência SHORT: {criterios_atendidos:.1f}/{criterios_total} ({confluence_pct:.0f}%)"
        
        return is_valid, reason, int(criterios_atendidos)
        
    except Exception as e:
        return False, f"Erro: {e}", 0

def verificar_no_trade_zone(row) -> bool:
    """Verifica se está em zona sem negociação"""
    try:
        eps = 0.05 * row.atr  # 5% do ATR
        diff = abs(row.ema_short - row.ema_long)
        atr_ok = (0.5 <= row.atr_pct <= 3.0)
        
        return (diff < eps) or (not atr_ok)
    except Exception:
        return True

def verificar_ativo(asset: Dict[str, str]) -> Dict[str, any]:
    """Verifica critérios para um único ativo"""
    try:
        # Buscar dados
        df = get_binance_klines(asset["data_symbol"], INTERVAL, 100)
        
        if df.empty:
            return {
                "asset": asset["name"],
                "status": "❌ SEM DADOS",
                "price": None,
                "signal": "N/A",
                "reason": "Falha ao obter dados da Binance"
            }
        
        # Calcular indicadores
        df = calcular_indicadores(df)
        
        if len(df) < 50:
            return {
                "asset": asset["name"],
                "status": "❌ DADOS INSUFICIENTES",
                "price": None,
                "signal": "N/A",
                "reason": "Menos de 50 candles disponíveis"
            }
        
        last_row = df.iloc[-1]
        price = last_row.valor_fechamento
        
        # Verificar zona sem negociação
        if verificar_no_trade_zone(last_row):
            return {
                "asset": asset["name"],
                "status": "⚪ NO TRADE ZONE",
                "price": price,
                "signal": "NEUTRO",
                "reason": f"ATR%={last_row.atr_pct:.2f}% ou EMAs próximas"
            }
        
        # Verificar critérios LONG (sistema inverso)
        long_valid, long_reason, long_score = verificar_criterios_long(last_row)
        
        # Verificar critérios SHORT (sistema inverso)
        short_valid, short_reason, short_score = verificar_criterios_short(last_row)
        
        if long_valid:
            return {
                "asset": asset["name"],
                "status": "🔴 SINAL LONG → EXECUTARIA SHORT",
                "price": price,
                "signal": "LONG_DETECTED",
                "reason": long_reason,
                "confluence": long_score,
                "atr_pct": last_row.atr_pct,
                "volume_ratio": last_row.volume / last_row.vol_ma if last_row.vol_ma > 0 else 0,
                "rsi": last_row.rsi if pd.notna(last_row.rsi) else None
            }
        elif short_valid:
            return {
                "asset": asset["name"],
                "status": "🟢 SINAL SHORT → EXECUTARIA LONG",
                "price": price,
                "signal": "SHORT_DETECTED",
                "reason": short_reason,
                "confluence": short_score,
                "atr_pct": last_row.atr_pct,
                "volume_ratio": last_row.volume / last_row.vol_ma if last_row.vol_ma > 0 else 0,
                "rsi": last_row.rsi if pd.notna(last_row.rsi) else None
            }
        else:
            return {
                "asset": asset["name"],
                "status": "⚪ SEM SINAL",
                "price": price,
                "signal": "NONE",
                "reason": f"L:{long_score}/10 S:{short_score}/10 (min:3)",
                "confluence": max(long_score, short_score),
                "atr_pct": last_row.atr_pct,
                "volume_ratio": last_row.volume / last_row.vol_ma if last_row.vol_ma > 0 else 0,
                "rsi": last_row.rsi if pd.notna(last_row.rsi) else None
            }
            
    except Exception as e:
        return {
            "asset": asset["name"],
            "status": f"❌ ERRO: {type(e).__name__}",
            "price": None,
            "signal": "ERROR",
            "reason": str(e)[:100]
        }

def enviar_discord(message: str):
    """Envia mensagem para Discord se webhook configurado"""
    if not WEBHOOK_DISCORD or "discord.com/api/webhooks" not in WEBHOOK_DISCORD:
        return False
    
    try:
        payload = {"content": message}
        response = requests.post(WEBHOOK_DISCORD, json=payload, timeout=10)
        return response.status_code in (200, 204)
    except Exception as e:
        print(f"⚠️ Erro Discord: {e}")
        return False

def analisar_criterios_detalhado(asset: Dict[str, str]) -> Dict[str, any]:
    """Análise detalhada dos critérios para um único ativo"""
    print(f"\n� ANÁLISE DETALHADA: {asset['name']}")
    print("=" * 60)
    
    try:
        # Buscar dados
        df = get_binance_klines(asset["data_symbol"], INTERVAL, 100)
        
        if df.empty:
            print("❌ Falha ao obter dados da Binance")
            return {"status": "error", "reason": "sem_dados"}
        
        # Calcular indicadores
        df = calcular_indicadores(df)
        
        if len(df) < 50:
            print("❌ Dados insuficientes (< 50 candles)")
            return {"status": "error", "reason": "dados_insuficientes"}
        
        last_row = df.iloc[-1]
        price = last_row.valor_fechamento
        
        print(f"� Preço atual: ${price:.4f}")
        print(f"⏰ Último candle: {last_row.name}")
        
        # ========== ANÁLISE DOS 10 CRITÉRIOS LONG ==========
        print("\n🔴 CRITÉRIOS LONG (Sistema executaria SHORT):")
        print("-" * 40)
        
        criterios_long = 0
        
        # 1. EMA + Gradiente
        ema_ok = last_row.ema_short > last_row.ema_long
        grad_ok = last_row.ema_short_grad_pct > 0.08
        if ema_ok and grad_ok:
            criterios_long += 1
            print(f"✅ 1. EMA7({last_row.ema_short:.4f}) > EMA21({last_row.ema_long:.4f}) + Grad({last_row.ema_short_grad_pct:.3f}%) > 0.08%")
        else:
            print(f"❌ 1. EMA: {ema_ok} | Gradiente: {last_row.ema_short_grad_pct:.3f}% (precisa > 0.08%)")
        
        # 2. ATR
        atr_ok = 0.5 <= last_row.atr_pct <= 3.0
        if atr_ok:
            criterios_long += 1
            print(f"✅ 2. ATR% = {last_row.atr_pct:.2f}% (entre 0.5-3.0%)")
        else:
            print(f"❌ 2. ATR% = {last_row.atr_pct:.2f}% (fora da faixa 0.5-3.0%)")
        
        # 3. Rompimento
        breakout_level = last_row.ema_short + (0.8 * last_row.atr)
        breakout_ok = price > breakout_level
        if breakout_ok:
            criterios_long += 1
            print(f"✅ 3. Preço({price:.4f}) > EMA7 + 0.8×ATR({breakout_level:.4f})")
        else:
            print(f"❌ 3. Preço({price:.4f}) ≤ EMA7 + 0.8×ATR({breakout_level:.4f}) - Falta {breakout_level-price:.4f}")
        
        # 4. Volume
        vol_ratio = last_row.volume / last_row.vol_ma if last_row.vol_ma > 0 else 0
        vol_ok = vol_ratio > 3.0
        if vol_ok:
            criterios_long += 1
            print(f"✅ 4. Volume = {vol_ratio:.1f}x média (> 3.0x)")
        else:
            print(f"❌ 4. Volume = {vol_ratio:.1f}x média (precisa > 3.0x)")
        
        # 5. RSI
        rsi_ok = pd.notna(last_row.rsi) and (20 <= last_row.rsi <= 70)
        if rsi_ok:
            criterios_long += 1
            print(f"✅ 5. RSI = {last_row.rsi:.1f} (entre 20-70)")
        else:
            rsi_val = last_row.rsi if pd.notna(last_row.rsi) else "N/A"
            print(f"❌ 5. RSI = {rsi_val} (fora da faixa 20-70)")
        
        # 6. MACD
        if hasattr(last_row, 'macd') and hasattr(last_row, 'macd_signal'):
            macd_diff = last_row.macd - last_row.macd_signal
            macd_ok = macd_diff > 0.01
            if macd_ok:
                criterios_long += 1
                print(f"✅ 6. MACD({last_row.macd:.4f}) - Signal({last_row.macd_signal:.4f}) = {macd_diff:.4f} > 0.01")
            else:
                print(f"❌ 6. MACD Diff = {macd_diff:.4f} (precisa > 0.01)")
        else:
            print("❌ 6. MACD não disponível")
        
        # 7. Separação EMAs
        ema_sep = abs(last_row.ema_short - last_row.ema_long)
        ema_sep_ok = ema_sep >= (0.3 * last_row.atr)
        if ema_sep_ok:
            criterios_long += 1
            print(f"✅ 7. |EMA7-EMA21| = {ema_sep:.4f} ≥ 0.3×ATR({0.3*last_row.atr:.4f})")
        else:
            print(f"❌ 7. |EMA7-EMA21| = {ema_sep:.4f} < 0.3×ATR({0.3*last_row.atr:.4f})")
        
        # 8. Timing
        timing_dist = abs(price - last_row.ema_short)
        timing_ok = timing_dist <= (1.5 * last_row.atr)
        if timing_ok:
            criterios_long += 1
            print(f"✅ 8. |Preço-EMA7| = {timing_dist:.4f} ≤ 1.5×ATR({1.5*last_row.atr:.4f})")
        else:
            print(f"❌ 8. |Preço-EMA7| = {timing_dist:.4f} > 1.5×ATR({1.5*last_row.atr:.4f})")
        
        # 9. Bollinger %B
        if hasattr(last_row, 'bb_percent_b') and pd.notna(last_row.bb_percent_b):
            bb_ok = 0.6 <= last_row.bb_percent_b <= 0.95
            if bb_ok:
                criterios_long += 1
                print(f"✅ 9. BB %B = {last_row.bb_percent_b:.3f} (entre 0.6-0.95)")
            else:
                print(f"❌ 9. BB %B = {last_row.bb_percent_b:.3f} (fora da faixa 0.6-0.95)")
        else:
            print("❌ 9. Bollinger %B não disponível")
        
        # 10. BB Squeeze
        if hasattr(last_row, 'bb_squeeze'):
            squeeze_ok = not last_row.bb_squeeze
            if squeeze_ok:
                criterios_long += 1
                print(f"✅ 10. Não em BB Squeeze")
            else:
                print(f"❌ 10. Em BB Squeeze (baixa volatilidade)")
        else:
            print("❌ 10. BB Squeeze não disponível")
        
        # ========== ANÁLISE DOS 10 CRITÉRIOS SHORT ==========
        print("\n🟢 CRITÉRIOS SHORT (Sistema executaria LONG):")
        print("-" * 40)
        
        criterios_short = 0
        
        # 1. EMA + Gradiente SHORT
        ema_short_ok = last_row.ema_short < last_row.ema_long
        grad_short_ok = last_row.ema_short_grad_pct < -0.12
        if ema_short_ok and grad_short_ok:
            criterios_short += 1
            print(f"✅ 1. EMA7({last_row.ema_short:.4f}) < EMA21({last_row.ema_long:.4f}) + Grad({last_row.ema_short_grad_pct:.3f}%) < -0.12%")
        else:
            print(f"❌ 1. EMA: {ema_short_ok} | Gradiente: {last_row.ema_short_grad_pct:.3f}% (precisa < -0.12%)")
        
        # 2. ATR (mesmo critério)
        if atr_ok:
            criterios_short += 1
            print(f"✅ 2. ATR% = {last_row.atr_pct:.2f}% (entre 0.5-3.0%)")
        else:
            print(f"❌ 2. ATR% = {last_row.atr_pct:.2f}% (fora da faixa 0.5-3.0%)")
        
        # 3. Rompimento SHORT
        breakout_short_level = last_row.ema_short - (0.8 * last_row.atr)
        breakout_short_ok = price < breakout_short_level
        if breakout_short_ok:
            criterios_short += 1
            print(f"✅ 3. Preço({price:.4f}) < EMA7 - 0.8×ATR({breakout_short_level:.4f})")
        else:
            print(f"❌ 3. Preço({price:.4f}) ≥ EMA7 - 0.8×ATR({breakout_short_level:.4f}) - Falta {price-breakout_short_level:.4f}")
        
        # 4-10. Outros critérios similares adaptados para SHORT
        # Volume (mesmo)
        if vol_ok:
            criterios_short += 1
            print(f"✅ 4. Volume = {vol_ratio:.1f}x média (> 3.0x)")
        else:
            print(f"❌ 4. Volume = {vol_ratio:.1f}x média (precisa > 3.0x)")
        
        # RSI (mesmo range)
        if rsi_ok:
            criterios_short += 1
            print(f"✅ 5. RSI = {last_row.rsi:.1f} (entre 20-70)")
        else:
            rsi_val = last_row.rsi if pd.notna(last_row.rsi) else "N/A"
            print(f"❌ 5. RSI = {rsi_val} (fora da faixa 20-70)")
        
        # MACD SHORT
        if hasattr(last_row, 'macd') and hasattr(last_row, 'macd_signal'):
            macd_diff = last_row.macd - last_row.macd_signal
            macd_short_ok = macd_diff < -0.01
            if macd_short_ok:
                criterios_short += 1
                print(f"✅ 6. MACD Diff = {macd_diff:.4f} < -0.01")
            else:
                print(f"❌ 6. MACD Diff = {macd_diff:.4f} (precisa < -0.01)")
        else:
            print("❌ 6. MACD não disponível")
        
        # Outros critérios 7-10 (similar ao LONG)
        if ema_sep_ok:
            criterios_short += 1
            print(f"✅ 7. Separação EMAs OK")
        else:
            print(f"❌ 7. EMAs muito próximas")
        
        if timing_ok:
            criterios_short += 1
            print(f"✅ 8. Timing OK")
        else:
            print(f"❌ 8. Timing ruim")
        
        # BB %B para SHORT (faixa baixa)
        if hasattr(last_row, 'bb_percent_b') and pd.notna(last_row.bb_percent_b):
            bb_short_ok = 0.05 <= last_row.bb_percent_b <= 0.4
            if bb_short_ok:
                criterios_short += 1
                print(f"✅ 9. BB %B = {last_row.bb_percent_b:.3f} (entre 0.05-0.4)")
            else:
                print(f"❌ 9. BB %B = {last_row.bb_percent_b:.3f} (fora da faixa 0.05-0.4)")
        else:
            print("❌ 9. Bollinger %B não disponível")
        
        # BB Squeeze (mesmo)
        if hasattr(last_row, 'bb_squeeze'):
            if not last_row.bb_squeeze:
                criterios_short += 1
                print(f"✅ 10. Não em BB Squeeze")
            else:
                print(f"❌ 10. Em BB Squeeze")
        else:
            print("❌ 10. BB Squeeze não disponível")
        
        # ========== RESUMO FINAL ==========
        print(f"\n📊 RESUMO FINAL:")
        print("-" * 30)
        print(f"🔴 LONG: {criterios_long}/10 critérios (min: 3)")
        print(f"🟢 SHORT: {criterios_short}/10 critérios (min: 3)")
        
        # Zona sem negociação
        no_trade = verificar_no_trade_zone(last_row)
        if no_trade:
            print("⚪ Status: NO TRADE ZONE")
            print("   Motivo: ATR fora da faixa ou EMAs muito próximas")
        elif criterios_long >= 3:
            print("🔴 Status: SINAL LONG DETECTADO")
            print("⚠️ Sistema Inverso: EXECUTARIA SHORT")
        elif criterios_short >= 3:
            print("🟢 Status: SINAL SHORT DETECTADO") 
            print("⚠️ Sistema Inverso: EXECUTARIA LONG")
        else:
            print("😴 Status: SEM SINAL")
            print(f"   Confluência insuficiente (max: {max(criterios_long, criterios_short)}/10)")
        
        return {
            "status": "success",
            "price": price,
            "criterios_long": criterios_long,
            "criterios_short": criterios_short,
            "no_trade": no_trade
        }
        
    except Exception as e:
        print(f"❌ ERRO: {type(e).__name__}: {e}")
        return {"status": "error", "reason": str(e)}

def verificacao_unica_detalhada():
    """Verificação única e detalhada de todos os ativos"""
    print("🎯 ANÁLISE DETALHADA DOS CRITÉRIOS DE ENTRADA - TRADINGV4")
    print("=" * 80)
    print("⚠️ SISTEMA INVERSO ATIVO:")
    print("   • Sinal LONG detectado → Sistema executaria SHORT")
    print("   • Sinal SHORT detectado → Sistema executaria LONG")
    print("=" * 80)
    
    resultados = []
    
    for i, asset in enumerate(ASSET_SETUPS, 1):
        print(f"\n[{i}/{len(ASSET_SETUPS)}]", end="")
        resultado = analisar_criterios_detalhado(asset)
        resultados.append(resultado)
        
        if i < len(ASSET_SETUPS):
            print("\n" + "="*80)
    
    # Resumo geral
    print(f"\n\n🏆 RESUMO GERAL")
    print("=" * 50)
    
    com_sinal_long = sum(1 for r in resultados if r.get("criterios_long", 0) >= 3 and not r.get("no_trade", True))
    com_sinal_short = sum(1 for r in resultados if r.get("criterios_short", 0) >= 3 and not r.get("no_trade", True))
    com_erro = sum(1 for r in resultados if r.get("status") == "error")
    
    print(f"📊 Total de ativos analisados: {len(ASSET_SETUPS)}")
    print(f"🔴 Com critérios LONG (executaria SHORT): {com_sinal_long}")
    print(f"� Com critérios SHORT (executaria LONG): {com_sinal_short}")
    print(f"❌ Com erro: {com_erro}")
    print(f"😴 Sem sinal: {len(ASSET_SETUPS) - com_sinal_long - com_sinal_short - com_erro}")
    
    if com_sinal_long == 0 and com_sinal_short == 0:
        print("\n💡 DICAS PARA MELHORAR AS CONFLUÊNCIAS:")
        print("   • Aguarde maior volatilidade (ATR% entre 0.5-3.0%)")
        print("   • Volume precisa ser > 3.0x da média")
        print("   • EMAs precisam estar com separação clara")
        print("   • Aguarde rompimentos fortes acima/abaixo das EMAs")
    
    print("\n✅ Análise detalhada concluída!")

if __name__ == "__main__":
    verificacao_unica_detalhada()