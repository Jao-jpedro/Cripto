#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ANÁLISE COMPARATIVA: DADOS REAIS vs CRITÉRIOS DE ENTRADA
============================================================
Compara os valores atuais dos ativos com os critérios necessários
para identificar exatamente o que está impedindo as entradas.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from typing import Dict, List, Tuple

# Critérios otimizados do tradingv4
CRITERIOS_LONG = {
    "ema_trending": {"descricao": "EMA7 > EMA21", "necessario": True},
    "gradient_long": {"descricao": "Gradiente EMA7 > 0.08%", "necessario": 0.08},
    "atr_range": {"descricao": "ATR% entre 0.5% - 3.0%", "min": 0.5, "max": 3.0},
    "breakout_long": {"descricao": "Preço > EMA7 + 0.8×ATR", "multiplicador": 0.8},
    "volume_spike": {"descricao": "Volume > 3.0x média", "necessario": 3.0},
    "rsi_range": {"descricao": "RSI entre 20-70", "min": 20, "max": 70},
    "macd_positive": {"descricao": "MACD > Signal + 0.01", "necessario": 0.01},
    "ema_separation": {"descricao": "|EMA7-EMA21| ≥ 0.3×ATR", "multiplicador": 0.3},
    "timing_entry": {"descricao": "|Preço-EMA7| ≤ 1.5×ATR", "multiplicador": 1.5},
    "bb_position": {"descricao": "BB %B entre 0.6-0.95", "min": 0.6, "max": 0.95}
}

CRITERIOS_SHORT = {
    "ema_trending": {"descricao": "EMA7 < EMA21", "necessario": True},
    "gradient_short": {"descricao": "Gradiente EMA7 < -0.12%", "necessario": -0.12},
    "atr_range": {"descricao": "ATR% entre 0.5% - 3.0%", "min": 0.5, "max": 3.0},
    "breakout_short": {"descricao": "Preço < EMA7 - 0.8×ATR", "multiplicador": 0.8},
    "volume_spike": {"descricao": "Volume > 3.0x média", "necessario": 3.0},
    "rsi_range": {"descricao": "RSI entre 20-70", "min": 20, "max": 70},
    "macd_negative": {"descricao": "MACD < Signal - 0.01", "necessario": -0.01},
    "ema_separation": {"descricao": "|EMA7-EMA21| ≥ 0.3×ATR", "multiplicador": 0.3},
    "timing_entry": {"descricao": "|Preço-EMA7| ≤ 1.5×ATR", "multiplicador": 1.5},
    "bb_position": {"descricao": "BB %B entre 0.05-0.4", "min": 0.05, "max": 0.4}
}

# Lista dos ativos
ASSET_SETUPS = [
    {"name": "BTC-USD", "data_symbol": "BTCUSDT"},
    {"name": "SOL-USD", "data_symbol": "SOLUSDT"},
    {"name": "ETH-USD", "data_symbol": "ETHUSDT"},
    {"name": "XRP-USD", "data_symbol": "XRPUSDT"},
    {"name": "DOGE-USD", "data_symbol": "DOGEUSDT"},
    {"name": "AVAX-USD", "data_symbol": "AVAXUSDT"},
    {"name": "ENA-USD", "data_symbol": "ENAUSDT"},
    {"name": "BNB-USD", "data_symbol": "BNBUSDT"},
    {"name": "SUI-USD", "data_symbol": "SUIUSDT"},
]

def get_binance_klines(symbol: str, interval: str = "15m", limit: int = 100) -> pd.DataFrame:
    """Busca dados da Binance"""
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
        df['valor_fechamento'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        return df.set_index('timestamp')
        
    except Exception as e:
        print(f"❌ Erro ao buscar dados da Binance para {symbol}: {e}")
        return pd.DataFrame()

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os indicadores necessários"""
    if df.empty:
        return df
    
    # EMAs
    df['ema_short'] = df['valor_fechamento'].ewm(span=7).mean()
    df['ema_long'] = df['valor_fechamento'].ewm(span=21).mean()
    
    # Gradiente EMA7
    df['ema_short_grad_pct'] = df['ema_short'].pct_change(periods=3) * 100
    
    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['valor_fechamento'].shift(1)),
            abs(df['low'] - df['valor_fechamento'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    # RSI
    delta = df['valor_fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

def formatar_status(condicao: bool) -> str:
    """Formata status visual"""
    return "✅" if condicao else "❌"

def analisar_ativo_detalhado(asset: Dict[str, str]) -> Dict:
    """Análise detalhada de um ativo"""
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISE COMPARATIVA: {asset['name']}")
    print(f"{'='*80}")
    
    try:
        # Buscar dados
        df = get_binance_klines(asset["data_symbol"], "15m", 100)
        
        if df.empty:
            print("❌ Falha ao obter dados da Binance")
            return {"erro": "sem_dados"}
        
        # Calcular indicadores
        df = calcular_indicadores(df)
        
        if len(df) < 50:
            print("❌ Dados insuficientes")
            return {"erro": "dados_insuficientes"}
        
        last_row = df.iloc[-1]
        price = last_row.valor_fechamento
        
        print(f"💰 Preço atual: ${price:.4f}")
        print(f"⏰ Última atualização: {last_row.name.strftime('%H:%M:%S UTC')}")
        
        # ========== TABELA COMPARATIVA LONG ==========
        print(f"\n🔴 CRITÉRIOS LONG → EXECUTARIA SHORT")
        print(f"{'─'*80}")
        print(f"{'CRITÉRIO':<35} {'VALOR ATUAL':<20} {'NECESSÁRIO':<20} {'STATUS':<5}")
        print(f"{'─'*80}")
        
        criterios_long_ok = 0
        
        # 1. EMA Trending
        ema_trend_ok = last_row.ema_short > last_row.ema_long
        criterios_long_ok += ema_trend_ok
        print(f"{'1. EMA7 > EMA21':<35} {f'{last_row.ema_short:.4f} > {last_row.ema_long:.4f}':<20} {'TRUE':<20} {formatar_status(ema_trend_ok)}")
        
        # 2. Gradiente
        grad_atual = last_row.ema_short_grad_pct if pd.notna(last_row.ema_short_grad_pct) else 0
        grad_ok = grad_atual > 0.08
        criterios_long_ok += grad_ok
        print(f"{'2. Gradiente EMA7':<35} {f'{grad_atual:.3f}%':<20} {'> 0.08%':<20} {formatar_status(grad_ok)}")
        
        # 3. ATR Range
        atr_ok = 0.5 <= last_row.atr_pct <= 3.0
        criterios_long_ok += atr_ok
        print(f"{'3. ATR%':<35} {f'{last_row.atr_pct:.2f}%':<20} {'0.5% - 3.0%':<20} {formatar_status(atr_ok)}")
        
        # 4. Breakout
        breakout_level = last_row.ema_short + (0.8 * last_row.atr)
        breakout_ok = price > breakout_level
        criterios_long_ok += breakout_ok
        gap_breakout = breakout_level - price if not breakout_ok else 0
        print(f"{'4. Rompimento LONG':<35} {f'{price:.4f}':<20} {f'> {breakout_level:.4f}':<20} {formatar_status(breakout_ok)}")
        if not breakout_ok:
            print(f"{'   └─ Falta para romper:':<35} {f'{gap_breakout:.4f}':<20} {'({gap_breakout/price*100:.2f}%)':<20}")
        
        # 5. Volume
        vol_ratio = last_row.volume / last_row.vol_ma if last_row.vol_ma > 0 else 0
        vol_ok = vol_ratio > 3.0
        criterios_long_ok += vol_ok
        gap_volume = 3.0 - vol_ratio if not vol_ok else 0
        print(f"{'5. Volume Spike':<35} {f'{vol_ratio:.1f}x':<20} {'> 3.0x':<20} {formatar_status(vol_ok)}")
        if not vol_ok:
            print(f"{'   └─ Falta para 3.0x:':<35} {f'{gap_volume:.1f}x':<20} {'({gap_volume/3.0*100:.0f}% mais)':<20}")
        
        # 6. RSI
        rsi_val = last_row.rsi if pd.notna(last_row.rsi) else 0
        rsi_ok = 20 <= rsi_val <= 70
        criterios_long_ok += rsi_ok
        print(f"{'6. RSI Range':<35} {f'{rsi_val:.1f}':<20} {'20 - 70':<20} {formatar_status(rsi_ok)}")
        if rsi_val < 20:
            print(f"{'   └─ Muito baixo:':<35} {'Oversold extremo':<20} {f'Falta {20-rsi_val:.1f} pts':<20}")
        elif rsi_val > 70:
            print(f"{'   └─ Muito alto:':<35} {'Overbought':<20} {f'Sobra {rsi_val-70:.1f} pts':<20}")
        
        # 7. MACD (se disponível)
        macd_ok = False
        if hasattr(last_row, 'macd') and hasattr(last_row, 'macd_signal'):
            if pd.notna(last_row.macd) and pd.notna(last_row.macd_signal):
                macd_diff = last_row.macd - last_row.macd_signal
                macd_ok = macd_diff > 0.01
                criterios_long_ok += macd_ok
                print(f"{'7. MACD Momentum':<35} {f'{macd_diff:.4f}':<20} {'> 0.01':<20} {formatar_status(macd_ok)}")
            else:
                print(f"{'7. MACD Momentum':<35} {'N/A':<20} {'> 0.01':<20} {'❓'}")
        else:
            print(f"{'7. MACD Momentum':<35} {'Não calculado':<20} {'> 0.01':<20} {'❓'}")
        
        # 8. Separação EMAs
        ema_sep = abs(last_row.ema_short - last_row.ema_long)
        ema_sep_min = 0.3 * last_row.atr
        ema_sep_ok = ema_sep >= ema_sep_min
        criterios_long_ok += ema_sep_ok
        print(f"{'8. Separação EMAs':<35} {f'{ema_sep:.4f}':<20} {f'>= {ema_sep_min:.4f}':<20} {formatar_status(ema_sep_ok)}")
        
        # 9. Timing
        timing_dist = abs(price - last_row.ema_short)
        timing_max = 1.5 * last_row.atr
        timing_ok = timing_dist <= timing_max
        criterios_long_ok += timing_ok
        print(f"{'9. Timing Entry':<35} {f'{timing_dist:.4f}':<20} {f'<= {timing_max:.4f}':<20} {formatar_status(timing_ok)}")
        
        # 10. Bollinger %B (se disponível)
        bb_ok = False
        print(f"{'10. Bollinger %B':<35} {'Não calculado':<20} {'0.6 - 0.95':<20} {'❓'}")
        
        print(f"{'─'*80}")
        print(f"{'TOTAL LONG:':<35} {f'{criterios_long_ok}/10':<20} {'Mín: 3/10':<20} {formatar_status(criterios_long_ok >= 3)}")
        
        # ========== TABELA COMPARATIVA SHORT ==========
        print(f"\n🟢 CRITÉRIOS SHORT → EXECUTARIA LONG")
        print(f"{'─'*80}")
        print(f"{'CRITÉRIO':<35} {'VALOR ATUAL':<20} {'NECESSÁRIO':<20} {'STATUS':<5}")
        print(f"{'─'*80}")
        
        criterios_short_ok = 0
        
        # 1. EMA Trending SHORT
        ema_trend_short_ok = last_row.ema_short < last_row.ema_long
        criterios_short_ok += ema_trend_short_ok
        print(f"{'1. EMA7 < EMA21':<35} {f'{last_row.ema_short:.4f} < {last_row.ema_long:.4f}':<20} {'TRUE':<20} {formatar_status(ema_trend_short_ok)}")
        
        # 2. Gradiente SHORT
        grad_short_ok = grad_atual < -0.12
        criterios_short_ok += grad_short_ok
        print(f"{'2. Gradiente EMA7':<35} {f'{grad_atual:.3f}%':<20} {'< -0.12%':<20} {formatar_status(grad_short_ok)}")
        
        # 3. ATR Range (mesmo)
        criterios_short_ok += atr_ok
        print(f"{'3. ATR%':<35} {f'{last_row.atr_pct:.2f}%':<20} {'0.5% - 3.0%':<20} {formatar_status(atr_ok)}")
        
        # 4. Breakout SHORT
        breakout_short_level = last_row.ema_short - (0.8 * last_row.atr)
        breakout_short_ok = price < breakout_short_level
        criterios_short_ok += breakout_short_ok
        gap_short = price - breakout_short_level if not breakout_short_ok else 0
        print(f"{'4. Rompimento SHORT':<35} {f'{price:.4f}':<20} {f'< {breakout_short_level:.4f}':<20} {formatar_status(breakout_short_ok)}")
        if not breakout_short_ok:
            print(f"{'   └─ Falta para romper:':<35} {f'{gap_short:.4f}':<20} {'({gap_short/price*100:.2f}%)':<20}")
        
        # 5-10. Outros critérios similares
        criterios_short_ok += vol_ok
        print(f"{'5. Volume Spike':<35} {f'{vol_ratio:.1f}x':<20} {'> 3.0x':<20} {formatar_status(vol_ok)}")
        
        criterios_short_ok += rsi_ok
        print(f"{'6. RSI Range':<35} {f'{rsi_val:.1f}':<20} {'20 - 70':<20} {formatar_status(rsi_ok)}")
        
        # MACD SHORT
        macd_short_ok = False
        if hasattr(last_row, 'macd') and hasattr(last_row, 'macd_signal'):
            if pd.notna(last_row.macd) and pd.notna(last_row.macd_signal):
                macd_diff = last_row.macd - last_row.macd_signal
                macd_short_ok = macd_diff < -0.01
                criterios_short_ok += macd_short_ok
                print(f"{'7. MACD Momentum':<35} {f'{macd_diff:.4f}':<20} {'< -0.01':<20} {formatar_status(macd_short_ok)}")
            else:
                print(f"{'7. MACD Momentum':<35} {'N/A':<20} {'< -0.01':<20} {'❓'}")
        else:
            print(f"{'7. MACD Momentum':<35} {'Não calculado':<20} {'< -0.01':<20} {'❓'}")
        
        criterios_short_ok += ema_sep_ok
        print(f"{'8. Separação EMAs':<35} {f'{ema_sep:.4f}':<20} {f'>= {ema_sep_min:.4f}':<20} {formatar_status(ema_sep_ok)}")
        
        criterios_short_ok += timing_ok
        print(f"{'9. Timing Entry':<35} {f'{timing_dist:.4f}':<20} {f'<= {timing_max:.4f}':<20} {formatar_status(timing_ok)}")
        
        print(f"{'10. Bollinger %B':<35} {'Não calculado':<20} {'0.05 - 0.4':<20} {'❓'}")
        
        print(f"{'─'*80}")
        print(f"{'TOTAL SHORT:':<35} {f'{criterios_short_ok}/10':<20} {'Mín: 3/10':<20} {formatar_status(criterios_short_ok >= 3)}")
        
        # ========== RESUMO E ANÁLISE ==========
        print(f"\n📊 RESUMO FINAL:")
        print(f"{'─'*50}")
        
        # No-trade zone
        no_trade = (last_row.atr_pct < 0.5 or last_row.atr_pct > 3.0) or (ema_sep < 0.05 * last_row.atr)
        
        if no_trade:
            print("⚪ STATUS: NO TRADE ZONE")
            print("   MOTIVO: ATR fora da faixa ou EMAs muito próximas")
        elif criterios_long_ok >= 3:
            print("🔴 STATUS: SINAL LONG DETECTADO")
            print("⚠️  SISTEMA INVERSO: EXECUTARIA SHORT")
        elif criterios_short_ok >= 3:
            print("🟢 STATUS: SINAL SHORT DETECTADO") 
            print("⚠️  SISTEMA INVERSO: EXECUTARIA LONG")
        else:
            print("😴 STATUS: SEM SINAL")
        
        print(f"\n🎯 ANÁLISE DOS BLOQUEADORES:")
        print(f"   • ATR muito baixo: {last_row.atr_pct:.2f}% (precisa 0.5-3.0%)")
        print(f"   • Volume insuficiente: {vol_ratio:.1f}x (precisa > 3.0x)")
        if rsi_val < 20 or rsi_val > 70:
            print(f"   • RSI em extremo: {rsi_val:.1f} (precisa 20-70)")
        print(f"   • MACD/BB não calculados: Perda de 3 critérios potenciais")
        
        # Distâncias para próximas confluências
        print(f"\n🎯 DISTÂNCIAS PARA CRITÉRIOS:")
        if not atr_ok:
            falta_atr = ((0.5 - last_row.atr_pct) / last_row.atr_pct * 100) if last_row.atr_pct < 0.5 else 0
            print(f"   • ATR precisa subir {falta_atr:.0f}% para atingir 0.5%")
        
        if not vol_ok:
            falta_vol = ((3.0 - vol_ratio) / vol_ratio * 100) if vol_ratio > 0 else 300
            print(f"   • Volume precisa subir {falta_vol:.0f}% para atingir 3.0x")
        
        if not breakout_ok:
            print(f"   • Preço precisa subir {gap_breakout/price*100:.2f}% para rompimento LONG")
        
        if not breakout_short_ok:
            print(f"   • Preço precisa cair {gap_short/price*100:.2f}% para rompimento SHORT")
        
        return {
            "asset": asset["name"],
            "price": price,
            "criterios_long": criterios_long_ok,
            "criterios_short": criterios_short_ok,
            "atr_pct": last_row.atr_pct,
            "volume_ratio": vol_ratio,
            "rsi": rsi_val,
            "no_trade": no_trade
        }
        
    except Exception as e:
        print(f"❌ ERRO: {type(e).__name__}: {e}")
        return {"erro": str(e)}

def main():
    """Análise comparativa de todos os ativos"""
    print("📊 ANÁLISE COMPARATIVA: DADOS REAIS vs CRITÉRIOS DE ENTRADA")
    print("=" * 80)
    print("🎯 Sistema Otimizado tradingv4 | Confluência mínima: 3/10 critérios")
    print("⚠️  SISTEMA INVERSO: Long detectado → Executa SHORT | Short detectado → Executa LONG")
    
    resultados = []
    
    for i, asset in enumerate(ASSET_SETUPS, 1):
        print(f"\n[{i}/{len(ASSET_SETUPS)}] Analisando {asset['name']}...")
        resultado = analisar_ativo_detalhado(asset)
        if "erro" not in resultado:
            resultados.append(resultado)
    
    # Resumo geral comparativo
    print(f"\n\n🏆 RESUMO COMPARATIVO GERAL")
    print("=" * 100)
    print(f"{'ATIVO':<12} {'PREÇO':<12} {'ATR%':<8} {'VOL':<8} {'RSI':<8} {'L':<3} {'S':<3} {'STATUS':<15}")
    print("=" * 100)
    
    for r in resultados:
        status = "NO TRADE"
        if r["criterios_long"] >= 3 and not r["no_trade"]:
            status = "LONG → SHORT"
        elif r["criterios_short"] >= 3 and not r["no_trade"]:
            status = "SHORT → LONG"
        elif r["criterios_long"] >= 2 or r["criterios_short"] >= 2:
            status = "QUASE"
        
        print(f"{r['asset']:<12} ${r['price']:<11.4f} {r['atr_pct']:<7.2f} {r['volume_ratio']:<7.1f} {r['rsi']:<7.1f} {r['criterios_long']:<3} {r['criterios_short']:<3} {status:<15}")
    
    print("=" * 100)
    print("📍 L = Critérios LONG | S = Critérios SHORT | Mínimo para entrada: 3")
    
    # Análise de mercado geral
    avg_atr = np.mean([r["atr_pct"] for r in resultados])
    avg_vol = np.mean([r["volume_ratio"] for r in resultados])
    avg_rsi = np.mean([r["rsi"] for r in resultados])
    
    print(f"\n📊 MÉDIAS DE MERCADO:")
    print(f"   • ATR médio: {avg_atr:.2f}% (ideal: 0.5-3.0%)")
    print(f"   • Volume médio: {avg_vol:.1f}x (ideal: >3.0x)")
    print(f"   • RSI médio: {avg_rsi:.1f} (ideal: 20-70)")
    
    print(f"\n💡 CONCLUSÃO:")
    if avg_atr < 0.5:
        print("   🔹 Mercado em baixa volatilidade - Aguardar breakouts")
    if avg_vol < 3.0:
        print("   🔹 Volume insuficiente - Aguardar eventos de alta liquidez")
    if avg_rsi < 30:
        print("   🔹 Mercado oversold - Possível reversão em breve")
    elif avg_rsi > 70:
        print("   🔹 Mercado overbought - Possível correção em breve")
    
    print("\n✅ Análise comparativa concluída!")

if __name__ == "__main__":
    main()
