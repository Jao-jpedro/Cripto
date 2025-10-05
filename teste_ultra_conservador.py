#!/usr/bin/env python3
"""
Estratégia Ultra-Conservadora para Capital Preservation
Foco em não perder dinheiro ao invés de maximizar ganhos
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

def test_ultra_conservative():
    """Testa estratégia ultra-conservadora"""
    
    print("🛡️ ESTRATÉGIA ULTRA-CONSERVADORA - CAPITAL PRESERVATION")
    print("🎯 Objetivo: NÃO PERDER DINHEIRO")
    print("="*70)
    
    # Configuração ULTRA-conservadora
    config = {
        'tp_pct': 5.0,       # TP muito pequeno: 5%
        'sl_pct': 2.0,       # SL muito pequeno: 2%
        'leverage': 3,       # Leverage muito baixo: 3x
        'atr_min': 0.3,      # ATR muito permissivo
        'atr_max': 1.0,      # ATR baixo (sem volatilidade extrema)
        'volume_mult': 1.2,  # Volume quase qualquer coisa
        'min_confluencia': 1, # Apenas 1 critério
        'ema_short': 5,      # EMAs mais rápidas
        'ema_long': 15,      # EMAs mais rápidas
        'breakout_k': 0.3,   # Breakout muito pequeno
    }
    
    print("📋 CONFIGURAÇÃO ULTRA-CONSERVADORA:")
    print(f"   TP/SL: {config['tp_pct']}%/{config['sl_pct']}% (R:R = {config['tp_pct']/config['sl_pct']:.1f}:1)")
    print(f"   Leverage: {config['leverage']}x (muito baixo)")
    print(f"   ATR Range: {config['atr_min']}-{config['atr_max']}% (baixa volatilidade)")
    print(f"   Volume: {config['volume_mult']}x média (muito permissivo)")
    print(f"   Confluência mín: {config['min_confluencia']} (máxima permissividade)")
    print(f"   EMAs: {config['ema_short']}/{config['ema_long']} (mais rápidas)")
    print()
    
    # Testar BTC primeiro
    btc_result = test_asset_ultra_conservative("BTC", "dados_reais_btc_1ano.csv", config)
    
    if btc_result and btc_result['total_return_pct'] > -10:
        print("✅ BTC teve resultado aceitável!")
        
        # Testar outros
        assets = [
            ("ETH", "dados_reais_eth_1ano.csv"),
            ("BNB", "dados_reais_bnb_1ano.csv"),
        ]
        
        results = [btc_result]
        for asset_name, filename in assets:
            if os.path.exists(filename):
                result = test_asset_ultra_conservative(asset_name, filename, config)
                if result:
                    results.append(result)
        
        analyze_ultra_conservative_results(results, config)
    else:
        print("❌ Mesmo ultra-conservador, BTC está perdendo")
        print("🔧 DIAGNÓSTICO ADICIONAL NECESSÁRIO")
        
        if btc_result:
            # Análise detalhada do BTC
            analyze_failing_strategy("BTC", "dados_reais_btc_1ano.csv", config)

def test_asset_ultra_conservative(asset_name, filename, config):
    """Testa asset com estratégia ultra-conservadora"""
    
    if not os.path.exists(filename):
        return None
    
    try:
        df = pd.read_csv(filename)
        
        # Padronizar colunas
        column_mapping = {
            'open': 'valor_abertura',
            'high': 'valor_maximo',
            'low': 'valor_minimo',
            'close': 'valor_fechamento',
            'volume': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Calcular indicadores
        df = calculate_ultra_conservative_indicators(df, config)
        
        # Simular
        result = simulate_ultra_conservative(df, config)
        result['asset'] = asset_name
        
        roi = result['total_return_pct']
        trades = result['num_trades']
        wr = result['win_rate'] * 100
        dd = result['max_drawdown']
        
        status = "✅" if roi > 0 else "⚠️" if roi > -10 else "❌"
        print(f"{status} {asset_name}: {roi:+6.1f}% | {trades:3d} trades | WR {wr:5.1f}% | DD {dd:5.1f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ {asset_name}: Erro - {str(e)[:40]}")
        return None

def calculate_ultra_conservative_indicators(df, config):
    """Calcula indicadores para estratégia ultra-conservadora"""
    df = df.copy()
    
    # EMAs mais rápidas
    df['ema_short'] = df['valor_fechamento'].ewm(span=config['ema_short']).mean()
    df['ema_long'] = df['valor_fechamento'].ewm(span=config['ema_long']).mean()
    
    # ATR
    high_low = df['valor_maximo'] - df['valor_minimo']
    high_close = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
    low_close = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=7).mean()  # ATR mais rápido
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    # Volume MA mais rápido
    df['vol_ma'] = df['volume'].rolling(window=10).mean()
    
    # RSI mais rápido
    delta = df['valor_fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def simulate_ultra_conservative(df, config):
    """Simula trading ultra-conservador"""
    
    balance = 1000.0
    trades = []
    position = None
    
    # Começar mais tarde para ter dados dos indicadores
    start_idx = max(config['ema_long'], 20)
    
    for i in range(start_idx, len(df) - 1):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Gerenciar posição
        if position:
            entry_price = position['entry_price']
            side = position['side']
            entry_balance = position['entry_balance']
            
            # P&L com leverage baixo
            if side == 'long':
                pnl_pct = ((current['valor_fechamento'] - entry_price) / entry_price) * config['leverage']
            else:
                pnl_pct = ((entry_price - current['valor_fechamento']) / entry_price) * config['leverage']
            
            # TP/SL ultra-conservador
            tp_pct = config['tp_pct'] / 100.0
            sl_pct = config['sl_pct'] / 100.0
            
            if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                final_balance = entry_balance * (1 + pnl_pct)
                
                trades.append({
                    'side': side,
                    'pnl_pct': pnl_pct,
                    'balance_after': final_balance
                })
                
                balance = final_balance
                position = None
                continue
        
        # Verificar entrada ultra-permissiva
        if not position:
            can_long, can_short = check_ultra_conservative_entry(current, prev, config)
            
            if can_long:
                position = {
                    'side': 'long',
                    'entry_price': current['valor_fechamento'],
                    'entry_balance': balance
                }
            elif can_short:
                position = {
                    'side': 'short',
                    'entry_price': current['valor_fechamento'],
                    'entry_balance': balance
                }
    
    # Métricas
    if not trades:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'final_balance': balance
        }
    
    total_return_pct = ((balance - 1000.0) / 1000.0) * 100
    wins = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    # Drawdown
    balances = [1000.0]
    for trade in trades:
        balances.append(trade['balance_after'])
    
    peak = balances[0]
    max_dd = 0
    for balance in balances:
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown': max_dd * 100,
        'final_balance': balance
    }

def check_ultra_conservative_entry(current, prev, config):
    """Critérios de entrada ultra-permissivos"""
    
    # EMA Cross
    ema_cross_bullish = (prev['ema_short'] <= prev['ema_long'] and 
                        current['ema_short'] > current['ema_long'])
    ema_cross_bearish = (prev['ema_short'] >= prev['ema_long'] and 
                        current['ema_short'] < current['ema_long'])
    
    # ATR baixo (sem volatilidade extrema)
    atr_ok = config['atr_min'] <= current['atr_pct'] <= config['atr_max']
    
    # Volume qualquer
    volume_ok = current['volume'] > (current['vol_ma'] * config['volume_mult'])
    
    # Decisões ultra-simples
    can_long = False
    can_short = False
    
    if ema_cross_bullish and atr_ok:
        can_long = True
    elif ema_cross_bearish and atr_ok:
        can_short = True
    
    # RSI force
    if current['rsi'] < 30:
        can_long = True
    elif current['rsi'] > 70:
        can_short = True
    
    return can_long, can_short

def analyze_failing_strategy(asset_name, filename, config):
    """Analisa por que a estratégia está falhando"""
    
    print(f"\n🔬 DIAGNÓSTICO DETALHADO - {asset_name}")
    print("-"*50)
    
    df = pd.read_csv(filename)
    
    # Padronizar
    column_mapping = {
        'open': 'valor_abertura',
        'high': 'valor_maximo',
        'low': 'valor_minimo',
        'close': 'valor_fechamento',
        'volume': 'volume'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Calcular retornos simples
    df['retorno_simples'] = df['valor_fechamento'].pct_change()
    df['retorno_acumulado'] = (1 + df['retorno_simples']).cumprod()
    
    # Estatísticas básicas
    print(f"📊 ESTATÍSTICAS BÁSICAS:")
    print(f"   Retorno total (buy and hold): {(df['retorno_acumulado'].iloc[-1] - 1) * 100:.1f}%")
    print(f"   Retorno médio diário: {df['retorno_simples'].mean() * 100:.3f}%")
    print(f"   Volatilidade diária: {df['retorno_simples'].std() * 100:.2f}%")
    
    # Dias positivos vs negativos
    dias_positivos = (df['retorno_simples'] > 0).sum()
    dias_negativos = (df['retorno_simples'] < 0).sum()
    
    print(f"   Dias positivos: {dias_positivos}/{len(df)} ({dias_positivos/len(df)*100:.1f}%)")
    print(f"   Dias negativos: {dias_negativos}/{len(df)} ({dias_negativos/len(df)*100:.1f}%)")
    
    # Impacto do leverage
    retorno_com_leverage = df['retorno_simples'] * config['leverage']
    print(f"\n⚠️ IMPACTO DO LEVERAGE {config['leverage']}x:")
    print(f"   Maior ganho diário: {retorno_com_leverage.max() * 100:.1f}%")
    print(f"   Maior perda diária: {retorno_com_leverage.min() * 100:.1f}%")
    
    # Dias com perdas > SL
    dias_liquidacao = (retorno_com_leverage < -config['sl_pct']/100).sum()
    print(f"   Dias que levariam à liquidação (SL {config['sl_pct']}%): {dias_liquidacao}")
    
    print(f"\n💡 CONCLUSÕES:")
    if dias_liquidacao > 10:
        print(f"   ❌ Muitos dias de liquidação - leverage ainda muito alto")
    if df['retorno_simples'].mean() <= 0:
        print(f"   ❌ Retorno médio negativo - asset em tendência de baixa")
    if df['retorno_simples'].std() > 0.05:
        print(f"   ❌ Volatilidade muito alta - difícil de tradear")

def analyze_ultra_conservative_results(results, config):
    """Analisa resultados ultra-conservadores"""
    
    print(f"\n📊 ANÁLISE ULTRA-CONSERVADORA")
    print("="*50)
    
    avg_roi = np.mean([r['total_return_pct'] for r in results])
    profitable = len([r for r in results if r['total_return_pct'] > 0])
    
    print(f"ROI Médio: {avg_roi:.1f}%")
    print(f"Assets Lucrativos: {profitable}/{len(results)}")
    
    if avg_roi > -5 and profitable >= len(results) * 0.5:
        print("✅ ESTRATÉGIA ULTRA-CONSERVADORA VIÁVEL!")
        print("💡 Pode ser base para desenvolvimento")
    else:
        print("❌ Mesmo ultra-conservadora, estratégia não funciona")
        print("🚨 PROBLEMA FUNDAMENTAL na abordagem")

if __name__ == "__main__":
    test_ultra_conservative()
