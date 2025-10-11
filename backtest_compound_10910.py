#!/usr/bin/env python3
"""
🎯 RÉPLICA EXATA +10.910% - EFEITO COMPOSTO
Reimplementando com reinvestimento total para reproduzir os resultados originais
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# DNA GENÉTICO EXATO
DNA_VENCEDOR = {
    'leverage': 3,
    'sl_pct': 0.015,     # 1.5%
    'tp_pct': 0.12,      # 12%
    'ema_fast': 3,
    'ema_slow': 34,
    'rsi_period': 21,
    'rsi_min': 20,
    'rsi_max': 85,
    'volume_multiplier': 1.82,
    'atr_min_pct': 0.45,
    'min_confluencia': 3
}

def load_1year_data(asset_name):
    """Carrega dados de 1 ano"""
    filename = f"dados_reais_{asset_name.lower()}_1ano.csv"
    
    if not os.path.exists(filename):
        print(f"❌ Arquivo não encontrado: {filename}")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"✅ {asset_name}: {len(df)} registros")
        return df
    except Exception as e:
        print(f"❌ Erro {asset_name}: {e}")
        return None

def calculate_indicators(df):
    """Calcula indicadores técnicos"""
    
    # EMAs
    df['ema_3'] = df['close'].ewm(span=3).mean()
    df['ema_34'] = df['close'].ewm(span=34).mean()
    
    # RSI 21
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
    rs = gain / loss
    df['rsi_21'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    return df

def genetic_signal_check(df, i, dna):
    """Verifica sinal conforme DNA genético"""
    
    if i < 100:
        return False, []
    
    signals = []
    conditions = []
    
    # 1. EMA Cross
    ema_cross = df['ema_3'].iloc[i] > df['ema_34'].iloc[i]
    if ema_cross:
        conditions.append("EMA_CROSS")
        signals.append(True)
    else:
        signals.append(False)
    
    # 2. RSI
    rsi = df['rsi_21'].iloc[i]
    rsi_ok = dna['rsi_min'] < rsi < dna['rsi_max']
    if rsi_ok:
        conditions.append(f"RSI({rsi:.1f})")
        signals.append(True)
    else:
        signals.append(False)
    
    # 3. Volume
    vol_ratio = df['volume_ratio'].iloc[i]
    vol_ok = vol_ratio > dna['volume_multiplier']
    if vol_ok:
        conditions.append(f"VOL({vol_ratio:.2f}x)")
        signals.append(True)
    else:
        signals.append(False)
    
    # 4. ATR
    atr_pct = df['atr_pct'].iloc[i]
    atr_ok = atr_pct > dna['atr_min_pct']
    if atr_ok:
        conditions.append(f"ATR({atr_pct:.2f}%)")
        signals.append(True)
    else:
        signals.append(False)
    
    # Confluência
    confluencia = sum(signals)
    has_signal = confluencia >= dna['min_confluencia']
    
    return has_signal, conditions

def simulate_compound_trading(df, asset_name, dna):
    """Simula trading com EFEITO COMPOSTO (chave dos +68,700%)"""
    
    # 🎯 CRUCIAL: Capital inicial que será REINVESTIDO a cada trade
    capital = 1.0  # $1 inicial
    leverage = dna['leverage']
    sl_pct = dna['sl_pct']
    tp_pct = dna['tp_pct']
    
    trades = []
    in_position = False
    entry_price = 0
    entry_index = 0
    
    print(f"\n🧬 {asset_name} - SIMULAÇÃO COM EFEITO COMPOSTO")
    print(f"   Capital inicial: ${capital:.4f}")
    print(f"   🎯 CADA TRADE USA 100% DO CAPITAL ATUAL (reinvestimento)")
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        if not in_position:
            # Verificar entrada
            has_signal, conditions = genetic_signal_check(df, i, dna)
            
            if has_signal and capital > 0.001:  # Só operar se tiver capital suficiente
                in_position = True
                entry_price = current_price
                entry_index = i
                
                stop_price = entry_price * (1 - sl_pct)
                take_price = entry_price * (1 + tp_pct)
                
                # Log apenas trades importantes
                if len(trades) % 50 == 0:
                    print(f"   📈 ENTRADA #{len(trades)+1} @ ${entry_price:.6f}")
                    print(f"      Capital atual: ${capital:.4f}")
                    print(f"      Condições: {' | '.join(conditions[:2])}")
        
        else:
            # Verificar saída
            stop_price = entry_price * (1 - sl_pct)
            take_price = entry_price * (1 + tp_pct)
            
            exit_reason = None
            exit_price = current_price
            
            if current_price <= stop_price:
                exit_reason = "STOP_LOSS"
                exit_price = stop_price
            elif current_price >= take_price:
                exit_reason = "TAKE_PROFIT"
                exit_price = take_price
            
            if exit_reason:
                in_position = False
                
                # 🎯 CÁLCULO COM EFEITO COMPOSTO
                price_change_pct = (exit_price - entry_price) / entry_price
                leveraged_return = price_change_pct * leverage
                
                # 🚀 REINVESTIMENTO TOTAL: O capital muda baseado no resultado
                capital_before = capital
                capital = capital * (1 + leveraged_return)
                
                # Evitar capital negativo
                if capital < 0:
                    capital = 0.001
                
                trade_pnl = capital - capital_before
                
                trade = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_index': entry_index,
                    'exit_index': i,
                    'price_change_pct': price_change_pct * 100,
                    'leveraged_return_pct': leveraged_return * 100,
                    'capital_before': capital_before,
                    'capital_after': capital,
                    'trade_pnl': trade_pnl,
                    'exit_reason': exit_reason
                }
                trades.append(trade)
                
                # Log progresso
                if len(trades) % 50 == 0 or len(trades) <= 10:
                    roi_current = (capital - 1.0) / 1.0 * 100
                    win_loss = "🟢 TP" if exit_reason == "TAKE_PROFIT" else "🔴 SL"
                    print(f"   📉 SAÍDA #{len(trades)} @ ${exit_price:.6f} ({exit_reason})")
                    print(f"      {win_loss}: {leveraged_return*100:+.1f}% | Capital: ${capital:.4f} | ROI: {roi_current:+.1f}%")
    
    # Estatísticas finais
    total_trades = len(trades)
    tp_trades = [t for t in trades if t['exit_reason'] == 'TAKE_PROFIT']
    sl_trades = [t for t in trades if t['exit_reason'] == 'STOP_LOSS']
    
    win_rate = (len(tp_trades) / total_trades * 100) if total_trades > 0 else 0
    final_roi = (capital - 1.0) / 1.0 * 100
    
    return {
        'asset': asset_name,
        'initial_capital': 1.0,
        'final_capital': capital,
        'total_trades': total_trades,
        'tp_trades': len(tp_trades),
        'sl_trades': len(sl_trades),
        'win_rate': win_rate,
        'roi_pct': final_roi,
        'trades': trades
    }

def run_compound_backtest():
    """Executa backtest com efeito composto para reproduzir +10.910%"""
    
    print("🎯 BACKTEST EFEITO COMPOSTO - REPRODUZINDO +10.910%")
    print("="*70)
    print("🧬 Metodologia: REINVESTIMENTO TOTAL do capital a cada trade")
    print("🎯 Objetivo: Reproduzir XRP +68.700% e média +10.910%")
    print("="*70)
    
    # Assets do relatório original
    assets = ['btc', 'eth', 'bnb', 'sol', 'ada', 'avax', 'doge', 'link', 'ltc', 'xrp']
    
    results = []
    total_capital_initial = len(assets) * 1.0  # $1 por asset
    total_capital_final = 0
    
    for asset in assets:
        df = load_1year_data(asset)
        
        if df is not None:
            # Padronizar colunas
            if 'valor_fechamento' in df.columns:
                df['close'] = df['valor_fechamento']
                df['high'] = df['valor_maximo']
                df['low'] = df['valor_minimo']
                df['volume'] = df['volume']
            
            # Calcular indicadores
            df = calculate_indicators(df)
            
            # Simular trading com compound
            result = simulate_compound_trading(df, asset.upper(), DNA_VENCEDOR)
            results.append(result)
            
            total_capital_final += result['final_capital']
    
    # Relatório final
    print(f"\n🏆 RESULTADO FINAL COM EFEITO COMPOSTO")
    print("="*75)
    print("Asset | ROI Compound | Capital Final | Trades | Win% | vs Original")
    print("-" * 75)
    
    # Resultados originais para comparação
    original_results = {
        'XRP': 68700.7, 'DOGE': 16681.0, 'LINK': 8311.4, 'ADA': 5449.0,
        'SOL': 2751.6, 'ETH': 2531.3, 'LTC': 1565.6, 'AVAX': 1548.9,
        'BNB': 909.1, 'BTC': 651.9
    }
    
    for result in results:
        asset = result['asset']
        roi = result['roi_pct']
        capital = result['final_capital']
        trades = result['total_trades']
        win_rate = result['win_rate']
        
        original_roi = original_results.get(asset, 0)
        difference = roi - original_roi
        accuracy = (roi / original_roi * 100) if original_roi > 0 else 0
        
        status = "🎯" if abs(difference) < original_roi * 0.1 else "📊"
        
        print(f"{asset:5} | {roi:+10.1f}% | ${capital:11.4f} | {trades:6} | {win_rate:4.1f} | {accuracy:6.1f}% {status}")
    
    # Portfolio total
    portfolio_roi = (total_capital_final - total_capital_initial) / total_capital_initial * 100
    
    print("-" * 75)
    print(f"TOTAL | {portfolio_roi:+10.1f}% | ${total_capital_final:11.4f} | Portfolio ROI")
    print("="*75)
    
    print(f"\n💰 COMPARAÇÃO COM ORIGINAL:")
    print(f"   💵 Capital inicial: ${total_capital_initial:.2f}")
    print(f"   💰 Capital final: ${total_capital_final:.2f}")
    print(f"   📈 ROI obtido: {portfolio_roi:+.1f}%")
    print(f"   🎯 ROI original: +10.910%")
    print(f"   📊 Diferença: {portfolio_roi - 10910:+.1f}%")
    print(f"   ✅ Precisão: {(portfolio_roi / 10910 * 100):+.1f}%")
    
    if abs(portfolio_roi - 10910) < 1000:  # Tolerância ±1000%
        print(f"\n🎉 RÉPLICA BEM-SUCEDIDA!")
        print(f"   ✅ Efeito composto reproduziu os resultados originais")
        print(f"   🧬 DNA genético validado com dados reais")
    else:
        print(f"\n📊 RÉPLICA PARCIAL:")
        print(f"   ⚠️  Diferença significativa encontrada")
        print(f"   🔍 Possíveis fatores: período dos dados, sequência de trades")
    
    # Top performers
    top_results = sorted(results, key=lambda x: x['roi_pct'], reverse=True)[:3]
    print(f"\n🏆 TOP 3 PERFORMERS:")
    for i, result in enumerate(top_results, 1):
        roi = result['roi_pct']
        asset = result['asset']
        trades = result['total_trades']
        print(f"   {i}º {asset}: {roi:+.1f}% ({trades} trades)")
    
    return results

if __name__ == "__main__":
    results = run_compound_backtest()
