#!/usr/bin/env python3
"""
🧬 BACKTEST RÉPLICA EXATA +10.910% ROI
Recria o backtest histórico que resultou no ROI de 10.910%
Usando exatamente os mesmos parâmetros do DNA genético vencedor
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração DNA GENÉTICO VENCEDOR (conforme RELATORIO_FINAL_DEFINITIVO.md)
DNA_GENETICO = {
    'leverage': 3,                    # Leverage 3x (otimizado)
    'sl_pct': 0.015,                 # Stop Loss 1.5% (ultra agressivo)
    'tp_pct': 0.12,                  # Take Profit 12% (balanço risco/retorno)
    'ema_fast': 3,                   # EMA rápida 3 períodos (ultra sensível)
    'ema_slow': 34,                  # EMA lenta 34 períodos (filtro tendência)
    'rsi_period': 21,                # RSI 21 períodos
    'rsi_min': 20,                   # RSI mínimo 20 (overbought)
    'rsi_max': 85,                   # RSI máximo 85 (oversold)
    'volume_multiplier': 1.82,       # Volume > 1.82x média (confirmação forte)
    'atr_min_pct': 0.45,            # ATR mínimo 0.45% (volatilidade adequada)
    'min_confluencia': 3             # Mínimo 3 critérios (agilidade vs precisão)
}

def load_1year_data(asset_name):
    """Carrega dados de 1 ano para o asset"""
    filename = f"dados_reais_{asset_name.lower()}_1ano.csv"
    
    if not os.path.exists(filename):
        print(f"❌ Arquivo não encontrado: {filename}")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"✅ Carregado {asset_name}: {len(df)} registros (1 ano)")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar {asset_name}: {e}")
        return None

def calculate_indicators(df):
    """Calcula indicadores técnicos conforme DNA genético"""
    
    # EMAs conforme DNA (3 e 34 períodos)
    df['ema_3'] = df['close'].ewm(span=3).mean()
    df['ema_34'] = df['close'].ewm(span=34).mean()
    
    # RSI 21 períodos conforme DNA
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
    rs = gain / loss
    df['rsi_21'] = 100 - (100 / (1 + rs))
    
    # Volume médio para comparação (20 períodos)
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # ATR (Average True Range) 14 períodos
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    return df

def genetic_signal_check(df, i, dna):
    """Verifica sinal de entrada conforme DNA genético"""
    
    if i < 100:  # Aguardar indicadores estabilizarem
        return False, []
    
    conditions = []
    signals = []
    
    # 1. EMA Cross (EMA 3 > EMA 34)
    ema_cross = df['ema_3'].iloc[i] > df['ema_34'].iloc[i]
    if ema_cross:
        conditions.append("✅ EMA_CROSS")
        signals.append(True)
    else:
        signals.append(False)
    
    # 2. RSI dentro da faixa (20 < RSI < 85)
    rsi = df['rsi_21'].iloc[i]
    rsi_ok = dna['rsi_min'] < rsi < dna['rsi_max']
    if rsi_ok:
        conditions.append(f"✅ RSI({rsi:.1f})")
        signals.append(True)
    else:
        signals.append(False)
    
    # 3. Volume boost (> 1.82x média)
    vol_ratio = df['volume_ratio'].iloc[i]
    vol_ok = vol_ratio > dna['volume_multiplier']
    if vol_ok:
        conditions.append(f"✅ VOL({vol_ratio:.2f}x)")
        signals.append(True)
    else:
        signals.append(False)
    
    # 4. ATR saudável (> 0.45%)
    atr_pct = df['atr_pct'].iloc[i]
    atr_ok = atr_pct > dna['atr_min_pct']
    if atr_ok:
        conditions.append(f"✅ ATR({atr_pct:.2f}%)")
        signals.append(True)
    else:
        signals.append(False)
    
    # Confluência: mínimo 3 critérios
    confluencia = sum(signals)
    has_signal = confluencia >= dna['min_confluencia']
    
    if has_signal:
        conditions.append(f"🎯 CONFLUÊNCIA({confluencia}/4)")
    
    return has_signal, conditions

def simulate_genetic_trading(df, asset_name, dna):
    """Simula trading com DNA genético por 1 ano"""
    
    initial_capital = 1.0  # $1 por asset conforme relatório
    balance = initial_capital
    position_size = initial_capital
    trades = []
    
    in_position = False
    entry_price = 0
    entry_index = 0
    
    print(f"\n🧬 Simulando {asset_name} com DNA Genético...")
    print(f"   Capital inicial: ${initial_capital:.2f}")
    print(f"   Leverage: {dna['leverage']}x")
    print(f"   SL: {dna['sl_pct']*100:.1f}% | TP: {dna['tp_pct']*100:.1f}%")
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        if not in_position:
            # Verificar sinal de entrada
            has_signal, conditions = genetic_signal_check(df, i, dna)
            
            if has_signal:
                # ABRIR POSIÇÃO LONG
                in_position = True
                entry_price = current_price
                entry_index = i
                
                # Calcular stops conforme DNA
                stop_price = entry_price * (1 - dna['sl_pct'])
                take_price = entry_price * (1 + dna['tp_pct'])
                
                print(f"   📈 ENTRADA #{len(trades)+1} @ ${entry_price:.6f}")
                print(f"      Condições: {' | '.join(conditions)}")
                print(f"      SL: ${stop_price:.6f} | TP: ${take_price:.6f}")
        
        else:
            # Verificar saída da posição
            stop_price = entry_price * (1 - dna['sl_pct'])
            take_price = entry_price * (1 + dna['tp_pct'])
            
            exit_reason = None
            exit_price = current_price
            
            # Verificar Stop Loss
            if current_price <= stop_price:
                exit_reason = "STOP_LOSS"
                exit_price = stop_price
            
            # Verificar Take Profit
            elif current_price >= take_price:
                exit_reason = "TAKE_PROFIT"
                exit_price = take_price
            
            if exit_reason:
                # FECHAR POSIÇÃO
                in_position = False
                
                # Calcular resultado com leverage
                price_change_pct = (exit_price - entry_price) / entry_price
                leveraged_return = price_change_pct * dna['leverage']
                trade_result = position_size * leveraged_return
                
                balance += trade_result
                
                # Salvar trade
                trade = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_index': entry_index,
                    'exit_index': i,
                    'price_change_pct': price_change_pct * 100,
                    'leveraged_return_pct': leveraged_return * 100,
                    'trade_result': trade_result,
                    'balance_after': balance,
                    'exit_reason': exit_reason
                }
                trades.append(trade)
                
                win_loss = "🟢 WIN" if trade_result > 0 else "🔴 LOSS"
                print(f"   📉 SAÍDA #{len(trades)} @ ${exit_price:.6f} ({exit_reason})")
                print(f"      {win_loss}: {leveraged_return*100:+.2f}% | Balance: ${balance:.6f}")
    
    # Calcular estatísticas
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['trade_result'] > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    final_roi = (balance - initial_capital) / initial_capital * 100
    
    result = {
        'asset': asset_name,
        'initial_capital': initial_capital,
        'final_balance': balance,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'roi_pct': final_roi,
        'trades': trades
    }
    
    return result

def run_genetic_backtest():
    """Executa backtest completo conforme DNA genético"""
    
    print("🧬 BACKTEST RÉPLICA EXATA +10.910% ROI")
    print("="*70)
    print("Baseado no RELATORIO_FINAL_DEFINITIVO.md")
    print("DNA Genético Vencedor: SL 1.5% | TP 12% | Leverage 3x")
    print("="*70)
    
    # Assets conforme relatório original (10 assets)
    assets = ['btc', 'eth', 'bnb', 'sol', 'ada', 'avax', 'doge', 'link', 'ltc', 'xrp']
    
    results = []
    total_invested = 0
    total_returned = 0
    
    for asset in assets:
        df = load_1year_data(asset)
        
        if df is not None:
            # Padronizar colunas se necessário
            if 'valor_fechamento' in df.columns:
                df['close'] = df['valor_fechamento']
                df['high'] = df['valor_maximo']
                df['low'] = df['valor_minimo']
                df['volume'] = df['volume']
            
            # Calcular indicadores
            df = calculate_indicators(df)
            
            # Simular trading
            result = simulate_genetic_trading(df, asset.upper(), DNA_GENETICO)
            results.append(result)
            
            total_invested += result['initial_capital']
            total_returned += result['final_balance']
    
    # Relatório final
    print(f"\n🏆 RESULTADO FINAL GENÉTICO")
    print("="*70)
    print("Asset | ROI Final | Balance | Trades | Win% | Status")
    print("-" * 70)
    
    for result in results:
        asset = result['asset']
        roi = result['roi_pct']
        balance = result['final_balance']
        trades = result['total_trades']
        win_rate = result['win_rate']
        
        # Status conforme ROI
        if roi > 5000:
            status = "🤯 EXPLOSÃO"
        elif roi > 1000:
            status = "🚀 EXCELENTE"
        elif roi > 500:
            status = "💎 ÓTIMO"
        elif roi > 100:
            status = "📈 BOM"
        else:
            status = "✅ POSITIVO"
        
        print(f"{asset:5} | {roi:+8.1f}% | ${balance:6.3f} | {trades:6} | {win_rate:4.1f} | {status}")
    
    # Totais
    portfolio_roi = (total_returned - total_invested) / total_invested * 100
    
    print("-" * 70)
    print(f"TOTAL | {portfolio_roi:+8.1f}% | ${total_returned:6.3f} | Portfolio ROI")
    print("="*70)
    
    print(f"\n💰 RESUMO FINANCEIRO:")
    print(f"   💵 Capital inicial: ${total_invested:.2f}")
    print(f"   💰 Capital final: ${total_returned:.2f}")
    print(f"   📈 ROI do portfolio: {portfolio_roi:+.1f}%")
    print(f"   🎯 Meta original: +10.910%")
    
    if abs(portfolio_roi - 10910) < 100:  # Tolerância de ±100%
        print(f"   ✅ RÉPLICA BEM-SUCEDIDA! (±100% da meta)")
    else:
        print(f"   ⚠️  Diferença da meta: {portfolio_roi - 10910:+.1f}%")
    
    print(f"\n🧬 DNA UTILIZADO:")
    for param, value in DNA_GENETICO.items():
        print(f"   {param}: {value}")
    
    return results

if __name__ == "__main__":
    results = run_genetic_backtest()
