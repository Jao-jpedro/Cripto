#!/usr/bin/env python3
"""
GANHOS REAIS COM TODOS OS 10 ASSETS DISPONÍVEIS
Banca $10 + Entradas $1 em cada asset
"""

import pandas as pd
import numpy as np
import os

def load_data(filename):
    """Carrega e padroniza dados"""
    if not os.path.exists(filename):
        return None
    
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
    
    return df

def calculate_indicators(df):
    """Calcula indicadores técnicos"""
    
    # EMAs
    df['ema_9'] = df['valor_fechamento'].ewm(span=9).mean()
    df['ema_21'] = df['valor_fechamento'].ewm(span=21).mean()
    df['ema_50'] = df['valor_fechamento'].ewm(span=50).mean()
    
    # RSI
    delta = df['valor_fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume médio
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    
    return df

def optimized_entry_signal(df, i):
    """Sinal de entrada com 4 critérios de confluência"""
    
    if i < 50:
        return False
    
    current_price = df['valor_fechamento'].iloc[i]
    prev_price = df['valor_fechamento'].iloc[i-1]
    
    conditions = []
    
    # 1. Tendência EMA
    ema_9 = df['ema_9'].iloc[i]
    ema_21 = df['ema_21'].iloc[i]
    ema_50 = df['ema_50'].iloc[i]
    ema_aligned = ema_9 > ema_21 > ema_50
    conditions.append(ema_aligned)
    
    # 2. Preço acima EMA
    price_above_ema = current_price > ema_9
    conditions.append(price_above_ema)
    
    # 3. RSI neutro
    rsi = df['rsi'].iloc[i]
    rsi_ok = 30 < rsi < 70
    conditions.append(rsi_ok)
    
    # 4. Momentum
    momentum_ok = current_price > prev_price
    conditions.append(momentum_ok)
    
    # 5. Volume
    volume = df['volume'].iloc[i]
    volume_ma = df['volume_ma'].iloc[i]
    volume_ok = volume > volume_ma
    conditions.append(volume_ok)
    
    # Confluência: 4 de 5
    return sum(conditions) >= 4

def simulate_asset(asset_name):
    """Simula estratégia para um asset"""
    
    filename = f"dados_reais_{asset_name.lower()}_1ano.csv"
    df = load_data(filename)
    
    if df is None:
        return None
    
    # Configuração otimizada
    LEVERAGE = 3
    SL_PCT = 0.04  # 4%
    TP_PCT = 0.10  # 10%
    INITIAL_BALANCE = 1.0  # $1 por asset
    
    df = calculate_indicators(df)
    
    balance = INITIAL_BALANCE
    trades = []
    position = None
    max_balance = INITIAL_BALANCE
    max_drawdown = 0
    
    for i in range(50, len(df) - 1):
        current_price = df['valor_fechamento'].iloc[i]
        
        # Entrada
        if position is None:
            if optimized_entry_signal(df, i):
                position = {
                    'entry_price': current_price,
                    'entry_time': i,
                    'side': 'buy'
                }
        
        # Saída
        elif position is not None:
            entry_price = position['entry_price']
            
            sl_level = entry_price * (1 - SL_PCT)
            tp_level = entry_price * (1 + TP_PCT)
            
            exit_reason = None
            exit_price = None
            
            if current_price <= sl_level:
                exit_reason = "SL"
                exit_price = sl_level
            elif current_price >= tp_level:
                exit_reason = "TP"
                exit_price = tp_level
            
            if exit_reason:
                price_change = (exit_price - entry_price) / entry_price
                pnl_leveraged = price_change * LEVERAGE
                trade_pnl = balance * pnl_leveraged
                balance += trade_pnl
                
                trades.append({
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_leveraged * 100,
                    'balance_after': balance
                })
                
                max_balance = max(max_balance, balance)
                current_drawdown = (max_balance - balance) / max_balance
                max_drawdown = max(max_drawdown, current_drawdown)
                
                position = None
    
    tp_trades = [t for t in trades if t['exit_reason'] == 'TP']
    sl_trades = [t for t in trades if t['exit_reason'] == 'SL']
    
    return {
        'asset': asset_name,
        'final_balance': balance,
        'total_return': (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        'num_trades': len(trades),
        'tp_trades': len(tp_trades),
        'sl_trades': len(sl_trades),
        'win_rate': len(tp_trades) / len(trades) * 100 if trades else 0,
        'max_drawdown': max_drawdown * 100
    }

def calculate_all_assets_gains():
    """Calcula ganhos com todos os assets disponíveis"""
    
    print("💰 GANHOS REAIS COM TODOS OS ASSETS DISPONÍVEIS")
    print("="*80)
    print("💵 Banca inicial: $10.00")
    print("📊 Investimento: $1.00 por asset")
    print("⚡ Configuração: Leverage 3x | TP 10% | SL 4% | Confluência 4")
    print()
    
    # Todos os assets disponíveis
    all_assets = ['btc', 'eth', 'bnb', 'sol', 'ada', 'avax', 'doge', 'link', 'ltc', 'xrp']
    
    results = []
    total_invested = 0
    total_returned = 0
    successful_assets = 0
    
    print("🎯 PERFORMANCE POR ASSET:")
    print("-"*70)
    print("Asset | ROI      | Final  | Trades | Win% | DD%   | Status")
    print("-" * 65)
    
    for asset in all_assets:
        result = simulate_asset(asset)
        
        if result is None:
            print(f"{asset.upper():5} | N/A      | N/A    | N/A    | N/A  | N/A   | ❌ Sem dados")
            continue
        
        results.append(result)
        
        asset_name = result['asset']
        roi = result['total_return']
        final_balance = result['final_balance']
        trades = result['num_trades']
        win_rate = result['win_rate']
        drawdown = result['max_drawdown']
        
        total_invested += 1.0
        total_returned += final_balance
        
        if roi > 0:
            successful_assets += 1
        
        # Status
        if roi > 300:
            status = "🚀"
        elif roi > 100:
            status = "🎉"
        elif roi > 0:
            status = "✅"
        else:
            status = "❌"
        
        print(f"{asset_name.upper():5} | {roi:+7.1f}% | ${final_balance:5.2f} | {trades:6} | {win_rate:4.1f} | {drawdown:5.1f} | {status}")
    
    # Totais
    total_profit = total_returned - total_invested
    average_roi = (total_returned / total_invested - 1) * 100 if total_invested > 0 else 0
    success_rate = (successful_assets / len(results)) * 100 if results else 0
    
    print("-" * 65)
    print(f"TOTAL | {average_roi:+7.1f}% | ${total_returned:5.2f} | {'':>6} | {'':>4} | {'':>5} | {success_rate:.0f}% lucr.")
    
    return {
        'total_invested': total_invested,
        'total_returned': total_returned,
        'total_profit': total_profit,
        'average_roi': average_roi,
        'successful_assets': successful_assets,
        'total_assets': len(results),
        'results': results
    }

def analyze_portfolio_scenarios(summary):
    """Analisa diferentes cenários de portfólio"""
    
    print(f"\n📊 ANÁLISE DE CENÁRIOS DE PORTFÓLIO:")
    print("="*60)
    
    total_invested = summary['total_invested']
    total_returned = summary['total_returned']
    total_profit = summary['total_profit']
    
    # Cenário 1: Portfólio completo
    print(f"🎯 CENÁRIO 1: PORTFÓLIO COMPLETO")
    print(f"   Investimento: ${total_invested:.2f}")
    print(f"   Retorno: ${total_returned:.2f}")
    print(f"   Lucro: ${total_profit:+.2f}")
    print(f"   ROI: {summary['average_roi']:+.1f}%")
    print(f"   Capital restante: ${10.0 - total_invested:.2f}")
    
    # Cenário 2: Apenas top performers
    results = summary['results']
    top_performers = sorted([r for r in results if r['total_return'] > 100], 
                          key=lambda x: x['total_return'], reverse=True)[:5]
    
    if top_performers:
        print(f"\n🏆 CENÁRIO 2: TOP 5 PERFORMERS")
        top_invested = len(top_performers) * 1.0
        top_returned = sum(r['final_balance'] for r in top_performers)
        top_profit = top_returned - top_invested
        top_roi = (top_returned / top_invested - 1) * 100
        
        print(f"   Assets: {', '.join([r['asset'].upper() for r in top_performers])}")
        print(f"   Investimento: ${top_invested:.2f}")
        print(f"   Retorno: ${top_returned:.2f}")
        print(f"   Lucro: ${top_profit:+.2f}")
        print(f"   ROI: {top_roi:+.1f}%")
        print(f"   Capital restante: ${10.0 - top_invested:.2f}")
    
    # Cenário 3: Estratégia sequencial
    print(f"\n🔄 CENÁRIO 3: ESTRATÉGIA SEQUENCIAL")
    print(f"   (Reinvestir ganhos sequencialmente)")
    
    if results:
        # Ordenar por ROI decrescente
        sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)
        
        current_balance = 10.0
        sequential_balance = current_balance
        
        print(f"   Sequência sugerida:")
        for i, result in enumerate(sorted_results[:5], 1):
            if sequential_balance >= 1.0:
                investment = 1.0
                roi = result['total_return']
                return_amount = investment * (1 + roi/100)
                profit = return_amount - investment
                sequential_balance = sequential_balance - investment + return_amount
                
                print(f"   {i}. {result['asset'].upper()}: $1.00 → ${return_amount:.2f} (+{roi:.1f}%) → Saldo: ${sequential_balance:.2f}")
        
        sequential_profit = sequential_balance - 10.0
        sequential_roi = (sequential_balance / 10.0 - 1) * 100
        
        print(f"   Resultado final: ${sequential_balance:.2f}")
        print(f"   Lucro total: ${sequential_profit:+.2f}")
        print(f"   ROI total: {sequential_roi:+.1f}%")

def project_compound_growth(summary):
    """Projeta crescimento composto"""
    
    print(f"\n📈 PROJEÇÃO DE CRESCIMENTO COMPOSTO:")
    print("="*60)
    
    avg_roi_decimal = summary['average_roi'] / 100
    initial_bankroll = 10.0
    
    periods = [1, 2, 3, 6, 12, 24]
    
    print(f"ROI médio por ciclo: {summary['average_roi']:.1f}%")
    print(f"Banca inicial: ${initial_bankroll:.2f}")
    print()
    print("Ciclos | Banca Final | Lucro Acumulado | ROI Acumulado")
    print("-" * 55)
    
    for period in periods:
        compound_balance = initial_bankroll * ((1 + avg_roi_decimal) ** period)
        compound_profit = compound_balance - initial_bankroll
        compound_roi = (compound_balance / initial_bankroll - 1) * 100
        
        if compound_balance < 1000:
            print(f"{period:6} | ${compound_balance:11.2f} | ${compound_profit:14.2f} | {compound_roi:11.1f}%")
        elif compound_balance < 1000000:
            print(f"{period:6} | ${compound_balance:11,.0f} | ${compound_profit:14,.0f} | {compound_roi:11,.0f}%")
        else:
            print(f"{period:6} | ${compound_balance:11,.0f} | ${compound_profit:14,.0f} | {compound_roi:11,.0f}%")

def main():
    # Calcular ganhos com todos os assets
    summary = calculate_all_assets_gains()
    
    # Análises adicionais
    analyze_portfolio_scenarios(summary)
    project_compound_growth(summary)
    
    print(f"\n" + "="*80)
    print("🎉 RESUMO EXECUTIVO - TODOS OS ASSETS:")
    print("="*80)
    
    total_invested = summary['total_invested']
    total_returned = summary['total_returned']
    total_profit = summary['total_profit']
    success_rate = (summary['successful_assets'] / summary['total_assets']) * 100
    
    print(f"💵 Banca inicial: $10.00")
    print(f"📊 Assets disponíveis: {summary['total_assets']}")
    print(f"💰 Total investido: ${total_invested:.2f}")
    print(f"🎯 Total retornado: ${total_returned:.2f}")
    print(f"✅ Lucro líquido: ${total_profit:+.2f}")
    print(f"📈 ROI médio: {summary['average_roi']:+.1f}%")
    print(f"🏆 Taxa de sucesso: {success_rate:.0f}%")
    print(f"💎 Capital reserva: ${10.0 - total_invested:.2f}")
    
    if total_profit > 15:
        print(f"🚀 RESULTADO EXCEPCIONAL!")
    elif total_profit > 10:
        print(f"🎉 RESULTADO EXCELENTE!")
    elif total_profit > 5:
        print(f"✅ RESULTADO SATISFATÓRIO!")
    else:
        print(f"📊 RESULTADO MODERADO")

if __name__ == "__main__":
    main()
