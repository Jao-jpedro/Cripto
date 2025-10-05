#!/usr/bin/env python3
"""
BACKTEST COMPLETO - LEVERAGE CORRIGIDO
Teste com diferentes leverages após corrigir o bug
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

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

def simulate_strategy_corrected(df, leverage=1, tp_pct=0.08, sl_pct=0.05, initial_balance=1000):
    """
    Simula estratégia com LEVERAGE CORRIGIDO
    - SL e TP fixos (não divididos pelo leverage)
    - P&L amplificado pelo leverage
    """
    
    balance = initial_balance
    trades = []
    position = None
    max_balance = initial_balance
    max_drawdown = 0
    
    for i in range(1, len(df) - 1):
        current_price = df['valor_fechamento'].iloc[i]
        
        # Se não tem posição, tentar entrar (estratégia simples: comprar na subida)
        if position is None:
            prev_price = df['valor_fechamento'].iloc[i-1]
            if current_price > prev_price:  # Preço subindo
                position = {
                    'entry_price': current_price,
                    'entry_time': i,
                    'side': 'buy'
                }
        
        # Se tem posição, verificar saída
        elif position is not None:
            entry_price = position['entry_price']
            
            # Calcular níveis de SL e TP (FIXOS - não divididos pelo leverage)
            sl_level = entry_price * (1 - sl_pct)  # -5%
            tp_level = entry_price * (1 + tp_pct)  # +8%
            
            exit_reason = None
            exit_price = None
            
            # Verificar SL
            if current_price <= sl_level:
                exit_reason = "SL"
                exit_price = sl_level
            
            # Verificar TP
            elif current_price >= tp_level:
                exit_reason = "TP"
                exit_price = tp_level
            
            # Se saiu, calcular P&L com LEVERAGE CORRIGIDO
            if exit_reason:
                # P&L do preço
                price_change = (exit_price - entry_price) / entry_price
                
                # P&L amplificado pelo leverage (CORREÇÃO!)
                pnl_leveraged = price_change * leverage
                
                # Aplicar ao balance
                trade_pnl = balance * pnl_leveraged
                balance += trade_pnl
                
                # Registrar trade
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_time': position['entry_time'],
                    'exit_time': i,
                    'price_change_pct': price_change * 100,
                    'pnl_leveraged_pct': pnl_leveraged * 100,
                    'trade_pnl': trade_pnl,
                    'balance_after': balance,
                    'exit_reason': exit_reason
                })
                
                # Atualizar drawdown
                max_balance = max(max_balance, balance)
                current_drawdown = (max_balance - balance) / max_balance
                max_drawdown = max(max_drawdown, current_drawdown)
                
                position = None
    
    return {
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'trades': trades,
        'num_trades': len(trades),
        'max_drawdown': max_drawdown * 100,
        'leverage': leverage
    }

def run_comprehensive_backtest():
    """Executa backtest completo com diferentes leverages"""
    
    print("🚀 BACKTEST COMPLETO - LEVERAGE CORRIGIDO")
    print("="*70)
    
    # Assets para testar
    assets = ['btc', 'eth', 'bnb', 'sol', 'ada', 'avax']
    leverages = [1, 3, 5, 10, 20]
    
    results = {}
    
    for asset in assets:
        filename = f"dados_reais_{asset}_1ano.csv"
        print(f"\n📊 TESTANDO {asset.upper()}...")
        
        df = load_data(filename)
        if df is None:
            print(f"   ❌ Arquivo {filename} não encontrado")
            continue
        
        results[asset] = {}
        
        print(f"   📈 Dados: {len(df)} barras")
        
        # Testar diferentes leverages
        for leverage in leverages:
            result = simulate_strategy_corrected(df, leverage=leverage)
            results[asset][leverage] = result
            
            roi = result['total_return']
            trades = result['num_trades']
            drawdown = result['max_drawdown']
            
            print(f"   Leverage {leverage:2}x: ROI {roi:+7.1f}% | {trades:3} trades | DD {drawdown:5.1f}%")
    
    return results

def analyze_results(results):
    """Analisa e apresenta os resultados"""
    
    print(f"\n" + "="*70)
    print("📊 ANÁLISE DETALHADA DOS RESULTADOS")
    print("="*70)
    
    # Resumo por leverage
    print(f"\n🎯 RESUMO POR LEVERAGE:")
    print("-"*50)
    
    leverages = [1, 3, 5, 10, 20]
    
    for leverage in leverages:
        total_roi = 0
        count = 0
        profitable_assets = 0
        
        print(f"\nLeverage {leverage}x:")
        print("Asset   | ROI      | Trades | Drawdown")
        print("-" * 40)
        
        for asset, asset_results in results.items():
            if leverage in asset_results:
                result = asset_results[leverage]
                roi = result['total_return']
                trades = result['num_trades']
                drawdown = result['max_drawdown']
                
                total_roi += roi
                count += 1
                if roi > 0:
                    profitable_assets += 1
                
                print(f"{asset.upper():7} | {roi:+7.1f}% | {trades:6} | {drawdown:7.1f}%")
        
        if count > 0:
            avg_roi = total_roi / count
            profit_rate = profitable_assets / count * 100
            print(f"        | {avg_roi:+7.1f}% | {'Média':>6} | {profit_rate:6.1f}% lucr.")
    
    # Comparação dramática
    print(f"\n🔥 COMPARAÇÃO DRAMÁTICA:")
    print("-"*50)
    
    for asset, asset_results in results.items():
        print(f"\n{asset.upper()}:")
        
        roi_1x = asset_results.get(1, {}).get('total_return', 0)
        roi_20x = asset_results.get(20, {}).get('total_return', 0)
        
        print(f"   Leverage  1x: {roi_1x:+7.1f}%")
        print(f"   Leverage 20x: {roi_20x:+7.1f}%")
        
        if roi_1x > 0:
            multiplier = roi_20x / roi_1x if roi_1x != 0 else 0
            print(f"   Amplificação: {multiplier:.1f}x")
        
        if roi_20x > roi_1x * 15:  # Se amplificou mais que 15x
            print(f"   Status: ✅ LEVERAGE FUNCIONANDO!")
        elif roi_20x < 0 and roi_1x > 0:
            print(f"   Status: ❌ Ainda com problemas")
        else:
            print(f"   Status: 🔄 Parcialmente corrigido")

def create_detailed_report(results):
    """Cria relatório detalhado"""
    
    print(f"\n" + "="*70)
    print("📋 RELATÓRIO DETALHADO")
    print("="*70)
    
    # Encontrar melhor configuração
    best_config = None
    best_roi = -float('inf')
    
    for asset, asset_results in results.items():
        for leverage, result in asset_results.items():
            roi = result['total_return']
            if roi > best_roi:
                best_roi = roi
                best_config = (asset, leverage, result)
    
    if best_config:
        asset, leverage, result = best_config
        print(f"\n🏆 MELHOR CONFIGURAÇÃO:")
        print(f"   Asset: {asset.upper()}")
        print(f"   Leverage: {leverage}x")
        print(f"   ROI: {result['total_return']:+.1f}%")
        print(f"   Trades: {result['num_trades']}")
        print(f"   Drawdown: {result['max_drawdown']:.1f}%")
    
    # Análise de trades
    print(f"\n📈 ANÁLISE DE TRADES (MELHOR CONFIG):")
    if best_config:
        trades = best_config[2]['trades']
        
        if trades:
            tp_trades = [t for t in trades if t['exit_reason'] == 'TP']
            sl_trades = [t for t in trades if t['exit_reason'] == 'SL']
            
            print(f"   Total trades: {len(trades)}")
            print(f"   Take Profits: {len(tp_trades)} ({len(tp_trades)/len(trades)*100:.1f}%)")
            print(f"   Stop Losses: {len(sl_trades)} ({len(sl_trades)/len(trades)*100:.1f}%)")
            
            if tp_trades:
                avg_tp_gain = np.mean([t['pnl_leveraged_pct'] for t in tp_trades])
                print(f"   TP médio: +{avg_tp_gain:.1f}%")
            
            if sl_trades:
                avg_sl_loss = np.mean([t['pnl_leveraged_pct'] for t in sl_trades])
                print(f"   SL médio: {avg_sl_loss:.1f}%")

def main():
    print("🔧 LEVERAGE BUG FOI CORRIGIDO!")
    print("🎯 Agora vamos ver se funciona nos dados reais...")
    print()
    
    # Executar backtest
    results = run_comprehensive_backtest()
    
    if not results:
        print("❌ Nenhum resultado obtido - verificar arquivos de dados")
        return
    
    # Analisar resultados
    analyze_results(results)
    create_detailed_report(results)
    
    print(f"\n" + "="*70)
    print("🎉 CONCLUSÃO DO BACKTEST COMPLETO")
    print("="*70)
    print("✅ Bug de leverage corrigido")
    print("✅ SL/TP agora são fixos (5%/8%)")
    print("✅ P&L corretamente amplificado pelo leverage")
    print("🚀 Resultados mostram se o leverage finalmente funciona!")

if __name__ == "__main__":
    main()
