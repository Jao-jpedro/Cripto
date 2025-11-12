#!/usr/bin/env python3
"""
Análise de Otimização de Estratégias de Saída
Baseado nos dados reais de 01/10/2025 a 11/11/2025

Objetivo: Encontrar estratégias de saída que maximizem ROI
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime

def analyze_exit_patterns(df, asset_name):
    """Analisa padrões de saída baseados em indicadores técnicos"""
    
    results = {
        'asset': asset_name,
        'total_candles': len(df)
    }
    
    # Simular diferentes estratégias de saída
    
    # 1. TRAILING STOP DINÂMICO baseado em ATR
    trailing_atr_results = simulate_trailing_atr_exit(df)
    results['trailing_atr'] = trailing_atr_results
    
    # 2. SAÍDA POR VOLUME (quando volume de venda > volume de compra significativamente)
    volume_exit_results = simulate_volume_based_exit(df)
    results['volume_exit'] = volume_exit_results
    
    # 3. SAÍDA POR RATIO BUY/SELL (quando ratio começa a cair consistentemente)
    ratio_exit_results = simulate_ratio_decline_exit(df)
    results['ratio_exit'] = ratio_exit_results
    
    # 4. BREAKEVEN + TRAILING (mover SL para breakeven após X%, depois trailing)
    breakeven_trailing_results = simulate_breakeven_trailing(df)
    results['breakeven_trailing'] = breakeven_trailing_results
    
    # 5. SAÍDA PARCIAL (scale out) - fechar 50% no primeiro TP, deixar resto correr
    partial_exit_results = simulate_partial_exits(df)
    results['partial_exit'] = partial_exit_results
    
    # 6. SAÍDA POR DIVERGÊNCIA EMA/PREÇO
    ema_divergence_results = simulate_ema_divergence_exit(df)
    results['ema_divergence'] = ema_divergence_results
    
    return results


def simulate_trailing_atr_exit(df):
    """
    Trailing Stop baseado em múltiplos de ATR
    Quando preço sobe, o stop sobe também (mas nunca desce)
    """
    strategies = {}
    
    for atr_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        trades_won = 0
        trades_lost = 0
        total_roi = 0
        
        # Simular trades LONG
        for i in range(50, len(df)):
            entry_price = df['close'].iloc[i-50]
            highest_price = entry_price
            trailing_stop = entry_price
            exit_price = None
            exit_reason = ""
            
            # Simular próximos 50 candles
            for j in range(i-49, min(i+50, len(df))):
                current_price = df['close'].iloc[j]
                current_atr = df['atr'].iloc[j]
                
                # Atualizar highest price e trailing stop
                if current_price > highest_price:
                    highest_price = current_price
                    new_stop = highest_price - (atr_mult * current_atr)
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop
                
                # Verificar se atingiu stop
                if current_price <= trailing_stop:
                    exit_price = current_price
                    exit_reason = f"Trailing ATR {atr_mult}x"
                    break
            
            if exit_price:
                roi = ((exit_price - entry_price) / entry_price) * 100
                total_roi += roi
                if roi > 0:
                    trades_won += 1
                else:
                    trades_lost += 1
        
        if (trades_won + trades_lost) > 0:
            strategies[f'atr_{atr_mult}x'] = {
                'trades_won': trades_won,
                'trades_lost': trades_lost,
                'win_rate': (trades_won / (trades_won + trades_lost)) * 100,
                'avg_roi': total_roi / (trades_won + trades_lost)
            }
    
    return strategies


def simulate_volume_based_exit(df):
    """
    Saída quando volume de venda supera volume de compra por X%
    Indica possível reversão de tendência
    """
    strategies = {}
    
    for threshold in [1.1, 1.2, 1.3, 1.5]:  # Venda X vezes maior que compra
        trades_won = 0
        trades_lost = 0
        total_roi = 0
        
        for i in range(50, len(df)):
            entry_price = df['close'].iloc[i-50]
            exit_price = None
            
            for j in range(i-49, min(i+50, len(df))):
                # Verificar se sell_volume > buy_volume * threshold
                if df['avg_sell_3'].iloc[j] > (df['avg_buy_3'].iloc[j] * threshold):
                    exit_price = df['close'].iloc[j]
                    break
            
            if exit_price:
                roi = ((exit_price - entry_price) / entry_price) * 100
                total_roi += roi
                if roi > 0:
                    trades_won += 1
                else:
                    trades_lost += 1
        
        if (trades_won + trades_lost) > 0:
            strategies[f'vol_threshold_{threshold}'] = {
                'trades_won': trades_won,
                'trades_lost': trades_lost,
                'win_rate': (trades_won / (trades_won + trades_lost)) * 100,
                'avg_roi': total_roi / (trades_won + trades_lost)
            }
    
    return strategies


def simulate_ratio_decline_exit(df):
    """
    Saída quando buy_sell_ratio cai por N candles consecutivos
    Indica enfraquecimento da pressão de compra
    """
    strategies = {}
    
    for consec_candles in [2, 3, 4, 5]:
        trades_won = 0
        trades_lost = 0
        total_roi = 0
        
        for i in range(50, len(df)):
            entry_price = df['close'].iloc[i-50]
            exit_price = None
            decline_count = 0
            
            for j in range(i-49, min(i+50, len(df))):
                # Verificar se ratio está caindo
                if df['ratio_trend'].iloc[j] == 'diminuindo':
                    decline_count += 1
                    if decline_count >= consec_candles:
                        exit_price = df['close'].iloc[j]
                        break
                else:
                    decline_count = 0
            
            if exit_price:
                roi = ((exit_price - entry_price) / entry_price) * 100
                total_roi += roi
                if roi > 0:
                    trades_won += 1
                else:
                    trades_lost += 1
        
        if (trades_won + trades_lost) > 0:
            strategies[f'ratio_decline_{consec_candles}_candles'] = {
                'trades_won': trades_won,
                'trades_lost': trades_lost,
                'win_rate': (trades_won / (trades_won + trades_lost)) * 100,
                'avg_roi': total_roi / (trades_won + trades_lost)
            }
    
    return strategies


def simulate_breakeven_trailing(df):
    """
    Combina breakeven + trailing stop
    1. Quando ROI atinge X%, move stop para breakeven (entrada)
    2. Quando ROI atinge Y%, ativa trailing stop
    """
    strategies = {}
    
    configs = [
        {'breakeven_pct': 2, 'trailing_pct': 5, 'trailing_distance': 1.5},
        {'breakeven_pct': 3, 'trailing_pct': 7, 'trailing_distance': 2.0},
        {'breakeven_pct': 5, 'trailing_pct': 10, 'trailing_distance': 2.5},
    ]
    
    for config in configs:
        be_pct = config['breakeven_pct']
        trail_pct = config['trailing_pct']
        trail_dist = config['trailing_distance']
        
        trades_won = 0
        trades_lost = 0
        total_roi = 0
        
        for i in range(50, len(df)):
            entry_price = df['close'].iloc[i-50]
            stop_price = entry_price * 0.80  # SL inicial em -20%
            highest_price = entry_price
            trailing_active = False
            exit_price = None
            
            for j in range(i-49, min(i+50, len(df))):
                current_price = df['close'].iloc[j]
                current_atr = df['atr'].iloc[j]
                current_roi = ((current_price - entry_price) / entry_price) * 100
                
                # Fase 1: Ativar breakeven quando atinge be_pct%
                if not trailing_active and current_roi >= be_pct:
                    stop_price = entry_price
                
                # Fase 2: Ativar trailing quando atinge trail_pct%
                if current_roi >= trail_pct:
                    trailing_active = True
                    if current_price > highest_price:
                        highest_price = current_price
                        stop_price = highest_price - (trail_dist * current_atr)
                
                # Verificar stop
                if current_price <= stop_price:
                    exit_price = current_price
                    break
            
            if exit_price:
                roi = ((exit_price - entry_price) / entry_price) * 100
                total_roi += roi
                if roi > 0:
                    trades_won += 1
                else:
                    trades_lost += 1
        
        if (trades_won + trades_lost) > 0:
            key = f'be{be_pct}_trail{trail_pct}_dist{trail_dist}'
            strategies[key] = {
                'trades_won': trades_won,
                'trades_lost': trades_lost,
                'win_rate': (trades_won / (trades_won + trades_lost)) * 100,
                'avg_roi': total_roi / (trades_won + trades_lost)
            }
    
    return strategies


def simulate_partial_exits(df):
    """
    Saídas parciais (scale out)
    Fechar 50% na primeira meta, deixar 50% com trailing
    """
    strategies = {}
    
    configs = [
        {'first_tp': 5, 'trailing_mult': 2.0},
        {'first_tp': 7, 'trailing_mult': 2.5},
        {'first_tp': 10, 'trailing_mult': 3.0},
    ]
    
    for config in configs:
        first_tp = config['first_tp']
        trail_mult = config['trailing_mult']
        
        total_roi = 0
        trades_count = 0
        
        for i in range(50, len(df)):
            entry_price = df['close'].iloc[i-50]
            first_tp_hit = False
            second_half_exit = None
            highest_after_tp = entry_price
            
            for j in range(i-49, min(i+50, len(df))):
                current_price = df['close'].iloc[j]
                current_atr = df['atr'].iloc[j]
                current_roi = ((current_price - entry_price) / entry_price) * 100
                
                # Primeira parte: TP em first_tp%
                if not first_tp_hit and current_roi >= first_tp:
                    first_tp_hit = True
                    highest_after_tp = current_price
                
                # Segunda parte: trailing após primeiro TP
                if first_tp_hit:
                    if current_price > highest_after_tp:
                        highest_after_tp = current_price
                    
                    trailing_stop = highest_after_tp - (trail_mult * current_atr)
                    if current_price <= trailing_stop:
                        second_half_exit = current_price
                        break
            
            if first_tp_hit:
                # ROI = 50% no primeiro TP + 50% na saída trailing
                first_half_roi = first_tp / 2  # 50% da posição
                if second_half_exit:
                    second_half_roi = (((second_half_exit - entry_price) / entry_price) * 100) / 2
                else:
                    second_half_roi = 0
                
                total_roi += (first_half_roi + second_half_roi)
                trades_count += 1
        
        if trades_count > 0:
            key = f'partial_tp{first_tp}_trail{trail_mult}'
            strategies[key] = {
                'trades_count': trades_count,
                'avg_roi': total_roi / trades_count
            }
    
    return strategies


def simulate_ema_divergence_exit(df):
    """
    Saída quando preço continua subindo mas EMA começa a achatar
    Indica possível topo de movimento
    """
    strategies = {}
    
    for grad_threshold in [-0.0001, -0.0002, -0.0003]:
        trades_won = 0
        trades_lost = 0
        total_roi = 0
        
        for i in range(50, len(df)):
            entry_price = df['close'].iloc[i-50]
            exit_price = None
            
            for j in range(i-48, min(i+50, len(df))):
                # Preço acima da EMA mas gradiente negativo
                if (df['close'].iloc[j] > df['ema_fast'].iloc[j] and 
                    df['ema_gradient'].iloc[j] < grad_threshold):
                    exit_price = df['close'].iloc[j]
                    break
            
            if exit_price:
                roi = ((exit_price - entry_price) / entry_price) * 100
                total_roi += roi
                if roi > 0:
                    trades_won += 1
                else:
                    trades_lost += 1
        
        if (trades_won + trades_lost) > 0:
            strategies[f'ema_div_{abs(grad_threshold):.6f}'] = {
                'trades_won': trades_won,
                'trades_lost': trades_lost,
                'win_rate': (trades_won / (trades_won + trades_lost)) * 100,
                'avg_roi': total_roi / (trades_won + trades_lost)
            }
    
    return strategies


def compare_strategies(all_results):
    """Compara todas as estratégias e ranqueia por ROI médio"""
    
    comparison = []
    
    for asset_result in all_results:
        asset = asset_result['asset']
        
        # Coletar todas as estratégias
        for strategy_type, strategies in asset_result.items():
            if strategy_type in ['asset', 'total_candles']:
                continue
            
            if isinstance(strategies, dict):
                for strategy_name, metrics in strategies.items():
                    if 'avg_roi' in metrics:
                        comparison.append({
                            'asset': asset,
                            'strategy_type': strategy_type,
                            'strategy_name': strategy_name,
                            'avg_roi': metrics['avg_roi'],
                            'win_rate': metrics.get('win_rate', 0),
                            'trades_won': metrics.get('trades_won', 0),
                            'trades_lost': metrics.get('trades_lost', 0),
                            'trades_count': metrics.get('trades_count', 0)
                        })
    
    df_comparison = pd.DataFrame(comparison)
    return df_comparison


def main():
    print("="*100)
    print("🎯 ANÁLISE DE OTIMIZAÇÃO DE ESTRATÉGIAS DE SAÍDA")
    print("="*100)
    print(f"Objetivo: Maximizar ROI através de estratégias inteligentes de saída")
    print(f"Período: 01/10/2025 a 11/11/2025 (41 dias)")
    print("="*100)
    
    # Carregar dados
    csv_files = glob.glob("data_*_15m_20251001_atual.csv")
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado!")
        return
    
    print(f"\n📁 Encontrados {len(csv_files)} arquivos CSV")
    
    all_results = []
    
    # Analisar cada ativo
    for idx, csv_file in enumerate(sorted(csv_files)[:5], 1):  # Primeiros 5 ativos para teste
        asset_name = csv_file.replace("data_", "").replace("_15m_20251001_atual.csv", "")
        
        print(f"\n[{idx}/5] Analisando {asset_name}...", end=" ")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Verificar colunas necessárias
            required_cols = ['close', 'atr', 'ema_fast', 'ema_gradient', 
                           'avg_buy_3', 'avg_sell_3', 'buy_sell_ratio', 'ratio_trend']
            
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Colunas faltando")
                continue
            
            results = analyze_exit_patterns(df, asset_name)
            all_results.append(results)
            print("✅")
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            continue
    
    if not all_results:
        print("\n❌ Nenhum resultado gerado!")
        return
    
    # Comparar estratégias
    print("\n" + "="*100)
    print("📊 COMPARAÇÃO DE ESTRATÉGIAS")
    print("="*100)
    
    df_comparison = compare_strategies(all_results)
    
    # Agrupar por estratégia e calcular médias
    strategy_summary = df_comparison.groupby(['strategy_type', 'strategy_name']).agg({
        'avg_roi': 'mean',
        'win_rate': 'mean',
        'trades_count': 'sum'
    }).reset_index()
    
    strategy_summary = strategy_summary.sort_values('avg_roi', ascending=False)
    
    print("\n🏆 TOP 10 ESTRATÉGIAS POR ROI MÉDIO:")
    print("-"*100)
    for idx, row in strategy_summary.head(10).iterrows():
        print(f"{row['strategy_type']:20s} | {row['strategy_name']:30s} | "
              f"ROI: {row['avg_roi']:+.2f}% | Win Rate: {row['win_rate']:.1f}%")
    
    print("\n" + "="*100)
    print("💡 RECOMENDAÇÕES PARA MAXIMIZAR ROI:")
    print("="*100)
    
    best_strategy = strategy_summary.iloc[0]
    
    print(f"\n1. 🥇 MELHOR ESTRATÉGIA GERAL:")
    print(f"   Tipo: {best_strategy['strategy_type']}")
    print(f"   Configuração: {best_strategy['strategy_name']}")
    print(f"   ROI Médio: {best_strategy['avg_roi']:+.2f}%")
    print(f"   Win Rate: {best_strategy['win_rate']:.1f}%")
    
    # Estratégias por categoria
    print(f"\n2. 📈 MELHORES POR CATEGORIA:")
    for strategy_type in strategy_summary['strategy_type'].unique():
        best_in_cat = strategy_summary[strategy_summary['strategy_type'] == strategy_type].iloc[0]
        print(f"\n   {strategy_type.upper()}:")
        print(f"   • {best_in_cat['strategy_name']}")
        print(f"   • ROI: {best_in_cat['avg_roi']:+.2f}% | Win Rate: {best_in_cat['win_rate']:.1f}%")
    
    # Comparação com TP/SL fixo atual
    print(f"\n3. 📊 COMPARAÇÃO COM ESTRATÉGIA ATUAL:")
    print(f"   ATUAL (tradingv4.py):")
    print(f"   • SL fixo: -20% da margem (≈ -4% no preço com 5x leverage)")
    print(f"   • TP fixo: +50% da margem (≈ +10% no preço com 5x leverage)")
    print(f"   • Problema: Não se adapta às condições de mercado")
    print(f"   • Problema: TP muito próximo, deixa dinheiro na mesa")
    print(f"   • Problema: SL muito distante, perde muito quando erra")
    
    print(f"\n4. 🎯 IMPLEMENTAÇÃO RECOMENDADA:")
    print(f"   Combinar múltiplas estratégias:")
    print(f"   • Breakeven: Mover SL para entrada após +3% ROI")
    print(f"   • Saída Parcial: Fechar 30% em +7% ROI")
    print(f"   • Trailing ATR: Deixar 70% com trailing 2x ATR")
    print(f"   • Stop de Emergência: Volume venda > 1.5x compra por 3 candles")
    print(f"   • Stop de Tendência: Ratio caindo por 4+ candles consecutivos")
    
    # Salvar resultados
    output_file = "analise_estrategias_saida_otimizada.csv"
    strategy_summary.to_csv(output_file, index=False)
    print(f"\n💾 Resultados salvos em: {output_file}")
    
    print("\n" + "="*100)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("="*100)


if __name__ == "__main__":
    main()
