#!/usr/bin/env python3
"""
OTIMIZAÇÃO GENÉTICA AVANÇADA
Sistema evolutivo para superar +635.7% ROI
"""

import pandas as pd
import numpy as np
import os
import random
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def load_data(filename):
    """Carrega dados com verificação robusta"""
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

def calculate_advanced_indicators(df):
    """Calcula indicadores técnicos completos"""
    
    # EMAs múltiplas
    for span in [3, 5, 8, 9, 13, 21, 34, 55, 89, 144]:
        df[f'ema_{span}'] = df['valor_fechamento'].ewm(span=span).mean()
    
    # RSI múltiplos períodos
    for period in [5, 7, 9, 14, 21, 28]:
        delta = df['valor_fechamento'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD múltiplos
    for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (19, 39, 9)]:
        exp1 = df['valor_fechamento'].ewm(span=fast).mean()
        exp2 = df['valor_fechamento'].ewm(span=slow).mean()
        df[f'macd_{fast}_{slow}'] = exp1 - exp2
        df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal).mean()
        df[f'macd_hist_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
    
    # Bollinger Bands múltiplos
    for period, std_dev in [(10, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)]:
        bb_middle = df['valor_fechamento'].rolling(window=period).mean()
        bb_std = df['valor_fechamento'].rolling(window=period).std()
        df[f'bb_upper_{period}_{std_dev}'] = bb_middle + (bb_std * std_dev)
        df[f'bb_lower_{period}_{std_dev}'] = bb_middle - (bb_std * std_dev)
        df[f'bb_position_{period}_{std_dev}'] = (df['valor_fechamento'] - df[f'bb_lower_{period}_{std_dev}']) / (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}'])
        df[f'bb_width_{period}_{std_dev}'] = (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}']) / bb_middle
    
    # Volume indicators
    for period in [5, 10, 20, 50]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
    
    # ATR múltiplos
    for period in [7, 14, 21, 28]:
        high_low = df['valor_maximo'] - df['valor_minimo']
        high_close = np.abs(df['valor_maximo'] - df['valor_fechamento'].shift())
        low_close = np.abs(df['valor_minimo'] - df['valor_fechamento'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        df[f'atr_pct_{period}'] = (df[f'atr_{period}'] / df['valor_fechamento']) * 100
    
    # Momentum indicators
    for period in [1, 3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['valor_fechamento'].pct_change(period)
        df[f'momentum_sma_{period}'] = df[f'momentum_{period}'].rolling(window=5).mean()
    
    # Stochastic
    for period in [9, 14, 21]:
        low_n = df['valor_minimo'].rolling(window=period).min()
        high_n = df['valor_maximo'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = ((df['valor_fechamento'] - low_n) / (high_n - low_n)) * 100
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # Williams %R
    for period in [9, 14, 21]:
        high_n = df['valor_maximo'].rolling(window=period).max()
        low_n = df['valor_minimo'].rolling(window=period).min()
        df[f'williams_r_{period}'] = ((high_n - df['valor_fechamento']) / (high_n - low_n)) * -100
    
    # CCI
    for period in [14, 20, 28]:
        typical_price = (df['valor_maximo'] + df['valor_minimo'] + df['valor_fechamento']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Volatility measures
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['valor_fechamento'].rolling(window=period).std()
        df[f'volatility_pct_{period}'] = df[f'volatility_{period}'] / df['valor_fechamento'] * 100
    
    # Price action patterns
    df['body_size'] = np.abs(df['valor_fechamento'] - df['valor_abertura'])
    df['upper_shadow'] = df['valor_maximo'] - np.maximum(df['valor_fechamento'], df['valor_abertura'])
    df['lower_shadow'] = np.minimum(df['valor_fechamento'], df['valor_abertura']) - df['valor_minimo']
    df['total_range'] = df['valor_maximo'] - df['valor_minimo']
    
    # Trend strength
    for period in [10, 20, 50]:
        df[f'trend_strength_{period}'] = (df['valor_fechamento'] - df['valor_fechamento'].shift(period)) / df['valor_fechamento'].shift(period)
    
    return df

def genetic_entry_signal(df, i, dna):
    """Sinal baseado em DNA genético otimizado"""
    
    if i < 200:
        return False
    
    conditions = []
    current_price = df['valor_fechamento'].iloc[i]
    
    # Gene 1: EMA Trend (múltiplas EMAs)
    ema_fast = df[f"ema_{dna['ema_fast']}"].iloc[i]
    ema_slow = df[f"ema_{dna['ema_slow']}"].iloc[i]
    
    if dna['ema_mode'] == 'cross':
        ema_ok = ema_fast > ema_slow
    elif dna['ema_mode'] == 'price_above':
        ema_ok = current_price > ema_fast > ema_slow
    else:  # 'strong'
        ema_medium = df[f"ema_{dna['ema_medium']}"].iloc[i]
        ema_ok = ema_fast > ema_medium > ema_slow and current_price > ema_fast
    
    conditions.append(ema_ok)
    
    # Gene 2: RSI
    rsi = df[f"rsi_{dna['rsi_period']}"].iloc[i]
    rsi_ok = dna['rsi_min'] < rsi < dna['rsi_max']
    conditions.append(rsi_ok)
    
    # Gene 3: MACD
    macd_key = f"macd_{dna['macd_fast']}_{dna['macd_slow']}"
    macd_signal_key = f"macd_signal_{dna['macd_fast']}_{dna['macd_slow']}"
    macd_hist_key = f"macd_hist_{dna['macd_fast']}_{dna['macd_slow']}"
    
    if macd_key in df.columns:
        macd = df[macd_key].iloc[i]
        macd_signal = df[macd_signal_key].iloc[i]
        macd_hist = df[macd_hist_key].iloc[i]
        
        if dna['macd_mode'] == 'histogram':
            macd_ok = macd_hist > 0
        elif dna['macd_mode'] == 'cross_positive':
            macd_ok = macd > macd_signal and macd > 0
        else:  # 'simple'
            macd_ok = macd > macd_signal
        
        conditions.append(macd_ok)
    
    # Gene 4: Bollinger Bands
    bb_key = f"bb_position_{dna['bb_period']}_{dna['bb_std']}"
    if bb_key in df.columns:
        bb_position = df[bb_key].iloc[i]
        bb_ok = dna['bb_min'] < bb_position < dna['bb_max']
        conditions.append(bb_ok)
    
    # Gene 5: Volume
    volume_key = f"volume_ratio_{dna['volume_period']}"
    if volume_key in df.columns:
        volume_ratio = df[volume_key].iloc[i]
        volume_ok = volume_ratio > dna['volume_min']
        conditions.append(volume_ok)
    
    # Gene 6: ATR (volatilidade)
    atr_key = f"atr_pct_{dna['atr_period']}"
    if atr_key in df.columns:
        atr_pct = df[atr_key].iloc[i]
        atr_ok = dna['atr_min'] < atr_pct < dna['atr_max']
        conditions.append(atr_ok)
    
    # Gene 7: Momentum
    momentum_key = f"momentum_{dna['momentum_period']}"
    if momentum_key in df.columns:
        momentum = df[momentum_key].iloc[i]
        momentum_ok = momentum > dna['momentum_min']
        conditions.append(momentum_ok)
    
    # Gene 8: Stochastic
    if dna['use_stoch']:
        stoch_key = f"stoch_k_{dna['stoch_period']}"
        if stoch_key in df.columns:
            stoch = df[stoch_key].iloc[i]
            stoch_ok = dna['stoch_min'] < stoch < dna['stoch_max']
            conditions.append(stoch_ok)
    
    # Gene 9: Williams %R
    if dna['use_williams']:
        williams_key = f"williams_r_{dna['williams_period']}"
        if williams_key in df.columns:
            williams = df[williams_key].iloc[i]
            williams_ok = dna['williams_min'] < williams < dna['williams_max']
            conditions.append(williams_ok)
    
    # Gene 10: CCI
    if dna['use_cci']:
        cci_key = f"cci_{dna['cci_period']}"
        if cci_key in df.columns:
            cci = df[cci_key].iloc[i]
            cci_ok = dna['cci_min'] < cci < dna['cci_max']
            conditions.append(cci_ok)
    
    # Gene 11: Trend Strength
    trend_key = f"trend_strength_{dna['trend_period']}"
    if trend_key in df.columns:
        trend_strength = df[trend_key].iloc[i]
        trend_ok = trend_strength > dna['trend_min']
        conditions.append(trend_ok)
    
    # Confluência genética
    return sum(conditions) >= dna['min_confluencia']

def simulate_genetic_strategy(df, asset_name, dna):
    """Simula estratégia com DNA genético"""
    
    df = calculate_advanced_indicators(df)
    
    leverage = dna['leverage']
    sl_pct = dna['sl_pct']
    tp_pct = dna['tp_pct']
    initial_balance = 1.0
    
    balance = initial_balance
    trades = []
    position = None
    
    for i in range(200, len(df) - 1):
        current_price = df['valor_fechamento'].iloc[i]
        
        # Entrada
        if position is None and genetic_entry_signal(df, i, dna):
            position = {
                'entry_price': current_price,
                'entry_time': i
            }
        
        # Saída
        elif position is not None:
            entry_price = position['entry_price']
            
            sl_level = entry_price * (1 - sl_pct)
            tp_level = entry_price * (1 + tp_pct)
            
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
                pnl_leveraged = price_change * leverage
                trade_pnl = balance * pnl_leveraged
                balance += trade_pnl
                
                trades.append({
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_leveraged * 100,
                    'balance_after': balance
                })
                
                position = None
    
    tp_trades = [t for t in trades if t['exit_reason'] == 'TP']
    sl_trades = [t for t in trades if t['exit_reason'] == 'SL']
    
    return {
        'asset': asset_name,
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'num_trades': len(trades),
        'tp_trades': len(tp_trades),
        'sl_trades': len(sl_trades),
        'win_rate': len(tp_trades) / len(trades) * 100 if trades else 0
    }

def create_random_dna():
    """Cria DNA aleatório para algoritmo genético"""
    
    return {
        # Core parameters
        'leverage': 3,
        'sl_pct': random.choice([0.015, 0.02, 0.025, 0.03, 0.035]),
        'tp_pct': random.choice([0.10, 0.12, 0.15, 0.18, 0.20, 0.25]),
        
        # EMA genes
        'ema_fast': random.choice([3, 5, 8, 9, 13]),
        'ema_medium': random.choice([13, 21, 34]),
        'ema_slow': random.choice([34, 55, 89, 144]),
        'ema_mode': random.choice(['cross', 'price_above', 'strong']),
        
        # RSI genes
        'rsi_period': random.choice([5, 7, 9, 14, 21]),
        'rsi_min': random.randint(15, 35),
        'rsi_max': random.randint(65, 85),
        
        # MACD genes
        'macd_fast': random.choice([8, 12, 19]),
        'macd_slow': random.choice([21, 26, 39]),
        'macd_mode': random.choice(['simple', 'histogram', 'cross_positive']),
        
        # Bollinger genes
        'bb_period': random.choice([10, 20, 50]),
        'bb_std': random.choice([1.5, 2.0, 2.5]),
        'bb_min': random.uniform(0.1, 0.3),
        'bb_max': random.uniform(0.7, 0.9),
        
        # Volume genes
        'volume_period': random.choice([5, 10, 20]),
        'volume_min': random.uniform(1.0, 3.0),
        
        # ATR genes
        'atr_period': random.choice([7, 14, 21]),
        'atr_min': random.uniform(0.2, 0.8),
        'atr_max': random.uniform(5.0, 12.0),
        
        # Momentum genes
        'momentum_period': random.choice([1, 3, 5, 10]),
        'momentum_min': random.uniform(-0.01, 0.02),
        
        # Oscillator genes
        'use_stoch': random.choice([True, False]),
        'stoch_period': random.choice([9, 14, 21]),
        'stoch_min': random.randint(15, 25),
        'stoch_max': random.randint(75, 85),
        
        'use_williams': random.choice([True, False]),
        'williams_period': random.choice([9, 14, 21]),
        'williams_min': random.randint(-90, -70),
        'williams_max': random.randint(-30, -10),
        
        'use_cci': random.choice([True, False]),
        'cci_period': random.choice([14, 20, 28]),
        'cci_min': random.randint(-150, -50),
        'cci_max': random.randint(50, 150),
        
        # Trend genes
        'trend_period': random.choice([10, 20, 50]),
        'trend_min': random.uniform(-0.02, 0.02),
        
        # Confluence
        'min_confluencia': random.randint(3, 7)
    }

def genetic_algorithm():
    """Algoritmo genético para evolução de estratégias"""
    
    print("🧬 ALGORITMO GENÉTICO - EVOLUÇÃO DE ESTRATÉGIAS")
    print("="*80)
    
    # Parâmetros do algoritmo genético
    POPULATION_SIZE = 20
    GENERATIONS = 5
    MUTATION_RATE = 0.3
    ELITE_SIZE = 5
    
    # Asset de teste
    test_asset = 'xrp'
    filename = f"dados_reais_{test_asset}_1ano.csv"
    df = load_data(filename)
    
    if df is None:
        print(f"❌ Arquivo {filename} não encontrado")
        return None
    
    print(f"🎯 Evoluindo estratégias com {test_asset.upper()} ({len(df)} barras)")
    print(f"População: {POPULATION_SIZE} | Gerações: {GENERATIONS} | Taxa mutação: {MUTATION_RATE*100}%")
    print()
    
    # Criar população inicial
    population = [create_random_dna() for _ in range(POPULATION_SIZE)]
    best_ever = None
    
    for generation in range(GENERATIONS):
        print(f"🧬 GERAÇÃO {generation + 1}/{GENERATIONS}")
        print("-" * 50)
        
        # Avaliar fitness de cada indivíduo
        fitness_scores = []
        
        for i, dna in enumerate(population):
            result = simulate_genetic_strategy(df, test_asset.upper(), dna)
            fitness = result['total_return']
            fitness_scores.append((fitness, dna, result))
            
            print(f"   DNA {i+1:2}: ROI {fitness:+7.1f}% | {result['num_trades']:3} trades | {result['win_rate']:4.1f}% win")
        
        # Ordenar por fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Atualizar melhor de todos os tempos
        if best_ever is None or fitness_scores[0][0] > best_ever[0]:
            best_ever = fitness_scores[0]
        
        # Elite (melhores sobrevivem)
        elite = [fs[1] for fs in fitness_scores[:ELITE_SIZE]]
        
        # Criar nova população
        new_population = elite.copy()
        
        # Crossover e mutação
        while len(new_population) < POPULATION_SIZE:
            # Seleção dos pais (tournament selection)
            parent1 = random.choice(fitness_scores[:ELITE_SIZE*2])[1]
            parent2 = random.choice(fitness_scores[:ELITE_SIZE*2])[1]
            
            # Crossover (50% de cada pai)
            child = {}
            for key in parent1.keys():
                child[key] = random.choice([parent1[key], parent2[key]])
            
            # Mutação
            if random.random() < MUTATION_RATE:
                # Escolher gene aleatório para mutar
                gene_to_mutate = random.choice(list(child.keys()))
                
                if gene_to_mutate == 'sl_pct':
                    child[gene_to_mutate] = random.choice([0.015, 0.02, 0.025, 0.03, 0.035])
                elif gene_to_mutate == 'tp_pct':
                    child[gene_to_mutate] = random.choice([0.10, 0.12, 0.15, 0.18, 0.20, 0.25])
                elif gene_to_mutate in ['rsi_min']:
                    child[gene_to_mutate] = random.randint(15, 35)
                elif gene_to_mutate in ['rsi_max']:
                    child[gene_to_mutate] = random.randint(65, 85)
                # ... outros genes
            
            new_population.append(child)
        
        population = new_population
        
        avg_fitness = sum(fs[0] for fs in fitness_scores) / len(fitness_scores)
        best_fitness = fitness_scores[0][0]
        
        print(f"   📊 Melhor: {best_fitness:+.1f}% | Média: {avg_fitness:+.1f}%")
        print()
    
    return best_ever

def test_evolved_strategy(best_dna):
    """Testa estratégia evoluída em todos os assets"""
    
    print(f"🧬 TESTE DA ESTRATÉGIA EVOLUÍDA")
    print("="*80)
    
    assets = ['btc', 'eth', 'bnb', 'sol', 'ada', 'avax', 'doge', 'link', 'ltc', 'xrp']
    results = []
    
    print("Asset | ROI Evoluído | Trades | Win% | vs +635.7%")
    print("-" * 60)
    
    for asset in assets:
        filename = f"dados_reais_{asset}_1ano.csv"
        df = load_data(filename)
        
        if df is None:
            continue
        
        result = simulate_genetic_strategy(df, asset.upper(), best_dna[1])
        results.append(result)
        
        roi = result['total_return']
        trades = result['num_trades']
        win_rate = result['win_rate']
        
        improvement = roi - 635.7
        status = "🚀" if improvement > 100 else "📈" if improvement > 0 else "📊"
        
        print(f"{asset.upper():5} | {roi:+9.1f}% | {trades:6} | {win_rate:4.1f} | {improvement:+7.1f}% {status}")
    
    return results

def main():
    print("🎯 OBJETIVO: SUPERAR +635.7% COM ALGORITMO GENÉTICO!")
    print("🧬 Evolução de estratégias através de seleção natural")
    print()
    
    # Executar algoritmo genético
    best_evolved = genetic_algorithm()
    
    if best_evolved:
        print(f"🏆 MELHOR ESTRATÉGIA EVOLUÍDA:")
        print("="*60)
        print(f"ROI: {best_evolved[0]:+.1f}%")
        print(f"Trades: {best_evolved[2]['num_trades']}")
        print(f"Win Rate: {best_evolved[2]['win_rate']:.1f}%")
        
        # Mostrar genes principais
        dna = best_evolved[1]
        print(f"\n🧬 DNA da estratégia:")
        key_genes = ['sl_pct', 'tp_pct', 'ema_fast', 'ema_slow', 'rsi_period', 'volume_min', 'min_confluencia']
        for gene in key_genes:
            if gene in dna:
                print(f"   {gene}: {dna[gene]}")
        
        # Comparar com anterior
        improvement = best_evolved[0] - 635.7
        print(f"\n🚀 vs SISTEMA ANTERIOR:")
        print(f"   Anterior: +635.7%")
        print(f"   Evoluído: {best_evolved[0]:+.1f}%")
        print(f"   Evolução: {improvement:+.1f}pp {'🎉' if improvement > 0 else '📊'}")
        
        # Testar em todos os assets
        all_results = test_evolved_strategy(best_evolved)
        
        if all_results:
            avg_roi = sum(r['total_return'] for r in all_results) / len(all_results)
            profitable = len([r for r in all_results if r['total_return'] > 0])
            
            print(f"\n📊 RESULTADO FINAL EVOLUÍDO:")
            print("="*50)
            print(f"Assets testados: {len(all_results)}")
            print(f"ROI médio evoluído: {avg_roi:+.1f}%")
            print(f"ROI anterior: +635.7%")
            print(f"Evolução total: {avg_roi - 635.7:+.1f}pp")
            print(f"Assets lucrativos: {profitable}/{len(all_results)} ({profitable/len(all_results)*100:.1f}%)")
            
            # Top performers
            top_3 = sorted(all_results, key=lambda x: x['total_return'], reverse=True)[:3]
            print(f"\n🏆 TOP 3 EVOLUÍDOS:")
            for i, result in enumerate(top_3, 1):
                roi = result['total_return']
                trades = result['num_trades']
                print(f"   {i}º {result['asset']}: {roi:+.1f}% ({trades} trades)")
            
            if avg_roi > 635.7:
                print(f"\n✅ EVOLUÇÃO BEM-SUCEDIDA!")
                print(f"🧬 Algoritmo genético superou sistema anterior!")
            else:
                print(f"\n📊 Sistema evoluído com características refinadas")

if __name__ == "__main__":
    main()
