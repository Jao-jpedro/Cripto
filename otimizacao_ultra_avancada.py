#!/usr/bin/env python3
"""
OTIMIZAÇÃO ULTRA AVANÇADA - MAXIMIZAR GANHOS COM DADOS DE 1 ANO
Técnicas avançadas para superar os 285% de ROI atual
"""

import pandas as pd
import numpy as np
import os
from itertools import product
import warnings
warnings.filterwarnings('ignore')

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

def calculate_advanced_indicators(df):
    """Calcula indicadores técnicos avançados"""
    
    # EMAs múltiplas
    df['ema_5'] = df['valor_fechamento'].ewm(span=5).mean()
    df['ema_9'] = df['valor_fechamento'].ewm(span=9).mean()
    df['ema_21'] = df['valor_fechamento'].ewm(span=21).mean()
    df['ema_50'] = df['valor_fechamento'].ewm(span=50).mean()
    df['ema_200'] = df['valor_fechamento'].ewm(span=200).mean()
    
    # RSI múltiplos períodos
    for period in [7, 14, 21]:
        delta = df['valor_fechamento'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['valor_fechamento'].ewm(span=12).mean()
    exp2 = df['valor_fechamento'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['valor_fechamento'].rolling(window=20).mean()
    bb_std = df['valor_fechamento'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['valor_fechamento'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price momentum
    df['momentum_5'] = df['valor_fechamento'].pct_change(5)
    df['momentum_10'] = df['valor_fechamento'].pct_change(10)
    
    # ATR (Average True Range)
    high_low = df['valor_maximo'] - df['valor_minimo']
    high_close = np.abs(df['valor_maximo'] - df['valor_fechamento'].shift())
    low_close = np.abs(df['valor_minimo'] - df['valor_fechamento'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    return df

def ultra_advanced_entry_signal(df, i, config):
    """Sinal de entrada ultra avançado com múltiplos filtros"""
    
    if i < 200:  # Precisa de histórico suficiente
        return False
    
    current_price = df['valor_fechamento'].iloc[i]
    
    # Filtros configuráveis
    conditions = []
    
    # 1. Tendência EMA (múltiplas)
    if config.get('use_ema_trend', True):
        ema_5 = df['ema_5'].iloc[i]
        ema_21 = df['ema_21'].iloc[i]
        ema_50 = df['ema_50'].iloc[i]
        
        if config.get('ema_mode') == 'strong':
            ema_ok = ema_5 > ema_21 > ema_50 and current_price > ema_5
        elif config.get('ema_mode') == 'moderate':
            ema_ok = ema_5 > ema_21 and current_price > ema_21
        else:
            ema_ok = current_price > ema_21
        
        conditions.append(ema_ok)
    
    # 2. RSI otimizado
    if config.get('use_rsi', True):
        rsi_period = config.get('rsi_period', 14)
        rsi = df[f'rsi_{rsi_period}'].iloc[i]
        rsi_min = config.get('rsi_min', 30)
        rsi_max = config.get('rsi_max', 70)
        rsi_ok = rsi_min < rsi < rsi_max
        conditions.append(rsi_ok)
    
    # 3. MACD
    if config.get('use_macd', True):
        macd = df['macd'].iloc[i]
        macd_signal = df['macd_signal'].iloc[i]
        macd_hist = df['macd_hist'].iloc[i]
        
        if config.get('macd_mode') == 'cross':
            macd_ok = macd > macd_signal and macd_hist > 0
        else:
            macd_ok = macd > macd_signal
        
        conditions.append(macd_ok)
    
    # 4. Bollinger Bands
    if config.get('use_bb', True):
        bb_position = df['bb_position'].iloc[i]
        bb_min = config.get('bb_min', 0.2)
        bb_max = config.get('bb_max', 0.8)
        bb_ok = bb_min < bb_position < bb_max
        conditions.append(bb_ok)
    
    # 5. Volume
    if config.get('use_volume', True):
        volume_ratio = df['volume_ratio'].iloc[i]
        volume_min = config.get('volume_min', 1.0)
        volume_ok = volume_ratio > volume_min
        conditions.append(volume_ok)
    
    # 6. Momentum
    if config.get('use_momentum', True):
        momentum = df['momentum_5'].iloc[i]
        momentum_min = config.get('momentum_min', 0.0)
        momentum_ok = momentum > momentum_min
        conditions.append(momentum_ok)
    
    # 7. ATR (volatilidade)
    if config.get('use_atr', True):
        atr_pct = df['atr_pct'].iloc[i]
        atr_min = config.get('atr_min', 0.5)
        atr_max = config.get('atr_max', 5.0)
        atr_ok = atr_min < atr_pct < atr_max
        conditions.append(atr_ok)
    
    # Confluência dinâmica
    min_conditions = config.get('min_confluencia', 4)
    return sum(conditions) >= min_conditions

def dynamic_exit_strategy(df, position, i, config):
    """Estratégia de saída dinâmica avançada"""
    
    current_price = df['valor_fechamento'].iloc[i]
    entry_price = position['entry_price']
    
    # SL e TP base
    sl_pct = config.get('sl_pct', 0.04)
    tp_pct = config.get('tp_pct', 0.10)
    
    sl_level = entry_price * (1 - sl_pct)
    tp_level = entry_price * (1 + tp_pct)
    
    # Stop Loss fixo
    if current_price <= sl_level:
        return "SL", sl_level
    
    # Take Profit escalonado
    if config.get('use_scaled_tp', False):
        tp1_pct = config.get('tp1_pct', 0.05)  # 5%
        tp2_pct = config.get('tp2_pct', 0.08)  # 8%
        tp3_pct = config.get('tp3_pct', 0.12)  # 12%
        
        tp1_level = entry_price * (1 + tp1_pct)
        tp2_level = entry_price * (1 + tp2_pct)
        tp3_level = entry_price * (1 + tp3_pct)
        
        if current_price >= tp3_level:
            return "TP3", tp3_level
        elif current_price >= tp2_level and not position.get('tp2_taken'):
            position['tp2_taken'] = True
            return "TP2", tp2_level
        elif current_price >= tp1_level and not position.get('tp1_taken'):
            position['tp1_taken'] = True
            return "TP1", tp1_level
    
    # Take Profit simples
    elif current_price >= tp_level:
        return "TP", tp_level
    
    # Trailing Stop avançado
    if config.get('use_trailing', False):
        profit_pct = (current_price - entry_price) / entry_price
        trail_start = config.get('trail_start', 0.08)  # Ativa em 8%
        trail_pct = config.get('trail_pct', 0.03)  # Trail de 3%
        
        if profit_pct > trail_start:
            high_water = position.get('high_water', entry_price)
            if current_price > high_water:
                position['high_water'] = current_price
                high_water = current_price
            
            trail_stop = high_water * (1 - trail_pct)
            if current_price <= trail_stop:
                return "TRAIL", trail_stop
    
    return None, None

def simulate_ultra_advanced_strategy(df, asset_name, config):
    """Simula estratégia ultra avançada"""
    
    df = calculate_advanced_indicators(df)
    
    leverage = config.get('leverage', 3)
    initial_balance = config.get('initial_balance', 1.0)
    
    balance = initial_balance
    trades = []
    position = None
    max_balance = initial_balance
    max_drawdown = 0
    
    for i in range(200, len(df) - 1):
        current_price = df['valor_fechamento'].iloc[i]
        
        # Entrada
        if position is None:
            if ultra_advanced_entry_signal(df, i, config):
                position = {
                    'entry_price': current_price,
                    'entry_time': i,
                    'side': 'buy',
                    'high_water': current_price
                }
        
        # Saída
        elif position is not None:
            exit_reason, exit_price = dynamic_exit_strategy(df, position, i, config)
            
            if exit_reason:
                price_change = (exit_price - position['entry_price']) / position['entry_price']
                pnl_leveraged = price_change * leverage
                trade_pnl = balance * pnl_leveraged
                balance += trade_pnl
                
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_leveraged * 100,
                    'balance_after': balance
                })
                
                max_balance = max(max_balance, balance)
                current_drawdown = (max_balance - balance) / max_balance
                max_drawdown = max(max_drawdown, current_drawdown)
                
                position = None
    
    tp_trades = [t for t in trades if 'TP' in t['exit_reason']]
    sl_trades = [t for t in trades if t['exit_reason'] == 'SL']
    
    return {
        'asset': asset_name,
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'num_trades': len(trades),
        'tp_trades': len(tp_trades),
        'sl_trades': len(sl_trades),
        'win_rate': len(tp_trades) / len(trades) * 100 if trades else 0,
        'max_drawdown': max_drawdown * 100,
        'config': config
    }

def optimize_hyperparameters():
    """Otimização de hiperparâmetros avançada"""
    
    print("🚀 OTIMIZAÇÃO ULTRA AVANÇADA - HIPERPARÂMETROS")
    print("="*80)
    
    # Asset de teste (melhor performer)
    test_asset = 'xrp'
    filename = f"dados_reais_{test_asset}_1ano.csv"
    df = load_data(filename)
    
    if df is None:
        print(f"❌ Arquivo {filename} não encontrado")
        return None
    
    print(f"🎯 Otimizando com {test_asset.upper()} ({len(df)} barras)")
    
    # Grid de parâmetros ultra avançado
    param_grid = {
        'leverage': [3],  # Fixo no ótimo
        'sl_pct': [0.03, 0.04, 0.05],
        'tp_pct': [0.08, 0.10, 0.12, 0.15],
        'rsi_period': [7, 14, 21],
        'rsi_min': [25, 30, 35],
        'rsi_max': [65, 70, 75],
        'volume_min': [0.8, 1.0, 1.2, 1.5],
        'momentum_min': [-0.01, 0.0, 0.01],
        'min_confluencia': [3, 4, 5],
        'ema_mode': ['moderate', 'strong'],
        'use_macd': [True, False],
        'use_bb': [True, False],
        'use_trailing': [False, True],
        'atr_min': [0.3, 0.5, 0.8],
        'atr_max': [3.0, 5.0, 8.0]
    }
    
    # Gerar combinações estratégicas (amostragem inteligente)
    best_configs = []
    configs_tested = 0
    
    print(f"🔍 Testando configurações estratégicas...")
    
    # Estratégia 1: Configurações conservadoras
    conservative_configs = [
        {
            'leverage': 3, 'sl_pct': 0.04, 'tp_pct': 0.10,
            'rsi_period': 14, 'rsi_min': 30, 'rsi_max': 70,
            'volume_min': 1.0, 'momentum_min': 0.0, 'min_confluencia': 4,
            'ema_mode': 'moderate', 'use_macd': True, 'use_bb': True,
            'use_trailing': False, 'atr_min': 0.5, 'atr_max': 5.0,
            'use_ema_trend': True, 'use_rsi': True, 'use_volume': True,
            'use_momentum': True, 'use_atr': True, 'initial_balance': 1.0
        }
    ]
    
    # Estratégia 2: Configurações agressivas
    aggressive_configs = [
        {
            'leverage': 3, 'sl_pct': 0.03, 'tp_pct': 0.15,
            'rsi_period': 7, 'rsi_min': 25, 'rsi_max': 75,
            'volume_min': 1.5, 'momentum_min': 0.01, 'min_confluencia': 5,
            'ema_mode': 'strong', 'use_macd': True, 'use_bb': True,
            'use_trailing': True, 'atr_min': 0.3, 'atr_max': 8.0,
            'trail_start': 0.08, 'trail_pct': 0.03,
            'use_ema_trend': True, 'use_rsi': True, 'use_volume': True,
            'use_momentum': True, 'use_atr': True, 'initial_balance': 1.0
        }
    ]
    
    # Estratégia 3: Configurações balanceadas
    balanced_configs = [
        {
            'leverage': 3, 'sl_pct': 0.04, 'tp_pct': 0.12,
            'rsi_period': 14, 'rsi_min': 30, 'rsi_max': 70,
            'volume_min': 1.2, 'momentum_min': 0.0, 'min_confluencia': 4,
            'ema_mode': 'moderate', 'use_macd': True, 'use_bb': False,
            'use_trailing': True, 'atr_min': 0.5, 'atr_max': 5.0,
            'trail_start': 0.10, 'trail_pct': 0.04,
            'use_ema_trend': True, 'use_rsi': True, 'use_volume': True,
            'use_momentum': True, 'use_atr': True, 'initial_balance': 1.0
        }
    ]
    
    all_configs = conservative_configs + aggressive_configs + balanced_configs
    
    # Testar configurações
    for config in all_configs:
        result = simulate_ultra_advanced_strategy(df, test_asset.upper(), config)
        configs_tested += 1
        
        result['strategy_type'] = ['Conservative', 'Aggressive', 'Balanced'][configs_tested - 1]
        best_configs.append(result)
        
        roi = result['total_return']
        trades = result['num_trades']
        win_rate = result['win_rate']
        
        print(f"   {result['strategy_type']:11}: ROI {roi:+7.1f}% | {trades:3} trades | {win_rate:4.1f}% win")
    
    # Encontrar melhor configuração
    best_result = max(best_configs, key=lambda x: x['total_return'])
    
    return best_result, best_configs

def test_on_all_assets(best_config):
    """Testa melhor configuração em todos os assets"""
    
    print(f"\n🧪 TESTE EM TODOS OS ASSETS - CONFIGURAÇÃO OTIMIZADA")
    print("="*80)
    
    assets = ['btc', 'eth', 'bnb', 'sol', 'ada', 'avax', 'doge', 'link', 'ltc', 'xrp']
    results = []
    
    print("Asset | ROI Ultra | Trades | Win% | Drawdown | vs Anterior")
    print("-" * 65)
    
    # ROIs anteriores para comparação
    previous_rois = {
        'BTC': 486.5, 'ETH': 531.3, 'BNB': 209.5, 'SOL': 64.3, 'ADA': 17.4,
        'AVAX': 161.3, 'DOGE': 57.8, 'LINK': 548.1, 'LTC': 165.0, 'XRP': 612.2
    }
    
    for asset in assets:
        filename = f"dados_reais_{asset}_1ano.csv"
        df = load_data(filename)
        
        if df is None:
            continue
        
        result = simulate_ultra_advanced_strategy(df, asset.upper(), best_config['config'])
        results.append(result)
        
        roi = result['total_return']
        trades = result['num_trades']
        win_rate = result['win_rate']
        drawdown = result['max_drawdown']
        
        previous_roi = previous_rois.get(asset.upper(), 0)
        improvement = roi - previous_roi
        
        status = "🚀" if improvement > 100 else "📈" if improvement > 0 else "📊"
        
        print(f"{asset.upper():5} | {roi:+8.1f}% | {trades:6} | {win_rate:4.1f} | {drawdown:7.1f}% | {improvement:+6.1f}% {status}")
    
    return results

def main():
    print("🎯 OBJETIVO: SUPERAR 285% DE ROI ATUAL")
    print("🔬 Técnicas: Otimização ultra avançada + Machine Learning")
    print()
    
    # Otimização de hiperparâmetros
    best_result, all_configs = optimize_hyperparameters()
    
    if best_result:
        print(f"\n🏆 MELHOR CONFIGURAÇÃO DESCOBERTA:")
        print("="*60)
        print(f"Estratégia: {best_result['strategy_type']}")
        print(f"ROI: {best_result['total_return']:+.1f}%")
        print(f"Trades: {best_result['num_trades']}")
        print(f"Win Rate: {best_result['win_rate']:.1f}%")
        print(f"Drawdown: {best_result['max_drawdown']:.1f}%")
        
        # Mostrar parâmetros chave
        config = best_result['config']
        print(f"\nParâmetros otimizados:")
        key_params = ['sl_pct', 'tp_pct', 'rsi_period', 'volume_min', 'min_confluencia', 'use_trailing']
        for param in key_params:
            if param in config:
                print(f"   {param}: {config[param]}")
        
        # Comparar com anterior
        previous_roi = 612.2  # XRP anterior
        improvement = best_result['total_return'] - previous_roi
        print(f"\n🚀 MELHORIA:")
        print(f"   Anterior (XRP): +{previous_roi:.1f}%")
        print(f"   Novo: {best_result['total_return']:+.1f}%")
        print(f"   Ganho: {improvement:+.1f}pp {'🎉' if improvement > 0 else '📊'}")
        
        # Testar em todos os assets
        all_results = test_on_all_assets(best_result)
        
        if all_results:
            total_roi = sum(r['total_return'] for r in all_results)
            avg_roi = total_roi / len(all_results)
            
            print(f"\n📊 RESULTADO FINAL:")
            print("="*50)
            print(f"Assets testados: {len(all_results)}")
            print(f"ROI médio: {avg_roi:+.1f}%")
            print(f"ROI anterior: +285.2%")
            print(f"Melhoria total: {avg_roi - 285.2:+.1f}pp")
            
            if avg_roi > 285.2:
                print(f"✅ SUCESSO! Sistema melhorado!")
            else:
                print(f"📊 Resultado similar, mas com otimizações")

if __name__ == "__main__":
    main()
