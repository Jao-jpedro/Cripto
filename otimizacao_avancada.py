#!/usr/bin/env python3
"""
OTIMIZAÇÃO AVANÇADA - MAXIMIZAR GANHOS ALÉM DOS 201.8%
Vamos testar múltiplas otimizações para superar o resultado atual
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import itertools

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

def calculate_technical_indicators(df):
    """Calcula indicadores técnicos para melhor entrada"""
    
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
    
    # Bollinger Bands
    df['bb_middle'] = df['valor_fechamento'].rolling(window=20).mean()
    bb_std = df['valor_fechamento'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    return df

def advanced_entry_logic(df, i, config):
    """Lógica avançada de entrada com múltiplos filtros"""
    
    if i < 50:  # Precisa de histórico para indicadores
        return False
    
    current_price = df['valor_fechamento'].iloc[i]
    
    # Filtros básicos
    ema_9 = df['ema_9'].iloc[i]
    ema_21 = df['ema_21'].iloc[i]
    ema_50 = df['ema_50'].iloc[i]
    rsi = df['rsi'].iloc[i]
    volume = df['volume'].iloc[i]
    volume_ma = df['volume_ma'].iloc[i]
    
    # Condições otimizadas
    conditions = []
    
    # 1. Tendência de alta (EMAs alinhadas)
    if config.get('use_ema_trend', True):
        ema_aligned = ema_9 > ema_21 > ema_50
        conditions.append(ema_aligned)
    
    # 2. Preço acima das EMAs
    if config.get('use_price_above_ema', True):
        price_above_ema = current_price > ema_9
        conditions.append(price_above_ema)
    
    # 3. RSI não sobrecomprado
    if config.get('use_rsi_filter', True):
        rsi_ok = 30 < rsi < 70  # Evita extremos
        conditions.append(rsi_ok)
    
    # 4. Volume acima da média
    if config.get('use_volume_filter', True):
        volume_ok = volume > volume_ma * config.get('volume_multiplier', 1.2)
        conditions.append(volume_ok)
    
    # 5. Momentum positivo
    if config.get('use_momentum', True):
        momentum_ok = current_price > df['valor_fechamento'].iloc[i-1]
        conditions.append(momentum_ok)
    
    # 6. Filtro de confluência
    min_confluencia = config.get('min_confluencia', 3)
    
    return sum(conditions) >= min_confluencia

def dynamic_exit_logic(df, position, i, config):
    """Lógica dinâmica de saída com trailing stop e múltiplos TPs"""
    
    current_price = df['valor_fechamento'].iloc[i]
    entry_price = position['entry_price']
    
    # SL e TP base (fixos)
    sl_pct = config.get('sl_pct', 0.05)
    tp_pct = config.get('tp_pct', 0.08)
    
    sl_level = entry_price * (1 - sl_pct)
    tp_level = entry_price * (1 + tp_pct)
    
    # Stop Loss fixo
    if current_price <= sl_level:
        return "SL", sl_level
    
    # Take Profit dinâmico com múltiplos níveis
    if config.get('use_dynamic_tp', False):
        
        # TP1: 4% (fechar 30%)
        tp1_level = entry_price * 1.04
        if current_price >= tp1_level and not position.get('tp1_hit', False):
            position['tp1_hit'] = True
            return "TP1", tp1_level
        
        # TP2: 8% (fechar 50%)  
        tp2_level = entry_price * 1.08
        if current_price >= tp2_level and not position.get('tp2_hit', False):
            position['tp2_hit'] = True
            return "TP2", tp2_level
        
        # TP3: 15% (fechar resto)
        tp3_level = entry_price * 1.15
        if current_price >= tp3_level:
            return "TP3", tp3_level
    
    else:
        # TP fixo tradicional
        if current_price >= tp_level:
            return "TP", tp_level
    
    # Trailing Stop avançado
    if config.get('use_trailing_stop', False):
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct > 0.10:  # Só ativa trailing se +10%
            trailing_pct = config.get('trailing_pct', 0.03)  # 3% trailing
            high_water = position.get('high_water', entry_price)
            
            if current_price > high_water:
                position['high_water'] = current_price
                high_water = current_price
            
            trailing_stop = high_water * (1 - trailing_pct)
            
            if current_price <= trailing_stop:
                return "TRAILING", trailing_stop
    
    return None, None

def simulate_optimized_strategy(df, config):
    """Simula estratégia otimizada"""
    
    df = calculate_technical_indicators(df)
    
    balance = config.get('initial_balance', 1000)
    leverage = config.get('leverage', 3)
    trades = []
    position = None
    max_balance = balance
    max_drawdown = 0
    
    for i in range(50, len(df) - 1):
        current_price = df['valor_fechamento'].iloc[i]
        
        # Entrada
        if position is None:
            if advanced_entry_logic(df, i, config):
                position = {
                    'entry_price': current_price,
                    'entry_time': i,
                    'side': 'buy',
                    'high_water': current_price
                }
        
        # Saída
        elif position is not None:
            exit_reason, exit_price = dynamic_exit_logic(df, position, i, config)
            
            if exit_reason:
                # Calcular P&L
                price_change = (exit_price - position['entry_price']) / position['entry_price']
                pnl_leveraged = price_change * leverage
                trade_pnl = balance * pnl_leveraged
                balance += trade_pnl
                
                trades.append({
                    'entry_price': position['entry_price'],
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
        'total_return': (balance - config['initial_balance']) / config['initial_balance'] * 100,
        'trades': trades,
        'num_trades': len(trades),
        'max_drawdown': max_drawdown * 100,
        'config': config
    }

def optimize_parameters():
    """Otimiza parâmetros para máximo ganho"""
    
    print("🚀 OTIMIZAÇÃO AVANÇADA - MAXIMIZAR GANHOS")
    print("="*60)
    
    # Asset de referência (melhor performer)
    filename = "dados_reais_btc_1ano.csv"
    df = load_data(filename)
    
    if df is None:
        print("❌ Arquivo BTC não encontrado")
        return
    
    print(f"📊 Otimizando com {len(df)} barras de dados BTC...")
    
    # Parâmetros para otimizar
    parameter_grid = {
        'leverage': [3],  # Fixo no melhor
        'sl_pct': [0.03, 0.04, 0.05, 0.06],  # Testar SLs diferentes
        'tp_pct': [0.06, 0.08, 0.10, 0.12],  # Testar TPs diferentes
        'volume_multiplier': [1.0, 1.2, 1.5, 2.0],
        'min_confluencia': [2, 3, 4],
        'use_rsi_filter': [True, False],
        'use_trailing_stop': [True, False],
        'trailing_pct': [0.02, 0.03, 0.04]
    }
    
    best_result = None
    best_roi = -float('inf')
    configs_tested = 0
    
    # Gerar combinações (limitado para não explodir)
    sl_options = parameter_grid['sl_pct']
    tp_options = parameter_grid['tp_pct']
    volume_options = parameter_grid['volume_multiplier']
    confluencia_options = parameter_grid['min_confluencia']
    
    print(f"🔍 Testando combinações otimizadas...")
    
    for sl_pct in sl_options:
        for tp_pct in tp_options:
            for vol_mult in volume_options:
                for min_conf in confluencia_options:
                    
                    config = {
                        'initial_balance': 1000,
                        'leverage': 3,
                        'sl_pct': sl_pct,
                        'tp_pct': tp_pct,
                        'volume_multiplier': vol_mult,
                        'min_confluencia': min_conf,
                        'use_ema_trend': True,
                        'use_price_above_ema': True,
                        'use_rsi_filter': True,
                        'use_volume_filter': True,
                        'use_momentum': True,
                        'use_trailing_stop': False,  # Simplificar primeiro
                        'use_dynamic_tp': False
                    }
                    
                    result = simulate_optimized_strategy(df, config)
                    configs_tested += 1
                    
                    if result['total_return'] > best_roi:
                        best_roi = result['total_return']
                        best_result = result
                    
                    # Progress
                    if configs_tested % 20 == 0:
                        print(f"   Testado {configs_tested} configs... Melhor ROI: {best_roi:+.1f}%")
    
    return best_result, configs_tested

def test_advanced_features():
    """Testa features avançadas separadamente"""
    
    print(f"\n🧪 TESTE DE FEATURES AVANÇADAS:")
    print("="*50)
    
    filename = "dados_reais_btc_1ano.csv"
    df = load_data(filename)
    
    # Config base (atual melhor)
    base_config = {
        'initial_balance': 1000,
        'leverage': 3,
        'sl_pct': 0.05,
        'tp_pct': 0.08,
        'volume_multiplier': 1.5,
        'min_confluencia': 3,
        'use_ema_trend': True,
        'use_price_above_ema': True,
        'use_rsi_filter': True,
        'use_volume_filter': True,
        'use_momentum': True,
        'use_trailing_stop': False,
        'use_dynamic_tp': False
    }
    
    # Teste 1: Trailing Stop
    config_trailing = base_config.copy()
    config_trailing['use_trailing_stop'] = True
    config_trailing['trailing_pct'] = 0.03
    
    result_base = simulate_optimized_strategy(df, base_config)
    result_trailing = simulate_optimized_strategy(df, config_trailing)
    
    print(f"📈 Base config: {result_base['total_return']:+.1f}% ({result_base['num_trades']} trades)")
    print(f"📈 Com trailing: {result_trailing['total_return']:+.1f}% ({result_trailing['num_trades']} trades)")
    
    improvement = result_trailing['total_return'] - result_base['total_return']
    print(f"   Melhoria: {improvement:+.1f}pp")
    
    # Teste 2: TP dinâmico
    config_dynamic = base_config.copy()
    config_dynamic['use_dynamic_tp'] = True
    
    result_dynamic = simulate_optimized_strategy(df, config_dynamic)
    print(f"📈 Com TP dinâmico: {result_dynamic['total_return']:+.1f}% ({result_dynamic['num_trades']} trades)")
    
    improvement2 = result_dynamic['total_return'] - result_base['total_return']
    print(f"   Melhoria: {improvement2:+.1f}pp")

def main():
    print("🎯 OBJETIVO: SUPERAR OS +201.8% ATUAIS!")
    print("🔧 Estratégias: Otimização de parâmetros + Features avançadas")
    print()
    
    # Otimização de parâmetros
    best_result, configs_tested = optimize_parameters()
    
    if best_result:
        print(f"\n🏆 MELHOR CONFIGURAÇÃO ENCONTRADA:")
        print("="*50)
        config = best_result['config']
        print(f"ROI: {best_result['total_return']:+.1f}%")
        print(f"Trades: {best_result['num_trades']}")
        print(f"Drawdown: {best_result['max_drawdown']:.1f}%")
        print(f"Configs testadas: {configs_tested}")
        print()
        print("Parâmetros otimizados:")
        for key, value in config.items():
            if key not in ['initial_balance']:
                print(f"   {key}: {value}")
        
        # Comparar com resultado anterior
        previous_roi = 201.8
        improvement = best_result['total_return'] - previous_roi
        print(f"\n🚀 MELHORIA vs ANTERIOR:")
        print(f"   Anterior: +{previous_roi:.1f}%")
        print(f"   Novo: {best_result['total_return']:+.1f}%")
        print(f"   Ganho: {improvement:+.1f}pp {'🎉' if improvement > 0 else '📊'}")
    
    # Testar features avançadas
    test_advanced_features()
    
    print(f"\n" + "="*60)
    print("🎯 PRÓXIMOS PASSOS PARA OTIMIZAÇÃO:")
    print("="*60)
    print("1. Implementar melhor config no trading.py")
    print("2. Testar com outros assets") 
    print("3. Adicionar features avançadas que funcionaram")
    print("4. Validar com backtest completo")

if __name__ == "__main__":
    main()
