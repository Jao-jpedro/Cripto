#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÃO EXTREMA - MACHINE LEARNING + GRID SEARCH
Testando configurações ainda mais avançadas para superar os +386.3% ROI
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
from itertools import product
import random
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES EXTREMAS PARA TESTAR
EXTREME_CONFIGS = {
    "HYPER_AGGRESSIVE": {
        "name": "DNA Hyper Agressivo",
        "stop_loss": 0.01,    # SL ultra baixo
        "take_profit": 0.35,  # TP gigante
        "leverage": 6,        # Leverage máximo
        "ema_fast": 2,        # EMA super rápida
        "ema_slow": 21,       # EMA mais responsiva
        "rsi_period": 14,     # RSI clássico
        "min_confluence": 4.0, # Menos restritivo
        "volume_multiplier": 0.8, # Volume menor
        "atr_min": 0.2,
        "atr_max": 4.0
    },
    
    "MOMENTUM_EXTREME": {
        "name": "DNA Momentum Extremo",
        "stop_loss": 0.012,
        "take_profit": 0.28,
        "leverage": 7,
        "ema_fast": 3,
        "ema_slow": 13,       # EMA muito rápida
        "rsi_period": 7,      # RSI ultra responsivo
        "min_confluence": 3.5,
        "volume_multiplier": 5.0, # Volume explosivo
        "atr_min": 0.8,       # ATR alto
        "atr_max": 3.5
    },
    
    "SCALP_MONSTER": {
        "name": "DNA Scalp Monster",
        "stop_loss": 0.008,   # SL minúsculo
        "take_profit": 0.25,
        "leverage": 8,        # Leverage máximo
        "ema_fast": 2,
        "ema_slow": 8,        # EMA super rápida
        "rsi_period": 5,      # RSI hiper responsivo
        "min_confluence": 3.0, # Muito permissivo
        "volume_multiplier": 0.5, # Qualquer volume
        "atr_min": 0.1,
        "atr_max": 5.0
    },
    
    "BREAKOUT_BEAST": {
        "name": "DNA Breakout Beast",
        "stop_loss": 0.02,
        "take_profit": 0.40,  # TP gigantesco
        "leverage": 5,
        "ema_fast": 5,
        "ema_slow": 55,       # EMA lenta para breakouts
        "rsi_period": 28,     # RSI mais lento
        "min_confluence": 7.0, # Mais seletivo
        "volume_multiplier": 10.0, # Volume extremo
        "atr_min": 1.0,       # ATR alto
        "atr_max": 2.5
    },
    
    "VOLATILITY_HUNTER": {
        "name": "DNA Volatility Hunter",
        "stop_loss": 0.025,
        "take_profit": 0.50,  # TP épico
        "leverage": 4,
        "ema_fast": 3,
        "ema_slow": 89,       # EMA muito lenta
        "rsi_period": 35,     # RSI lento
        "min_confluence": 8.0, # Ultra seletivo
        "volume_multiplier": 15.0, # Volume monstruoso
        "atr_min": 2.0,       # Alta volatilidade
        "atr_max": 4.0
    }
}

# Assets
ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def load_data(asset):
    symbol = asset.replace("-USD", "").lower()
    filename = f"dados_reais_{symbol}_1ano.csv"
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        df = df.rename(columns={
            'valor_fechamento': 'close',
            'valor_abertura': 'open',
            'valor_maximo': 'high',
            'valor_minimo': 'low'
        })
        
        return df
    except:
        return None

def calculate_indicators_advanced(df, config):
    # EMAs
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_short_grad_pct'] = df['ema_short'].pct_change() * 100
    df['ema_momentum'] = (df['ema_short'] / df['ema_long'] - 1) * 100
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma']
    
    # ATR avançado
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Indicadores extras
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volatility'] = df['price_change_pct'].rolling(window=20).std()
    
    # Momentum score
    df['momentum_score'] = (
        (df['ema_short_grad_pct'] > 0).astype(int) * 2 +
        (df['ema_momentum'] > 0).astype(int) * 2 +
        (df['vol_surge'] > 1.5).astype(int) * 2 +
        (df['atr_pct'] > 0.5).astype(int) * 1
    )
    
    return df

def entry_condition_extreme(row, config) -> Tuple[bool, str]:
    confluence_score = 0
    max_score = 12
    reasons = []
    
    # 1. EMA System (peso 3)
    ema_score = 0
    if row.ema_short > row.ema_long:
        ema_score += 1
        if row.ema_short_grad_pct > 0.1:
            ema_score += 1
        if row.ema_momentum > 1:
            ema_score += 1
    confluence_score += ema_score
    if ema_score > 0:
        reasons.append(f"EMA({ema_score})")
    
    # 2. Breakout Power (peso 2.5)
    breakout_score = 0
    if row.close > row.ema_short * 1.005:
        breakout_score += 1
        if row.close > row.ema_short * 1.01:
            breakout_score += 1
        if row.atr_pct > 0.8:
            breakout_score += 0.5
    confluence_score += breakout_score
    if breakout_score > 0:
        reasons.append(f"Break({breakout_score:.1f})")
    
    # 3. Volume Explosion (peso 2.5)
    volume_score = 0
    if hasattr(row, 'vol_surge') and row.vol_surge > config['volume_multiplier']:
        volume_score += 2.5
        reasons.append("Vol-Exp")
    elif hasattr(row, 'vol_surge') and row.vol_surge > 1.5:
        volume_score += 1.5
        reasons.append("Vol-High")
    confluence_score += volume_score
    
    # 4. Volatility Range (peso 2)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 2
        reasons.append("ATR-OK")
    elif row.atr_pct > config['atr_max'] * 0.8:
        confluence_score += 1
        reasons.append("ATR-High")
    
    # 5. RSI Condition (peso 1.5)
    if pd.notna(row.rsi):
        if 20 <= row.rsi <= 80:
            confluence_score += 1.5
            reasons.append("RSI-OK")
        elif 15 <= row.rsi <= 85:
            confluence_score += 1
            reasons.append("RSI-Med")
    
    # 6. Momentum Boost (peso 0.5)
    if hasattr(row, 'momentum_score') and row.momentum_score >= 5:
        confluence_score += 0.5
        reasons.append("Momentum")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/12 [{','.join(reasons[:4])}]"
    
    return is_valid, reason

def simulate_trading_extreme(df, asset, config):
    capital = 4.0
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period']):
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = entry_condition_extreme(row, config)
            
            if should_enter:
                entry_price = row.close
                position_size = capital * config['leverage']
                shares = position_size / entry_price
                
                stop_loss = entry_price * (1 - config['stop_loss'])
                take_profit = entry_price * (1 + config['take_profit'])
                
                position = {
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_bar': i,
                    'entry_time': row.timestamp,
                    'capital_used': capital,
                    'reason': reason
                }
                
        else:
            current_price = row.close
            exit_reason = None
            
            if current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'
            elif current_price >= position['take_profit']:
                exit_reason = 'TAKE_PROFIT'
            elif i - position['entry_bar'] >= 72:  # 3 dias timeout
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                
                trade = {
                    'asset': asset,
                    'exit_reason': exit_reason,
                    'pnl_gross': pnl_gross,
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_bars': i - position['entry_bar'],
                    'entry_reason': position['reason']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def generate_random_configs(num_configs=10):
    """Gerar configurações aleatórias para teste"""
    configs = {}
    
    for i in range(num_configs):
        config = {
            "name": f"DNA Random {i+1}",
            "stop_loss": round(random.uniform(0.005, 0.03), 3),
            "take_profit": round(random.uniform(0.15, 0.60), 2),
            "leverage": random.choice([3, 4, 5, 6, 7, 8]),
            "ema_fast": random.choice([2, 3, 5, 8]),
            "ema_slow": random.choice([13, 21, 34, 55, 89]),
            "rsi_period": random.choice([5, 7, 14, 21, 28]),
            "min_confluence": round(random.uniform(2.0, 9.0), 1),
            "volume_multiplier": round(random.uniform(0.5, 20.0), 1),
            "atr_min": round(random.uniform(0.1, 1.0), 1),
            "atr_max": round(random.uniform(2.0, 5.0), 1)
        }
        configs[f"RANDOM_{i+1}"] = config
    
    return configs

def run_extreme_test(config_name, config):
    print(f"\n🚀 TESTANDO: {config['name']}")
    print("="*70)
    
    all_trades = []
    total_pnl = 0
    profitable_assets = 0
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        df = calculate_indicators_advanced(df, config)
        trades = simulate_trading_extreme(df, asset, config)
        
        if trades:
            asset_pnl = sum(t['pnl_gross'] for t in trades)
            roi = (asset_pnl / 4.0) * 100
            wins = len([t for t in trades if t['pnl_gross'] > 0])
            win_rate = (wins / len(trades)) * 100
            
            if asset_pnl > 0:
                profitable_assets += 1
                status = "🟢"
            else:
                status = "🔴"
            
            print(f"   {status} {asset}: {len(trades)} trades | {win_rate:.1f}% WR | {roi:+.1f}% ROI")
            
            total_pnl += asset_pnl
            all_trades.extend(trades)
    
    total_capital = len(ASSETS) * 4.0
    portfolio_roi = (total_pnl / total_capital) * 100
    total_trades = len(all_trades)
    total_wins = len([t for t in all_trades if t['pnl_gross'] > 0])
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"\n📊 RESULTADO:")
    print(f"   💰 ROI: {portfolio_roi:+.1f}%")
    print(f"   📈 PnL: ${total_pnl:+.2f}")
    print(f"   🎯 Trades: {total_trades}")
    print(f"   🏆 WR: {win_rate:.1f}%")
    print(f"   ✅ Assets+: {profitable_assets}/{len(ASSETS)}")
    
    return {
        'config_name': config_name,
        'config': config,
        'portfolio_roi': portfolio_roi,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profitable_assets': profitable_assets
    }

def main():
    print("🚀 OTIMIZAÇÃO EXTREMA - SUPERANDO +386.3% ROI")
    print("="*80)
    print("🎯 Testando configurações extremas + machine learning")
    
    all_results = []
    
    # 1. Configurações extremas predefinidas
    print("\n🔥 FASE 1: CONFIGURAÇÕES EXTREMAS")
    for config_name, config in EXTREME_CONFIGS.items():
        result = run_extreme_test(config_name, config)
        all_results.append(result)
    
    # 2. Configurações aleatórias (machine learning approach)
    print("\n🎲 FASE 2: CONFIGURAÇÕES ALEATÓRIAS (ML)")
    random_configs = generate_random_configs(15)
    for config_name, config in random_configs.items():
        result = run_extreme_test(config_name, config)
        all_results.append(result)
    
    # 3. Grid search nas melhores
    print("\n🔍 FASE 3: GRID SEARCH DAS MELHORES")
    
    # Pegar top 3 configurações
    top_configs = sorted(all_results, key=lambda x: x['portfolio_roi'], reverse=True)[:3]
    
    grid_configs = {}
    for i, top_config in enumerate(top_configs):
        base_config = top_config['config']
        
        # Testar variações do TP/SL
        for tp_mult in [0.9, 1.1, 1.2]:
            for sl_mult in [0.8, 1.2]:
                config_name = f"GRID_{i+1}_{tp_mult}_{sl_mult}"
                config = base_config.copy()
                config['name'] = f"Grid {i+1} - TP{tp_mult} SL{sl_mult}"
                config['take_profit'] = min(base_config['take_profit'] * tp_mult, 0.8)
                config['stop_loss'] = max(base_config['stop_loss'] * sl_mult, 0.005)
                
                grid_configs[config_name] = config
    
    for config_name, config in grid_configs.items():
        result = run_extreme_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("🏆 RANKING FINAL EXTREMO")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+")
    print("-" * 85)
    
    # Top 10
    for i, result in enumerate(all_results[:10], 1):
        name = result['config']['name'][:20]
        roi = result['portfolio_roi']
        pnl = result['total_pnl']
        trades = result['total_trades']
        wr = result['win_rate']
        assets = result['profitable_assets']
        
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2}"
        print(f"{emoji} | {name:<20} | {roi:+7.1f}% | ${pnl:+7.2f} | {trades:6} | {wr:4.1f}% | {assets:2}/16")
    
    # Configuração suprema
    supreme = all_results[0]
    improvement = (supreme['portfolio_roi'] - 386.3) / 386.3 * 100
    
    print(f"\n👑 CONFIGURAÇÃO SUPREMA ENCONTRADA:")
    print(f"   📛 Nome: {supreme['config']['name']}")
    print(f"   🚀 ROI: {supreme['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${supreme['total_pnl']:+.2f}")
    print(f"   📊 Trades: {supreme['total_trades']}")
    print(f"   🎯 Win Rate: {supreme['win_rate']:.1f}%")
    print(f"   📈 Melhoria: {improvement:+.1f}% sobre DNA Ultra Agressivo")
    
    print(f"\n🔧 PARÂMETROS SUPREMOS:")
    config = supreme['config']
    print(f"   🛑 Stop Loss: {config['stop_loss']*100:.1f}%")
    print(f"   🎯 Take Profit: {config['take_profit']*100:.1f}%")
    print(f"   📈 Leverage: {config['leverage']}x")
    print(f"   🌊 EMA: {config['ema_fast']}/{config['ema_slow']}")
    print(f"   📊 RSI: {config['rsi_period']} períodos")
    print(f"   🎲 Confluência: {config['min_confluence']}/12")
    print(f"   📈 Volume: {config['volume_multiplier']}x")
    print(f"   ⚡ ATR: {config['atr_min']}-{config['atr_max']}%")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"otimizacao_extrema_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 OTIMIZAÇÃO EXTREMA CONCLUÍDA!")
    print("="*80)

if __name__ == "__main__":
    main()
