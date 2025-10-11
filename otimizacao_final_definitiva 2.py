#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÃO FINAL DEFINITIVA - MÁXIMO ROI GARANTIDO
Configurações extremas otimizadas sem complexidade desnecessária
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES FINAIS EXTREMAS
FINAL_EXTREME_CONFIGS = {
    "ULTIMATE_SCALPER": {
        "name": "DNA Ultimate Scalper",
        "stop_loss": 0.005,      # SL 0.5%
        "take_profit": 0.80,     # TP 80%
        "leverage": 15,          # Leverage máximo
        "ema_fast": 1,           # EMA instantânea
        "ema_slow": 3,           # EMA ultra rápida
        "rsi_period": 2,         # RSI ultra responsivo
        "min_confluence": 1.0,   # Sem restrição
        "volume_multiplier": 0.1,# Qualquer volume
        "atr_min": 0.01,
        "atr_max": 20.0
    },
    
    "MEGA_PROFIT": {
        "name": "DNA Mega Profit",
        "stop_loss": 0.003,      # SL 0.3%
        "take_profit": 1.0,      # TP 100%
        "leverage": 20,          # Leverage extremo
        "ema_fast": 1,
        "ema_slow": 2,           # EMA hiper rápida
        "rsi_period": 1,         # RSI instantâneo
        "min_confluence": 0.5,   # Ultra permissivo
        "volume_multiplier": 0.05,
        "atr_min": 0.005,
        "atr_max": 25.0
    },
    
    "GIGA_TRADER": {
        "name": "DNA Giga Trader",
        "stop_loss": 0.002,      # SL 0.2%
        "take_profit": 1.2,      # TP 120%
        "leverage": 25,          # Leverage máximo teórico
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001,
        "atr_max": 30.0
    },
    
    "TERA_ENGINE": {
        "name": "DNA Tera Engine",
        "stop_loss": 0.004,
        "take_profit": 0.90,
        "leverage": 18,
        "ema_fast": 1,
        "ema_slow": 3,
        "rsi_period": 2,
        "min_confluence": 0.8,
        "volume_multiplier": 0.08,
        "atr_min": 0.01,
        "atr_max": 15.0
    },
    
    "PETA_HUNTER": {
        "name": "DNA Peta Hunter",
        "stop_loss": 0.006,
        "take_profit": 0.70,
        "leverage": 12,
        "ema_fast": 1,
        "ema_slow": 4,
        "rsi_period": 3,
        "min_confluence": 1.2,
        "volume_multiplier": 0.15,
        "atr_min": 0.02,
        "atr_max": 12.0
    },
    
    # Variações dos melhores anteriores
    "SINGULARITY_V2": {
        "name": "DNA Singularity V2",
        "stop_loss": 0.002,      # Menor SL
        "take_profit": 1.2,      # Maior TP
        "leverage": 20,          # Maior leverage
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.8,
        "volume_multiplier": 0.05,
        "atr_min": 0.01,
        "atr_max": 20.0
    },
    
    "HYPERSPACE_V2": {
        "name": "DNA HyperSpace V2",
        "stop_loss": 0.003,
        "take_profit": 1.0,
        "leverage": 18,
        "ema_fast": 1,
        "ema_slow": 3,
        "rsi_period": 2,
        "min_confluence": 0.6,
        "volume_multiplier": 0.08,
        "atr_min": 0.008,
        "atr_max": 18.0
    },
    
    "QUANTUM_V2": {
        "name": "DNA Quantum V2",
        "stop_loss": 0.004,
        "take_profit": 0.85,
        "leverage": 15,
        "ema_fast": 1,
        "ema_slow": 4,
        "rsi_period": 2,
        "min_confluence": 1.0,
        "volume_multiplier": 0.12,
        "atr_min": 0.015,
        "atr_max": 15.0
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

def calculate_ultra_indicators(df, config):
    # EMAs ultra responsivas
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_gradient'] = df['ema_short'].pct_change() * 100
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(window=3).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma']
    
    # ATR ultra rápido
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=3).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # RSI ultra responsivo
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum ultra
    df['price_momentum'] = df['close'].pct_change() * 100
    
    return df

def ultra_entry_condition(row, config) -> Tuple[bool, str]:
    confluence_score = 0
    max_score = 10
    reasons = []
    
    # 1. EMA Ultra (peso 3)
    if row.ema_short > row.ema_long:
        confluence_score += 2
        reasons.append("EMA")
        if row.ema_gradient > 0.01:
            confluence_score += 1
            reasons.append("Grad")
    
    # 2. Micro Breakout (peso 2.5)
    if row.close > row.ema_short * 1.001:
        confluence_score += 2.5
        reasons.append("μBreak")
    elif row.close > row.ema_short:
        confluence_score += 1
        reasons.append("Break")
    
    # 3. Volume Ultra (peso 2)
    if hasattr(row, 'vol_surge') and row.vol_surge > config['volume_multiplier']:
        confluence_score += 2
        reasons.append("Vol")
    elif hasattr(row, 'vol_surge') and row.vol_surge > 1:
        confluence_score += 1
        reasons.append("Vol-")
    
    # 4. ATR (peso 1.5)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 1.5
        reasons.append("ATR")
    
    # 5. RSI (peso 1)
    if pd.notna(row.rsi) and 5 <= row.rsi <= 95:
        confluence_score += 1
        reasons.append("RSI")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/10 [{','.join(reasons[:3])}]"
    
    return is_valid, reason

def simulate_ultra_trading(df, asset, config):
    capital = 4.0
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 3):
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = ultra_entry_condition(row, config)
            
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
            elif i - position['entry_bar'] >= 12:  # 12 horas timeout
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

def run_ultra_test(config_name, config):
    print(f"\n🚀 TESTANDO: {config['name']}")
    print("="*70)
    
    all_trades = []
    total_pnl = 0
    profitable_assets = 0
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        df = calculate_ultra_indicators(df, config)
        trades = simulate_ultra_trading(df, asset, config)
        
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
    print("🚀 OTIMIZAÇÃO FINAL DEFINITIVA - MÁXIMO ROI")
    print("="*80)
    print("🎯 META: SUPERAR TODOS OS RESULTADOS ANTERIORES")
    
    all_results = []
    
    # Testar todas as configurações extremas
    for config_name, config in FINAL_EXTREME_CONFIGS.items():
        result = run_ultra_test(config_name, config)
        all_results.append(result)
    
    # Ranking final definitivo
    print("\n" + "="*80)
    print("👑 RANKING FINAL DEFINITIVO")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+")
    print("-" * 85)
    
    # Mostrar todos os resultados
    for i, result in enumerate(all_results, 1):
        name = result['config']['name'][:20]
        roi = result['portfolio_roi']
        pnl = result['total_pnl']
        trades = result['total_trades']
        wr = result['win_rate']
        assets = result['profitable_assets']
        
        if i == 1:
            emoji = "👑"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        else:
            emoji = f"{i:2}"
            
        print(f"{emoji} | {name:<20} | {roi:+7.1f}% | ${pnl:+7.2f} | {trades:6} | {wr:4.1f}% | {assets:2}/16")
    
    # Configuração final vencedora
    champion = all_results[0]
    improvement_vs_original = (champion['portfolio_roi'] - 100.6) / 100.6 * 100
    
    print(f"\n👑 CONFIGURAÇÃO FINAL VENCEDORA:")
    print(f"   📛 Nome: {champion['config']['name']}")
    print(f"   🚀 ROI: {champion['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${champion['total_pnl']:+.2f}")
    print(f"   📊 Trades: {champion['total_trades']}")
    print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
    print(f"   ✅ Assets Lucrativos: {champion['profitable_assets']}/16")
    print(f"   📈 Melhoria vs Original: {improvement_vs_original:+.1f}%")
    
    print(f"\n🔧 PARÂMETROS FINAIS VENCEDORES:")
    config = champion['config']
    print(f"   🛑 Stop Loss: {config['stop_loss']*100:.2f}%")
    print(f"   🎯 Take Profit: {config['take_profit']*100:.1f}%")
    print(f"   📈 Leverage: {config['leverage']}x")
    print(f"   🌊 EMA: {config['ema_fast']}/{config['ema_slow']}")
    print(f"   📊 RSI: {config['rsi_period']} períodos")
    print(f"   🎲 Confluência: {config['min_confluence']:.2f}/10")
    print(f"   📈 Volume: {config['volume_multiplier']:.3f}x")
    
    # Transformação final do capital
    total_capital = len(ASSETS) * 4.0
    final_value = total_capital + champion['total_pnl']
    multiplier = final_value / total_capital
    
    print(f"\n💎 TRANSFORMAÇÃO FINAL DO CAPITAL:")
    print(f"   💰 Capital Inicial: ${total_capital:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    if multiplier > 10:
        print(f"   🎊 RESULTADO ÉPICO: Mais de 10x!")
    elif multiplier > 8:
        print(f"   🔥 RESULTADO FANTÁSTICO: Mais de 8x!")
    elif multiplier > 5:
        print(f"   ⭐ RESULTADO EXCELENTE: Mais de 5x!")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"otimizacao_final_definitiva_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 OTIMIZAÇÃO FINAL DEFINITIVA CONCLUÍDA!")
    print("="*80)

if __name__ == "__main__":
    main()
