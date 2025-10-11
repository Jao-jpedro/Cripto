#!/usr/bin/env python3
"""
🚀 TESTE FINAL - DNA AGRESSIVO OTIMIZADO
Refinando ainda mais a melhor configuração encontrada
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES FINAIS PARA TESTAR (baseadas na vencedora)
FINAL_CONFIGS = {
    "VENCEDOR": {
        "name": "DNA Agressivo Original",
        "stop_loss": 0.02,
        "take_profit": 0.18,
        "leverage": 4,
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 5.5,
        "volume_multiplier": 1.3,
        "atr_min": 0.3,
        "atr_max": 2.5
    },
    
    "ULTRA_AGRESSIVO": {
        "name": "DNA Ultra Agressivo",
        "stop_loss": 0.015,  # SL menor
        "take_profit": 0.22,  # TP ainda maior
        "leverage": 5,        # Leverage máximo
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 5.0,  # Menos restritivo
        "volume_multiplier": 1.2,
        "atr_min": 0.25,
        "atr_max": 3.0
    },
    
    "BALANCEADO_PLUS": {
        "name": "DNA Balanceado Plus",
        "stop_loss": 0.018,
        "take_profit": 0.20,
        "leverage": 4,
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 6.0,
        "volume_multiplier": 1.4,
        "atr_min": 0.35,
        "atr_max": 2.2
    },
    
    "MEGA_TP": {
        "name": "DNA Mega Take Profit",
        "stop_loss": 0.025,  # SL maior para compensar
        "take_profit": 0.30,  # TP gigante
        "leverage": 4,
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 6.5,  # Mais seletivo
        "volume_multiplier": 2.0,
        "atr_min": 0.4,
        "atr_max": 2.0
    },
    
    "VOLUME_HUNTER": {
        "name": "DNA Volume Hunter",
        "stop_loss": 0.02,
        "take_profit": 0.18,
        "leverage": 4,
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 5.0,
        "volume_multiplier": 3.0,  # Volume explosivo
        "atr_min": 0.5,      # ATR alto
        "atr_max": 2.5
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

def calculate_indicators(df, config):
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_short_grad_pct'] = df['ema_short'].pct_change() * 100
    
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def entry_condition_final(row, config) -> Tuple[bool, str]:
    confluence_score = 0
    max_score = 10
    reasons = []
    
    # 1. EMA Cross + Gradiente (peso 2)
    c1_ema = row.ema_short > row.ema_long
    c1_grad = row.ema_short_grad_pct > 0.05
    if c1_ema and c1_grad:
        confluence_score += 2
        reasons.append("EMA+Grad")
    elif c1_ema:
        confluence_score += 1
        reasons.append("EMA")
    
    # 2. ATR (peso 1.5)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 1.5
        reasons.append("ATR-OK")
    
    # 3. Rompimento (peso 1.5)
    if row.close > (row.ema_short + 0.5 * row.atr):
        confluence_score += 1.5
        reasons.append("Breakout")
    elif row.close > row.ema_short:
        confluence_score += 0.8
        reasons.append("Above-EMA")
    
    # 4. Volume (peso 2)
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    if volume_ratio > config['volume_multiplier']:
        confluence_score += 2
        reasons.append("Vol-High")
    elif volume_ratio > 1.5:
        confluence_score += 1
        reasons.append("Vol-Med")
    
    # 5. RSI (peso 1.5)
    if pd.notna(row.rsi):
        if 35 <= row.rsi <= 65:
            confluence_score += 1.5
            reasons.append("RSI-Good")
        elif 25 <= row.rsi <= 75:
            confluence_score += 1
            reasons.append("RSI-OK")
    else:
        confluence_score += 0.8
    
    # 6. Momentum extra (peso 1.5)
    momentum_score = 0
    if hasattr(row, 'ema_short_grad_pct') and row.ema_short_grad_pct > 0.1:
        momentum_score += 0.5
    if volume_ratio > 2.0:
        momentum_score += 0.5
    if row.atr_pct > 0.6:
        momentum_score += 0.5
    
    confluence_score += min(momentum_score, 1.5)
    if momentum_score > 0:
        reasons.append("Momentum")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/10 [{','.join(reasons[:3])}]"
    
    return is_valid, reason

def simulate_trading_final(df, asset, config):
    capital = 4.0
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < config['ema_slow']:
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = entry_condition_final(row, config)
            
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
                    'capital_used': capital
                }
                
        else:
            current_price = row.close
            exit_reason = None
            
            if current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'
            elif current_price >= position['take_profit']:
                exit_reason = 'TAKE_PROFIT'
            elif i - position['entry_bar'] >= 168:
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                
                trade = {
                    'asset': asset,
                    'exit_reason': exit_reason,
                    'pnl_gross': pnl_gross,
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_bars': i - position['entry_bar']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def run_final_test(config_name, config):
    print(f"\n🚀 TESTANDO: {config['name']}")
    print("="*70)
    
    all_trades = []
    total_pnl = 0
    profitable_assets = 0
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        df = calculate_indicators(df, config)
        trades = simulate_trading_final(df, asset, config)
        
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
    
    # Análise de saídas
    exit_analysis = {}
    for trade in all_trades:
        reason = trade['exit_reason']
        if reason not in exit_analysis:
            exit_analysis[reason] = {'count': 0, 'pnl': 0}
        exit_analysis[reason]['count'] += 1
        exit_analysis[reason]['pnl'] += trade['pnl_gross']
    
    print(f"\n📊 RESULTADO FINAL:")
    print(f"   💰 ROI Portfolio: {portfolio_roi:+.1f}%")
    print(f"   📈 PnL Total: ${total_pnl:+.2f}")
    print(f"   🎯 Total Trades: {total_trades}")
    print(f"   🏆 Win Rate: {win_rate:.1f}%")
    print(f"   ✅ Assets Lucrativos: {profitable_assets}/{len(ASSETS)} ({(profitable_assets/len(ASSETS)*100):.1f}%)")
    
    print(f"\n📊 Análise de Saídas:")
    for reason, data in exit_analysis.items():
        pct = (data['count'] / total_trades) * 100
        avg_pnl = data['pnl'] / data['count']
        print(f"   {reason}: {data['count']} ({pct:.1f}%) | Avg: ${avg_pnl:+.2f}")
    
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
    print("🚀 TESTE FINAL - DNA AGRESSIVO OTIMIZADO")
    print("="*80)
    print("🎯 Refinando a configuração vencedora para máximo ROI")
    
    all_results = []
    
    for config_name, config in FINAL_CONFIGS.items():
        result = run_final_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("🏆 RANKING FINAL - CONFIGURAÇÕES OTIMIZADAS")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+")
    print("-" * 85)
    
    for i, result in enumerate(all_results, 1):
        name = result['config']['name'][:20]
        roi = result['portfolio_roi']
        pnl = result['total_pnl']
        trades = result['total_trades']
        wr = result['win_rate']
        assets = result['profitable_assets']
        
        print(f"{i}   | {name:<20} | {roi:+7.1f}% | ${pnl:+7.2f} | {trades:6} | {wr:4.1f}% | {assets:2}/16")
    
    # Configuração final vencedora
    winner = all_results[0]
    print(f"\n🥇 CONFIGURAÇÃO FINAL VENCEDORA:")
    print(f"   📛 Nome: {winner['config']['name']}")
    print(f"   🚀 ROI: {winner['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${winner['total_pnl']:+.2f}")
    print(f"   📊 Trades: {winner['total_trades']}")
    print(f"   🎯 Win Rate: {winner['win_rate']:.1f}%")
    print(f"   ✅ Assets Lucrativos: {winner['profitable_assets']}/16")
    
    print(f"\n🔧 PARÂMETROS FINAIS OTIMIZADOS:")
    config = winner['config']
    print(f"   🛑 Stop Loss: {config['stop_loss']*100:.1f}%")
    print(f"   🎯 Take Profit: {config['take_profit']*100:.1f}%")
    print(f"   📈 Leverage: {config['leverage']}x")
    print(f"   🌊 EMA: {config['ema_fast']}/{config['ema_slow']}")
    print(f"   📊 RSI: {config['rsi_period']} períodos")
    print(f"   🎲 Confluência mín: {config['min_confluence']}/10")
    print(f"   📈 Volume mín: {config['volume_multiplier']}x")
    print(f"   ⚡ ATR: {config['atr_min']}-{config['atr_max']}%")
    
    # Salvar resultado final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dna_final_otimizado_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 OTIMIZAÇÃO FINAL CONCLUÍDA!")
    print("="*80)

if __name__ == "__main__":
    main()
