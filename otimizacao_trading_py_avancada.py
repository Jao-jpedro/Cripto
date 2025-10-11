#!/usr/bin/env python3
"""
🧬 OTIMIZAÇÃO AVANÇADA TRADING.PY - MÚLTIPLOS BACKTESTS
Testando diferentes configurações para maximizar ROI mantendo $4 por entrada
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES PARA TESTAR
DNA_CONFIGS = {
    "ORIGINAL": {
        "name": "DNA Original",
        "stop_loss": 0.015,
        "take_profit": 0.12,
        "leverage": 3,
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 6.0,
        "volume_multiplier": 1.5,
        "atr_min": 0.25,
        "atr_max": 2.0
    },
    
    "AGRESSIVO": {
        "name": "DNA Agressivo - TP Alto",
        "stop_loss": 0.02,   # SL 2%
        "take_profit": 0.18, # TP 18% (maior)
        "leverage": 4,       # Leverage maior
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 5.5,  # Menos restritivo
        "volume_multiplier": 1.3,
        "atr_min": 0.3,
        "atr_max": 2.5
    },
    
    "CONSERVADOR": {
        "name": "DNA Conservador - SL Baixo",
        "stop_loss": 0.01,   # SL 1%
        "take_profit": 0.08, # TP 8%
        "leverage": 2,       # Leverage menor
        "ema_fast": 5,
        "ema_slow": 21,
        "rsi_period": 14,
        "min_confluence": 7.0,  # Mais restritivo
        "volume_multiplier": 2.0,
        "atr_min": 0.4,
        "atr_max": 1.5
    },
    
    "SCALPING": {
        "name": "DNA Scalping - Trades Rápidos",
        "stop_loss": 0.008,  # SL 0.8%
        "take_profit": 0.04, # TP 4%
        "leverage": 5,       # Leverage alto
        "ema_fast": 2,       # EMAs mais rápidas
        "ema_slow": 13,
        "rsi_period": 7,
        "min_confluence": 4.0,  # Muito menos restritivo
        "volume_multiplier": 1.2,
        "atr_min": 0.2,
        "atr_max": 3.0
    },
    
    "SWING": {
        "name": "DNA Swing - Trades Longos",
        "stop_loss": 0.03,   # SL 3%
        "take_profit": 0.25, # TP 25%
        "leverage": 3,
        "ema_fast": 8,       # EMAs mais lentas
        "ema_slow": 55,
        "rsi_period": 28,
        "min_confluence": 7.5,  # Muito restritivo
        "volume_multiplier": 2.5,
        "atr_min": 0.5,
        "atr_max": 1.8
    },
    
    "MOMENTUM": {
        "name": "DNA Momentum - Força Tendência",
        "stop_loss": 0.015,
        "take_profit": 0.15,
        "leverage": 3,
        "ema_fast": 3,
        "ema_slow": 34,
        "rsi_period": 21,
        "min_confluence": 5.0,
        "volume_multiplier": 3.0,  # Volume muito alto
        "atr_min": 0.6,      # ATR alto (momentum)
        "atr_max": 2.0
    },
    
    "BREAKOUT": {
        "name": "DNA Breakout - Rompimentos",
        "stop_loss": 0.025,
        "take_profit": 0.20,
        "leverage": 4,
        "ema_fast": 5,
        "ema_slow": 21,
        "rsi_period": 14,
        "min_confluence": 6.0,
        "volume_multiplier": 4.0,  # Volume explosivo
        "atr_min": 0.8,      # ATR muito alto
        "atr_max": 3.0
    }
}

# Assets com dados reais
ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def load_data(asset):
    """Carrega dados do asset"""
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
    except Exception as e:
        return None

def calculate_indicators(df, config):
    """Calcula indicadores baseado na configuração"""
    # EMAs
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_short_grad_pct'] = df['ema_short'].pct_change() * 100
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    # ATR
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
    
    return df

def entry_condition_optimized(row, config) -> Tuple[bool, str]:
    """Condição de entrada otimizada por configuração"""
    confluence_score = 0
    max_score = 10
    
    # 1. EMA Cross + Gradiente
    c1_ema = row.ema_short > row.ema_long
    c1_grad = row.ema_short_grad_pct > 0.05
    if c1_ema and c1_grad:
        confluence_score += 1
    elif c1_ema:
        confluence_score += 0.5
    
    # 2. ATR
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 1
    
    # 3. Rompimento
    if row.close > (row.ema_short + 0.3 * row.atr):
        confluence_score += 1
    elif row.close > row.ema_short:
        confluence_score += 0.5
    
    # 4. Volume
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    if volume_ratio > config['volume_multiplier']:
        confluence_score += 1
    elif volume_ratio > 1.0:
        confluence_score += 0.5
    
    # 5. RSI
    if pd.notna(row.rsi):
        if 30 <= row.rsi <= 70:
            confluence_score += 1
        elif 25 <= row.rsi <= 75:
            confluence_score += 0.5
    else:
        confluence_score += 0.5
    
    # 6-10. Outros critérios (simplificados)
    confluence_score += 2.5  # Pontos base
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"Confluência: {confluence_score:.1f}/{max_score}"
    
    return is_valid, reason

def simulate_trading_optimized(df, asset, config):
    """Simula trading com configuração otimizada"""
    capital = 4.0
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < config['ema_slow']:
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = entry_condition_optimized(row, config)
            
            if should_enter:
                entry_price = row.close
                position_size = capital * config['leverage']
                shares = position_size / entry_price
                
                stop_loss = entry_price * (1 - config['stop_loss'])
                take_profit = entry_price * (1 + config['take_profit'])
                
                position = {
                    'type': 'LONG',
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
            elif i - position['entry_bar'] >= 168:  # 1 semana timeout
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                pnl_percent = (pnl_gross / (position['capital_used'] * config['leverage'])) * 100
                
                trade = {
                    'asset': asset,
                    'exit_reason': exit_reason,
                    'pnl_gross': pnl_gross,
                    'pnl_percent': pnl_percent,
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_bars': i - position['entry_bar']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def run_backtest_config(config_name, config):
    """Executa backtest para uma configuração específica"""
    print(f"\n🧬 TESTANDO: {config['name']}")
    print("="*60)
    
    all_trades = []
    total_pnl = 0
    asset_results = {}
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        df = calculate_indicators(df, config)
        trades = simulate_trading_optimized(df, asset, config)
        
        if trades:
            wins = [t for t in trades if t['pnl_gross'] > 0]
            win_rate = len(wins) / len(trades) * 100
            asset_pnl = sum(t['pnl_gross'] for t in trades)
            roi = (asset_pnl / 4.0) * 100
            
            asset_results[asset] = {
                'trades': len(trades),
                'win_rate': win_rate,
                'pnl': asset_pnl,
                'roi': roi
            }
            
            total_pnl += asset_pnl
            all_trades.extend(trades)
        else:
            asset_results[asset] = {'trades': 0, 'win_rate': 0, 'pnl': 0, 'roi': 0}
    
    # Calcular métricas finais
    total_capital = len(ASSETS) * 4.0
    portfolio_roi = (total_pnl / total_capital) * 100
    
    total_trades = len(all_trades)
    wins = [t for t in all_trades if t['pnl_gross'] > 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    
    # Estatísticas de saída
    exit_stats = {}
    for trade in all_trades:
        reason = trade['exit_reason']
        if reason not in exit_stats:
            exit_stats[reason] = {'count': 0, 'pnl': 0}
        exit_stats[reason]['count'] += 1
        exit_stats[reason]['pnl'] += trade['pnl_gross']
    
    result = {
        'config_name': config_name,
        'config': config,
        'portfolio_roi': portfolio_roi,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'asset_results': asset_results,
        'exit_stats': exit_stats
    }
    
    # Imprimir resultado
    print(f"📊 RESULTADO {config['name']}:")
    print(f"   💰 ROI Portfolio: {portfolio_roi:+.1f}%")
    print(f"   📈 PnL Total: ${total_pnl:+.2f}")
    print(f"   🎯 Total Trades: {total_trades}")
    print(f"   🏆 Win Rate: {win_rate:.1f}%")
    
    if exit_stats:
        print(f"   📊 Saídas:")
        for reason, data in exit_stats.items():
            pct = (data['count'] / total_trades) * 100
            avg_pnl = data['pnl'] / data['count']
            print(f"      {reason}: {data['count']} ({pct:.1f}%) | Avg: ${avg_pnl:+.2f}")
    
    return result

def main():
    print("🧬 OTIMIZAÇÃO AVANÇADA TRADING.PY - MÚLTIPLOS BACKTESTS")
    print("="*80)
    print("🎯 Objetivo: Maximizar ROI mantendo entradas de $4")
    print("📊 Testando 7 configurações diferentes do DNA genético")
    
    all_results = []
    
    # Executar todos os backtests
    for config_name, config in DNA_CONFIGS.items():
        result = run_backtest_config(config_name, config)
        all_results.append(result)
    
    # Comparar resultados
    print("\n" + "="*80)
    print("🏆 COMPARAÇÃO FINAL DE CONFIGURAÇÕES")
    print("="*80)
    
    # Ordenar por ROI
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\n📊 RANKING POR ROI:")
    print("Pos | Configuração        | ROI      | Trades | Win Rate | PnL")
    print("-" * 70)
    
    for i, result in enumerate(all_results, 1):
        name = result['config']['name'][:18]
        roi = result['portfolio_roi']
        trades = result['total_trades']
        wr = result['win_rate']
        pnl = result['total_pnl']
        
        print(f"{i:2}  | {name:<18} | {roi:+7.1f}% | {trades:6} | {wr:6.1f}% | ${pnl:+7.2f}")
    
    # Melhor configuração
    best = all_results[0]
    print(f"\n🥇 MELHOR CONFIGURAÇÃO: {best['config']['name']}")
    print(f"   🚀 ROI: {best['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${best['total_pnl']:+.2f}")
    print(f"   📊 Trades: {best['total_trades']}")
    print(f"   🎯 Win Rate: {best['win_rate']:.1f}%")
    
    print(f"\n🔧 PARÂMETROS VENCEDORES:")
    for key, value in best['config'].items():
        if key != 'name':
            print(f"   {key}: {value}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"otimizacao_trading_py_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    
    print(f"\n🎊 OTIMIZAÇÃO CONCLUÍDA!")
    print("="*80)

if __name__ == "__main__":
    main()
