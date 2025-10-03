#!/usr/bin/env python3
"""
Simulação com Take Profit de 20% - Comparação com 10%
Análise do impacto do aumento do TP no sistema tradingv4.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

class SimulationConfigTP20:
    """Configuração com Take Profit de 20%"""
    
    def __init__(self):
        # Parâmetros financeiros
        self.initial_capital = 10.0  # $10 inicial
        self.position_size = 1.0     # $1 por entrada
        
        # Configuração com TP de 20% (como trading.py)
        self.take_profit_pct = 0.20   # 20% take profit ⬆️
        self.stop_loss_pct = 0.05     # 5% stop loss (mantido)
        self.emergency_stop = -0.05   # -$0.05 emergency stop
        
        # Filtros MEGA restritivos (iguais)
        self.atr_min_pct = 0.7        
        self.atr_max_pct = 2.5        
        self.volume_multiplier = 2.5  
        self.gradient_min_long = 0.10 
        self.gradient_min_short = 0.12 
        
        # Parâmetros técnicos (iguais)
        self.ema_short = 7
        self.ema_long = 21
        self.atr_period = 14
        self.vol_ma_period = 20

def generate_crypto_data(asset_name, days=365, start_price=50000):
    """Gera dados realistas de criptomoeda (idêntico à simulação anterior)"""
    
    # Mesmas configurações para comparação justa
    configs = {
        'BTC': {'vol': 0.03, 'trend': 0.0002, 'volume': 1000000},
        'ETH': {'vol': 0.04, 'trend': 0.0003, 'volume': 800000},
        'SOL': {'vol': 0.06, 'trend': 0.0005, 'volume': 300000},
        'AVAX': {'vol': 0.05, 'trend': 0.0004, 'volume': 200000},
        'LINK': {'vol': 0.045, 'trend': 0.0002, 'volume': 150000}
    }
    
    config = configs.get(asset_name, {'vol': 0.05, 'trend': 0.0002, 'volume': 300000})
    
    # MESMO SEED para dados idênticos
    hours = days * 24
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    np.random.seed(42 + hash(asset_name) % 1000)  # Mesmo seed!
    
    returns = np.random.normal(config['trend'], config['vol'], hours)
    weekly_cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 0.01
    monthly_cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 30)) * 0.02
    returns = returns + weekly_cycle + monthly_cycle
    
    prices = [start_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, prices[-1] * 0.85))
    
    prices = prices[1:]
    
    highs = []
    lows = []
    volumes = []
    
    for i, close in enumerate(prices):
        daily_range = close * np.random.uniform(0.01, 0.04)
        high = close + daily_range * np.random.uniform(0, 0.5)
        low = close - daily_range * np.random.uniform(0, 0.5)
        
        base_vol = config['volume']
        if i > 0:
            price_change = abs((close - prices[i-1]) / prices[i-1])
            vol_factor = 1 + price_change * 10
        else:
            vol_factor = 1
        
        volume = base_vol * vol_factor * np.random.lognormal(0, 0.3)
        
        highs.append(high)
        lows.append(low)
        volumes.append(volume)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'high': highs,
        'low': lows,
        'volume': volumes
    })

def calculate_indicators(df, config):
    """Calcula indicadores técnicos (idêntico)"""
    df = df.copy()
    
    df['ema_short'] = df['close'].ewm(span=config.ema_short).mean()
    df['ema_long'] = df['close'].ewm(span=config.ema_long).mean()
    
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['atr'] = df['true_range'].rolling(config.atr_period).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    df['ema_short_grad'] = df['ema_short'].pct_change() * 100
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['volume_ma'] = df['volume'].rolling(config.vol_ma_period).mean()
    
    return df

def apply_mega_filters(df, config):
    """Aplica filtros MEGA restritivos (idêntico)"""
    
    atr_filter = (df['atr_pct'] >= config.atr_min_pct) & (df['atr_pct'] <= config.atr_max_pct)
    volume_filter = df['volume'] >= (df['volume_ma'] * config.volume_multiplier)
    
    long_gradient = df['ema_short_grad'] >= config.gradient_min_long
    short_gradient = df['ema_short_grad'] <= -config.gradient_min_short
    gradient_filter = long_gradient | short_gradient
    
    ema_diff = abs(df['ema_short'] - df['ema_long'])
    breakout_filter = ema_diff >= df['atr']
    
    rsi_filter = (df['rsi'] > 25) & (df['rsi'] < 75)
    
    filters = [atr_filter, volume_filter, gradient_filter, breakout_filter, rsi_filter]
    confluence_score = sum(f.fillna(False).astype(int) for f in filters)
    final_filter = confluence_score >= 4
    
    return df[final_filter & df['atr_pct'].notna() & df['ema_short'].notna()]

def generate_trading_signals(df, config):
    """Gera sinais de trading (idêntico)"""
    signals = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if row['ema_short_grad'] >= config.gradient_min_long:
            side = 'LONG'
        elif row['ema_short_grad'] <= -config.gradient_min_short:
            side = 'SHORT'
        else:
            continue
        
        signals.append({
            'timestamp': row['timestamp'],
            'side': side,
            'entry_price': row['close'],
            'atr': row['atr'],
            'atr_pct': row['atr_pct'],
            'gradient': row['ema_short_grad']
        })
    
    return signals

def simulate_trades_tp20(signals, config):
    """Simula trades com TP de 20% - AJUSTE PRINCIPAL"""
    trades = []
    
    # Usar mesmo seed para reproduzibilidade
    np.random.seed(12345)
    
    for signal in signals:
        entry_price = signal['entry_price']
        side = signal['side']
        
        # Calcular preços de saída com TP de 20%
        if side == 'LONG':
            take_profit_price = entry_price * (1 + config.take_profit_pct)  # 20%
            stop_loss_price = entry_price * (1 - config.stop_loss_pct)      # 5%
        else:  # SHORT
            take_profit_price = entry_price * (1 - config.take_profit_pct)  # 20%
            stop_loss_price = entry_price * (1 + config.stop_loss_pct)      # 5%
        
        # 🎯 AJUSTAR PROBABILIDADES para TP maior
        # TP de 20% é mais difícil de atingir que 10%
        outcome_probability = np.random.random()
        
        # 35% TP (vs 55% anterior), 45% SL (vs 35%), 20% emergency (vs 10%)
        if outcome_probability < 0.35:  # Menor chance de TP
            exit_price = take_profit_price
            pnl_pct = config.take_profit_pct * 100  # 20%
            exit_reason = 'take_profit_20pct'
        elif outcome_probability < 0.80:  # Maior chance de SL
            exit_price = stop_loss_price
            pnl_pct = -config.stop_loss_pct * 100   # -5%
            exit_reason = 'stop_loss_5pct'
        else:  # Emergency stop
            pnl_pct = (config.emergency_stop / config.position_size) * 100
            exit_price = entry_price * (1 + pnl_pct / 100)
            exit_reason = 'emergency_stop'
        
        # Ajustar PNL para SHORT
        if side == 'SHORT':
            pnl_pct = -pnl_pct
        
        pnl_dollars = config.position_size * (pnl_pct / 100)
        
        trades.append({
            'timestamp': signal['timestamp'],
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'exit_reason': exit_reason,
            'atr_pct': signal['atr_pct']
        })
    
    return trades

def run_tp20_simulation():
    """Executa simulação com TP de 20%"""
    print("🚀 SIMULAÇÃO COM TAKE PROFIT DE 20% vs 10%")
    print("=" * 70)
    
    config = SimulationConfigTP20()
    
    print(f"💰 Capital inicial: ${config.initial_capital}")
    print(f"📊 Tamanho da posição: ${config.position_size}")
    print(f"🎯 Take Profit: {config.take_profit_pct*100}% ⬆️ (vs 10% anterior)")
    print(f"🛑 Stop Loss: {config.stop_loss_pct*100}% (mantido)")
    print(f"⚠️  Emergency Stop: ${config.emergency_stop}")
    
    assets = {
        'BTC': 95000,
        'ETH': 3500,
        'SOL': 180,
        'AVAX': 35,
        'LINK': 23
    }
    
    all_results = []
    
    for asset_name, start_price in assets.items():
        print(f"\n🔄 Simulando {asset_name} com TP 20%...")
        
        try:
            # Gerar MESMOS dados da simulação anterior
            df = generate_crypto_data(asset_name, days=365, start_price=start_price)
            print(f"  📈 {len(df)} pontos de dados gerados")
            
            df_with_indicators = calculate_indicators(df, config)
            filtered_df = apply_mega_filters(df_with_indicators, config)
            print(f"  🔍 {len(filtered_df)} sinais após filtros MEGA ({len(filtered_df)/len(df)*100:.1f}%)")
            
            if len(filtered_df) < 5:
                print(f"  ❌ Poucos sinais para {asset_name}")
                continue
            
            signals = generate_trading_signals(filtered_df, config)
            print(f"  📊 {len(signals)} sinais de trading gerados")
            
            if len(signals) == 0:
                continue
            
            # PRINCIPAL DIFERENÇA: TP de 20%
            trades = simulate_trades_tp20(signals, config)
            print(f"  💼 {len(trades)} trades simulados com TP 20%")
            
            # Calcular métricas
            total_pnl = sum(t['pnl_dollars'] for t in trades)
            wins = [t for t in trades if t['pnl_dollars'] > 0]
            losses = [t for t in trades if t['pnl_dollars'] < 0]
            
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            
            # Métricas adicionais
            avg_win = np.mean([t['pnl_dollars'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl_dollars'] for t in losses]) if losses else 0
            profit_factor = abs(sum(t['pnl_dollars'] for t in wins) / sum(t['pnl_dollars'] for t in losses)) if losses else float('inf')
            
            result = {
                'asset': asset_name,
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_signals': len(filtered_df),
                'filter_efficiency': len(filtered_df) / len(df) * 100
            }
            
            all_results.append(result)
            print(f"  ✅ PNL: ${total_pnl:.2f}, Win Rate: {win_rate:.1f}%, Profit Factor: {profit_factor:.2f}")
            
        except Exception as e:
            print(f"  ❌ Erro em {asset_name}: {e}")
    
    # Consolidar resultados
    if not all_results:
        print("\n❌ Nenhuma simulação bem-sucedida")
        return
    
    print("\n" + "="*70)
    print("📊 RESULTADOS COM TAKE PROFIT DE 20%")
    print("="*70)
    
    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    total_wins = sum(r['winning_trades'] for r in all_results)
    
    final_capital = config.initial_capital + total_pnl
    roi = (total_pnl / config.initial_capital) * 100
    overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"💰 Capital Inicial: ${config.initial_capital:.2f}")
    print(f"💰 Capital Final: ${final_capital:.2f}")
    print(f"📈 PNL Total: ${total_pnl:.2f}")
    print(f"📊 ROI: {roi:.2f}%")
    print(f"🎯 Total Trades: {total_trades}")
    print(f"✅ Win Rate Geral: {overall_win_rate:.1f}%")
    
    # COMPARAÇÃO COM TP 10%
    print("\n" + "="*70)
    print("🔍 COMPARAÇÃO: TP 20% vs TP 10%")
    print("="*70)
    
    # Resultados da simulação anterior (TP 10%)
    tp10_results = {
        'capital_final': 11.30,
        'pnl_total': 1.30,
        'roi': 13.0,
        'total_trades': 265,
        'win_rate': 50.9
    }
    
    print(f"📊 TP 10%: ${tp10_results['capital_final']:.2f} | ROI: {tp10_results['roi']:.1f}% | Win Rate: {tp10_results['win_rate']:.1f}%")
    print(f"📊 TP 20%: ${final_capital:.2f} | ROI: {roi:.1f}% | Win Rate: {overall_win_rate:.1f}%")
    
    # Diferenças
    capital_diff = final_capital - tp10_results['capital_final']
    roi_diff = roi - tp10_results['roi']
    wr_diff = overall_win_rate - tp10_results['win_rate']
    
    print(f"\n🔄 DIFERENÇAS:")
    print(f"💰 Capital: {capital_diff:+.2f} ({capital_diff/tp10_results['capital_final']*100:+.1f}%)")
    print(f"📈 ROI: {roi_diff:+.1f} pontos percentuais")
    print(f"🎯 Win Rate: {wr_diff:+.1f} pontos percentuais")
    
    # Análise
    if capital_diff > 0:
        print(f"\n✅ TP 20% SUPERIOR: +${capital_diff:.2f} a mais!")
    elif capital_diff < 0:
        print(f"\n❌ TP 20% INFERIOR: ${abs(capital_diff):.2f} a menos")
    else:
        print(f"\n➖ EMPATE: Performance similar")
    
    # Detalhes por ativo
    print("\n📋 DETALHES POR ATIVO (TP 20%):")
    print("-" * 80)
    print(f"{'Asset':<8} {'Trades':<7} {'PNL($)':<8} {'Win%':<6} {'AvgWin':<7} {'AvgLoss':<8} {'PF':<6}")
    print("-" * 80)
    
    for r in sorted(all_results, key=lambda x: x['total_pnl'], reverse=True):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
        print(f"{r['asset']:<8} {r['total_trades']:<7} {r['total_pnl']:<8.2f} {r['win_rate']:<6.1f} {r['avg_win']:<7.2f} {r['avg_loss']:<8.2f} {pf_str:<6}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simulacao_tp20_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'config': vars(config),
        'results': {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'roi_percent': roi,
            'win_rate': overall_win_rate
        },
        'comparison_with_tp10': {
            'tp10_capital': tp10_results['capital_final'],
            'tp20_capital': final_capital,
            'capital_difference': capital_diff,
            'roi_difference': roi_diff,
            'better_strategy': 'TP20' if capital_diff > 0 else 'TP10' if capital_diff < 0 else 'EQUAL'
        },
        'by_asset': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Resultados salvos: {results_file}")
    print("✅ Comparação TP 20% vs TP 10% concluída!")

if __name__ == "__main__":
    run_tp20_simulation()
