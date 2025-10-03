#!/usr/bin/env python3
"""
Simulação Simplificada de 1 ano com tradingv4.py
Análise de performance macro com $10 iniciais e entradas de $1
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

class SimpleSimulationConfig:
    """Configuração para simulação simplificada"""
    
    def __init__(self):
        # Parâmetros financeiros
        self.initial_capital = 10.0  # $10 inicial
        self.position_size = 1.0     # $1 por entrada
        
        # Configuração conforme tradingv4.py
        self.take_profit_pct = 0.10   # 10% take profit
        self.stop_loss_pct = 0.05     # 5% stop loss
        self.emergency_stop = -0.05   # -$0.05 emergency stop
        
        # Filtros MEGA restritivos
        self.atr_min_pct = 0.7        # ATR mínimo 0.7%
        self.atr_max_pct = 2.5        # ATR máximo 2.5%
        self.volume_multiplier = 2.5  # Volume 2.5x média
        self.gradient_min_long = 0.10 # Gradient para LONG
        self.gradient_min_short = 0.12 # Gradient para SHORT
        
        # Parâmetros técnicos
        self.ema_short = 7
        self.ema_long = 21
        self.atr_period = 14
        self.vol_ma_period = 20

def generate_crypto_data(asset_name, days=365, start_price=50000):
    """Gera dados realistas de criptomoeda"""
    
    # Configurações por ativo
    configs = {
        'BTC': {'vol': 0.03, 'trend': 0.0002, 'volume': 1000000},
        'ETH': {'vol': 0.04, 'trend': 0.0003, 'volume': 800000},
        'SOL': {'vol': 0.06, 'trend': 0.0005, 'volume': 300000},
        'AVAX': {'vol': 0.05, 'trend': 0.0004, 'volume': 200000},
        'LINK': {'vol': 0.045, 'trend': 0.0002, 'volume': 150000}
    }
    
    config = configs.get(asset_name, {'vol': 0.05, 'trend': 0.0002, 'volume': 300000})
    
    # Gerar dados horários
    hours = days * 24
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Random walk com seed baseado no ativo
    np.random.seed(42 + hash(asset_name) % 1000)
    
    # Gerar retornos
    returns = np.random.normal(config['trend'], config['vol'], hours)
    
    # Adicionar ciclos
    weekly_cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 0.01
    monthly_cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 30)) * 0.02
    returns = returns + weekly_cycle + monthly_cycle
    
    # Gerar preços
    prices = [start_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, prices[-1] * 0.85))  # Limitar quedas
    
    prices = prices[1:]
    
    # Gerar high/low/volume
    highs = []
    lows = []
    volumes = []
    
    for i, close in enumerate(prices):
        daily_range = close * np.random.uniform(0.01, 0.04)
        high = close + daily_range * np.random.uniform(0, 0.5)
        low = close - daily_range * np.random.uniform(0, 0.5)
        
        # Volume com correlação com movimento de preço
        base_vol = config['volume']
        if i > 0:
            price_change = abs((close - prices[i-1]) / prices[i-1])
            vol_factor = 1 + price_change * 10  # Mais volume em movimentos grandes
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
    """Calcula indicadores técnicos"""
    df = df.copy()
    
    # EMAs
    df['ema_short'] = df['close'].ewm(span=config.ema_short).mean()
    df['ema_long'] = df['close'].ewm(span=config.ema_long).mean()
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['atr'] = df['true_range'].rolling(config.atr_period).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Gradient EMA curta (aproximação)
    df['ema_short_grad'] = df['ema_short'].pct_change() * 100
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume MA
    df['volume_ma'] = df['volume'].rolling(config.vol_ma_period).mean()
    
    return df

def apply_mega_filters(df, config):
    """Aplica filtros MEGA restritivos"""
    
    # Filtro 1: ATR entre 0.7% e 2.5%
    atr_filter = (df['atr_pct'] >= config.atr_min_pct) & (df['atr_pct'] <= config.atr_max_pct)
    
    # Filtro 2: Volume acima de 2.5x da média
    volume_filter = df['volume'] >= (df['volume_ma'] * config.volume_multiplier)
    
    # Filtro 3: Gradient significativo
    long_gradient = df['ema_short_grad'] >= config.gradient_min_long
    short_gradient = df['ema_short_grad'] <= -config.gradient_min_short
    gradient_filter = long_gradient | short_gradient
    
    # Filtro 4: Breakout das EMAs (1 ATR de separação)
    ema_diff = abs(df['ema_short'] - df['ema_long'])
    breakout_filter = ema_diff >= df['atr']
    
    # Filtro 5: RSI não em extremos
    rsi_filter = (df['rsi'] > 25) & (df['rsi'] < 75)
    
    # Confluence: pelo menos 4 dos 5 filtros
    filters = [atr_filter, volume_filter, gradient_filter, breakout_filter, rsi_filter]
    confluence_score = sum(f.fillna(False).astype(int) for f in filters)
    final_filter = confluence_score >= 4
    
    return df[final_filter & df['atr_pct'].notna() & df['ema_short'].notna()]

def generate_trading_signals(df, config):
    """Gera sinais de trading"""
    signals = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Determinar direção baseado no gradient
        if row['ema_short_grad'] >= config.gradient_min_long:
            side = 'LONG'
        elif row['ema_short_grad'] <= -config.gradient_min_short:
            side = 'SHORT'
        else:
            continue
        
        # Adicionar sinal
        signals.append({
            'timestamp': row['timestamp'],
            'side': side,
            'entry_price': row['close'],
            'atr': row['atr'],
            'atr_pct': row['atr_pct'],
            'gradient': row['ema_short_grad']
        })
    
    return signals

def simulate_trades(signals, config):
    """Simula execução dos trades"""
    trades = []
    
    for signal in signals:
        entry_price = signal['entry_price']
        side = signal['side']
        
        # Calcular preços de saída
        if side == 'LONG':
            take_profit_price = entry_price * (1 + config.take_profit_pct)
            stop_loss_price = entry_price * (1 - config.stop_loss_pct)
        else:  # SHORT
            take_profit_price = entry_price * (1 - config.take_profit_pct)
            stop_loss_price = entry_price * (1 + config.stop_loss_pct)
        
        # Simular saída aleatória (para simplificar)
        # Em um backtest real, seria baseado em dados futuros
        outcome_probability = np.random.random()
        
        # 55% de chance de take profit, 35% stop loss, 10% emergency
        if outcome_probability < 0.55:
            exit_price = take_profit_price
            pnl_pct = config.take_profit_pct * 100
            exit_reason = 'take_profit'
        elif outcome_probability < 0.90:
            exit_price = stop_loss_price
            pnl_pct = -config.stop_loss_pct * 100
            exit_reason = 'stop_loss'
        else:
            # Emergency stop
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

def run_simple_simulation():
    """Executa simulação simplificada"""
    print("🚀 SIMULAÇÃO SIMPLIFICADA DE 1 ANO - TRADINGV4 MEGA RESTRITIVO")
    print("=" * 70)
    
    config = SimpleSimulationConfig()
    
    print(f"💰 Capital inicial: ${config.initial_capital}")
    print(f"📊 Tamanho da posição: ${config.position_size}")
    print(f"🎯 Take Profit: {config.take_profit_pct*100}%")
    print(f"🛑 Stop Loss: {config.stop_loss_pct*100}%")
    print(f"⚠️  Emergency Stop: ${config.emergency_stop}")
    
    # Assets para testar
    assets = {
        'BTC': 95000,
        'ETH': 3500,
        'SOL': 180,
        'AVAX': 35,
        'LINK': 23
    }
    
    all_results = []
    
    for asset_name, start_price in assets.items():
        print(f"\n🔄 Simulando {asset_name}...")
        
        try:
            # Gerar dados
            df = generate_crypto_data(asset_name, days=365, start_price=start_price)
            print(f"  📈 {len(df)} pontos de dados gerados")
            
            # Calcular indicadores
            df_with_indicators = calculate_indicators(df, config)
            print(f"  🔧 Indicadores calculados")
            
            # Aplicar filtros
            filtered_df = apply_mega_filters(df_with_indicators, config)
            print(f"  🔍 {len(filtered_df)} sinais após filtros MEGA ({len(filtered_df)/len(df)*100:.1f}%)")
            
            if len(filtered_df) < 5:
                print(f"  ❌ Poucos sinais para {asset_name}")
                continue
            
            # Gerar sinais
            signals = generate_trading_signals(filtered_df, config)
            print(f"  📊 {len(signals)} sinais de trading gerados")
            
            if len(signals) == 0:
                print(f"  ❌ Nenhum sinal para {asset_name}")
                continue
            
            # Simular trades
            trades = simulate_trades(signals, config)
            print(f"  💼 {len(trades)} trades simulados")
            
            # Calcular métricas
            total_pnl = sum(t['pnl_dollars'] for t in trades)
            wins = [t for t in trades if t['pnl_dollars'] > 0]
            losses = [t for t in trades if t['pnl_dollars'] < 0]
            
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            
            result = {
                'asset': asset_name,
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_signals': len(filtered_df),
                'filter_efficiency': len(filtered_df) / len(df) * 100
            }
            
            all_results.append(result)
            print(f"  ✅ PNL: ${total_pnl:.2f}, Win Rate: {win_rate:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Erro em {asset_name}: {e}")
    
    # Consolidar resultados
    if not all_results:
        print("\n❌ Nenhuma simulação bem-sucedida")
        return
    
    print("\n" + "="*70)
    print("📊 RESULTADOS CONSOLIDADOS")
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
    
    # Detalhes por ativo
    print("\n📋 DETALHES POR ATIVO:")
    print("-" * 70)
    print(f"{'Asset':<8} {'Trades':<7} {'PNL($)':<8} {'Win%':<6} {'Signals':<8} {'Filter%':<8}")
    print("-" * 70)
    
    for r in sorted(all_results, key=lambda x: x['total_pnl'], reverse=True):
        print(f"{r['asset']:<8} {r['total_trades']:<7} {r['total_pnl']:<8.2f} {r['win_rate']:<6.1f} {r['total_signals']:<8} {r['filter_efficiency']:<8.1f}")
    
    # Projeções
    if roi > 0:
        monthly_roi = roi / 12
        projected_6m = config.initial_capital * (1 + (monthly_roi * 6) / 100)
        projected_2y = config.initial_capital * ((1 + roi / 100) ** 2)
        
        print(f"\n🔮 PROJEÇÕES:")
        print(f"📊 ROI Mensal: {monthly_roi:.2f}%")
        print(f"📈 Capital em 6 meses: ${projected_6m:.2f}")
        print(f"📈 Capital em 2 anos: ${projected_2y:.2f}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simulacao_simples_{timestamp}.json"
    
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
        'by_asset': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Resultados salvos: {results_file}")
    print("✅ Simulação concluída!")

if __name__ == "__main__":
    run_simple_simulation()
