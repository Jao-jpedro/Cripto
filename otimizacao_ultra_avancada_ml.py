#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÃO ULTRA AVANÇADA - ML & MARKET MICROSTRUCTURE
Implementação de técnicas avançadas de machine learning e análise de microestrutura de mercado
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# LEVERAGES MÁXIMOS REAIS DA HYPERLIQUID
HYPERLIQUID_MAX_LEVERAGE = {
    "BTC-USD": 40, "SOL-USD": 20, "ETH-USD": 25, "XRP-USD": 20,
    "DOGE-USD": 10, "AVAX-USD": 10, "ENA-USD": 10, "BNB-USD": 10,
    "SUI-USD": 10, "ADA-USD": 10, "LINK-USD": 10, "WLD-USD": 10,
    "AAVE-USD": 10, "CRV-USD": 10, "LTC-USD": 10, "NEAR-USD": 10
}

# CONFIGURAÇÕES ULTRA AVANÇADAS COM ML
ULTRA_ADVANCED_CONFIGS = {
    "DNA_QUANTUM_TRADER": {
        "name": "DNA Quantum Trader",
        "stop_loss": 0.0015,     # SL 0.15%
        "take_profit": 1.5,      # TP 150%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.1,
        "volume_multiplier": 0.001,
        "atr_min": 0.0005,
        "atr_max": 40.0,
        "use_ml_signals": True,
        "use_market_microstructure": True,
        "volatility_adaptive": True
    },
    
    "DNA_NEURAL_SCALPER": {
        "name": "DNA Neural Scalper",
        "stop_loss": 0.001,      # SL 0.1%
        "take_profit": 2.0,      # TP 200%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.05,
        "volume_multiplier": 0.0005,
        "atr_min": 0.0001,
        "atr_max": 50.0,
        "use_ml_signals": True,
        "use_market_microstructure": True,
        "volatility_adaptive": True,
        "neural_boost": True
    },
    
    "DNA_ADAPTIVE_AI": {
        "name": "DNA Adaptive AI",
        "stop_loss": 0.002,      # SL 0.2%
        "take_profit": 1.8,      # TP 180%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.2,
        "volume_multiplier": 0.002,
        "atr_min": 0.001,
        "atr_max": 35.0,
        "use_ml_signals": True,
        "use_market_microstructure": True,
        "volatility_adaptive": True,
        "adaptive_parameters": True
    },
    
    "DNA_MOMENTUM_ML": {
        "name": "DNA Momentum ML",
        "stop_loss": 0.0008,     # SL 0.08%
        "take_profit": 2.5,      # TP 250%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.02,
        "volume_multiplier": 0.0001,
        "atr_min": 0.0001,
        "atr_max": 60.0,
        "use_ml_signals": True,
        "use_market_microstructure": True,
        "volatility_adaptive": True,
        "momentum_focus": True
    }
}

ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def get_leverage_for_asset(asset, config):
    """Retorna o leverage correto para o asset"""
    max_leverage = HYPERLIQUID_MAX_LEVERAGE.get(asset, 10)
    
    if config.get('use_max_leverage', False):
        return max_leverage
    elif 'leverage_multiplier' in config:
        return max(1, int(max_leverage * config['leverage_multiplier']))
    else:
        return max_leverage

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

def calculate_advanced_ml_indicators(df, config):
    """Indicadores avançados com machine learning e microestrutura"""
    
    # EMAs ultra responsivas
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_gradient'] = df['ema_short'].pct_change() * 100
    df['ema_acceleration'] = df['ema_gradient'].diff()
    
    # Análise de Volume Avançada
    df['vol_ma'] = df['volume'].rolling(window=2).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma']
    df['vol_profile'] = df['volume'] * df['close']
    df['vol_momentum'] = df['vol_surge'].pct_change()
    
    # ATR e Volatilidade Adaptativa
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=2).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    df['volatility_regime'] = df['atr_pct'].rolling(window=10).std()
    
    # RSI Ultra Responsivo com ML
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_velocity'] = df['rsi'].diff()
    
    # Momentum Multi-Timeframe
    df['momentum_1'] = df['close'].pct_change() * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_avg'] = (df['momentum_1'] + df['momentum_3'] + df['momentum_5']) / 3
    
    # Microestrutura de Mercado
    if config.get('use_market_microstructure', False):
        # Spread Implícito
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Pressão de Compra/Venda
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = 1 - df['buy_pressure']
        
        # Eficiência de Preço
        df['price_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Força Direcional
        df['directional_force'] = df['buy_pressure'] * df['vol_surge']
    
    # Sinais de Machine Learning Simulados
    if config.get('use_ml_signals', False):
        # Ensemble de múltiplos sinais
        df['ml_trend'] = (
            (df['ema_short'] > df['ema_long']).astype(int) * 0.3 +
            (df['momentum_avg'] > 0).astype(int) * 0.3 +
            (df['vol_surge'] > 1.2).astype(int) * 0.2 +
            (df['rsi'] < 80).astype(int) * 0.2
        )
        
        # Neural Network Simulado
        if config.get('neural_boost', False):
            df['neural_signal'] = (
                np.tanh(df['momentum_avg'] / 10) * 0.4 +
                np.tanh(df['vol_surge'] - 1) * 0.3 +
                np.tanh((df['rsi'] - 50) / 50) * 0.3
            )
        else:
            df['neural_signal'] = 0
    
    # Adaptação de Volatilidade
    if config.get('volatility_adaptive', False):
        df['vol_adj_factor'] = 1 + (df['volatility_regime'] / 100)
    else:
        df['vol_adj_factor'] = 1
    
    return df

def ultra_advanced_entry_condition(row, config) -> Tuple[bool, str]:
    """Condição de entrada ultra avançada com ML"""
    confluence_score = 0
    max_score = 15  # Aumentado para incluir novos sinais
    reasons = []
    
    # 1. Sistema EMA Avançado (peso 3)
    if row.ema_short > row.ema_long:
        confluence_score += 2
        reasons.append("EMA")
        if row.ema_gradient > 0.01:
            confluence_score += 1
            reasons.append("Grad+")
        if hasattr(row, 'ema_acceleration') and row.ema_acceleration > 0:
            confluence_score += 0.5
            reasons.append("Acc+")
    
    # 2. Breakout Ultra Sensível (peso 3)
    if row.close > row.ema_short * 1.0005:  # 0.05% breakout
        confluence_score += 3
        reasons.append("μμBreak")
    elif row.close > row.ema_short:
        confluence_score += 1.5
        reasons.append("μBreak")
    
    # 3. Volume Inteligente (peso 2.5)
    if hasattr(row, 'vol_surge') and row.vol_surge > config['volume_multiplier']:
        confluence_score += 2.5
        reasons.append("Vol++")
        if hasattr(row, 'vol_momentum') and row.vol_momentum > 0:
            confluence_score += 0.5
            reasons.append("VolMom")
    
    # 4. Microestrutura (peso 2)
    if config.get('use_market_microstructure', False) and hasattr(row, 'buy_pressure'):
        if row.buy_pressure > 0.6:
            confluence_score += 2
            reasons.append("BuyP")
        elif row.buy_pressure > 0.5:
            confluence_score += 1
            reasons.append("BuyP-")
    
    # 5. Machine Learning (peso 2)
    if config.get('use_ml_signals', False) and hasattr(row, 'ml_trend'):
        if row.ml_trend > 0.7:
            confluence_score += 2
            reasons.append("ML++")
        elif row.ml_trend > 0.5:
            confluence_score += 1
            reasons.append("ML+")
    
    # 6. Neural Network (peso 1.5)
    if config.get('neural_boost', False) and hasattr(row, 'neural_signal'):
        if row.neural_signal > 0.3:
            confluence_score += 1.5
            reasons.append("Neural")
        elif row.neural_signal > 0:
            confluence_score += 0.5
            reasons.append("Neural-")
    
    # 7. ATR Otimizado (peso 1)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 1
        reasons.append("ATR")
    
    # 8. Momentum Multi-TF (peso 1)
    if hasattr(row, 'momentum_avg') and row.momentum_avg > 0.01:
        confluence_score += 1
        reasons.append("Mom")
    
    # Ajuste por Volatilidade
    if config.get('volatility_adaptive', False) and hasattr(row, 'vol_adj_factor'):
        confluence_score *= row.vol_adj_factor
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/15 [{','.join(reasons[:4])}]"
    
    return is_valid, reason

def simulate_ultra_advanced_trading(df, asset, config):
    """Simulação de trading ultra avançada"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    # Parâmetros adaptativos
    if config.get('adaptive_parameters', False):
        base_sl = config['stop_loss']
        base_tp = config['take_profit']
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 5):
            continue
            
        row = df.iloc[i]
        
        # Adaptação de parâmetros baseada em volatilidade
        if config.get('adaptive_parameters', False):
            vol_factor = getattr(row, 'volatility_regime', 1) or 1
            current_sl = base_sl * (1 + vol_factor * 0.5)
            current_tp = base_tp * (1 + vol_factor * 0.2)
        else:
            current_sl = config['stop_loss']
            current_tp = config['take_profit']
        
        if position is None:
            should_enter, reason = ultra_advanced_entry_condition(row, config)
            
            if should_enter:
                entry_price = row.close
                position_size = capital * leverage
                shares = position_size / entry_price
                
                stop_loss = entry_price * (1 - current_sl)
                take_profit = entry_price * (1 + current_tp)
                
                position = {
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_bar': i,
                    'entry_time': row.timestamp,
                    'capital_used': capital,
                    'leverage_used': leverage,
                    'reason': reason,
                    'sl_used': current_sl,
                    'tp_used': current_tp
                }
                
        else:
            current_price = row.close
            exit_reason = None
            
            if current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'
            elif current_price >= position['take_profit']:
                exit_reason = 'TAKE_PROFIT'
            elif i - position['entry_bar'] >= 8:  # 8 horas timeout mais agressivo
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                
                trade = {
                    'asset': asset,
                    'exit_reason': exit_reason,
                    'pnl_gross': pnl_gross,
                    'leverage_used': position['leverage_used'],
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_bars': i - position['entry_bar'],
                    'entry_reason': position['reason'],
                    'sl_used': position['sl_used'],
                    'tp_used': position['tp_used']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def run_ultra_advanced_test(config_name, config):
    print(f"\n🚀 TESTANDO: {config['name']}")
    print("="*70)
    
    all_trades = []
    total_pnl = 0
    profitable_assets = 0
    leverage_summary = {}
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        leverage = get_leverage_for_asset(asset, config)
        leverage_summary[asset] = leverage
        
        df = calculate_advanced_ml_indicators(df, config)
        trades = simulate_ultra_advanced_trading(df, asset, config)
        
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
            
            print(f"   {status} {asset}: {len(trades)} trades | {leverage}x | {win_rate:.1f}% WR | {roi:+.1f}% ROI")
            
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
        'profitable_assets': profitable_assets,
        'leverage_summary': leverage_summary
    }

def main():
    print("🚀 OTIMIZAÇÃO ULTRA AVANÇADA - ML & MICROSTRUCTURE")
    print("="*80)
    print("🧠 USANDO: Machine Learning + Neural Networks + Microestrutura")
    
    all_results = []
    
    # Testar todas as configurações ultra avançadas
    for config_name, config in ULTRA_ADVANCED_CONFIGS.items():
        result = run_ultra_advanced_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING ULTRA AVANÇADO")
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
        
        if i == 1:
            emoji = "👑"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        else:
            emoji = f"{i:2}"
            
        print(f"{emoji} | {name:<20} | {roi:+7.1f}% | ${pnl:+7.2f} | {trades:6} | {wr:4.1f}% | {assets:2}/16")
    
    # Configuração vencedora
    champion = all_results[0]
    
    print(f"\n👑 CONFIGURAÇÃO ULTRA AVANÇADA VENCEDORA:")
    print(f"   📛 Nome: {champion['config']['name']}")
    print(f"   🚀 ROI: {champion['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${champion['total_pnl']:+.2f}")
    print(f"   📊 Trades: {champion['total_trades']}")
    print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
    print(f"   ✅ Assets Lucrativos: {champion['profitable_assets']}/16")
    
    print(f"\n🔧 PARÂMETROS ULTRA AVANÇADOS:")
    config = champion['config']
    print(f"   🛑 Stop Loss: {config['stop_loss']*100:.3f}%")
    print(f"   🎯 Take Profit: {config['take_profit']*100:.1f}%")
    print(f"   🧠 ML Signals: {config.get('use_ml_signals', False)}")
    print(f"   🔬 Microstructure: {config.get('use_market_microstructure', False)}")
    print(f"   📊 Volatility Adaptive: {config.get('volatility_adaptive', False)}")
    print(f"   🧬 Neural Boost: {config.get('neural_boost', False)}")
    
    # Transformação do capital
    total_capital = len(ASSETS) * 4.0
    final_value = total_capital + champion['total_pnl']
    multiplier = final_value / total_capital
    
    print(f"\n💎 TRANSFORMAÇÃO ULTRA AVANÇADA:")
    print(f"   💰 Capital Inicial: ${total_capital:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    # Comparação com versão anterior
    previous_roi = 1377.3  # ROI da versão realista
    improvement = champion['portfolio_roi'] - previous_roi
    
    print(f"\n📈 MELHORIA vs VERSÃO ANTERIOR:")
    print(f"   📊 ROI Anterior: +{previous_roi:.1f}%")
    print(f"   🚀 ROI Atual: +{champion['portfolio_roi']:.1f}%")
    print(f"   ⬆️ Melhoria: +{improvement:.1f}%")
    print(f"   📊 Melhoria Relativa: +{(improvement/previous_roi)*100:.1f}%")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"otimizacao_ultra_avancada_ml_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 OTIMIZAÇÃO ULTRA AVANÇADA CONCLUÍDA!")
    print("🧠 Sistema com IA e Machine Learning!")
    print("="*80)

if __name__ == "__main__":
    main()
