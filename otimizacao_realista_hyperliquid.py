#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÃO REALISTA - LEVERAGES MÁXIMOS DA HYPERLIQUID
Usando os leverages reais disponíveis para cada asset na Hyperliquid
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# LEVERAGES MÁXIMOS REAIS DA HYPERLIQUID POR ASSET
HYPERLIQUID_MAX_LEVERAGE = {
    "BTC-USD": 40,   # BTC tem leverage máximo de 40x
    "SOL-USD": 20,   # SOL tem leverage máximo de 20x
    "ETH-USD": 25,   # ETH tem leverage máximo de 25x
    "XRP-USD": 20,   # XRP tem leverage máximo de 20x
    "DOGE-USD": 10,  # DOGE tem leverage máximo de 10x
    "AVAX-USD": 10,  # AVAX tem leverage máximo de 10x
    "ENA-USD": 10,   # ENA tem leverage máximo de 10x
    "BNB-USD": 10,   # BNB tem leverage máximo de 10x
    "SUI-USD": 10,   # SUI tem leverage máximo de 10x
    "ADA-USD": 10,   # ADA tem leverage máximo de 10x
    "LINK-USD": 10,  # LINK tem leverage máximo de 10x
    "WLD-USD": 10,   # WLD tem leverage máximo de 10x
    "AAVE-USD": 10,  # AAVE tem leverage máximo de 10x
    "CRV-USD": 10,   # CRV tem leverage máximo de 10x
    "LTC-USD": 10,   # LTC tem leverage máximo de 10x
    "NEAR-USD": 10   # NEAR tem leverage máximo de 10x
}

# CONFIGURAÇÕES REALISTAS COM LEVERAGES MÁXIMOS
REALISTIC_CONFIGS = {
    "DNA_REALISTA_OTIMIZADO": {
        "name": "DNA Realista Otimizado",
        "stop_loss": 0.002,      # SL 0.2%
        "take_profit": 1.2,      # TP 120%
        "use_max_leverage": True, # Usar leverage máximo por asset
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001,
        "atr_max": 30.0
    },
    
    "DNA_CONSERVADOR_REALISTA": {
        "name": "DNA Conservador Realista",
        "stop_loss": 0.005,      # SL 0.5%
        "take_profit": 0.8,      # TP 80%
        "leverage_multiplier": 0.8, # 80% do leverage máximo
        "ema_fast": 1,
        "ema_slow": 3,
        "rsi_period": 2,
        "min_confluence": 1.0,
        "volume_multiplier": 0.05,
        "atr_min": 0.01,
        "atr_max": 20.0
    },
    
    "DNA_EQUILIBRADO_REALISTA": {
        "name": "DNA Equilibrado Realista",
        "stop_loss": 0.003,      # SL 0.3%
        "take_profit": 1.0,      # TP 100%
        "leverage_multiplier": 0.9, # 90% do leverage máximo
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.5,
        "volume_multiplier": 0.02,
        "atr_min": 0.005,
        "atr_max": 25.0
    },
    
    "DNA_AGRESSIVO_REALISTA": {
        "name": "DNA Agressivo Realista",
        "stop_loss": 0.001,      # SL 0.1%
        "take_profit": 1.5,      # TP 150%
        "use_max_leverage": True, # Usar leverage máximo
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.2,
        "volume_multiplier": 0.005,
        "atr_min": 0.001,
        "atr_max": 35.0
    }
}

# Assets
ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def get_leverage_for_asset(asset, config):
    """Retorna o leverage correto para o asset baseado na configuração"""
    max_leverage = HYPERLIQUID_MAX_LEVERAGE.get(asset, 10)  # Default 10x
    
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

def calculate_realistic_indicators(df, config):
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
    
    # Momentum
    df['price_momentum'] = df['close'].pct_change() * 100
    
    return df

def realistic_entry_condition(row, config) -> Tuple[bool, str]:
    confluence_score = 0
    max_score = 10
    reasons = []
    
    # 1. EMA System (peso 3)
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
    
    # 3. Volume (peso 2)
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

def simulate_realistic_trading(df, asset, config):
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 3):
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = realistic_entry_condition(row, config)
            
            if should_enter:
                entry_price = row.close
                position_size = capital * leverage
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
                    'leverage_used': leverage,
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
                    'leverage_used': position['leverage_used'],
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_bars': i - position['entry_bar'],
                    'entry_reason': position['reason']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def run_realistic_test(config_name, config):
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
        
        df = calculate_realistic_indicators(df, config)
        trades = simulate_realistic_trading(df, asset, config)
        
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
    
    print(f"\n⚙️ LEVERAGES UTILIZADOS:")
    for asset, lev in leverage_summary.items():
        max_lev = HYPERLIQUID_MAX_LEVERAGE[asset]
        pct = (lev / max_lev) * 100
        print(f"   {asset}: {lev}x de {max_lev}x máximo ({pct:.0f}%)")
    
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
    print("🚀 OTIMIZAÇÃO REALISTA - LEVERAGES MÁXIMOS HYPERLIQUID")
    print("="*80)
    print("🎯 META: TESTAR COM LEVERAGES REAIS DISPONÍVEIS")
    
    all_results = []
    
    # Mostrar leverages máximos disponíveis
    print(f"\n⚙️ LEVERAGES MÁXIMOS DISPONÍVEIS NA HYPERLIQUID:")
    for asset, max_lev in HYPERLIQUID_MAX_LEVERAGE.items():
        print(f"   {asset}: {max_lev}x")
    
    # Testar todas as configurações realistas
    for config_name, config in REALISTIC_CONFIGS.items():
        result = run_realistic_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING FINAL REALISTA")
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
    
    print(f"\n👑 CONFIGURAÇÃO REALISTA VENCEDORA:")
    print(f"   📛 Nome: {champion['config']['name']}")
    print(f"   🚀 ROI: {champion['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${champion['total_pnl']:+.2f}")
    print(f"   📊 Trades: {champion['total_trades']}")
    print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
    print(f"   ✅ Assets Lucrativos: {champion['profitable_assets']}/16")
    
    print(f"\n🔧 PARÂMETROS REALISTAS VENCEDORES:")
    config = champion['config']
    print(f"   🛑 Stop Loss: {config['stop_loss']*100:.2f}%")
    print(f"   🎯 Take Profit: {config['take_profit']*100:.1f}%")
    print(f"   📈 Leverage: Baseado nos máximos da Hyperliquid")
    print(f"   🌊 EMA: {config['ema_fast']}/{config['ema_slow']}")
    print(f"   📊 RSI: {config['rsi_period']} períodos")
    print(f"   🎲 Confluência: {config['min_confluence']:.2f}/10")
    print(f"   📈 Volume: {config['volume_multiplier']:.3f}x")
    
    # Transformação do capital
    total_capital = len(ASSETS) * 4.0
    final_value = total_capital + champion['total_pnl']
    multiplier = final_value / total_capital
    
    print(f"\n💎 TRANSFORMAÇÃO REALISTA DO CAPITAL:")
    print(f"   💰 Capital Inicial: ${total_capital:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    print(f"\n⚙️ LEVERAGE MÉDIO UTILIZADO:")
    total_leverage = sum(champion['leverage_summary'].values())
    avg_leverage = total_leverage / len(champion['leverage_summary'])
    print(f"   📊 Média: {avg_leverage:.1f}x")
    print(f"   📈 Máximo usado: {max(champion['leverage_summary'].values())}x")
    print(f"   📉 Mínimo usado: {min(champion['leverage_summary'].values())}x")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"otimizacao_realista_hyperliquid_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 OTIMIZAÇÃO REALISTA CONCLUÍDA!")
    print("✅ Configuração compatível com Hyperliquid!")
    print("="*80)

if __name__ == "__main__":
    main()
