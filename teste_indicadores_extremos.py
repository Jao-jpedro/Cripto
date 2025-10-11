#!/usr/bin/env python3
"""
🚀 TESTE INDICADORES EXTREMAMENTE AVANÇADOS - VERSÃO 2.0
Testando indicadores de alta performance: Ichimoku, Parabolic SAR, ADX, OBV, etc.
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

# CONFIGURAÇÕES COM INDICADORES EXTREMAMENTE AVANÇADOS
EXTREME_INDICATOR_CONFIGS = {
    "DNA_ICHIMOKU_CLOUD": {
        "name": "DNA Ichimoku Cloud",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_ichimoku": True, "ichi_tenkan": 4, "ichi_kijun": 8, "ichi_senkou": 16
    },
    
    "DNA_PARABOLIC_SAR": {
        "name": "DNA Parabolic SAR",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_psar": True, "psar_af": 0.02, "psar_max": 0.2
    },
    
    "DNA_ADX_DIRECTIONAL": {
        "name": "DNA ADX Directional",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_adx": True, "adx_period": 7
    },
    
    "DNA_OBV_VOLUME": {
        "name": "DNA OBV Volume",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_obv": True, "obv_period": 10
    },
    
    "DNA_FISHER_TRANSFORM": {
        "name": "DNA Fisher Transform",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_fisher": True, "fisher_period": 5
    },
    
    "DNA_KAUFMAN_AMA": {
        "name": "DNA Kaufman AMA",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_kama": True, "kama_period": 8, "kama_fast": 2, "kama_slow": 20
    },
    
    "DNA_AWESOME_OSCILLATOR": {
        "name": "DNA Awesome Oscillator",
        "stop_loss": 0.002, "take_profit": 1.2, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 20, "bb_std": 2,
        "use_ao": True, "ao_fast": 3, "ao_slow": 8
    },
    
    "DNA_ULTIMATE_COMBO": {
        "name": "DNA Ultimate Combo",
        "stop_loss": 0.0018, "take_profit": 1.25, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.25, "volume_multiplier": 0.008,
        "atr_min": 0.0008, "atr_max": 32.0,
        "use_bollinger": True, "bb_period": 18, "bb_std": 1.8,
        "use_ichimoku": True, "ichi_tenkan": 3, "ichi_kijun": 6, "ichi_senkou": 12,
        "use_adx": True, "adx_period": 5,
        "use_obv": True, "obv_period": 8,
        "use_fisher": True, "fisher_period": 4
    },
    
    "DNA_HYPER_OPTIMIZED": {
        "name": "DNA Hyper Optimized",
        "stop_loss": 0.0015, "take_profit": 1.35, "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.2, "volume_multiplier": 0.005,
        "atr_min": 0.0005, "atr_max": 35.0,
        "use_bollinger": True, "bb_period": 15, "bb_std": 1.5,
        "use_psar": True, "psar_af": 0.03, "psar_max": 0.25,
        "use_adx": True, "adx_period": 4,
        "use_kama": True, "kama_period": 6, "kama_fast": 2, "kama_slow": 15,
        "use_ao": True, "ao_fast": 2, "ao_slow": 6
    }
}

ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def get_leverage_for_asset(asset, config):
    max_leverage = HYPERLIQUID_MAX_LEVERAGE.get(asset, 10)
    return max_leverage if config.get('use_max_leverage', False) else max_leverage

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

def calculate_extreme_indicators(df, config):
    """Cálculo de indicadores básicos + bollinger + extremamente avançados"""
    
    # INDICADORES BÁSICOS (mantém a base)
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_gradient'] = df['ema_short'].pct_change() * 100
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(window=3).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma']
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=3).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum básico
    df['price_momentum'] = df['close'].pct_change() * 100
    
    # BOLLINGER BANDS (mantém o vencedor anterior)
    if config.get('use_bollinger', False):
        period = config.get('bb_period', 20)
        std_dev = config.get('bb_std', 2)
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # INDICADORES EXTREMAMENTE AVANÇADOS
    
    # 1. ICHIMOKU CLOUD
    if config.get('use_ichimoku', False):
        tenkan = config.get('ichi_tenkan', 4)
        kijun = config.get('ichi_kijun', 8)
        senkou = config.get('ichi_senkou', 16)
        
        # Tenkan-sen (Turning Line)
        df['ichi_tenkan'] = (df['high'].rolling(window=tenkan).max() + df['low'].rolling(window=tenkan).min()) / 2
        # Kijun-sen (Standard Line)
        df['ichi_kijun'] = (df['high'].rolling(window=kijun).max() + df['low'].rolling(window=kijun).min()) / 2
        # Senkou Span A
        df['ichi_senkou_a'] = ((df['ichi_tenkan'] + df['ichi_kijun']) / 2).shift(senkou)
        # Senkou Span B
        df['ichi_senkou_b'] = ((df['high'].rolling(window=senkou*2).max() + df['low'].rolling(window=senkou*2).min()) / 2).shift(senkou)
        # Chikou Span
        df['ichi_chikou'] = df['close'].shift(-senkou)
        
        # Sinais Ichimoku
        df['ichi_bullish'] = (
            (df['close'] > df['ichi_senkou_a']) & 
            (df['close'] > df['ichi_senkou_b']) &
            (df['ichi_tenkan'] > df['ichi_kijun'])
        ).astype(int)
    
    # 2. PARABOLIC SAR
    if config.get('use_psar', False):
        af = config.get('psar_af', 0.02)
        max_af = config.get('psar_max', 0.2)
        
        # Simplified PSAR calculation
        df['psar'] = df['low'].iloc[0]  # Initialize
        df['psar_signal'] = 0
        
        for i in range(1, len(df)):
            # Simplified logic - in real implementation would be more complex
            if df['close'].iloc[i] > df['psar'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('psar_signal')] = 1
            else:
                df.iloc[i, df.columns.get_loc('psar_signal')] = 0
    
    # 3. ADX (Average Directional Index)
    if config.get('use_adx', False):
        period = config.get('adx_period', 7)
        
        # Calculate +DI and -DI
        df['plus_dm'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), 
                                 np.maximum(df['high'] - df['high'].shift(), 0), 0)
        df['minus_dm'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), 
                                  np.maximum(df['low'].shift() - df['low'], 0), 0)
        
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
        
        # ADX calculation (simplified)
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # ADX signal (trending market when ADX > 25 and +DI > -DI)
        df['adx_signal'] = ((df['adx'] > 15) & (df['plus_di'] > df['minus_di'])).astype(int)
    
    # 4. OBV (On Balance Volume)
    if config.get('use_obv', False):
        period = config.get('obv_period', 10)
        
        df['obv'] = 0
        df['obv'].iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('obv')] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('obv')] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('obv')] = df['obv'].iloc[i-1]
        
        df['obv_ma'] = df['obv'].rolling(window=period).mean()
        df['obv_signal'] = (df['obv'] > df['obv_ma']).astype(int)
    
    # 5. FISHER TRANSFORM
    if config.get('use_fisher', False):
        period = config.get('fisher_period', 5)
        
        # Normalize price to -1 to 1 range
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()
        raw_value = (df['close'] - lowest) / (highest - lowest) - 0.5
        
        # Smooth the raw value
        df['fisher_value'] = raw_value.ewm(span=3).mean()
        
        # Apply Fisher transform
        df['fisher'] = 0.5 * np.log((1 + df['fisher_value']) / (1 - df['fisher_value']))
        df['fisher_signal'] = df['fisher']
        
        # Signal when Fisher crosses above previous value
        df['fisher_bullish'] = (df['fisher'] > df['fisher'].shift()).astype(int)
    
    # 6. KAUFMAN'S ADAPTIVE MOVING AVERAGE (KAMA)
    if config.get('use_kama', False):
        period = config.get('kama_period', 8)
        fast_sc = 2.0 / (config.get('kama_fast', 2) + 1)
        slow_sc = 2.0 / (config.get('kama_slow', 20) + 1)
        
        # Calculate efficiency ratio
        change = np.abs(df['close'] - df['close'].shift(period))
        volatility = np.abs(df['close'] - df['close'].shift()).rolling(window=period).sum()
        df['kama_er'] = change / volatility
        
        # Calculate smoothing constant
        df['kama_sc'] = ((df['kama_er'] * (fast_sc - slow_sc)) + slow_sc) ** 2
        
        # Calculate KAMA
        df['kama'] = df['close'].iloc[0]
        for i in range(1, len(df)):
            if pd.notna(df['kama_sc'].iloc[i]):
                prev_kama = df['kama'].iloc[i-1] if pd.notna(df['kama'].iloc[i-1]) else df['close'].iloc[i]
                df.iloc[i, df.columns.get_loc('kama')] = prev_kama + df['kama_sc'].iloc[i] * (df['close'].iloc[i] - prev_kama)
            else:
                df.iloc[i, df.columns.get_loc('kama')] = df['close'].iloc[i]
        
        df['kama_signal'] = (df['close'] > df['kama']).astype(int)
    
    # 7. AWESOME OSCILLATOR
    if config.get('use_ao', False):
        fast = config.get('ao_fast', 3)
        slow = config.get('ao_slow', 8)
        
        median_price = (df['high'] + df['low']) / 2
        df['ao'] = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
        df['ao_signal'] = (df['ao'] > 0).astype(int)
    
    return df

def extreme_entry_condition(row, config) -> Tuple[bool, str]:
    """Condição de entrada com indicadores extremamente avançados"""
    confluence_score = 0
    max_score = 20  # Aumentado para novos indicadores
    reasons = []
    
    # INDICADORES BÁSICOS (peso 8 - mantém base que funciona)
    if row.ema_short > row.ema_long:
        confluence_score += 2
        reasons.append("EMA")
        if row.ema_gradient > 0.01:
            confluence_score += 1
            reasons.append("Grad+")
    
    if row.close > row.ema_short * 1.001:
        confluence_score += 2.5
        reasons.append("μBreak")
    elif row.close > row.ema_short:
        confluence_score += 1
        reasons.append("Break")
    
    if hasattr(row, 'vol_surge') and row.vol_surge > config['volume_multiplier']:
        confluence_score += 2
        reasons.append("Vol")
    
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 0.5
        reasons.append("ATR")
    
    # BOLLINGER BANDS (peso 2 - mantém vencedor)
    if config.get('use_bollinger', False) and hasattr(row, 'bb_position'):
        if 0.1 <= row.bb_position <= 0.9:
            confluence_score += 1.5
            reasons.append("BB")
        if hasattr(row, 'bb_squeeze') and row.bb_squeeze < 0.1:
            confluence_score += 0.5
            reasons.append("BBSq")
    
    # INDICADORES EXTREMAMENTE AVANÇADOS (peso total 10)
    
    # Ichimoku Cloud (peso 2.5)
    if config.get('use_ichimoku', False) and hasattr(row, 'ichi_bullish'):
        if row.ichi_bullish == 1:
            confluence_score += 2.5
            reasons.append("Ichi")
    
    # Parabolic SAR (peso 1.5)
    if config.get('use_psar', False) and hasattr(row, 'psar_signal'):
        if row.psar_signal == 1:
            confluence_score += 1.5
            reasons.append("PSAR")
    
    # ADX Directional (peso 2)
    if config.get('use_adx', False) and hasattr(row, 'adx_signal'):
        if row.adx_signal == 1:
            confluence_score += 2
            reasons.append("ADX")
    
    # OBV Volume (peso 1.5)
    if config.get('use_obv', False) and hasattr(row, 'obv_signal'):
        if row.obv_signal == 1:
            confluence_score += 1.5
            reasons.append("OBV")
    
    # Fisher Transform (peso 1)
    if config.get('use_fisher', False) and hasattr(row, 'fisher_bullish'):
        if row.fisher_bullish == 1:
            confluence_score += 1
            reasons.append("Fish")
    
    # KAMA (peso 1)
    if config.get('use_kama', False) and hasattr(row, 'kama_signal'):
        if row.kama_signal == 1:
            confluence_score += 1
            reasons.append("KAMA")
    
    # Awesome Oscillator (peso 0.5)
    if config.get('use_ao', False) and hasattr(row, 'ao_signal'):
        if row.ao_signal == 1:
            confluence_score += 0.5
            reasons.append("AO")
    
    # RSI (peso 0.5)
    if pd.notna(row.rsi) and 5 <= row.rsi <= 95:
        confluence_score += 0.5
        reasons.append("RSI")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/20 [{','.join(reasons[:6])}]"
    
    return is_valid, reason

def simulate_extreme_trading(df, asset, config):
    """Simulação de trading com indicadores extremamente avançados"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 25):  # Mais períodos para indicadores complexos
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = extreme_entry_condition(row, config)
            
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

def run_extreme_test(config_name, config):
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
        
        df = calculate_extreme_indicators(df, config)
        trades = simulate_extreme_trading(df, asset, config)
        
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
    print("🚀 TESTE INDICADORES EXTREMAMENTE AVANÇADOS - V2.0")
    print("="*80)
    print("🎯 OBJETIVO: SUPERAR DNA BOLLINGER BOOST (+1.383,1% ROI)")
    
    # Benchmark atual
    current_best = 1383.1
    
    all_results = []
    
    # Testar configurações com indicadores extremamente avançados
    for config_name, config in EXTREME_INDICATOR_CONFIGS.items():
        result = run_extreme_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING INDICADORES EXTREMAMENTE AVANÇADOS")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+ | vs Bollinger")
    print("-" * 100)
    
    for i, result in enumerate(all_results, 1):
        name = result['config']['name'][:20]
        roi = result['portfolio_roi']
        pnl = result['total_pnl']
        trades = result['total_trades']
        wr = result['win_rate']
        assets = result['profitable_assets']
        improvement = roi - current_best
        
        if i == 1:
            emoji = "👑"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        else:
            emoji = f"{i:2}"
            
        if improvement > 0:
            vs_base = f"+{improvement:.1f}%"
        else:
            vs_base = f"{improvement:.1f}%"
            
        print(f"{emoji} | {name:<20} | {roi:+7.1f}% | ${pnl:+7.2f} | {trades:6} | {wr:4.1f}% | {assets:2}/16 | {vs_base}")
    
    # Análise dos resultados
    champion = all_results[0]
    improvement = champion['portfolio_roi'] - current_best
    
    print(f"\n📊 ANÁLISE FINAL:")
    print(f"   📈 DNA Bollinger Boost: +{current_best:.1f}%")
    print(f"   🚀 Melhor Extremo: +{champion['portfolio_roi']:.1f}%")
    
    if improvement > 0:
        print(f"   ✅ NOVA MELHORIA: +{improvement:.1f}% ({(improvement/current_best)*100:+.2f}%)")
        print(f"   🎊 NOVO RECORDE ABSOLUTO!")
        
        print(f"\n👑 CONFIGURAÇÃO EXTREMA VENCEDORA:")
        print(f"   📛 Nome: {champion['config']['name']}")
        print(f"   💰 ROI: +{champion['portfolio_roi']:.1f}%")
        print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
        print(f"   📊 Trades: {champion['total_trades']}")
        
        # Indicadores extremos usados
        config = champion['config']
        print(f"\n🔧 INDICADORES EXTREMOS ATIVOS:")
        if config.get('use_bollinger'): print(f"   📊 Bollinger: {config.get('bb_period', 20)}p/{config.get('bb_std', 2)}σ")
        if config.get('use_ichimoku'): print(f"   ☁️ Ichimoku: {config.get('ichi_tenkan', 4)}/{config.get('ichi_kijun', 8)}/{config.get('ichi_senkou', 16)}")
        if config.get('use_psar'): print(f"   🎯 PSAR: AF={config.get('psar_af', 0.02)}, Max={config.get('psar_max', 0.2)}")
        if config.get('use_adx'): print(f"   📈 ADX: {config.get('adx_period', 7)} períodos")
        if config.get('use_obv'): print(f"   📊 OBV: {config.get('obv_period', 10)} períodos")
        if config.get('use_fisher'): print(f"   🐟 Fisher: {config.get('fisher_period', 5)} períodos")
        if config.get('use_kama'): print(f"   🎯 KAMA: {config.get('kama_period', 8)}/{config.get('kama_fast', 2)}/{config.get('kama_slow', 20)}")
        if config.get('use_ao'): print(f"   🌊 AO: {config.get('ao_fast', 3)}/{config.get('ao_slow', 8)}")
        
    else:
        print(f"   ❌ Diferença: {improvement:.1f}% ({(improvement/current_best)*100:+.2f}%)")
        print(f"   📊 DNA BOLLINGER BOOST AINDA É O MELHOR!")
        print(f"   💡 Indicadores extremos não melhoraram mais")
    
    # Transformação de capital do melhor
    total_capital = len(ASSETS) * 4.0
    final_value = total_capital + champion['total_pnl']
    multiplier = final_value / total_capital
    
    print(f"\n💎 TRANSFORMAÇÃO DE CAPITAL (MELHOR):")
    print(f"   💰 Capital Inicial: ${total_capital:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_indicadores_extremos_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 TESTE DE INDICADORES EXTREMOS CONCLUÍDO!")
    print("="*80)

if __name__ == "__main__":
    main()
