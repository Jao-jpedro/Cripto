#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÃO COM INDICADORES AVANÇADOS - EXPERIMENTAL
Testando indicadores adicionais para melhorar ainda mais o DNA Realista Otimizado
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

# CONFIGURAÇÕES COM INDICADORES AVANÇADOS
ADVANCED_INDICATOR_CONFIGS = {
    "DNA_BOLLINGER_BOOST": {
        "name": "DNA Bollinger Boost",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True,
        "bb_period": 20, "bb_std": 2
    },
    
    "DNA_MACD_POWER": {
        "name": "DNA MACD Power",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_macd": True,
        "macd_fast": 3, "macd_slow": 8, "macd_signal": 3
    },
    
    "DNA_STOCH_MOMENTUM": {
        "name": "DNA Stoch Momentum",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_stochastic": True,
        "stoch_k": 5, "stoch_d": 3
    },
    
    "DNA_WILLIAMS_R": {
        "name": "DNA Williams %R",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_williams_r": True,
        "williams_period": 7
    },
    
    "DNA_CCI_MOMENTUM": {
        "name": "DNA CCI Momentum",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_cci": True,
        "cci_period": 10
    },
    
    "DNA_VWAP_ENHANCED": {
        "name": "DNA VWAP Enhanced",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_vwap": True,
        "vwap_period": 20
    },
    
    "DNA_MULTI_INDICATOR": {
        "name": "DNA Multi Indicator",
        "stop_loss": 0.002,
        "take_profit": 1.2,
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.25,  # Ligeiramente menos restritivo
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0,
        "use_bollinger": True, "bb_period": 15, "bb_std": 1.8,
        "use_macd": True, "macd_fast": 3, "macd_slow": 8, "macd_signal": 3,
        "use_stochastic": True, "stoch_k": 5, "stoch_d": 3
    },
    
    "DNA_ULTRA_COMBINED": {
        "name": "DNA Ultra Combined",
        "stop_loss": 0.0015,  # Mais agressivo
        "take_profit": 1.3,   # Ligeiramente mais conservador
        "use_max_leverage": True,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.2,  # Menos restritivo
        "volume_multiplier": 0.008,
        "atr_min": 0.0008, "atr_max": 35.0,
        "use_bollinger": True, "bb_period": 12, "bb_std": 1.5,
        "use_macd": True, "macd_fast": 2, "macd_slow": 6, "macd_signal": 2,
        "use_vwap": True, "vwap_period": 15,
        "use_williams_r": True, "williams_period": 5
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

def calculate_advanced_indicators(df, config):
    """Cálculo de indicadores básicos + avançados"""
    
    # INDICADORES BÁSICOS (que já funcionam bem)
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
    
    # INDICADORES AVANÇADOS (apenas se habilitados)
    
    # 1. BOLLINGER BANDS
    if config.get('use_bollinger', False):
        period = config.get('bb_period', 20)
        std_dev = config.get('bb_std', 2)
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 2. MACD
    if config.get('use_macd', False):
        fast = config.get('macd_fast', 3)
        slow = config.get('macd_slow', 8)
        signal = config.get('macd_signal', 3)
        
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # 3. STOCHASTIC
    if config.get('use_stochastic', False):
        k_period = config.get('stoch_k', 5)
        d_period = config.get('stoch_d', 3)
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        df['stoch_signal'] = ((df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < 80)).astype(int)
    
    # 4. WILLIAMS %R
    if config.get('use_williams_r', False):
        period = config.get('williams_period', 7)
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        df['williams_signal'] = (df['williams_r'] > -80).astype(int)
    
    # 5. CCI (Commodity Channel Index)
    if config.get('use_cci', False):
        period = config.get('cci_period', 10)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - ma_tp) / (0.015 * mad)
        df['cci_signal'] = ((df['cci'] > -100) & (df['cci'] < 100)).astype(int)
    
    # 6. VWAP
    if config.get('use_vwap', False):
        period = config.get('vwap_period', 20)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_num = (typical_price * df['volume']).rolling(window=period).sum()
        vwap_den = df['volume'].rolling(window=period).sum()
        df['vwap'] = vwap_num / vwap_den
        df['vwap_signal'] = (df['close'] > df['vwap']).astype(int)
    
    return df

def advanced_indicator_entry_condition(row, config) -> Tuple[bool, str]:
    """Condição de entrada com indicadores avançados"""
    confluence_score = 0
    max_score = 16  # Aumentado para incluir novos indicadores
    reasons = []
    
    # INDICADORES BÁSICOS (base que funciona - peso 8)
    
    # 1. EMA System (peso 3)
    if row.ema_short > row.ema_long:
        confluence_score += 2
        reasons.append("EMA")
        if row.ema_gradient > 0.01:
            confluence_score += 1
            reasons.append("Grad+")
    
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
    
    # 4. ATR (peso 0.5)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 0.5
        reasons.append("ATR")
    
    # INDICADORES AVANÇADOS (peso total 8)
    
    # 5. Bollinger Bands (peso 2)
    if config.get('use_bollinger', False) and hasattr(row, 'bb_position'):
        if 0.1 <= row.bb_position <= 0.9:  # Não nos extremos
            confluence_score += 1.5
            reasons.append("BB")
        if hasattr(row, 'bb_squeeze') and row.bb_squeeze < 0.1:  # Squeeze
            confluence_score += 0.5
            reasons.append("BBSq")
    
    # 6. MACD (peso 2)
    if config.get('use_macd', False) and hasattr(row, 'macd_cross'):
        if row.macd_cross == 1:  # MACD acima do signal
            confluence_score += 1.5
            reasons.append("MACD")
        if hasattr(row, 'macd_histogram') and row.macd_histogram > 0:
            confluence_score += 0.5
            reasons.append("MACDh")
    
    # 7. Stochastic (peso 1.5)
    if config.get('use_stochastic', False) and hasattr(row, 'stoch_signal'):
        if row.stoch_signal == 1:
            confluence_score += 1.5
            reasons.append("Stoch")
    
    # 8. Williams %R (peso 1)
    if config.get('use_williams_r', False) and hasattr(row, 'williams_signal'):
        if row.williams_signal == 1:
            confluence_score += 1
            reasons.append("WillR")
    
    # 9. CCI (peso 1)
    if config.get('use_cci', False) and hasattr(row, 'cci_signal'):
        if row.cci_signal == 1:
            confluence_score += 1
            reasons.append("CCI")
    
    # 10. VWAP (peso 0.5)
    if config.get('use_vwap', False) and hasattr(row, 'vwap_signal'):
        if row.vwap_signal == 1:
            confluence_score += 0.5
            reasons.append("VWAP")
    
    # 11. RSI (peso 0.5)
    if pd.notna(row.rsi) and 5 <= row.rsi <= 95:
        confluence_score += 0.5
        reasons.append("RSI")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/16 [{','.join(reasons[:5])}]"
    
    return is_valid, reason

def simulate_advanced_trading(df, asset, config):
    """Simulação de trading com indicadores avançados"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 20):  # Mais períodos para indicadores avançados
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = advanced_indicator_entry_condition(row, config)
            
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

def run_advanced_test(config_name, config):
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
        
        df = calculate_advanced_indicators(df, config)
        trades = simulate_advanced_trading(df, asset, config)
        
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
    print("🚀 TESTE DE INDICADORES AVANÇADOS")
    print("="*80)
    print("🎯 OBJETIVO: SUPERAR DNA REALISTA OTIMIZADO (+1.377% ROI)")
    
    # Benchmark atual
    current_best = 1377.3
    
    all_results = []
    
    # Testar configurações com indicadores avançados
    for config_name, config in ADVANCED_INDICATOR_CONFIGS.items():
        result = run_advanced_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING INDICADORES AVANÇADOS")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+ | vs Base")
    print("-" * 95)
    
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
    print(f"   📈 DNA Realista Base: +{current_best:.1f}%")
    print(f"   🚀 Melhor com Indicadores: +{champion['portfolio_roi']:.1f}%")
    
    if improvement > 0:
        print(f"   ✅ MELHORIA: +{improvement:.1f}% ({(improvement/current_best)*100:+.1f}%)")
        print(f"   🎊 NOVO RECORDE! Indicadores adicionais FUNCIONARAM!")
        
        print(f"\n👑 NOVA CONFIGURAÇÃO VENCEDORA:")
        print(f"   📛 Nome: {champion['config']['name']}")
        print(f"   💰 ROI: +{champion['portfolio_roi']:.1f}%")
        print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
        print(f"   📊 Trades: {champion['total_trades']}")
        
        # Indicadores usados
        config = champion['config']
        print(f"\n🔧 INDICADORES ADICIONAIS:")
        if config.get('use_bollinger'): print(f"   📊 Bollinger Bands: {config.get('bb_period', 20)} períodos")
        if config.get('use_macd'): print(f"   📈 MACD: {config.get('macd_fast', 3)}/{config.get('macd_slow', 8)}")
        if config.get('use_stochastic'): print(f"   🎲 Stochastic: {config.get('stoch_k', 5)}/{config.get('stoch_d', 3)}")
        if config.get('use_williams_r'): print(f"   📉 Williams %R: {config.get('williams_period', 7)} períodos")
        if config.get('use_cci'): print(f"   🌊 CCI: {config.get('cci_period', 10)} períodos")
        if config.get('use_vwap'): print(f"   💧 VWAP: {config.get('vwap_period', 20)} períodos")
        
    else:
        print(f"   ❌ Diferença: {improvement:.1f}% ({(improvement/current_best)*100:+.1f}%)")
        print(f"   📊 DNA Realista AINDA É O MELHOR!")
        print(f"   💡 Indicadores adicionais não melhoraram a performance")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_indicadores_avancados_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 TESTE DE INDICADORES CONCLUÍDO!")
    print("="*80)

if __name__ == "__main__":
    main()
