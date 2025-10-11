#!/usr/bin/env python3
"""
🚀 TESTE ESTRATÉGIAS DE SAÍDA INTELIGENTE - DNA REALISTA OTIMIZADO
Testando diferentes maneiras de sair com lucro ao invés de TP fixo
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

# CONFIGURAÇÕES COM DIFERENTES ESTRATÉGIAS DE SAÍDA
SMART_EXIT_CONFIGS = {
    "DNA_TRAILING_STOP": {
        "name": "DNA Trailing Stop",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "trailing_stop",
        "trailing_stop_pct": 0.8,  # 0.8% trailing stop
        "min_profit": 0.5          # 0.5% lucro mínimo antes de ativar trailing
    },
    
    "DNA_EMA_EXIT": {
        "name": "DNA EMA Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "ema_break",
        "ema_exit_period": 3,      # Sai quando preço cruza abaixo da EMA 3
        "min_profit": 0.3          # 0.3% lucro mínimo
    },
    
    "DNA_MOMENTUM_EXIT": {
        "name": "DNA Momentum Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "momentum_reversal",
        "momentum_threshold": -0.3, # Sai quando momentum fica negativo
        "min_profit": 0.4
    },
    
    "DNA_VOLUME_EXIT": {
        "name": "DNA Volume Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "volume_spike",
        "volume_exit_threshold": 3.0, # Sai quando volume > 3x da média
        "min_profit": 0.6
    },
    
    "DNA_VOLATILITY_EXIT": {
        "name": "DNA Volatility Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "volatility_spike",
        "volatility_threshold": 2.0, # Sai quando ATR > 2x da média
        "min_profit": 0.5
    },
    
    "DNA_FIBONACCI_EXIT": {
        "name": "DNA Fibonacci Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "fibonacci_levels",
        "fib_levels": [0.382, 0.618, 1.0, 1.618], # Níveis de Fibonacci
        "partial_exit": True       # Saída parcial em cada nível
    },
    
    "DNA_TIME_DECAY_EXIT": {
        "name": "DNA Time Decay Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "time_decay",
        "max_hold_hours": 8,       # Máximo 8 horas
        "profit_decay_factor": 0.9 # TP reduz 10% a cada hora
    },
    
    "DNA_ADAPTIVE_EXIT": {
        "name": "DNA Adaptive Exit",
        "stop_loss": 0.002, "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3, "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 30.0, "use_max_leverage": True,
        "exit_strategy": "adaptive_multi",
        "min_profit": 0.4,
        "use_trailing": True, "trailing_pct": 0.6,
        "use_ema_exit": True, "ema_exit_period": 4,
        "use_momentum": True, "momentum_threshold": -0.2
    },
    
    "DNA_SMART_COMBO": {
        "name": "DNA Smart Combo",
        "stop_loss": 0.0015,       # SL mais agressivo
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.25,    # Confluência mais baixa
        "volume_multiplier": 0.008,
        "atr_min": 0.0008, "atr_max": 35.0, "use_max_leverage": True,
        "exit_strategy": "smart_hybrid",
        "base_tp": 1.5,            # TP base 150%
        "dynamic_adjustment": True,
        "volatility_multiplier": 1.2,
        "momentum_boost": True
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

def calculate_smart_indicators(df, config):
    """Indicadores para estratégias de saída inteligente"""
    
    # Indicadores básicos (DNA Realista)
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
    
    # Momentum
    df['price_momentum'] = df['close'].pct_change() * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    
    # INDICADORES PARA SAÍDAS INTELIGENTES
    
    # EMA para saída
    if 'ema_exit_period' in config:
        df['ema_exit'] = df['close'].ewm(span=config['ema_exit_period']).mean()
    
    # ATR para volatilidade
    df['atr_ma'] = df['atr_pct'].rolling(window=5).mean()
    df['volatility_spike'] = df['atr_pct'] / df['atr_ma']
    
    # Volume para saída
    df['vol_ma_long'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = df['volume'] / df['vol_ma_long']
    
    return df

def realistic_entry_condition(row, config) -> Tuple[bool, str]:
    """Mantém as condições de entrada do DNA Realista Otimizado"""
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

def smart_exit_logic(current_price, position, current_row, config, bars_held):
    """Lógica de saída inteligente baseada na estratégia configurada"""
    strategy = config.get('exit_strategy', 'fixed_tp')
    entry_price = position['entry_price']
    current_profit_pct = (current_price - entry_price) / entry_price
    
    # Verificar lucro mínimo (se configurado)
    min_profit = config.get('min_profit', 0) / 100
    if current_profit_pct < min_profit:
        return False, None
    
    if strategy == "trailing_stop":
        # Trailing Stop Loss
        if not hasattr(position, 'highest_price'):
            position['highest_price'] = current_price
        
        if current_price > position['highest_price']:
            position['highest_price'] = current_price
        
        trailing_pct = config.get('trailing_stop_pct', 0.8) / 100
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop:
            return True, "TRAILING_STOP"
    
    elif strategy == "ema_break":
        # Saída quando preço cruza abaixo da EMA
        if hasattr(current_row, 'ema_exit') and current_price < current_row.ema_exit:
            return True, "EMA_BREAK"
    
    elif strategy == "momentum_reversal":
        # Saída quando momentum reverte
        threshold = config.get('momentum_threshold', -0.3)
        if hasattr(current_row, 'price_momentum') and current_row.price_momentum < threshold:
            return True, "MOMENTUM_REVERSAL"
    
    elif strategy == "volume_spike":
        # Saída em spike de volume (possível reversão)
        threshold = config.get('volume_exit_threshold', 3.0)
        if hasattr(current_row, 'volume_spike') and current_row.volume_spike > threshold:
            return True, "VOLUME_SPIKE"
    
    elif strategy == "volatility_spike":
        # Saída quando volatilidade aumenta muito
        threshold = config.get('volatility_threshold', 2.0)
        if hasattr(current_row, 'volatility_spike') and current_row.volatility_spike > threshold:
            return True, "VOLATILITY_SPIKE"
    
    elif strategy == "fibonacci_levels":
        # Saída em níveis de Fibonacci
        fib_levels = config.get('fib_levels', [0.618, 1.0, 1.618])
        for level in fib_levels:
            if current_profit_pct >= level / 100:
                return True, f"FIBONACCI_{level}"
    
    elif strategy == "time_decay":
        # TP reduz com o tempo
        max_hours = config.get('max_hold_hours', 8)
        decay_factor = config.get('profit_decay_factor', 0.9)
        
        if bars_held >= max_hours:
            return True, "TIME_DECAY"
        
        # TP dinâmico que reduz com o tempo
        base_tp = 1.2  # 120% base
        current_tp = base_tp * (decay_factor ** bars_held)
        
        if current_profit_pct >= current_tp / 100:
            return True, f"TIME_TP_{current_tp:.1f}%"
    
    elif strategy == "adaptive_multi":
        # Combina múltiplas estratégias
        
        # Trailing stop
        if config.get('use_trailing', False):
            if not hasattr(position, 'highest_price'):
                position['highest_price'] = current_price
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            trailing_pct = config.get('trailing_pct', 0.6) / 100
            trailing_stop = position['highest_price'] * (1 - trailing_pct)
            if current_price <= trailing_stop:
                return True, "ADAPTIVE_TRAILING"
        
        # EMA break
        if config.get('use_ema_exit', False):
            if hasattr(current_row, 'ema_exit') and current_price < current_row.ema_exit:
                return True, "ADAPTIVE_EMA"
        
        # Momentum
        if config.get('use_momentum', False):
            threshold = config.get('momentum_threshold', -0.2)
            if hasattr(current_row, 'price_momentum') and current_row.price_momentum < threshold:
                return True, "ADAPTIVE_MOMENTUM"
    
    elif strategy == "smart_hybrid":
        # Estratégia híbrida inteligente
        base_tp = config.get('base_tp', 1.5) / 100
        
        # Ajuste por volatilidade
        if config.get('dynamic_adjustment', False):
            vol_mult = config.get('volatility_multiplier', 1.2)
            if hasattr(current_row, 'atr_pct'):
                if current_row.atr_pct > 2.0:  # Alta volatilidade
                    target_tp = base_tp * vol_mult
                else:
                    target_tp = base_tp
            else:
                target_tp = base_tp
        else:
            target_tp = base_tp
        
        # Boost por momentum
        if config.get('momentum_boost', False):
            if hasattr(current_row, 'price_momentum') and current_row.price_momentum > 1.0:
                target_tp *= 1.3  # 30% boost em momentum forte
        
        if current_profit_pct >= target_tp:
            return True, f"SMART_HYBRID_{target_tp*100:.1f}%"
    
    return False, None

def simulate_smart_exit_trading(df, asset, config):
    """Simulação com estratégias de saída inteligente"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 5):
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = realistic_entry_condition(row, config)
            
            if should_enter:
                entry_price = row.close
                position_size = capital * leverage
                shares = position_size / entry_price
                
                stop_loss = entry_price * (1 - config['stop_loss'])
                
                position = {
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'entry_bar': i,
                    'entry_time': row.timestamp,
                    'capital_used': capital,
                    'leverage_used': leverage,
                    'reason': reason
                }
                
        else:
            current_price = row.close
            bars_held = i - position['entry_bar']
            exit_reason = None
            
            # Stop Loss tradicional
            if current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'
            
            # Timeout (caso não tenha saída inteligente)
            elif bars_held >= 12:
                exit_reason = 'TIMEOUT'
            
            # Estratégia de saída inteligente
            else:
                should_exit, smart_reason = smart_exit_logic(current_price, position, row, config, bars_held)
                if should_exit:
                    exit_reason = smart_reason
            
            if exit_reason:
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                
                trade = {
                    'asset': asset,
                    'exit_reason': exit_reason,
                    'pnl_gross': pnl_gross,
                    'leverage_used': position['leverage_used'],
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_bars': bars_held,
                    'entry_reason': position['reason']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def run_smart_exit_test(config_name, config):
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
        
        df = calculate_smart_indicators(df, config)
        trades = simulate_smart_exit_trading(df, asset, config)
        
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
    print("🚀 TESTE ESTRATÉGIAS DE SAÍDA INTELIGENTE")
    print("="*80)
    print("🎯 OBJETIVO: SUPERAR DNA REALISTA OTIMIZADO (+1.377,3% ROI)")
    print("💡 FOCO: Mudar estratégia de TP fixo para saídas inteligentes")
    
    # Benchmark atual
    current_best = 1377.3
    
    all_results = []
    
    # Testar configurações com saídas inteligentes
    for config_name, config in SMART_EXIT_CONFIGS.items():
        result = run_smart_exit_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING ESTRATÉGIAS DE SAÍDA INTELIGENTE")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+ | vs DNA Base")
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
    print(f"   📈 DNA Realista Base: +{current_best:.1f}%")
    print(f"   🚀 Melhor Saída Inteligente: +{champion['portfolio_roi']:.1f}%")
    
    if improvement > 0:
        print(f"   ✅ MELHORIA COM SAÍDA INTELIGENTE: +{improvement:.1f}% ({(improvement/current_best)*100:+.2f}%)")
        print(f"   🎊 NOVO RECORDE COM ESTRATÉGIA DE SAÍDA!")
        
        print(f"\n👑 ESTRATÉGIA DE SAÍDA VENCEDORA:")
        print(f"   📛 Nome: {champion['config']['name']}")
        print(f"   💰 ROI: +{champion['portfolio_roi']:.1f}%")
        print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
        print(f"   📊 Trades: {champion['total_trades']}")
        
        # Detalhes da estratégia
        config = champion['config']
        print(f"\n🔧 ESTRATÉGIA DE SAÍDA:")
        print(f"   🎯 Tipo: {config.get('exit_strategy', 'N/A')}")
        if 'trailing_stop_pct' in config:
            print(f"   📉 Trailing Stop: {config['trailing_stop_pct']}%")
        if 'min_profit' in config:
            print(f"   💰 Lucro Mínimo: {config['min_profit']}%")
        if 'ema_exit_period' in config:
            print(f"   📈 EMA Saída: {config['ema_exit_period']} períodos")
        
    else:
        print(f"   ❌ Diferença: {improvement:.1f}% ({(improvement/current_best)*100:+.2f}%)")
        print(f"   📊 DNA REALISTA COM TP FIXO AINDA É MELHOR!")
        print(f"   💡 Estratégias de saída inteligente não melhoraram")
    
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
    filename = f"teste_saidas_inteligentes_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 TESTE DE SAÍDAS INTELIGENTES CONCLUÍDO!")
    print("="*80)

if __name__ == "__main__":
    main()
