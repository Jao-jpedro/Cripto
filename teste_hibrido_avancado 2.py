#!/usr/bin/env python3
"""
🚀 TESTE HÍBRIDO: MELHOR SAÍDA INTELIGENTE + OTIMIZAÇÕES EXTRAS
Combinando DNA Trailing Stop com melhorias adicionais
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

# CONFIGURAÇÕES HÍBRIDAS OPTIMIZADAS
HYBRID_CONFIGS = {
    "DNA_TRAILING_ULTRA": {
        "name": "DNA Trailing Ultra",
        "stop_loss": 0.0015,           # SL mais apertado
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.25,        # Confluência mais baixa
        "volume_multiplier": 0.008,    # Volume mais sensível
        "atr_min": 0.0008, "atr_max": 35.0, "use_max_leverage": True,
        "exit_strategy": "trailing_stop",
        "trailing_stop_pct": 0.6,      # Trailing mais apertado
        "min_profit": 0.3              # Lucro mínimo menor
    },
    
    "DNA_TRAILING_ADAPTIVE": {
        "name": "DNA Trailing Adaptive",
        "stop_loss": 0.0018,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.28,
        "volume_multiplier": 0.009,
        "atr_min": 0.0009, "atr_max": 32.0, "use_max_leverage": True,
        "exit_strategy": "trailing_adaptive",
        "trailing_base": 0.8,          # Trailing base
        "trailing_volatility_adj": True, # Ajuste por volatilidade
        "min_profit": 0.4,
        "profit_acceleration": True     # Acelera trailing em alta
    },
    
    "DNA_TRAILING_MOMENTUM": {
        "name": "DNA Trailing Momentum",
        "stop_loss": 0.0016,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.27,
        "volume_multiplier": 0.009,
        "atr_min": 0.0007, "atr_max": 33.0, "use_max_leverage": True,
        "exit_strategy": "trailing_momentum",
        "trailing_stop_pct": 0.7,
        "momentum_boost": True,        # Boost em momentum forte
        "momentum_threshold": 1.5,     # 1.5% momentum para boost
        "min_profit": 0.35
    },
    
    "DNA_SMART_TRAILING": {
        "name": "DNA Smart Trailing",
        "stop_loss": 0.0017,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.26,
        "volume_multiplier": 0.0085,
        "atr_min": 0.0008, "atr_max": 34.0, "use_max_leverage": True,
        "exit_strategy": "smart_trailing",
        "base_trailing": 0.8,
        "smart_adjustments": True,
        "volume_factor": True,         # Ajuste por volume
        "volatility_factor": True,     # Ajuste por volatilidade
        "time_factor": True,           # Trailing melhora com tempo
        "min_profit": 0.4
    },
    
    "DNA_MICRO_TRAILING": {
        "name": "DNA Micro Trailing",
        "stop_loss": 0.0012,           # SL super agressivo
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.2,         # Confluência muito baixa
        "volume_multiplier": 0.006,    # Volume super sensível
        "atr_min": 0.0005, "atr_max": 40.0, "use_max_leverage": True,
        "exit_strategy": "micro_trailing",
        "trailing_stop_pct": 0.4,      # Trailing super apertado
        "min_profit": 0.2,             # Lucro mínimo muito baixo
        "micro_management": True
    },
    
    "DNA_QUANTUM_TRAILING": {
        "name": "DNA Quantum Trailing",
        "stop_loss": 0.0014,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.22,
        "volume_multiplier": 0.007,
        "atr_min": 0.0006, "atr_max": 38.0, "use_max_leverage": True,
        "exit_strategy": "quantum_trailing",
        "quantum_levels": [0.3, 0.6, 1.0, 1.5, 2.0], # Níveis quânticos
        "quantum_trailing": [0.3, 0.5, 0.7, 0.9, 1.2], # Trailing por nível
        "level_profits": True
    },
    
    "DNA_NEURAL_TRAILING": {
        "name": "DNA Neural Trailing",
        "stop_loss": 0.0019,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.29,
        "volume_multiplier": 0.01,
        "atr_min": 0.001, "atr_max": 31.0, "use_max_leverage": True,
        "exit_strategy": "neural_trailing",
        "neural_weights": {
            "momentum": 0.3,
            "volume": 0.25,
            "volatility": 0.2,
            "time": 0.15,
            "profit": 0.1
        },
        "adaptive_learning": True,
        "min_profit": 0.45
    },
    
    "DNA_FIBONACCI_TRAILING": {
        "name": "DNA Fibonacci Trailing",
        "stop_loss": 0.0016,
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.26,
        "volume_multiplier": 0.0085,
        "atr_min": 0.0008, "atr_max": 33.0, "use_max_leverage": True,
        "exit_strategy": "fibonacci_trailing",
        "fib_levels": [0.382, 0.618, 1.0, 1.618, 2.618],
        "fib_trailing": [0.2, 0.4, 0.6, 0.8, 1.0],
        "partial_exits": True,
        "min_profit": 0.38
    },
    
    "DNA_ULTIMATE_HYBRID": {
        "name": "DNA Ultimate Hybrid",
        "stop_loss": 0.0013,           # Mais agressivo que o melhor
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.18,        # Muito mais baixo
        "volume_multiplier": 0.005,    # Super sensível
        "atr_min": 0.0004, "atr_max": 45.0, "use_max_leverage": True,
        "exit_strategy": "ultimate_hybrid",
        
        # Multi-estratégia
        "use_trailing": True, "trailing_pct": 0.5,
        "use_momentum": True, "momentum_exit": True,
        "use_volume": True, "volume_exit": True,
        "use_time": True, "time_decay": True,
        "use_fib": True, "fib_exits": True,
        
        # Parâmetros ultra otimizados
        "min_profit": 0.15,
        "ultra_responsive": True,
        "max_position_time": 6         # Máximo 6 horas
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
    """Indicadores avançados para estratégias híbridas"""
    
    # Indicadores básicos
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_gradient'] = df['ema_short'].pct_change() * 100
    
    # Volume avançado
    df['vol_ma'] = df['volume'].rolling(window=3).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma']
    df['vol_ma_long'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = df['volume'] / df['vol_ma_long']
    
    # ATR e volatilidade
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=3).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    df['atr_ma'] = df['atr_pct'].rolling(window=5).mean()
    df['volatility_spike'] = df['atr_pct'] / df['atr_ma']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum avançado
    df['price_momentum'] = df['close'].pct_change() * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_ma'] = df['price_momentum'].rolling(window=3).mean()
    
    # Indicadores extras para híbridos
    df['price_acceleration'] = df['price_momentum'].diff()
    df['volume_momentum'] = df['vol_surge'].rolling(window=3).mean()
    df['volatility_trend'] = df['atr_pct'].rolling(window=3).mean()
    
    return df

def realistic_entry_condition_enhanced(row, config) -> Tuple[bool, str]:
    """Condições de entrada aprimoradas para configurações híbridas"""
    confluence_score = 0
    max_score = 12  # Aumentado para mais precisão
    reasons = []
    
    # 1. EMA System (peso 3)
    if row.ema_short > row.ema_long:
        confluence_score += 2.5
        reasons.append("EMA")
        if row.ema_gradient > 0.008:  # Mais sensível
            confluence_score += 0.8
            reasons.append("Grad+")
    
    # 2. Micro Breakout (peso 3)
    if row.close > row.ema_short * 1.0008:  # Breakout mais micro
        confluence_score += 2.8
        reasons.append("μBreak+")
    elif row.close > row.ema_short:
        confluence_score += 1.5
        reasons.append("Break")
    
    # 3. Volume Ultra-Sensível (peso 2.5)
    if hasattr(row, 'vol_surge') and row.vol_surge > config['volume_multiplier']:
        confluence_score += 2.5
        reasons.append("Vol")
        if row.vol_surge > config['volume_multiplier'] * 2:
            confluence_score += 0.5
            reasons.append("Vol++")
    elif hasattr(row, 'vol_surge') and row.vol_surge > 1:
        confluence_score += 1
        reasons.append("Vol-")
    
    # 4. ATR Expandido (peso 2)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 2
        reasons.append("ATR")
        # Bonus para volatilidade ideal
        if 0.5 <= row.atr_pct <= 3.0:
            confluence_score += 0.5
            reasons.append("ATR+")
    
    # 5. RSI Melhorado (peso 1.5)
    if pd.notna(row.rsi) and 3 <= row.rsi <= 97:  # Range expandido
        confluence_score += 1.5
        reasons.append("RSI")
    
    # 6. Momentum Extra (peso 1)
    if hasattr(row, 'price_momentum') and row.price_momentum > 0:
        confluence_score += 0.8
        reasons.append("Mom")
        if row.price_momentum > 0.5:
            confluence_score += 0.4
            reasons.append("Mom+")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/12 [{','.join(reasons[:4])}]"
    
    return is_valid, reason

def advanced_trailing_logic(current_price, position, current_row, config, bars_held):
    """Lógica de trailing stop avançada para configurações híbridas"""
    strategy = config.get('exit_strategy', 'trailing_stop')
    entry_price = position['entry_price']
    current_profit_pct = (current_price - entry_price) / entry_price
    
    # Verificar lucro mínimo
    min_profit = config.get('min_profit', 0) / 100
    if current_profit_pct < min_profit:
        return False, None
    
    # Inicializar highest_price se não existir
    if not hasattr(position, 'highest_price'):
        position['highest_price'] = current_price
        position['trailing_active'] = False
    
    # Atualizar highest_price
    if current_price > position['highest_price']:
        position['highest_price'] = current_price
        position['trailing_active'] = True
    
    if strategy == "trailing_adaptive":
        # Trailing que se adapta à volatilidade
        base_trailing = config.get('trailing_base', 0.8) / 100
        
        if config.get('trailing_volatility_adj', False):
            # Ajustar trailing baseado na volatilidade
            if hasattr(current_row, 'atr_pct'):
                if current_row.atr_pct > 2.0:  # Alta volatilidade
                    trailing_pct = base_trailing * 1.5  # Trailing mais largo
                elif current_row.atr_pct < 0.5:  # Baixa volatilidade
                    trailing_pct = base_trailing * 0.7  # Trailing mais apertado
                else:
                    trailing_pct = base_trailing
            else:
                trailing_pct = base_trailing
        else:
            trailing_pct = base_trailing
        
        # Aceleração de lucro
        if config.get('profit_acceleration', False) and current_profit_pct > 0.015:  # 1.5%+
            trailing_pct *= 0.8  # Trailing mais apertado em alta
        
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"ADAPTIVE_TRAIL_{trailing_pct*100:.1f}%"
    
    elif strategy == "trailing_momentum":
        # Trailing com boost de momentum
        base_trailing = config.get('trailing_stop_pct', 0.7) / 100
        
        if config.get('momentum_boost', False):
            threshold = config.get('momentum_threshold', 1.5)
            if hasattr(current_row, 'price_momentum') and current_row.price_momentum > threshold:
                trailing_pct = base_trailing * 0.6  # 40% mais apertado em momentum forte
            else:
                trailing_pct = base_trailing
        else:
            trailing_pct = base_trailing
        
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"MOMENTUM_TRAIL_{trailing_pct*100:.1f}%"
    
    elif strategy == "smart_trailing":
        # Trailing inteligente com múltiplos fatores
        base_trailing = config.get('base_trailing', 0.8) / 100
        adjustment_factor = 1.0
        
        if config.get('smart_adjustments', False):
            # Ajuste por volume
            if config.get('volume_factor', False) and hasattr(current_row, 'volume_spike'):
                if current_row.volume_spike > 3.0:  # Volume alto = trailing mais largo
                    adjustment_factor *= 1.3
                elif current_row.volume_spike < 0.5:  # Volume baixo = trailing mais apertado
                    adjustment_factor *= 0.8
            
            # Ajuste por volatilidade
            if config.get('volatility_factor', False) and hasattr(current_row, 'volatility_spike'):
                if current_row.volatility_spike > 2.0:  # Volatilidade alta
                    adjustment_factor *= 1.2
                elif current_row.volatility_spike < 0.7:  # Volatilidade baixa
                    adjustment_factor *= 0.9
            
            # Ajuste por tempo
            if config.get('time_factor', False):
                if bars_held > 4:  # Posição antiga = trailing mais apertado
                    adjustment_factor *= 0.85
                elif bars_held < 2:  # Posição nova = trailing mais largo
                    adjustment_factor *= 1.1
        
        trailing_pct = base_trailing * adjustment_factor
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"SMART_TRAIL_{trailing_pct*100:.1f}%"
    
    elif strategy == "micro_trailing":
        # Trailing super apertado para capturar micro movimentos
        trailing_pct = config.get('trailing_stop_pct', 0.4) / 100
        
        if config.get('micro_management', False):
            # Micro ajustes baseados em tick por tick
            if current_profit_pct > 0.01:  # 1%+ = trailing mais apertado
                trailing_pct *= 0.8
            elif current_profit_pct > 0.005:  # 0.5%+ = trailing normal
                trailing_pct *= 1.0
            else:  # <0.5% = trailing um pouco mais largo
                trailing_pct *= 1.2
        
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"MICRO_TRAIL_{trailing_pct*100:.2f}%"
    
    elif strategy == "quantum_trailing":
        # Trailing em níveis quânticos
        quantum_levels = config.get('quantum_levels', [0.3, 0.6, 1.0, 1.5, 2.0])
        quantum_trailing = config.get('quantum_trailing', [0.3, 0.5, 0.7, 0.9, 1.2])
        
        # Determinar nível quântico atual
        current_level = 0
        for i, level in enumerate(quantum_levels):
            if current_profit_pct >= level / 100:
                current_level = i
        
        if current_level < len(quantum_trailing):
            trailing_pct = quantum_trailing[current_level] / 100
            trailing_stop = position['highest_price'] * (1 - trailing_pct)
            
            if current_price <= trailing_stop and position['trailing_active']:
                return True, f"QUANTUM_L{current_level}_{trailing_pct*100:.1f}%"
    
    elif strategy == "neural_trailing":
        # Trailing com rede neural simulada
        weights = config.get('neural_weights', {})
        base_trailing = 0.8 / 100
        
        neural_factor = 0
        
        # Input: momentum
        if hasattr(current_row, 'price_momentum'):
            momentum_norm = max(-1, min(1, current_row.price_momentum / 3))  # Normalizar
            neural_factor += momentum_norm * weights.get('momentum', 0.3)
        
        # Input: volume
        if hasattr(current_row, 'volume_spike'):
            volume_norm = max(0, min(1, (current_row.volume_spike - 1) / 4))  # Normalizar
            neural_factor += volume_norm * weights.get('volume', 0.25)
        
        # Input: volatilidade
        if hasattr(current_row, 'volatility_spike'):
            vol_norm = max(0, min(1, (current_row.volatility_spike - 1) / 3))  # Normalizar
            neural_factor += vol_norm * weights.get('volatility', 0.2)
        
        # Input: tempo
        time_norm = min(1, bars_held / 8)  # Normalizar por 8 horas
        neural_factor += time_norm * weights.get('time', 0.15)
        
        # Input: lucro
        profit_norm = min(1, current_profit_pct / 0.05)  # Normalizar por 5%
        neural_factor += profit_norm * weights.get('profit', 0.1)
        
        # Aplicar função de ativação (sigmoid simplificada)
        activation = 1 / (1 + np.exp(-neural_factor))
        trailing_pct = base_trailing * (0.5 + activation)  # 0.5x a 1.5x do base
        
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"NEURAL_TRAIL_{trailing_pct*100:.1f}%"
    
    elif strategy == "fibonacci_trailing":
        # Trailing baseado em níveis de Fibonacci
        fib_levels = config.get('fib_levels', [0.382, 0.618, 1.0, 1.618, 2.618])
        fib_trailing = config.get('fib_trailing', [0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Encontrar nível de Fibonacci atual
        current_fib_level = 0
        for i, level in enumerate(fib_levels):
            if current_profit_pct >= level / 100:
                current_fib_level = i
        
        if current_fib_level < len(fib_trailing):
            trailing_pct = fib_trailing[current_fib_level] / 100
            trailing_stop = position['highest_price'] * (1 - trailing_pct)
            
            if current_price <= trailing_stop and position['trailing_active']:
                return True, f"FIB_TRAIL_L{current_fib_level}_{trailing_pct*100:.1f}%"
    
    elif strategy == "ultimate_hybrid":
        # Estratégia híbrida definitiva
        base_trailing = config.get('trailing_pct', 0.5) / 100
        
        # Multi-factor trailing
        if config.get('ultra_responsive', False):
            # Ajustes ultra responsivos
            factor = 1.0
            
            # Momentum factor
            if hasattr(current_row, 'price_momentum'):
                if current_row.price_momentum > 1.0:
                    factor *= 0.7  # Trailing mais apertado
                elif current_row.price_momentum < -0.5:
                    factor *= 1.3  # Trailing mais largo
            
            # Volume factor
            if hasattr(current_row, 'volume_spike'):
                if current_row.volume_spike > 2.5:
                    factor *= 1.2  # Trailing mais largo
                elif current_row.volume_spike < 0.8:
                    factor *= 0.8  # Trailing mais apertado
            
            # Time factor
            max_time = config.get('max_position_time', 6)
            if bars_held >= max_time:
                return True, "TIME_EXIT"
            
            time_factor = 1 - (bars_held / max_time) * 0.3  # Trailing fica mais apertado
            factor *= time_factor
            
            trailing_pct = base_trailing * factor
        else:
            trailing_pct = base_trailing
        
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"ULTIMATE_TRAIL_{trailing_pct*100:.2f}%"
    
    else:
        # Trailing stop padrão
        trailing_pct = config.get('trailing_stop_pct', 0.8) / 100
        trailing_stop = position['highest_price'] * (1 - trailing_pct)
        
        if current_price <= trailing_stop and position['trailing_active']:
            return True, f"TRAILING_{trailing_pct*100:.1f}%"
    
    return False, None

def simulate_hybrid_trading(df, asset, config):
    """Simulação com estratégias híbridas avançadas"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 10):
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = realistic_entry_condition_enhanced(row, config)
            
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
            
            # Timeout máximo
            elif bars_held >= 12:
                exit_reason = 'TIMEOUT'
            
            # Estratégia de trailing avançada
            else:
                should_exit, trail_reason = advanced_trailing_logic(current_price, position, row, config, bars_held)
                if should_exit:
                    exit_reason = trail_reason
            
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

def run_hybrid_test(config_name, config):
    print(f"\n🚀 TESTANDO: {config['name']}")
    print("="*70)
    
    all_trades = []
    total_pnl = 0
    profitable_assets = 0
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        leverage = get_leverage_for_asset(asset, config)
        
        df = calculate_advanced_indicators(df, config)
        trades = simulate_hybrid_trading(df, asset, config)
        
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
        'profitable_assets': profitable_assets
    }

def main():
    print("🚀 TESTE HÍBRIDO: TRAILING STOP APRIMORADO")
    print("="*80)
    print("🎯 OBJETIVO: SUPERAR DNA REALISTA OTIMIZADO (+1.377,3% ROI)")
    print("💡 ESTRATÉGIA: Trailing Stop + Otimizações Avançadas")
    
    # Benchmarks
    dna_realista = 1377.3
    best_trailing = 1375.3
    
    all_results = []
    
    # Testar configurações híbridas
    for config_name, config in HYBRID_CONFIGS.items():
        result = run_hybrid_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING ESTRATÉGIAS HÍBRIDAS AVANÇADAS")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi'], reverse=True)
    
    print(f"\nPos | Configuração           | ROI      | PnL      | Trades | WR    | Assets+ | vs DNA Base | vs Trail Base")
    print("-" * 110)
    
    for i, result in enumerate(all_results, 1):
        name = result['config']['name'][:20]
        roi = result['portfolio_roi']
        pnl = result['total_pnl']
        trades = result['total_trades']
        wr = result['win_rate']
        assets = result['profitable_assets']
        vs_dna = roi - dna_realista
        vs_trail = roi - best_trailing
        
        if i == 1:
            emoji = "👑"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        else:
            emoji = f"{i:2}"
            
        vs_dna_str = f"+{vs_dna:.1f}%" if vs_dna > 0 else f"{vs_dna:.1f}%"
        vs_trail_str = f"+{vs_trail:.1f}%" if vs_trail > 0 else f"{vs_trail:.1f}%"
            
        print(f"{emoji} | {name:<20} | {roi:+7.1f}% | ${pnl:+7.2f} | {trades:6} | {wr:4.1f}% | {assets:2}/16 | {vs_dna_str:>8} | {vs_trail_str:>10}")
    
    # Análise final
    champion = all_results[0]
    improvement_dna = champion['portfolio_roi'] - dna_realista
    improvement_trail = champion['portfolio_roi'] - best_trailing
    
    print(f"\n📊 ANÁLISE COMPARATIVA:")
    print(f"   📈 DNA Realista Base: +{dna_realista:.1f}%")
    print(f"   🎯 DNA Trailing Simples: +{best_trailing:.1f}%")
    print(f"   🚀 Melhor Híbrido: +{champion['portfolio_roi']:.1f}%")
    
    if improvement_dna > 0:
        print(f"   ✅ MELHORIA vs DNA REALISTA: +{improvement_dna:.1f}% ({(improvement_dna/dna_realista)*100:+.2f}%)")
        print(f"   🎊 NOVO RECORDE ABSOLUTO!")
    else:
        print(f"   ❌ vs DNA Realista: {improvement_dna:.1f}% ({(improvement_dna/dna_realista)*100:+.2f}%)")
    
    if improvement_trail > 0:
        print(f"   ✅ MELHORIA vs Trailing Simples: +{improvement_trail:.1f}%")
    else:
        print(f"   ❌ vs Trailing Simples: {improvement_trail:.1f}%")
    
    # Detalhes do campeão
    if champion['portfolio_roi'] > max(dna_realista, best_trailing):
        print(f"\n👑 NOVA ESTRATÉGIA CAMPEÃ:")
        print(f"   📛 Nome: {champion['config']['name']}")
        print(f"   💰 ROI: +{champion['portfolio_roi']:.1f}%")
        print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
        print(f"   📊 Trades: {champion['total_trades']}")
        print(f"   ✅ Assets Lucrativos: {champion['profitable_assets']}/16")
        
        # Transformação de capital
        total_capital = 64.0
        final_value = total_capital + champion['total_pnl']
        multiplier = final_value / total_capital
        
        print(f"\n💎 TRANSFORMAÇÃO DE CAPITAL:")
        print(f"   💰 Capital Inicial: $64.00")
        print(f"   🚀 Valor Final: ${final_value:.2f}")
        print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
        
        # Comparar com DNA Realista
        dna_final = 64.0 * (1 + dna_realista/100)
        extra_profit = final_value - dna_final
        print(f"   💵 Lucro Extra vs DNA Realista: ${extra_profit:+.2f}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_hibrido_avancado_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 TESTE HÍBRIDO AVANÇADO CONCLUÍDO!")

if __name__ == "__main__":
    main()
