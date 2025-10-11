#!/usr/bin/env python3
"""
🚀 DNA MACRO TRAILING - OTIMIZADO PARA TAXAS HYPERLIQUID
Estratégia que considera custos reais: menos trades, mais lucro por trade
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# TAXAS REAIS DA HYPERLIQUID
HYPERLIQUID_FEES = {
    "maker_fee": 0.0002,     # 0.02%
    "taker_fee": 0.0005,     # 0.05%
    "funding_rate_avg": 0.0001,  # 0.01% por 8h
}

# LEVERAGES MÁXIMOS
HYPERLIQUID_MAX_LEVERAGE = {
    "BTC-USD": 40, "SOL-USD": 20, "ETH-USD": 25, "XRP-USD": 20,
    "DOGE-USD": 10, "AVAX-USD": 10, "ENA-USD": 10, "BNB-USD": 10,
    "SUI-USD": 10, "ADA-USD": 10, "LINK-USD": 10, "WLD-USD": 10,
    "AAVE-USD": 10, "CRV-USD": 10, "LTC-USD": 10, "NEAR-USD": 10
}

# CONFIGURAÇÕES DNA MACRO TRAILING
DNA_MACRO_CONFIGS = {
    "DNA_MACRO_CONSERVATIVE": {
        "name": "DNA Macro Conservative",
        "stop_loss": 0.008,            # 0.8% SL mais largo
        "take_profit": 2.5,            # 250% TP alto
        "ema_fast": 3, "ema_slow": 8, "rsi_period": 5,
        "min_confluence": 0.7,         # Confluência alta (seletivo)
        "volume_multiplier": 0.05,     # Volume mais exigente
        "atr_min": 0.5, "atr_max": 5.0, "use_max_leverage": True,
        "min_profit_target": 1.5,     # 1.5% mínimo para compensar taxas
        "max_trades_per_day": 2,       # Máximo 2 trades por dia
        "cooldown_hours": 8,           # 8h de cooldown entre trades
    },
    
    "DNA_MACRO_BALANCED": {
        "name": "DNA Macro Balanced",
        "stop_loss": 0.006,            # 0.6% SL
        "take_profit": 2.0,            # 200% TP
        "ema_fast": 2, "ema_slow": 5, "rsi_period": 3,
        "min_confluence": 0.6,         # Confluência moderada-alta
        "volume_multiplier": 0.03,
        "atr_min": 0.3, "atr_max": 8.0, "use_max_leverage": True,
        "min_profit_target": 1.2,     # 1.2% mínimo
        "max_trades_per_day": 3,
        "cooldown_hours": 6,
    },
    
    "DNA_MACRO_AGGRESSIVE": {
        "name": "DNA Macro Aggressive",
        "stop_loss": 0.005,            # 0.5% SL
        "take_profit": 1.8,            # 180% TP
        "ema_fast": 2, "ema_slow": 4, "rsi_period": 2,
        "min_confluence": 0.5,         # Confluência moderada
        "volume_multiplier": 0.02,
        "atr_min": 0.2, "atr_max": 10.0, "use_max_leverage": True,
        "min_profit_target": 1.0,     # 1.0% mínimo
        "max_trades_per_day": 4,
        "cooldown_hours": 4,
    },
    
    "DNA_MACRO_TRAILING_SMART": {
        "name": "DNA Macro Trailing Smart",
        "stop_loss": 0.004,            # 0.4% SL
        "ema_fast": 2, "ema_slow": 4, "rsi_period": 2,
        "min_confluence": 0.4,         # Confluência moderada-baixa
        "volume_multiplier": 0.015,
        "atr_min": 0.15, "atr_max": 12.0, "use_max_leverage": True,
        
        # TRAILING INTELIGENTE
        "exit_strategy": "smart_trailing",
        "initial_target": 1.5,        # 1.5% target inicial
        "trailing_activation": 0.8,   # Ativa trailing aos 0.8%
        "trailing_distance": 0.4,     # 0.4% de trailing
        "profit_scaling": True,       # Escala o trailing com lucro
        "max_hold_hours": 24,         # Máximo 24h
        "min_profit_target": 0.8,     # 0.8% mínimo
    },
    
    "DNA_MACRO_VOLUME_FILTER": {
        "name": "DNA Macro Volume Filter",
        "stop_loss": 0.0035,          # 0.35% SL
        "take_profit": 1.5,           # 150% TP
        "ema_fast": 1, "ema_slow": 3, "rsi_period": 2,
        "min_confluence": 0.35,
        "volume_multiplier": 0.01,
        "atr_min": 0.1, "atr_max": 15.0, "use_max_leverage": True,
        
        # FILTROS ANTI-CHURNING
        "volume_filter": True,
        "min_volume_spike": 2.0,      # Volume 2x acima da média
        "momentum_filter": True,
        "min_momentum": 0.5,          # 0.5% momentum mínimo
        "trend_filter": True,
        "min_trend_strength": 0.3,    # Força de tendência
        "min_profit_target": 0.7,
    },
    
    "DNA_MACRO_BREAKOUT": {
        "name": "DNA Macro Breakout",
        "stop_loss": 0.003,           # 0.3% SL agressivo
        "take_profit": 1.2,           # 120% TP
        "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
        "min_confluence": 0.3,
        "volume_multiplier": 0.008,
        "atr_min": 0.08, "atr_max": 20.0, "use_max_leverage": True,
        
        # FOCO EM BREAKOUTS
        "breakout_filter": True,
        "min_breakout_strength": 0.3, # 0.3% breakout mínimo
        "volume_confirmation": True,
        "momentum_confirmation": True,
        "exit_on_weakness": True,      # Sai em primeiro sinal de fraqueza
        "min_profit_target": 0.6,
    }
}

ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def calculate_trade_cost(position_size, avg_hold_hours=6):
    """Calcula custo total de um trade na Hyperliquid"""
    # Trading fees (entrada + saída)
    trading_cost = position_size * HYPERLIQUID_FEES["taker_fee"] * 2
    
    # Funding fees (baseado em horas)
    funding_periods = avg_hold_hours / 8
    funding_cost = position_size * HYPERLIQUID_FEES["funding_rate_avg"] * funding_periods
    
    total_cost = trading_cost + funding_cost
    cost_percentage = (total_cost / position_size) * 100
    
    return total_cost, cost_percentage

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

def calculate_macro_indicators(df, config):
    """Indicadores para estratégias macro com filtros anti-churning"""
    
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_slow'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_gradient'] = df['ema_fast'].pct_change() * 100
    df['ema_separation'] = ((df['ema_fast'] - df['ema_slow']) / df['ema_slow']) * 100
    
    # Volume com filtros
    df['vol_ma_short'] = df['volume'].rolling(window=5).mean()
    df['vol_ma_long'] = df['volume'].rolling(window=20).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma_short']
    df['vol_trend'] = df['vol_ma_short'] / df['vol_ma_long']
    
    # ATR melhorado
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    df['atr_trend'] = df['atr_pct'].rolling(window=5).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum e tendência
    df['price_momentum'] = df['close'].pct_change() * 100
    df['momentum_3'] = df['close'].pct_change(3) * 100
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_ma'] = df['price_momentum'].rolling(window=3).mean()
    
    # Força de tendência
    df['trend_strength'] = np.abs(df['ema_separation'])
    df['trend_direction'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    
    # Breakout detection
    df['price_range'] = df['high'] - df['low']
    df['breakout_strength'] = df['price_range'] / df['close'] * 100
    df['breakout_volume'] = df['vol_surge'] > 1.5
    
    return df

def macro_entry_condition(row, config, last_trade_time=None) -> Tuple[bool, str]:
    """Condições de entrada para estratégias macro (anti-churning)"""
    
    # Verificar cooldown
    if last_trade_time and hasattr(row, 'timestamp'):
        hours_since_last = (row.timestamp - last_trade_time).total_seconds() / 3600
        cooldown = config.get('cooldown_hours', 0)
        if hours_since_last < cooldown:
            return False, f"COOLDOWN_{hours_since_last:.1f}h"
    
    confluence_score = 0
    max_score = 10
    reasons = []
    
    # 1. EMA System (peso 2.5)
    if row.ema_fast > row.ema_slow:
        confluence_score += 2
        reasons.append("EMA")
        if row.ema_gradient > 0.02:  # Gradiente forte
            confluence_score += 0.5
            reasons.append("EMA+")
    
    # 2. Tendência forte (peso 2)
    if hasattr(row, 'trend_strength') and row.trend_strength > config.get('min_trend_strength', 0.2):
        confluence_score += 2
        reasons.append("TREND")
    
    # 3. Volume significativo (peso 2)
    if hasattr(row, 'vol_surge') and row.vol_surge > config['volume_multiplier']:
        confluence_score += 1.5
        reasons.append("VOL")
        if row.vol_surge > config['volume_multiplier'] * 3:  # Volume explosivo
            confluence_score += 0.5
            reasons.append("VOL++")
    
    # 4. ATR apropriado (peso 1.5)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 1.5
        reasons.append("ATR")
    
    # 5. Momentum (peso 1.5)
    if hasattr(row, 'price_momentum') and row.price_momentum > config.get('min_momentum', 0.1):
        confluence_score += 1
        reasons.append("MOM")
        if row.price_momentum > 1.0:  # Momentum forte
            confluence_score += 0.5
            reasons.append("MOM+")
    
    # 6. RSI (peso 0.5)
    if pd.notna(row.rsi) and 20 <= row.rsi <= 80:  # RSI neutro
        confluence_score += 0.5
        reasons.append("RSI")
    
    # Filtros específicos
    if config.get('breakout_filter', False):
        min_breakout = config.get('min_breakout_strength', 0.2)
        if hasattr(row, 'breakout_strength') and row.breakout_strength < min_breakout:
            return False, f"BREAKOUT_WEAK_{row.breakout_strength:.2f}%"
    
    if config.get('volume_confirmation', False):
        if hasattr(row, 'vol_surge') and row.vol_surge < 1.5:
            return False, f"VOLUME_LOW_{row.vol_surge:.1f}"
    
    # Verificar confluência
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/10 [{','.join(reasons[:3])}]"
    
    return is_valid, reason

def macro_exit_logic(current_price, position, current_row, config, bars_held):
    """Lógica de saída para estratégias macro"""
    entry_price = position['entry_price']
    current_profit_pct = (current_price - entry_price) / entry_price
    
    # Stop Loss
    if current_price <= position['stop_loss']:
        return True, 'STOP_LOSS'
    
    # Take Profit fixo (se configurado)
    if 'take_profit' in config:
        tp_pct = config['take_profit'] / 100
        if current_profit_pct >= tp_pct:
            return True, f'TAKE_PROFIT_{config["take_profit"]:.1f}%'
    
    # Timeout
    max_hours = config.get('max_hold_hours', 48)
    if bars_held >= max_hours:
        return True, 'TIMEOUT'
    
    # Trailing inteligente
    if config.get('exit_strategy') == 'smart_trailing':
        # Verificar se atingiu lucro mínimo para trailing
        trailing_activation = config.get('trailing_activation', 0.8) / 100
        if current_profit_pct >= trailing_activation:
            
            # Inicializar trailing
            if not hasattr(position, 'highest_price'):
                position['highest_price'] = current_price
                position['trailing_active'] = True
            
            # Atualizar highest
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            # Calcular trailing distance
            base_distance = config.get('trailing_distance', 0.4) / 100
            
            # Escalar trailing com lucro (se configurado)
            if config.get('profit_scaling', False):
                if current_profit_pct > 0.02:  # 2%+
                    trailing_distance = base_distance * 0.8  # Trailing mais apertado
                elif current_profit_pct > 0.01:  # 1%+
                    trailing_distance = base_distance
                else:
                    trailing_distance = base_distance * 1.2  # Trailing mais largo
            else:
                trailing_distance = base_distance
            
            trailing_stop = position['highest_price'] * (1 - trailing_distance)
            
            if current_price <= trailing_stop:
                return True, f'TRAILING_{trailing_distance*100:.1f}%'
    
    # Saída por fraqueza (se configurado)
    if config.get('exit_on_weakness', False):
        if hasattr(current_row, 'price_momentum') and current_row.price_momentum < -0.3:
            # Só sai se tiver lucro mínimo
            min_profit = config.get('min_profit_target', 0.5) / 100
            if current_profit_pct >= min_profit:
                return True, 'WEAKNESS'
    
    return False, None

def simulate_macro_trading(df, asset, config):
    """Simulação com estratégias macro incluindo custos reais"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    total_fees = 0
    last_trade_time = None
    daily_trades = {}
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 20):
            continue
            
        row = df.iloc[i]
        current_date = row.timestamp.date()
        
        # Controle de trades por dia
        max_daily = config.get('max_trades_per_day', 10)
        if current_date not in daily_trades:
            daily_trades[current_date] = 0
        
        if position is None:
            # Verificar limite diário
            if daily_trades[current_date] >= max_daily:
                continue
                
            should_enter, reason = macro_entry_condition(row, config, last_trade_time)
            
            if should_enter:
                entry_price = row.close
                position_size = capital * leverage
                shares = position_size / entry_price
                
                stop_loss = entry_price * (1 - config['stop_loss'])
                
                # Calcular custo estimado do trade
                avg_hold_hours = config.get('max_hold_hours', 12) / 2  # Média
                trade_cost, cost_pct = calculate_trade_cost(position_size, avg_hold_hours)
                
                # Verificar se lucro mínimo justifica o trade
                min_profit_target = config.get('min_profit_target', 0.5) / 100
                if cost_pct / 100 > min_profit_target * 0.8:  # Custo > 80% do target
                    continue  # Skip trade não rentável
                
                position = {
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'entry_bar': i,
                    'entry_time': row.timestamp,
                    'capital_used': capital,
                    'leverage_used': leverage,
                    'reason': reason,
                    'position_size': position_size,
                    'estimated_cost': trade_cost
                }
                
                daily_trades[current_date] += 1
                last_trade_time = row.timestamp
                
        else:
            current_price = row.close
            bars_held = i - position['entry_bar']
            exit_reason = None
            
            should_exit, exit_reason = macro_exit_logic(current_price, position, row, config, bars_held)
            
            if should_exit:
                # Calcular PnL
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                
                # Calcular custos reais
                hold_hours = bars_held
                actual_cost, cost_pct = calculate_trade_cost(position['position_size'], hold_hours)
                total_fees += actual_cost
                
                pnl_net = pnl_gross - actual_cost
                
                trade = {
                    'asset': asset,
                    'exit_reason': exit_reason,
                    'pnl_gross': pnl_gross,
                    'pnl_net': pnl_net,
                    'fees_paid': actual_cost,
                    'leverage_used': position['leverage_used'],
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'duration_hours': bars_held,
                    'entry_reason': position['reason'],
                    'position_size': position['position_size'],
                    'cost_percentage': cost_pct
                }
                
                trades.append(trade)
                position = None
    
    return trades, total_fees

def run_macro_test(config_name, config):
    print(f"\n🚀 TESTANDO: {config['name']}")
    print("="*70)
    
    all_trades = []
    total_pnl_gross = 0
    total_pnl_net = 0
    total_fees = 0
    profitable_assets = 0
    
    for asset in ASSETS:
        df = load_data(asset)
        if df is None:
            continue
        
        leverage = get_leverage_for_asset(asset, config)
        
        df = calculate_macro_indicators(df, config)
        trades, asset_fees = simulate_macro_trading(df, asset, config)
        
        if trades:
            asset_pnl_gross = sum(t['pnl_gross'] for t in trades)
            asset_pnl_net = sum(t['pnl_net'] for t in trades)
            
            roi_gross = (asset_pnl_gross / 4.0) * 100
            roi_net = (asset_pnl_net / 4.0) * 100
            
            wins_net = len([t for t in trades if t['pnl_net'] > 0])
            win_rate_net = (wins_net / len(trades)) * 100
            
            if asset_pnl_net > 0:
                profitable_assets += 1
                status = "🟢"
            else:
                status = "🔴"
            
            print(f"   {status} {asset}: {len(trades)} trades | {leverage}x | {win_rate_net:.1f}% WR | ROI: {roi_gross:+.1f}%→{roi_net:+.1f}%")
            
            total_pnl_gross += asset_pnl_gross
            total_pnl_net += asset_pnl_net
            total_fees += asset_fees
            all_trades.extend(trades)
    
    # Resultado geral
    total_capital = len(ASSETS) * 4.0
    portfolio_roi_gross = (total_pnl_gross / total_capital) * 100
    portfolio_roi_net = (total_pnl_net / total_capital) * 100
    
    total_trades = len(all_trades)
    total_wins = len([t for t in all_trades if t['pnl_net'] > 0])
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    fee_impact = portfolio_roi_gross - portfolio_roi_net
    
    print(f"\n📊 RESULTADO:")
    print(f"   💰 ROI Bruto: {portfolio_roi_gross:+.1f}%")
    print(f"   💵 ROI Líquido: {portfolio_roi_net:+.1f}%")
    print(f"   💸 Impacto Taxas: -{fee_impact:.1f}%")
    print(f"   🎯 Trades: {total_trades}")
    print(f"   🏆 WR: {win_rate:.1f}%")
    print(f"   ✅ Assets+: {profitable_assets}/{len(ASSETS)}")
    print(f"   🏦 Fees Totais: ${total_fees:.2f}")
    
    return {
        'config_name': config_name,
        'config': config,
        'portfolio_roi_gross': portfolio_roi_gross,
        'portfolio_roi_net': portfolio_roi_net,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profitable_assets': profitable_assets,
        'total_fees': total_fees,
        'fee_impact': fee_impact
    }

def main():
    print("🚀 DNA MACRO TRAILING - OTIMIZADO PARA HYPERLIQUID")
    print("="*80)
    print("🎯 FOCO: Menos trades, mais lucro por trade, custos controlados")
    
    # Custo mínimo por trade
    sample_position = 4.0 * 20  # $80 position média
    min_cost, min_cost_pct = calculate_trade_cost(sample_position, 6)
    print(f"💡 Custo mínimo por trade: ${min_cost:.2f} ({min_cost_pct:.2f}%)")
    print(f"📊 Logo, cada trade precisa de pelo menos {min_cost_pct*2:.1f}% de lucro")
    
    all_results = []
    
    # Testar todas as configurações macro
    for config_name, config in DNA_MACRO_CONFIGS.items():
        result = run_macro_test(config_name, config)
        all_results.append(result)
    
    # Ranking
    print("\n" + "="*80)
    print("👑 RANKING DNA MACRO STRATEGIES")
    print("="*80)
    
    all_results.sort(key=lambda x: x['portfolio_roi_net'], reverse=True)
    
    dna_realista_benchmark = 1377.3
    
    print(f"\nPos | Config                    | ROI Líq. | Trades | WR    | Assets+ | Fees   | vs DNA Realista")
    print("-" * 100)
    
    for i, result in enumerate(all_results, 1):
        name = result['config']['name'][:24]
        roi_net = result['portfolio_roi_net']
        trades = result['total_trades']
        wr = result['win_rate']
        assets = result['profitable_assets']
        fees = result['total_fees']
        vs_realista = roi_net - dna_realista_benchmark
        
        if i == 1:
            emoji = "👑"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        else:
            emoji = f"{i:2}"
            
        vs_str = f"+{vs_realista:.1f}%" if vs_realista > 0 else f"{vs_realista:.1f}%"
        
        print(f"{emoji} | {name:<24} | {roi_net:+7.1f}% | {trades:6} | {wr:4.1f}% | {assets:2}/16 | ${fees:5.0f} | {vs_str}")
    
    # Análise do vencedor
    champion = all_results[0]
    vs_realista = champion['portfolio_roi_net'] - dna_realista_benchmark
    
    print(f"\n📊 ANÁLISE DO VENCEDOR:")
    print(f"   🏆 Estratégia: {champion['config']['name']}")
    print(f"   💰 ROI Líquido: {champion['portfolio_roi_net']:+.1f}%")
    print(f"   🎯 Total Trades: {champion['total_trades']}")
    print(f"   💸 Total Fees: ${champion['total_fees']:.2f}")
    print(f"   📊 Fee Impact: -{champion['fee_impact']:.1f}%")
    
    if vs_realista > 0:
        print(f"   ✅ vs DNA Realista: +{vs_realista:.1f}% (VENCEDOR!)")
        print(f"   🚀 ESTRATÉGIA MACRO VALIDADA!")
    else:
        print(f"   ❌ vs DNA Realista: {vs_realista:.1f}%")
        print(f"   📊 Ainda inferior ao benchmark")
    
    # Transformação de capital
    capital_total = 64.0
    final_value = capital_total + champion['portfolio_roi_net'] * capital_total / 100
    multiplier = final_value / capital_total
    
    print(f"\n💎 TRANSFORMAÇÃO DE CAPITAL (VENCEDOR):")
    print(f"   💰 Capital Inicial: ${capital_total:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dna_macro_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 TESTE DNA MACRO CONCLUÍDO!")

if __name__ == "__main__":
    main()
