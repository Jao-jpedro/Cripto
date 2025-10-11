#!/usr/bin/env python3
"""
🚀 OTIMIZAÇÃO HÍBRIDA FINAL - MELHOR DOS DOIS MUNDOS
Combinando a simplicidade do DNA Realista com melhorias seletivas de ML
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

# CONFIGURAÇÕES HÍBRIDAS - REFINAMENTO DO CAMPEÃO
HYBRID_SUPREME_CONFIGS = {
    "DNA_REALISTA_SUPREMO": {
        "name": "DNA Realista Supremo",
        "stop_loss": 0.0015,     # SL 0.15% (mais agressivo)
        "take_profit": 1.3,      # TP 130% (mais conservador)
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.25,  # Ligeiramente mais seletivo
        "volume_multiplier": 0.008, # Melhor detecção de volume
        "atr_min": 0.0008,
        "atr_max": 35.0,
        "use_smart_exit": True   # Exit inteligente
    },
    
    "DNA_PERFECT_BALANCE": {
        "name": "DNA Perfect Balance",
        "stop_loss": 0.0018,     # SL 0.18%
        "take_profit": 1.4,      # TP 140%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.2,
        "volume_multiplier": 0.005,
        "atr_min": 0.001,
        "atr_max": 40.0,
        "use_smart_exit": True,
        "use_momentum_filter": True
    },
    
    "DNA_ULTRA_REFINADO": {
        "name": "DNA Ultra Refinado",
        "stop_loss": 0.001,      # SL 0.1% (ultra agressivo)
        "take_profit": 1.6,      # TP 160%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.15,  # Menos restritivo
        "volume_multiplier": 0.003,
        "atr_min": 0.0005,
        "atr_max": 45.0,
        "use_smart_exit": True,
        "use_momentum_filter": True,
        "use_trend_acceleration": True
    },
    
    "DNA_GIGA_REFINADO": {
        "name": "DNA Giga Refinado",
        "stop_loss": 0.0012,     # SL 0.12%
        "take_profit": 1.8,      # TP 180%
        "use_max_leverage": True,
        "ema_fast": 1,
        "ema_slow": 2,
        "rsi_period": 1,
        "min_confluence": 0.1,   # Muito menos restritivo
        "volume_multiplier": 0.001,
        "atr_min": 0.0003,
        "atr_max": 50.0,
        "use_smart_exit": True,
        "use_momentum_filter": True,
        "use_trend_acceleration": True,
        "ultra_sensitive": True
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

def calculate_hybrid_indicators(df, config):
    """Indicadores híbridos - base realista + melhorias seletivas"""
    
    # EMAs ultra responsivas (mantém base que funciona)
    df['ema_short'] = df['close'].ewm(span=config['ema_fast']).mean()
    df['ema_long'] = df['close'].ewm(span=config['ema_slow']).mean()
    df['ema_gradient'] = df['ema_short'].pct_change() * 100
    
    # Volume inteligente (melhoria do base)
    df['vol_ma'] = df['volume'].rolling(window=3).mean()
    df['vol_surge'] = df['volume'] / df['vol_ma']
    df['vol_strength'] = np.log1p(df['vol_surge'])  # Log para suavizar outliers
    
    # ATR melhorado
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=3).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # RSI ultra responsivo (base que funciona)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum básico (simples e efetivo)
    df['price_momentum'] = df['close'].pct_change() * 100
    
    # MELHORIAS SELETIVAS APENAS SE CONFIGURADO
    
    # Filtro de momento (só se habilitado)
    if config.get('use_momentum_filter', False):
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_strength'] = (df['price_momentum'] + df['momentum_3']) / 2
    else:
        df['momentum_strength'] = df['price_momentum']
    
    # Aceleração de tendência (só se habilitado)
    if config.get('use_trend_acceleration', False):
        df['ema_acceleration'] = df['ema_gradient'].diff()
        df['trend_power'] = df['ema_gradient'] + (df['ema_acceleration'] * 0.5)
    else:
        df['trend_power'] = df['ema_gradient']
    
    # Sensibilidade ultra (só se habilitado)
    if config.get('ultra_sensitive', False):
        df['micro_breakout'] = (df['close'] / df['ema_short'] - 1) * 1000  # Micro movimentos
        df['volume_burst'] = df['vol_surge'] > (1 + config['volume_multiplier'] * 0.5)
    else:
        df['micro_breakout'] = 0
        df['volume_burst'] = False
    
    return df

def hybrid_supreme_entry_condition(row, config) -> Tuple[bool, str]:
    """Condição de entrada híbrida suprema"""
    confluence_score = 0
    max_score = 12  # Score mais controlado
    reasons = []
    
    # 1. Sistema EMA (base sólida - peso 3)
    if row.ema_short > row.ema_long:
        confluence_score += 2
        reasons.append("EMA")
        if row.trend_power > 0.01:
            confluence_score += 1
            reasons.append("Trend+")
    
    # 2. Breakout inteligente (peso 3)
    breakout_threshold = 1.001 if config.get('ultra_sensitive', False) else 1.002
    if row.close > row.ema_short * breakout_threshold:
        confluence_score += 3
        reasons.append("Break")
        if config.get('ultra_sensitive', False) and row.micro_breakout > 1:
            confluence_score += 0.5
            reasons.append("μBreak")
    elif row.close > row.ema_short:
        confluence_score += 1.5
        reasons.append("Break-")
    
    # 3. Volume inteligente (peso 2.5)
    if row.vol_surge > config['volume_multiplier']:
        confluence_score += 2
        reasons.append("Vol")
        if hasattr(row, 'vol_strength') and row.vol_strength > 1:
            confluence_score += 0.5
            reasons.append("Vol+")
        if config.get('ultra_sensitive', False) and row.volume_burst:
            confluence_score += 0.5
            reasons.append("Burst")
    
    # 4. ATR (peso 1.5)
    if config['atr_min'] <= row.atr_pct <= config['atr_max']:
        confluence_score += 1.5
        reasons.append("ATR")
    
    # 5. RSI (peso 1)
    if pd.notna(row.rsi) and 5 <= row.rsi <= 95:
        confluence_score += 1
        reasons.append("RSI")
    
    # 6. Momentum (peso 1 - só se habilitado)
    if config.get('use_momentum_filter', False):
        if hasattr(row, 'momentum_strength') and row.momentum_strength > 0.01:
            confluence_score += 1
            reasons.append("Mom")
    
    is_valid = confluence_score >= config['min_confluence']
    reason = f"{confluence_score:.1f}/12 [{','.join(reasons[:4])}]"
    
    return is_valid, reason

def simulate_hybrid_supreme_trading(df, asset, config):
    """Simulação de trading híbrida suprema"""
    capital = 4.0
    leverage = get_leverage_for_asset(asset, config)
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < max(config['ema_slow'], config['rsi_period'], 3):
            continue
            
        row = df.iloc[i]
        
        if position is None:
            should_enter, reason = hybrid_supreme_entry_condition(row, config)
            
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
            
            # Exit inteligente (se habilitado)
            if config.get('use_smart_exit', False):
                # Saída antecipada por momentum negativo
                if hasattr(row, 'momentum_strength') and row.momentum_strength < -0.5:
                    exit_reason = 'MOMENTUM_EXIT'
                # Saída antecipada por quebra de EMA
                elif row.close < row.ema_short * 0.998:
                    exit_reason = 'EMA_BREAK'
            
            # Saídas normais
            if not exit_reason:
                if current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                elif current_price >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                elif i - position['entry_bar'] >= 10:  # Timeout 10 horas
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

def run_hybrid_supreme_test(config_name, config):
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
        
        df = calculate_hybrid_indicators(df, config)
        trades = simulate_hybrid_supreme_trading(df, asset, config)
        
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
    print("🚀 OTIMIZAÇÃO HÍBRIDA SUPREMA - FINAL BOSS")
    print("="*80)
    print("🎯 OBJETIVO: SUPERAR +1.377% ROI COM MELHORIAS INTELIGENTES")
    
    all_results = []
    
    # Testar todas as configurações híbridas supremas
    for config_name, config in HYBRID_SUPREME_CONFIGS.items():
        result = run_hybrid_supreme_test(config_name, config)
        all_results.append(result)
    
    # Ranking final
    print("\n" + "="*80)
    print("👑 RANKING HÍBRIDO SUPREMO")
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
    previous_best = 1377.3  # ROI do DNA Realista Otimizado
    improvement = champion['portfolio_roi'] - previous_best
    
    print(f"\n👑 CAMPEÃO HÍBRIDO SUPREMO:")
    print(f"   📛 Nome: {champion['config']['name']}")
    print(f"   🚀 ROI: {champion['portfolio_roi']:+.1f}%")
    print(f"   💰 PnL: ${champion['total_pnl']:+.2f}")
    print(f"   📊 Trades: {champion['total_trades']}")
    print(f"   🎯 Win Rate: {champion['win_rate']:.1f}%")
    print(f"   ✅ Assets Lucrativos: {champion['profitable_assets']}/16")
    
    print(f"\n📈 COMPARAÇÃO COM MELHOR ANTERIOR:")
    print(f"   📊 DNA Realista: +{previous_best:.1f}%")
    print(f"   🚀 Híbrido Supremo: +{champion['portfolio_roi']:.1f}%")
    
    if improvement > 0:
        print(f"   ✅ MELHORIA: +{improvement:.1f}% ({(improvement/previous_best)*100:+.1f}%)")
        print("   🎊 NOVO RECORDE ALCANÇADO!")
    else:
        print(f"   ❌ Diferença: {improvement:.1f}% ({(improvement/previous_best)*100:+.1f}%)")
        print("   📊 DNA Realista ainda é o melhor")
    
    # Transformação do capital
    total_capital = len(ASSETS) * 4.0
    final_value = total_capital + champion['total_pnl']
    multiplier = final_value / total_capital
    
    print(f"\n💎 TRANSFORMAÇÃO DE CAPITAL:")
    print(f"   💰 Capital Inicial: ${total_capital:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"otimizacao_hibrida_suprema_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos: {filename}")
    print(f"\n🎊 OTIMIZAÇÃO HÍBRIDA SUPREMA CONCLUÍDA!")
    print("🏆 MELHOR DOS DOIS MUNDOS TESTADO!")
    print("="*80)

if __name__ == "__main__":
    main()
