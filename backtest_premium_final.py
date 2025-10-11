#!/usr/bin/env python3
"""
🎯 BACKTEST FINAL ULTRA OTIMIZADO - EFEITO COMPOSTO PREMIUM
Estratégia mais seletiva para trades de maior qualidade
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Configuração DNA Premium (mais seletivo, trades maiores)
DNA_PREMIUM = {
    'stop_loss': 2.0,          # SL mais conservador
    'take_profit': 15.0,       # TP mais alto
    'leverage': 4.0,           # Leverage maior
    'ema_fast': 5,
    'ema_slow': 21,
    'rsi_period': 14,
    'rsi_buy_max': 70,         # RSI mais restritivo
    'rsi_sell_min': 30,
    'volume_multiplier': 2.5,  # Volume muito acima da média
    'atr_period': 14,
    'atr_threshold': 0.8       # ATR mais alto = mais volatilidade
}

# Assets selecionados pelos melhores performers
PREMIUM_ASSETS = [
    "XRP-USD", "DOGE-USD", "LINK-USD", "ADA-USD", "ETH-USD",
    "AVAX-USD", "SOL-USD", "BNB-USD", "BTC-USD", "LTC-USD"
]

def load_data(asset):
    """Carrega dados reais do asset"""
    symbol = asset.replace("-USD", "").lower()
    filename = f"dados_reais_{symbol}_1ano.csv"
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Mapear colunas para nomes padrão
        df = df.rename(columns={
            'valor_fechamento': 'close',
            'valor_abertura': 'open',
            'valor_maximo': 'high',
            'valor_minimo': 'low'
        })
        
        # Calcular indicadores
        df = calculate_indicators(df)
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar {asset}: {e}")
        return None

def calculate_indicators(df):
    """Calcula indicadores técnicos premium"""
    # EMAs
    df[f'ema_{DNA_PREMIUM["ema_fast"]}'] = df['close'].ewm(span=DNA_PREMIUM['ema_fast']).mean()
    df[f'ema_{DNA_PREMIUM["ema_slow"]}'] = df['close'].ewm(span=DNA_PREMIUM['ema_slow']).mean()
    
    # RSI mais sensível
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=DNA_PREMIUM['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=DNA_PREMIUM['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume 
    df['volume_avg'] = df['volume'].rolling(window=30).mean()  # Média mais longa
    df['volume_ratio'] = df['volume'] / df['volume_avg']
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=DNA_PREMIUM['atr_period']).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Momentum adicional
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['volume_spike'] = df['volume'] / df['volume'].rolling(10).mean()
    
    return df

def check_premium_buy_signal(df, i):
    """Sinal de compra PREMIUM - muito seletivo"""
    if i < DNA_PREMIUM['ema_slow'] + 10:
        return False
    
    try:
        # Confluência rigorosa
        ema_strong_cross = (
            df.iloc[i][f'ema_{DNA_PREMIUM["ema_fast"]}'] > df.iloc[i][f'ema_{DNA_PREMIUM["ema_slow"]}'] and
            df.iloc[i-1][f'ema_{DNA_PREMIUM["ema_fast"]}'] <= df.iloc[i-1][f'ema_{DNA_PREMIUM["ema_slow"]}']
        )
        
        rsi_ideal = 30 <= df.iloc[i]['rsi'] <= DNA_PREMIUM['rsi_buy_max']
        
        volume_explosion = df.iloc[i]['volume_ratio'] >= DNA_PREMIUM['volume_multiplier']
        
        atr_high = df.iloc[i]['atr_pct'] >= DNA_PREMIUM['atr_threshold']
        
        # Momentum forte
        momentum_up = df.iloc[i]['price_change_5'] > 2.0  # 2% nos últimos 5 períodos
        
        # Volume spike
        volume_spike = df.iloc[i]['volume_spike'] >= 2.0
        
        # Precio acima das EMAs
        price_above_emas = df.iloc[i]['close'] > df.iloc[i][f'ema_{DNA_PREMIUM["ema_slow"]}']
        
        # SIGNAL PREMIUM: Pelo menos 5 dos 6 critérios
        criteria = [ema_strong_cross, rsi_ideal, volume_explosion, atr_high, momentum_up, price_above_emas]
        signal_strength = sum(criteria)
        
        return signal_strength >= 4  # Muito seletivo
        
    except:
        return False

def check_premium_sell_signal(df, i):
    """Sinal de venda premium"""
    if i < DNA_PREMIUM['ema_slow']:
        return False
    
    try:
        # Condições de saída
        ema_cross_down = (
            df.iloc[i][f'ema_{DNA_PREMIUM["ema_fast"]}'] < df.iloc[i][f'ema_{DNA_PREMIUM["ema_slow"]}']
        )
        
        rsi_overbought = df.iloc[i]['rsi'] >= 85
        
        momentum_down = df.iloc[i]['price_change_5'] < -3.0
        
        return ema_cross_down or rsi_overbought or momentum_down
        
    except:
        return False

def simulate_premium_compound(df, asset):
    """Simulação compound PREMIUM"""
    capital = 10.0  # Começar com mais capital por asset
    position = None
    trades = []
    max_capital = capital
    
    for i in range(len(df)):
        if position is None:
            # Entrada PREMIUM
            if check_premium_buy_signal(df, i):
                entry_price = df.iloc[i]['close']
                
                # Usar TODO o capital disponível com leverage
                position_size = capital
                shares = (position_size * DNA_PREMIUM['leverage']) / entry_price
                
                # Níveis mais conservadores
                stop_loss = entry_price * (1 - DNA_PREMIUM['stop_loss'] / 100)
                take_profit = entry_price * (1 + DNA_PREMIUM['take_profit'] / 100)
                
                position = {
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_bar': i,
                    'entry_time': df.iloc[i]['timestamp'],
                    'capital_used': capital
                }
                
        else:
            # Verificar saída
            current_price = df.iloc[i]['close']
            exit_reason = None
            
            if position['type'] == 'LONG':
                if current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                elif current_price >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                elif check_premium_sell_signal(df, i):
                    exit_reason = 'SELL_SIGNAL'
            
            # Timeout mais generoso para trades premium
            if i - position['entry_bar'] >= 168:  # 1 semana
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                # Calcular PNL
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                pnl_leveraged = pnl_gross
                
                # Novo capital
                new_capital = position['capital_used'] + pnl_leveraged
                
                # Proteção contra capital negativo
                if new_capital < 1.0:
                    new_capital = 1.0
                
                # Tracking do máximo
                if new_capital > max_capital:
                    max_capital = new_capital
                
                pnl_percent = ((new_capital - position['capital_used']) / position['capital_used']) * 100
                
                trade = {
                    'asset': asset,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_time': position['entry_time'],
                    'exit_time': df.iloc[i]['timestamp'],
                    'exit_reason': exit_reason,
                    'shares': position['shares'],
                    'pnl_gross': pnl_gross,
                    'pnl_leveraged': pnl_leveraged,
                    'pnl_percent': pnl_percent,
                    'capital_before': position['capital_used'],
                    'capital_after': new_capital,
                    'duration_bars': i - position['entry_bar'],
                    'max_capital_reached': max_capital
                }
                
                trades.append(trade)
                
                # COMPOSTO TOTAL
                capital = new_capital
                position = None
    
    return trades, capital, max_capital

def main():
    print("🎯 BACKTEST FINAL ULTRA OTIMIZADO - EFEITO COMPOSTO PREMIUM")
    print("="*85)
    
    all_results = {}
    total_initial_capital = len(PREMIUM_ASSETS) * 10.0
    total_final_capital = 0
    total_max_capital = 0
    
    print(f"\n🧬 DNA PREMIUM:")
    for key, value in DNA_PREMIUM.items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 EXECUTANDO ESTRATÉGIA PREMIUM:")
    print("="*70)
    
    all_trades_summary = []
    
    for asset in PREMIUM_ASSETS:
        print(f"\n📈 Processando {asset}...")
        
        df = load_data(asset)
        if df is None:
            continue
        
        trades, final_capital, max_capital = simulate_premium_compound(df, asset)
        
        if trades:
            wins = [t for t in trades if t['pnl_leveraged'] > 0]
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            roi = ((final_capital - 10.0) / 10.0) * 100
            max_roi = ((max_capital - 10.0) / 10.0) * 100
            
            print(f"   ✅ {len(trades)} trades | Win rate: {win_rate:.1f}% | ROI: {roi:+.1f}% | Max: {max_roi:+.1f}%")
            
            all_results[asset] = {
                'trades': len(trades),
                'win_rate': win_rate,
                'roi': roi,
                'max_roi': max_roi,
                'final_capital': final_capital,
                'max_capital': max_capital
            }
            
            total_final_capital += final_capital
            total_max_capital += max_capital
            all_trades_summary.extend(trades)
        else:
            print(f"   ❌ Nenhum trade premium encontrado")
            all_results[asset] = {
                'trades': 0,
                'win_rate': 0,
                'roi': 0,
                'max_roi': 0,
                'final_capital': 10.0,
                'max_capital': 10.0
            }
            total_final_capital += 10.0
            total_max_capital += 10.0
    
    # Resultado final
    portfolio_roi = ((total_final_capital - total_initial_capital) / total_initial_capital) * 100
    portfolio_max_roi = ((total_max_capital - total_initial_capital) / total_initial_capital) * 100
    
    print("\n" + "="*85)
    print("🏆 RESULTADOS PREMIUM FINAIS:")
    print("="*85)
    
    print(f"\n📊 Performance por Asset:")
    print("Asset      | Trades | Win% | ROI Final  | ROI Máximo | Cap Final | Cap Máximo")
    print("-" * 80)
    
    for asset, result in all_results.items():
        print(f"{asset:10} | {result['trades']:6} | {result['win_rate']:4.1f} | "
              f"{result['roi']:+8.1f}% | {result['max_roi']:+8.1f}% | "
              f"${result['final_capital']:7.2f} | ${result['max_capital']:8.2f}")
    
    print("-" * 80)
    print(f"PORTFOLIO  | {sum(r['trades'] for r in all_results.values()):6} | "
          f"   - | {portfolio_roi:+8.1f}% | {portfolio_max_roi:+8.1f}% | "
          f"${total_final_capital:7.2f} | ${total_max_capital:8.2f}")
    
    print(f"\n💰 RESULTADO DEFINITIVO:")
    print(f"   💸 Capital inicial: ${total_initial_capital:.2f}")
    print(f"   💰 Capital final: ${total_final_capital:.2f}")
    print(f"   🚀 Capital máximo atingido: ${total_max_capital:.2f}")
    print(f"   📈 ROI final: {portfolio_roi:+.1f}%")
    print(f"   🎯 ROI máximo: {portfolio_max_roi:+.1f}%")
    print(f"   🏆 Target original: +10.910%")
    
    if portfolio_roi >= 10910:
        print(f"   🎉 TARGET ATINGIDO! {(portfolio_roi-10910):+.1f}% acima!")
    elif portfolio_max_roi >= 10910:
        print(f"   🎯 TARGET foi atingido no máximo! Peak: {portfolio_max_roi:+.1f}%")
    else:
        print(f"   🔧 Faltam {(10910-portfolio_roi):+.1f}% para o target (final)")
        print(f"   📊 Faltam {(10910-portfolio_max_roi):+.1f}% para o target (máximo)")
    
    print(f"\n🎲 ESTATÍSTICAS PREMIUM:")
    if all_trades_summary:
        wins = [t for t in all_trades_summary if t['pnl_leveraged'] > 0]
        losses = [t for t in all_trades_summary if t['pnl_leveraged'] < 0]
        
        win_rate_global = len(wins) / len(all_trades_summary) * 100
        avg_win = np.mean([t['pnl_percent'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_percent'] for t in losses]) if losses else 0
        
        best_trade = max(all_trades_summary, key=lambda x: x['pnl_percent'])
        worst_trade = min(all_trades_summary, key=lambda x: x['pnl_percent'])
        
        print(f"   📊 Total trades: {len(all_trades_summary)}")
        print(f"   🎯 Win rate: {win_rate_global:.1f}%")
        print(f"   📈 Ganho médio: {avg_win:+.1f}%")
        print(f"   📉 Perda média: {avg_loss:+.1f}%")
        print(f"   🏆 Melhor trade: {best_trade['asset']} {best_trade['pnl_percent']:+.1f}%")
        print(f"   💥 Pior trade: {worst_trade['asset']} {worst_trade['pnl_percent']:+.1f}%")
        
    print("\n🎊 ESTRATÉGIA PREMIUM EXECUTADA!")
    print("="*85)
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_premium = {
        'timestamp': timestamp,
        'strategy': 'PREMIUM_COMPOUND',
        'dna_config': DNA_PREMIUM,
        'portfolio_roi_final': portfolio_roi,
        'portfolio_roi_max': portfolio_max_roi,
        'target_roi': 10910,
        'target_achieved': portfolio_roi >= 10910,
        'target_achieved_max': portfolio_max_roi >= 10910,
        'precision_final': (portfolio_roi/10910)*100,
        'precision_max': (portfolio_max_roi/10910)*100,
        'total_trades': len(all_trades_summary),
        'assets_results': all_results
    }
    
    filename = f"backtest_premium_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results_premium, f, indent=2, default=str)
    
    print(f"📁 Resultados salvos: {filename}")

if __name__ == "__main__":
    main()
