#!/usr/bin/env python3
"""
Estratégia SPOT (Sem Leverage) para trading.py
Baseado na descoberta: leverage é o problema principal
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

def test_spot_trading():
    """Testa estratégia spot (sem leverage)"""
    
    print("💎 ESTRATÉGIA SPOT TRADING - SEM LEVERAGE")
    print("🎯 Baseado na descoberta: LEVERAGE é o problema")
    print("="*65)
    
    # Configuração SPOT (sem leverage)
    config = {
        'tp_pct': 8.0,       # TP: 8% do preço
        'sl_pct': 4.0,       # SL: 4% do preço  
        'leverage': 1,       # SEM LEVERAGE!
        'atr_min': 0.5,      # ATR filtro
        'atr_max': 2.0,      # ATR máximo
        'volume_mult': 1.5,  # Volume 1.5x
        'min_confluencia': 2, # 2 critérios
        'ema_short': 9,      # EMA rápida
        'ema_long': 21,      # EMA lenta
        'rsi_oversold': 30,  # RSI oversold
        'rsi_overbought': 70 # RSI overbought
    }
    
    print("📋 CONFIGURAÇÃO SPOT TRADING:")
    print(f"   TP/SL: {config['tp_pct']}%/{config['sl_pct']}% do PREÇO (não da margem)")
    print(f"   Leverage: {config['leverage']}x (SEM LEVERAGE)")
    print(f"   ATR Range: {config['atr_min']}-{config['atr_max']}%")
    print(f"   Volume: {config['volume_mult']}x média")
    print(f"   EMAs: {config['ema_short']}/{config['ema_long']}")
    print(f"   RSI: {config['rsi_oversold']}-{config['rsi_overbought']}")
    print()
    
    # Testar assets principais
    assets_to_test = [
        ("BTC", "dados_reais_btc_1ano.csv"),
        ("ETH", "dados_reais_eth_1ano.csv"),
        ("BNB", "dados_reais_bnb_1ano.csv"),
        ("SOL", "dados_reais_sol_1ano.csv"),
        ("ADA", "dados_reais_ada_1ano.csv"),
        ("AVAX", "dados_reais_avax_1ano.csv")
    ]
    
    results = []
    
    print("🧪 TESTANDO SPOT TRADING...")
    print("-"*50)
    
    for asset_name, filename in assets_to_test:
        if os.path.exists(filename):
            result = test_spot_asset(asset_name, filename, config)
            if result:
                results.append(result)
        else:
            print(f"⚠️ {asset_name}: Arquivo não encontrado")
    
    if results:
        analyze_spot_results(results, config)
    else:
        print("❌ Nenhum resultado válido!")

def test_spot_asset(asset_name, filename, config):
    """Testa trading spot em um asset"""
    
    try:
        # Carregar dados
        df = pd.read_csv(filename)
        
        # Padronizar colunas
        column_mapping = {
            'open': 'valor_abertura',
            'high': 'valor_maximo', 
            'low': 'valor_minimo',
            'close': 'valor_fechamento',
            'volume': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Calcular indicadores
        df = calculate_spot_indicators(df, config)
        
        # Simular spot trading
        result = simulate_spot_trading(df, config)
        result['asset'] = asset_name
        
        # Comparar com buy and hold
        buy_hold_return = ((df['valor_fechamento'].iloc[-1] - df['valor_fechamento'].iloc[0]) / 
                          df['valor_fechamento'].iloc[0]) * 100
        result['buy_hold_return'] = buy_hold_return
        
        # Log resultado
        roi = result['total_return_pct']
        trades = result['num_trades']
        wr = result['win_rate'] * 100
        dd = result['max_drawdown']
        bh = buy_hold_return
        
        if roi > bh:
            status = "🟢 SUPEROU"
        elif roi > 0:
            status = "🟡 POSITIVO"
        else:
            status = "🔴 NEGATIVO"
            
        print(f"{status} {asset_name}: {roi:+6.1f}% vs B&H {bh:+5.1f}% | {trades:3d} trades | WR {wr:4.1f}% | DD {dd:4.1f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ {asset_name}: Erro - {str(e)[:50]}")
        return None

def calculate_spot_indicators(df, config):
    """Calcula indicadores para spot trading"""
    df = df.copy()
    
    # EMAs
    df['ema_short'] = df['valor_fechamento'].ewm(span=config['ema_short']).mean()
    df['ema_long'] = df['valor_fechamento'].ewm(span=config['ema_long']).mean()
    
    # ATR
    high_low = df['valor_maximo'] - df['valor_minimo']
    high_close = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
    low_close = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    # RSI
    delta = df['valor_fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def simulate_spot_trading(df, config):
    """Simula spot trading (sem leverage)"""
    
    balance_usd = 1000.0  # Capital em USD
    crypto_amount = 0.0   # Quantidade de crypto
    trades = []
    position_type = None  # 'long' ou None
    entry_price = 0
    
    for i in range(max(config['ema_long'], 20), len(df) - 1):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        current_price = current['valor_fechamento']
        
        # Se tem posição, verificar saída
        if position_type == 'long':
            # Calcular P&L
            current_value = crypto_amount * current_price
            pnl_pct = (current_value - balance_usd) / balance_usd
            
            # Verificar TP/SL (baseado no preço, não na margem)
            price_change_pct = (current_price - entry_price) / entry_price
            
            tp_pct = config['tp_pct'] / 100.0
            sl_pct = config['sl_pct'] / 100.0
            
            if price_change_pct >= tp_pct or price_change_pct <= -sl_pct:
                # Vender tudo
                balance_usd = crypto_amount * current_price
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'balance_after': balance_usd,
                    'reason': 'TP' if price_change_pct >= tp_pct else 'SL'
                })
                
                crypto_amount = 0.0
                position_type = None
                continue
        
        # Se não tem posição, verificar entrada
        if position_type is None:
            can_long = check_spot_entry(current, prev, config)
            
            if can_long:
                # Comprar crypto com todo o USD
                crypto_amount = balance_usd / current_price
                entry_price = current_price
                position_type = 'long'
                # balance_usd = 0.0  # Todo dinheiro em crypto
    
    # Se terminou com posição aberta, fechar no último preço
    if position_type == 'long':
        final_price = df['valor_fechamento'].iloc[-1]
        balance_usd = crypto_amount * final_price
        pnl_pct = (balance_usd - 1000.0) / 1000.0
        
        trades.append({
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl_pct': pnl_pct,
            'balance_after': balance_usd,
            'reason': 'END'
        })
    
    # Métricas
    if not trades:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'final_balance': balance_usd
        }
    
    total_return_pct = ((balance_usd - 1000.0) / 1000.0) * 100
    wins = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    # Drawdown
    balances = [1000.0]
    for trade in trades:
        balances.append(trade['balance_after'])
    
    peak = balances[0]
    max_dd = 0
    for balance in balances:
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown': max_dd * 100,
        'final_balance': balance_usd
    }

def check_spot_entry(current, prev, config):
    """Verifica entrada para spot trading"""
    
    criterios = 0
    
    # 1. EMA Cross bullish
    ema_cross_bullish = (prev['ema_short'] <= prev['ema_long'] and 
                        current['ema_short'] > current['ema_long'])
    if ema_cross_bullish:
        criterios += 1
    
    # 2. ATR em range
    atr_ok = config['atr_min'] <= current['atr_pct'] <= config['atr_max']
    if atr_ok:
        criterios += 1
    
    # 3. Volume
    volume_ok = current['volume'] > (current['vol_ma'] * config['volume_mult'])
    if volume_ok:
        criterios += 1
    
    # 4. RSI oversold (força extra)
    if current['rsi'] < config['rsi_oversold']:
        criterios += 2
    
    # 5. Preço acima de EMA longa (tendência de alta)
    if current['valor_fechamento'] > current['ema_long']:
        criterios += 1
    
    return criterios >= config['min_confluencia']

def analyze_spot_results(results, config):
    """Analisa resultados do spot trading"""
    
    print(f"\n" + "="*65)
    print("📊 ANÁLISE SPOT TRADING - SEM LEVERAGE")
    print("="*65)
    
    # Métricas gerais
    avg_roi = np.mean([r['total_return_pct'] for r in results])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in results])
    profitable_count = len([r for r in results if r['total_return_pct'] > 0])
    outperformed_bh = len([r for r in results if r['total_return_pct'] > r['buy_hold_return']])
    total_trades = sum([r['num_trades'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
    avg_drawdown = np.mean([r['max_drawdown'] for r in results])
    
    print(f"\n📈 PERFORMANCE GERAL:")
    print(f"   ROI Médio Spot: {avg_roi:.1f}%")
    print(f"   ROI Médio B&H: {avg_buy_hold:.1f}%")
    print(f"   Assets Lucrativos: {profitable_count}/{len(results)}")
    print(f"   Superou B&H: {outperformed_bh}/{len(results)}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate Médio: {avg_win_rate*100:.1f}%")
    print(f"   Drawdown Médio: {avg_drawdown:.1f}%")
    
    print(f"\n💰 COMPARAÇÃO DETALHADA:")
    for result in sorted(results, key=lambda x: x['total_return_pct'] - x['buy_hold_return'], reverse=True):
        asset = result['asset']
        spot_roi = result['total_return_pct']
        bh_roi = result['buy_hold_return']
        diff = spot_roi - bh_roi
        trades = result['num_trades']
        wr = result['win_rate'] * 100
        
        if diff > 5:
            status = "🟢 MUITO MELHOR"
        elif diff > 0:
            status = "🟡 MELHOR     "
        elif diff > -10:
            status = "🟠 PRÓXIMO    "
        else:
            status = "🔴 PIOR       "
            
        print(f"   {status} {asset}: Spot {spot_roi:+6.1f}% vs B&H {bh_roi:+6.1f}% = {diff:+5.1f}% | {trades} trades")
    
    # Avaliação final
    print(f"\n🎯 AVALIAÇÃO DO SPOT TRADING:")
    
    criteria = [
        (avg_roi > 0, f"ROI Médio Positivo: {avg_roi:.1f}%"),
        (profitable_count >= len(results) * 0.6, f"Maioria Lucrativa: {profitable_count}/{len(results)}"),
        (outperformed_bh >= len(results) * 0.4, f"Supera B&H em ≥40%: {outperformed_bh}/{len(results)}"),
        (avg_drawdown < 30, f"Drawdown Controlado: {avg_drawdown:.1f}%"),
        (avg_win_rate > 0.4, f"Win Rate Aceitável: {avg_win_rate*100:.1f}%")
    ]
    
    passed = sum([c[0] for c in criteria])
    
    for criterion_passed, description in criteria:
        status = "✅" if criterion_passed else "❌"
        print(f"   {status} {description}")
    
    print(f"\n📊 CRITÉRIOS ATENDIDOS: {passed}/5")
    
    if passed >= 4:
        print(f"\n🎉 SPOT TRADING APROVADO!")
        print(f"✅ Estratégia viável sem leverage")
        print(f"✅ Performance consistente")
        print(f"🚀 RECOMENDAÇÃO: Implementar versão spot no trading.py")
        
        # Configuração recomendada
        print(f"\n📋 CONFIGURAÇÃO SPOT RECOMENDADA:")
        print(f"   LEVERAGE = 1 (sem leverage)")
        print(f"   TP_PCT = {config['tp_pct']} (% do preço)")
        print(f"   SL_PCT = {config['sl_pct']} (% do preço)")
        print(f"   VOLUME_MULTIPLIER = {config['volume_mult']}")
        print(f"   MIN_CONFLUENCIA = {config['min_confluencia']}")
        
    elif passed >= 3:
        print(f"\n⚠️ SPOT TRADING PROMISSOR")
        print(f"💡 Pode ser viável com ajustes")
        print(f"🔧 Considerar refinamentos nos filtros")
        
    else:
        print(f"\n❌ SPOT TRADING AINDA INSATISFATÓRIO")
        print(f"🔧 Necessita revisão fundamental da estratégia")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_spot_trading_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'config': config,
        'summary': {
            'avg_roi': avg_roi,
            'avg_buy_hold': avg_buy_hold,
            'profitable_assets': profitable_count,
            'outperformed_bh': outperformed_bh,
            'total_assets': len(results),
            'criteria_passed': passed
        },
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {filename}")

if __name__ == "__main__":
    test_spot_trading()
