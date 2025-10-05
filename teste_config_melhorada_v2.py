#!/usr/bin/env python3
"""
Teste da Configuração Melhorada V2 do trading.py
Foca em consistência ao invés de ROI máximo
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_improved_config():
    """Testa a configuração melhorada com dados históricos"""
    
    print("🚀 TESTE DA CONFIGURAÇÃO MELHORADA V2 - TRADING.PY")
    print("🎯 Foco em CONSISTÊNCIA e REDUÇÃO DE RISCO")
    print("="*70)
    
    # Configuração melhorada
    config = {
        'tp_pct': 25.0,      # Mais conservador que 30%
        'sl_pct': 10.0,      # Mantido
        'atr_min': 0.5,      # Mais restritivo
        'atr_max': 3.0,      # Mais restritivo que 4.0%
        'volume_mult': 3.0,  # Mantido
        'min_confluencia': 3, # Menos restritivo que 4
        'ema_short': 7,      # Mantido
        'ema_long': 21,      # Tradicional (era 24)
        'breakout_k': 0.8,   # Mais conservador que 1.0
        'leverage': 20
    }
    
    print("📋 CONFIGURAÇÃO MELHORADA:")
    print(f"   TP/SL: {config['tp_pct']}%/{config['sl_pct']}% (R:R = {config['tp_pct']/config['sl_pct']:.1f}:1)")
    print(f"   ATR Range: {config['atr_min']}-{config['atr_max']}%")
    print(f"   Volume: {config['volume_mult']}x média")
    print(f"   EMAs: {config['ema_short']}/{config['ema_long']}")
    print(f"   Confluência mín: {config['min_confluencia']}")
    print(f"   Breakout K: {config['breakout_k']}")
    print()
    
    # Assets para teste
    data_files = [
        ("BTC", "dados_reais_btc_1ano.csv"),
        ("ETH", "dados_reais_eth_1ano.csv"),
        ("BNB", "dados_reais_bnb_1ano.csv"),
        ("SOL", "dados_reais_sol_1ano.csv"),
        ("ADA", "dados_reais_ada_1ano.csv"),
        ("AVAX", "dados_reais_avax_1ano.csv")
    ]
    
    results = []
    
    print("🧪 TESTANDO EM ASSETS PRINCIPAIS...")
    print("-"*50)
    
    for asset_name, filename in data_files:
        if not os.path.exists(filename):
            print(f"⚠️ {asset_name}: Arquivo {filename} não encontrado")
            continue
            
        try:
            result = test_single_asset(asset_name, filename, config)
            if result:
                results.append(result)
                
                # Log resultado imediato
                roi = result['total_return_pct']
                trades = result['num_trades']
                wr = result['win_rate'] * 100
                dd = result['max_drawdown']
                
                status = "✅" if roi > 0 else "❌"
                print(f"{status} {asset_name}: {roi:+.1f}% | {trades} trades | WR {wr:.1f}% | DD {dd:.1f}%")
                
        except Exception as e:
            print(f"❌ {asset_name}: Erro - {str(e)[:50]}")
    
    if not results:
        print("❌ Nenhum resultado válido obtido!")
        return
    
    # Análise dos resultados
    analyze_results(results, config)

def test_single_asset(asset_name, filename, config):
    """Testa configuração em um único asset"""
    
    df = load_and_prepare_data(filename)
    if df is None or len(df) < 100:
        return None
    
    # Calcular indicadores
    df = calculate_indicators(df, config)
    
    # Simular trading
    balance = 1000.0
    trades = []
    position = None
    
    for i in range(config['ema_long'], len(df) - 1):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Detectar EMA Cross
        ema_cross_bullish = (prev['ema_short'] <= prev['ema_long'] and 
                           current['ema_short'] > current['ema_long'])
        ema_cross_bearish = (prev['ema_short'] >= prev['ema_long'] and 
                           current['ema_short'] < current['ema_long'])
        
        # Gerenciar posição existente
        if position:
            entry_price = position['entry_price']
            side = position['side']
            entry_balance = position['entry_balance']
            
            # Calcular P&L
            if side == 'long':
                pnl_pct = ((current['valor_fechamento'] - entry_price) / entry_price) * config['leverage']
            else:
                pnl_pct = ((entry_price - current['valor_fechamento']) / entry_price) * config['leverage']
            
            # Verificar saída
            tp_pct = config['tp_pct'] / 100.0
            sl_pct = config['sl_pct'] / 100.0
            
            if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                final_balance = entry_balance * (1 + pnl_pct)
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current['valor_fechamento'],
                    'side': side,
                    'pnl_pct': pnl_pct,
                    'balance_after': final_balance
                })
                
                balance = final_balance
                position = None
                continue
        
        # Verificar entrada
        if not position:
            can_long, can_short = check_entry_conditions(current, ema_cross_bullish, ema_cross_bearish, config)
            
            if can_long:
                position = {
                    'side': 'long',
                    'entry_price': current['valor_fechamento'],
                    'entry_balance': balance
                }
            elif can_short:
                position = {
                    'side': 'short',
                    'entry_price': current['valor_fechamento'],
                    'entry_balance': balance
                }
    
    # Calcular métricas
    if not trades:
        return {
            'asset': asset_name,
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
    
    total_return_pct = ((balance - 1000.0) / 1000.0) * 100
    wins = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    # Calcular drawdown
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
        'asset': asset_name,
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown': max_dd * 100,
        'final_balance': balance
    }

def load_and_prepare_data(filename):
    """Carrega e prepara dados"""
    try:
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
        
        required_cols = ['valor_abertura', 'valor_maximo', 'valor_minimo', 'valor_fechamento', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
            
        return df.reset_index(drop=True)
        
    except Exception:
        return None

def calculate_indicators(df, config):
    """Calcula indicadores técnicos"""
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

def check_entry_conditions(row, ema_cross_bullish, ema_cross_bearish, config):
    """Verifica condições de entrada com confluência"""
    
    criterios_long = 0
    criterios_short = 0
    
    # 1. EMA Cross
    if ema_cross_bullish:
        criterios_long += 1
    if ema_cross_bearish:
        criterios_short += 1
    
    # 2. ATR em range aceitável
    atr_ok = config['atr_min'] <= row['atr_pct'] <= config['atr_max']
    if atr_ok:
        criterios_long += 1
        criterios_short += 1
    
    # 3. Volume acima da média
    volume_ok = row['volume'] > (row['vol_ma'] * config['volume_mult'])
    if volume_ok:
        criterios_long += 1
        criterios_short += 1
    
    # 4. Breakout confirmado
    if ema_cross_bullish:
        breakout_long = row['valor_fechamento'] > (row['ema_short'] + config['breakout_k'] * row['atr'])
        if breakout_long:
            criterios_long += 1
    
    if ema_cross_bearish:
        breakout_short = row['valor_fechamento'] < (row['ema_short'] - config['breakout_k'] * row['atr'])
        if breakout_short:
            criterios_short += 1
    
    # 5. RSI force (peso extra)
    if row['rsi'] < 20:  # Oversold
        criterios_long += 2
    elif row['rsi'] > 80:  # Overbought
        criterios_short += 2
    
    can_long = criterios_long >= config['min_confluencia']
    can_short = criterios_short >= config['min_confluencia']
    
    return can_long, can_short

def analyze_results(results, config):
    """Analisa e exibe resultados"""
    
    print("\n" + "="*70)
    print("📊 ANÁLISE DOS RESULTADOS - CONFIGURAÇÃO MELHORADA V2")
    print("="*70)
    
    # Métricas gerais
    profitable_assets = [r for r in results if r['total_return_pct'] > 0]
    total_roi = np.mean([r['total_return_pct'] for r in results])
    total_trades = sum([r['num_trades'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
    avg_drawdown = np.mean([r['max_drawdown'] for r in results])
    
    print(f"\n📈 PERFORMANCE GERAL:")
    print(f"   ROI Médio: {total_roi:.1f}%")
    print(f"   Assets Lucrativos: {len(profitable_assets)}/{len(results)}")
    print(f"   Total de Trades: {total_trades}")
    print(f"   Win Rate Médio: {avg_win_rate*100:.1f}%")
    print(f"   Drawdown Médio: {avg_drawdown:.1f}%")
    
    print(f"\n💰 RESULTADOS POR ASSET:")
    for result in sorted(results, key=lambda x: x['total_return_pct'], reverse=True):
        asset = result['asset']
        roi = result['total_return_pct']
        trades = result['num_trades']
        wr = result['win_rate'] * 100
        dd = result['max_drawdown']
        
        status = "🟢" if roi > 20 else "🟡" if roi > 0 else "🔴"
        print(f"   {status} {asset}: {roi:+7.1f}% | {trades:3d} trades | WR {wr:5.1f}% | DD {dd:5.1f}%")
    
    # Comparação com configuração anterior
    print(f"\n🔄 COMPARAÇÃO COM CONFIGURAÇÃO ANTERIOR:")
    print(f"   ✅ TP/SL: 25%/10% (era 30%/10%) - Mais conservador")
    print(f"   ✅ ATR Max: 3.0% (era 4.0%) - Menos volátil")
    print(f"   ✅ Confluência: 3 (era 4) - Mais oportunidades")
    print(f"   ✅ EMA: 7/21 (era 7/24) - Mais tradicional")
    print(f"   ✅ Breakout K: 0.8 (era 1.0) - Entrada mais cedo")
    
    # Recomendações
    if total_roi > 50 and len(profitable_assets) >= len(results) * 0.6:
        print(f"\n🎉 CONFIGURAÇÃO APROVADA!")
        print(f"   ✅ ROI médio > 50%")
        print(f"   ✅ Maioria dos assets lucrativa")
        print(f"   🚀 RECOMENDAÇÃO: Implementar no trading.py")
    elif total_roi > 0 and avg_drawdown < 50:
        print(f"\n⚠️ CONFIGURAÇÃO MODERADA")
        print(f"   ✅ ROI positivo")
        print(f"   ✅ Drawdown controlado")
        print(f"   💡 RECOMENDAÇÃO: Pode ser usada com cautela")
    else:
        print(f"\n❌ CONFIGURAÇÃO PRECISA MELHORIAS")
        print(f"   ⚠️ ROI baixo ou drawdown alto")
        print(f"   🔧 RECOMENDAÇÃO: Ajustar parâmetros")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_config_melhorada_v2_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'config': config,
        'summary': {
            'avg_roi': total_roi,
            'profitable_assets': len(profitable_assets),
            'total_assets': len(results),
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'avg_drawdown': avg_drawdown
        },
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {filename}")

if __name__ == "__main__":
    test_improved_config()
