#!/usr/bin/env python3
"""
Teste Final da Versão Conservadora V3 do trading.py
Baseado no diagnóstico: filtros permissivos, leverage baixo, TP/SL conservador
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_conservative_config():
    """Testa a configuração conservadora baseada no diagnóstico"""
    
    print("🛡️ TESTE DA VERSÃO CONSERVADORA V3 - TRADING.PY")
    print("🎯 Baseado no DIAGNÓSTICO: Filtros permissivos + Baixo risco")
    print("="*75)
    
    # Configuração conservadora baseada no diagnóstico
    config = {
        'tp_pct': 15.0,      # Conservador (era 25%)
        'sl_pct': 5.0,       # Conservador (era 10%)
        'leverage': 10,      # Reduzido (era 20)
        'atr_min': 0.4,      # Mais permissivo (era 0.5)
        'atr_max': 2.0,      # Mais conservador (era 3.0)
        'volume_mult': 1.5,  # Muito mais permissivo (era 3.0)
        'min_confluencia': 2, # Mais permissivo (era 3)
        'ema_short': 7,      # Mantido
        'ema_long': 21,      # Mantido
        'breakout_k': 0.5,   # Muito conservador (era 0.8)
    }
    
    print("📋 CONFIGURAÇÃO CONSERVADORA V3:")
    print(f"   TP/SL: {config['tp_pct']}%/{config['sl_pct']}% (R:R = {config['tp_pct']/config['sl_pct']:.1f}:1)")
    print(f"   Leverage: {config['leverage']}x (reduzido de 20x)")
    print(f"   ATR Range: {config['atr_min']}-{config['atr_max']}% (mais permissivo)")
    print(f"   Volume: {config['volume_mult']}x média (muito mais permissivo)")
    print(f"   Confluência mín: {config['min_confluencia']} (mais oportunidades)")
    print(f"   Breakout K: {config['breakout_k']} (entradas mais cedo)")
    print()
    
    # Testar primeiro no BTC
    print("🧪 TESTE INICIAL - BTC")
    print("-"*40)
    
    btc_result = test_single_asset_conservative("BTC", "dados_reais_btc_1ano.csv", config)
    
    if btc_result and btc_result['total_return_pct'] > -50:
        print("✅ BTC teve resultado aceitável, testando outros assets...")
        
        # Testar outros assets
        assets_to_test = [
            ("ETH", "dados_reais_eth_1ano.csv"),
            ("BNB", "dados_reais_bnb_1ano.csv"),
            ("SOL", "dados_reais_sol_1ano.csv"),
            ("ADA", "dados_reais_ada_1ano.csv"),
            ("AVAX", "dados_reais_avax_1ano.csv")
        ]
        
        all_results = [btc_result]
        
        for asset_name, filename in assets_to_test:
            if os.path.exists(filename):
                result = test_single_asset_conservative(asset_name, filename, config)
                if result:
                    all_results.append(result)
        
        # Análise final
        analyze_conservative_results(all_results, config)
        
    else:
        print("❌ BTC teve resultado muito ruim, a estratégia precisa ser revista")
        if btc_result:
            print(f"   ROI BTC: {btc_result['total_return_pct']:.1f}%")
            print(f"   Trades: {btc_result['num_trades']}")
            print(f"   Win Rate: {btc_result['win_rate']*100:.1f}%")

def test_single_asset_conservative(asset_name, filename, config):
    """Testa configuração conservadora em um asset"""
    
    if not os.path.exists(filename):
        print(f"⚠️ {asset_name}: Arquivo não encontrado")
        return None
    
    try:
        df = load_data_robust(filename)
        if df is None or len(df) < 100:
            print(f"⚠️ {asset_name}: Dados insuficientes")
            return None
        
        # Calcular indicadores
        df = calculate_indicators_conservative(df, config)
        
        # Simular trading conservador
        result = simulate_conservative_trading(df, config)
        result['asset'] = asset_name
        
        # Log resultado
        roi = result['total_return_pct']
        trades = result['num_trades']
        wr = result['win_rate'] * 100
        dd = result['max_drawdown']
        
        status = "✅" if roi > 0 else "⚠️" if roi > -30 else "❌"
        print(f"{status} {asset_name}: {roi:+6.1f}% | {trades:3d} trades | WR {wr:5.1f}% | DD {dd:5.1f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ {asset_name}: Erro - {str(e)[:40]}")
        return None

def load_data_robust(filename):
    """Carrega dados com tratamento robusto"""
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
        
        # Verificar colunas essenciais
        required = ['valor_abertura', 'valor_maximo', 'valor_minimo', 'valor_fechamento', 'volume']
        if not all(col in df.columns for col in required):
            return None
        
        # Limpar dados
        df = df.dropna()
        df = df[df['valor_fechamento'] > 0]
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception:
        return None

def calculate_indicators_conservative(df, config):
    """Calcula indicadores para estratégia conservadora"""
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

def simulate_conservative_trading(df, config):
    """Simula trading com estratégia conservadora"""
    
    balance = 1000.0
    trades = []
    position = None
    
    for i in range(config['ema_long'], len(df) - 1):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Gerenciar posição existente
        if position:
            entry_price = position['entry_price']
            side = position['side']
            entry_balance = position['entry_balance']
            
            # Calcular P&L com leverage reduzido
            if side == 'long':
                pnl_pct = ((current['valor_fechamento'] - entry_price) / entry_price) * config['leverage']
            else:
                pnl_pct = ((entry_price - current['valor_fechamento']) / entry_price) * config['leverage']
            
            # Verificar saída (TP/SL conservador)
            tp_pct = config['tp_pct'] / 100.0
            sl_pct = config['sl_pct'] / 100.0
            
            if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                final_balance = entry_balance * (1 + pnl_pct)
                
                trades.append({
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': current['valor_fechamento'],
                    'pnl_pct': pnl_pct,
                    'balance_after': final_balance
                })
                
                balance = final_balance
                position = None
                continue
        
        # Verificar entrada (critérios permissivos)
        if not position:
            can_long, can_short = check_conservative_entry(current, prev, config)
            
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
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'final_balance': balance
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
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown': max_dd * 100,
        'final_balance': balance
    }

def check_conservative_entry(current, prev, config):
    """Verifica entrada com critérios conservadores e permissivos"""
    
    # EMA Cross
    ema_cross_bullish = (prev['ema_short'] <= prev['ema_long'] and 
                        current['ema_short'] > current['ema_long'])
    ema_cross_bearish = (prev['ema_short'] >= prev['ema_long'] and 
                        current['ema_short'] < current['ema_long'])
    
    criterios_long = 0
    criterios_short = 0
    
    # 1. EMA Cross (peso 1)
    if ema_cross_bullish:
        criterios_long += 1
    if ema_cross_bearish:
        criterios_short += 1
    
    # 2. ATR em range (mais permissivo)
    atr_ok = config['atr_min'] <= current['atr_pct'] <= config['atr_max']
    if atr_ok:
        criterios_long += 1
        criterios_short += 1
    
    # 3. Volume (muito mais permissivo)
    volume_ok = current['volume'] > (current['vol_ma'] * config['volume_mult'])
    if volume_ok:
        criterios_long += 1
        criterios_short += 1
    
    # 4. RSI force (peso extra para entradas extremas)
    if current['rsi'] < 25:  # Oversold mais permissivo
        criterios_long += 2
    elif current['rsi'] > 75:  # Overbought mais permissivo
        criterios_short += 2
    
    # Decisão com confluência mínima baixa
    can_long = criterios_long >= config['min_confluencia']
    can_short = criterios_short >= config['min_confluencia']
    
    return can_long, can_short

def analyze_conservative_results(results, config):
    """Analisa resultados da versão conservadora"""
    
    print("\n" + "="*75)
    print("📊 ANÁLISE FINAL - VERSÃO CONSERVADORA V3")
    print("="*75)
    
    if not results:
        print("❌ Nenhum resultado válido!")
        return
    
    # Métricas gerais
    avg_roi = np.mean([r['total_return_pct'] for r in results])
    profitable_count = len([r for r in results if r['total_return_pct'] > 0])
    total_trades = sum([r['num_trades'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
    avg_drawdown = np.mean([r['max_drawdown'] for r in results])
    
    print(f"\n📈 PERFORMANCE GERAL:")
    print(f"   ROI Médio: {avg_roi:.1f}%")
    print(f"   Assets Lucrativos: {profitable_count}/{len(results)}")
    print(f"   Total de Trades: {total_trades}")
    print(f"   Win Rate Médio: {avg_win_rate*100:.1f}%")
    print(f"   Drawdown Médio: {avg_drawdown:.1f}%")
    
    print(f"\n💰 RESULTADOS DETALHADOS:")
    for result in sorted(results, key=lambda x: x['total_return_pct'], reverse=True):
        asset = result['asset']
        roi = result['total_return_pct']
        trades = result['num_trades']
        wr = result['win_rate'] * 100
        dd = result['max_drawdown']
        
        if roi > 20:
            status = "🟢 EXCELENTE"
        elif roi > 0:
            status = "🟡 BOM      "
        elif roi > -20:
            status = "🟠 ACEITÁVEL"
        else:
            status = "🔴 RUIM     "
            
        print(f"   {status} {asset}: {roi:+7.1f}% | {trades:3d} trades | WR {wr:5.1f}% | DD {dd:5.1f}%")
    
    # Avaliação da configuração
    print(f"\n🎯 AVALIAÇÃO DA CONFIGURAÇÃO CONSERVADORA:")
    
    success_criteria = [
        (avg_roi > 0, f"ROI Médio Positivo: {avg_roi:.1f}%"),
        (profitable_count >= len(results) * 0.5, f"Maioria Lucrativa: {profitable_count}/{len(results)}"),
        (avg_win_rate > 0.4, f"Win Rate Aceitável: {avg_win_rate*100:.1f}%"),
        (avg_drawdown < 50, f"Drawdown Controlado: {avg_drawdown:.1f}%"),
        (total_trades > 50, f"Trades Suficientes: {total_trades}")
    ]
    
    passed_criteria = sum([criterion[0] for criterion in success_criteria])
    
    for passed, description in success_criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {description}")
    
    print(f"\n📊 CRITÉRIOS ATENDIDOS: {passed_criteria}/5")
    
    if passed_criteria >= 4:
        print(f"\n🎉 CONFIGURAÇÃO CONSERVADORA APROVADA!")
        print(f"✅ Boa relação risco/retorno")
        print(f"✅ Drawdown controlado") 
        print(f"✅ Performance consistente")
        print(f"🚀 RECOMENDAÇÃO: Implementar no trading.py")
    elif passed_criteria >= 3:
        print(f"\n⚠️ CONFIGURAÇÃO MODERADAMENTE APROVADA")
        print(f"💡 Pode ser usada com monitoramento rigoroso")
        print(f"🔧 Considerar ajustes finos nos parâmetros")
    else:
        print(f"\n❌ CONFIGURAÇÃO AINDA PRECISA MELHORIAS")
        print(f"🔧 Revisar parâmetros fundamentais")
        print(f"📊 Considerar estratégia diferente")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_conservador_v3_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'config': config,
        'summary': {
            'avg_roi': avg_roi,
            'profitable_assets': profitable_count,
            'total_assets': len(results),
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'avg_drawdown': avg_drawdown,
            'criteria_passed': passed_criteria
        },
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {filename}")

if __name__ == "__main__":
    test_conservative_config()
