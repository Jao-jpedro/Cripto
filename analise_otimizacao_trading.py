#!/usr/bin/env python3
"""
Análise Detalhada dos Resultados de Otimização do trading.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_optimization_results():
    """Analisa os resultados da otimização"""
    
    # Encontrar o arquivo mais recente
    files = [f for f in os.listdir('.') if f.startswith('otimizacao_trading_py_') and f.endswith('.json')]
    if not files:
        print("❌ Nenhum arquivo de otimização encontrado!")
        return
    
    latest_file = sorted(files)[-1]
    print(f"📊 Analisando: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print("\n🔍 ANÁLISE DETALHADA DOS RESULTADOS")
    print("="*60)
    
    # Analisar os top 5 resultados
    top_results = data['top_results'][:5]
    
    for i, result in enumerate(top_results, 1):
        config = result['config']
        asset_results = result['asset_results']
        
        print(f"\n📋 CONFIGURAÇÃO #{i}")
        print("-"*40)
        print(f"TP/SL: {config['tp_pct']}%/{config['sl_pct']}% (R:R = {config['tp_pct']/config['sl_pct']:.1f})")
        print(f"ATR: {config['atr_min']}-{config['atr_max']}%")
        print(f"Volume: {config['volume_mult']}x")
        print(f"EMAs: {config['ema_short']}/{config['ema_long']}")
        print(f"Confluência: {config['min_confluencia']}")
        print(f"ROI Médio: {result['avg_return_pct']:.1f}%")
        print(f"Win Rate: {result['avg_win_rate']*100:.1f}%")
        print(f"Max DD: {result['avg_max_drawdown']:.1f}%")
        
        # Analisar assets individualmente
        profitable_assets = [a for a in asset_results if a['total_return_pct'] > 0]
        losing_assets = [a for a in asset_results if a['total_return_pct'] < 0]
        
        print(f"\n💰 Assets Lucrativos: {len(profitable_assets)}/10")
        for asset in profitable_assets:
            print(f"   ✅ {asset['asset']}: +{asset['total_return_pct']:.1f}% ({asset['num_trades']} trades)")
            
        print(f"\n📉 Assets com Perdas: {len(losing_assets)}/10")
        for asset in losing_assets[:3]:  # Mostrar só os 3 piores
            print(f"   ❌ {asset['asset']}: {asset['total_return_pct']:.1f}% ({asset['num_trades']} trades)")
        
        # Verificar padrões
        high_win_rates = [a for a in asset_results if a['win_rate'] > 0.5 and a['num_trades'] > 10]
        if high_win_rates:
            print(f"\n🎯 Assets com Alta Win Rate (>50%):")
            for asset in high_win_rates:
                print(f"   🔥 {asset['asset']}: WR={asset['win_rate']*100:.1f}% | ROI={asset['total_return_pct']:.1f}%")

def identify_problems():
    """Identifica problemas na estratégia atual"""
    
    print("\n🚨 PROBLEMAS IDENTIFICADOS")
    print("="*60)
    
    problems = [
        "1. Win Rate muito baixo (~42% médio)",
        "2. Drawdowns extremos (>90% em muitos casos)", 
        "3. Resultados inconsistentes entre assets",
        "4. Alguns assets têm ROI negativo extremo (-99%+)",
        "5. Configuração pode estar overfitted para poucos assets (BNB, AVAX)"
    ]
    
    for problem in problems:
        print(f"   ❌ {problem}")
    
    print("\n💡 POSSÍVEIS SOLUÇÕES")
    print("-"*40)
    
    solutions = [
        "1. Revisar lógica de entrada (talvez muito restritiva)",
        "2. Implementar stop loss mais conservador",
        "3. Testar configurações mais balanceadas",
        "4. Considerar filtros de tendência macro",
        "5. Validar se dados históricos estão corretos"
    ]
    
    for solution in solutions:
        print(f"   ✅ {solution}")

def suggest_improved_config():
    """Sugere configuração melhorada baseada na análise"""
    
    print("\n🎯 CONFIGURAÇÃO MELHORADA SUGERIDA")
    print("="*60)
    
    print("""
📋 PARÂMETROS CONSERVADORES PARA TRADING.PY:

# Configuração de Risk/Reward mais balanceada
TP_PCT = 25.0                    # TP menor que o atual (era 30%)
SL_PCT = 10.0                    # SL igual ao atual  
RISK_REWARD = 2.5:1              # Mais conservador que 2.5:1 da otimização

# Filtros mais seletivos
ATR_PCT_MIN = 0.5                # Mais restritivo que 0.3%
ATR_PCT_MAX = 3.0                # Mais restritivo que 4.0%
VOLUME_MULTIPLIER = 3.0          # Manter o mesmo
MIN_CONFLUENCIA = 3              # Menos restritivo que 4

# EMAs mais tradicionais  
EMA_SHORT_SPAN = 7               # Manter
EMA_LONG_SPAN = 21               # Voltar para 21 (mais tradicional que 24)
BREAKOUT_K_ATR = 0.8             # Mais conservador que 1.0

💡 JUSTIFICATIVA:
- TP/SL mais conservador para melhor win rate
- ATR range mais restritivo para evitar mercados muito voláteis
- Confluência 3 ao invés de 4 para mais oportunidades
- EMA 21 é configuração mais testada historicamente
- Breakout menor para entradas mais cedo na tendência
""")

def create_improved_test():
    """Cria um teste com a configuração melhorada"""
    
    improved_config = {
        'tp_pct': 25.0,
        'sl_pct': 10.0,
        'atr_min': 0.5,
        'atr_max': 3.0,
        'volume_mult': 3.0,
        'min_confluencia': 3,
        'ema_short': 7,
        'ema_long': 21,
        'breakout_k': 0.8
    }
    
    # Salvar configuração sugerida
    output = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'type': 'improved_config',
        'description': 'Configuração melhorada baseada em análise dos resultados',
        'config': improved_config,
        'reasoning': {
            'tp_sl_ratio': '2.5:1 para melhor balanceamento',
            'atr_range': 'Mais restritivo para evitar volatilidade extrema',
            'confluencia': 'Reduzido para mais oportunidades',
            'ema_period': 'EMA 21 é mais tradicional e testada',
            'breakout_k': 'Menor para entradas mais precoces'
        }
    }
    
    filename = f"config_melhorada_trading_py_{output['timestamp']}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Configuração melhorada salva em: {filename}")
    
    return improved_config

def main():
    print("🔬 ANÁLISE DOS RESULTADOS DE OTIMIZAÇÃO - TRADING.PY")
    print("🎯 Identificando problemas e sugerindo melhorias")
    print()
    
    # Analisar resultados
    analyze_optimization_results()
    
    # Identificar problemas  
    identify_problems()
    
    # Sugerir melhorias
    suggest_improved_config()
    
    # Criar configuração melhorada
    improved_config = create_improved_test()
    
    print("\n🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    print("="*50)
    print("1. Implementar a configuração melhorada no trading.py")
    print("2. Testar com dados históricos menores (3-6 meses)")
    print("3. Validar lógica de entrada/saída")
    print("4. Considerar implementar filtros de tendência macro")
    print("5. Testar em modo paper trading antes de ir ao vivo")
    
    print(f"\n💡 A configuração atual pode estar muito agressiva.")
    print(f"🎯 Foque em consistência ao invés de ROI máximo.")

if __name__ == "__main__":
    main()
