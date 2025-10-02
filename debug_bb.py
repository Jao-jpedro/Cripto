#!/usr/bin/env python3
"""
Teste específico para debug dos Bollinger Bands
"""

import sys
import os
import pandas as pd
import numpy as np

# Adicionar o diretório atual ao path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingv4 import compute_indicators, BacktestParams, TradingLearner

# Criar dados de teste com 260 pontos (suficiente para passar na validação)
num_points = 260
base_price = 100
prices = []
volumes = []

for i in range(num_points):
    price_change = np.random.normal(0, 0.02) * base_price
    base_price += price_change
    prices.append(base_price)
    volumes.append(1000 + np.random.randint(-200, 200))

test_data = {
    'open': [p * 0.999 for p in prices],
    'high': [p * 1.001 for p in prices],
    'low': [p * 0.998 for p in prices], 
    'close': prices,
    'valor_fechamento': prices,
    'volume': volumes
}

df = pd.DataFrame(test_data)

# Processar com indicadores
p = BacktestParams()
df_with_indicators = compute_indicators(df, p)

print("=== COLUNAS DISPONÍVEIS ===")
print(df_with_indicators.columns.tolist())

print("\n=== ÚLTIMAS 5 LINHAS DOS BOLLINGER BANDS ===")
bb_cols = [col for col in df_with_indicators.columns if 'bb_' in col]
print(f"Colunas BB encontradas: {bb_cols}")

if bb_cols:
    print(df_with_indicators[bb_cols].tail())

print("\n=== TESTE DE EXTRAÇÃO FEATURES_RAW ===")
learner = TradingLearner()

print(f"DataFrame shape: {df_with_indicators.shape}")

try:
    features_raw = learner.extract_features_raw("TEST", "buy", df_with_indicators, prices[-1])
    print(f"Função extract_features_raw executou com sucesso. Tamanho: {len(features_raw)}")
    
    if len(features_raw) > 0:
        print("Bollinger Bands em features_raw:")
        bb_found = False
        for key, value in features_raw.items():
            if 'bb_' in key:
                print(f"  {key}: {value}")
                bb_found = True
                
        if not bb_found:
            print("  Nenhum Bollinger Band encontrado!")
            
        print("\n=== TESTE DE BINNING ===")
        features_binned = learner.bin_features(features_raw)
        
        print("Bollinger Bands em features_binned:")
        for key, value in features_binned.items():
            if 'bb_' in key:
                print(f"  {key}: {value}")
    else:
        print("⚠️ features_raw está vazio")
        
except Exception as e:
    print(f"Erro na função extract_features_raw: {e}")
    import traceback
    traceback.print_exc()
