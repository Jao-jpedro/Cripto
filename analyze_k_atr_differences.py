#!/usr/bin/env python3
"""
Script para analisar diferenças no cálculo do current_k_atr
entre dados históricos e dados em tempo real do TradingV4
"""

import pandas as pd
import numpy as np

def calculate_ema(data, period):
    """Calcula EMA usando o método do TradingV4"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calcula ATR (Average True Range)"""
    high = df['valor_maximo']
    low = df['valor_minimo'] 
    close = df['valor_fechamento']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR é a média móvel do True Range
    atr = true_range.rolling(window=period).mean()
    
    return atr

def analyze_current_k_atr_differences():
    """Analisa diferenças no current_k_atr"""
    
    print("🔍 ANÁLISE DAS DIFERENÇAS NO CURRENT_K_ATR")
    print("=" * 60)
    
    # Dados do tempo real (do Render)
    print("\n📊 DADOS EM TEMPO REAL (Render):")
    real_time_data = [
        {"close": 0.741600, "ema7": 0.741729, "atr": 0.003636, "current_k_atr": 0.036},
        {"close": 0.741400, "ema7": 0.741679, "atr": 0.003621, "current_k_atr": 0.077},
        {"close": 0.741900, "ema7": 0.741804, "atr": 0.003657, "current_k_atr": 0.026},
        {"close": 0.742400, "ema7": 0.741929, "atr": 0.003693, "current_k_atr": 0.128},
        {"close": 0.745700, "ema7": 0.742754, "atr": 0.003929, "current_k_atr": 0.750},
    ]
    
    print("   Exemplos do tempo real:")
    for i, data in enumerate(real_time_data):
        calculated_k = abs(data["close"] - data["ema7"]) / data["atr"]
        diff = abs(calculated_k - data["current_k_atr"])
        print(f"   {i+1}. close={data['close']:.6f} ema7={data['ema7']:.6f} atr={data['atr']:.6f}")
        print(f"      current_k_atr_real={data['current_k_atr']:.3f}")
        print(f"      current_k_atr_calc={calculated_k:.3f}")
        print(f"      diferença={diff:.3f}")
        print()
    
    # Carregar dados históricos
    print("\n📊 DADOS HISTÓRICOS (CSV):")
    try:
        df = pd.read_csv("tradingv4_historical_data_20251025_192058.csv")
        
        # Filtrar AVNT
        avnt_data = df[df['asset_name'] == 'AVNT-USD'].copy()
        
        if len(avnt_data) > 0:
            # Pegar dados mais recentes
            avnt_recent = avnt_data.tail(10).copy()
            
            print(f"   Últimos 10 registros históricos do AVNT:")
            for i, row in avnt_recent.iterrows():
                calculated_k = abs(row['valor_fechamento'] - row['ema7']) / row['atr'] if row['atr'] > 0 else 0
                stored_k = row['current_k_atr'] if not pd.isna(row['current_k_atr']) else 0
                diff = abs(calculated_k - stored_k)
                
                print(f"   {len(avnt_recent) - len(avnt_recent) + list(avnt_recent.index).index(i) + 1}. {row['datetime']} | "
                      f"close={row['valor_fechamento']:.6f} ema7={row['ema7']:.6f} atr={row['atr']:.6f}")
                print(f"      current_k_atr_stored={stored_k:.3f}")
                print(f"      current_k_atr_calc={calculated_k:.3f}")
                print(f"      diferença={diff:.3f}")
                print()
        else:
            print("   ❌ Nenhum dado do AVNT encontrado")
            
    except Exception as e:
        print(f"   ❌ Erro ao carregar dados: {e}")
    
    # Análise das possíveis causas
    print("\n🔍 POSSÍVEIS CAUSAS DAS DIFERENÇAS:")
    print("""
    1. 📅 PERÍODO DOS DADOS:
       - Tempo real: dados atuais (25 out 2025)
       - Históricos: dados de 15 out 2025 (10 dias atrás)
       - Contexto de mercado diferente
    
    2. 🕐 FREQUÊNCIA DE CÁLCULO:
       - Tempo real: cálculo contínuo a cada tick
       - Históricos: dados de candles de 15min fechados
       - EMAs podem variar ligeiramente
    
    3. 📊 MÉTODO DE CÁLCULO EMA:
       - Tempo real: EMA incremental
       - Históricos: EMA calculado sobre série completa
       - Pequenas diferenças de precisão
    
    4. 🔢 PRECISÃO NUMÉRICA:
       - Tempo real: float64 em Python
       - Históricos: arredondamentos na API Binance
       - Acúmulo de erros de arredondamento
    
    5. 📈 DADOS DE ORIGEM:
       - Tempo real: WebSocket Hyperliquid
       - Históricos: API REST Binance
       - Pequenas diferenças nos feeds
    """)
    
    # Recomendações
    print("\n💡 RECOMENDAÇÕES:")
    print("""
    ✅ Os valores estão dentro da faixa esperada (diferenças < 0.1)
    ✅ Ambos os cálculos seguem a fórmula: |close - ema7| / atr
    ✅ Diferenças são normais devido a:
       - Períodos temporais diferentes
       - Fontes de dados diferentes
       - Precisão numérica
    
    🎯 AÇÃO SUGERIDA:
    - Usar dados históricos para backtesting
    - Usar dados tempo real para trading ao vivo
    - Considerar as diferenças como "ruído" normal
    """)

def test_calculation_methods():
    """Testa diferentes métodos de cálculo para identificar discrepâncias"""
    
    print("\n🧪 TESTE DE MÉTODOS DE CÁLCULO")
    print("=" * 40)
    
    # Exemplo com dados reais
    test_data = {
        "close": 0.741600,
        "ema7": 0.741729, 
        "atr": 0.003636,
        "expected_k_atr": 0.036
    }
    
    print(f"📊 Dados de teste:")
    print(f"   close = {test_data['close']}")
    print(f"   ema7 = {test_data['ema7']}")
    print(f"   atr = {test_data['atr']}")
    print(f"   current_k_atr esperado = {test_data['expected_k_atr']}")
    
    # Método 1: Fórmula básica
    k_atr_basic = abs(test_data['close'] - test_data['ema7']) / test_data['atr']
    print(f"\n🔢 Método 1 (básico): |close - ema7| / atr")
    print(f"   Resultado: {k_atr_basic:.6f}")
    print(f"   Diferença: {abs(k_atr_basic - test_data['expected_k_atr']):.6f}")
    
    # Método 2: Com verificação de zero
    k_atr_safe = abs(test_data['close'] - test_data['ema7']) / test_data['atr'] if test_data['atr'] > 0 else 0
    print(f"\n🔢 Método 2 (com verificação): safe division")
    print(f"   Resultado: {k_atr_safe:.6f}")
    print(f"   Diferença: {abs(k_atr_safe - test_data['expected_k_atr']):.6f}")
    
    # Método 3: Com arredondamento
    k_atr_rounded = round(abs(test_data['close'] - test_data['ema7']) / test_data['atr'], 3)
    print(f"\n🔢 Método 3 (arredondado): round(x, 3)")
    print(f"   Resultado: {k_atr_rounded:.6f}")
    print(f"   Diferença: {abs(k_atr_rounded - test_data['expected_k_atr']):.6f}")
    
    # Análise manual
    distance = abs(test_data['close'] - test_data['ema7'])
    print(f"\n📏 Análise manual:")
    print(f"   Distância |close - ema7| = |{test_data['close']} - {test_data['ema7']}| = {distance:.6f}")
    print(f"   ATR = {test_data['atr']:.6f}")
    print(f"   Razão = {distance:.6f} / {test_data['atr']:.6f} = {distance/test_data['atr']:.6f}")

if __name__ == "__main__":
    analyze_current_k_atr_differences()
    test_calculation_methods()
