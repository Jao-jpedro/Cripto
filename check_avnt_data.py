#!/usr/bin/env python3
"""
Script para verificar dados atuais da Binance para AVNT
e comparar com dados históricos extraídos
"""

import requests
import pandas as pd
from datetime import datetime

def check_binance_avnt():
    """Verifica dados atuais do AVNT na Binance"""
    
    print("🔍 Verificando dados do AVNT na Binance...")
    
    # 1. Verificar preço atual
    current_price_url = "https://api.binance.com/api/v3/ticker/price?symbol=AVNTUSDT"
    response = requests.get(current_price_url)
    
    if response.status_code == 200:
        current_data = response.json()
        print(f"💰 Preço atual AVNTUSDT: ${current_data['price']}")
    else:
        print("❌ Erro ao buscar preço atual")
    
    # 2. Verificar dados históricos recentes (últimas 10 candles de 15m)
    klines_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'AVNTUSDT',
        'interval': '15m',
        'limit': 10
    }
    
    response = requests.get(klines_url, params=params)
    
    if response.status_code == 200:
        klines_data = response.json()
        print(f"\n📊 Últimas {len(klines_data)} candles de 15m:")
        
        for i, candle in enumerate(klines_data):
            timestamp = int(candle[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])
            
            print(f"   {i+1}. {dt.strftime('%Y-%m-%d %H:%M')} | "
                  f"O:{open_price:.6f} H:{high_price:.6f} L:{low_price:.6f} "
                  f"C:{close_price:.6f} V:{volume:.2f}")
    else:
        print("❌ Erro ao buscar dados históricos")

def check_csv_avnt():
    """Verifica dados do AVNT no arquivo CSV"""
    
    print("\n🔍 Verificando dados do AVNT no arquivo CSV...")
    
    csv_file = "tradingv4_historical_data_20251025_192058.csv"
    
    try:
        df = pd.read_csv(csv_file)
        
        # Filtrar apenas dados do AVNT
        avnt_data = df[df['asset_name'] == 'AVNT-USD']
        
        if len(avnt_data) > 0:
            print(f"📊 Registros do AVNT no CSV: {len(avnt_data)}")
            print("\n📈 Primeiros 10 registros:")
            
            for i, row in avnt_data.head(10).iterrows():
                print(f"   {i+1}. {row['datetime']} | "
                      f"O:{row['valor_abertura']:.6f} H:{row['valor_maximo']:.6f} "
                      f"L:{row['valor_minimo']:.6f} C:{row['valor_fechamento']:.6f} "
                      f"V:{row['volume_compra']:.2f}")
                      
            print(f"\n📊 Estatísticas do valor_fechamento AVNT:")
            print(f"   Mínimo: ${avnt_data['valor_fechamento'].min():.6f}")
            print(f"   Máximo: ${avnt_data['valor_fechamento'].max():.6f}")
            print(f"   Média: ${avnt_data['valor_fechamento'].mean():.6f}")
            
        else:
            print("❌ Nenhum dado do AVNT encontrado no CSV")
            
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")

if __name__ == "__main__":
    print("🔍 VERIFICAÇÃO DOS DADOS DO AVNT")
    print("=" * 50)
    
    # Verificar dados da Binance
    check_binance_avnt()
    
    # Verificar dados do CSV
    check_csv_avnt()
    
    print("\n" + "=" * 50)
    print("✅ Verificação completa")
