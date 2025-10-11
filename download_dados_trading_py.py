#!/usr/bin/env python3
"""
📥 DOWNLOAD DADOS REAIS 1 ANO - ASSETS TRADING.PY
Baixa dados históricos de 1 ano para todos os assets do trading.py
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os

# Assets do trading.py (18 assets)
TRADING_ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD", 
    "PUMP-USD", "AVNT-USD", "LINK-USD", "WLD-USD", "AAVE-USD",
    "CRV-USD", "LTC-USD", "NEAR-USD"
]

# Mapeamento para símbolos Binance
SYMBOL_MAPPING = {
    "BTC-USD": "BTCUSDT",
    "SOL-USD": "SOLUSDT", 
    "ETH-USD": "ETHUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
    "AVAX-USD": "AVAXUSDT",
    "ENA-USD": "ENAUSDT",
    "BNB-USD": "BNBUSDT",
    "SUI-USD": "SUIUSDT",
    "ADA-USD": "ADAUSDT",
    "PUMP-USD": "PUMPUSDT",
    "AVNT-USD": "AVNTUSDT", 
    "LINK-USD": "LINKUSDT",
    "WLD-USD": "WLDUSDT",
    "AAVE-USD": "AAVEUSDT",
    "CRV-USD": "CRVUSDT",
    "LTC-USD": "LTCUSDT",
    "NEAR-USD": "NEARUSDT"
}

def check_existing_files():
    """Verifica quais arquivos já existem"""
    existing = []
    missing = []
    
    for asset in TRADING_ASSETS:
        symbol = asset.replace("-USD", "").lower()
        filename = f"dados_reais_{symbol}_1ano.csv"
        
        if os.path.exists(filename):
            existing.append(asset)
        else:
            missing.append(asset)
    
    return existing, missing

def download_binance_data(symbol, start_date, end_date):
    """Baixa dados da Binance Vision API"""
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Converter datas para timestamps
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_timestamp
    
    while current_start < end_timestamp:
        try:
            params = {
                'symbol': symbol,
                'interval': '1h',  # 1 hora como nos outros arquivos
                'startTime': current_start,
                'endTime': min(current_start + (1000 * 60 * 60 * 1000), end_timestamp),  # Max 1000 candles
                'limit': 1000
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_start = data[-1][6] + 1  # Próximo timestamp após o último
            
            print(f"   📊 Baixados {len(data)} candles para {symbol}")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"   ❌ Erro baixando {symbol}: {e}")
            time.sleep(1)
            continue
    
    return all_data

def process_binance_data(raw_data, asset):
    """Processa dados da Binance para formato padrão"""
    if not raw_data:
        return None
    
    df_data = []
    
    for candle in raw_data:
        timestamp = datetime.fromtimestamp(candle[0] / 1000)
        
        df_data.append({
            'data': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'valor_fechamento': float(candle[4]),  # Close
            'valor_maximo': float(candle[2]),      # High
            'valor_minimo': float(candle[3]),      # Low
            'valor_abertura': float(candle[1]),    # Open
            'volume': float(candle[5])             # Volume
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset=['timestamp'])
    
    return df

def download_asset_data(asset):
    """Baixa dados de 1 ano para um asset específico"""
    symbol_binance = SYMBOL_MAPPING.get(asset)
    if not symbol_binance:
        print(f"❌ Símbolo não mapeado para {asset}")
        return False
    
    print(f"\n📥 Baixando {asset} ({symbol_binance})...")
    
    # Período de 1 ano
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"   📅 Período: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    
    # Baixar dados
    raw_data = download_binance_data(symbol_binance, start_date, end_date)
    
    if not raw_data:
        print(f"   ❌ Nenhum dado baixado para {asset}")
        return False
    
    # Processar dados
    df = process_binance_data(raw_data, asset)
    
    if df is None or len(df) == 0:
        print(f"   ❌ Erro processando dados para {asset}")
        return False
    
    # Salvar arquivo
    symbol_lower = asset.replace("-USD", "").lower()
    filename = f"dados_reais_{symbol_lower}_1ano.csv"
    
    df.to_csv(filename, index=False)
    
    print(f"   ✅ Salvos {len(df)} candles em {filename}")
    print(f"   📊 Período real: {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]}")
    
    return True

def main():
    print("📥 DOWNLOAD DADOS REAIS 1 ANO - ASSETS TRADING.PY")
    print("="*70)
    
    # Verificar arquivos existentes
    existing, missing = check_existing_files()
    
    print(f"\n📊 STATUS DOS ARQUIVOS:")
    print(f"   ✅ Existentes: {len(existing)} assets")
    print(f"   📥 Faltando: {len(missing)} assets")
    
    if existing:
        print(f"\n✅ ASSETS COM DADOS:")
        for asset in existing:
            print(f"   • {asset}")
    
    if missing:
        print(f"\n📥 ASSETS PARA BAIXAR:")
        for asset in missing:
            print(f"   • {asset}")
            
        print(f"\n🚀 INICIANDO DOWNLOADS...")
        
        successful = 0
        failed = 0
        
        for asset in missing:
            try:
                if download_asset_data(asset):
                    successful += 1
                else:
                    failed += 1
                    
                # Pequena pausa entre downloads
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Erro geral baixando {asset}: {e}")
                failed += 1
        
        print(f"\n" + "="*70)
        print(f"🏆 RESUMO DO DOWNLOAD:")
        print(f"   ✅ Sucessos: {successful}")
        print(f"   ❌ Falhas: {failed}")
        print(f"   📊 Total processado: {successful + failed}")
        
        if successful > 0:
            print(f"\n🎉 Downloads concluídos com sucesso!")
            
            # Re-verificar status final
            existing_final, missing_final = check_existing_files()
            print(f"\n📊 STATUS FINAL:")
            print(f"   ✅ Total com dados: {len(existing_final)}/{len(TRADING_ASSETS)} assets")
            
            if missing_final:
                print(f"   ❌ Ainda faltando: {missing_final}")
            else:
                print(f"   🎊 TODOS OS 18 ASSETS COMPLETOS!")
        
    else:
        print(f"\n🎊 TODOS OS DADOS JÁ ESTÃO DISPONÍVEIS!")
        print(f"   ✅ {len(existing)}/{len(TRADING_ASSETS)} assets completos")

    print("\n🧬 DOWNLOAD CONCLUÍDO!")
    print("="*70)

if __name__ == "__main__":
    main()
