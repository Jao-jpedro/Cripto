#!/usr/bin/env python3
"""
🔍 VERIFICAÇÃO BINANCE - PUMP E AVNT
Verifica se os símbolos PUMP e AVNT existem na Binance e tenta baixar
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json

def check_binance_symbols():
    """Verifica todos os símbolos disponíveis na Binance"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
        
        return symbols
        
    except Exception as e:
        print(f"❌ Erro buscando símbolos: {e}")
        return []

def search_pump_avnt_symbols(all_symbols):
    """Procura variações dos símbolos PUMP e AVNT"""
    pump_variants = []
    avnt_variants = []
    
    for symbol in all_symbols:
        if 'PUMP' in symbol:
            pump_variants.append(symbol)
        if 'AVNT' in symbol or 'AVANT' in symbol:
            avnt_variants.append(symbol)
    
    return pump_variants, avnt_variants

def try_download_symbol(symbol, asset_name):
    """Tenta baixar dados de um símbolo específico"""
    print(f"\n🔍 Testando {symbol} para {asset_name}...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Testar com dados de 7 dias primeiro
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    try:
        params = {
            'symbol': symbol,
            'interval': '1h',
            'startTime': start_timestamp,
            'endTime': end_timestamp,
            'limit': 100
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data and len(data) > 0:
            print(f"   ✅ {symbol} disponível! {len(data)} candles encontrados")
            print(f"   📊 Primeiro candle: {datetime.fromtimestamp(data[0][0]/1000)}")
            print(f"   📊 Último candle: {datetime.fromtimestamp(data[-1][0]/1000)}")
            return True, data
        else:
            print(f"   ❌ {symbol} sem dados")
            return False, None
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print(f"   ❌ {symbol} não existe na Binance")
        else:
            print(f"   ❌ Erro HTTP {e.response.status_code} para {symbol}")
        return False, None
    except Exception as e:
        print(f"   ❌ Erro testando {symbol}: {e}")
        return False, None

def download_full_year_data(symbol, asset_name):
    """Baixa dados completos de 1 ano"""
    print(f"\n📥 Baixando dados de 1 ano para {symbol}...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Período de 1 ano
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_timestamp
    
    while current_start < end_timestamp:
        try:
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': current_start,
                'endTime': min(current_start + (1000 * 60 * 60 * 1000), end_timestamp),
                'limit': 1000
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_start = data[-1][6] + 1
            
            print(f"   📊 Baixados {len(data)} candles...")
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Erro baixando dados: {e}")
            break
    
    if all_data:
        # Processar e salvar
        df_data = []
        
        for candle in all_data:
            timestamp = datetime.fromtimestamp(candle[0] / 1000)
            
            df_data.append({
                'data': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'valor_fechamento': float(candle[4]),
                'valor_maximo': float(candle[2]),
                'valor_minimo': float(candle[3]),
                'valor_abertura': float(candle[1]),
                'volume': float(candle[5])
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Salvar arquivo
        symbol_lower = asset_name.replace("-USD", "").lower()
        filename = f"dados_reais_{symbol_lower}_1ano.csv"
        
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Salvos {len(df)} candles em {filename}")
        print(f"   📊 Período: {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]}")
        
        return True
    else:
        print(f"   ❌ Nenhum dado baixado")
        return False

def main():
    print("🔍 VERIFICAÇÃO BINANCE - PUMP E AVNT")
    print("="*60)
    
    print("\n📡 Buscando todos os símbolos da Binance...")
    all_symbols = check_binance_symbols()
    
    if not all_symbols:
        print("❌ Não foi possível buscar símbolos da Binance")
        return
    
    print(f"✅ Encontrados {len(all_symbols)} símbolos ativos")
    
    # Procurar variações
    print(f"\n🔍 Procurando variações de PUMP e AVNT...")
    pump_variants, avnt_variants = search_pump_avnt_symbols(all_symbols)
    
    print(f"\n📊 PUMP Variants encontradas:")
    if pump_variants:
        for variant in pump_variants:
            print(f"   • {variant}")
    else:
        print("   ❌ Nenhuma variação de PUMP encontrada")
    
    print(f"\n📊 AVNT Variants encontradas:")
    if avnt_variants:
        for variant in avnt_variants:
            print(f"   • {variant}")
    else:
        print("   ❌ Nenhuma variação de AVNT encontrada")
    
    # Testar símbolos específicos
    test_symbols = [
        ('PUMPUSDT', 'PUMP-USD'),
        ('PUMPUSDC', 'PUMP-USD'),
        ('PUMPUSD', 'PUMP-USD'),
        ('AVNTUSDT', 'AVNT-USD'),
        ('AVNTUSDC', 'AVNT-USD'),
        ('AVNTUSD', 'AVNT-USD'),
        ('AVENTUSDT', 'AVNT-USD'),
        ('AVENTUSDC', 'AVNT-USD')
    ]
    
    print(f"\n🧪 TESTANDO SÍMBOLOS ESPECÍFICOS:")
    print("="*50)
    
    successful_downloads = []
    
    for symbol, asset_name in test_symbols:
        success, sample_data = try_download_symbol(symbol, asset_name)
        
        if success:
            # Tentar download completo
            if download_full_year_data(symbol, asset_name):
                successful_downloads.append((symbol, asset_name))
    
    # Testar variações encontradas
    if pump_variants or avnt_variants:
        print(f"\n🧪 TESTANDO VARIAÇÕES ENCONTRADAS:")
        print("="*40)
        
        all_variants = []
        for variant in pump_variants:
            all_variants.append((variant, 'PUMP-USD'))
        for variant in avnt_variants:
            all_variants.append((variant, 'AVNT-USD'))
        
        for symbol, asset_name in all_variants:
            success, sample_data = try_download_symbol(symbol, asset_name)
            
            if success and (symbol, asset_name) not in successful_downloads:
                if download_full_year_data(symbol, asset_name):
                    successful_downloads.append((symbol, asset_name))
    
    print(f"\n" + "="*60)
    print(f"🏆 RESULTADO FINAL:")
    print("="*60)
    
    if successful_downloads:
        print(f"✅ Downloads bem-sucedidos:")
        for symbol, asset_name in successful_downloads:
            print(f"   • {asset_name} → {symbol}")
    else:
        print(f"❌ Nenhum símbolo PUMP ou AVNT disponível na Binance")
        
        print(f"\n💡 POSSÍVEIS RAZÕES:")
        print(f"   • Símbolos não listados na Binance")
        print(f"   • Tokens muito novos")
        print(f"   • Símbolos disponíveis em outras exchanges")
        print(f"   • Nomes diferentes (ex: ADVENTURE → ADVT)")
    
    print(f"\n🧬 VERIFICAÇÃO CONCLUÍDA!")

if __name__ == "__main__":
    main()
