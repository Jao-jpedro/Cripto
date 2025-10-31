#!/usr/bin/env python3
"""
Script para extrair dados históricos de 1 ano dos assets do TradingV4
com todas as métricas utilizadas no sistema de trading.

Inclui:
- Dados OHLCV básicos
- EMAs (7 e 21)
- ATR e ATR%
- Volume e Volume MA
- Gradientes das EMAs
- current_k_atr (distância do preço à EMA7 em unidades de ATR)
- Candles consecutivos positivos/negativos
- RSI e MACD
- Todos os indicadores técnicos do TradingV4
"""

import pandas as pd
import numpy as np
import requests
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

# Configurações independentes (sem importar tradingv4.py para evitar conflitos)
@dataclass
class AssetSetup:
    name: str
    data_symbol: str
    hl_symbol: str
    leverage: int
    stop_pct: float = 0.20
    take_pct: float = 0.50
    usd_env: Optional[str] = None

ASSET_SETUPS: List[AssetSetup] = [
    AssetSetup("AVNT-USD", "AVNTUSDT", "AVNT/USDC:USDC", 5, usd_env="USD_PER_TRADE_AVNT"),
    AssetSetup("ZEC-USD", "ZECUSDT", "ZEC/USDC:USDC", 5, usd_env="USD_PER_TRADE_ZEC"),
    AssetSetup("ASTER-USD", "ASTERUSDT", "ASTER/USDC:USDC", 5, usd_env="USD_PER_TRADE_ASTER"),
    AssetSetup("ETH-USD", "ETHUSDT", "ETH/USDC:USDC", 25, usd_env="USD_PER_TRADE_ETH"),
    AssetSetup("TAO-USD", "TAOUSDT", "TAO/USDC:USDC", 5, usd_env="USD_PER_TRADE_TAO"),
    AssetSetup("XRP-USD", "XRPUSDT", "XRP/USDC:USDC", 20, usd_env="USD_PER_TRADE_XRP"),
    AssetSetup("DOGE-USD", "DOGEUSDT", "DOGE/USDC:USDC", 10, usd_env="USD_PER_TRADE_DOGE"),
    AssetSetup("AVAX-USD", "AVAXUSDT", "AVAX/USDC:USDC", 10, usd_env="USD_PER_TRADE_AVAX"),
    AssetSetup("ENA-USD", "ENAUSDT", "ENA/USDC:USDC", 10, usd_env="USD_PER_TRADE_ENA"),
    AssetSetup("BNB-USD", "BNBUSDT", "BNB/USDC:USDC", 10, usd_env="USD_PER_TRADE_BNB"),
    AssetSetup("SUI-USD", "SUIUSDT", "SUI/USDC:USDC", 10, usd_env="USD_PER_TRADE_SUI"),
    AssetSetup("ADA-USD", "ADAUSDT", "ADA/USDC:USDC", 10, usd_env="USD_PER_TRADE_ADA"),
    AssetSetup("PUMP-USD", "PUMPUSDT", "PUMP/USDC:USDC", 10, usd_env="USD_PER_TRADE_PUMP"),
    AssetSetup("PAXG-USD", "PAXGUSDT", "PAXG/USDC:USDC", 10, usd_env="USD_PER_TRADE_PAXG"),
    AssetSetup("LINK-USD", "LINKUSDT", "LINK/USDC:USDC", 10, usd_env="USD_PER_TRADE_LINK"),
    AssetSetup("WLD-USD", "WLDUSDT", "WLD/USDC:USDC", 10, usd_env="USD_PER_TRADE_WLD"),
    AssetSetup("AAVE-USD", "AAVEUSDT", "AAVE/USDC:USDC", 10, usd_env="USD_PER_TRADE_AAVE"),
    AssetSetup("CRV-USD", "CRVUSDT", "CRV/USDC:USDC", 10, usd_env="USD_PER_TRADE_CRV"),
    AssetSetup("LTC-USD", "LTCUSDT", "LTC/USDC:USDC", 10, usd_env="USD_PER_TRADE_LTC"),
    AssetSetup("NEAR-USD", "NEARUSDT", "NEAR/USDC:USDC", 10, usd_env="USD_PER_TRADE_NEAR"),
]

# Configurações
UTC = timezone.utc
BASE_URL = "https://api.binance.com/api/v3/"
TIMEFRAME = "15m"  # Mesmo timeframe do TradingV4
LOOKBACK_DAYS = 365  # 1 ano de dados

def get_binance_data_safe(symbol, interval='15m', limit=1000):
    """Baixa dados históricos da Binance com tratamento de erros aprimorado"""
    try:
        print(f"🔄 Baixando dados para {symbol}...")
        
        # Teste simples primeiro - verificar se o símbolo existe
        test_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        test_response = requests.get(test_url, timeout=10)
        
        if test_response.status_code != 200:
            print(f"❌ Símbolo {symbol} não encontrado na Binance")
            return pd.DataFrame()
        
        # URL da API Binance para dados históricos (klines)
        url = "https://api.binance.com/api/v3/klines"
        
        # Parâmetros da requisição
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        print(f"🔗 URL: {url}")
        print(f"📋 Parâmetros: {params}")
        
        # Fazer a requisição com timeout
        response = requests.get(url, params=params, timeout=30)
        print(f"📈 Status Code: {response.status_code}")
        print(f"📄 Response text (primeiros 200 chars): {response.text[:200]}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"📊 Dados recebidos: {len(data)} registros")
                
                if data and len(data) > 0:
                    # Converter para DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'volume', 'close_time', 'quote_asset_volume', 
                        'number_of_trades', 'taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Converter timestamp para datetime
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Converter preços e volume para float
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    print(f"✅ DataFrame criado com {len(df)} linhas")
                    print(f"📅 Período: {df['datetime'].min()} a {df['datetime'].max()}")
                    return df
                else:
                    print(f"⚠️ Dados vazios para {symbol}")
                    return pd.DataFrame()
            except ValueError as ve:
                print(f"❌ Erro JSON para {symbol}: {ve}")
                return pd.DataFrame()
        else:
            print(f"❌ Erro HTTP {response.status_code}: {response.text}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Erro ao baixar dados para {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    """Versão segura da função get_binance_data com retry e tratamento de erro"""
    print(f"   🔍 Buscando dados para {symbol} de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    all_data = []
    current_start = start_timestamp
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PythonRequests/2.x",
        "Accept": "application/json",
    })
    
    while current_start < end_timestamp:
        url = f"{BASE_URL}klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_timestamp,
            "limit": 1000
        }
        
        print(f"   📡 Requisição: {url} com {params['symbol']}")
        
        for attempt in range(3):  # 3 tentativas
            try:
                response = session.get(url, params=params, timeout=15)
                print(f"   📊 Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        print(f"   ⚠️ Lista vazia retornada para {symbol}")
                        return []
                    
                    print(f"   ✅ {len(data)} candles recebidos")
                    all_data.extend(data)
                    current_start = int(data[-1][0]) + 1
                    break
                elif response.status_code == 400:
                    print(f"   ❌ Erro 400 - Símbolo {symbol} pode não existir na Binance")
                    return []
                else:
                    print(f"   ❌ Erro {response.status_code} para {symbol}: {response.text[:200]}")
                    if attempt == 2:
                        return []
                    time.sleep(1 * (attempt + 1))
            except Exception as e:
                print(f"   ❌ Erro na tentativa {attempt+1} para {symbol}: {e}")
                if attempt == 2:
                    return []
                time.sleep(1 * (attempt + 1))
        
        time.sleep(0.2)  # Rate limiting mais conservador
    
    if not all_data:
        print(f"   ❌ Nenhum dado coletado para {symbol}")
        return []
    
    print(f"   ✅ Total: {len(all_data)} candles coletados para {symbol}")
    
    formatted_data = [{
        "data": item[0],
        "valor_abertura": round(float(item[1]), 8),
        "valor_maximo": round(float(item[2]), 8),
        "valor_minimo": round(float(item[3]), 8),
        "valor_fechamento": round(float(item[4]), 8),
        "criptomoeda": symbol,
        "volume_compra": float(item[5]),
        "volume_venda": float(item[7]),
        "numero_trades": int(item[8]),
        "volume_ativo_compra": float(item[9]),
        "volume_ativo_venda": float(item[10])
    } for item in all_data]
    
    return formatted_data

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calcula EMA com span específico"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula ATR (Average True Range)"""
    high = df['valor_maximo']
    low = df['valor_minimo'] 
    close = df['valor_fechamento']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_gradient(series: pd.Series, window: int = 3) -> pd.Series:
    """Calcula gradiente (taxa de mudança) percentual"""
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    gradients = []
    for i in range(len(series)):
        if i < window - 1:
            gradients.append(np.nan)
        else:
            # Calcular inclinação dos últimos 'window' pontos
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(len(y))
            if len(y) > 1:
                # Regressão linear simples
                slope = np.polyfit(x, y, 1)[0]
                # Converter para percentual baseado no valor atual
                if series.iloc[i] != 0:
                    gradient_pct = (slope / series.iloc[i]) * 100
                else:
                    gradient_pct = 0
                gradients.append(gradient_pct)
            else:
                gradients.append(np.nan)
    
    return pd.Series(gradients, index=series.index)

def calculate_consecutive_candles(df: pd.DataFrame) -> pd.Series:
    """Calcula quantos candles consecutivos são positivos ou negativos"""
    close = df['valor_fechamento']
    open_price = df['valor_abertura']
    
    # 1 para candle positivo (close > open), -1 para negativo
    candle_direction = np.where(close > open_price, 1, -1)
    
    consecutive = []
    current_count = 0
    current_direction = 0
    
    for direction in candle_direction:
        if direction == current_direction:
            current_count += 1 if direction == 1 else -1
        else:
            current_direction = direction
            current_count = 1 if direction == 1 else -1
        consecutive.append(current_count)
    
    return pd.Series(consecutive, index=df.index)

def calculate_rsi_simple(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calcula RSI simples"""
    close = df['valor_fechamento']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd_simple(df: pd.DataFrame, short_window: int = 7, long_window: int = 40, signal_window: int = 9):
    """Calcula MACD simples"""
    close = df['valor_fechamento']
    
    ema_short = calculate_ema(close, short_window)
    ema_long = calculate_ema(close, long_window)
    
    macd_line = ema_short - ema_long
    macd_signal = calculate_ema(macd_line, signal_window)
    macd_histogram = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_histogram

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona todos os indicadores técnicos usados no TradingV4"""
    print("   📊 Calculando indicadores técnicos...")
    
    # Mapear colunas do Binance para o formato esperado pelo TradingV4
    df = df.rename(columns={
        'close': 'valor_fechamento',
        'high': 'valor_maximo', 
        'low': 'valor_minimo',
        'open': 'valor_abertura',
        'volume': 'volume_compra'
    })
    
    # EMAs
    df['ema7'] = calculate_ema(df['valor_fechamento'], 7)
    df['ema21'] = calculate_ema(df['valor_fechamento'], 21)
    
    # ATR e ATR%
    df['atr'] = calculate_atr(df, period=14)
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    # Volume MA
    df['vol_ma'] = df['volume_compra'].rolling(window=20).mean()
    
    # Gradientes das EMAs (taxa de mudança percentual)
    df['grad_pct_ema7'] = calculate_gradient(df['ema7'], window=3)
    df['grad_pct_ema21'] = calculate_gradient(df['ema21'], window=3)
    
    # current_k_atr: distância do preço à EMA7 em unidades de ATR
    df['current_k_atr'] = abs(df['valor_fechamento'] - df['ema7']) / df['atr']
    
    # Candles consecutivos
    df['consecutive_candles'] = calculate_consecutive_candles(df)
    
    # RSI
    df['rsi'] = calculate_rsi_simple(df, window=14)
    
    # MACD
    macd_line, macd_signal, macd_histogram = calculate_macd_simple(df)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_histogram
    
    # Indicativo MACD
    df['macd_indicativo'] = np.where(df['macd'] > df['macd_signal'], 'Alta', 
                                   np.where(df['macd'] < df['macd_signal'], 'Baixa', 'Neutro'))
    
    # Breakout thresholds (bandas de rompimento)
    breakout_k = 3.0  # Valor atual do TradingV4
    df['breakout_upper'] = df['ema7'] + (breakout_k * df['atr'])
    df['breakout_lower'] = df['ema7'] - (breakout_k * df['atr'])
    
    # Distância percentual das bandas
    df['dist_breakout_upper_pct'] = ((df['breakout_upper'] - df['valor_fechamento']) / df['valor_fechamento']) * 100
    df['dist_breakout_lower_pct'] = ((df['valor_fechamento'] - df['breakout_lower']) / df['valor_fechamento']) * 100
    
    # EMA spread (diferença entre EMAs)
    df['ema_spread'] = df['ema7'] - df['ema21']
    df['ema_spread_pct'] = (df['ema_spread'] / df['valor_fechamento']) * 100
    
    # Volume ratio
    df['vol_ratio'] = df['volume_compra'] / df['vol_ma']
    
    # High/Low ranges
    df['hl_range'] = df['valor_maximo'] - df['valor_minimo']
    df['hl_range_pct'] = (df['hl_range'] / df['valor_fechamento']) * 100
    
    print("   ✅ Indicadores calculados com sucesso!")
    return df

def download_asset_data(asset: AssetSetup) -> Optional[pd.DataFrame]:
    """Baixa dados históricos de um asset específico"""
    print(f"\n📥 Baixando dados para {asset.name} ({asset.data_symbol})...")
    
    # Calcular período de 1 ano
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    print(f"   📅 Período: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # Baixar dados
        raw_data = get_binance_data_safe(asset.data_symbol, TIMEFRAME, 1000)
        
        if raw_data.empty:
            print(f"   ❌ Falha ao baixar dados para {asset.name}")
            return None
        
        # Converter para DataFrame
        df = raw_data.copy()  # raw_data já é um DataFrame processado
        
        print(f"   📊 {len(df)} candles baixados")
        
        # Adicionar metadados do asset
        df['asset_name'] = asset.name
        df['asset_leverage'] = asset.leverage
        df['hl_symbol'] = asset.hl_symbol
        
        # Adicionar todos os indicadores
        df = add_all_indicators(df)
        
        print(f"   ✅ Dados processados para {asset.name}")
        return df
        
    except Exception as e:
        print(f"   ❌ Erro ao processar {asset.name}: {e}")
        return None

def main():
    """Função principal"""
    print("🚀 Iniciando extração de dados históricos do TradingV4")
    print(f"📋 {len(ASSET_SETUPS)} assets configurados")
    print(f"⏰ Timeframe: {TIMEFRAME}")
    print(f"📅 Período: {LOOKBACK_DAYS} dias")
    
    all_data = []
    successful_downloads = 0
    failed_downloads = 0
    
    for i, asset in enumerate(ASSET_SETUPS, 1):
        print(f"\n[{i}/{len(ASSET_SETUPS)}] Processando {asset.name}...")
        
        df = download_asset_data(asset)
        
        if df is not None:
            all_data.append(df)
            successful_downloads += 1
            print(f"   ✅ Sucesso! {len(df)} registros")
        else:
            failed_downloads += 1
            print(f"   ❌ Falha!")
        
        # Pequena pausa para evitar rate limiting
        time.sleep(0.5)
    
    # Consolidar todos os dados
    if all_data:
        print(f"\n📊 Consolidando dados de {successful_downloads} assets...")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['asset_name', 'datetime']).reset_index(drop=True)
        
        # Salvar em arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tradingv4_historical_data_{timestamp}.csv"
        
        print(f"💾 Salvando em {filename}...")
        combined_df.to_csv(filename, index=False)
        
        # Estatísticas finais
        print(f"\n📈 RELATÓRIO FINAL:")
        print(f"✅ Assets processados com sucesso: {successful_downloads}")
        print(f"❌ Assets com falha: {failed_downloads}")
        print(f"📊 Total de registros: {len(combined_df):,}")
        print(f"📁 Arquivo salvo: {filename}")
        print(f"💾 Tamanho do arquivo: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")
        
        # Mostrar preview dos dados
        print(f"\n🔍 PREVIEW DOS DADOS:")
        print("Colunas disponíveis:")
        for col in combined_df.columns:
            print(f"  • {col}")
        
        print(f"\nPrimeiros registros:")
        print(combined_df.head(3).to_string())
        
        print(f"\nEstatísticas por asset:")
        summary = combined_df.groupby('asset_name').agg({
            'data': ['min', 'max', 'count'],
            'valor_fechamento': ['min', 'max', 'mean'],
            'atr_pct': 'mean',
            'current_k_atr': 'mean'
        }).round(4)
        print(summary)
        
    else:
        print("\n❌ Nenhum dado foi baixado com sucesso!")
        
    print(f"\n🏁 Extração concluída!")

if __name__ == "__main__":
    main()
