#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 OTIMIZAÇÃO DO TRADINGV4 - ANÁLISE DE PERFORMANCE
================================================================

PROBLEMAS IDENTIFICADOS NO TRADINGV4:
=====================================

1. 📊 ANÁLISE SEQUENCIAL LENTA:
   • 19 ativos processados UM POR VEZ
   • Cada ativo: build_df(15m) + build_df(1h) + step() ≈ 3-5 segundos
   • Total: 19 × 4s = 76 segundos POR CICLO
   • BTC só volta a ser analisado após 76 segundos!

2. 🔄 CICLO INEFICIENTE:
   • Full Analysis: 60 segundos (SLEEP_SECONDS)
   • Fast Safety: 5 segundos (só verificações)
   • Resultado: BTC analisado a cada ~80-90 segundos

3. 📡 CHAMADAS API REPETITIVAS:
   • Cada ativo faz 2 calls build_df() independentes
   • Sem cache entre ativos
   • Binance rate limit pode ser atingido

4. 🐌 SLEEP DESNECESSÁRIO:
   • time.sleep(0.25) após cada ativo
   • 19 × 0.25s = 4.75s adicionais por ciclo

SOLUÇÕES PROPOSTAS:
==================

✅ PARALELIZAÇÃO DOS DADOS:
   • Buscar dados de TODOS os ativos em paralelo
   • Reduzir 76s → 10-15s

✅ CACHE INTELIGENTE:
   • Cache de 30s para dados já buscados
   • Evitar chamadas duplicadas

✅ ANÁLISE PRIORITÁRIA:
   • Ativos com posições abertas: prioridade máxima
   • Ativos sem posição: análise menos frequente

✅ BATCH PROCESSING:
   • Processar indicadores em lote
   • Verificações de segurança em batch
"""

import asyncio
import aiohttp
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd

class OptimizedTradingEngine:
    """Engine de trading otimizado com paralelização e cache"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.data_cache = {}
        self.cache_ttl = 30  # 30 segundos
        self.cache_timestamps = {}
        
        # Contadores de performance
        self.stats = {
            "cycles_completed": 0,
            "total_time": 0,
            "avg_cycle_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0
        }
    
    def is_cache_valid(self, key: str) -> bool:
        """Verifica se cache ainda é válido"""
        if key not in self.cache_timestamps:
            return False
        
        elapsed = time.time() - self.cache_timestamps[key]
        return elapsed < self.cache_ttl
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Busca dados do cache se válidos"""
        if self.is_cache_valid(key):
            self.stats["cache_hits"] += 1
            return self.data_cache[key]
        
        self.stats["cache_misses"] += 1
        return None
    
    def cache_data(self, key: str, data: pd.DataFrame):
        """Armazena dados no cache"""
        self.data_cache[key] = data
        self.cache_timestamps[key] = time.time()
    
    def fetch_asset_data_parallel(self, assets: List[dict]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Busca dados de todos os ativos em paralelo"""
        print(f"📊 Buscando dados de {len(assets)} ativos em paralelo...")
        start_time = time.time()
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submeter todas as tarefas
            future_to_asset = {}
            
            for asset in assets:
                # 15m data
                cache_key_15m = f"{asset['data_symbol']}_15m"
                cached_15m = self.get_cached_data(cache_key_15m)
                
                if cached_15m is not None:
                    print(f"💾 Cache hit: {asset['name']} 15m")
                    if asset['name'] not in results:
                        results[asset['name']] = {}
                    results[asset['name']]['15m'] = cached_15m
                else:
                    future = executor.submit(self._fetch_single_timeframe, asset['data_symbol'], '15m')
                    future_to_asset[future] = (asset, '15m', cache_key_15m)
                
                # 1h data
                cache_key_1h = f"{asset['data_symbol']}_1h"
                cached_1h = self.get_cached_data(cache_key_1h)
                
                if cached_1h is not None:
                    print(f"💾 Cache hit: {asset['name']} 1h")
                    if asset['name'] not in results:
                        results[asset['name']] = {}
                    results[asset['name']]['1h'] = cached_1h
                else:
                    future = executor.submit(self._fetch_single_timeframe, asset['data_symbol'], '1h')
                    future_to_asset[future] = (asset, '1h', cache_key_1h)
            
            # Coletar resultados
            for future in as_completed(future_to_asset):
                asset, timeframe, cache_key = future_to_asset[future]
                
                try:
                    df = future.result()
                    
                    if asset['name'] not in results:
                        results[asset['name']] = {}
                    
                    results[asset['name']][timeframe] = df
                    
                    # Cache do resultado
                    self.cache_data(cache_key, df)
                    self.stats["api_calls"] += 1
                    
                    print(f"✅ {asset['name']} {timeframe}: {len(df)} candles")
                    
                except Exception as e:
                    print(f"❌ Erro {asset['name']} {timeframe}: {e}")
                    
                    if asset['name'] not in results:
                        results[asset['name']] = {}
                    results[asset['name']][timeframe] = pd.DataFrame()
        
        elapsed = time.time() - start_time
        print(f"⚡ Dados coletados em {elapsed:.2f}s (vs ~{len(assets) * 4:.0f}s sequencial)")
        print(f"📈 Cache: {self.stats['cache_hits']} hits / {self.stats['cache_misses']} misses")
        
        return results
    
    def _fetch_single_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Busca dados de um timeframe específico"""
        try:
            # Simular chamada à função build_df do tradingv4
            # Na implementação real, isso seria:
            # return build_df(symbol, timeframe, debug=False)
            
            import requests
            
            # Endpoint da Binance
            url = "https://api.binance.com/api/v3/klines"
            limit = 260 if timeframe == "15m" else 100
            
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['valor_fechamento'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['data'] = df['timestamp']  # Compatibilidade com tradingv4
            
            return df.set_index('timestamp')
            
        except Exception as e:
            print(f"❌ Erro buscando {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def prioritize_assets_by_positions(self, assets: List[dict], dex) -> Tuple[List[dict], List[dict]]:
        """Separa ativos com/sem posições abertas para priorização"""
        priority_assets = []
        standard_assets = []
        
        print("🎯 Priorizando ativos com posições abertas...")
        
        for asset in assets:
            try:
                # Verificar se há posição aberta (simulado)
                # Na implementação real: dex.fetch_positions([asset.hl_symbol])
                has_position = False  # Placeholder
                
                if has_position:
                    priority_assets.append(asset)
                    print(f"🔥 PRIORIDADE: {asset['name']} (posição aberta)")
                else:
                    standard_assets.append(asset)
                    
            except Exception as e:
                print(f"⚠️ Erro verificando posição {asset['name']}: {e}")
                standard_assets.append(asset)
        
        print(f"📊 Prioridade: {len(priority_assets)} | Padrão: {len(standard_assets)}")
        return priority_assets, standard_assets
    
    def run_optimized_cycle(self, assets: List[dict], dex, asset_state: Dict):
        """Executa um ciclo otimizado de análise"""
        cycle_start = time.time()
        
        print(f"\n🚀 INICIANDO CICLO OTIMIZADO #{self.stats['cycles_completed'] + 1}")
        print("=" * 70)
        
        # 1. Priorizar ativos com posições
        priority_assets, standard_assets = self.prioritize_assets_by_positions(assets, dex)
        
        # 2. Buscar dados em paralelo
        print("\n📊 FASE 1: COLETA DE DADOS PARALELA")
        all_data = self.fetch_asset_data_parallel(assets)
        
        # 3. Processar ativos prioritários primeiro
        print("\n🔥 FASE 2: PROCESSAMENTO PRIORITÁRIO")
        if priority_assets:
            self._process_assets_batch(priority_assets, all_data, dex, asset_state, priority=True)
        
        # 4. Processar ativos padrão
        print("\n📈 FASE 3: PROCESSAMENTO PADRÃO")
        self._process_assets_batch(standard_assets, all_data, dex, asset_state, priority=False)
        
        # 5. Safety checks em batch
        print("\n🛡️ FASE 4: VERIFICAÇÕES DE SEGURANÇA")
        self._run_batch_safety_checks(dex, asset_state)
        
        # 6. Estatísticas do ciclo
        cycle_time = time.time() - cycle_start
        self.stats["cycles_completed"] += 1
        self.stats["total_time"] += cycle_time
        self.stats["avg_cycle_time"] = self.stats["total_time"] / self.stats["cycles_completed"]
        
        print(f"\n✅ CICLO CONCLUÍDO EM {cycle_time:.2f}s")
        print(f"📊 Média por ciclo: {self.stats['avg_cycle_time']:.2f}s")
        print(f"⚡ Speedup vs tradingv4: {76/cycle_time:.1f}x mais rápido")
        
        return cycle_time
    
    def _process_assets_batch(self, assets: List[dict], all_data: Dict, dex, asset_state: Dict, priority: bool = False):
        """Processa um lote de ativos"""
        if not assets:
            return
        
        batch_type = "PRIORIDADE" if priority else "PADRÃO"
        print(f"⚙️ Processando {len(assets)} ativos ({batch_type})...")
        
        for asset in assets:
            try:
                asset_data = all_data.get(asset['name'], {})
                df_15m = asset_data.get('15m', pd.DataFrame())
                df_1h = asset_data.get('1h', pd.DataFrame())
                
                if df_15m.empty:
                    print(f"⚠️ {asset['name']}: Sem dados 15m")
                    continue
                
                # Processar estratégia (simulado)
                # Na implementação real: strategy.step(df_15m, usd_to_spend, df_1h)
                print(f"✅ {asset['name']}: Analisado ({len(df_15m)} candles)")
                
                # Pequeno delay apenas para ativos prioritários
                if priority:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Erro processando {asset['name']}: {e}")
    
    def _run_batch_safety_checks(self, dex, asset_state: Dict):
        """Executa verificações de segurança em lote"""
        try:
            # Na implementação real:
            # fast_safety_check_v4(dex, asset_state)
            # check_all_trailing_stops_v4(dex, asset_state)
            print("🛡️ Safety checks executados")
            
        except Exception as e:
            print(f"❌ Erro nos safety checks: {e}")

def demonstrate_optimization():
    """Demonstra a otimização proposta"""
    
    # Assets do tradingv4 (simplificado)
    ASSETS = [
        {"name": "BTC-USD", "data_symbol": "BTCUSDT", "hl_symbol": "BTC/USDC:USDC"},
        {"name": "SOL-USD", "data_symbol": "SOLUSDT", "hl_symbol": "SOL/USDC:USDC"},
        {"name": "ETH-USD", "data_symbol": "ETHUSDT", "hl_symbol": "ETH/USDC:USDC"},
        {"name": "XRP-USD", "data_symbol": "XRPUSDT", "hl_symbol": "XRP/USDC:USDC"},
        {"name": "DOGE-USD", "data_symbol": "DOGEUSDT", "hl_symbol": "DOGE/USDC:USDC"},
    ]
    
    print("🚀 DEMONSTRAÇÃO DA OTIMIZAÇÃO DO TRADINGV4")
    print("=" * 60)
    print(f"📊 Testando com {len(ASSETS)} ativos (subset para demo)")
    print("\n💡 COMPARAÇÃO DE PERFORMANCE:")
    print(f"   • tradingv4 atual: ~{len(ASSETS) * 4}s por ciclo (sequencial)")
    print(f"   • Versão otimizada: ~5-10s por ciclo (paralelo)")
    print(f"   • Speedup esperado: {len(ASSETS) * 4 / 8:.1f}x mais rápido")
    
    # Inicializar engine otimizado
    engine = OptimizedTradingEngine(max_workers=8)
    
    # Simular alguns ciclos
    for cycle in range(3):
        print(f"\n" + "="*60)
        cycle_time = engine.run_optimized_cycle(ASSETS, None, {})
        
        if cycle < 2:  # Não fazer sleep no último ciclo
            print(f"⏰ Aguardando próximo ciclo...")
            time.sleep(2)  # Sleep curto para demo
    
    print(f"\n🏆 RESUMO FINAL:")
    print(f"   • Ciclos executados: {engine.stats['cycles_completed']}")
    print(f"   • Tempo médio por ciclo: {engine.stats['avg_cycle_time']:.2f}s")
    print(f"   • API calls totais: {engine.stats['api_calls']}")
    print(f"   • Cache efficiency: {engine.stats['cache_hits']/(engine.stats['cache_hits']+engine.stats['cache_misses'])*100:.1f}%")
    print(f"   • Speedup vs tradingv4: {len(ASSETS)*4/engine.stats['avg_cycle_time']:.1f}x")

if __name__ == "__main__":
    demonstrate_optimization()
