#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE COMPARATIVO: TP 40% vs TP 30%
=====================================

Este script testa se aumentar o Take Profit de 30% para 40% melhora
ou piora a performance do sistema, mantendo todos os outros parâmetros
iguais à configuração vencedora.

CONFIGURAÇÃO BASE (VENCEDORA):
• TP: 30% | SL: 10%
• ATR: 0.5-3.0%
• Volume: 3.0x
• ROI Histórico: +2,190%

TESTE:
• TP: 40% | SL: 10% (mesmo setup)
• Comparação direta de performance
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

class TP40TestEngine:
    """Engine de teste para comparar TP 40% vs TP 30%"""
    
    def __init__(self):
        self.results = {
            'tp30': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0},
            'tp40': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        }
        
        # Configuração base (vencedora)
        self.config = {
            'initial_capital': 10.0,
            'position_size': 1.0,
            'stop_loss_pct': 0.1,  # 10%
            'emergency_stop': -0.05,
            'atr_min_pct': 0.5,
            'atr_max_pct': 3.0,
            'volume_multiplier': 3.0,
            'gradient_min_long': 0.08,
            'gradient_min_short': 0.12,
            'rsi_min': 20,
            'rsi_max': 70,
            'min_confluence': 3
        }
        
        # Assets para teste
        self.assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
        
    def fetch_binance_data(self, symbol, interval='1h', limit=1000):
        """Busca dados históricos da Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Converter para tipos numéricos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Dados {symbol} carregados: {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao buscar {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calcula indicadores técnicos"""
        # EMAs
        df['ema7'] = df['close'].ewm(span=7).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Gradients
        df['grad_ema7'] = ((df['ema7'] - df['ema7'].shift()) / df['ema7'].shift()) * 100
        df['grad_ema21'] = ((df['ema21'] - df['ema21'].shift()) / df['ema21'].shift()) * 100
        
        return df
        
    def check_filters(self, row):
        """Verifica se passa nos filtros restritivos"""
        filters_passed = 0
        
        # 1. ATR saudável (0.5% - 3.0%)
        if self.config['atr_min_pct'] <= row['atr_pct'] <= self.config['atr_max_pct']:
            filters_passed += 1
            
        # 2. Volume acima de 3.0x
        if row['volume_ratio'] >= self.config['volume_multiplier']:
            filters_passed += 1
            
        # 3. RSI não extremo (20-70)
        if self.config['rsi_min'] <= row['rsi'] <= self.config['rsi_max']:
            filters_passed += 1
            
        # 4. Gradient significativo
        if abs(row['grad_ema7']) >= self.config['gradient_min_long']:
            filters_passed += 1
            
        # 5. Breakout das EMAs
        ema_diff = abs(row['ema7'] - row['ema21'])
        if ema_diff >= row['atr']:
            filters_passed += 1
            
        return filters_passed >= self.config['min_confluence']
    
    def simulate_trade(self, df, tp_pct):
        """Simula trades com TP específico"""
        trades = 0
        wins = 0
        losses = 0
        pnl = 0
        
        position = None
        
        for i in range(50, len(df)):  # Skip first 50 for indicators
            row = df.iloc[i]
            
            # Se não tem posição, busca entrada
            if position is None:
                if self.check_filters(row):
                    # Determina direção
                    if row['ema7'] > row['ema21'] and row['grad_ema7'] >= self.config['gradient_min_long']:
                        direction = 'LONG'
                    elif row['ema7'] < row['ema21'] and row['grad_ema7'] <= -self.config['gradient_min_short']:
                        direction = 'SHORT'
                    else:
                        continue
                        
                    # SISTEMA INVERSO: inverte o sinal
                    direction = 'SHORT' if direction == 'LONG' else 'LONG'
                    
                    position = {
                        'direction': direction,
                        'entry_price': row['close'],
                        'entry_index': i
                    }
                    
            # Se tem posição, verifica saída
            else:
                current_price = row['close']
                entry_price = position['entry_price']
                
                if position['direction'] == 'LONG':
                    pct_change = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pct_change = ((entry_price - current_price) / entry_price) * 100
                
                # Verifica TP
                if pct_change >= tp_pct:
                    trades += 1
                    wins += 1
                    pnl += self.config['position_size'] * (tp_pct / 100)
                    position = None
                    
                # Verifica SL
                elif pct_change <= -self.config['stop_loss_pct'] * 100:
                    trades += 1
                    losses += 1
                    pnl -= self.config['position_size'] * (self.config['stop_loss_pct'])
                    position = None
                    
        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'pnl': pnl,
            'win_rate': (wins / trades * 100) if trades > 0 else 0
        }
    
    def run_comparison(self):
        """Executa comparação completa"""
        print("🚀 INICIANDO TESTE COMPARATIVO: TP 40% vs TP 30%")
        print("=" * 60)
        
        total_results = {
            'tp30': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0},
            'tp40': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        }
        
        asset_details = []
        
        for asset in self.assets:
            print(f"\n📊 Testando {asset}...")
            
            # Busca dados
            df = self.fetch_binance_data(asset)
            if df is None or len(df) < 100:
                continue
                
            # Calcula indicadores
            df = self.calculate_indicators(df)
            
            # Testa TP 30%
            result_tp30 = self.simulate_trade(df, 30.0)
            
            # Testa TP 40%
            result_tp40 = self.simulate_trade(df, 40.0)
            
            # Accumula resultados
            for key in ['trades', 'wins', 'losses', 'pnl']:
                total_results['tp30'][key] += result_tp30[key]
                total_results['tp40'][key] += result_tp40[key]
            
            asset_details.append({
                'asset': asset,
                'tp30': result_tp30,
                'tp40': result_tp40
            })
            
            print(f"  TP 30%: {result_tp30['trades']} trades | PnL: ${result_tp30['pnl']:.2f}")
            print(f"  TP 40%: {result_tp40['trades']} trades | PnL: ${result_tp40['pnl']:.2f}")
        
        # Calcula win rates totais
        total_results['tp30']['win_rate'] = (
            total_results['tp30']['wins'] / total_results['tp30']['trades'] * 100
        ) if total_results['tp30']['trades'] > 0 else 0
        
        total_results['tp40']['win_rate'] = (
            total_results['tp40']['wins'] / total_results['tp40']['trades'] * 100
        ) if total_results['tp40']['trades'] > 0 else 0
        
        # Resultados finais
        self.print_results(total_results, asset_details)
        
        # Salva resultados
        self.save_results(total_results, asset_details)
        
    def print_results(self, total_results, asset_details):
        """Exibe resultados formatados"""
        print("\n" + "=" * 60)
        print("🏆 RESULTADOS FINAIS")
        print("=" * 60)
        
        tp30 = total_results['tp30']
        tp40 = total_results['tp40']
        
        print(f"\n📊 TP 30% (BASELINE):")
        print(f"   💰 PnL Total: ${tp30['pnl']:.2f}")
        print(f"   📈 Capital Final: ${self.config['initial_capital'] + tp30['pnl']:.2f}")
        print(f"   🎯 Trades: {tp30['trades']}")
        print(f"   ✅ Win Rate: {tp30['win_rate']:.1f}%")
        print(f"   📊 ROI: {(tp30['pnl'] / self.config['initial_capital']) * 100:.1f}%")
        
        print(f"\n🚀 TP 40% (TESTE):")
        print(f"   💰 PnL Total: ${tp40['pnl']:.2f}")
        print(f"   📈 Capital Final: ${self.config['initial_capital'] + tp40['pnl']:.2f}")
        print(f"   🎯 Trades: {tp40['trades']}")
        print(f"   ✅ Win Rate: {tp40['win_rate']:.1f}%")
        print(f"   📊 ROI: {(tp40['pnl'] / self.config['initial_capital']) * 100:.1f}%")
        
        # Comparação
        pnl_diff = tp40['pnl'] - tp30['pnl']
        roi_diff = ((tp40['pnl'] / self.config['initial_capital']) - 
                   (tp30['pnl'] / self.config['initial_capital'])) * 100
        
        print(f"\n🔍 COMPARAÇÃO:")
        print(f"   💰 Diferença PnL: ${pnl_diff:.2f}")
        print(f"   📊 Diferença ROI: {roi_diff:.1f}%")
        
        if pnl_diff > 0:
            print(f"   🏆 VENCEDOR: TP 40% (+${pnl_diff:.2f})")
        elif pnl_diff < 0:
            print(f"   🏆 VENCEDOR: TP 30% (+${abs(pnl_diff):.2f})")
        else:
            print(f"   🤝 EMPATE")
            
        print("\n📋 DETALHES POR ATIVO:")
        for detail in asset_details:
            asset = detail['asset']
            tp30_pnl = detail['tp30']['pnl']
            tp40_pnl = detail['tp40']['pnl']
            diff = tp40_pnl - tp30_pnl
            
            winner = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            print(f"   {winner} {asset}: TP30=${tp30_pnl:.2f} | TP40=${tp40_pnl:.2f} | Diff=${diff:.2f}")
    
    def save_results(self, total_results, asset_details):
        """Salva resultados em JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"teste_tp40_vs_tp30_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'test_description': 'Comparação TP 40% vs TP 30% mantendo outros parâmetros',
            'config': self.config,
            'total_results': total_results,
            'asset_details': asset_details,
            'conclusion': {
                'tp30_pnl': total_results['tp30']['pnl'],
                'tp40_pnl': total_results['tp40']['pnl'],
                'difference': total_results['tp40']['pnl'] - total_results['tp30']['pnl'],
                'better_strategy': 'TP40' if total_results['tp40']['pnl'] > total_results['tp30']['pnl'] else 'TP30',
                'roi_tp30': (total_results['tp30']['pnl'] / self.config['initial_capital']) * 100,
                'roi_tp40': (total_results['tp40']['pnl'] / self.config['initial_capital']) * 100
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n💾 Resultados salvos em: {filename}")

def main():
    """Função principal"""
    print("🧪 TESTE DE OTIMIZAÇÃO: TP 40% vs TP 30%")
    print("Baseado na configuração vencedora de 2190% ROI")
    print("=" * 60)
    
    engine = TP40TestEngine()
    engine.run_comparison()
    
    print("\n✅ Teste concluído!")

if __name__ == "__main__":
    main()
