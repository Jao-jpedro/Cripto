#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE EXTENSIVO: TP 40% vs TP 30%
===================================

Teste mais robusto com:
• Mais ativos (10 ativos como na simulação original)
• Mais dados históricos (6 meses)
• Intervalos 4h para melhor representatividade
• Análise estatística mais detalhada
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

class ExtensiveTP40Test:
    """Teste extensivo TP 40% vs TP 30%"""
    
    def __init__(self):
        # Configuração idêntica à simulação vencedora
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
        
        # Mesmos 10 ativos da simulação vencedora
        self.assets = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'LINKUSDT',
            'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'LTCUSDT'
        ]
        
        self.results = {
            'tp30': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'asset_details': {}},
            'tp40': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'asset_details': {}}
        }
        
    def fetch_extended_data(self, symbol, interval='4h', months=6):
        """Busca dados históricos extensivos (6 meses)"""
        try:
            print(f"📡 Buscando {symbol} ({interval}, {months} meses)...")
            
            # Calcula timestamp de 6 meses atrás
            end_time = datetime.now()
            start_time = end_time - timedelta(days=months * 30)
            start_timestamp = int(start_time.timestamp() * 1000)
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_timestamp,
                'limit': 1500  # Máximo da Binance
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                print(f"❌ Erro HTTP {response.status_code} para {symbol}")
                return None
                
            data = response.json()
            
            if not data:
                print(f"❌ Sem dados para {symbol}")
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Conversões
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            print(f"✅ {symbol}: {len(df)} candles de {df['timestamp'].min()} a {df['timestamp'].max()}")
            time.sleep(0.1)  # Rate limit
            return df
            
        except Exception as e:
            print(f"❌ Erro ao buscar {symbol}: {e}")
            return None
    
    def calculate_advanced_indicators(self, df):
        """Calcula todos os indicadores necessários"""
        # EMAs
        df['ema7'] = df['close'].ewm(span=7).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
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
        df['grad_ema7'] = ((df['ema7'] - df['ema7'].shift(1)) / df['ema7'].shift(1)) * 100
        df['grad_ema21'] = ((df['ema21'] - df['ema21'].shift(1)) / df['ema21'].shift(1)) * 100
        
        # Remove NaN
        df = df.dropna()
        
        return df
        
    def check_confluence_filters(self, row):
        """Verifica filtros de confluência (mesmo critério da simulação vencedora)"""
        filters = []
        
        # 1. ATR saudável (0.5% - 3.0%)
        atr_ok = self.config['atr_min_pct'] <= row['atr_pct'] <= self.config['atr_max_pct']
        filters.append(atr_ok)
        
        # 2. Volume acima de 3.0x
        vol_ok = row['volume_ratio'] >= self.config['volume_multiplier']
        filters.append(vol_ok)
        
        # 3. RSI não extremo (20-70)
        rsi_ok = self.config['rsi_min'] <= row['rsi'] <= self.config['rsi_max']
        filters.append(rsi_ok)
        
        # 4. Gradient EMA7 significativo
        grad_ok = abs(row['grad_ema7']) >= self.config['gradient_min_long']
        filters.append(grad_ok)
        
        # 5. Breakout das EMAs (separação > 1 ATR)
        ema_separation = abs(row['ema7'] - row['ema21'])
        breakout_ok = ema_separation >= row['atr']
        filters.append(breakout_ok)
        
        # Conta quantos filtros passaram
        filters_passed = sum(filters)
        
        return filters_passed >= self.config['min_confluence'], filters_passed
    
    def simulate_extended_trading(self, df, tp_pct, asset):
        """Simulação de trading com estatísticas detalhadas"""
        trades = []
        position = None
        
        print(f"  🔄 Simulando {asset} com TP {tp_pct}%...")
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            # Se não tem posição, busca entrada
            if position is None:
                confluence_passed, filters_count = self.check_confluence_filters(row)
                
                if confluence_passed:
                    # Determina direção original
                    if (row['ema7'] > row['ema21'] and 
                        row['grad_ema7'] >= self.config['gradient_min_long']):
                        original_direction = 'LONG'
                    elif (row['ema7'] < row['ema21'] and 
                          row['grad_ema7'] <= -self.config['gradient_min_short']):
                        original_direction = 'SHORT'
                    else:
                        continue
                        
                    # SISTEMA INVERSO: inverte sinal
                    direction = 'SHORT' if original_direction == 'LONG' else 'LONG'
                    
                    position = {
                        'direction': direction,
                        'entry_price': row['close'],
                        'entry_time': row['timestamp'],
                        'entry_index': i,
                        'filters_count': filters_count
                    }
                    
            # Se tem posição, verifica saída
            else:
                current_price = row['close']
                entry_price = position['entry_price']
                
                if position['direction'] == 'LONG':
                    pct_change = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pct_change = ((entry_price - current_price) / entry_price) * 100
                
                trade_closed = False
                
                # Verifica TP
                if pct_change >= tp_pct:
                    trade_result = {
                        'asset': asset,
                        'direction': position['direction'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'entry_time': position['entry_time'],
                        'exit_time': row['timestamp'],
                        'duration': i - position['entry_index'],
                        'pct_change': pct_change,
                        'result': 'WIN',
                        'pnl': self.config['position_size'] * (tp_pct / 100),
                        'filters_count': position['filters_count']
                    }
                    trades.append(trade_result)
                    trade_closed = True
                    
                # Verifica SL
                elif pct_change <= -self.config['stop_loss_pct'] * 100:
                    trade_result = {
                        'asset': asset,
                        'direction': position['direction'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'entry_time': position['entry_time'],
                        'exit_time': row['timestamp'],
                        'duration': i - position['entry_index'],
                        'pct_change': pct_change,
                        'result': 'LOSS',
                        'pnl': -self.config['position_size'] * self.config['stop_loss_pct'],
                        'filters_count': position['filters_count']
                    }
                    trades.append(trade_result)
                    trade_closed = True
                
                if trade_closed:
                    position = None
                    
        # Calcula estatísticas
        total_trades = len(trades)
        wins = len([t for t in trades if t['result'] == 'WIN'])
        losses = len([t for t in trades if t['result'] == 'LOSS'])
        total_pnl = sum([t['pnl'] for t in trades])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'trades': total_trades,
            'wins': wins,
            'losses': losses,
            'pnl': total_pnl,
            'win_rate': win_rate,
            'trade_details': trades
        }
    
    def run_extensive_test(self):
        """Executa teste extensivo completo"""
        print("🚀 TESTE EXTENSIVO: TP 40% vs TP 30%")
        print("📊 Configuração: 10 ativos | 6 meses | Intervalos 4h")
        print("=" * 70)
        
        all_results = {
            'tp30': {},
            'tp40': {}
        }
        
        for asset in self.assets:
            print(f"\n📈 Processando {asset}...")
            
            # Busca dados extensivos
            df = self.fetch_extended_data(asset, interval='4h', months=6)
            if df is None or len(df) < 200:
                print(f"⚠️ Dados insuficientes para {asset}")
                continue
                
            # Calcula indicadores
            df = self.calculate_advanced_indicators(df)
            if len(df) < 100:
                print(f"⚠️ Poucos dados após indicadores para {asset}")
                continue
            
            # Simula TP 30%
            result_tp30 = self.simulate_extended_trading(df, 30.0, asset)
            all_results['tp30'][asset] = result_tp30
            
            # Simula TP 40%
            result_tp40 = self.simulate_extended_trading(df, 40.0, asset)
            all_results['tp40'][asset] = result_tp40
            
            # Mostra resultado
            print(f"  📊 TP 30%: {result_tp30['trades']} trades | PnL: ${result_tp30['pnl']:.2f} | WR: {result_tp30['win_rate']:.1f}%")
            print(f"  📊 TP 40%: {result_tp40['trades']} trades | PnL: ${result_tp40['pnl']:.2f} | WR: {result_tp40['win_rate']:.1f}%")
            
            # Atualiza totais
            for tp_type in ['tp30', 'tp40']:
                result = all_results[tp_type][asset]
                self.results[tp_type]['trades'] += result['trades']
                self.results[tp_type]['wins'] += result['wins']
                self.results[tp_type]['losses'] += result['losses']
                self.results[tp_type]['pnl'] += result['pnl']
                self.results[tp_type]['asset_details'][asset] = result
        
        # Calcula win rates totais
        for tp_type in ['tp30', 'tp40']:
            total_trades = self.results[tp_type]['trades']
            if total_trades > 0:
                self.results[tp_type]['win_rate'] = (
                    self.results[tp_type]['wins'] / total_trades * 100
                )
            else:
                self.results[tp_type]['win_rate'] = 0
        
        # Apresenta resultados finais
        self.present_final_results(all_results)
        
        # Salva resultados
        self.save_extensive_results(all_results)
    
    def present_final_results(self, all_results):
        """Apresenta resultados finais formatados"""
        print("\n" + "=" * 70)
        print("🏆 RESULTADOS EXTENSIVOS FINAIS")
        print("=" * 70)
        
        tp30 = self.results['tp30']
        tp40 = self.results['tp40']
        
        print(f"\n📊 TP 30% (BASELINE VENCEDORA):")
        print(f"   💰 PnL Total: ${tp30['pnl']:.2f}")
        print(f"   📈 Capital Final: ${self.config['initial_capital'] + tp30['pnl']:.2f}")
        print(f"   🎯 Total Trades: {tp30['trades']}")
        print(f"   ✅ Trades Vencedores: {tp30['wins']}")
        print(f"   ❌ Trades Perdedores: {tp30['losses']}")
        print(f"   📊 Win Rate: {tp30['win_rate']:.1f}%")
        print(f"   💹 ROI: {(tp30['pnl'] / self.config['initial_capital']) * 100:.1f}%")
        
        print(f"\n🚀 TP 40% (TESTE OTIMIZADO):")
        print(f"   💰 PnL Total: ${tp40['pnl']:.2f}")
        print(f"   📈 Capital Final: ${self.config['initial_capital'] + tp40['pnl']:.2f}")
        print(f"   🎯 Total Trades: {tp40['trades']}")
        print(f"   ✅ Trades Vencedores: {tp40['wins']}")
        print(f"   ❌ Trades Perdedores: {tp40['losses']}")
        print(f"   📊 Win Rate: {tp40['win_rate']:.1f}%")
        print(f"   💹 ROI: {(tp40['pnl'] / self.config['initial_capital']) * 100:.1f}%")
        
        # Análise comparativa
        pnl_diff = tp40['pnl'] - tp30['pnl']
        roi_diff = ((tp40['pnl'] / self.config['initial_capital']) - 
                   (tp30['pnl'] / self.config['initial_capital'])) * 100
        trade_diff = tp40['trades'] - tp30['trades']
        wr_diff = tp40['win_rate'] - tp30['win_rate']
        
        print(f"\n🔍 ANÁLISE COMPARATIVA:")
        print(f"   💰 Diferença PnL: ${pnl_diff:.2f}")
        print(f"   💹 Diferença ROI: {roi_diff:.1f}%")
        print(f"   🎯 Diferença Trades: {trade_diff}")
        print(f"   📊 Diferença Win Rate: {wr_diff:.1f}%")
        
        # Determina vencedor
        if pnl_diff > 0.1:  # Margem de $0.10
            print(f"   🏆 VENCEDOR CLARO: TP 40% (+${pnl_diff:.2f})")
            recommendation = "Recomendação: MUDAR para TP 40%"
        elif pnl_diff < -0.1:
            print(f"   🏆 VENCEDOR CLARO: TP 30% (+${abs(pnl_diff):.2f})")
            recommendation = "Recomendação: MANTER TP 30%"
        else:
            print(f"   🤝 EMPATE TÉCNICO (diferença < $0.10)")
            recommendation = "Recomendação: MANTER TP 30% (configuração testada)"
            
        print(f"   💡 {recommendation}")
        
        # Top performers por ativo
        print(f"\n📋 PERFORMANCE POR ATIVO:")
        for asset in self.assets:
            if asset in all_results['tp30'] and asset in all_results['tp40']:
                tp30_pnl = all_results['tp30'][asset]['pnl']
                tp40_pnl = all_results['tp40'][asset]['pnl']
                diff = tp40_pnl - tp30_pnl
                
                if diff > 0.05:
                    emoji = "🚀"
                elif diff < -0.05:
                    emoji = "📉"
                else:
                    emoji = "➡️"
                    
                print(f"   {emoji} {asset:8} | TP30: ${tp30_pnl:6.2f} | TP40: ${tp40_pnl:6.2f} | Diff: ${diff:6.2f}")
    
    def save_extensive_results(self, all_results):
        """Salva resultados extensivos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"teste_extensivo_tp40_vs_tp30_{timestamp}.json"
        
        # Prepara dados para salvar (sem trade_details para reduzir tamanho)
        save_data = {
            'timestamp': timestamp,
            'test_type': 'extensive_tp40_vs_tp30',
            'description': 'Teste extensivo TP 40% vs TP 30% com 6 meses de dados',
            'config': self.config,
            'assets_tested': self.assets,
            'total_results': self.results,
            'summary_by_asset': {}
        }
        
        # Adiciona resumo por ativo
        for asset in self.assets:
            if asset in all_results['tp30'] and asset in all_results['tp40']:
                save_data['summary_by_asset'][asset] = {
                    'tp30': {k: v for k, v in all_results['tp30'][asset].items() if k != 'trade_details'},
                    'tp40': {k: v for k, v in all_results['tp40'][asset].items() if k != 'trade_details'}
                }
        
        # Adiciona conclusão
        pnl_diff = self.results['tp40']['pnl'] - self.results['tp30']['pnl']
        save_data['conclusion'] = {
            'winner': 'TP40' if pnl_diff > 0.1 else 'TP30' if pnl_diff < -0.1 else 'TIE',
            'pnl_difference': pnl_diff,
            'roi_difference': ((self.results['tp40']['pnl'] / self.config['initial_capital']) - 
                              (self.results['tp30']['pnl'] / self.config['initial_capital'])) * 100,
            'recommendation': 'MUDAR para TP 40%' if pnl_diff > 0.1 else 'MANTER TP 30%'
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
            
        print(f"\n💾 Resultados salvos em: {filename}")

def main():
    """Função principal"""
    print("🧪 TESTE EXTENSIVO DE OTIMIZAÇÃO")
    print("Comparando TP 40% vs TP 30% com dados históricos robustos")
    print("=" * 70)
    
    test_engine = ExtensiveTP40Test()
    test_engine.run_extensive_test()
    
    print("\n✅ Teste extensivo concluído!")
    print("📊 Verifique o arquivo JSON para análise detalhada")

if __name__ == "__main__":
    main()
