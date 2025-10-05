#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 TESTE VALIDAÇÃO: TP 40% vs TP 30% com Dados de 1 Ano
======================================================

Este teste usa os MESMOS dados de 1 ano que geraram os 2190% ROI
para validar se TP 40% melhora ou piora a performance.

OBJETIVO:
- TP 30% deve reproduzir ~2190% ROI
- TP 40% será comparado com esta baseline
- Usa dados_reais_*_1ano.csv (mesmos da simulação vencedora)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ValidationTP40Test:
    """Teste de validação com dados de 1 ano"""
    
    def __init__(self):
        # Configuração EXATA da simulação vencedora (2190% ROI)
        self.config = {
            'initial_capital': 10.0,
            'position_size': 1.0,
            'take_profit_pct_30': 0.30,  # 30% - baseline vencedora
            'take_profit_pct_40': 0.40,  # 40% - teste
            'stop_loss_pct': 0.10,       # 10%
            'emergency_stop': -0.05,
            'atr_min_pct': 0.5,
            'atr_max_pct': 3.0,
            'volume_multiplier': 3.0,
            'gradient_min_long': 0.08,
            'gradient_min_short': 0.12,
            'breakout_atr_mult': 0.8,
            'rsi_min': 20,
            'rsi_max': 70,
            'min_confluence': 3
        }
        
        # Mesmos 10 ativos da simulação de 2190% ROI
        self.assets = [
            'btc', 'eth', 'sol', 'avax', 'link',
            'ada', 'doge', 'xrp', 'bnb', 'ltc'
        ]
        
        self.results = {
            'tp30': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'by_asset': {}},
            'tp40': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'by_asset': {}}
        }
        
    def load_real_data(self, asset):
        """Carrega dados reais de 1 ano do CSV"""
        filename = f"dados_reais_{asset}_1ano.csv"
        
        if not os.path.exists(filename):
            print(f"❌ Arquivo não encontrado: {filename}")
            return None
            
        try:
            print(f"📂 Carregando {filename}...")
            
            df = pd.read_csv(filename)
            
            # Renomeia colunas para padrão
            df = df.rename(columns={
                'valor_fechamento': 'close',
                'valor_abertura': 'open', 
                'valor_maximo': 'high',
                'valor_minimo': 'low',
                'volume': 'volume',
                'timestamp': 'timestamp'
            })
            
            # Converte timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ordena por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ {asset.upper()}: {len(df)} candles de {df['timestamp'].min()} a {df['timestamp'].max()}")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            print(f"❌ Erro ao carregar {filename}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calcula indicadores técnicos EXATOS da simulação vencedora"""
        # EMAs (7 e 21 períodos)
        df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # ATR (14 períodos)
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14, min_periods=14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Volume MA (20 períodos)
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI (14 períodos)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Gradients EMA
        df['grad_ema7'] = ((df['ema7'] - df['ema7'].shift(1)) / df['ema7'].shift(1)) * 100
        df['grad_ema21'] = ((df['ema21'] - df['ema21'].shift(1)) / df['ema21'].shift(1)) * 100
        
        # Remove NaNs
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def check_entry_filters(self, row):
        """Verifica filtros de entrada EXATOS da configuração vencedora"""
        filters_passed = []
        
        # 1. ATR saudável (0.5% - 3.0%)
        atr_ok = self.config['atr_min_pct'] <= row['atr_pct'] <= self.config['atr_max_pct']
        filters_passed.append(atr_ok)
        
        # 2. Volume acima de 3.0x
        vol_ok = row['volume_ratio'] >= self.config['volume_multiplier']
        filters_passed.append(vol_ok)
        
        # 3. RSI não extremo (20-70)
        rsi_ok = self.config['rsi_min'] <= row['rsi'] <= self.config['rsi_max']
        filters_passed.append(rsi_ok)
        
        # 4. Gradient EMA7 significativo
        grad_ok = abs(row['grad_ema7']) >= self.config['gradient_min_long']
        filters_passed.append(grad_ok)
        
        # 5. Breakout das EMAs (separação > 0.8 * ATR)
        ema_separation = abs(row['ema7'] - row['ema21'])
        breakout_ok = ema_separation >= (self.config['breakout_atr_mult'] * row['atr'])
        filters_passed.append(breakout_ok)
        
        # Verifica confluência mínima
        filters_count = sum(filters_passed)
        return filters_count >= self.config['min_confluence'], filters_count
    
    def simulate_trading(self, df, tp_pct, asset_name):
        """Simula trading com configuração EXATA da simulação vencedora"""
        trades = []
        position = None
        
        print(f"  🔄 Simulando {asset_name.upper()} com TP {tp_pct*100:.0f}%...")
        
        for i in range(50, len(df)):  # Skip first 50 for indicators stabilization
            row = df.iloc[i]
            
            # Se não tem posição, verifica entrada
            if position is None:
                confluence_ok, filters_count = self.check_entry_filters(row)
                
                if confluence_ok:
                    # Determina direção baseada em EMA e gradient
                    direction = None
                    
                    if (row['ema7'] > row['ema21'] and 
                        row['grad_ema7'] >= self.config['gradient_min_long']):
                        direction = 'LONG'
                    elif (row['ema7'] < row['ema21'] and 
                          row['grad_ema7'] <= -self.config['gradient_min_short']):
                        direction = 'SHORT'
                    
                    if direction:
                        # SISTEMA INVERSO: inverte o sinal (como na simulação original)
                        final_direction = 'SHORT' if direction == 'LONG' else 'LONG'
                        
                        position = {
                            'direction': final_direction,
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
                
                # Verifica Take Profit
                if pct_change >= tp_pct * 100:
                    trade_result = {
                        'asset': asset_name,
                        'direction': position['direction'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pct_change': pct_change,
                        'result': 'WIN',
                        'pnl': self.config['position_size'] * tp_pct,
                        'duration_hours': i - position['entry_index']
                    }
                    trades.append(trade_result)
                    trade_closed = True
                    
                # Verifica Stop Loss
                elif pct_change <= -self.config['stop_loss_pct'] * 100:
                    trade_result = {
                        'asset': asset_name,
                        'direction': position['direction'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pct_change': pct_change,
                        'result': 'LOSS',
                        'pnl': -self.config['position_size'] * self.config['stop_loss_pct'],
                        'duration_hours': i - position['entry_index']
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
        
        print(f"    📊 {total_trades} trades | PnL: ${total_pnl:.2f} | WR: {win_rate:.1f}%")
        
        return {
            'trades': total_trades,
            'wins': wins,
            'losses': losses,
            'pnl': total_pnl,
            'win_rate': win_rate,
            'trade_details': trades
        }
    
    def run_validation_test(self):
        """Executa teste de validação completo"""
        print("🎯 TESTE DE VALIDAÇÃO: TP 40% vs TP 30%")
        print("📊 Usando dados REAIS de 1 ano (mesmos da simulação 2190% ROI)")
        print("=" * 75)
        
        for asset in self.assets:
            print(f"\n📈 Processando {asset.upper()}...")
            
            # Carrega dados reais
            df = self.load_real_data(asset)
            if df is None or len(df) < 100:
                print(f"⚠️ Pulando {asset} - dados insuficientes")
                continue
                
            # Calcula indicadores
            df = self.calculate_indicators(df)
            if len(df) < 50:
                print(f"⚠️ Pulando {asset} - poucos dados após indicadores")
                continue
            
            # Simula TP 30% (baseline que deveria dar 2190% ROI)
            result_tp30 = self.simulate_trading(df, self.config['take_profit_pct_30'], asset)
            self.results['tp30']['by_asset'][asset] = result_tp30
            
            # Simula TP 40% (teste)
            result_tp40 = self.simulate_trading(df, self.config['take_profit_pct_40'], asset)
            self.results['tp40']['by_asset'][asset] = result_tp40
            
            # Atualiza totais
            for tp_key, result in [('tp30', result_tp30), ('tp40', result_tp40)]:
                self.results[tp_key]['trades'] += result['trades']
                self.results[tp_key]['wins'] += result['wins']
                self.results[tp_key]['losses'] += result['losses']
                self.results[tp_key]['pnl'] += result['pnl']
        
        # Calcula win rates finais
        for tp_key in ['tp30', 'tp40']:
            total_trades = self.results[tp_key]['trades']
            if total_trades > 0:
                self.results[tp_key]['win_rate'] = (
                    self.results[tp_key]['wins'] / total_trades * 100
                )
            else:
                self.results[tp_key]['win_rate'] = 0
        
        # Apresenta resultados
        self.present_validation_results()
        
        # Salva resultados
        self.save_validation_results()
    
    def present_validation_results(self):
        """Apresenta resultados de validação"""
        print("\n" + "=" * 75)
        print("🏆 RESULTADOS DE VALIDAÇÃO")
        print("=" * 75)
        
        tp30 = self.results['tp30']
        tp40 = self.results['tp40']
        
        # Calcula ROI
        roi_tp30 = (tp30['pnl'] / self.config['initial_capital']) * 100
        roi_tp40 = (tp40['pnl'] / self.config['initial_capital']) * 100
        
        print(f"\n📊 TP 30% (BASELINE - deveria ser ~2190% ROI):")
        print(f"   💰 PnL Total: ${tp30['pnl']:.2f}")
        print(f"   📈 Capital Final: ${self.config['initial_capital'] + tp30['pnl']:.2f}")
        print(f"   🎯 Total Trades: {tp30['trades']}")
        print(f"   ✅ Wins: {tp30['wins']} | ❌ Losses: {tp30['losses']}")
        print(f"   📊 Win Rate: {tp30['win_rate']:.1f}%")
        print(f"   💹 ROI: {roi_tp30:.1f}% {'✅' if roi_tp30 > 2000 else '⚠️ (esperado ~2190%)'}")
        
        print(f"\n🚀 TP 40% (TESTE):")
        print(f"   💰 PnL Total: ${tp40['pnl']:.2f}")
        print(f"   📈 Capital Final: ${self.config['initial_capital'] + tp40['pnl']:.2f}")
        print(f"   🎯 Total Trades: {tp40['trades']}")
        print(f"   ✅ Wins: {tp40['wins']} | ❌ Losses: {tp40['losses']}")
        print(f"   📊 Win Rate: {tp40['win_rate']:.1f}%")
        print(f"   💹 ROI: {roi_tp40:.1f}%")
        
        # Comparação
        pnl_diff = tp40['pnl'] - tp30['pnl']
        roi_diff = roi_tp40 - roi_tp30
        
        print(f"\n🔍 COMPARAÇÃO:")
        print(f"   💰 Diferença PnL: ${pnl_diff:.2f}")
        print(f"   💹 Diferença ROI: {roi_diff:.1f}%")
        
        if pnl_diff > 1.0:
            winner = "🏆 VENCEDOR: TP 40%"
            recommendation = "RECOMENDAÇÃO: Mudar para TP 40%"
        elif pnl_diff < -1.0:
            winner = "🏆 VENCEDOR: TP 30%"
            recommendation = "RECOMENDAÇÃO: Manter TP 30%"
        else:
            winner = "🤝 EMPATE TÉCNICO"
            recommendation = "RECOMENDAÇÃO: Manter TP 30% (testado)"
            
        print(f"   {winner}")
        print(f"   💡 {recommendation}")
        
        # Performance por ativo
        print(f"\n📋 PERFORMANCE POR ATIVO:")
        for asset in self.assets:
            if asset in self.results['tp30']['by_asset'] and asset in self.results['tp40']['by_asset']:
                tp30_pnl = self.results['tp30']['by_asset'][asset]['pnl']
                tp40_pnl = self.results['tp40']['by_asset'][asset]['pnl']
                diff = tp40_pnl - tp30_pnl
                
                emoji = "🚀" if diff > 0.1 else "📉" if diff < -0.1 else "➡️"
                print(f"   {emoji} {asset.upper():5} | TP30: ${tp30_pnl:7.2f} | TP40: ${tp40_pnl:7.2f} | Diff: ${diff:7.2f}")
    
    def save_validation_results(self):
        """Salva resultados de validação"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validacao_tp40_vs_tp30_1ano_{timestamp}.json"
        
        # Calcula ROI
        roi_tp30 = (self.results['tp30']['pnl'] / self.config['initial_capital']) * 100
        roi_tp40 = (self.results['tp40']['pnl'] / self.config['initial_capital']) * 100
        
        data = {
            'timestamp': timestamp,
            'test_type': 'validation_tp40_vs_tp30_1year_data',
            'description': 'Validação usando dados reais de 1 ano que geraram 2190% ROI',
            'config': self.config,
            'assets_tested': self.assets,
            'results': {
                'tp30': {
                    **self.results['tp30'],
                    'roi_percent': roi_tp30,
                    'final_capital': self.config['initial_capital'] + self.results['tp30']['pnl']
                },
                'tp40': {
                    **self.results['tp40'],
                    'roi_percent': roi_tp40,
                    'final_capital': self.config['initial_capital'] + self.results['tp40']['pnl']
                }
            },
            'comparison': {
                'pnl_difference': self.results['tp40']['pnl'] - self.results['tp30']['pnl'],
                'roi_difference': roi_tp40 - roi_tp30,
                'better_strategy': 'TP40' if self.results['tp40']['pnl'] > self.results['tp30']['pnl'] else 'TP30',
                'tp30_validates_2190_roi': roi_tp30 > 2000
            }
        }
        
        # Remove trade_details para reduzir tamanho do arquivo
        for tp_key in ['tp30', 'tp40']:
            for asset in data['results'][tp_key]['by_asset']:
                if 'trade_details' in data['results'][tp_key]['by_asset'][asset]:
                    del data['results'][tp_key]['by_asset'][asset]['trade_details']
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"\n💾 Resultados salvos em: {filename}")

def main():
    """Função principal"""
    print("🎯 TESTE DE VALIDAÇÃO COM DADOS DE 1 ANO")
    print("Verificando se TP 30% reproduz os 2190% ROI e testando TP 40%")
    print("=" * 75)
    
    validator = ValidationTP40Test()
    validator.run_validation_test()
    
    print("\n✅ Teste de validação concluído!")

if __name__ == "__main__":
    main()
