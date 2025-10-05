#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 TESTE EXATO: Replicação da Simulação de 2.190% ROI
===================================================

Este script replica EXATAMENTE a lógica que gerou os 2.190% ROI
baseado no arquivo otimizacao_parametros.py original.

CONFIGURAÇÃO EXATA (do JSON de 2.190% ROI):
• TP: 30% | SL: 10%
• ATR: 0.5-3.0%
• Volume: 3.0x
• Gradient LONG: ≥0.08% | SHORT: ≥0.12%
• RSI: 20-70
• Confluência: 3/5 filtros
• Breakout: 0.8x ATR
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ExactReplicationTest:
    """Replica exatamente a lógica da simulação de 2.190% ROI"""
    
    def __init__(self):
        # Configuração EXATA da simulação vencedora (do JSON)
        self.params = {
            'initial_capital': 10.0,
            'position_size': 1.0,
            'emergency_stop': -0.05,
            'ema_short': 7,
            'ema_long': 21,
            'atr_period': 14,
            'vol_ma_period': 20,
            'take_profit_pct': 0.3,  # 30%
            'stop_loss_pct': 0.1,    # 10%
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
        
        # Mesmos 10 ativos da simulação original
        self.assets = ['btc', 'eth', 'sol', 'avax', 'link', 'ada', 'doge', 'xrp', 'bnb', 'ltc']
        self.real_data = {}
        
    def load_real_data(self):
        """Carrega dados reais exatamente como na simulação original"""
        print("📂 Carregando dados reais (mesmo formato da simulação 2190% ROI)...")
        
        for asset in self.assets:
            try:
                filename = f"dados_reais_{asset}_1ano.csv"
                if not os.path.exists(filename):
                    print(f"  ❌ Arquivo não encontrado: {filename}")
                    continue
                    
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.real_data[asset.upper()] = df
                print(f"  ✅ {asset.upper()}: {len(df)} pontos")
                
            except Exception as e:
                print(f"  ❌ Erro ao carregar {asset}: {e}")
        
        print(f"📊 {len(self.real_data)} ativos carregados")
        return len(self.real_data) > 0
    
    def calculate_indicators(self, df):
        """Calcula indicadores EXATAMENTE como na simulação original"""
        df = df.copy()
        
        # EMAs (mesmos parâmetros)
        df['ema_short'] = df['valor_fechamento'].ewm(span=self.params['ema_short']).mean()
        df['ema_long'] = df['valor_fechamento'].ewm(span=self.params['ema_long']).mean()
        
        # ATR (mesmo método)
        df['high_low'] = df['valor_maximo'] - df['valor_minimo']
        df['high_close_prev'] = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
        df['low_close_prev'] = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df['atr'] = df['true_range'].rolling(self.params['atr_period']).mean()
        df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
        
        # Gradient (mesmo cálculo)
        df['ema_short_grad'] = df['ema_short'].pct_change() * 100
        
        # RSI (mesmo método)
        delta = df['valor_fechamento'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(self.params['vol_ma_period']).mean()
        
        return df
    
    def apply_filters(self, df):
        """Aplica filtros EXATAMENTE como na simulação original"""
        
        # Filtro ATR (0.5% - 3.0%)
        atr_filter = (
            (df['atr_pct'] >= self.params['atr_min_pct']) & 
            (df['atr_pct'] <= self.params['atr_max_pct'])
        )
        
        # Filtro Volume (≥ 3.0x)
        volume_filter = df['volume'] >= (df['volume_ma'] * self.params['volume_multiplier'])
        
        # Filtro Gradient
        long_gradient = df['ema_short_grad'] >= self.params['gradient_min_long']
        short_gradient = df['ema_short_grad'] <= -self.params['gradient_min_short']
        gradient_filter = long_gradient | short_gradient
        
        # Filtro Breakout (0.8x ATR)
        ema_diff = abs(df['ema_short'] - df['ema_long'])
        breakout_filter = ema_diff >= (df['atr'] * self.params['breakout_atr_mult'])
        
        # Filtro RSI (20-70)
        rsi_filter = (
            (df['rsi'] > self.params['rsi_min']) & 
            (df['rsi'] < self.params['rsi_max'])
        )
        
        # Confluence (≥ 3 de 5 filtros)
        filters = [atr_filter, volume_filter, gradient_filter, breakout_filter, rsi_filter]
        confluence_count = sum(filters)
        confluence_filter = confluence_count >= self.params['min_confluence']
        
        return df[confluence_filter].copy()
    
    def generate_signals(self, filtered_df):
        """Gera sinais EXATAMENTE como na simulação original"""
        signals = []
        
        for _, row in filtered_df.iterrows():
            # Determina direção baseada em EMA e gradient
            if (row['ema_short'] > row['ema_long'] and 
                row['ema_short_grad'] >= self.params['gradient_min_long']):
                side = 'LONG'
            elif (row['ema_short'] < row['ema_long'] and 
                  row['ema_short_grad'] <= -self.params['gradient_min_short']):
                side = 'SHORT'
            else:
                continue
            
            signals.append({
                'side': side,
                'entry_price': row['valor_fechamento'],
                'timestamp': row['timestamp'],
                'atr_pct': row['atr_pct']
            })
        
        return signals
    
    def simulate_trades(self, signals):
        """Simula trades EXATAMENTE como na simulação original"""
        trades = []
        
        for signal in signals:
            entry_price = signal['entry_price']
            side = signal['side']
            
            # Calcula preços de TP e SL
            if side == 'LONG':
                take_profit_price = entry_price * (1 + self.params['take_profit_pct'])
                stop_loss_price = entry_price * (1 - self.params['stop_loss_pct'])
            else:  # SHORT
                take_profit_price = entry_price * (1 - self.params['take_profit_pct'])
                stop_loss_price = entry_price * (1 + self.params['stop_loss_pct'])
            
            # Determina probabilidade baseada em TP/SL ratio (como no original)
            tp_sl_ratio = self.params['take_profit_pct'] / self.params['stop_loss_pct']
            
            # Lógica EXATA da simulação original
            if tp_sl_ratio <= 2:  # TP/SL <= 2:1
                tp_prob = 0.45
            elif tp_sl_ratio <= 4:  # TP/SL 2:1 a 4:1
                tp_prob = 0.35
            elif tp_sl_ratio <= 6:  # TP/SL 4:1 a 6:1
                tp_prob = 0.25
            else:  # TP/SL > 6:1
                tp_prob = 0.20
            
            # Para TP 30% / SL 10% = ratio 3:1 → tp_prob = 0.35
            
            outcome = np.random.random()
            
            if outcome < tp_prob:
                exit_price = take_profit_price
                pnl_pct = self.params['take_profit_pct'] * 100
                exit_reason = 'take_profit'
            elif outcome < (tp_prob + 0.7):  # 70% dos restantes vai para SL
                exit_price = stop_loss_price
                pnl_pct = -self.params['stop_loss_pct'] * 100
                exit_reason = 'stop_loss'
            else:  # Emergency stop
                pnl_pct = (self.params['emergency_stop'] / self.params['position_size']) * 100
                exit_price = entry_price * (1 + pnl_pct / 100)
                exit_reason = 'emergency_stop'
            
            if side == 'SHORT':
                pnl_pct = -pnl_pct
            
            pnl_dollars = self.params['position_size'] * (pnl_pct / 100)
            
            trades.append({
                'side': side,
                'pnl_dollars': pnl_dollars,
                'exit_reason': exit_reason,
                'atr_pct': signal['atr_pct']
            })
        
        return trades
    
    def test_tp40_vs_tp30(self):
        """Testa TP 40% vs TP 30% usando a lógica EXATA"""
        
        print("🎯 TESTE EXATO: Replicando lógica da simulação 2.190% ROI")
        print("=" * 70)
        
        # Carrega dados
        if not self.load_real_data():
            print("❌ Falha ao carregar dados")
            return
        
        # Fixa seed para reprodutibilidade
        np.random.seed(42)
        
        results = {
            'tp30': {'total_pnl': 0, 'total_trades': 0, 'total_wins': 0, 'asset_results': []},
            'tp40': {'total_pnl': 0, 'total_trades': 0, 'total_wins': 0, 'asset_results': []}
        }
        
        print(f"\n🔄 Processando {len(self.real_data)} ativos...")
        
        for asset_name, df in self.real_data.items():
            print(f"\n📈 {asset_name}...")
            
            try:
                # Calcula indicadores
                df_indicators = self.calculate_indicators(df)
                
                # Aplica filtros
                filtered_df = self.apply_filters(df_indicators)
                print(f"  🔍 Após filtros: {len(filtered_df)} sinais de {len(df_indicators)} pontos ({len(filtered_df)/len(df_indicators)*100:.1f}%)")
                
                if len(filtered_df) < 5:
                    print(f"  ⚠️ Poucos sinais para {asset_name}")
                    continue
                
                # Gera sinais
                signals = self.generate_signals(filtered_df)
                print(f"  📊 Sinais válidos: {len(signals)}")
                
                if len(signals) == 0:
                    continue
                
                # Testa TP 30% (baseline)
                trades_tp30 = self.simulate_trades(signals)
                asset_pnl_tp30 = sum(t['pnl_dollars'] for t in trades_tp30)
                asset_wins_tp30 = len([t for t in trades_tp30 if t['pnl_dollars'] > 0])
                
                # Testa TP 40% (modifica temporariamente)
                original_tp = self.params['take_profit_pct']
                self.params['take_profit_pct'] = 0.4  # 40%
                trades_tp40 = self.simulate_trades(signals)
                asset_pnl_tp40 = sum(t['pnl_dollars'] for t in trades_tp40)
                asset_wins_tp40 = len([t for t in trades_tp40 if t['pnl_dollars'] > 0])
                self.params['take_profit_pct'] = original_tp  # Restaura
                
                # Atualiza resultados
                results['tp30']['total_pnl'] += asset_pnl_tp30
                results['tp30']['total_trades'] += len(trades_tp30)
                results['tp30']['total_wins'] += asset_wins_tp30
                results['tp30']['asset_results'].append({
                    'asset': asset_name,
                    'trades': len(trades_tp30),
                    'pnl': asset_pnl_tp30,
                    'signals': len(signals)
                })
                
                results['tp40']['total_pnl'] += asset_pnl_tp40
                results['tp40']['total_trades'] += len(trades_tp40)
                results['tp40']['total_wins'] += asset_wins_tp40
                results['tp40']['asset_results'].append({
                    'asset': asset_name,
                    'trades': len(trades_tp40),
                    'pnl': asset_pnl_tp40,
                    'signals': len(signals)
                })
                
                print(f"  📊 TP 30%: {len(trades_tp30)} trades | PnL: ${asset_pnl_tp30:.2f}")
                print(f"  📊 TP 40%: {len(trades_tp40)} trades | PnL: ${asset_pnl_tp40:.2f}")
                
            except Exception as e:
                print(f"  ❌ Erro em {asset_name}: {e}")
        
        # Calcula métricas finais
        for key in ['tp30', 'tp40']:
            total_trades = results[key]['total_trades']
            if total_trades > 0:
                results[key]['final_capital'] = self.params['initial_capital'] + results[key]['total_pnl']
                results[key]['roi'] = (results[key]['total_pnl'] / self.params['initial_capital']) * 100
                results[key]['win_rate'] = (results[key]['total_wins'] / total_trades) * 100
            else:
                results[key]['final_capital'] = self.params['initial_capital']
                results[key]['roi'] = 0
                results[key]['win_rate'] = 0
        
        # Apresenta resultados
        self.present_exact_results(results)
        
        # Salva resultados
        self.save_exact_results(results)
        
        return results
    
    def present_exact_results(self, results):
        """Apresenta resultados da replicação exata"""
        print("\n" + "=" * 70)
        print("🏆 RESULTADOS DA REPLICAÇÃO EXATA")
        print("=" * 70)
        
        tp30 = results['tp30']
        tp40 = results['tp40']
        
        print(f"\n📊 TP 30% (BASELINE - esperado ~2190% ROI):")
        print(f"   💰 PnL Total: ${tp30['total_pnl']:.2f}")
        print(f"   📈 Capital Final: ${tp30['final_capital']:.2f}")
        print(f"   💹 ROI: {tp30['roi']:.1f}%")
        print(f"   🎯 Total Trades: {tp30['total_trades']}")
        print(f"   ✅ Win Rate: {tp30['win_rate']:.1f}%")
        
        # Validação do resultado
        if tp30['roi'] > 2000:
            validation = "✅ VALIDADO! ROI próximo aos 2190% esperados"
        elif tp30['roi'] > 1000:
            validation = "⚠️ ROI alto, mas abaixo do esperado"
        else:
            validation = "❌ ROI muito abaixo do esperado - verificar implementação"
        print(f"   {validation}")
        
        print(f"\n🚀 TP 40% (TESTE):")
        print(f"   💰 PnL Total: ${tp40['total_pnl']:.2f}")
        print(f"   📈 Capital Final: ${tp40['final_capital']:.2f}")
        print(f"   💹 ROI: {tp40['roi']:.1f}%")
        print(f"   🎯 Total Trades: {tp40['total_trades']}")
        print(f"   ✅ Win Rate: {tp40['win_rate']:.1f}%")
        
        # Comparação
        pnl_diff = tp40['total_pnl'] - tp30['total_pnl']
        roi_diff = tp40['roi'] - tp30['roi']
        
        print(f"\n🔍 COMPARAÇÃO:")
        print(f"   💰 Diferença PnL: ${pnl_diff:.2f}")
        print(f"   💹 Diferença ROI: {roi_diff:.1f}%")
        
        if pnl_diff > 1.0:
            winner = "🏆 VENCEDOR: TP 40%"
            recommendation = "RECOMENDAÇÃO: Considerar mudança para TP 40%"
        elif pnl_diff < -1.0:
            winner = "🏆 VENCEDOR: TP 30%"
            recommendation = "RECOMENDAÇÃO: Manter TP 30%"
        else:
            winner = "🤝 EMPATE TÉCNICO"
            recommendation = "RECOMENDAÇÃO: Manter TP 30% (configuração testada)"
            
        print(f"   {winner}")
        print(f"   💡 {recommendation}")
        
        # Performance por ativo
        print(f"\n📋 PERFORMANCE POR ATIVO:")
        print("-" * 70)
        print(f"{'Asset':<6} {'TP30 PnL':<10} {'TP40 PnL':<10} {'Diferença':<12} {'Trades':<8}")
        print("-" * 70)
        
        for i, asset_tp30 in enumerate(tp30['asset_results']):
            asset_tp40 = tp40['asset_results'][i] if i < len(tp40['asset_results']) else {'pnl': 0, 'trades': 0}
            diff = asset_tp40['pnl'] - asset_tp30['pnl']
            
            print(f"{asset_tp30['asset']:<6} ${asset_tp30['pnl']:<9.2f} ${asset_tp40['pnl']:<9.2f} ${diff:<11.2f} {asset_tp30['trades']:<8}")
    
    def save_exact_results(self, results):
        """Salva resultados da replicação exata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"teste_exato_tp40_vs_tp30_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'test_type': 'exact_replication_2190_roi',
            'description': 'Replicação exata da lógica que gerou 2190% ROI para testar TP 40% vs TP 30%',
            'exact_params': self.params,
            'results': results,
            'validation': {
                'tp30_roi_expected': 2190,
                'tp30_roi_actual': results['tp30']['roi'],
                'validation_passed': results['tp30']['roi'] > 2000,
                'tp40_vs_tp30_winner': 'TP40' if results['tp40']['total_pnl'] > results['tp30']['total_pnl'] else 'TP30'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"\n💾 Resultados salvos em: {filename}")

def main():
    """Função principal"""
    print("🎯 TESTE DE REPLICAÇÃO EXATA")
    print("Baseado na lógica original que gerou 2.190% ROI")
    print("=" * 70)
    
    tester = ExactReplicationTest()
    results = tester.test_tp40_vs_tp30()
    
    print("\n✅ Teste de replicação exata concluído!")

if __name__ == "__main__":
    main()
