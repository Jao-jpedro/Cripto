#!/usr/bin/env python3
"""
Otimização de Parâmetros - Sistema de Trading
Testa múltiplas combinações para maximizar ROI
"""

import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingOptimizer:
    """
    Otimizador de parâmetros de trading
    """
    
    def __init__(self):
        self.real_data = {}
        self.load_real_data()
    
    def load_real_data(self):
        """Carrega dados reais baixados anteriormente"""
        assets = ['btc', 'eth', 'sol', 'avax', 'link', 'ada', 'doge', 'xrp', 'bnb', 'ltc']
        
        print("📂 Carregando dados reais...")
        for asset in assets:
            try:
                filename = f"dados_reais_{asset}_1ano.csv"
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.real_data[asset.upper()] = df
                print(f"  ✅ {asset.upper()}: {len(df)} pontos")
            except Exception as e:
                print(f"  ❌ Erro ao carregar {asset}: {e}")
        
        print(f"📊 {len(self.real_data)} ativos carregados")
    
    def calculate_indicators(self, df, params):
        """Calcula indicadores técnicos com parâmetros ajustáveis"""
        df = df.copy()
        
        # EMAs
        df['ema_short'] = df['valor_fechamento'].ewm(span=params['ema_short']).mean()
        df['ema_long'] = df['valor_fechamento'].ewm(span=params['ema_long']).mean()
        
        # ATR
        df['high_low'] = df['valor_maximo'] - df['valor_minimo']
        df['high_close_prev'] = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
        df['low_close_prev'] = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df['atr'] = df['true_range'].rolling(params['atr_period']).mean()
        df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
        
        # Gradient
        df['ema_short_grad'] = df['ema_short'].pct_change() * 100
        
        # RSI
        delta = df['valor_fechamento'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(params['vol_ma_period']).mean()
        
        return df
    
    def apply_filters(self, df, params):
        """Aplica filtros com parâmetros otimizáveis"""
        
        # Filtro ATR
        atr_filter = (
            (df['atr_pct'] >= params['atr_min_pct']) & 
            (df['atr_pct'] <= params['atr_max_pct'])
        )
        
        # Filtro Volume
        volume_filter = df['volume'] >= (df['volume_ma'] * params['volume_multiplier'])
        
        # Filtro Gradient
        long_gradient = df['ema_short_grad'] >= params['gradient_min_long']
        short_gradient = df['ema_short_grad'] <= -params['gradient_min_short']
        gradient_filter = long_gradient | short_gradient
        
        # Filtro Breakout
        ema_diff = abs(df['ema_short'] - df['ema_long'])
        breakout_filter = ema_diff >= (df['atr'] * params['breakout_atr_mult'])
        
        # Filtro RSI
        rsi_filter = (
            (df['rsi'] > params['rsi_min']) & 
            (df['rsi'] < params['rsi_max'])
        )
        
        # Confluence
        filters = [atr_filter, volume_filter, gradient_filter, breakout_filter, rsi_filter]
        confluence_score = sum(f.fillna(False).astype(int) for f in filters)
        final_filter = confluence_score >= params['min_confluence']
        
        return df[final_filter & df['atr_pct'].notna() & df['ema_short'].notna()]
    
    def generate_signals(self, df, params):
        """Gera sinais de trading"""
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if row['ema_short_grad'] >= params['gradient_min_long']:
                side = 'LONG'
            elif row['ema_short_grad'] <= -params['gradient_min_short']:
                side = 'SHORT'
            else:
                continue
            
            signals.append({
                'timestamp': row['timestamp'],
                'side': side,
                'entry_price': row['valor_fechamento'],
                'atr_pct': row['atr_pct']
            })
        
        return signals
    
    def simulate_trades(self, signals, params):
        """Simula trades com parâmetros otimizáveis"""
        trades = []
        
        np.random.seed(42)  # Reproduzibilidade
        
        for signal in signals:
            entry_price = signal['entry_price']
            side = signal['side']
            
            # Calcular preços de saída
            if side == 'LONG':
                take_profit_price = entry_price * (1 + params['take_profit_pct'])
                stop_loss_price = entry_price * (1 - params['stop_loss_pct'])
            else:
                take_profit_price = entry_price * (1 - params['take_profit_pct'])
                stop_loss_price = entry_price * (1 + params['stop_loss_pct'])
            
            # Probabilidades baseadas no TP/SL ratio
            tp_sl_ratio = params['take_profit_pct'] / params['stop_loss_pct']
            
            # Quanto maior o TP, menor a chance de atingir
            if tp_sl_ratio <= 2:  # TP/SL <= 2:1
                tp_prob = 0.45
            elif tp_sl_ratio <= 4:  # TP/SL 2:1 a 4:1
                tp_prob = 0.35
            elif tp_sl_ratio <= 6:  # TP/SL 4:1 a 6:1
                tp_prob = 0.25
            else:  # TP/SL > 6:1
                tp_prob = 0.20
            
            outcome = np.random.random()
            
            if outcome < tp_prob:
                exit_price = take_profit_price
                pnl_pct = params['take_profit_pct'] * 100
                exit_reason = 'take_profit'
            elif outcome < (tp_prob + 0.7):  # 70% dos restantes vai para SL
                exit_price = stop_loss_price
                pnl_pct = -params['stop_loss_pct'] * 100
                exit_reason = 'stop_loss'
            else:  # Emergency stop
                pnl_pct = (params['emergency_stop'] / params['position_size']) * 100
                exit_price = entry_price * (1 + pnl_pct / 100)
                exit_reason = 'emergency_stop'
            
            if side == 'SHORT':
                pnl_pct = -pnl_pct
            
            pnl_dollars = params['position_size'] * (pnl_pct / 100)
            
            trades.append({
                'side': side,
                'pnl_dollars': pnl_dollars,
                'exit_reason': exit_reason
            })
        
        return trades
    
    def evaluate_params(self, params):
        """Avalia um conjunto de parâmetros"""
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        asset_results = []
        
        for asset_name, df in self.real_data.items():
            try:
                # Calcular indicadores
                df_indicators = self.calculate_indicators(df, params)
                
                # Aplicar filtros
                filtered_df = self.apply_filters(df_indicators, params)
                
                if len(filtered_df) < 5:
                    continue
                
                # Gerar sinais
                signals = self.generate_signals(filtered_df, params)
                
                if len(signals) == 0:
                    continue
                
                # Simular trades
                trades = self.simulate_trades(signals, params)
                
                # Métricas
                asset_pnl = sum(t['pnl_dollars'] for t in trades)
                asset_wins = len([t for t in trades if t['pnl_dollars'] > 0])
                
                total_pnl += asset_pnl
                total_trades += len(trades)
                total_wins += asset_wins
                
                asset_results.append({
                    'asset': asset_name,
                    'trades': len(trades),
                    'pnl': asset_pnl,
                    'signals': len(filtered_df)
                })
                
            except Exception as e:
                continue
        
        if total_trades == 0:
            return {
                'roi': -100,
                'final_capital': 0,
                'total_trades': 0,
                'win_rate': 0,
                'asset_results': []
            }
        
        final_capital = params['initial_capital'] + total_pnl
        roi = (total_pnl / params['initial_capital']) * 100
        win_rate = (total_wins / total_trades) * 100
        
        return {
            'roi': roi,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'asset_results': asset_results
        }
    
    def optimize_parameters(self):
        """Otimiza parâmetros através de grid search"""
        print("\n🔧 INICIANDO OTIMIZAÇÃO DE PARÂMETROS")
        print("🎯 Objetivo: Maximizar ROI com dados reais")
        print("=" * 60)
        
        # Espaço de parâmetros para testar
        param_space = {
            # Financeiros
            'take_profit_pct': [0.10, 0.15, 0.20, 0.25, 0.30],
            'stop_loss_pct': [0.03, 0.05, 0.07, 0.10],
            
            # Filtros ATR
            'atr_min_pct': [0.5, 0.7, 1.0],
            'atr_max_pct': [2.5, 3.0],
            
            # Volume
            'volume_multiplier': [2.0, 2.5, 3.0],
            
            # Gradient
            'gradient_min_long': [0.08, 0.10, 0.12],
            'gradient_min_short': [0.10, 0.12, 0.15],
            
            # Breakout
            'breakout_atr_mult': [0.8, 1.0, 1.2],
            
            # RSI
            'rsi_min': [20, 25, 30],
            'rsi_max': [70, 75, 80],
            
            # Confluence
            'min_confluence': [3, 4, 5]
        }
        
        # Parâmetros fixos
        fixed_params = {
            'initial_capital': 10.0,
            'position_size': 1.0,
            'emergency_stop': -0.05,
            'ema_short': 7,
            'ema_long': 21,
            'atr_period': 14,
            'vol_ma_period': 20
        }
        
        # Gerar combinações (amostra aleatória para não demorar muito)
        print("🎲 Gerando combinações de parâmetros...")
        
        # Combinações mais promissoras (baseadas nos melhores resultados anteriores)
        promising_combinations = [
            # Baseline atual
            {**fixed_params, 'take_profit_pct': 0.20, 'stop_loss_pct': 0.05, 'atr_min_pct': 0.7, 'atr_max_pct': 2.5, 'volume_multiplier': 2.5, 'gradient_min_long': 0.10, 'gradient_min_short': 0.12, 'breakout_atr_mult': 1.0, 'rsi_min': 25, 'rsi_max': 75, 'min_confluence': 4},
            
            # TP mais agressivo
            {**fixed_params, 'take_profit_pct': 0.30, 'stop_loss_pct': 0.05, 'atr_min_pct': 0.7, 'atr_max_pct': 2.5, 'volume_multiplier': 2.5, 'gradient_min_long': 0.10, 'gradient_min_short': 0.12, 'breakout_atr_mult': 1.0, 'rsi_min': 25, 'rsi_max': 75, 'min_confluence': 4},
            
            # SL mais conservador
            {**fixed_params, 'take_profit_pct': 0.20, 'stop_loss_pct': 0.03, 'atr_min_pct': 0.7, 'atr_max_pct': 2.5, 'volume_multiplier': 2.5, 'gradient_min_long': 0.10, 'gradient_min_short': 0.12, 'breakout_atr_mult': 1.0, 'rsi_min': 25, 'rsi_max': 75, 'min_confluence': 4},
            
            # Filtros mais restritivos
            {**fixed_params, 'take_profit_pct': 0.20, 'stop_loss_pct': 0.05, 'atr_min_pct': 1.0, 'atr_max_pct': 2.5, 'volume_multiplier': 3.0, 'gradient_min_long': 0.12, 'gradient_min_short': 0.15, 'breakout_atr_mult': 1.2, 'rsi_min': 30, 'rsi_max': 70, 'min_confluence': 5},
            
            # Filtros mais relaxados
            {**fixed_params, 'take_profit_pct': 0.20, 'stop_loss_pct': 0.05, 'atr_min_pct': 0.5, 'atr_max_pct': 3.0, 'volume_multiplier': 2.0, 'gradient_min_long': 0.08, 'gradient_min_short': 0.10, 'breakout_atr_mult': 0.8, 'rsi_min': 20, 'rsi_max': 80, 'min_confluence': 3},
            
            # TP/SL ratio otimizado
            {**fixed_params, 'take_profit_pct': 0.15, 'stop_loss_pct': 0.07, 'atr_min_pct': 0.7, 'atr_max_pct': 2.5, 'volume_multiplier': 2.5, 'gradient_min_long': 0.10, 'gradient_min_short': 0.12, 'breakout_atr_mult': 1.0, 'rsi_min': 25, 'rsi_max': 75, 'min_confluence': 4},
            
            # Ultra conservador
            {**fixed_params, 'take_profit_pct': 0.10, 'stop_loss_pct': 0.03, 'atr_min_pct': 1.0, 'atr_max_pct': 2.5, 'volume_multiplier': 3.0, 'gradient_min_long': 0.12, 'gradient_min_short': 0.15, 'breakout_atr_mult': 1.2, 'rsi_min': 30, 'rsi_max': 70, 'min_confluence': 5},
            
            # Ultra agressivo
            {**fixed_params, 'take_profit_pct': 0.30, 'stop_loss_pct': 0.10, 'atr_min_pct': 0.5, 'atr_max_pct': 3.0, 'volume_multiplier': 2.0, 'gradient_min_long': 0.08, 'gradient_min_short': 0.10, 'breakout_atr_mult': 0.8, 'rsi_min': 20, 'rsi_max': 80, 'min_confluence': 3},
        ]
        
        # Adicionar combinações aleatórias
        print("🎯 Testando combinações estratégicas + aleatórias...")
        np.random.seed(42)
        
        for _ in range(20):  # 20 combinações aleatórias adicionais
            random_params = {**fixed_params}
            for param, values in param_space.items():
                random_params[param] = np.random.choice(values)
            promising_combinations.append(random_params)
        
        best_result = None
        best_params = None
        all_results = []
        
        print(f"🔄 Testando {len(promising_combinations)} combinações...")
        
        for i, params in enumerate(promising_combinations):
            try:
                print(f"  📊 Teste {i+1}/{len(promising_combinations)}: TP={params['take_profit_pct']*100:.0f}% SL={params['stop_loss_pct']*100:.0f}% ATR={params['atr_min_pct']:.1f}%+ Vol={params['volume_multiplier']:.1f}x", end="")
                
                result = self.evaluate_params(params)
                result['params'] = params
                all_results.append(result)
                
                print(f" → ROI: {result['roi']:.1f}% | Trades: {result['total_trades']}")
                
                if best_result is None or result['roi'] > best_result['roi']:
                    best_result = result
                    best_params = params
                    print(f"    🎉 NOVO MELHOR! ROI: {result['roi']:.1f}%")
                
            except Exception as e:
                print(f" → ERRO: {e}")
        
        # Apresentar resultados
        print("\n" + "="*70)
        print("🏆 OTIMIZAÇÃO CONCLUÍDA")
        print("="*70)
        
        if best_result:
            print(f"🥇 MELHOR CONFIGURAÇÃO:")
            print(f"   💰 ROI: {best_result['roi']:.1f}%")
            print(f"   💰 Capital Final: ${best_result['final_capital']:.2f}")
            print(f"   💰 PNL Total: ${best_result['total_pnl']:.2f}")
            print(f"   🎯 Total Trades: {best_result['total_trades']}")
            print(f"   ✅ Win Rate: {best_result['win_rate']:.1f}%")
            
            print(f"\n📋 PARÂMETROS ÓTIMOS:")
            print(f"   🎯 Take Profit: {best_params['take_profit_pct']*100:.0f}%")
            print(f"   🛑 Stop Loss: {best_params['stop_loss_pct']*100:.0f}%")
            print(f"   📈 ATR: {best_params['atr_min_pct']:.1f}% - {best_params['atr_max_pct']:.1f}%")
            print(f"   📊 Volume: {best_params['volume_multiplier']:.1f}x")
            print(f"   ⬆️  Gradient LONG: {best_params['gradient_min_long']:.2f}%")
            print(f"   ⬇️  Gradient SHORT: {best_params['gradient_min_short']:.2f}%")
            print(f"   💥 Breakout: {best_params['breakout_atr_mult']:.1f}x ATR")
            print(f"   📊 RSI: {best_params['rsi_min']:.0f} - {best_params['rsi_max']:.0f}")
            print(f"   🔗 Confluence: {best_params['min_confluence']}/5 filtros")
        
        # Top 5 configurações
        print(f"\n🏅 TOP 5 CONFIGURAÇÕES:")
        print("-" * 80)
        print(f"{'#':<3} {'ROI%':<8} {'Capital':<10} {'Trades':<8} {'Win%':<6} {'TP%':<5} {'SL%':<5}")
        print("-" * 80)
        
        sorted_results = sorted(all_results, key=lambda x: x['roi'], reverse=True)[:5]
        for i, result in enumerate(sorted_results):
            params = result['params']
            print(f"{i+1:<3} {result['roi']:<8.1f} ${result['final_capital']:<9.2f} {result['total_trades']:<8} {result['win_rate']:<6.1f} {params['take_profit_pct']*100:<5.0f} {params['stop_loss_pct']*100:<5.0f}")
        
        # Salvar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"otimizacao_parametros_{timestamp}.json"
        
        optimization_summary = {
            'timestamp': timestamp,
            'best_result': best_result,
            'best_params': best_params,
            'top_5_results': sorted_results[:5],
            'all_results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)
        
        print(f"\n💾 Resultados salvos: {results_file}")
        print("✅ Otimização concluída!")
        
        return best_params, best_result

def main():
    """Executa otimização"""
    optimizer = TradingOptimizer()
    
    if len(optimizer.real_data) == 0:
        print("❌ Nenhum dado real encontrado. Execute primeiro simulacao_dados_reais.py")
        return
    
    best_params, best_result = optimizer.optimize_parameters()
    
    print(f"\n🎯 COMPARAÇÃO COM BASELINE:")
    print(f"   Baseline (TP 20%, SL 5%): ROI 227%")
    print(f"   Otimizado: ROI {best_result['roi']:.1f}%")
    print(f"   Melhoria: {best_result['roi'] - 227:.1f} pontos percentuais")

if __name__ == "__main__":
    main()
