#!/usr/bin/env python3
"""
Otimização Avançada do trading.py para Maximizar ROI
Testa múltiplas combinações de parâmetros nos dados históricos de 1 ano
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import json
from typing import Dict, List, Tuple, Any
import concurrent.futures
from dataclasses import dataclass

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class OptimizationConfig:
    """Configuração para testes de otimização"""
    # Take Profit variations (% da margem)
    tp_values = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    
    # Stop Loss variations (% da margem)  
    sl_values = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    
    # ATR Range variations
    atr_min_values = [0.3, 0.5, 0.8, 1.0]
    atr_max_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    # Volume multiplier variations
    volume_mult_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # Confluência variations
    confluencia_values = [2, 3, 4, 5]
    
    # EMA periods variations
    ema_short_values = [5, 7, 10, 12]
    ema_long_values = [18, 21, 24, 30]
    
    # Breakout multiplier variations
    breakout_k_values = [0.5, 0.8, 1.0, 1.2, 1.5]

class TradingOptimizer:
    def __init__(self):
        self.data_files = [
            "dados_reais_btc_1ano.csv",
            "dados_reais_eth_1ano.csv", 
            "dados_reais_bnb_1ano.csv",
            "dados_reais_sol_1ano.csv",
            "dados_reais_ada_1ano.csv",
            "dados_reais_avax_1ano.csv",
            "dados_reais_doge_1ano.csv",
            "dados_reais_link_1ano.csv",
            "dados_reais_ltc_1ano.csv",
            "dados_reais_xrp_1ano.csv"
        ]
        self.available_data = []
        self._check_data_availability()
        
    def _check_data_availability(self):
        """Verifica quais arquivos de dados estão disponíveis"""
        for file in self.data_files:
            if os.path.exists(file):
                self.available_data.append(file)
        
        print(f"📊 Dados disponíveis: {len(self.available_data)}/{len(self.data_files)} assets")
        for file in self.available_data:
            asset = file.replace("dados_reais_", "").replace("_1ano.csv", "").upper()
            print(f"   ✅ {asset}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Carrega e prepara dados de um asset"""
        try:
            df = pd.read_csv(filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            elif 'data' in df.columns:
                df['timestamp'] = pd.to_datetime(df['data'])
                df = df.sort_values('timestamp')
            
            # Padronizar nomes das colunas
            column_mapping = {
                'open': 'valor_abertura',
                'high': 'valor_maximo', 
                'low': 'valor_minimo',
                'close': 'valor_fechamento',
                'volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # Verificar colunas essenciais
            required_cols = ['valor_abertura', 'valor_maximo', 'valor_minimo', 'valor_fechamento', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Colunas faltando em {filename}: {missing_cols}")
                return None
                
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Erro ao carregar {filename}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame, ema_short: int = 7, ema_long: int = 21, 
                           atr_period: int = 14, vol_period: int = 20) -> pd.DataFrame:
        """Calcula indicadores técnicos"""
        df = df.copy()
        
        # EMAs
        df['ema_short'] = df['valor_fechamento'].ewm(span=ema_short).mean()
        df['ema_long'] = df['valor_fechamento'].ewm(span=ema_long).mean()
        
        # ATR
        high_low = df['valor_maximo'] - df['valor_minimo']
        high_close = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
        low_close = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
        
        # Volume MA
        df['vol_ma'] = df['volume'].rolling(window=vol_period).mean()
        
        # RSI
        delta = df['valor_fechamento'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def check_confluencia(self, row, ema_cross_bullish: bool, ema_cross_bearish: bool,
                         atr_min: float, atr_max: float, volume_mult: float, 
                         breakout_k: float, min_confluencia: int) -> Tuple[bool, bool, int]:
        """Verifica critérios de confluência para entrada"""
        criterios_long = 0
        criterios_short = 0
        
        # 1. EMA Cross
        if ema_cross_bullish:
            criterios_long += 1
        if ema_cross_bearish:
            criterios_short += 1
            
        # 2. ATR Range
        atr_ok = atr_min <= row['atr_pct'] <= atr_max
        if atr_ok:
            criterios_long += 1
            criterios_short += 1
            
        # 3. Volume
        volume_ok = row['volume'] > (row['vol_ma'] * volume_mult)
        if volume_ok:
            criterios_long += 1
            criterios_short += 1
            
        # 4. Breakout (simplificado)
        if ema_cross_bullish:
            breakout_long = row['valor_fechamento'] > (row['ema_short'] + breakout_k * row['atr'])
            if breakout_long:
                criterios_long += 1
                
        if ema_cross_bearish:
            breakout_short = row['valor_fechamento'] < (row['ema_short'] - breakout_k * row['atr'])
            if breakout_short:
                criterios_short += 1
        
        # 5. RSI Force
        if row['rsi'] < 20:  # Oversold - Force LONG
            criterios_long += 2
        elif row['rsi'] > 80:  # Overbought - Force SHORT
            criterios_short += 2
            
        can_long = criterios_long >= min_confluencia
        can_short = criterios_short >= min_confluencia
        
        return can_long, can_short, max(criterios_long, criterios_short)
    
    def simulate_trading(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Simula trading com uma configuração específica"""
        
        # Aplicar indicadores
        df = self.calculate_indicators(
            df, 
            ema_short=config['ema_short'], 
            ema_long=config['ema_long']
        )
        
        balance = 1000.0  # Capital inicial
        trades = []
        position = None
        leverage = 20
        
        for i in range(config['ema_long'], len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Detectar EMA Cross
            ema_cross_bullish = (prev['ema_short'] <= prev['ema_long'] and 
                               current['ema_short'] > current['ema_long'])
            ema_cross_bearish = (prev['ema_short'] >= prev['ema_long'] and 
                               current['ema_short'] < current['ema_long'])
            
            # Se já tem posição, verificar saída
            if position:
                entry_price = position['entry_price']
                side = position['side']
                entry_balance = position['entry_balance']
                
                # Calcular P&L atual
                if side == 'long':
                    pnl_pct = ((current['valor_fechamento'] - entry_price) / entry_price) * leverage
                else:  # short
                    pnl_pct = ((entry_price - current['valor_fechamento']) / entry_price) * leverage
                
                # Verificar TP/SL
                tp_pct = config['tp_pct'] / 100.0
                sl_pct = config['sl_pct'] / 100.0
                
                exit_trade = False
                exit_reason = ""
                
                if pnl_pct >= tp_pct:
                    exit_trade = True
                    exit_reason = "TP"
                elif pnl_pct <= -sl_pct:
                    exit_trade = True
                    exit_reason = "SL"
                
                if exit_trade:
                    final_balance = entry_balance * (1 + pnl_pct)
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current['valor_fechamento'],
                        'side': side,
                        'pnl_pct': pnl_pct,
                        'reason': exit_reason,
                        'balance_before': entry_balance,
                        'balance_after': final_balance
                    })
                    
                    balance = final_balance
                    position = None
                    continue
            
            # Se não tem posição, verificar entrada
            if not position:
                can_long, can_short, criterios = self.check_confluencia(
                    current, ema_cross_bullish, ema_cross_bearish,
                    config['atr_min'], config['atr_max'], config['volume_mult'],
                    config['breakout_k'], config['min_confluencia']
                )
                
                if can_long:
                    position = {
                        'side': 'long',
                        'entry_price': current['valor_fechamento'],
                        'entry_balance': balance
                    }
                elif can_short:
                    position = {
                        'side': 'short', 
                        'entry_price': current['valor_fechamento'],
                        'entry_balance': balance
                    }
        
        # Calcular métricas
        if not trades:
            return {
                'final_balance': balance,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0
            }
        
        total_return_pct = ((balance - 1000.0) / 1000.0) * 100
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        
        # Calcular drawdown
        balances = [1000.0]
        for trade in trades:
            balances.append(trade['balance_after'])
        
        peak = balances[0]
        max_dd = 0
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'final_balance': balance,
            'total_return_pct': total_return_pct,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd * 100
        }
    
    def test_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Testa uma configuração em todos os assets disponíveis"""
        results = []
        
        for data_file in self.available_data:
            df = self.load_data(data_file)
            if df is None or len(df) < 100:
                continue
                
            asset = data_file.replace("dados_reais_", "").replace("_1ano.csv", "").upper()
            result = self.simulate_trading(df, config)
            result['asset'] = asset
            results.append(result)
        
        if not results:
            return None
            
        # Agregar resultados
        total_return = np.mean([r['total_return_pct'] for r in results])
        total_trades = sum([r['num_trades'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results if r['num_trades'] > 0])
        avg_max_dd = np.mean([r['max_drawdown'] for r in results])
        
        return {
            'config': config,
            'avg_return_pct': total_return,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'avg_max_drawdown': avg_max_dd,
            'num_assets': len(results),
            'asset_results': results
        }
    
    def optimize_parameters(self, max_combinations: int = 200, top_results: int = 10):
        """Otimiza parâmetros testando diferentes combinações"""
        
        print("🚀 INICIANDO OTIMIZAÇÃO AVANÇADA DO TRADING.PY")
        print("=" * 60)
        print(f"📊 Assets disponíveis: {len(self.available_data)}")
        print(f"🎯 Máximo de combinações: {max_combinations}")
        print(f"🏆 Top resultados: {top_results}")
        print()
        
        opt_config = OptimizationConfig()
        
        # Gerar combinações de parâmetros
        all_combinations = list(itertools.product(
            opt_config.tp_values,
            opt_config.sl_values,
            opt_config.atr_min_values,
            opt_config.atr_max_values,
            opt_config.volume_mult_values,
            opt_config.confluencia_values,
            opt_config.ema_short_values,
            opt_config.ema_long_values,
            opt_config.breakout_k_values
        ))
        
        # Filtrar combinações válidas
        valid_combinations = []
        for combo in all_combinations:
            tp, sl, atr_min, atr_max, vol_mult, confluencia, ema_s, ema_l, breakout_k = combo
            
            # Filtros de validade
            if (tp > sl and  # TP > SL
                atr_min < atr_max and  # ATR range válido
                ema_s < ema_l and  # EMA válido
                tp <= 50 and sl >= 5):  # Limites razoáveis
                valid_combinations.append(combo)
        
        print(f"📋 Combinações válidas: {len(valid_combinations)}")
        
        # Limitar combinações se necessário
        if len(valid_combinations) > max_combinations:
            # Usar amostragem estratificada
            step = len(valid_combinations) // max_combinations
            valid_combinations = valid_combinations[::step][:max_combinations]
            print(f"📋 Combinações selecionadas: {len(valid_combinations)}")
        
        # Testar combinações
        results = []
        
        print("\n🧪 TESTANDO COMBINAÇÕES...")
        print("-" * 60)
        
        for i, combo in enumerate(valid_combinations):
            tp, sl, atr_min, atr_max, vol_mult, confluencia, ema_s, ema_l, breakout_k = combo
            
            config = {
                'tp_pct': tp,
                'sl_pct': sl,
                'atr_min': atr_min,
                'atr_max': atr_max,
                'volume_mult': vol_mult,
                'min_confluencia': confluencia,
                'ema_short': ema_s,
                'ema_long': ema_l,
                'breakout_k': breakout_k
            }
            
            result = self.test_configuration(config)
            if result and result['total_trades'] > 10:  # Mínimo de trades
                results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"   ✅ Testadas: {i+1}/{len(valid_combinations)} | Válidas: {len(results)}")
        
        if not results:
            print("❌ Nenhum resultado válido encontrado!")
            return
        
        # Ordenar por retorno médio
        results.sort(key=lambda x: x['avg_return_pct'], reverse=True)
        
        # Salvar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"otimizacao_trading_py_{timestamp}.json"
        
        output_data = {
            'timestamp': timestamp,
            'total_combinations_tested': len(results),
            'assets_used': len(self.available_data),
            'top_results': results[:top_results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Mostrar resultados
        self.display_results(results[:top_results], filename)
    
    def display_results(self, results: List[Dict], filename: str):
        """Exibe os melhores resultados"""
        
        print("\n" + "="*80)
        print("🏆 TOP CONFIGURAÇÕES PARA MÁXIMO ROI - TRADING.PY")
        print("="*80)
        
        for i, result in enumerate(results[:10], 1):
            config = result['config']
            
            print(f"\n🥇 #{i} - ROI: {result['avg_return_pct']:.1f}%")
            print("-" * 50)
            print(f"📊 TP: {config['tp_pct']}% | SL: {config['sl_pct']}% | R:R = {config['tp_pct']/config['sl_pct']:.1f}")
            print(f"📈 ATR: {config['atr_min']}-{config['atr_max']}% | Volume: {config['volume_mult']}x")
            print(f"🎯 EMA: {config['ema_short']}/{config['ema_long']} | Confluência: {config['min_confluencia']}")
            print(f"⚡ Breakout K: {config['breakout_k']} | Win Rate: {result['avg_win_rate']:.1f}%")
            print(f"📉 Max DD: {result['avg_max_drawdown']:.1f}% | Trades: {result['total_trades']}")
            
            # Mostrar performance por asset
            print(f"💰 Performance por Asset:")
            for asset_result in result['asset_results']:
                asset = asset_result['asset']
                roi = asset_result['total_return_pct']
                trades = asset_result['num_trades']
                wr = asset_result['win_rate'] * 100
                print(f"   {asset}: {roi:+.1f}% ({trades} trades, {wr:.1f}% WR)")
        
        print(f"\n💾 Resultados salvos em: {filename}")
        print("\n🎯 RECOMENDAÇÃO PARA IMPLEMENTAÇÃO:")
        
        best = results[0]
        best_config = best['config']
        
        print(f"""
📋 CONFIGURAÇÃO ÓTIMA:
   TP_PCT = {best_config['tp_pct']}
   SL_PCT = {best_config['sl_pct']} 
   ATR_PCT_MIN = {best_config['atr_min']}
   ATR_PCT_MAX = {best_config['atr_max']}
   VOLUME_MULTIPLIER = {best_config['volume_mult']}
   MIN_CONFLUENCIA = {best_config['min_confluencia']}
   EMA_SHORT_SPAN = {best_config['ema_short']}
   EMA_LONG_SPAN = {best_config['ema_long']}
   BREAKOUT_K_ATR = {best_config['breakout_k']}

💡 ROI ESPERADO: {best['avg_return_pct']:.1f}% anual
🎯 Risk/Reward: {best_config['tp_pct']/best_config['sl_pct']:.1f}:1
📊 Win Rate Médio: {best['avg_win_rate']:.1f}%
📉 Drawdown Máximo: {best['avg_max_drawdown']:.1f}%
""")

def main():
    optimizer = TradingOptimizer()
    
    if len(optimizer.available_data) == 0:
        print("❌ Nenhum dado histórico encontrado!")
        print("🔍 Certifique-se de ter os arquivos dados_reais_*_1ano.csv")
        return
    
    print("🎯 Iniciando otimização para MAXIMIZAR ROI do trading.py")
    print("⏱️ Este processo pode levar alguns minutos...")
    print()
    
    optimizer.optimize_parameters(max_combinations=300, top_results=15)
    
    print("\n🎉 OTIMIZAÇÃO CONCLUÍDA!")
    print("💡 Use a configuração recomendada para atualizar trading.py")

if __name__ == "__main__":
    main()
