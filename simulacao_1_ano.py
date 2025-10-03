#!/usr/bin/env python3
"""
Simulação de 1 ano com tradingv4.py
Análise de performance macro com $10 iniciais e entradas de $1
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Importar as funções do tradingv4.py
from tradingv4 import BacktestParams, backtest_ema_gradient, _ensure_base_cols, compute_indicators

class SimulationConfig:
    """Configuração específica para simulação de 1 ano"""
    
    def __init__(self):
        # Parâmetros financeiros
        self.initial_capital = 10.0  # $10 inicial
        self.position_size = 1.0     # $1 por entrada
        
        # Configuração do tradingv4.py com filtros MEGA restritivos
        self.params = BacktestParams()
        self.params.atr_pct_min = 0.7     # ATR mínimo do tradingv4
        self.params.atr_pct_max = 2.5     # ATR máximo
        self.params.breakout_k_atr = 1.0  # Breakout 1.0+ ATR
        self.params.cooldown_bars = 3
        self.params.post_cooldown_confirm_bars = 1
        
        # Take profit e stop loss conforme tradingv4.py
        self.take_profit_pct = 0.10   # 10% take profit
        self.stop_loss_pct = 0.05     # 5% stop loss
        self.emergency_stop = -0.05   # -$0.05 emergency stop
        
        # Volume mínimo (2.5x média do tradingv4)
        self.volume_multiplier = 2.5
        
        # Gradient mínimo conforme MEGA filters
        self.gradient_min_long = 0.10   # 0.10% para LONG
        self.gradient_min_short = 0.12  # 0.12% para SHORT

def load_historical_data():
    """Carrega dados históricos de múltiplos ativos"""
    assets = [
        'btc_usd', 'eth_usd', 'sol_usd', 'avax_usd', 'link_usd',
        'ada_usd', 'doge_usd', 'xrp_usd', 'bnb_usd', 'ltc_usd',
        'aave_usd', 'crv_usd', 'near_usd', 'sui_usd', 'wld_usd',
        'ena_usd', 'avnt_usd', 'pump_usd', 'hype_usd'
    ]
    
    all_data = {}
    base_path = '/Users/joaoreis/Documents/GitHub/Cripto'
    
    for asset in assets:
        file_path = os.path.join(base_path, f'trade_log_{asset}.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) > 1000:  # Só usar dados com pelo menos 1000 pontos
                    print(f"✅ Carregado {asset}: {len(df)} registros")
                    all_data[asset] = df
                else:
                    print(f"⚠️  {asset}: apenas {len(df)} registros (pulando)")
            except Exception as e:
                print(f"❌ Erro ao carregar {asset}: {e}")
        else:
            print(f"❌ Arquivo não encontrado: {file_path}")
    
    return all_data

def prepare_data_for_backtest(df):
    """Prepara dados para o formato esperado pelo backtest"""
    try:
        # Garantir que temos as colunas necessárias
        if 'timestamp' in df.columns:
            df['data'] = pd.to_datetime(df['timestamp'])
        elif 'data' not in df.columns:
            print("❌ Coluna de data não encontrada")
            return None
            
        # Mapear colunas de preço
        price_cols = ['close', 'valor_fechamento', 'price']
        price_col = None
        for col in price_cols:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            print("❌ Coluna de preço não encontrada")
            return None
            
        # Criar dataframe padronizado
        df_clean = df.copy()
        df_clean['valor_fechamento'] = pd.to_numeric(df_clean[price_col], errors='coerce')
        
        # Volume (se existir)
        if 'volume' in df.columns:
            df_clean['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
        else:
            df_clean['volume'] = 0
            
        # High e Low (se existirem)
        if 'high' in df.columns:
            df_clean['valor_maximo'] = pd.to_numeric(df['high'], errors='coerce')
        if 'low' in df.columns:
            df_clean['valor_minimo'] = pd.to_numeric(df['low'], errors='coerce')
            
        # Filtrar último ano se tivermos dados suficientes
        df_clean = df_clean.sort_values('data').reset_index(drop=True)
        if len(df_clean) > 8760:  # Mais de 1 ano de dados horários
            df_clean = df_clean.tail(8760).reset_index(drop=True)
            
        # Remover NaN
        df_clean = df_clean.dropna(subset=['valor_fechamento']).reset_index(drop=True)
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Erro ao preparar dados: {e}")
        return None

def apply_mega_filters(df, config):
    """Aplica os filtros MEGA restritivos do tradingv4.py"""
    try:
        # Calcular indicadores
        df = compute_indicators(df, config.params)
        
        # Filtros de qualidade de sinal
        conditions = []
        
        # 1. ATR Filter (0.7% mínimo)
        atr_pct = (df['atr'] / df['valor_fechamento']) * 100
        conditions.append(atr_pct >= config.params.atr_pct_min)
        
        # 2. Volume Filter (2.5x média)
        vol_ma = df['volume'].rolling(window=config.params.vol_ma_period).mean()
        conditions.append(df['volume'] >= (vol_ma * config.volume_multiplier))
        
        # 3. Gradient Filter
        if 'ema_short_grad' in df.columns:
            # LONG: gradient >= 0.10%
            long_grad = df['ema_short_grad'] >= config.gradient_min_long
            # SHORT: gradient <= -0.12%
            short_grad = df['ema_short_grad'] <= -config.gradient_min_short
            conditions.append(long_grad | short_grad)
        
        # 4. Breakout Filter (1.0+ ATR)
        if all(col in df.columns for col in ['ema_short', 'ema_long']):
            ema_diff = abs(df['ema_short'] - df['ema_long'])
            breakout_threshold = config.params.breakout_k_atr * df['atr']
            conditions.append(ema_diff >= breakout_threshold)
        
        # Aplicar todos os filtros
        final_filter = pd.Series(True, index=df.index)
        for condition in conditions:
            if len(condition) == len(df):
                final_filter &= condition
        
        # Estatísticas dos filtros
        total_points = len(df)
        filtered_points = final_filter.sum()
        filter_rate = (filtered_points / total_points) * 100 if total_points > 0 else 0
        
        print(f"  📊 Filtros MEGA: {filtered_points}/{total_points} pontos ({filter_rate:.1f}%)")
        
        return df[final_filter].reset_index(drop=True) if filtered_points > 0 else pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Erro nos filtros MEGA: {e}")
        return pd.DataFrame()

def simulate_trading(df, asset_name, config):
    """Simula trading para um ativo específico"""
    print(f"\n🔄 Simulando {asset_name}...")
    
    # Preparar dados
    df_clean = prepare_data_for_backtest(df)
    if df_clean is None or len(df_clean) < 100:
        print(f"❌ Dados insuficientes para {asset_name}")
        return None
    
    print(f"  📈 Dados limpos: {len(df_clean)} pontos")
    
    # Aplicar filtros MEGA
    df_filtered = apply_mega_filters(df_clean, config)
    if len(df_filtered) < 10:
        print(f"❌ Filtros muito restritivos para {asset_name} ({len(df_filtered)} pontos)")
        return None
    
    try:
        # Executar backtest
        results = backtest_ema_gradient(df_filtered, config.params)
        
        if not results or 'trades' not in results:
            print(f"❌ Backtest falhou para {asset_name}")
            return None
        
        trades = results['trades']
        if len(trades) == 0:
            print(f"⚠️  Nenhum trade gerado para {asset_name}")
            return None
        
        # Calcular métricas
        metrics = calculate_trade_metrics(trades, config)
        metrics['asset'] = asset_name
        metrics['total_signals'] = len(df_filtered)
        metrics['total_data_points'] = len(df_clean)
        
        print(f"  ✅ {len(trades)} trades, PNL: ${metrics['total_pnl']:.2f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro na simulação de {asset_name}: {e}")
        return None

def calculate_trade_metrics(trades, config):
    """Calcula métricas detalhadas dos trades"""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'max_win': 0,
            'max_loss': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
    
    # Converter trades para DataFrame se necessário
    if isinstance(trades, list):
        df_trades = pd.DataFrame(trades)
    else:
        df_trades = trades.copy()
    
    # Calcular PNL de cada trade baseado na posição de $1
    pnls = []
    for _, trade in df_trades.iterrows():
        if 'pnl_pct' in trade:
            pnl = config.position_size * (trade['pnl_pct'] / 100)
        elif 'exit_price' in trade and 'entry_price' in trade:
            if trade.get('side', '').upper() == 'LONG':
                pnl_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
            else:  # SHORT
                pnl_pct = ((trade['entry_price'] - trade['exit_price']) / trade['entry_price']) * 100
            pnl = config.position_size * (pnl_pct / 100)
        else:
            pnl = 0
        pnls.append(pnl)
    
    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    # Calcular drawdown
    cumulative_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    return {
        'total_trades': len(pnls),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': (len(wins) / len(pnls) * 100) if len(pnls) > 0 else 0,
        'total_pnl': np.sum(pnls),
        'max_win': np.max(wins) if len(wins) > 0 else 0,
        'max_loss': np.min(losses) if len(losses) > 0 else 0,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'profit_factor': abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf'),
        'max_drawdown': max_drawdown
    }

def run_simulation():
    """Executa a simulação completa de 1 ano"""
    print("🚀 Iniciando Simulação de 1 Ano - TradingV4 MEGA Restritivo")
    print("=" * 60)
    
    # Configuração
    config = SimulationConfig()
    print(f"💰 Capital inicial: ${config.initial_capital}")
    print(f"📊 Tamanho da posição: ${config.position_size}")
    print(f"🎯 Take Profit: {config.take_profit_pct*100}%")
    print(f"🛑 Stop Loss: {config.stop_loss_pct*100}%")
    print(f"⚠️  Emergency Stop: ${config.emergency_stop}")
    
    # Carregar dados
    print("\n📂 Carregando dados históricos...")
    all_data = load_historical_data()
    
    if not all_data:
        print("❌ Nenhum dado carregado. Verificar arquivos CSV.")
        return
    
    print(f"✅ {len(all_data)} ativos carregados")
    
    # Simular cada ativo
    all_results = []
    
    for asset_name, df in all_data.items():
        try:
            result = simulate_trading(df, asset_name, config)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Erro geral em {asset_name}: {e}")
    
    if not all_results:
        print("❌ Nenhuma simulação bem-sucedida")
        return
    
    # Consolidar resultados
    print("\n" + "="*60)
    print("📊 RESULTADOS CONSOLIDADOS - SIMULAÇÃO 1 ANO")
    print("="*60)
    
    # Métricas totais
    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    total_wins = sum(r['winning_trades'] for r in all_results)
    total_losses = sum(r['losing_trades'] for r in all_results)
    
    final_capital = config.initial_capital + total_pnl
    roi = (total_pnl / config.initial_capital) * 100
    
    print(f"💰 Capital Inicial: ${config.initial_capital:.2f}")
    print(f"💰 Capital Final: ${final_capital:.2f}")
    print(f"📈 PNL Total: ${total_pnl:.2f}")
    print(f"📊 ROI Total: {roi:.2f}%")
    print(f"🎯 Total de Trades: {total_trades}")
    print(f"✅ Trades Vencedores: {total_wins}")
    print(f"❌ Trades Perdedores: {total_losses}")
    
    if total_trades > 0:
        overall_win_rate = (total_wins / total_trades) * 100
        print(f"📊 Win Rate Geral: {overall_win_rate:.1f}%")
    
    # Detalhes por ativo
    print("\n📋 DETALHES POR ATIVO:")
    print("-" * 80)
    print(f"{'Ativo':<12} {'Trades':<7} {'PNL($)':<8} {'Win%':<6} {'Signals':<8} {'Filter%':<8}")
    print("-" * 80)
    
    for result in sorted(all_results, key=lambda x: x['total_pnl'], reverse=True):
        asset = result['asset'].replace('_usd', '').upper()
        trades = result['total_trades']
        pnl = result['total_pnl']
        win_rate = result['win_rate']
        signals = result['total_signals']
        filter_eff = (signals / result['total_data_points'] * 100) if result['total_data_points'] > 0 else 0
        
        print(f"{asset:<12} {trades:<7} {pnl:<8.2f} {win_rate:<6.1f} {signals:<8} {filter_eff:<8.1f}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simulacao_1_ano_results_{timestamp}.json"
    
    simulation_summary = {
        'timestamp': timestamp,
        'config': {
            'initial_capital': config.initial_capital,
            'position_size': config.position_size,
            'take_profit_pct': config.take_profit_pct,
            'stop_loss_pct': config.stop_loss_pct,
            'emergency_stop': config.emergency_stop
        },
        'results': {
            'total_assets': len(all_results),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'roi_percent': roi,
            'overall_win_rate': overall_win_rate if total_trades > 0 else 0
        },
        'by_asset': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(simulation_summary, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {results_file}")
    print("\n✅ Simulação concluída!")
    
    return simulation_summary

if __name__ == "__main__":
    run_simulation()
