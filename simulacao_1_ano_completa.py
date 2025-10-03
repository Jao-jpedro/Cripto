#!/usr/bin/env python3
"""
Simulação de 1 ano com tradingv4.py - Dados Sintéticos Realistas
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
sys.path.append('/Users/joaoreis/Documents/GitHub/Cripto')
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

def generate_realistic_crypto_data(asset_name, days=365, start_price=50000):
    """Gera dados realistas de criptomoeda para simulação"""
    
    # Configurações específicas por ativo
    asset_configs = {
        'BTC': {'volatility': 0.03, 'trend': 0.0002, 'volume_base': 1000000},
        'ETH': {'volatility': 0.04, 'trend': 0.0003, 'volume_base': 800000},
        'SOL': {'volatility': 0.06, 'trend': 0.0005, 'volume_base': 300000},
        'AVAX': {'volatility': 0.05, 'trend': 0.0004, 'volume_base': 200000},
        'LINK': {'volatility': 0.045, 'trend': 0.0002, 'volume_base': 150000},
        'ADA': {'volatility': 0.05, 'trend': 0.0001, 'volume_base': 400000},
        'DOGE': {'volatility': 0.08, 'trend': -0.0001, 'volume_base': 600000}
    }
    
    # Configuração padrão se o ativo não estiver na lista
    config = asset_configs.get(asset_name, {'volatility': 0.05, 'trend': 0.0002, 'volume_base': 300000})
    
    # Gerar timestamps (dados de 1 hora)
    hours = days * 24
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Gerar preços usando random walk com tendência
    np.random.seed(42 + hash(asset_name) % 1000)  # Seed baseado no nome do ativo
    
    # Movimento base
    returns = np.random.normal(config['trend'], config['volatility'], hours)
    
    # Adicionar ciclos e volatilidade clusters
    cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 0.01  # Ciclo semanal
    monthly_cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 30)) * 0.02  # Ciclo mensal
    
    # Clusters de volatilidade (períodos de alta e baixa volatilidade)
    volatility_regime = np.random.choice([0.5, 1.0, 1.5], hours, p=[0.6, 0.3, 0.1])
    returns = returns * volatility_regime
    
    returns = returns + cycle + monthly_cycle
    
    # Gerar preços
    prices = [start_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, prices[-1] * 0.8))  # Limitar quedas muito abruptas
    
    prices = prices[1:]  # Remover o preço inicial duplicado
    
    # Gerar high/low/open baseado no close
    highs = []
    lows = []
    opens = []
    
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close
        else:
            # Open próximo ao close anterior com pequena variação
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.002))
        
        # High e Low baseados na volatilidade do período
        daily_range = close * np.random.uniform(0.01, 0.05)  # 1-5% de range intraday
        high = max(open_price, close) + daily_range * np.random.uniform(0, 1)
        low = min(open_price, close) - daily_range * np.random.uniform(0, 1)
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
    
    # Gerar volume realista
    base_volume = config['volume_base']
    volume_variations = np.random.lognormal(0, 0.5, hours)  # Distribuição log-normal
    volumes = base_volume * volume_variations
    
    # Volume maior em movimentos de preço maiores
    price_changes = np.abs(np.diff([start_price] + prices))
    price_change_factor = 1 + (price_changes / np.mean(price_changes))
    volumes = volumes * price_change_factor
    
    # Criar DataFrame
    df = pd.DataFrame({
        'data': timestamps,
        'valor_fechamento': prices,
        'valor_maximo': highs,
        'valor_minimo': lows,
        'valor_abertura': opens,
        'volume': volumes,
        'volume_compra': volumes * np.random.uniform(0.4, 0.6, hours),
        'volume_venda': volumes * np.random.uniform(0.4, 0.6, hours)
    })
    
    return df

def apply_mega_filters_enhanced(df, config):
    """Aplica os filtros MEGA restritivos do tradingv4.py - versão aprimorada"""
    try:
        if len(df) < 50:
            return pd.DataFrame()
        
        # Garantir que o DataFrame tem as colunas corretas antes de calcular indicadores
        required_base_cols = ['valor_fechamento', 'volume', 'data']
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Colunas base não encontradas: {missing_cols}")
            return pd.DataFrame()
        
        # Calcular indicadores usando a função do tradingv4.py
        try:
            df_with_indicators = compute_indicators(df, config.params)
        except Exception as e:
            print(f"❌ Erro ao calcular indicadores: {e}")
            return pd.DataFrame()
        
        # Verificar se os indicadores foram criados
        required_indicator_cols = ['atr', 'atr_pct', 'ema_short', 'ema_long']
        missing_indicators = [col for col in required_indicator_cols if col not in df_with_indicators.columns]
        if missing_indicators:
            print(f"❌ Indicadores não criados: {missing_indicators}")
            return pd.DataFrame()
        
        # Filtros de qualidade de sinal
        conditions = []
        
        # 1. ATR Filter (0.7% mínimo) - usar a coluna atr_pct já calculada
        atr_filter = (
            (df_with_indicators['atr_pct'] >= config.params.atr_pct_min) & 
            (df_with_indicators['atr_pct'] <= config.params.atr_pct_max)
        )
        conditions.append(atr_filter)
        
        # 2. Volume Filter (2.5x média)
        vol_ma = df_with_indicators['volume'].rolling(window=config.params.vol_ma_period).mean()
        volume_filter = df_with_indicators['volume'] >= (vol_ma * config.volume_multiplier)
        conditions.append(volume_filter)
        
        # 3. Breakout Filter (1.0+ ATR) - usando o parâmetro correto
        ema_diff = abs(df_with_indicators['ema_short'] - df_with_indicators['ema_long'])
        breakout_threshold = config.params.breakout_k_atr * df_with_indicators['atr']
        breakout_filter = ema_diff >= breakout_threshold
        conditions.append(breakout_filter)
        
        # 4. Gradient Filter (se disponível)
        if 'ema_short_grad' in df_with_indicators.columns:
            # LONG: gradient >= 0.10%
            long_grad = df_with_indicators['ema_short_grad'] >= config.gradient_min_long
            # SHORT: gradient <= -0.12%
            short_grad = df_with_indicators['ema_short_grad'] <= -config.gradient_min_short
            gradient_filter = long_grad | short_grad
            conditions.append(gradient_filter)
        
        # 5. RSI Filter (evitar extremos) - se disponível
        if 'rsi' in df_with_indicators.columns:
            rsi_filter = (df_with_indicators['rsi'] > 25) & (df_with_indicators['rsi'] < 75)
            conditions.append(rsi_filter)
        
        # 6. MACD Confirmation - se disponível
        if all(col in df_with_indicators.columns for col in ['macd', 'macd_signal']):
            macd_diff = df_with_indicators['macd'] - df_with_indicators['macd_signal']
            macd_consistent = (
                (macd_diff > 0) & (macd_diff.shift(1) > 0) |  # MACD positivo consistente
                (macd_diff < 0) & (macd_diff.shift(1) < 0)    # MACD negativo consistente
            )
            conditions.append(macd_consistent)
        
        # Aplicar todos os filtros (confluence mínima)
        if len(conditions) == 0:
            return pd.DataFrame()
        
        # Calcular score de confluence
        confluence_score = pd.Series(0, index=df_with_indicators.index)
        for condition in conditions:
            # Limpar NaN values
            clean_condition = condition.fillna(False)
            confluence_score += clean_condition.astype(int)
        
        # Exigir pelo menos 60% dos filtros ativos para maior flexibilidade
        min_confluence = max(2, int(len(conditions) * 0.6))  # Pelo menos 2 filtros ou 60%
        final_filter = confluence_score >= min_confluence
        
        # Remover linhas com NaN nos indicadores principais
        valid_rows = (
            df_with_indicators['atr_pct'].notna() & 
            df_with_indicators['ema_short'].notna() & 
            df_with_indicators['ema_long'].notna() &
            df_with_indicators['atr'].notna()
        )
        final_filter = final_filter & valid_rows
        
        # Estatísticas dos filtros
        total_points = len(df_with_indicators)
        filtered_points = final_filter.sum()
        filter_rate = (filtered_points / total_points) * 100 if total_points > 0 else 0
        
        print(f"  📊 MEGA Confluence: {filtered_points}/{total_points} pontos ({filter_rate:.1f}%) - {len(conditions)} filtros")
        
        return df_with_indicators[final_filter].reset_index(drop=True) if filtered_points > 0 else pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Erro nos filtros MEGA: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def simulate_trading_with_management(df, asset_name, config):
    """Simula trading com gerenciamento de risco aprimorado"""
    print(f"\n🔄 Simulando {asset_name}...")
    
    if len(df) < 100:
        print(f"❌ Dados insuficientes para {asset_name}")
        return None
    
    print(f"  📈 Dados disponíveis: {len(df)} pontos")
    
    # Aplicar filtros MEGA
    df_filtered = apply_mega_filters_enhanced(df, config)
    if len(df_filtered) < 5:
        print(f"❌ Filtros muito restritivos para {asset_name} ({len(df_filtered)} sinais)")
        return None
    
    try:
        # Executar backtest
        results = backtest_ema_gradient(df_filtered, config.params)
        
        if not results or 'trades' not in results:
            print(f"❌ Backtest falhou para {asset_name}")
            return None
        
        trades = results['trades']
        if len(trades) == 0:
            print(f"⚠️  Nenhum trade executado para {asset_name}")
            return None
        
        # Aplicar gestão de risco personalizada
        enhanced_trades = apply_risk_management(trades, config)
        
        # Calcular métricas
        metrics = calculate_enhanced_metrics(enhanced_trades, config, df_filtered)
        metrics['asset'] = asset_name
        metrics['total_signals'] = len(df_filtered)
        metrics['total_data_points'] = len(df)
        metrics['filter_efficiency'] = (len(df_filtered) / len(df)) * 100
        
        print(f"  ✅ {len(enhanced_trades)} trades, PNL: ${metrics['total_pnl']:.2f}, Win Rate: {metrics['win_rate']:.1f}%")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro na simulação de {asset_name}: {e}")
        return None

def apply_risk_management(trades, config):
    """Aplica gestão de risco adicional aos trades"""
    if not trades or len(trades) == 0:
        return []
    
    enhanced_trades = []
    
    for trade in trades:
        # Converter para dict se necessário
        if hasattr(trade, 'to_dict'):
            trade_dict = trade.to_dict()
        elif isinstance(trade, dict):
            trade_dict = trade.copy()
        else:
            continue
        
        # Aplicar take profit de 10%
        if 'pnl_pct' in trade_dict:
            original_pnl = trade_dict['pnl_pct']
            
            # Limitar ganhos a 10% (take profit)
            if original_pnl > config.take_profit_pct * 100:
                trade_dict['pnl_pct'] = config.take_profit_pct * 100
                trade_dict['exit_reason'] = 'take_profit_10pct'
            
            # Limitar perdas a -5% (stop loss)
            elif original_pnl < -config.stop_loss_pct * 100:
                trade_dict['pnl_pct'] = -config.stop_loss_pct * 100
                trade_dict['exit_reason'] = 'stop_loss_5pct'
            
            # Emergency stop em -$0.05
            pnl_dollars = config.position_size * (trade_dict['pnl_pct'] / 100)
            if pnl_dollars <= config.emergency_stop:
                trade_dict['pnl_pct'] = (config.emergency_stop / config.position_size) * 100
                trade_dict['exit_reason'] = 'emergency_stop'
        
        enhanced_trades.append(trade_dict)
    
    return enhanced_trades

def calculate_enhanced_metrics(trades, config, signals_df):
    """Calcula métricas detalhadas e aprimoradas"""
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
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'total_return_pct': 0
        }
    
    # Calcular PNL em dólares para cada trade
    pnls = []
    for trade in trades:
        if 'pnl_pct' in trade:
            pnl_dollars = config.position_size * (trade['pnl_pct'] / 100)
        else:
            pnl_dollars = 0
        pnls.append(pnl_dollars)
    
    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    # Calcular drawdown
    cumulative_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(np.concatenate([[0], cumulative_pnl]))
    drawdown = cumulative_pnl - running_max[1:]
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # Sharpe ratio simplificado
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Anualizado
    else:
        sharpe_ratio = 0
    
    # Return total
    total_pnl = np.sum(pnls)
    total_return_pct = (total_pnl / config.initial_capital) * 100
    
    return {
        'total_trades': len(pnls),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': (len(wins) / len(pnls) * 100) if len(pnls) > 0 else 0,
        'total_pnl': total_pnl,
        'max_win': np.max(wins) if len(wins) > 0 else 0,
        'max_loss': np.min(losses) if len(losses) > 0 else 0,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'profit_factor': abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf'),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_return_pct': total_return_pct
    }

def run_comprehensive_simulation():
    """Executa simulação completa e abrangente de 1 ano"""
    print("🚀 SIMULAÇÃO COMPLETA DE 1 ANO - TRADINGV4 MEGA RESTRITIVO")
    print("=" * 70)
    
    # Configuração
    config = SimulationConfig()
    print(f"💰 Capital inicial: ${config.initial_capital}")
    print(f"📊 Tamanho da posição: ${config.position_size}")
    print(f"🎯 Take Profit: {config.take_profit_pct*100}%")
    print(f"🛑 Stop Loss: {config.stop_loss_pct*100}%")
    print(f"⚠️  Emergency Stop: ${config.emergency_stop}")
    print(f"📈 ATR mínimo: {config.params.atr_pct_min}%")
    print(f"📊 Volume multiplier: {config.volume_multiplier}x")
    
    # Assets para simulação
    crypto_assets = {
        'BTC': 95000,
        'ETH': 3500,
        'SOL': 180,
        'AVAX': 35,
        'LINK': 23,
        'ADA': 0.85,
        'DOGE': 0.35
    }
    
    print(f"\n📂 Gerando dados sintéticos para {len(crypto_assets)} ativos...")
    
    # Simular cada ativo
    all_results = []
    total_signals = 0
    total_data_points = 0
    
    for asset_name, start_price in crypto_assets.items():
        try:
            # Gerar dados sintéticos
            print(f"\n🎲 Gerando dados para {asset_name} (preço inicial: ${start_price})")
            df = generate_realistic_crypto_data(asset_name, days=365, start_price=start_price)
            
            # Simular trading
            result = simulate_trading_with_management(df, asset_name, config)
            
            if result:
                all_results.append(result)
                total_signals += result['total_signals']
                total_data_points += result['total_data_points']
                
        except Exception as e:
            print(f"❌ Erro geral em {asset_name}: {e}")
    
    if not all_results:
        print("❌ Nenhuma simulação bem-sucedida")
        return
    
    # Análise consolidada
    print("\n" + "="*70)
    print("📊 ANÁLISE CONSOLIDADA - RESULTADOS DE 1 ANO")
    print("="*70)
    
    # Métricas totais
    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    total_wins = sum(r['winning_trades'] for r in all_results)
    total_losses = sum(r['losing_trades'] for r in all_results)
    
    final_capital = config.initial_capital + total_pnl
    roi = (total_pnl / config.initial_capital) * 100
    
    # Performance principal
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
        
        # Estatísticas adicionais
        avg_trade_pnl = total_pnl / total_trades
        print(f"💱 PNL Médio por Trade: ${avg_trade_pnl:.3f}")
        
        # ROI anualizado
        print(f"📈 ROI Anualizado: {roi:.2f}%")
        
        # Eficiência dos filtros
        filter_efficiency = (total_signals / total_data_points) * 100 if total_data_points > 0 else 0
        print(f"🔍 Eficiência dos Filtros: {filter_efficiency:.2f}% ({total_signals}/{total_data_points} sinais)")
    
    # Análise por performance
    print("\n📋 RANKING POR PERFORMANCE:")
    print("-" * 85)
    print(f"{'Ativo':<8} {'Trades':<7} {'PNL($)':<9} {'ROI%':<7} {'Win%':<6} {'Sharpe':<7} {'Signals':<8}")
    print("-" * 85)
    
    for result in sorted(all_results, key=lambda x: x['total_pnl'], reverse=True):
        asset = result['asset']
        trades = result['total_trades']
        pnl = result['total_pnl']
        roi_asset = (pnl / config.position_size) * 100 if trades > 0 else 0
        win_rate = result['win_rate']
        sharpe = result.get('sharpe_ratio', 0)
        signals = result['total_signals']
        
        print(f"{asset:<8} {trades:<7} {pnl:<9.2f} {roi_asset:<7.1f} {win_rate:<6.1f} {sharpe:<7.2f} {signals:<8}")
    
    # Análise de risco
    print("\n🛡️  ANÁLISE DE RISCO:")
    print("-" * 50)
    
    max_drawdowns = [r.get('max_drawdown', 0) for r in all_results]
    total_max_drawdown = min(max_drawdowns) if max_drawdowns else 0
    
    print(f"📉 Drawdown Máximo Total: ${total_max_drawdown:.2f}")
    
    profit_factors = [r.get('profit_factor', 0) for r in all_results if r.get('profit_factor', 0) != float('inf')]
    if profit_factors:
        avg_profit_factor = np.mean(profit_factors)
        print(f"💹 Profit Factor Médio: {avg_profit_factor:.2f}")
    
    # Projeções
    print("\n🔮 PROJEÇÕES BASEADAS NA PERFORMANCE:")
    print("-" * 50)
    
    if roi > 0:
        monthly_roi = roi / 12
        projected_6_months = config.initial_capital * (1 + (monthly_roi * 6) / 100)
        projected_2_years = config.initial_capital * ((1 + roi / 100) ** 2)
        
        print(f"📊 ROI Mensal Médio: {monthly_roi:.2f}%")
        print(f"📈 Projeção 6 meses: ${projected_6_months:.2f}")
        print(f"📈 Projeção 2 anos: ${projected_2_years:.2f}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simulacao_1_ano_completa_{timestamp}.json"
    
    simulation_summary = {
        'metadata': {
            'timestamp': timestamp,
            'simulation_type': 'TradingV4_MEGA_Restrictive_1_Year',
            'total_assets': len(all_results),
            'data_source': 'synthetic_realistic'
        },
        'config': {
            'initial_capital': config.initial_capital,
            'position_size': config.position_size,
            'take_profit_pct': config.take_profit_pct,
            'stop_loss_pct': config.stop_loss_pct,
            'emergency_stop': config.emergency_stop,
            'atr_min_pct': config.params.atr_pct_min,
            'volume_multiplier': config.volume_multiplier,
            'filters_applied': 'MEGA_Restrictive_Confluence'
        },
        'results': {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'roi_percent': roi,
            'overall_win_rate': overall_win_rate if total_trades > 0 else 0,
            'total_signals_generated': total_signals,
            'filter_efficiency_pct': filter_efficiency,
            'max_drawdown': total_max_drawdown
        },
        'by_asset': all_results,
        'risk_analysis': {
            'max_drawdown_total': total_max_drawdown,
            'avg_profit_factor': np.mean(profit_factors) if profit_factors else 0,
            'sharpe_ratios': [r.get('sharpe_ratio', 0) for r in all_results]
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(simulation_summary, f, indent=2, default=str)
    
    print(f"\n💾 Resultados completos salvos em: {results_file}")
    print("\n✅ SIMULAÇÃO DE 1 ANO CONCLUÍDA COM SUCESSO!")
    
    return simulation_summary

if __name__ == "__main__":
    run_comprehensive_simulation()
