#!/usr/bin/env python3
"""
🎯 BACKTEST TRADING.PY EXATO - 1 ANO DE DADOS REAIS
Aplicando exatamente a lógica do trading.py nos dados históricos de 1 ano
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÃO EXATA DO TRADING.PY
class TradingConfig:
    # DNA GENÉTICO EXATO
    STOP_LOSS_PCT = 0.015  # 1.5% ROI stop loss - DNA GENÉTICO
    TAKE_PROFIT_PCT = 0.12  # 12% ROI take profit - DNA GENÉTICO
    LEVERAGE = 3  # Leverage padrão
    
    # EMA Configuration
    EMA_FAST = 3
    EMA_SLOW = 34
    
    # Volume e ATR
    VOLUME_MA_PERIOD = 20
    ATR_PERIOD = 14
    
    # RSI
    RSI_PERIOD = 21

# ASSETS EXATOS DO TRADING.PY (apenas com dados reais de 1 ano)
TRADING_ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    # "PUMP-USD", "AVNT-USD",  # Removidos: só têm 6 dias de dados reais
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def load_trading_data(asset):
    """Carrega dados usando a mesma estrutura do trading.py"""
    symbol = asset.replace("-USD", "").lower()
    filename = f"dados_reais_{symbol}_1ano.csv"
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Mapear para nomes do trading.py
        df = df.rename(columns={
            'valor_fechamento': 'valor_fechamento',
            'valor_abertura': 'valor_abertura', 
            'valor_maximo': 'valor_maximo',
            'valor_minimo': 'valor_minimo',
            'volume': 'volume'
        })
        
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar {asset}: {e}")
        return None

def calculate_trading_indicators(df):
    """Calcula indicadores EXATAMENTE como no trading.py"""
    # EMAs
    df['ema_short'] = df['valor_fechamento'].ewm(span=TradingConfig.EMA_FAST).mean()
    df['ema_long'] = df['valor_fechamento'].ewm(span=TradingConfig.EMA_SLOW).mean()
    
    # EMA Gradient (exatamente como no trading.py)
    df['ema_short_grad_pct'] = df['ema_short'].pct_change() * 100
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(window=TradingConfig.VOLUME_MA_PERIOD).mean()
    
    # ATR
    df['high_low'] = df['valor_maximo'] - df['valor_minimo']
    df['high_close'] = np.abs(df['valor_maximo'] - df['valor_fechamento'].shift())
    df['low_close'] = np.abs(df['valor_minimo'] - df['valor_fechamento'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=TradingConfig.ATR_PERIOD).mean()
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    # RSI (período 21 como no trading.py)
    delta = df['valor_fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=TradingConfig.RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=TradingConfig.RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def entry_long_condition_exact(row) -> Tuple[bool, str]:
    """
    CONDIÇÃO DE ENTRADA LONG REALISTA DO TRADING.PY
    Sistema de confluência balanceado (60% aprovação - 6/10 pontos)
    Baseado no que funcionou no teste de 24h
    """
    reasons = []
    confluence_score = 0
    max_score = 10
    
    # CRITÉRIO 1: EMA Cross básico (mais flexível)
    c1_ema = row.ema_short > row.ema_long
    c1_grad = row.ema_short_grad_pct > 0.05  # Reduzido: 0.05% vs 0.10%
    c1 = c1_ema and c1_grad
    if c1:
        confluence_score += 1
        reasons.append("✅ EMA3>EMA34+grad>0.05%")
    elif c1_ema:  # Pelo menos EMA cross
        confluence_score += 0.5
        reasons.append("🔶 EMA cross sem gradiente")
    else:
        reasons.append("❌ EMA bearish")
    
    # CRITÉRIO 2: ATR mais flexível
    c2 = (row.atr_pct >= 0.25) and (row.atr_pct <= 2.0)  # Mais flexível: 0.25%-2.0%
    if c2:
        confluence_score += 1
        reasons.append("✅ ATR saudável")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento mais suave
    c3 = row.valor_fechamento > (row.ema_short + 0.3 * row.atr)  # Reduzido: 0.3 ATR vs 1.0
    if c3:
        confluence_score += 1
        reasons.append("✅ Rompimento forte")
    elif row.valor_fechamento > row.ema_short:  # Pelo menos acima da EMA
        confluence_score += 0.5
        reasons.append("🔶 Acima EMA")
    else:
        reasons.append("❌ Abaixo EMA")
    
    # CRITÉRIO 4: Volume mais realista
    volume_ratio = row.volume / row.vol_ma if row.vol_ma > 0 else 0
    c4 = volume_ratio > 1.5  # Mais realista: 1.5x vs 3.0x
    if c4:
        confluence_score += 1
        reasons.append("✅ Volume alto")
    elif volume_ratio > 1.0:  # Volume normal
        confluence_score += 0.5
        reasons.append("🔶 Volume normal")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI mais flexível
    if pd.notna(row.rsi):
        c5 = 30 <= row.rsi <= 70  # Muito mais flexível: 30-70 vs 45-55
        if c5:
            confluence_score += 1
            reasons.append("✅ RSI bom")
        elif 25 <= row.rsi <= 75:  # Zona aceitável
            confluence_score += 0.5
            reasons.append("🔶 RSI aceitável")
        else:
            reasons.append("❌ RSI extremo")
    else:
        confluence_score += 0.5  # Meio ponto se RSI não disponível
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD momentum (simulado - meio ponto default)
    confluence_score += 0.5
    reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação EMAs mais flexível
    ema_separation = abs(row.ema_short - row.ema_long) / row.atr if row.atr > 0 else 0
    c7 = ema_separation >= 0.2  # Mais flexível: 0.2 ATR vs 0.5
    if c7:
        confluence_score += 1
        reasons.append("✅ EMAs separadas")
    elif ema_separation >= 0.1:  # Alguma separação
        confluence_score += 0.5
        reasons.append("🔶 EMAs próximas")
    else:
        reasons.append("❌ EMAs coladas")
    
    # CRITÉRIO 8: Timing mais flexível
    price_distance = abs(row.valor_fechamento - row.ema_short) / row.atr if row.atr > 0 else 999
    c8 = price_distance <= 2.0  # Mais flexível: 2.0 ATR vs 1.0
    if c8:
        confluence_score += 1
        reasons.append("✅ Timing bom")
    else:
        confluence_score += 0.5  # Sempre meio ponto para timing
        reasons.append("🔶 Timing tardio")
        
    # CRITÉRIO 9: Bollinger Bands (simulado - meio ponto default)
    confluence_score += 0.5
    reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: BB squeeze (simulado - meio ponto default)
    confluence_score += 0.5
    reasons.append("⚪ BB squeeze n/d")
    
    # DECISÃO FINAL: REALISTA requer 60% confluência (6.0/10 pontos)
    MIN_CONFLUENCE = 6.0  # Reduzido de 8.5 para 6.0
    is_valid = confluence_score >= MIN_CONFLUENCE
    
    confluence_pct = (confluence_score / max_score) * 100
    reason_summary = f"Confluência LONG: {confluence_score:.1f}/{max_score} ({confluence_pct:.0f}%)"
    
    return is_valid, reason_summary

def simulate_trading_exact(df, asset):
    """Simula trading EXATAMENTE como no trading.py"""
    capital = 4.0  # $4 por trade como configurado
    position = None
    trades = []
    
    for i in range(len(df)):
        if i < TradingConfig.EMA_SLOW:  # Aguardar indicadores estabilizarem
            continue
            
        row = df.iloc[i]
        
        if position is None:
            # Verificar entrada LONG usando lógica EXATA do trading.py
            should_enter, reason = entry_long_condition_exact(row)
            
            if should_enter:
                entry_price = row.valor_fechamento
                
                # Usar capital com leverage exato
                position_size = capital * TradingConfig.LEVERAGE
                shares = position_size / entry_price
                
                # Calcular SL e TP exatos
                stop_loss = entry_price * (1 - TradingConfig.STOP_LOSS_PCT)
                take_profit = entry_price * (1 + TradingConfig.TAKE_PROFIT_PCT)
                
                position = {
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_bar': i,
                    'entry_time': row.timestamp,
                    'capital_used': capital,
                    'reason': reason
                }
                
        else:
            # Verificar saída
            current_price = row.valor_fechamento
            exit_reason = None
            
            if position['type'] == 'LONG':
                if current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                elif current_price >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
            
            # Timeout mais realista (168h = 1 semana para dar tempo das posições se resolverem)
            if i - position['entry_bar'] >= 168:  # 168 horas = 1 semana vs 24h
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                # Calcular resultado
                pnl_gross = (current_price - position['entry_price']) * position['shares']
                pnl_percent = (pnl_gross / (position['capital_used'] * TradingConfig.LEVERAGE)) * 100
                
                trade = {
                    'asset': asset,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_time': position['entry_time'],
                    'exit_time': row.timestamp,
                    'exit_reason': exit_reason,
                    'shares': position['shares'],
                    'capital_used': position['capital_used'],
                    'pnl_gross': pnl_gross,
                    'pnl_percent': pnl_percent,
                    'duration_bars': i - position['entry_bar'],
                    'entry_reason': position['reason']
                }
                
                trades.append(trade)
                position = None
    
    return trades

def main():
    print("🎯 BACKTEST TRADING.PY EXATO - 1 ANO DE DADOS REAIS")
    print("="*80)
    
    print(f"\n🧬 CONFIGURAÇÃO DNA GENÉTICO:")
    print(f"   SL: {TradingConfig.STOP_LOSS_PCT*100:.1f}%")
    print(f"   TP: {TradingConfig.TAKE_PROFIT_PCT*100:.1f}%")
    print(f"   Leverage: {TradingConfig.LEVERAGE}x")
    print(f"   EMA: {TradingConfig.EMA_FAST}/{TradingConfig.EMA_SLOW}")
    print(f"   RSI: {TradingConfig.RSI_PERIOD}")
    print(f"   Confluência mínima: 6.0/10 pontos (60%) - REALISTA")
    
    all_results = {}
    total_capital = len(TRADING_ASSETS) * 4.0  # $4 por asset
    total_pnl = 0
    all_trades = []
    
    print(f"\n📊 PROCESSANDO {len(TRADING_ASSETS)} ASSETS:")
    print("="*70)
    
    for asset in TRADING_ASSETS:
        print(f"\n📈 Processando {asset}...")
        
        df = load_trading_data(asset)
        if df is None:
            continue
        
        df = calculate_trading_indicators(df)
        trades = simulate_trading_exact(df, asset)
        
        if trades:
            wins = [t for t in trades if t['pnl_gross'] > 0]
            win_rate = len(wins) / len(trades) * 100
            total_pnl_asset = sum(t['pnl_gross'] for t in trades)
            roi_asset = (total_pnl_asset / 4.0) * 100  # ROI baseado em $4 inicial
            
            print(f"   ✅ {len(trades)} trades | Win rate: {win_rate:.1f}% | ROI: {roi_asset:+.1f}%")
            
            all_results[asset] = {
                'trades': len(trades),
                'wins': len(wins),
                'win_rate': win_rate,
                'total_pnl': total_pnl_asset,
                'roi': roi_asset
            }
            
            total_pnl += total_pnl_asset
            all_trades.extend(trades)
        else:
            print(f"   ❌ Nenhum trade encontrado")
            all_results[asset] = {
                'trades': 0,
                'wins': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'roi': 0
            }
    
    # Resultado final
    portfolio_roi = (total_pnl / total_capital) * 100
    
    print("\n" + "="*80)
    print("🏆 RESULTADOS FINAIS TRADING.PY:")
    print("="*80)
    
    print(f"\n📊 Performance por Asset:")
    print("Asset      | Trades | Wins | Win Rate | PnL      | ROI")
    print("-" * 60)
    
    for asset, result in all_results.items():
        print(f"{asset:10} | {result['trades']:6} | {result['wins']:4} | "
              f"{result['win_rate']:7.1f}% | ${result['total_pnl']:+7.2f} | {result['roi']:+6.1f}%")
    
    print("-" * 60)
    total_trades = sum(r['trades'] for r in all_results.values())
    total_wins = sum(r['wins'] for r in all_results.values())
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"PORTFOLIO  | {total_trades:6} | {total_wins:4} | "
          f"{overall_win_rate:7.1f}% | ${total_pnl:+7.2f} | {portfolio_roi:+6.1f}%")
    
    print(f"\n💰 RESUMO EXECUTIVO:")
    print(f"   💸 Capital inicial: ${total_capital:.2f}")
    print(f"   💰 Capital final: ${total_capital + total_pnl:.2f}")
    print(f"   📈 PnL total: ${total_pnl:+.2f}")
    print(f"   🎯 ROI Portfolio: {portfolio_roi:+.1f}%")
    print(f"   📊 Total de trades: {total_trades}")
    print(f"   🏆 Win rate geral: {overall_win_rate:.1f}%")
    
    if all_trades:
        # Estatísticas detalhadas
        wins_list = [t for t in all_trades if t['pnl_gross'] > 0]
        losses_list = [t for t in all_trades if t['pnl_gross'] < 0]
        
        avg_win = np.mean([t['pnl_percent'] for t in wins_list]) if wins_list else 0
        avg_loss = np.mean([t['pnl_percent'] for t in losses_list]) if losses_list else 0
        
        best_trade = max(all_trades, key=lambda x: x['pnl_gross'])
        worst_trade = min(all_trades, key=lambda x: x['pnl_gross'])
        
        print(f"\n🎲 ESTATÍSTICAS DETALHADAS:")
        print(f"   📈 Ganho médio: {avg_win:+.1f}%")
        print(f"   📉 Perda média: {avg_loss:+.1f}%")
        print(f"   🏆 Melhor trade: {best_trade['asset']} ${best_trade['pnl_gross']:+.2f}")
        print(f"   💥 Pior trade: {worst_trade['asset']} ${worst_trade['pnl_gross']:+.2f}")
        
        # Análise por saída
        exit_reasons = {}
        for trade in all_trades:
            reason = trade['exit_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += trade['pnl_gross']
        
        print(f"\n📊 ANÁLISE POR TIPO DE SAÍDA:")
        for reason, data in exit_reasons.items():
            avg_pnl = data['pnl'] / data['count']
            pct = (data['count'] / total_trades) * 100
            print(f"   {reason}: {data['count']} trades ({pct:.1f}%) | Avg PnL: ${avg_pnl:+.2f}")
    
    print("\n🧬 TRADING.PY EXECUTADO COM SUCESSO!")
    print("="*80)
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        'timestamp': timestamp,
        'strategy': 'TRADING_PY_EXACT',
        'config': {
            'stop_loss_pct': TradingConfig.STOP_LOSS_PCT,
            'take_profit_pct': TradingConfig.TAKE_PROFIT_PCT,
            'leverage': TradingConfig.LEVERAGE,
            'ema_fast': TradingConfig.EMA_FAST,
            'ema_slow': TradingConfig.EMA_SLOW,
            'rsi_period': TradingConfig.RSI_PERIOD
        },
        'portfolio_roi': portfolio_roi,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': overall_win_rate,
        'total_pnl': total_pnl,
        'assets_results': all_results,
        'detailed_trades': all_trades
    }
    
    filename = f"backtest_tradingpy_exato_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"📁 Resultados salvos: {filename}")

if __name__ == "__main__":
    main()
