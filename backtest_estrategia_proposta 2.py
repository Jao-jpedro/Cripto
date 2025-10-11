#!/usr/bin/env python3
"""
🔬 BACKTEST ESTRATÉGIA PROPOSTA - ROI ANUAL EXATO
===============================================
🎯 Testar parâmetros específicos propostos pelo usuário
📊 Calcular ROI anual real com dados de 1 ano

PARÂMETROS DA ESTRATÉGIA PROPOSTA:
- Stop Loss: 2.0%
- Take Profit: 18.0%  
- Leverage: 4x
- EMA: 3/34
- RSI: 21 períodos
- Min Confluence: 5.5/10 (55%)
- Volume: 1.3x média
- ATR Range: 0.3% - 2.5%
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProposedStrategyConfig:
    """Configuração exata da estratégia proposta"""
    stop_loss_pct: float = 0.02         # 2.0% SL
    take_profit_pct: float = 0.18       # 18.0% TP
    leverage: int = 4                   # 4x leverage
    ema_fast: int = 3                   # EMA rápida
    ema_slow: int = 34                  # EMA lenta
    rsi_period: int = 21                # RSI período
    min_confluence: float = 5.5         # 5.5/10 critérios (55%)
    volume_multiplier: float = 1.3      # 1.3x volume
    atr_min_pct: float = 0.30          # ATR mínimo 0.3%
    atr_max_pct: float = 2.50          # ATR máximo 2.5%

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos"""
    df = df.copy()
    
    # Mapear colunas
    df['close'] = df['valor_fechamento']
    df['high'] = df['valor_maximo']
    df['low'] = df['valor_minimo']
    df['open'] = df['valor_abertura']
    
    # EMAs (3 e 34)
    df['ema3'] = df['close'].ewm(span=3).mean()
    df['ema34'] = df['close'].ewm(span=34).mean()
    
    # Gradientes EMA
    df['ema3_grad'] = df['ema3'].pct_change(periods=3) * 100
    df['ema34_grad'] = df['ema34'].pct_change(periods=21) * 100
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # RSI (21 períodos)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def proposed_confluence_long(row, config: ProposedStrategyConfig) -> Tuple[float, List[str]]:
    """Sistema de confluência da estratégia proposta para LONG (5.5/10 critérios)"""
    reasons = []
    score = 0.0
    max_score = 10.0
    
    # CRITÉRIO 1: EMA Cross + Gradiente (peso 2.0)
    c1_ema = row.ema3 > row.ema34
    c1_grad = row.ema3_grad > 0.08  # 0.08% mínimo
    if c1_ema and c1_grad:
        score += 2.0
        reasons.append("✅ EMA3>EMA34+grad>0.08%")
    elif c1_ema:
        score += 1.0
        reasons.append("🔶 EMA3>EMA34 (grad fraco)")
    else:
        reasons.append("❌ EMA inadequada")
    
    # CRITÉRIO 2: ATR Range (0.3% - 2.5%)
    if config.atr_min_pct <= row.atr_pct <= config.atr_max_pct:
        score += 1.0
        reasons.append("✅ ATR no range")
    else:
        reasons.append("❌ ATR fora do range")
    
    # CRITÉRIO 3: Rompimento significativo
    if row.close > (row.ema3 + 0.8 * row.atr):
        score += 1.0
        reasons.append("✅ Rompimento")
    else:
        reasons.append("❌ Sem rompimento")
    
    # CRITÉRIO 4: Volume 1.3x
    if row.volume_ratio > config.volume_multiplier:
        score += 1.0
        reasons.append("✅ Volume alto")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI 21 períodos (30-70)
    if hasattr(row, 'rsi') and not pd.isna(row.rsi):
        if 30 <= row.rsi <= 70:
            score += 1.0
            reasons.append("✅ RSI ok")
        else:
            reasons.append("❌ RSI extremo")
    else:
        score += 0.5
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD positivo
    if hasattr(row, 'macd') and hasattr(row, 'macd_signal') and not pd.isna(row.macd):
        if row.macd > row.macd_signal:
            score += 1.0
            reasons.append("✅ MACD positivo")
        else:
            reasons.append("❌ MACD negativo")
    else:
        score += 0.5
        reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação EMAs
    if row.atr > 0:
        ema_separation = abs(row.ema3 - row.ema34) / row.atr
        if ema_separation >= 0.5:
            score += 1.0
            reasons.append("✅ EMAs separadas")
        else:
            reasons.append("❌ EMAs próximas")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 8: Timing preciso
    if row.atr > 0:
        price_distance = abs(row.close - row.ema3) / row.atr
        if price_distance <= 1.5:
            score += 1.0
            reasons.append("✅ Timing bom")
        else:
            reasons.append("❌ Timing tardio")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 9: Bollinger Bands posição
    if hasattr(row, 'bb_percent_b') and not pd.isna(row.bb_percent_b):
        if 0.6 <= row.bb_percent_b <= 1.0:
            score += 0.5
            reasons.append("✅ BB superior")
        else:
            score += 0.2
            reasons.append("🔶 BB neutro")
    else:
        score += 0.3
        reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: Momentum EMA34
    if hasattr(row, 'ema34_grad') and not pd.isna(row.ema34_grad):
        if row.ema34_grad > 0:
            score += 0.5
            reasons.append("✅ Momentum geral")
        else:
            score += 0.2
            reasons.append("🔶 Momentum neutro")
    else:
        score += 0.3
        reasons.append("⚪ Momentum n/d")
    
    return score, reasons

def proposed_confluence_short(row, config: ProposedStrategyConfig) -> Tuple[float, List[str]]:
    """Sistema de confluência da estratégia proposta para SHORT (5.5/10 critérios)"""
    reasons = []
    score = 0.0
    max_score = 10.0
    
    # CRITÉRIO 1: EMA Cross + Gradiente (peso 2.0)
    c1_ema = row.ema3 < row.ema34
    c1_grad = row.ema3_grad < -0.08  # -0.08% mínimo
    if c1_ema and c1_grad:
        score += 2.0
        reasons.append("✅ EMA3<EMA34+grad<-0.08%")
    elif c1_ema:
        score += 1.0
        reasons.append("🔶 EMA3<EMA34 (grad fraco)")
    else:
        reasons.append("❌ EMA inadequada")
    
    # CRITÉRIO 2: ATR Range (0.3% - 2.5%)
    if config.atr_min_pct <= row.atr_pct <= config.atr_max_pct:
        score += 1.0
        reasons.append("✅ ATR no range")
    else:
        reasons.append("❌ ATR fora do range")
    
    # CRITÉRIO 3: Rompimento significativo
    if row.close < (row.ema3 - 0.8 * row.atr):
        score += 1.0
        reasons.append("✅ Rompimento")
    else:
        reasons.append("❌ Sem rompimento")
    
    # CRITÉRIO 4: Volume 1.3x
    if row.volume_ratio > config.volume_multiplier:
        score += 1.0
        reasons.append("✅ Volume alto")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI 21 períodos (30-70)
    if hasattr(row, 'rsi') and not pd.isna(row.rsi):
        if 30 <= row.rsi <= 70:
            score += 1.0
            reasons.append("✅ RSI ok")
        else:
            reasons.append("❌ RSI extremo")
    else:
        score += 0.5
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD negativo
    if hasattr(row, 'macd') and hasattr(row, 'macd_signal') and not pd.isna(row.macd):
        if row.macd < row.macd_signal:
            score += 1.0
            reasons.append("✅ MACD negativo")
        else:
            reasons.append("❌ MACD positivo")
    else:
        score += 0.5
        reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação EMAs
    if row.atr > 0:
        ema_separation = abs(row.ema3 - row.ema34) / row.atr
        if ema_separation >= 0.5:
            score += 1.0
            reasons.append("✅ EMAs separadas")
        else:
            reasons.append("❌ EMAs próximas")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 8: Timing preciso
    if row.atr > 0:
        price_distance = abs(row.close - row.ema3) / row.atr
        if price_distance <= 1.5:
            score += 1.0
            reasons.append("✅ Timing bom")
        else:
            reasons.append("❌ Timing tardio")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 9: Bollinger Bands posição
    if hasattr(row, 'bb_percent_b') and not pd.isna(row.bb_percent_b):
        if 0.0 <= row.bb_percent_b <= 0.4:
            score += 0.5
            reasons.append("✅ BB inferior")
        else:
            score += 0.2
            reasons.append("🔶 BB neutro")
    else:
        score += 0.3
        reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: Momentum EMA34
    if hasattr(row, 'ema34_grad') and not pd.isna(row.ema34_grad):
        if row.ema34_grad < 0:
            score += 0.5
            reasons.append("✅ Momentum geral")
        else:
            score += 0.2
            reasons.append("🔶 Momentum neutro")
    else:
        score += 0.3
        reasons.append("⚪ Momentum n/d")
    
    return score, reasons

def calculate_hyperliquid_fees(notional_value: float) -> float:
    """Calcula taxas da Hyperliquid"""
    return notional_value * 0.0007  # 0.07% total

def backtest_proposed_strategy(symbol: str, config: ProposedStrategyConfig) -> Dict:
    """Executa backtest da estratégia proposta"""
    
    # Carregar dados
    filename = f"dados_reais_{symbol.lower()}_1ano.csv"
    if not os.path.exists(filename):
        return {"error": f"Arquivo {filename} não encontrado"}
    
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calcular indicadores
    df = calculate_indicators(df)
    
    # Variáveis de controle
    capital = 64.0  # Capital inicial
    position = None
    trades = []
    equity_curve = [capital]
    
    for i in range(50, len(df)):  # Começar após período de warmup
        row = df.iloc[i]
        
        if position is None:
            # Verificar entrada LONG
            score_long, reasons_long = proposed_confluence_long(row, config)
            
            if score_long >= config.min_confluence:
                # ENTRADA LONG
                entry_price = row.close
                notional_value = capital * config.leverage
                position_size = notional_value / entry_price
                
                # Calcular stops
                sl_price = entry_price * (1 - config.stop_loss_pct)
                tp_price = entry_price * (1 + config.take_profit_pct)
                
                position = {
                    'side': 'long',
                    'entry_price': entry_price,
                    'entry_time': row.timestamp,
                    'size': position_size,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'notional': notional_value,
                    'confluence': score_long,
                    'reasons': reasons_long[:3]
                }
                continue
            
            # Verificar entrada SHORT
            score_short, reasons_short = proposed_confluence_short(row, config)
            
            if score_short >= config.min_confluence:
                # ENTRADA SHORT
                entry_price = row.close
                notional_value = capital * config.leverage
                position_size = notional_value / entry_price
                
                # Calcular stops
                sl_price = entry_price * (1 + config.stop_loss_pct)
                tp_price = entry_price * (1 - config.take_profit_pct)
                
                position = {
                    'side': 'short',
                    'entry_price': entry_price,
                    'entry_time': row.timestamp,
                    'size': position_size,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'notional': notional_value,
                    'confluence': score_short,
                    'reasons': reasons_short[:3]
                }
                continue
        
        else:
            # Gerenciar posição existente
            current_price = row.close
            
            if position['side'] == 'long':
                # Verificar SL/TP LONG
                if current_price <= position['sl_price']:
                    # Stop Loss
                    pnl_pct = -config.stop_loss_pct
                    close_reason = "SL"
                elif current_price >= position['tp_price']:
                    # Take Profit
                    pnl_pct = config.take_profit_pct
                    close_reason = "TP"
                else:
                    continue
            
            else:  # SHORT
                # Verificar SL/TP SHORT
                if current_price >= position['sl_price']:
                    # Stop Loss
                    pnl_pct = -config.stop_loss_pct
                    close_reason = "SL"
                elif current_price <= position['tp_price']:
                    # Take Profit
                    pnl_pct = config.take_profit_pct
                    close_reason = "TP"
                else:
                    continue
            
            # FECHAR POSIÇÃO
            pnl_raw = capital * config.leverage * pnl_pct
            fees = calculate_hyperliquid_fees(position['notional'])
            pnl_net = pnl_raw - fees
            
            capital += pnl_net
            equity_curve.append(capital)
            
            trade = {
                'symbol': symbol,
                'side': position['side'],
                'entry_time': position['entry_time'],
                'exit_time': row.timestamp,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': position['size'],
                'pnl_pct': pnl_pct,
                'pnl_raw': pnl_raw,
                'fees': fees,
                'pnl_net': pnl_net,
                'close_reason': close_reason,
                'confluence': position['confluence'],
                'reasons': position['reasons']
            }
            trades.append(trade)
            position = None
    
    # Calcular métricas
    if not trades:
        return {
            'symbol': symbol,
            'trades': 0,
            'roi_bruto': 0,
            'roi_liquido': 0,
            'win_rate': 0,
            'total_fees': 0,
            'final_capital': capital
        }
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl_net'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    
    total_pnl_raw = sum(t['pnl_raw'] for t in trades)
    total_fees = sum(t['fees'] for t in trades)
    total_pnl_net = sum(t['pnl_net'] for t in trades)
    
    roi_bruto = (total_pnl_raw / 64.0) * 100
    roi_liquido = ((capital - 64.0) / 64.0) * 100
    
    return {
        'symbol': symbol,
        'trades': total_trades,
        'win_rate': win_rate,
        'roi_bruto': roi_bruto,
        'roi_liquido': roi_liquido,
        'total_fees': total_fees,
        'final_capital': capital,
        'trades_detail': trades,
        'equity_curve': equity_curve
    }

def main():
    """Executa backtest completo da estratégia proposta"""
    
    print("🔬 BACKTEST ESTRATÉGIA PROPOSTA - ROI ANUAL EXATO")
    print("=" * 70)
    print("📊 Parâmetros: SL 2.0% | TP 18.0% | LEV 4x | Confluence 5.5/10")
    print("💡 Volume: 1.3x | ATR: 0.3%-2.5% | EMA: 3/34 | RSI: 21")
    print()
    
    config = ProposedStrategyConfig()
    
    # Assets para testar
    assets = [
        'btc', 'sol', 'eth', 'xrp', 'doge', 'avax', 'ena', 'bnb',
        'sui', 'ada', 'link', 'wld', 'aave', 'crv', 'ltc', 'near'
    ]
    
    all_results = []
    total_capital_inicial = 64.0 * len(assets)
    total_capital_final = 0.0
    total_trades = 0
    total_fees = 0.0
    
    print("🚀 EXECUTANDO BACKTESTS:")
    print("=" * 50)
    
    for asset in assets:
        result = backtest_proposed_strategy(asset, config)
        
        if 'error' in result:
            print(f"   ❌ {asset.upper()}: {result['error']}")
            continue
        
        all_results.append(result)
        total_capital_final += result['final_capital']
        total_trades += result['trades']
        total_fees += result['total_fees']
        
        # Status do asset
        status = "🟢" if result['roi_liquido'] > 0 else "🔴"
        print(f"   {status} {result['symbol'].upper()}: {result['trades']} trades | "
              f"{result['win_rate']:.1f}% WR | ROI: {result['roi_bruto']:.1f}%→{result['roi_liquido']:.1f}%")
    
    # Resultados finais
    if all_results:
        roi_total_bruto = ((total_capital_final + sum(r['total_fees'] for r in all_results) - total_capital_inicial) / total_capital_inicial) * 100
        roi_total_liquido = ((total_capital_final - total_capital_inicial) / total_capital_inicial) * 100
        impacto_taxas = roi_total_bruto - roi_total_liquido
        
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        assets_positivos = len([r for r in all_results if r['roi_liquido'] > 0])
        
        print()
        print("📊 RESULTADO FINAL DA ESTRATÉGIA PROPOSTA:")
        print("=" * 60)
        print(f"   💰 ROI Bruto: {roi_total_bruto:.1f}%")
        print(f"   💵 ROI Líquido: {roi_total_liquido:.1f}%")
        print(f"   💸 Impacto Taxas: {impacto_taxas:.1f}%")
        print(f"   🎯 Total Trades: {total_trades}")
        print(f"   🏆 Win Rate Médio: {avg_win_rate:.1f}%")
        print(f"   ✅ Assets Positivos: {assets_positivos}/{len(all_results)}")
        print(f"   🏦 Fees Totais: ${total_fees:.2f}")
        print()
        
        print("🔍 COMPARAÇÕES:")
        print("-" * 30)
        print(f"   📈 Configuração Vencedora: +9,480% ROI")
        print(f"   🔬 Estratégia Proposta: {roi_total_liquido:+.1f}% ROI")
        
        diferenca = roi_total_liquido - 9480.4
        if diferenca > 0:
            print(f"   🎉 MELHOR EM: +{diferenca:.1f}%")
        else:
            print(f"   ⚠️  PIOR EM: {diferenca:.1f}%")
        
        # Salvar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"estrategia_proposta_backtest_{timestamp}.json"
        
        resultado_completo = {
            'config': {
                'stop_loss_pct': config.stop_loss_pct,
                'take_profit_pct': config.take_profit_pct,
                'leverage': config.leverage,
                'min_confluence': config.min_confluence,
                'volume_multiplier': config.volume_multiplier,
                'atr_min_pct': config.atr_min_pct,
                'atr_max_pct': config.atr_max_pct
            },
            'summary': {
                'roi_total_bruto': roi_total_bruto,
                'roi_total_liquido': roi_total_liquido,
                'impacto_taxas': impacto_taxas,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'assets_positivos': assets_positivos,
                'total_fees': total_fees,
                'vs_configuracao_vencedora': diferenca
            },
            'results_by_asset': all_results
        }
        
        with open(filename, 'w') as f:
            json.dump(resultado_completo, f, indent=2, default=str)
        
        print(f"📁 Resultados salvos: {filename}")
    
    print("\n🎊 BACKTEST ESTRATÉGIA PROPOSTA CONCLUÍDO!")

if __name__ == "__main__":
    main()
