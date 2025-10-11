#!/usr/bin/env python3
"""
🏆 DNA HYPERLIQUID ULTIMATE - MÁXIMO ROI POSSÍVEL
===============================================
🎯 Objetivo: Aceitar limitações da Hyperliquid e maximizar ROI real
💡 Estratégia: Encontrar o sweet spot entre frequency e quality
📊 Abordagem: Testar diferentes configurações para achar o ápice

🔬 METODOLOGIA:
1. Testar variações do DNA Realista (já sabemos que funciona)
2. Micro-ajustes em confluence, volume, timing
3. Encontrar configuração que maximize ROI líquido
4. Validar com dados reais de 1 ano

⚡ HIPÓTESE: DNA Realista pode ser ligeiramente melhorado com ajustes finos
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
class HyperliquidConfig:
    """Configurações para maximizar ROI na Hyperliquid"""
    # Core DNA (base comprovada)
    stop_loss_pct: float = 0.015      # 1.5% SL
    take_profit_pct: float = 0.12     # 12% TP  
    leverage: int = 3                 # 3x leverage
    
    # Confluence (sweet spot entre quality e quantity)
    min_confluence: float = 3.0       # Base: 3/10 critérios
    
    # Volume (balanceado)
    volume_multiplier: float = 1.3    # Base: 1.3x
    
    # Timing e breakout
    min_atr_breakout: float = 0.8     # 0.8 ATR (menos restritivo)
    max_timing_distance: float = 1.5  # 1.5 ATR (mais flexível)
    min_ema_gradient: float = 0.08    # 0.08% (original)
    
    # Risk Management
    atr_min_pct: float = 0.35         # ATR mínimo 0.35%
    atr_max_pct: float = 1.50         # ATR máximo 1.5%

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos para Hyperliquid"""
    df = df.copy()
    
    # Mapear colunas para formato padrão
    df['close'] = df['valor_fechamento']
    df['high'] = df['valor_maximo']
    df['low'] = df['valor_minimo']
    df['open'] = df['valor_abertura']
    
    # EMAs DNA (3 e 34)
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
    
    # RSI
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

def balanced_confluence_long(row, config: HyperliquidConfig) -> Tuple[float, List[str]]:
    """Sistema de confluência balanceado para LONG (otimizado para Hyperliquid)"""
    reasons = []
    score = 0.0
    max_score = 10.0
    
    # CRITÉRIO 1: EMA + Gradiente (CORE - peso duplo)
    c1_ema = row.ema3 > row.ema34
    c1_grad = row.ema3_grad > config.min_ema_gradient
    if c1_ema and c1_grad:
        score += 2.0  # Peso duplo para critério principal
        reasons.append(f"✅ EMA3>EMA34+grad>{config.min_ema_gradient}%")
    elif c1_ema:
        score += 1.0  # EMA alinhada pelo menos
        reasons.append("🔶 EMA3>EMA34 (grad fraco)")
    else:
        reasons.append("❌ EMA inadequada")
    
    # CRITÉRIO 2: ATR razoável
    if config.atr_min_pct <= row.atr_pct <= config.atr_max_pct:
        score += 1.0
        reasons.append("✅ ATR saudável")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento (flexível)
    if row.close > (row.ema3 + config.min_atr_breakout * row.atr):
        score += 1.0
        reasons.append("✅ Rompimento")
    else:
        reasons.append("❌ Sem rompimento")
    
    # CRITÉRIO 4: Volume (balanceado)
    if row.volume_ratio > config.volume_multiplier:
        score += 1.0
        reasons.append("✅ Volume alto")
    elif row.volume_ratio > (config.volume_multiplier * 0.8):  # 80% do threshold
        score += 0.5
        reasons.append("🔶 Volume ok")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI (flexível)
    if hasattr(row, 'rsi') and not pd.isna(row.rsi):
        if 30 <= row.rsi <= 70:  # Zona ampla mas razoável
            score += 1.0
            reasons.append("✅ RSI ok")
        else:
            reasons.append("❌ RSI extremo")
    else:
        score += 0.5
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD (se disponível)
    if hasattr(row, 'macd') and hasattr(row, 'macd_signal') and not pd.isna(row.macd):
        if row.macd > row.macd_signal:
            score += 1.0
            reasons.append("✅ MACD positivo")
        else:
            score += 0.3
            reasons.append("🔶 MACD neutro")
    else:
        score += 0.5
        reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação EMAs
    if row.atr > 0:
        ema_separation = abs(row.ema3 - row.ema34) / row.atr
        if ema_separation >= 0.3:  # Menos restritivo
            score += 1.0
            reasons.append("✅ EMAs separadas")
        else:
            score += 0.3
            reasons.append("🔶 EMAs próximas")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 8: Timing (flexível)
    if row.atr > 0:
        price_distance = abs(row.close - row.ema3) / row.atr
        if price_distance <= config.max_timing_distance:
            score += 1.0
            reasons.append("✅ Timing bom")
        else:
            score += 0.3
            reasons.append("🔶 Timing tardio")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 9: Bollinger Bands (se disponível)
    if hasattr(row, 'bb_percent_b') and not pd.isna(row.bb_percent_b):
        if 0.5 <= row.bb_percent_b <= 1.0:  # Zona superior
            score += 0.5
            reasons.append("✅ BB superior")
        else:
            score += 0.2
            reasons.append("🔶 BB neutro")
    else:
        score += 0.3
        reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: Momentum geral
    if hasattr(row, 'ema34_grad') and not pd.isna(row.ema34_grad):
        if row.ema34_grad > 0:  # Tendência geral positiva
            score += 0.5
            reasons.append("✅ Momentum geral")
        else:
            score += 0.2
            reasons.append("🔶 Momentum neutro")
    else:
        score += 0.3
        reasons.append("⚪ Momentum n/d")
    
    return score, reasons

def balanced_confluence_short(row, config: HyperliquidConfig) -> Tuple[float, List[str]]:
    """Sistema de confluência balanceado para SHORT (otimizado para Hyperliquid)"""
    reasons = []
    score = 0.0
    max_score = 10.0
    
    # CRITÉRIO 1: EMA + Gradiente (CORE - peso duplo)
    c1_ema = row.ema3 < row.ema34
    c1_grad = row.ema3_grad < -config.min_ema_gradient
    if c1_ema and c1_grad:
        score += 2.0  # Peso duplo para critério principal
        reasons.append(f"✅ EMA3<EMA34+grad<-{config.min_ema_gradient}%")
    elif c1_ema:
        score += 1.0  # EMA alinhada pelo menos
        reasons.append("🔶 EMA3<EMA34 (grad fraco)")
    else:
        reasons.append("❌ EMA inadequada")
    
    # CRITÉRIO 2: ATR razoável
    if config.atr_min_pct <= row.atr_pct <= config.atr_max_pct:
        score += 1.0
        reasons.append("✅ ATR saudável")
    else:
        reasons.append("❌ ATR inadequado")
    
    # CRITÉRIO 3: Rompimento (flexível)
    if row.close < (row.ema3 - config.min_atr_breakout * row.atr):
        score += 1.0
        reasons.append("✅ Rompimento")
    else:
        reasons.append("❌ Sem rompimento")
    
    # CRITÉRIO 4: Volume (balanceado)
    if row.volume_ratio > config.volume_multiplier:
        score += 1.0
        reasons.append("✅ Volume alto")
    elif row.volume_ratio > (config.volume_multiplier * 0.8):  # 80% do threshold
        score += 0.5
        reasons.append("🔶 Volume ok")
    else:
        reasons.append("❌ Volume baixo")
    
    # CRITÉRIO 5: RSI (flexível)
    if hasattr(row, 'rsi') and not pd.isna(row.rsi):
        if 30 <= row.rsi <= 70:  # Zona ampla mas razoável
            score += 1.0
            reasons.append("✅ RSI ok")
        else:
            reasons.append("❌ RSI extremo")
    else:
        score += 0.5
        reasons.append("⚪ RSI n/d")
    
    # CRITÉRIO 6: MACD (se disponível)
    if hasattr(row, 'macd') and hasattr(row, 'macd_signal') and not pd.isna(row.macd):
        if row.macd < row.macd_signal:
            score += 1.0
            reasons.append("✅ MACD negativo")
        else:
            score += 0.3
            reasons.append("🔶 MACD neutro")
    else:
        score += 0.5
        reasons.append("⚪ MACD n/d")
    
    # CRITÉRIO 7: Separação EMAs
    if row.atr > 0:
        ema_separation = abs(row.ema3 - row.ema34) / row.atr
        if ema_separation >= 0.3:  # Menos restritivo
            score += 1.0
            reasons.append("✅ EMAs separadas")
        else:
            score += 0.3
            reasons.append("🔶 EMAs próximas")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 8: Timing (flexível)
    if row.atr > 0:
        price_distance = abs(row.close - row.ema3) / row.atr
        if price_distance <= config.max_timing_distance:
            score += 1.0
            reasons.append("✅ Timing bom")
        else:
            score += 0.3
            reasons.append("🔶 Timing tardio")
    else:
        reasons.append("❌ ATR zero")
    
    # CRITÉRIO 9: Bollinger Bands (se disponível)
    if hasattr(row, 'bb_percent_b') and not pd.isna(row.bb_percent_b):
        if 0.0 <= row.bb_percent_b <= 0.5:  # Zona inferior
            score += 0.5
            reasons.append("✅ BB inferior")
        else:
            score += 0.2
            reasons.append("🔶 BB neutro")
    else:
        score += 0.3
        reasons.append("⚪ BB n/d")
    
    # CRITÉRIO 10: Momentum geral
    if hasattr(row, 'ema34_grad') and not pd.isna(row.ema34_grad):
        if row.ema34_grad < 0:  # Tendência geral negativa
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
    """Calcula taxas da Hyperliquid (0.02% maker + 0.05% taker + funding)"""
    maker_fee = notional_value * 0.0002    # 0.02%
    taker_fee = notional_value * 0.0005    # 0.05%
    funding_fee = notional_value * 0.0001  # 0.01% (conservador)
    return maker_fee + taker_fee + funding_fee

def backtest_hyperliquid_ultimate(symbol: str, config: HyperliquidConfig) -> Dict:
    """Executa backtest para maximizar ROI na Hyperliquid"""
    
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
            score_long, reasons_long = balanced_confluence_long(row, config)
            
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
            score_short, reasons_short = balanced_confluence_short(row, config)
            
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

def test_multiple_configurations():
    """Testa múltiplas configurações para encontrar o máximo ROI"""
    
    print("🏆 DNA HYPERLIQUID ULTIMATE - BUSCA PELO MÁXIMO ROI")
    print("=" * 80)
    print("🎯 Testando variações para encontrar configuração ótima")
    print()
    
    # Assets para testar (focar nos principais)
    assets = ['btc', 'sol', 'eth', 'xrp', 'doge', 'avax']
    
    # Configurações para testar
    configs_to_test = [
        # Config 1: DNA Realista Original (baseline)
        HyperliquidConfig(min_confluence=3.0, volume_multiplier=1.3, min_atr_breakout=0.8),
        
        # Config 2: Ligeiramente mais restritivo
        HyperliquidConfig(min_confluence=3.5, volume_multiplier=1.4, min_atr_breakout=0.8),
        
        # Config 3: Mais permissivo
        HyperliquidConfig(min_confluence=2.5, volume_multiplier=1.2, min_atr_breakout=0.7),
        
        # Config 4: Volume focus
        HyperliquidConfig(min_confluence=3.0, volume_multiplier=1.5, min_atr_breakout=0.8),
        
        # Config 5: Timing focus
        HyperliquidConfig(min_confluence=3.0, volume_multiplier=1.3, min_atr_breakout=0.9),
        
        # Config 6: Balanced premium
        HyperliquidConfig(min_confluence=3.2, volume_multiplier=1.35, min_atr_breakout=0.85),
    ]
    
    config_names = [
        "DNA Original",
        "Ligeiramente Restritivo", 
        "Mais Permissivo",
        "Volume Focus",
        "Timing Focus", 
        "Balanced Premium"
    ]
    
    best_roi = -999999
    best_config = None
    best_config_name = ""
    all_results = []
    
    for i, (config, name) in enumerate(zip(configs_to_test, config_names)):
        print(f"🚀 TESTANDO CONFIG {i+1}: {name}")
        print("-" * 60)
        
        total_roi = 0
        total_trades = 0
        total_fees = 0
        config_results = []
        
        for asset in assets:
            result = backtest_hyperliquid_ultimate(asset, config)
            
            if 'error' not in result:
                config_results.append(result)
                total_roi += result['roi_liquido']
                total_trades += result['trades']
                total_fees += result['total_fees']
                
                status = "🟢" if result['roi_liquido'] > 0 else "🔴"
                print(f"   {status} {result['symbol'].upper()}: {result['trades']} trades | "
                      f"{result['win_rate']:.1f}% WR | ROI: {result['roi_liquido']:.1f}%")
        
        avg_roi = total_roi / len(assets) if assets else 0
        
        print(f"   📊 RESULTADO: ROI Médio: {avg_roi:.1f}% | Trades: {total_trades} | Fees: ${total_fees:.2f}")
        
        if avg_roi > best_roi:
            best_roi = avg_roi
            best_config = config
            best_config_name = name
            
        all_results.append({
            'name': name,
            'config': config,
            'avg_roi': avg_roi,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'results': config_results
        })
        
        print()
    
    # Resultado final
    print("🏆 RESULTADO FINAL - MÁXIMO ROI ENCONTRADO:")
    print("=" * 60)
    print(f"🥇 MELHOR CONFIGURAÇÃO: {best_config_name}")
    print(f"💰 ROI MÁXIMO: {best_roi:.1f}%")
    print()
    print("📊 RANKING DE CONFIGURAÇÕES:")
    
    # Ordenar por ROI
    all_results.sort(key=lambda x: x['avg_roi'], reverse=True)
    
    for i, result in enumerate(all_results):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f" {i+1}"
        print(f"{medal} {result['name']}: {result['avg_roi']:.1f}% ROI | "
              f"{result['total_trades']} trades | ${result['total_fees']:.2f} fees")
    
    # Salvar melhor configuração
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperliquid_ultimate_config_{timestamp}.json"
    
    best_result = {
        'best_config_name': best_config_name,
        'best_roi': best_roi,
        'best_config': {
            'min_confluence': best_config.min_confluence,
            'volume_multiplier': best_config.volume_multiplier,
            'min_atr_breakout': best_config.min_atr_breakout,
            'max_timing_distance': best_config.max_timing_distance,
            'min_ema_gradient': best_config.min_ema_gradient
        },
        'all_results': all_results
    }
    
    with open(filename, 'w') as f:
        json.dump(best_result, f, indent=2, default=str)
    
    print(f"\n📁 Melhor configuração salva: {filename}")
    print("\n🎊 BUSCA PELO MÁXIMO ROI CONCLUÍDA!")

if __name__ == "__main__":
    test_multiple_configurations()
