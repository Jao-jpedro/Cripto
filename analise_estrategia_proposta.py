#!/usr/bin/env python3
"""
📊 ANÁLISE DETALHADA DA ESTRATÉGIA PROPOSTA
==========================================
🎯 Comparação: Estratégia Proposta vs DNA Hyperliquid Ultimate
💰 Análise de capital, risco, e potencial de retorno

ESTRATÉGIA PROPOSTA:
- Stop Loss: 2.0% (vs 1.5% atual)
- Take Profit: 18.0% (vs 12.0% atual)  
- Leverage: 4x (vs 3x atual)
- EMA: 3/34 (igual ao atual)
- RSI: 21 períodos (igual ao atual)
- Min Confluence: 5.5/10 (vs 2.5/10 atual)
- Volume: 1.3x (vs 1.2x atual)
- ATR Range: 0.3-2.5% (vs 0.35-1.5% atual)
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
class ProposedStrategy:
    """Estratégia proposta pelo usuário"""
    stop_loss_pct: float = 0.02         # 2.0% SL (vs 1.5%)
    take_profit_pct: float = 0.18       # 18.0% TP (vs 12.0%)
    leverage: int = 4                   # 4x leverage (vs 3x)
    ema_fast: int = 3                   # EMA rápida
    ema_slow: int = 34                  # EMA lenta
    rsi_period: int = 21                # RSI período
    min_confluence: float = 5.5         # 5.5/10 critérios (vs 2.5/10)
    volume_multiplier: float = 1.3      # 1.3x volume (vs 1.2x)
    atr_min_pct: float = 0.30          # ATR mínimo 0.3% (vs 0.35%)
    atr_max_pct: float = 2.50          # ATR máximo 2.5% (vs 1.5%)

@dataclass
class CurrentWinning:
    """Configuração vencedora atual"""
    stop_loss_pct: float = 0.015        # 1.5% SL
    take_profit_pct: float = 0.12       # 12.0% TP
    leverage: int = 3                   # 3x leverage
    ema_fast: int = 3                   # EMA rápida
    ema_slow: int = 34                  # EMA lenta
    rsi_period: int = 21                # RSI período
    min_confluence: float = 2.5         # 2.5/10 critérios
    volume_multiplier: float = 1.2      # 1.2x volume
    atr_min_pct: float = 0.35          # ATR mínimo 0.35%
    atr_max_pct: float = 1.50          # ATR máximo 1.5%

def calculate_strategy_metrics():
    """Calcula métricas detalhadas das duas estratégias"""
    
    print("📊 ANÁLISE DETALHADA: ESTRATÉGIA PROPOSTA vs CONFIGURAÇÃO VENCEDORA")
    print("=" * 80)
    
    proposed = ProposedStrategy()
    current = CurrentWinning()
    
    # Capital inicial base
    base_capital = 64.0  # $64 por asset
    
    print("💰 ANÁLISE DE CAPITAL E RISCO:")
    print("=" * 50)
    
    # Cálculo do capital investido por trade
    proposed_capital_per_trade = base_capital * proposed.leverage
    current_capital_per_trade = base_capital * current.leverage
    
    print(f"💵 Capital por Trade:")
    print(f"   📈 Estratégia Proposta: ${proposed_capital_per_trade:.0f} (4x leverage)")
    print(f"   🏆 Configuração Atual: ${current_capital_per_trade:.0f} (3x leverage)")
    print(f"   📊 Diferença: +${proposed_capital_per_trade - current_capital_per_trade:.0f} (+{((proposed_capital_per_trade/current_capital_per_trade)-1)*100:.0f}%)")
    
    print(f"\n💸 Risco por Trade:")
    proposed_risk = proposed_capital_per_trade * proposed.stop_loss_pct
    current_risk = current_capital_per_trade * current.stop_loss_pct
    
    print(f"   📈 Estratégia Proposta: ${proposed_risk:.2f} (2.0% SL)")
    print(f"   🏆 Configuração Atual: ${current_risk:.2f} (1.5% SL)")
    print(f"   📊 Diferença: +${proposed_risk - current_risk:.2f} (+{((proposed_risk/current_risk)-1)*100:.0f}%)")
    
    print(f"\n💎 Potencial de Lucro por Trade:")
    proposed_profit = proposed_capital_per_trade * proposed.take_profit_pct
    current_profit = current_capital_per_trade * current.take_profit_pct
    
    print(f"   📈 Estratégia Proposta: ${proposed_profit:.2f} (18.0% TP)")
    print(f"   🏆 Configuração Atual: ${current_profit:.2f} (12.0% TP)")
    print(f"   📊 Diferença: +${proposed_profit - current_profit:.2f} (+{((proposed_profit/current_profit)-1)*100:.0f}%)")
    
    # Risk/Reward Ratio
    print(f"\n⚖️ Risk/Reward Ratio:")
    proposed_rr = proposed.take_profit_pct / proposed.stop_loss_pct
    current_rr = current.take_profit_pct / current.stop_loss_pct
    
    print(f"   📈 Estratégia Proposta: {proposed_rr:.1f}:1 (18%/2%)")
    print(f"   🏆 Configuração Atual: {current_rr:.1f}:1 (12%/1.5%)")
    print(f"   📊 Diferença: {proposed_rr - current_rr:+.1f}")
    
    print("\n🔍 ANÁLISE DE PARÂMETROS:")
    print("=" * 50)
    
    print("📊 Confluence (Critérios de Entrada):")
    print(f"   📈 Proposta: {proposed.min_confluence}/10 ({proposed.min_confluence*10:.0f}% confluence)")
    print(f"   🏆 Atual: {current.min_confluence}/10 ({current.min_confluence*10:.0f}% confluence)")
    
    if proposed.min_confluence > current.min_confluence:
        print(f"   ⚠️  MAIS RESTRITIVO: -{((proposed.min_confluence - current.min_confluence) * 10):.0f}% oportunidades")
    else:
        print(f"   ✅ MENOS RESTRITIVO: +{((current.min_confluence - proposed.min_confluence) * 10):.0f}% oportunidades")
    
    print(f"\n📈 Volume Filter:")
    print(f"   📈 Proposta: {proposed.volume_multiplier}x média")
    print(f"   🏆 Atual: {current.volume_multiplier}x média")
    
    if proposed.volume_multiplier > current.volume_multiplier:
        print(f"   ⚠️  MAIS RESTRITIVO: Filtra mais trades")
    else:
        print(f"   ✅ MENOS RESTRITIVO: Permite mais trades")
    
    print(f"\n🎯 ATR Range (Volatilidade):")
    print(f"   📈 Proposta: {proposed.atr_min_pct}% - {proposed.atr_max_pct}%")
    print(f"   🏆 Atual: {current.atr_min_pct}% - {current.atr_max_pct}%")
    
    proposed_range = proposed.atr_max_pct - proposed.atr_min_pct
    current_range = current.atr_max_pct - current.atr_min_pct
    
    print(f"   📊 Range Proposto: {proposed_range:.2f}% ({((proposed_range/current_range)-1)*100:+.0f}% vs atual)")
    
    print("\n⚡ ANÁLISE DE FREQUENCY vs QUALITY:")
    print("=" * 50)
    
    # Estimativa de frequência baseada nos parâmetros
    current_freq_score = (10 - current.min_confluence) + (2 - current.volume_multiplier) + (current.atr_max_pct - current.atr_min_pct)
    proposed_freq_score = (10 - proposed.min_confluence) + (2 - proposed.volume_multiplier) + (proposed.atr_max_pct - proposed.atr_min_pct)
    
    print(f"🔢 Frequency Score (maior = mais trades):")
    print(f"   📈 Estratégia Proposta: {proposed_freq_score:.1f}")
    print(f"   🏆 Configuração Atual: {current_freq_score:.1f}")
    
    if proposed_freq_score > current_freq_score:
        print(f"   📈 MAIS TRADES ESPERADOS: +{((proposed_freq_score/current_freq_score)-1)*100:.0f}%")
    else:
        print(f"   📉 MENOS TRADES ESPERADOS: {((proposed_freq_score/current_freq_score)-1)*100:.0f}%")
    
    print("\n💸 ANÁLISE DE TAXAS HYPERLIQUID:")
    print("=" * 50)
    
    # Custo base por trade na Hyperliquid
    hyperliquid_fee_rate = 0.0007  # 0.07% total (maker + taker + funding)
    
    proposed_fee_per_trade = proposed_capital_per_trade * hyperliquid_fee_rate
    current_fee_per_trade = current_capital_per_trade * hyperliquid_fee_rate
    
    print(f"💳 Taxa por Trade:")
    print(f"   📈 Estratégia Proposta: ${proposed_fee_per_trade:.3f}")
    print(f"   🏆 Configuração Atual: ${current_fee_per_trade:.3f}")
    print(f"   📊 Diferença: +${proposed_fee_per_trade - current_fee_per_trade:.3f}")
    
    # Break-even analysis
    print(f"\n⚖️ Break-even Analysis:")
    proposed_min_profit_needed = proposed_fee_per_trade / proposed_capital_per_trade * 100
    current_min_profit_needed = current_fee_per_trade / current_capital_per_trade * 100
    
    print(f"   📈 Proposta precisa de: {proposed_min_profit_needed:.3f}% lucro para break-even")
    print(f"   🏆 Atual precisa de: {current_min_profit_needed:.3f}% lucro para break-even")
    
    print("\n🎯 PROJEÇÃO DE PERFORMANCE:")
    print("=" * 50)
    
    # Simulação simplificada baseada nos nossos dados
    base_roi = 9480.4  # ROI da configuração vencedora atual
    
    # Ajustes estimados
    leverage_impact = (proposed.leverage / current.leverage) - 1  # +33% com 4x vs 3x
    confluence_impact = (current.min_confluence / proposed.min_confluence) - 1  # -55% trades com 5.5 vs 2.5
    rr_impact = (proposed_rr / current_rr) - 1  # +12.5% com melhor R:R
    
    print(f"📊 Impactos Estimados:")
    print(f"   ⚡ Leverage (4x vs 3x): {leverage_impact:+.0%}")
    print(f"   🎯 Confluence (5.5 vs 2.5): {confluence_impact:+.0%} trades")
    print(f"   💎 Risk/Reward: {rr_impact:+.0%}")
    
    # Projeção conservadora
    estimated_roi = base_roi * (1 + leverage_impact) * (1 + confluence_impact * 0.5) * (1 + rr_impact * 0.3)
    
    print(f"\n🏆 PROJEÇÃO FINAL:")
    print(f"   🏅 ROI Atual (Vencedora): +{base_roi:.0f}%")
    print(f"   📈 ROI Estimado (Proposta): +{estimated_roi:.0f}%")
    print(f"   📊 Diferença Projetada: {estimated_roi - base_roi:+.0f}%")
    
    if estimated_roi > base_roi:
        print(f"   ✅ POTENCIAL MELHORIA: +{((estimated_roi/base_roi)-1)*100:.0f}%")
    else:
        print(f"   ⚠️  POTENCIAL PIORA: {((estimated_roi/base_roi)-1)*100:.0f}%")
    
    print("\n🎯 RECOMENDAÇÃO:")
    print("=" * 30)
    
    if estimated_roi > base_roi * 1.1:  # Se >10% melhor
        print("✅ TESTE RECOMENDADO: Estratégia promissora!")
        print("💡 Vamos implementar e testar com dados reais.")
    elif estimated_roi > base_roi * 0.9:  # Se entre -10% e +10%
        print("🔶 TESTE CAUTELOSO: Resultado incerto.")
        print("💡 Pode valer testar com amostra menor.")
    else:
        print("❌ NÃO RECOMENDADO: Likely pior que atual.")
        print("💡 Manter configuração vencedora atual.")
    
    return {
        'proposed': proposed,
        'current': current,
        'estimated_roi': estimated_roi,
        'base_roi': base_roi,
        'recommendation': 'test' if estimated_roi > base_roi * 1.1 else 'caution' if estimated_roi > base_roi * 0.9 else 'skip'
    }

def main():
    """Executa análise completa da estratégia proposta"""
    print("🔬 INICIANDO ANÁLISE DETALHADA DA ESTRATÉGIA PROPOSTA")
    print()
    
    result = calculate_strategy_metrics()
    
    print(f"\n📁 Análise concluída!")
    print(f"💰 Capital por trade: $256 (4x leverage)")
    print(f"📊 Confluence: 55% (5.5/10 critérios)")
    print(f"⚖️  R:R Ratio: 9:1 (18%/2%)")
    print(f"🎯 Projeção: {result['recommendation'].upper()}")

if __name__ == "__main__":
    main()
