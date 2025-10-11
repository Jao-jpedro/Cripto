#!/usr/bin/env python3
"""
🔬 ANÁLISE CAPITAL $35 COM ENTRADAS DE $4
=========================================
🎯 Otimizar ROI com capital limitado e tamanho de entrada fixo
💰 Capital Total: $35
📊 Entrada por Trade: $4

CENÁRIOS A ANALISAR:
1. $4 capital + diferentes leverages
2. Gestão de risco com $35 total
3. Múltiplas posições simultâneas
4. Estratégia de compound
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

def analyze_capital_35_entries_4():
    """Analisa melhor estratégia com $35 capital e entradas $4"""
    
    print("🔬 ANÁLISE: $35 CAPITAL COM ENTRADAS $4")
    print("=" * 50)
    print("🎯 Encontrando estratégia ótima para capital limitado")
    print()
    
    # Parâmetros base
    total_capital = 35.0
    entry_size = 4.0
    max_positions = int(total_capital / entry_size)  # 8 posições máximo
    
    print("💰 SITUAÇÃO ATUAL:")
    print("=" * 20)
    print(f"💵 Capital Total: ${total_capital}")
    print(f"📊 Entrada por Trade: ${entry_size}")
    print(f"🎯 Máximo Posições Simultâneas: {max_positions}")
    print(f"📈 Capital Restante: ${total_capital - (max_positions * entry_size)}")
    
    # Parâmetros da estratégia vencedora
    stop_loss_pct = 0.015  # 1.5%
    take_profit_pct = 0.12  # 12%
    win_rate = 0.19  # 19%
    
    print(f"\n📊 CENÁRIOS DE LEVERAGE:")
    print("=" * 30)
    
    # Testar diferentes leverages
    leverages = [1, 2, 3, 4, 5, 6, 8, 10]
    scenarios = []
    
    for leverage in leverages:
        notional = entry_size * leverage
        
        # Risco e retorno por trade
        risk_per_trade = notional * stop_loss_pct
        profit_per_trade = notional * take_profit_pct
        
        # Taxas Hyperliquid
        fee_rate = 0.0008  # 0.08%
        fees_per_trade = notional * fee_rate
        
        # Break-even
        breakeven_pct = (fees_per_trade / notional) * 100
        
        # ROI teórico (baseado na estratégia vencedora)
        # Usando frequência de trades da análise anterior
        annual_trades = 1287
        winning_trades = int(annual_trades * win_rate)
        losing_trades = annual_trades - winning_trades
        
        # PnL anual para UMA posição
        gross_profit = winning_trades * profit_per_trade
        gross_loss = losing_trades * risk_per_trade
        total_fees = annual_trades * fees_per_trade
        net_pnl = gross_profit - gross_loss - total_fees
        roi_single = (net_pnl / entry_size) * 100
        
        # Com múltiplas posições (se capital permitir)
        if max_positions >= 1:
            total_capital_used = min(max_positions * entry_size, total_capital)
            positions_possible = int(total_capital_used / entry_size)
            total_net_pnl = net_pnl * positions_possible
            roi_total_capital = (total_net_pnl / total_capital) * 100
        else:
            roi_total_capital = 0
            positions_possible = 0
        
        scenario = {
            'leverage': leverage,
            'notional': notional,
            'risk_per_trade': risk_per_trade,
            'profit_per_trade': profit_per_trade,
            'fees_per_trade': fees_per_trade,
            'breakeven_pct': breakeven_pct,
            'roi_single_position': roi_single,
            'positions_possible': positions_possible,
            'roi_total_capital': roi_total_capital,
            'net_pnl_annual': total_net_pnl if 'total_net_pnl' in locals() else net_pnl
        }
        scenarios.append(scenario)
        
        print(f"⚡ {leverage}x Leverage:")
        print(f"   📊 Notional: ${notional}")
        print(f"   🔴 Risco: ${risk_per_trade:.2f} ({(risk_per_trade/entry_size)*100:.1f}%)")
        print(f"   🟢 Lucro: ${profit_per_trade:.2f} ({(profit_per_trade/entry_size)*100:.1f}%)")
        print(f"   💸 Taxa: ${fees_per_trade:.4f} ({breakeven_pct:.3f}%)")
        print(f"   📈 ROI (1 pos): {roi_single:+.0f}%")
        print(f"   🎯 ROI (total): {roi_total_capital:+.0f}%")
        print()
    
    # Encontrar melhor cenário
    best_scenario = max(scenarios, key=lambda x: x['roi_total_capital'])
    
    print("🏆 MELHOR CENÁRIO:")
    print("=" * 20)
    print(f"⚡ Leverage Ótimo: {best_scenario['leverage']}x")
    print(f"📊 Notional por Trade: ${best_scenario['notional']}")
    print(f"🎯 Posições Simultâneas: {best_scenario['positions_possible']}")
    print(f"💰 Capital Utilizado: ${best_scenario['positions_possible'] * entry_size}")
    print(f"📈 ROI Total do Capital: {best_scenario['roi_total_capital']:+.0f}%")
    print(f"💎 PnL Anual Esperado: ${best_scenario['net_pnl_annual']:+.0f}")
    
    print(f"\n🔍 COMPARAÇÃO COM ESTRATÉGIA VENCEDORA:")
    print("=" * 45)
    
    # Estratégia vencedora: $64 capital, 3x leverage = +9,480% ROI
    winner_roi = 9480
    ratio_vs_winner = best_scenario['roi_total_capital'] / winner_roi
    
    print(f"🏅 Estratégia Vencedora: +{winner_roi}% ROI ($64 capital)")
    print(f"💡 Nossa Estratégia: {best_scenario['roi_total_capital']:+.0f}% ROI ($35 capital)")
    print(f"📊 Ratio: {ratio_vs_winner:.3f}x ({ratio_vs_winner*100:.1f}%)")
    
    if ratio_vs_winner > 0.5:
        print("✅ DESEMPENHO MUITO BOM (>50% da estratégia vencedora)")
    elif ratio_vs_winner > 0.3:
        print("🔶 DESEMPENHO RAZOÁVEL (30-50% da estratégia vencedora)")
    else:
        print("⚠️ DESEMPENHO LIMITADO (<30% da estratégia vencedora)")
    
    print(f"\n💡 ESTRATÉGIA DE GESTÃO DE RISCO:")
    print("=" * 35)
    
    # Calcular risco máximo por trade
    max_risk_per_trade = best_scenario['risk_per_trade']
    max_risk_percentage = (max_risk_per_trade / total_capital) * 100
    
    print(f"🔴 Risco Máximo por Trade: ${max_risk_per_trade:.2f}")
    print(f"📊 Risco % do Capital Total: {max_risk_percentage:.2f}%")
    
    if max_risk_percentage > 10:
        print("⚠️ RISCO ALTO (>10% por trade)")
        print("💡 Considere reduzir leverage ou diversificar")
    elif max_risk_percentage > 5:
        print("🔶 RISCO MODERADO (5-10% por trade)")
        print("✅ Aceitável com boa gestão")
    else:
        print("✅ RISCO BAIXO (<5% por trade)")
        print("🚀 Configuração conservadora")
    
    # Estratégia de compound
    print(f"\n🚀 ESTRATÉGIA DE CRESCIMENTO:")
    print("=" * 30)
    
    # Simulação de compound com reinvestimento
    months_to_double = 12 / (best_scenario['roi_total_capital'] / 100) if best_scenario['roi_total_capital'] > 0 else float('inf')
    
    if months_to_double < 12:
        print(f"📈 Tempo para dobrar capital: {months_to_double:.1f} meses")
        print("🎯 ESTRATÉGIA: Reinvestir lucros mensalmente")
        
        # Projeção de crescimento
        monthly_roi = best_scenario['roi_total_capital'] / 12 / 100
        projected_capitals = []
        current_capital = total_capital
        
        for month in range(1, 13):
            current_capital *= (1 + monthly_roi)
            projected_capitals.append(current_capital)
        
        print(f"\n📊 PROJEÇÃO DE CRESCIMENTO (COMPOUND):")
        for i, capital in enumerate(projected_capitals[:6], 1):
            print(f"   Mês {i:2d}: ${capital:.0f}")
        print(f"   ...")
        print(f"   Mês 12: ${projected_capitals[-1]:.0f}")
        
    else:
        print("⚠️ ROI baixo para estratégia de compound efetiva")
        print("💡 Foque em consistência e baixo risco")
    
    print(f"\n🎯 RECOMENDAÇÕES FINAIS:")
    print("=" * 25)
    
    print(f"✅ Use leverage {best_scenario['leverage']}x para otimizar ROI")
    print(f"✅ Mantenha entradas de $4 por trade")
    print(f"✅ Máximo {best_scenario['positions_possible']} posições simultâneas")
    print(f"✅ Reserve ${total_capital - (best_scenario['positions_possible'] * entry_size)} como margem de segurança")
    print(f"✅ ROI esperado: {best_scenario['roi_total_capital']:+.0f}% anual")
    
    # Configuração específica para trading.py
    print(f"\n⚙️ CONFIGURAÇÃO PARA TRADING.PY:")
    print("=" * 35)
    print(f"POSITION_SIZE = {entry_size}")
    print(f"MAX_LEVERAGE = {best_scenario['leverage']}")
    print(f"MAX_POSITIONS = {best_scenario['positions_possible']}")
    print(f"STOP_LOSS = {stop_loss_pct}")
    print(f"TAKE_PROFIT = {take_profit_pct}")
    
    return best_scenario

def analyze_multiple_strategies():
    """Analisa estratégias alternativas"""
    
    print(f"\n🔄 ESTRATÉGIAS ALTERNATIVAS:")
    print("=" * 30)
    
    total_capital = 35.0
    entry_size = 4.0
    
    # Estratégia 1: Conservative (lower leverage, mais posições)
    print("📊 ESTRATÉGIA CONSERVADORA:")
    print("   ⚡ Leverage: 2x")
    print("   🎯 Posições: 8 simultâneas")
    print("   🔴 Risco por trade: $0.12 (0.34% capital total)")
    print("   📈 ROI estimado: ~2.000%")
    
    # Estratégia 2: Aggressive (higher leverage, menos posições)
    print(f"\n📊 ESTRATÉGIA AGRESSIVA:")
    print("   ⚡ Leverage: 8x")
    print("   🎯 Posições: 4-6 simultâneas")
    print("   🔴 Risco por trade: $0.48 (1.37% capital total)")
    print("   📈 ROI estimado: ~8.000%")
    
    # Estratégia 3: Balanced
    print(f"\n📊 ESTRATÉGIA BALANCEADA:")
    print("   ⚡ Leverage: 4x")
    print("   🎯 Posições: 6-8 simultâneas")
    print("   🔴 Risco por trade: $0.24 (0.69% capital total)")
    print("   📈 ROI estimado: ~5.000%")

def main():
    """Executa análise completa"""
    print("🔬 INICIANDO ANÁLISE CAPITAL $35 ENTRADAS $4...")
    print()
    
    best_scenario = analyze_capital_35_entries_4()
    analyze_multiple_strategies()
    
    print(f"\n🎊 RESULTADO FINAL:")
    print(f"💰 Melhor configuração: {best_scenario['leverage']}x leverage")
    print(f"📈 ROI esperado: {best_scenario['roi_total_capital']:+.0f}%")
    print(f"🎯 Entradas de $4 são VIÁVEIS com seu capital!")

if __name__ == "__main__":
    main()
