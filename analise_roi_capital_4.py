#!/usr/bin/env python3
"""
🔬 ANÁLISE ROI COM CAPITAL $4 E LEVERAGE 4X
==========================================
🎯 Verificar se ROI mantém-se igual com diferente capital/leverage
💰 Comparação: $64 (3x) vs $4 (4x) por trade

CENÁRIO ATUAL (VENCEDOR):
- Capital: $64
- Leverage: 3x  
- Notional: $192 por trade
- ROI: +9.480%

CENÁRIO PROPOSTO:
- Capital: $4
- Leverage: 4x
- Notional: $16 por trade
- ROI: ?
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

def analyze_roi_with_different_capital():
    """Analisa ROI com capital $4 e leverage 4x"""
    
    print("🔬 ANÁLISE ROI: $4 CAPITAL COM LEVERAGE 4X")
    print("=" * 60)
    print("🎯 Comparando com configuração vencedora")
    print()
    
    # Configuração atual (vencedora)
    current_capital = 64.0
    current_leverage = 3
    current_notional = current_capital * current_leverage  # $192
    
    # Configuração proposta
    new_capital = 4.0
    new_leverage = 4
    new_notional = new_capital * new_leverage  # $16
    
    # Parâmetros da estratégia (inalterados)
    stop_loss_pct = 0.015  # 1.5%
    take_profit_pct = 0.12  # 12%
    
    print("💰 COMPARAÇÃO DE CONFIGURAÇÕES:")
    print("=" * 40)
    
    print("📊 CONFIGURAÇÃO ATUAL (VENCEDORA):")
    print(f"   💵 Capital: ${current_capital}")
    print(f"   ⚡ Leverage: {current_leverage}x")
    print(f"   📊 Notional por Trade: ${current_notional}")
    print(f"   💰 ROI Alcançado: +9.480%")
    
    print(f"\n📊 CONFIGURAÇÃO PROPOSTA:")
    print(f"   💵 Capital: ${new_capital}")
    print(f"   ⚡ Leverage: {new_leverage}x")
    print(f"   📊 Notional por Trade: ${new_notional}")
    print(f"   💰 ROI Esperado: ?")
    
    print(f"\n🔍 ANÁLISE DE IMPACTO:")
    print("=" * 25)
    
    # Calcular diferenças
    notional_ratio = new_notional / current_notional
    capital_ratio = new_capital / current_capital
    leverage_ratio = new_leverage / current_leverage
    
    print(f"📊 Notional Ratio: {notional_ratio:.3f} ({new_notional}/{current_notional})")
    print(f"💰 Capital Ratio: {capital_ratio:.3f} ({new_capital}/{current_capital})")
    print(f"⚡ Leverage Ratio: {leverage_ratio:.3f} ({new_leverage}/{current_leverage})")
    
    print(f"\n💸 ANÁLISE DE RISCO E RETORNO:")
    print("=" * 35)
    
    # Calcular risco e retorno por trade para ambas configurações
    
    # Configuração atual
    current_risk_per_trade = current_notional * stop_loss_pct  # $2.88
    current_profit_per_trade = current_notional * take_profit_pct  # $23.04
    current_risk_pct_capital = (current_risk_per_trade / current_capital) * 100  # 4.5%
    current_profit_pct_capital = (current_profit_per_trade / current_capital) * 100  # 36%
    
    # Configuração proposta
    new_risk_per_trade = new_notional * stop_loss_pct  # $0.24
    new_profit_per_trade = new_notional * take_profit_pct  # $1.92
    new_risk_pct_capital = (new_risk_per_trade / new_capital) * 100  # 6%
    new_profit_pct_capital = (new_profit_per_trade / new_capital) * 100  # 48%
    
    print("🔴 RISCO POR TRADE:")
    print(f"   💰 Atual: ${current_risk_per_trade:.2f} ({current_risk_pct_capital:.1f}% do capital)")
    print(f"   🎯 Proposta: ${new_risk_per_trade:.2f} ({new_risk_pct_capital:.1f}% do capital)")
    
    print(f"\n🟢 LUCRO POR TRADE:")
    print(f"   💰 Atual: ${current_profit_per_trade:.2f} ({current_profit_pct_capital:.1f}% do capital)")
    print(f"   🎯 Proposta: ${new_profit_per_trade:.2f} ({new_profit_pct_capital:.1f}% do capital)")
    
    print(f"\n📊 ANÁLISE DE TAXAS:")
    print("=" * 25)
    
    # Taxas Hyperliquid: 0.08% do notional
    hyperliquid_fee_rate = 0.0008  # 0.08%
    
    current_fees = current_notional * hyperliquid_fee_rate
    new_fees = new_notional * hyperliquid_fee_rate
    
    print(f"💸 Taxas por Trade:")
    print(f"   💰 Atual: ${current_fees:.4f} ({(current_fees/current_capital)*100:.3f}% do capital)")
    print(f"   🎯 Proposta: ${new_fees:.4f} ({(new_fees/new_capital)*100:.3f}% do capital)")
    
    # Break-even analysis
    current_breakeven = (current_fees / current_notional) * 100
    new_breakeven = (new_fees / new_notional) * 100
    
    print(f"\n⚖️ BREAK-EVEN:")
    print(f"   💰 Atual: {current_breakeven:.3f}% lucro mínimo")
    print(f"   🎯 Proposta: {new_breakeven:.3f}% lucro mínimo")
    
    print(f"\n🧮 CÁLCULO DE ROI TEÓRICO:")
    print("=" * 35)
    
    # Baseado nos dados da estratégia vencedora
    total_trades = 1287  # Total de trades no ano
    winning_trades = int(total_trades * 0.19)  # 19% win rate médio
    losing_trades = total_trades - winning_trades
    
    print(f"📊 Dados da Estratégia Vencedora:")
    print(f"   🎯 Total Trades: {total_trades}")
    print(f"   ✅ Trades Vencedores: {winning_trades} (19%)")
    print(f"   ❌ Trades Perdedores: {losing_trades} (81%)")
    
    # Calcular PnL para configuração atual
    current_gross_profit = winning_trades * current_profit_per_trade
    current_gross_loss = losing_trades * current_risk_per_trade
    current_total_fees = total_trades * current_fees
    current_net_pnl = current_gross_profit - current_gross_loss - current_total_fees
    current_roi = (current_net_pnl / current_capital) * 100
    
    print(f"\n💰 CONFIGURAÇÃO ATUAL:")
    print(f"   🟢 Lucro Bruto: ${current_gross_profit:.0f}")
    print(f"   🔴 Perda Bruta: ${current_gross_loss:.0f}")
    print(f"   💸 Taxas Totais: ${current_total_fees:.0f}")
    print(f"   💎 PnL Líquido: ${current_net_pnl:.0f}")
    print(f"   📈 ROI Calculado: {current_roi:.0f}% (vs +9.480% real)")
    
    # Calcular PnL para configuração proposta
    new_gross_profit = winning_trades * new_profit_per_trade
    new_gross_loss = losing_trades * new_risk_per_trade
    new_total_fees = total_trades * new_fees
    new_net_pnl = new_gross_profit - new_gross_loss - new_total_fees
    new_roi = (new_net_pnl / new_capital) * 100
    
    print(f"\n🎯 CONFIGURAÇÃO PROPOSTA:")
    print(f"   🟢 Lucro Bruto: ${new_gross_profit:.2f}")
    print(f"   🔴 Perda Bruta: ${new_gross_loss:.2f}")
    print(f"   💸 Taxas Totais: ${new_total_fees:.2f}")
    print(f"   💎 PnL Líquido: ${new_net_pnl:.2f}")
    print(f"   📈 ROI Calculado: {new_roi:.0f}%")
    
    print(f"\n🏆 COMPARAÇÃO DIRETA:")
    print("=" * 25)
    
    roi_difference = new_roi - 9480.4
    roi_ratio = new_roi / 9480.4
    
    print(f"📊 ROI Configuração Atual: +9.480%")
    print(f"📊 ROI Configuração Proposta: {new_roi:+.0f}%")
    print(f"📊 Diferença: {roi_difference:+.0f}%")
    print(f"📊 Ratio: {roi_ratio:.3f}x")
    
    if abs(roi_difference) < 100:  # Diferença < 100%
        print("✅ ROI PRATICAMENTE IGUAL!")
    elif new_roi > 9480.4:
        print("🚀 ROI SUPERIOR!")
    else:
        print("⚠️ ROI INFERIOR!")
    
    print(f"\n🔍 ANÁLISE DETALHADA:")
    print("=" * 25)
    
    print("💡 FATORES QUE AFETAM O ROI:")
    
    # O ROI percentual deve ser igual pois:
    # 1. Mesma estratégia (mesmo win rate, mesmo SL/TP %)
    # 2. Leverage compensa o capital menor
    # 3. Taxas são proporcionais ao notional
    
    leverage_compensation = new_leverage / current_leverage  # 4/3 = 1.33
    capital_reduction = new_capital / current_capital  # 4/64 = 0.0625
    effective_multiplier = leverage_compensation * capital_reduction  # 1.33 * 0.0625 = 0.083
    
    print(f"   ⚡ Leverage aumentou: {leverage_compensation:.3f}x")
    print(f"   💰 Capital diminuiu: {capital_reduction:.3f}x")
    print(f"   📊 Efeito Combinado: {effective_multiplier:.3f}x")
    print(f"   💸 Notional Final: {notional_ratio:.3f}x do original")
    
    print(f"\n🎯 CONCLUSÃO TEÓRICA:")
    print("=" * 25)
    
    theoretical_roi = 9480.4  # Deveria ser igual
    
    print("📊 TEORICAMENTE:")
    print("   ✅ Mesmo SL/TP % = Mesmo risco/retorno relativo")
    print("   ✅ Leverage compensa capital menor")
    print("   ✅ Taxas proporcionais ao notional")
    print("   ✅ ROI % deveria ser IGUAL")
    
    print(f"\n⚠️ DIFERENÇAS PRÁTICAS:")
    print("   🔸 Notional menor = menor exposição absoluta")
    print("   🔸 Menos impacto de slippage")
    print("   🔸 Execução mais rápida")
    print("   🔸 Risco absoluto menor")
    
    print(f"\n📋 RESPOSTA DIRETA:")
    print("=" * 25)
    
    if abs(new_roi - 9480.4) < 500:  # Diferença < 500%
        answer = "SIM"
        explanation = "ROI praticamente igual"
    else:
        answer = "NÃO"
        explanation = f"ROI diferente: {new_roi:.0f}%"
    
    print(f"🎯 {answer}, {explanation}")
    print(f"💰 Com $4 capital + 4x leverage: ~{new_roi:.0f}% ROI")
    print(f"📊 vs $64 capital + 3x leverage: +9.480% ROI")
    
    return {
        "new_capital": new_capital,
        "new_leverage": new_leverage,
        "new_notional": new_notional,
        "new_roi": new_roi,
        "current_roi": 9480.4,
        "roi_difference": roi_difference,
        "same_roi": abs(roi_difference) < 500
    }

def main():
    """Executa análise de ROI com capital diferente"""
    print("🔬 INICIANDO ANÁLISE DE ROI COM $4 CAPITAL...")
    print()
    
    result = analyze_roi_with_different_capital()
    
    print(f"\n🎊 RESULTADO FINAL:")
    print(f"💰 Capital $4 + Leverage 4x = ROI ~{result['new_roi']:.0f}%")
    
    if result['same_roi']:
        print("✅ ROI é PRATICAMENTE IGUAL!")
    else:
        print(f"⚠️ ROI é DIFERENTE ({result['roi_difference']:+.0f}%)")

if __name__ == "__main__":
    main()
