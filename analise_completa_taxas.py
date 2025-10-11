#!/usr/bin/env python3
"""
🚀 ANÁLISE COMPLETA - DNA MICRO TRAILING COM TAXAS HYPERLIQUID
Verificando se o ROI de +1.400,1% se mantém com custos reais
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# TAXAS REAIS DA HYPERLIQUID (Outubro 2025)
HYPERLIQUID_FEES = {
    # Taxas de Trading
    "maker_fee": 0.0002,     # 0.02% para maker orders
    "taker_fee": 0.0005,     # 0.05% para taker orders
    
    # Taxas de Funding (a cada 8 horas)
    "funding_rate_avg": 0.0001,  # 0.01% média por funding (8h)
    "funding_rate_max": 0.005,   # 0.5% máximo em momentos extremos
    
    # Taxas de Slippage (impacto no mercado)
    "slippage_small": 0.0001,   # 0.01% para posições pequenas
    "slippage_medium": 0.0003,  # 0.03% para posições médias
    "slippage_large": 0.0008,   # 0.08% para posições grandes
    
    # Taxa de Spread (bid-ask)
    "spread_tight": 0.0001,     # 0.01% em mercados líquidos
    "spread_normal": 0.0003,    # 0.03% em condições normais
    "spread_wide": 0.001,       # 0.1% em mercados voláteis
}

# LEVERAGES MÁXIMOS REAIS DA HYPERLIQUID
HYPERLIQUID_MAX_LEVERAGE = {
    "BTC-USD": 40, "SOL-USD": 20, "ETH-USD": 25, "XRP-USD": 20,
    "DOGE-USD": 10, "AVAX-USD": 10, "ENA-USD": 10, "BNB-USD": 10,
    "SUI-USD": 10, "ADA-USD": 10, "LINK-USD": 10, "WLD-USD": 10,
    "AAVE-USD": 10, "CRV-USD": 10, "LTC-USD": 10, "NEAR-USD": 10
}

# CONFIGURAÇÃO CAMPEÃ: DNA MICRO TRAILING
DNA_MICRO_TRAILING_CONFIG = {
    "name": "DNA Micro Trailing + Taxas",
    "stop_loss": 0.0012,           # 0.12%
    "ema_fast": 1, "ema_slow": 2, "rsi_period": 1,
    "min_confluence": 0.2,         # Confluência baixa
    "volume_multiplier": 0.006,    # Volume super sensível
    "atr_min": 0.0005, "atr_max": 40.0, "use_max_leverage": True,
    "exit_strategy": "micro_trailing",
    "trailing_stop_pct": 0.4,      # 0.4% trailing
    "min_profit": 0.3,             # 0.3% lucro mínimo (ajustado para fees)
    "micro_management": True,
    
    # CONFIGURAÇÕES DE TAXAS
    "include_fees": True,
    "trading_style": "aggressive",  # aggressive, moderate, conservative
    "position_size_category": "medium"  # small, medium, large
}

ASSETS = [
    "BTC-USD", "SOL-USD", "ETH-USD", "XRP-USD", "DOGE-USD",
    "AVAX-USD", "ENA-USD", "BNB-USD", "SUI-USD", "ADA-USD",
    "LINK-USD", "WLD-USD", "AAVE-USD", "CRV-USD", "LTC-USD", "NEAR-USD"
]

def calculate_hyperliquid_fees(entry_price, exit_price, position_size, hold_time_hours, config, asset):
    """
    Calcula todas as taxas da Hyperliquid para um trade
    """
    total_fees = 0
    fee_breakdown = {}
    
    # 1. Taxa de Entrada (Taker fee - assumindo market order)
    entry_fee = position_size * HYPERLIQUID_FEES["taker_fee"]
    total_fees += entry_fee
    fee_breakdown["entry_fee"] = entry_fee
    
    # 2. Taxa de Saída (Taker fee - assumindo market order)
    exit_value = (exit_price / entry_price) * position_size
    exit_fee = exit_value * HYPERLIQUID_FEES["taker_fee"]
    total_fees += exit_fee
    fee_breakdown["exit_fee"] = exit_fee
    
    # 3. Taxas de Funding (calculadas por período de 8 horas)
    funding_periods = max(1, hold_time_hours / 8)  # Mínimo 1 período
    
    # Funding rate varia baseado no ativo e volatilidade
    if asset in ["BTC-USD", "ETH-USD"]:
        funding_rate = HYPERLIQUID_FEES["funding_rate_avg"] * 0.8  # Ativos estáveis
    elif asset in ["SOL-USD", "XRP-USD"]:
        funding_rate = HYPERLIQUID_FEES["funding_rate_avg"]  # Ativos médios
    else:
        funding_rate = HYPERLIQUID_FEES["funding_rate_avg"] * 1.3  # Altcoins mais voláteis
    
    funding_fee = position_size * funding_rate * funding_periods
    total_fees += funding_fee
    fee_breakdown["funding_fee"] = funding_fee
    
    # 4. Slippage baseado no tamanho da posição
    size_category = config.get("position_size_category", "medium")
    if size_category == "small":
        slippage_rate = HYPERLIQUID_FEES["slippage_small"]
    elif size_category == "large":
        slippage_rate = HYPERLIQUID_FEES["slippage_large"]
    else:
        slippage_rate = HYPERLIQUID_FEES["slippage_medium"]
    
    # Slippage afeta entrada e saída
    entry_slippage = position_size * slippage_rate
    exit_slippage = exit_value * slippage_rate
    total_slippage = entry_slippage + exit_slippage
    total_fees += total_slippage
    fee_breakdown["slippage"] = total_slippage
    
    # 5. Spread (bid-ask) - afeta entrada e saída
    # Spread varia com volatilidade
    atr_estimate = abs((exit_price - entry_price) / entry_price)
    if atr_estimate > 0.02:  # Alta volatilidade
        spread_rate = HYPERLIQUID_FEES["spread_wide"]
    elif atr_estimate > 0.005:  # Volatilidade normal
        spread_rate = HYPERLIQUID_FEES["spread_normal"]
    else:  # Baixa volatilidade
        spread_rate = HYPERLIQUID_FEES["spread_tight"]
    
    entry_spread = position_size * spread_rate
    exit_spread = exit_value * spread_rate
    total_spread = entry_spread + exit_spread
    total_fees += total_spread
    fee_breakdown["spread"] = total_spread
    
    fee_breakdown["total_fees"] = total_fees
    fee_breakdown["fee_percentage"] = (total_fees / position_size) * 100
    
    return total_fees, fee_breakdown

def analyze_fees_impact_simplified():
    """Análise simplificada baseada nos resultados do DNA Micro Trailing"""
    print("🚀 ANÁLISE RÁPIDA: DNA MICRO TRAILING COM TAXAS HYPERLIQUID")
    print("="*80)
    
    # Dados do teste anterior (sem taxas)
    roi_bruto = 1400.1  # DNA Micro Trailing
    trades_estimados = 25363  # Total de trades
    capital_total = 64.0  # $4 por asset x 16 assets
    pnl_bruto = capital_total * (roi_bruto / 100)
    
    print("📊 DADOS BASE (SEM TAXAS):")
    print(f"   💰 Capital Inicial: ${capital_total:.2f}")
    print(f"   📈 ROI Bruto: +{roi_bruto:.1f}%")
    print(f"   💵 PnL Bruto: ${pnl_bruto:.2f}")
    print(f"   🎯 Trades Estimados: {trades_estimados:,}")
    
    print(f"\n💰 ESTRUTURA DE TAXAS HYPERLIQUID:")
    print(f"   📊 Maker Fee: {HYPERLIQUID_FEES['maker_fee']*100:.3f}%")
    print(f"   📈 Taker Fee: {HYPERLIQUID_FEES['taker_fee']*100:.3f}%")
    print(f"   🔄 Funding Rate Médio: {HYPERLIQUID_FEES['funding_rate_avg']*100:.3f}% / 8h")
    print(f"   💫 Slippage Médio: {HYPERLIQUID_FEES['slippage_medium']*100:.3f}%")
    print(f"   📏 Spread Normal: {HYPERLIQUID_FEES['spread_normal']*100:.3f}%")
    
    # Cálculo de taxas por trade (conservador)
    posicao_media_por_trade = capital_total * 20 / trades_estimados  # Leverage médio 20x
    
    # Fees por trade
    entry_fee_per_trade = posicao_media_por_trade * HYPERLIQUID_FEES["taker_fee"]
    exit_fee_per_trade = posicao_media_por_trade * HYPERLIQUID_FEES["taker_fee"] * 1.05  # 5% de profit médio
    funding_fee_per_trade = posicao_media_por_trade * HYPERLIQUID_FEES["funding_rate_avg"] * 2  # 2 períodos médios
    slippage_per_trade = posicao_media_por_trade * HYPERLIQUID_FEES["slippage_medium"] * 2  # Entrada + saída
    spread_per_trade = posicao_media_por_trade * HYPERLIQUID_FEES["spread_normal"] * 2  # Entrada + saída
    
    total_fees_per_trade = entry_fee_per_trade + exit_fee_per_trade + funding_fee_per_trade + slippage_per_trade + spread_per_trade
    total_fees_all_trades = total_fees_per_trade * trades_estimados
    
    print(f"\n💸 CÁLCULO DE TAXAS:")
    print(f"   💼 Posição Média por Trade: ${posicao_media_por_trade:.2f}")
    print(f"   📊 Taxa de Entrada: ${entry_fee_per_trade:.4f}")
    print(f"   📈 Taxa de Saída: ${exit_fee_per_trade:.4f}")
    print(f"   🔄 Taxa de Funding: ${funding_fee_per_trade:.4f}")
    print(f"   💫 Slippage: ${slippage_per_trade:.4f}")
    print(f"   📏 Spread: ${spread_per_trade:.4f}")
    print(f"   💸 Total por Trade: ${total_fees_per_trade:.4f}")
    print(f"   🏦 Taxas Totais: ${total_fees_all_trades:.2f}")
    
    # Resultado líquido
    pnl_liquido = pnl_bruto - total_fees_all_trades
    roi_liquido = (pnl_liquido / capital_total) * 100
    fee_impact = roi_bruto - roi_liquido
    fee_percentage = (total_fees_all_trades / capital_total) * 100
    
    print(f"\n📊 RESULTADO FINAL COM TAXAS:")
    print(f"   📈 ROI Bruto (sem taxas): +{roi_bruto:.1f}%")
    print(f"   💵 ROI Líquido (com taxas): +{roi_liquido:.1f}%")
    print(f"   💸 Impacto das Taxas: -{fee_impact:.1f}%")
    print(f"   📊 Taxas como % do Capital: {fee_percentage:.2f}%")
    
    # Comparação com benchmarks
    dna_realista_roi = 1377.3
    
    print(f"\n🎯 COMPARAÇÃO:")
    print(f"   🥇 DNA Realista Original: +{dna_realista_roi:.1f}%")
    print(f"   🚀 DNA Micro (sem taxas): +{roi_bruto:.1f}%")
    print(f"   💎 DNA Micro (com taxas): +{roi_liquido:.1f}%")
    
    vs_realista = roi_liquido - dna_realista_roi
    reduction_percentage = (fee_impact / roi_bruto) * 100
    
    if vs_realista > 0:
        print(f"   ✅ vs DNA Realista: +{vs_realista:.1f}% (AINDA MELHOR!)")
        status = "VENCEDOR"
    else:
        print(f"   ❌ vs DNA Realista: {vs_realista:.1f}%")
        status = "INFERIOR"
    
    print(f"   📉 Redução por Taxas: {reduction_percentage:.1f}%")
    
    # Transformação de capital
    final_value = capital_total + pnl_liquido
    multiplier = final_value / capital_total
    
    print(f"\n💎 TRANSFORMAÇÃO DE CAPITAL (COM TAXAS):")
    print(f"   💰 Capital Inicial: ${capital_total:.2f}")
    print(f"   💸 Taxas Pagas: ${total_fees_all_trades:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    # Análise de sensibilidade
    print(f"\n🔬 ANÁLISE DE SENSIBILIDADE:")
    
    scenarios = [
        ("Otimista (70% Maker)", 0.7 * HYPERLIQUID_FEES["maker_fee"] + 0.3 * HYPERLIQUID_FEES["taker_fee"]),
        ("Conservador (30% Maker)", 0.3 * HYPERLIQUID_FEES["maker_fee"] + 0.7 * HYPERLIQUID_FEES["taker_fee"]),
        ("Pessimista (100% Taker)", HYPERLIQUID_FEES["taker_fee"])
    ]
    
    for scenario_name, trading_fee in scenarios:
        scenario_fees_per_trade = (posicao_media_por_trade * trading_fee * 2.05 + 
                                 funding_fee_per_trade + slippage_per_trade + spread_per_trade)
        scenario_total_fees = scenario_fees_per_trade * trades_estimados
        scenario_pnl = pnl_bruto - scenario_total_fees
        scenario_roi = (scenario_pnl / capital_total) * 100
        scenario_vs_realista = scenario_roi - dna_realista_roi
        
        status_emoji = "✅" if scenario_vs_realista > 0 else "❌"
        print(f"   {status_emoji} {scenario_name}: ROI {scenario_roi:+.1f}% (vs Realista: {scenario_vs_realista:+.1f}%)")
    
    # Conclusão final
    print(f"\n🎊 CONCLUSÃO FINAL:")
    if vs_realista > 0:
        print(f"   ✅ DNA MICRO TRAILING CONTINUA SENDO VENCEDOR!")
        print(f"   🚀 Mesmo com taxas reais, supera DNA Realista em {vs_realista:.1f}%")
        print(f"   💎 ROI líquido de {roi_liquido:.1f}% é EXCEPCIONAL!")
        print(f"   🏆 Estratégia validada para implementação real")
    else:
        print(f"   ⚠️  Taxas impactaram significativamente a performance")
        print(f"   📊 ROI líquido ficou {abs(vs_realista):.1f}% abaixo do DNA Realista")
        print(f"   🔧 Necessário ajustar estratégia para compensar custos")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    if vs_realista > 0:
        print(f"   🎯 Implementar DNA Micro Trailing com configuração otimizada")
        print(f"   📊 Monitorar taxa de funding em tempo real")
        print(f"   ⚡ Usar limit orders quando possível para reduzir taxas")
        print(f"   🎛️ Ajustar min_profit para {DNA_MICRO_TRAILING_CONFIG['min_profit']}% considerando fees")
    else:
        print(f"   🔧 Ajustar parâmetros para reduzir número de trades")
        print(f"   📈 Aumentar min_profit para compensar taxas")
        print(f"   🎯 Considerar trailing stop menos agressivo")
    
    return {
        'roi_gross': roi_bruto,
        'roi_net': roi_liquido,
        'fee_impact': fee_impact,
        'total_fees': total_fees_all_trades,
        'vs_realista': vs_realista,
        'status': status,
        'final_value': final_value,
        'multiplier': multiplier
    }

if __name__ == "__main__":
    resultado = analyze_fees_impact_simplified()
