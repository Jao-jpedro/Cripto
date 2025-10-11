#!/usr/bin/env python3
"""
🚀 ANÁLISE PRECISA - TAXAS HYPERLIQUID POR QUANTIDADE E TEMPO
Recálculo considerando o impacto real das taxas por volume e duração
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# TAXAS REAIS DA HYPERLIQUID (estrutura correta)
HYPERLIQUID_FEES = {
    # Taxas de Trading (por notional value)
    "maker_fee": 0.0002,     # 0.02% sobre valor notional
    "taker_fee": 0.0005,     # 0.05% sobre valor notional
    
    # Taxas de Funding (por notional value, a cada 8 horas)
    "funding_rate_typical": 0.0001,  # 0.01% típico
    "funding_rate_range": (-0.005, 0.005),  # -0.5% a +0.5%
    
    # Outras taxas
    "withdrawal_fee": 0.0,   # Sem taxa de retirada
    "gas_fee": 0.0          # Sem gas fees na L1
}

# Dados do DNA Micro Trailing (do teste anterior)
DNA_MICRO_RESULTS = {
    "total_trades": 25363,
    "roi_gross": 1400.1,  # %
    "capital_initial": 64.0,  # $
    "pnl_gross": 896.06,  # $
    "assets": 16,
    "avg_trades_per_asset": 1585,  # 25363 / 16
}

# Leverages por asset
LEVERAGE_BY_ASSET = {
    "BTC-USD": 40, "SOL-USD": 20, "ETH-USD": 25, "XRP-USD": 20,
    "DOGE-USD": 10, "AVAX-USD": 10, "ENA-USD": 10, "BNB-USD": 10,
    "SUI-USD": 10, "ADA-USD": 10, "LINK-USD": 10, "WLD-USD": 10,
    "AAVE-USD": 10, "CRV-USD": 10, "LTC-USD": 10, "NEAR-USD": 10
}

def calculate_precise_fees():
    print("🚀 ANÁLISE PRECISA: TAXAS HYPERLIQUID POR QUANTIDADE E TEMPO")
    print("="*80)
    
    # Dados base
    total_trades = DNA_MICRO_RESULTS["total_trades"]
    capital_per_asset = DNA_MICRO_RESULTS["capital_initial"] / 16  # $4 por asset
    
    print("📊 ESTRUTURA REAL DAS TAXAS HYPERLIQUID:")
    print("   💰 Taxas aplicadas sobre VALOR NOTIONAL das posições")
    print("   ⏰ Funding fees aplicadas A CADA 8 HORAS em posições abertas")
    print("   📈 Maker: 0.02% | Taker: 0.05% sobre valor da posição")
    print("   🔄 Funding: ~0.01% a cada 8h (pode ser -0.5% a +0.5%)")
    
    total_trading_fees = 0
    total_funding_fees = 0
    total_notional_volume = 0
    
    assets = list(LEVERAGE_BY_ASSET.keys())
    
    print(f"\n📊 CÁLCULO POR ASSET:")
    print("Asset        | Trades | Leverage | Notional/Trade | Trading Fees | Funding Fees | Total Fees")
    print("-" * 90)
    
    for asset in assets:
        leverage = LEVERAGE_BY_ASSET[asset]
        trades_per_asset = total_trades // 16  # Distribuição uniforme
        
        # Valor notional por trade
        position_size = capital_per_asset * leverage  # $4 * leverage
        notional_per_trade = position_size
        total_notional_asset = notional_per_trade * trades_per_asset
        
        # Trading fees (entrada + saída)
        # Assumindo 100% taker fees (market orders para trailing stop)
        trading_fee_per_trade = notional_per_trade * HYPERLIQUID_FEES["taker_fee"] * 2  # entrada + saída
        total_trading_fees_asset = trading_fee_per_trade * trades_per_asset
        
        # Funding fees
        # DNA Micro Trailing tem trades rápidos, estimativa: 2-4 horas médias por posição
        avg_hours_per_trade = 3  # Baseado no trailing agressivo
        funding_periods_per_trade = avg_hours_per_trade / 8  # Frações de 8h
        
        # Funding rate médio (conservador)
        avg_funding_rate = HYPERLIQUID_FEES["funding_rate_typical"]
        funding_fee_per_trade = notional_per_trade * avg_funding_rate * funding_periods_per_trade
        total_funding_fees_asset = funding_fee_per_trade * trades_per_asset
        
        total_fees_asset = total_trading_fees_asset + total_funding_fees_asset
        
        print(f"{asset:<12} | {trades_per_asset:>6} | {leverage:>8}x | ${notional_per_trade:>12.2f} | ${total_trading_fees_asset:>10.2f} | ${total_funding_fees_asset:>10.2f} | ${total_fees_asset:>9.2f}")
        
        total_trading_fees += total_trading_fees_asset
        total_funding_fees += total_funding_fees_asset
        total_notional_volume += total_notional_asset
    
    total_fees = total_trading_fees + total_funding_fees
    
    print("-" * 90)
    print(f"{'TOTAL':<12} | {total_trades:>6} | {'---':>8} | {'---':>12} | ${total_trading_fees:>10.2f} | ${total_funding_fees:>10.2f} | ${total_fees:>9.2f}")
    
    # Análise do impacto
    capital_initial = DNA_MICRO_RESULTS["capital_initial"]
    pnl_gross = DNA_MICRO_RESULTS["pnl_gross"]
    roi_gross = DNA_MICRO_RESULTS["roi_gross"]
    
    pnl_net = pnl_gross - total_fees
    roi_net = (pnl_net / capital_initial) * 100
    fee_impact = roi_gross - roi_net
    
    print(f"\n📊 RESUMO FINANCEIRO:")
    print(f"   💰 Capital Inicial: ${capital_initial:.2f}")
    print(f"   📈 PnL Bruto: ${pnl_gross:.2f}")
    print(f"   💸 Trading Fees: ${total_trading_fees:.2f}")
    print(f"   🔄 Funding Fees: ${total_funding_fees:.2f}")
    print(f"   🏦 Total Fees: ${total_fees:.2f}")
    print(f"   💎 PnL Líquido: ${pnl_net:.2f}")
    
    print(f"\n📊 IMPACTO NO ROI:")
    print(f"   📈 ROI Bruto: +{roi_gross:.1f}%")
    print(f"   💵 ROI Líquido: +{roi_net:.1f}%")
    print(f"   💸 Redução: -{fee_impact:.1f}%")
    print(f"   📊 Fees como % Capital: {(total_fees/capital_initial)*100:.2f}%")
    
    # Volume notional
    print(f"\n📊 VOLUME DE TRADING:")
    print(f"   💼 Volume Notional Total: ${total_notional_volume:,.2f}")
    print(f"   📊 Volume/Capital Ratio: {total_notional_volume/capital_initial:.1f}x")
    print(f"   🎯 Fees como % Volume: {(total_fees/total_notional_volume)*100:.4f}%")
    
    # Comparação com DNA Realista
    dna_realista_roi = 1377.3
    vs_realista = roi_net - dna_realista_roi
    
    print(f"\n🎯 COMPARAÇÃO COM DNA REALISTA:")
    print(f"   🥇 DNA Realista Original: +{dna_realista_roi:.1f}%")
    print(f"   🚀 DNA Micro (sem taxas): +{roi_gross:.1f}%")
    print(f"   💎 DNA Micro (com taxas): +{roi_net:.1f}%")
    
    if vs_realista > 0:
        print(f"   ✅ Diferença: +{vs_realista:.1f}% (AINDA VENCEDOR!)")
        status = "VENCEDOR"
    else:
        print(f"   ❌ Diferença: {vs_realista:.1f}% (INFERIOR)")
        status = "PERDEDOR"
    
    # Análise de sensibilidade
    print(f"\n🔬 ANÁLISE DE SENSIBILIDADE:")
    
    scenarios = [
        ("Cenário Otimista", 2, 0.7, 0.3),  # 2h médias, 70% maker
        ("Cenário Base", 3, 0.0, 1.0),      # 3h médias, 100% taker
        ("Cenário Conservador", 4, 0.0, 1.0), # 4h médias, 100% taker
        ("Cenário Pessimista", 6, 0.0, 1.0), # 6h médias, 100% taker, funding alto
    ]
    
    for scenario_name, avg_hours, maker_ratio, taker_ratio in scenarios:
        # Recalcular trading fees
        scenario_trading_fees = 0
        scenario_funding_fees = 0
        
        for asset in assets:
            leverage = LEVERAGE_BY_ASSET[asset]
            trades_per_asset = total_trades // 16
            notional_per_trade = capital_per_asset * leverage
            
            # Trading fees ajustadas
            maker_fee = notional_per_trade * HYPERLIQUID_FEES["maker_fee"] * 2 * maker_ratio
            taker_fee = notional_per_trade * HYPERLIQUID_FEES["taker_fee"] * 2 * taker_ratio
            trading_fee_per_trade = maker_fee + taker_fee
            scenario_trading_fees += trading_fee_per_trade * trades_per_asset
            
            # Funding fees ajustadas
            funding_periods = avg_hours / 8
            funding_rate = HYPERLIQUID_FEES["funding_rate_typical"]
            if scenario_name == "Cenário Pessimista":
                funding_rate *= 2  # Funding rate mais alto
            
            funding_fee_per_trade = notional_per_trade * funding_rate * funding_periods
            scenario_funding_fees += funding_fee_per_trade * trades_per_asset
        
        scenario_total_fees = scenario_trading_fees + scenario_funding_fees
        scenario_pnl_net = pnl_gross - scenario_total_fees
        scenario_roi_net = (scenario_pnl_net / capital_initial) * 100
        scenario_vs_realista = scenario_roi_net - dna_realista_roi
        
        status_emoji = "✅" if scenario_vs_realista > 0 else "❌"
        print(f"   {status_emoji} {scenario_name}: ROI {scenario_roi_net:+.1f}% | vs Realista: {scenario_vs_realista:+.1f}% | Fees: ${scenario_total_fees:.2f}")
    
    # Transformação de capital final
    final_value = capital_initial + pnl_net
    multiplier = final_value / capital_initial
    
    print(f"\n💎 TRANSFORMAÇÃO DE CAPITAL (TAXAS PRECISAS):")
    print(f"   💰 Capital Inicial: ${capital_initial:.2f}")
    print(f"   💸 Taxas Totais: ${total_fees:.2f}")
    print(f"   🚀 Valor Final: ${final_value:.2f}")
    print(f"   📈 Multiplicação: {multiplier:.2f}x em 1 ano")
    
    # Conclusão final
    print(f"\n🎊 CONCLUSÃO FINAL (ANÁLISE PRECISA):")
    if vs_realista > 0:
        print(f"   ✅ DNA MICRO TRAILING AINDA É VENCEDOR!")
        print(f"   🚀 Supera DNA Realista em {vs_realista:.1f}% mesmo com taxas precisas")
        print(f"   💎 ROI líquido de {roi_net:.1f}% valida a estratégia")
        print(f"   🏆 Implementação recomendada")
    else:
        print(f"   ❌ TAXAS INVIABILIZAM A ESTRATÉGIA")
        print(f"   📊 Déficit de {abs(vs_realista):.1f}% vs DNA Realista")
        print(f"   🔧 Necessário ajustar parâmetros significativamente")
        print(f"   ⚠️  Considerar estratégias com menos trades")
    
    print(f"\n💡 FATORES CRÍTICOS IDENTIFICADOS:")
    print(f"   📊 Volume notional alto: ${total_notional_volume:,.0f}")
    print(f"   🎯 Número de trades: {total_trades:,}")
    print(f"   ⏰ Duração média por trade: crítica para funding fees")
    print(f"   🏦 Trading fees dominam: ${total_trading_fees:.2f} vs ${total_funding_fees:.2f}")
    
    return {
        'roi_gross': roi_gross,
        'roi_net': roi_net,
        'total_fees': total_fees,
        'trading_fees': total_trading_fees,
        'funding_fees': total_funding_fees,
        'vs_realista': vs_realista,
        'status': status,
        'notional_volume': total_notional_volume
    }

if __name__ == "__main__":
    resultado = calculate_precise_fees()
