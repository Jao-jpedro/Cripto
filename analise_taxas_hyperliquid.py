#!/usr/bin/env python3
"""
Análise do impacto das taxas da Hyperliquid no ROI otimizado
"""

def analisar_taxas_hyperliquid():
    # Dados da otimização (sem taxas)
    trades_total = 50606
    roi_bruto = 2190.0
    capital_inicial = 10.0
    capital_final_bruto = 229.0
    pnl_bruto = 219.0
    
    # Taxas da Hyperliquid
    taxa_maker = 0.0002  # 0.02% para maker orders
    taxa_taker = 0.0005  # 0.05% para taker orders
    # Assumindo mix 50/50 maker/taker (conservador)
    taxa_media = (taxa_maker + taxa_taker) / 2  # 0.035%
    
    print("💰 ANÁLISE COM TAXAS DA HYPERLIQUID:")
    print()
    print("📊 DADOS BRUTOS (SEM TAXAS):")
    print(f"   Total de Trades: {trades_total:,}")
    print(f"   ROI Bruto: {roi_bruto:.1f}%")
    print(f"   Capital Final Bruto: ${capital_final_bruto:.2f}")
    print(f"   PnL Bruto: ${pnl_bruto:.2f}")
    print()
    print("🏦 ESTRUTURA DE TAXAS HYPERLIQUID:")
    print(f"   Taxa Maker: {taxa_maker*100:.3f}%")
    print(f"   Taxa Taker: {taxa_taker*100:.3f}%")
    print(f"   Taxa Média (50/50): {taxa_media*100:.3f}%")
    print()
    
    # Calcular taxas totais
    posicao_media = 1.0  # Baseado nos parâmetros do sistema ($1 por trade)
    taxas_totais = trades_total * posicao_media * taxa_media
    
    print("💸 IMPACTO DAS TAXAS:")
    print(f"   Posição Média por Trade: ${posicao_media:.2f}")
    print(f"   Taxa por Trade: ${posicao_media * taxa_media:.4f}")
    print(f"   Taxas Totais: ${taxas_totais:.2f}")
    print()
    
    # Calcular resultado líquido
    capital_final_liquido = capital_final_bruto - taxas_totais
    pnl_liquido = pnl_bruto - taxas_totais
    roi_liquido = (pnl_liquido / capital_inicial) * 100
    
    print("📈 RESULTADO LÍQUIDO (COM TAXAS):")
    print(f"   Capital Final Líquido: ${capital_final_liquido:.2f}")
    print(f"   PnL Líquido: ${pnl_liquido:.2f}")
    print(f"   ROI Líquido: {roi_liquido:.1f}%")
    print()
    print("🔍 COMPARAÇÃO:")
    print(f"   ROI Bruto: {roi_bruto:.1f}%")
    print(f"   ROI Líquido: {roi_liquido:.1f}%")
    print(f"   Impacto das Taxas: -{roi_bruto - roi_liquido:.1f} pontos percentuais")
    print(f"   Redução: {((roi_bruto - roi_liquido) / roi_bruto) * 100:.1f}%")
    print()
    
    # Análise de sensibilidade
    print("🔬 ANÁLISE DE SENSIBILIDADE:")
    cenarios = [
        ("Só Maker (0.02%)", taxa_maker),
        ("Só Taker (0.05%)", taxa_taker),
        ("70% Maker / 30% Taker", 0.7 * taxa_maker + 0.3 * taxa_taker),
        ("30% Maker / 70% Taker", 0.3 * taxa_maker + 0.7 * taxa_taker),
    ]
    
    for nome, taxa in cenarios:
        taxas_cenario = trades_total * posicao_media * taxa
        roi_cenario = ((pnl_bruto - taxas_cenario) / capital_inicial) * 100
        print(f"   {nome}: ROI {roi_cenario:.1f}% (taxas: ${taxas_cenario:.2f})")
    
    print()
    print("💡 CONCLUSÕES:")
    if roi_liquido > 1000:
        print(f"   ✅ Mesmo com taxas, ROI de {roi_liquido:.1f}% é EXCEPCIONAL!")
    print(f"   📊 Taxas consomem {((roi_bruto - roi_liquido) / roi_bruto) * 100:.1f}% do lucro")
    print(f"   🎯 Sistema continua sendo altamente lucrativo")
    
    return {
        'roi_bruto': roi_bruto,
        'roi_liquido': roi_liquido,
        'taxas_totais': taxas_totais,
        'impacto_percentual': ((roi_bruto - roi_liquido) / roi_bruto) * 100
    }

if __name__ == "__main__":
    resultado = analisar_taxas_hyperliquid()
