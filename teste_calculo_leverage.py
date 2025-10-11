#!/usr/bin/env python3
"""
🧮 TESTE: CÁLCULO CORRETO DE LEVERAGE
====================================
Esclarecendo como funciona o PnL com leverage
"""

def exemplo_leverage():
    """Exemplo prático de como funciona o leverage"""
    
    print("🎯 EXEMPLO: TRADE COM LEVERAGE")
    print("="*40)
    
    # Parâmetros
    capital_investido = 4.0
    leverage = 10
    preco_entrada = 100.0  # $100 por unidade
    movimento_preco = 0.12  # 12% de alta
    
    # Cálculos
    notional = capital_investido * leverage
    quantidade = notional / preco_entrada
    preco_saida = preco_entrada * (1 + movimento_preco)
    
    # PnL
    valor_final = quantidade * preco_saida
    pnl_bruto = valor_final - notional
    roi_percent = (pnl_bruto / capital_investido) * 100
    
    print(f"💰 Capital Investido: ${capital_investido}")
    print(f"⚡ Leverage: {leverage}x")
    print(f"📊 Notional: ${notional}")
    print(f"🎯 Quantidade comprada: {quantidade} unidades")
    print(f"📈 Preço entrada: ${preco_entrada}")
    print(f"📈 Preço saída (+12%): ${preco_saida}")
    print()
    print(f"💵 Valor da posição na saída: ${valor_final}")
    print(f"💵 Valor da posição na entrada: ${notional}")
    print(f"💰 PnL Bruto: ${pnl_bruto}")
    print(f"📊 ROI sobre capital: {roi_percent}%")
    print()
    
    # Conclusão
    if abs(pnl_bruto - 4.80) < 0.01:
        print("✅ CORRETO: PnL = $4,80 (120% ROI)")
    elif abs(pnl_bruto - 0.48) < 0.01:
        print("❌ ERRADO: PnL = $0,48 (12% ROI)")
    else:
        print(f"🤔 PnL = ${pnl_bruto}")
    
    print("\n" + "="*50)
    print("📝 EXPLICAÇÃO:")
    print("Com $4 e leverage 10x, você controla $40")
    print("Se o ativo sobe 12%, você ganha $4,80")
    print("Isso representa 120% de retorno sobre seus $4")
    print("Você NÃO ganha apenas $0,48!")

def exemplo_comparacao():
    """Comparação: com e sem leverage"""
    
    print("\n🔍 COMPARAÇÃO: COM E SEM LEVERAGE")
    print("="*45)
    
    capital = 4.0
    movimento = 0.12  # 12%
    
    # Sem leverage
    print(f"❌ SEM LEVERAGE:")
    print(f"   Capital: ${capital}")
    print(f"   Movimento: {movimento*100}%")
    print(f"   Ganho: ${capital * movimento} ({movimento*100}% ROI)")
    
    # Com leverage 10x
    print(f"\n✅ COM LEVERAGE 10x:")
    print(f"   Capital: ${capital}")
    print(f"   Notional: ${capital * 10}")
    print(f"   Movimento: {movimento*100}%")
    print(f"   Ganho: ${capital * 10 * movimento} ({movimento*10*100}% ROI)")

if __name__ == "__main__":
    exemplo_leverage()
    exemplo_comparacao()
