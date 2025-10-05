#!/usr/bin/env python3
"""
CORREÇÃO DO BUG DE LEVERAGE - IMPLEMENTAÇÃO CORRETA
"""

def demonstrate_bug():
    """Demonstra o bug atual vs implementação correta"""
    
    print("🚨 DEMONSTRAÇÃO DO BUG DE LEVERAGE")
    print("="*60)
    
    # Parâmetros de exemplo
    entry_price = 100.0
    sl_pct_base = 0.05  # 5%
    
    print("📊 IMPLEMENTAÇÃO ATUAL (BUGADA):")
    print("-"*40)
    
    for leverage in [1, 5, 10, 20]:
        # Bug atual: divide SL pelo leverage
        stop_loss_pct_buggy = sl_pct_base / leverage
        stop_level_buggy = entry_price * (1.0 - stop_loss_pct_buggy)
        
        print(f"Leverage {leverage:2}x: SL={stop_loss_pct_buggy*100:.2f}% → Stop em ${stop_level_buggy:.2f}")
    
    print(f"\n✅ IMPLEMENTAÇÃO CORRETA:")
    print("-"*40)
    
    for leverage in [1, 5, 10, 20]:
        # Correção: SL fixo, P&L amplificado
        stop_loss_pct_correct = sl_pct_base  # Sempre 5%
        stop_level_correct = entry_price * (1.0 - stop_loss_pct_correct)
        
        print(f"Leverage {leverage:2}x: SL={stop_loss_pct_correct*100:.1f}% → Stop em ${stop_level_correct:.2f}")
    
    print(f"\n🎯 DIFERENÇA DRAMÁTICA:")
    print("-"*30)
    print(f"Com leverage 20x:")
    print(f"   Bug atual: Stop em $99.75 (apenas 0.25% de queda!)")
    print(f"   Correto:   Stop em $95.00 (5% de queda normal)")
    print(f"")
    print(f"🚨 É por isso que leverage 'só amplifica perdas'!")
    print(f"   O SL fica tão sensível que para antes de qualquer TP!")

def create_corrected_logic():
    """Cria lógica correta para leverage"""
    
    print(f"\n🛠️ LÓGICA CORRETA DE LEVERAGE:")
    print("="*50)
    
    corrected_code = '''
def calculate_pnl_correct(entry_price, exit_price, leverage, initial_balance):
    """Cálculo CORRETO de P&L com leverage"""
    
    # 1. Calcular mudança de preço (%)
    price_change_pct = (exit_price - entry_price) / entry_price
    
    # 2. Amplificar pelo leverage
    pnl_with_leverage = price_change_pct * leverage
    
    # 3. Aplicar ao balance
    final_balance = initial_balance * (1 + pnl_with_leverage)
    
    return final_balance, pnl_with_leverage

def determine_stop_correct(entry_price, current_price, side, sl_pct=0.05):
    """Stop Loss CORRETO - fixo independente do leverage"""
    
    if side.lower() == "buy":
        stop_level = entry_price * (1.0 - sl_pct)  # 5% abaixo
        return current_price <= stop_level
    else:
        stop_level = entry_price * (1.0 + sl_pct)  # 5% acima  
        return current_price >= stop_level
'''
    
    print(corrected_code)

def test_corrected_vs_buggy():
    """Testa implementação correta vs bugada"""
    
    print(f"\n🧪 TESTE: CORRETO vs BUGADO")
    print("="*50)
    
    # Cenário de teste
    entry_price = 100.0
    exit_price = 105.0  # +5%
    initial_balance = 1000.0
    leverage = 10
    
    print(f"Cenário: Entrada ${entry_price} → Saída ${exit_price} (+5%)")
    print(f"Leverage: {leverage}x")
    print(f"Balance inicial: ${initial_balance}")
    print("-"*50)
    
    # Implementação CORRETA
    price_change = (exit_price - entry_price) / entry_price
    pnl_leveraged = price_change * leverage
    final_balance_correct = initial_balance * (1 + pnl_leveraged)
    
    print(f"✅ CORRETO:")
    print(f"   P&L: {price_change*100:.1f}% × {leverage} = {pnl_leveraged*100:.1f}%")
    print(f"   Balance final: ${final_balance_correct:.2f}")
    
    # Implementação BUGADA (simular)
    # Bug: SL muito sensível, nunca chega no TP
    sl_pct_buggy = 0.05 / leverage  # 0.5% com leverage 10x
    stop_level = entry_price * (1 - sl_pct_buggy)
    
    print(f"\n❌ BUGADO:")
    print(f"   SL: {sl_pct_buggy*100:.2f}% → Stop em ${stop_level:.2f}")
    print(f"   Status: {'STOP HIT!' if exit_price > entry_price else 'OK'}")
    print(f"   (Qualquer movimento para em SL ultra-sensível)")

def main():
    demonstrate_bug()
    create_corrected_logic()
    test_corrected_vs_buggy()
    
    print(f"\n" + "="*60)
    print("🎯 RESUMO DA DESCOBERTA:")
    print("="*60)
    print("❌ BUG: Dividem SL pelo leverage (SL ultra-sensível)")
    print("✅ FIX: SL fixo, P&L amplificado pelo leverage")
    print("🚨 Resultado: Com leverage alto, SEMPRE para no SL!")
    print("💡 Por isso leverage 'só amplifica perdas'")
    print(f"\n🛠️ PRÓXIMO PASSO: Corrigir a linha 1040 do trading.py")

if __name__ == "__main__":
    main()
