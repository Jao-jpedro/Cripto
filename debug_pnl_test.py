#!/usr/bin/env python3
"""
DEBUG VERSION - 1 TRADE TEST
"""

def test_pnl_calculation():
    print("🧪 TESTE P&L CALCULATION")
    print("-" * 30)
    
    # Parâmetros do teste
    entry_price = 100.0
    exit_price = 110.0  # +10%
    leverage = 10
    initial_balance = 1000.0
    
    # Método 1: Cálculo direto
    price_change_pct = (exit_price - entry_price) / entry_price
    pnl_with_leverage = price_change_pct * leverage
    final_balance_method1 = initial_balance * (1 + pnl_with_leverage)
    
    print(f"MÉTODO 1 (Direto):")
    print(f"   Price change: {price_change_pct*100:+.1f}%")
    print(f"   P&L leveraged: {pnl_with_leverage*100:+.1f}%")
    print(f"   Final balance: ${final_balance_method1:.2f}")
    
    # Método 2: Como exchange real
    position_size = initial_balance * leverage  # $10,000 com leverage 10x
    pnl_absolute = position_size * price_change_pct  # $1,000
    final_balance_method2 = initial_balance + pnl_absolute
    
    print(f"\nMÉTODO 2 (Exchange):")
    print(f"   Position size: ${position_size:.2f}")
    print(f"   P&L absolute: ${pnl_absolute:+.2f}")
    print(f"   Final balance: ${final_balance_method2:.2f}")
    
    # Método 3: Margem (como Hyperliquid)
    margin_used = initial_balance / leverage  # $100 de margem para $1000 de posição
    position_value = initial_balance  # $1000 de posição
    pnl_margin = position_value * price_change_pct  # $100
    final_balance_method3 = initial_balance + pnl_margin
    
    print(f"\nMÉTODO 3 (Margem):")
    print(f"   Margin used: ${margin_used:.2f}")
    print(f"   Position value: ${position_value:.2f}")
    print(f"   P&L: ${pnl_margin:+.2f}")
    print(f"   Final balance: ${final_balance_method3:.2f}")
    
    print(f"\n🎯 TODOS DEVERIAM DAR O MESMO RESULTADO!")
    print(f"   Resultado esperado: ${final_balance_method1:.2f}")

if __name__ == "__main__":
    test_pnl_calculation()
