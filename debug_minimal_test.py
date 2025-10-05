#!/usr/bin/env python3
"""
TESTE MINIMAL - 1 TRADE PARA ENCONTRAR O BUG
"""

def test_single_trade():
    """Teste com um único trade para debug"""
    
    print("🧪 TESTE MINIMAL - 1 TRADE")
    print("="*40)
    
    # Dados do trade
    entry_price = 100.0
    exit_price = 110.0  # +10%
    leverage = 10
    balance = 1000.0
    
    print(f"Entrada: ${entry_price}")
    print(f"Saída: ${exit_price}")
    print(f"Leverage: {leverage}x")
    print(f"Balance inicial: ${balance}")
    print("-"*40)
    
    # Cálculo CORRETO
    price_change = (exit_price - entry_price) / entry_price
    pnl_percent = price_change * leverage
    pnl_value = balance * pnl_percent
    final_balance = balance + pnl_value
    
    print(f"📊 CÁLCULO CORRETO:")
    print(f"   Mudança preço: {price_change*100:+.1f}%")
    print(f"   P&L com leverage: {pnl_percent*100:+.1f}%")
    print(f"   P&L em valor: ${pnl_value:+.2f}")
    print(f"   Balance final: ${final_balance:.2f}")
    
    return final_balance

def analyze_trading_py_logic():
    """Analisa a lógica do trading.py para encontrar o bug"""
    
    print(f"\n🔍 ANALISANDO TRADING.PY")
    print("="*40)
    
    try:
        with open('/Users/joaoreis/Documents/GitHub/Cripto/trading.py', 'r') as f:
            content = f.read()
        
        # Procurar por funções de cálculo P&L
        lines = content.split('\n')
        
        pnl_related_lines = []
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['pnl', 'profit', 'loss', 'balance', 'leverage']):
                pnl_related_lines.append((i+1, line.strip()))
        
        print("📝 LINHAS RELACIONADAS A P&L/LEVERAGE:")
        for line_num, line in pnl_related_lines[:20]:  # Primeiras 20
            print(f"   {line_num:3}: {line}")
        
        # Procurar funções específicas
        if 'def calcular_pnl' in content:
            print(f"\n✅ Função calcular_pnl encontrada")
        else:
            print(f"\n❌ Função calcular_pnl NÃO encontrada")
        
        if 'def processar_trade' in content:
            print(f"✅ Função processar_trade encontrada")
        else:
            print(f"❌ Função processar_trade NÃO encontrada")
            
    except FileNotFoundError:
        print("❌ Arquivo trading.py não encontrado")

def create_debug_version():
    """Cria versão debug do trading para testar"""
    
    print(f"\n🛠️ CRIANDO VERSÃO DEBUG")
    print("="*40)
    
    debug_code = '''#!/usr/bin/env python3
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
    
    print(f"\\nMÉTODO 2 (Exchange):")
    print(f"   Position size: ${position_size:.2f}")
    print(f"   P&L absolute: ${pnl_absolute:+.2f}")
    print(f"   Final balance: ${final_balance_method2:.2f}")
    
    # Método 3: Margem (como Hyperliquid)
    margin_used = initial_balance / leverage  # $100 de margem para $1000 de posição
    position_value = initial_balance  # $1000 de posição
    pnl_margin = position_value * price_change_pct  # $100
    final_balance_method3 = initial_balance + pnl_margin
    
    print(f"\\nMÉTODO 3 (Margem):")
    print(f"   Margin used: ${margin_used:.2f}")
    print(f"   Position value: ${position_value:.2f}")
    print(f"   P&L: ${pnl_margin:+.2f}")
    print(f"   Final balance: ${final_balance_method3:.2f}")
    
    print(f"\\n🎯 TODOS DEVERIAM DAR O MESMO RESULTADO!")
    print(f"   Resultado esperado: ${final_balance_method1:.2f}")

if __name__ == "__main__":
    test_pnl_calculation()
'''
    
    with open('/Users/joaoreis/Documents/GitHub/Cripto/debug_pnl_test.py', 'w') as f:
        f.write(debug_code)
    
    print("✅ Arquivo debug_pnl_test.py criado")

def main():
    print("🚨 INVESTIGAÇÃO DO BUG DE LEVERAGE")
    print("="*50)
    print("🎯 O usuário está correto: algo está muito errado!")
    print("📊 BTC subiu 81.6%, leverage deveria AMPLIFICAR ganhos")
    print("❌ Mas nossa estratégia deu -100%")
    print()
    
    expected_balance = test_single_trade()
    analyze_trading_py_logic()
    create_debug_version()
    
    print(f"\n" + "="*50)
    print("🔧 PRÓXIMOS PASSOS:")
    print("="*50)
    print("1. Executar debug_pnl_test.py")
    print("2. Comparar com lógica do trading.py")
    print("3. Encontrar onde está o bug")
    print("4. Corrigir a implementação")
    print(f"5. Re-testar com dados reais")

if __name__ == "__main__":
    main()
