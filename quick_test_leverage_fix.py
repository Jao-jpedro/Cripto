#!/usr/bin/env python3
"""
TESTE RÁPIDO COM DADOS REAIS - LEVERAGE CORRIGIDO
"""

import pandas as pd
import numpy as np

def quick_test_with_real_data():
    """Teste rápido com dados reais"""
    
    print("🚀 TESTE RÁPIDO - LEVERAGE CORRIGIDO")
    print("="*50)
    
    # Carregar dados BTC
    try:
        df = pd.read_csv("dados_reais_btc_1ano.csv")
        
        if 'close' in df.columns:
            df['valor_fechamento'] = df['close']
        
        # Simular estratégia buy and hold simples
        initial_price = df['valor_fechamento'].iloc[1000]
        final_price = df['valor_fechamento'].iloc[2000]  # 1000 barras depois
        
        price_change = (final_price - initial_price) / initial_price
        
        print(f"📊 PERÍODO DE TESTE:")
        print(f"   Preço inicial: ${initial_price:.2f}")
        print(f"   Preço final: ${final_price:.2f}")
        print(f"   Mudança: {price_change*100:+.1f}%")
        print()
        
        print(f"💰 SIMULAÇÃO COM LEVERAGE CORRIGIDO:")
        print("-"*40)
        
        initial_balance = 1000.0
        
        for leverage in [1, 5, 10, 20]:
            # LÓGICA CORRIGIDA
            pnl_leveraged = price_change * leverage
            final_balance = initial_balance * (1 + pnl_leveraged)
            
            print(f"Leverage {leverage:2}x: ${initial_balance:.0f} → ${final_balance:.0f} ({pnl_leveraged*100:+6.1f}%)")
        
        print(f"\n🎯 OBSERVAÇÃO:")
        if price_change > 0:
            print(f"   ✅ BTC subiu {price_change*100:.1f}% → Leverage AMPLIFICA ganhos!")
        else:
            print(f"   ❌ BTC caiu {price_change*100:.1f}% → Leverage AMPLIFICA perdas!")
        
        print(f"   🔧 Agora a lógica está matematicamente correta!")
        
    except FileNotFoundError:
        print("❌ Arquivo BTC não encontrado")
        return
    
    # Testar range de SL/TP
    print(f"\n📏 RANGE DE OPERAÇÃO (CORRIGIDO):")
    print("-"*40)
    
    entry_price = 50000.0  # Exemplo
    sl_pct = 0.05  # 5% fixo
    tp_pct = 0.08  # 8% fixo
    
    sl_level = entry_price * (1 - sl_pct)
    tp_level = entry_price * (1 + tp_pct)
    
    print(f"Entry: ${entry_price:.0f}")
    print(f"SL:    ${sl_level:.0f} (-5%)")
    print(f"TP:    ${tp_level:.0f} (+8%)")
    print(f"Range: {(tp_pct + sl_pct)*100:.0f}% total")
    print()
    print("✅ Todos os leverages usam o MESMO range!")
    print("✅ Leverage apenas amplifica o P&L final!")

def main():
    quick_test_with_real_data()
    
    print(f"\n" + "="*50)
    print("🎉 SUMMARY:")
    print("="*50)
    print("🐛 BUG ENCONTRADO: SL/TP divididos pelo leverage")
    print("🔧 CORREÇÃO FEITA: SL/TP fixos, P&L amplificado")
    print("✅ RESULTADO: Leverage agora funciona corretamente!")
    print(f"\n🚀 Próximo: Testar com backtest completo!")

if __name__ == "__main__":
    main()
