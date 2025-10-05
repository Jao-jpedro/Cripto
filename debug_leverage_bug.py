#!/usr/bin/env python3
"""
Investigação da Lógica de Leverage - Algo está errado!
O usuário está certo: se SL causa perdas, leverage deveria amplificar ganhos também
"""

import os
import sys
import pandas as pd
import numpy as np

def investigate_leverage_logic():
    """Investiga se há bug na lógica de leverage"""
    
    print("🔍 INVESTIGAÇÃO: LÓGICA DE LEVERAGE SUSPEITA")
    print("🎯 O usuário está certo: algo não bate!")
    print("="*60)
    
    # Testar com dados simples para debug
    print("📊 TESTE MATEMÁTICO SIMPLES:")
    print("-"*40)
    
    # Simular cenários simples
    entry_price = 100.0
    
    scenarios = [
        ("Ganho 5%", 105.0),
        ("Perda 5%", 95.0),
        ("Ganho 10%", 110.0),
        ("Perda 10%", 90.0),
    ]
    
    leverages = [1, 3, 5, 10, 20]
    
    print("Cenário | Preço | Leverage | P&L % | Esperado")
    print("-" * 50)
    
    for scenario_name, exit_price in scenarios:
        price_change = (exit_price - entry_price) / entry_price
        
        for lev in leverages:
            pnl_with_leverage = price_change * lev
            
            print(f"{scenario_name:8} | {exit_price:5.0f} | {lev:8}x | {pnl_with_leverage*100:+6.1f}% | {'✅' if abs(pnl_with_leverage) > abs(price_change) else '❌'}")
    
    print(f"\n💡 CONCLUSÃO MATEMÁTICA:")
    print(f"   Se leverage AMPLIFICA perdas...")
    print(f"   Então DEVE amplificar ganhos na mesma proporção!")
    print(f"   Se não está acontecendo, há BUG na implementação!")

def test_leverage_bug_in_real_data():
    """Testa se há bug na implementação com dados reais"""
    
    print(f"\n🔬 TESTE COM DADOS REAIS - BUSCANDO O BUG")
    print("-"*50)
    
    filename = "dados_reais_btc_1ano.csv"
    if not os.path.exists(filename):
        print("❌ Arquivo BTC não encontrado")
        return
    
    df = pd.read_csv(filename)
    
    # Padronizar colunas
    column_mapping = {
        'open': 'valor_abertura',
        'high': 'valor_maximo',
        'low': 'valor_minimo',
        'close': 'valor_fechamento',
        'volume': 'volume'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    print(f"📊 Dados carregados: {len(df)} barras")
    
    # Calcular retornos diários
    df['retorno_diario'] = df['valor_fechamento'].pct_change()
    
    # Estatísticas dos retornos
    retornos = df['retorno_diario'].dropna()
    
    print(f"\n📈 ANÁLISE DOS RETORNOS DIÁRIOS:")
    print(f"   Retorno médio: {retornos.mean()*100:.4f}%")
    print(f"   Retornos positivos: {(retornos > 0).sum()}/{len(retornos)} ({(retornos > 0).mean()*100:.1f}%)")
    print(f"   Retornos negativos: {(retornos < 0).sum()}/{len(retornos)} ({(retornos < 0).mean()*100:.1f}%)")
    
    # Análise por leverage
    print(f"\n🔍 IMPACTO DO LEVERAGE NOS RETORNOS:")
    print("-"*40)
    
    for leverage in [1, 5, 10, 20]:
        retornos_lev = retornos * leverage
        
        # Contar dias que passariam de diferentes SLs
        sl_2pct = (retornos_lev < -0.02).sum()
        sl_5pct = (retornos_lev < -0.05).sum()
        sl_10pct = (retornos_lev < -0.10).sum()
        
        # Contar dias que passariam de diferentes TPs
        tp_5pct = (retornos_lev > 0.05).sum()
        tp_10pct = (retornos_lev > 0.10).sum()
        tp_20pct = (retornos_lev > 0.20).sum()
        
        # Retorno acumulado teórico
        retorno_acumulado = (1 + retornos_lev).prod() - 1
        
        print(f"Leverage {leverage:2}x:")
        print(f"   Retorno acumulado teórico: {retorno_acumulado*100:+7.1f}%")
        print(f"   Dias SL 2%/5%/10%: {sl_2pct}/{sl_5pct}/{sl_10pct}")
        print(f"   Dias TP 5%/10%/20%: {tp_5pct}/{tp_10pct}/{tp_20pct}")
        
        if leverage > 1:
            ratio_tp_sl = tp_10pct / max(sl_10pct, 1)
            print(f"   Ratio TP/SL (10%): {ratio_tp_sl:.2f}")
        print()

def simulate_correct_leverage():
    """Simula como DEVERIA funcionar o leverage correto"""
    
    print(f"🧪 SIMULAÇÃO: COMO DEVERIA FUNCIONAR O LEVERAGE")
    print("-"*50)
    
    # Carregar dados BTC
    filename = "dados_reais_btc_1ano.csv"
    if not os.path.exists(filename):
        print("❌ Arquivo BTC não encontrado")
        return
    
    df = pd.read_csv(filename)
    
    # Padronizar
    if 'close' in df.columns:
        df['valor_fechamento'] = df['close']
    
    # Simular estratégia simples com diferentes leverages
    print("💰 SIMULAÇÃO BUY AND HOLD COM LEVERAGE:")
    print("-"*40)
    
    initial_price = df['valor_fechamento'].iloc[100]  # Preço inicial
    final_price = df['valor_fechamento'].iloc[-100]   # Preço final
    
    price_change = (final_price - initial_price) / initial_price
    
    print(f"Preço inicial: ${initial_price:.2f}")
    print(f"Preço final: ${final_price:.2f}")
    print(f"Mudança de preço: {price_change*100:+.1f}%")
    print()
    
    initial_balance = 1000.0
    
    for leverage in [1, 5, 10, 20]:
        # Com leverage, o retorno deveria ser ampliado
        leveraged_return = price_change * leverage
        final_balance = initial_balance * (1 + leveraged_return)
        
        print(f"Leverage {leverage:2}x: {initial_balance:.0f} → {final_balance:.0f} ({leveraged_return*100:+6.1f}%)")
    
    print(f"\n💡 OBSERVAÇÃO CRUCIAL:")
    print(f"   Se BTC subiu {price_change*100:+.1f}%, então:")
    print(f"   - Leverage 20x DEVERIA dar {price_change*20*100:+.1f}%")
    print(f"   - Se nossa estratégia dá -100%, há BUG GRAVE!")

def identify_possible_bugs():
    """Identifica possíveis bugs na implementação"""
    
    print(f"\n🚨 POSSÍVEIS BUGS NA IMPLEMENTAÇÃO:")
    print("="*50)
    
    bugs = [
        "1. Cálculo incorreto do P&L com leverage",
        "2. Stop Loss aplicado na direção errada", 
        "3. Entrada/saída invertida (sistema inverso residual)",
        "4. Acumulação incorreta de perdas",
        "5. Confusão entre % do preço vs % da margem",
        "6. Timing incorreto de entrada/saída",
        "7. Gestão incorreta de posições",
        "8. Cálculo errado do balance após trades"
    ]
    
    for bug in bugs:
        print(f"   ❌ {bug}")
    
    print(f"\n🔧 PRÓXIMOS PASSOS PARA DEBUG:")
    print("-"*30)
    
    steps = [
        "1. Revisar fórmula de cálculo P&L",
        "2. Verificar se sistema inverso foi 100% removido",
        "3. Testar com 1 trade manual",
        "4. Comparar com cálculo matemático esperado",
        "5. Verificar gestão de balance",
        "6. Analisar logs de trades individuais"
    ]
    
    for step in steps:
        print(f"   🔍 {step}")

def create_minimal_test():
    """Cria teste mínimo para encontrar o bug"""
    
    print(f"\n🧮 TESTE MÍNIMO PARA ENCONTRAR O BUG:")
    print("="*50)
    
    print("""
# TESTE MANUAL MÍNIMO
entry_price = 100.0
exit_price = 110.0  # +10%
leverage = 10

# Cálculo CORRETO esperado:
price_change = (110 - 100) / 100 = 0.10 (10%)
pnl_with_leverage = 0.10 * 10 = 1.00 (100%)
final_balance = 1000 * (1 + 1.00) = 2000

# Se nossa implementação não dá isso, há BUG!
""")
    
    print("💡 SUGESTÃO IMEDIATA:")
    print("   Vamos criar um teste com APENAS 1 trade")
    print("   Para ver exatamente onde está o erro!")

def main():
    investigate_leverage_logic()
    test_leverage_bug_in_real_data()
    simulate_correct_leverage()
    identify_possible_bugs()
    create_minimal_test()
    
    print(f"\n" + "="*60)
    print("🎯 CONCLUSÃO: O USUÁRIO ESTÁ CORRETO!")
    print("="*60)
    print("✅ Matematicamente, leverage DEVE amplificar ganhos E perdas")
    print("❌ Se só amplifica perdas, há BUG na implementação")
    print("🔍 Precisamos debuggar a lógica de P&L com leverage")
    print("🛠️ Próximo passo: criar teste minimal para encontrar o erro")

if __name__ == "__main__":
    main()
