#!/usr/bin/env python3
"""
DIAGNÓSTICO COMPLETO: MOTIVOS DE FECHAMENTO PREMATURO
Identifica TODOS os mecanismos que podem fechar posições antes do TP 12%
"""

def diagnosticar_fechamentos_prematuros():
    print("🔍 DIAGNÓSTICO: FECHAMENTOS PREMATUROS NO TRADING.PY")
    print("=" * 65)
    
    print("\n📋 MECANISMOS DE FECHAMENTO IDENTIFICADOS:")
    
    print("\n1️⃣ STOP LOSS FIXO (Correto ✅):")
    print("   • Configuração: 1.5% ROI = 0.5% movimento de preço")
    print("   • Localização: Linhas 4027, 4036, 4045")
    print("   • Status: ✅ Configurado corretamente")
    
    print("\n2️⃣ TAKE PROFIT FIXO (Correto ✅):")
    print("   • Configuração: 12% ROI = 4% movimento de preço")
    print("   • Localização: Linha 4034")
    print("   • Status: ✅ Configurado corretamente")
    
    print("\n3️⃣ ROI HARD STOP (POSSÍVEL PROBLEMA ⚠️):")
    print("   • Configuração: ROI_HARD_STOP = -5.0% (linha 47)")
    print("   • Localização: Linhas 4438, 5776")
    print("   • Comportamento: Fecha posição se ROI <= -5%")
    print("   • ⚠️ PROBLEMA: Pode estar fechando antes do SL 1.5%!")
    
    print("\n4️⃣ UNREALIZED PNL HARD STOP (POSSÍVEL PROBLEMA ⚠️):")
    print("   • Configuração: UNREALIZED_PNL_HARD_STOP = -0.05 (-5 cents)")
    print("   • Localização: Linha 48, usado em linha 5783")
    print("   • Comportamento: Fecha se PnL <= -$0.05")
    print("   • ⚠️ PROBLEMA: Muito restritivo para $12 de posição!")
    
    print("\n5️⃣ FAST SAFETY CHECK (AGRESSIVO ⚠️):")
    print("   • Função: fast_safety_check_v4() - linha 5713")
    print("   • Frequência: Executado APÓS CADA ASSET (linha 6013)")
    print("   • Comportamento: Verifica ROI e PnL constantemente")
    print("   • ⚠️ PROBLEMA: Verificações muito frequentes!")
    
    print("\n6️⃣ TRAILING STOP (Desabilitado ✅):")
    print("   • Configuração: ENABLE_TRAILING_STOP = False")
    print("   • Status: ✅ Corretamente desabilitado")
    
    print("\n🔍 CÁLCULOS DOS PROBLEMAS:")
    
    # Posição $12 com 3x leverage = $4 capital
    capital = 4
    leverage = 3
    posicao_total = capital * leverage  # $12
    
    print(f"\n💰 Para posição de ${posicao_total} (${capital} capital, {leverage}x leverage):")
    
    # ROI Hard Stop
    roi_loss = abs(-5.0 / 100)  # 5%
    valor_perda_roi = capital * roi_loss
    print(f"• ROI Hard Stop (-5%): Perda de ${valor_perda_roi:.2f} fecha posição")
    
    # PnL Hard Stop  
    pnl_limit = 0.05  # 5 cents
    print(f"• PnL Hard Stop: Perda de ${pnl_limit:.2f} fecha posição")
    print(f"  ⚠️ CRÍTICO: ${pnl_limit:.2f} é apenas {(pnl_limit/capital)*100:.1f}% do capital!")
    
    # SL DNA (correto)
    sl_roi = 1.5 / 100
    valor_perda_sl = capital * sl_roi
    print(f"• SL DNA (1.5%): Perda de ${valor_perda_sl:.2f} para fechar")
    
    print(f"\n🚨 CONFLITOS IDENTIFICADOS:")
    print(f"1. PnL Hard Stop (${pnl_limit:.2f}) < SL DNA (${valor_perda_sl:.2f})")
    print(f"   ➜ PnL fecha ANTES do SL chegar!")
    print(f"2. ROI Hard Stop (-5%) pode fechar antes do SL (-1.5%)")
    print(f"3. Fast Safety roda a cada iteração = fechamentos prematuros")
    
    print(f"\n🎯 SOLUÇÕES NECESSÁRIAS:")
    print(f"1. Aumentar UNREALIZED_PNL_HARD_STOP para > ${valor_perda_sl:.2f}")
    print(f"2. Ajustar ROI_HARD_STOP para < -1.5% (ex: -1.0%)")
    print(f"3. Reduzir frequência do fast_safety_check_v4")
    print(f"4. Ou DESABILITAR completamente os hard stops")

if __name__ == "__main__":
    diagnosticar_fechamentos_prematuros()
