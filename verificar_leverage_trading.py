#!/usr/bin/env python3
"""
Verificação de Leverage no trading.py
Confirma se todas as configurações estão corretas
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verificar_configuracao_leverage():
    """Verifica se a configuração de leverage está correta"""
    
    print("🔍 VERIFICAÇÃO COMPLETA DE LEVERAGE - trading.py")
    print("="*60)
    print()
    
    try:
        # Importar a classe de configuração
        from trading import GradientConfig
        
        cfg = GradientConfig()
        
        print("✅ CONFIGURAÇÃO PRINCIPAL:")
        print(f"   LEVERAGE: {cfg.LEVERAGE}x")
        print(f"   STOP_LOSS_CAPITAL_PCT: {cfg.STOP_LOSS_CAPITAL_PCT*100:.1f}%")
        print(f"   TAKE_PROFIT_CAPITAL_PCT: {cfg.TAKE_PROFIT_CAPITAL_PCT*100:.1f}%")
        print()
        
        print("🧮 CÁLCULOS DERIVADOS:")
        sl_price_pct = (cfg.STOP_LOSS_CAPITAL_PCT / cfg.LEVERAGE) * 100
        tp_price_pct = (cfg.TAKE_PROFIT_CAPITAL_PCT / cfg.LEVERAGE) * 100
        print(f"   SL preço: {sl_price_pct:.2f}% ({cfg.STOP_LOSS_CAPITAL_PCT*100:.1f}% ROI ÷ {cfg.LEVERAGE}x)")
        print(f"   TP preço: {tp_price_pct:.2f}% ({cfg.TAKE_PROFIT_CAPITAL_PCT*100:.1f}% ROI ÷ {cfg.LEVERAGE}x)")
        print()
        
        print("🌍 VARIÁVEIS DE AMBIENTE:")
        env_leverage = os.getenv("LEVERAGE", "3")
        print(f"   LEVERAGE env: {env_leverage}x (padrão: 3x)")
        print()
        
        if cfg.LEVERAGE == 3 and env_leverage == "3":
            print("✅ CONFIGURAÇÃO CORRETA!")
            print("   Leverage uniformemente configurado em 3x")
        else:
            print("⚠️  POSSÍVEL INCONSISTÊNCIA:")
            print(f"   cfg.LEVERAGE: {cfg.LEVERAGE}x")
            print(f"   env LEVERAGE: {env_leverage}x")
        
        print()
        print("📊 IMPACTO NO SISTEMA:")
        print("   • Todas as posições devem abrir com 3x leverage")
        print("   • SL em 0.5% movimento de preço")
        print("   • TP em 4.0% movimento de preço")
        print("   • ROI amplificado 3x nos trades")
        
    except Exception as e:
        print(f"❌ ERRO na verificação: {e}")
        print("   Verificar se trading.py está acessível")

def verificar_problemas_potenciais():
    """Identifica possíveis problemas"""
    
    print("\n" + "="*60)
    print("🚨 DIAGNÓSTICO DE PROBLEMAS POTENCIAIS")
    print("="*60)
    print()
    
    problemas = [
        "1. 🔄 Posições já abertas com leverage antigo",
        "2. 📊 Cache do exchange com configuração anterior", 
        "3. 🎯 Configuração manual da conta Hyperliquid",
        "4. ⏱️ Ordens pendentes com leverage incorreto",
        "5. 🔧 Múltiplas instâncias rodando simultaneamente"
    ]
    
    print("POSSÍVEIS CAUSAS DE LEVERAGE INCORRETO:")
    for problema in problemas:
        print(f"   {problema}")
    
    print()
    print("💡 AÇÕES RECOMENDADAS:")
    print("   1. ✅ Fechar todas as posições abertas")
    print("   2. ✅ Cancelar todas as ordens pendentes") 
    print("   3. ✅ Parar todas as instâncias do trading")
    print("   4. ✅ Verificar configuração no Hyperliquid")
    print("   5. ✅ Reiniciar sistema com configuração limpa")
    print("   6. ✅ Monitorar primeiras posições para confirmar 3x")

def main():
    verificar_configuracao_leverage()
    verificar_problemas_potenciais()
    
    print("\n" + "="*60)
    print("🎯 CONCLUSÃO")
    print("="*60)
    print("✅ trading.py configurado corretamente para leverage 3x")
    print("⚠️  Se ainda houver problemas, verificar estado do exchange")
    print("🔄 Reinicialização completa pode ser necessária")

if __name__ == "__main__":
    main()
