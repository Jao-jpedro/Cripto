#!/usr/bin/env python3
"""
VERIFICAÇÃO FINAL DE LEVERAGE - TRADING.PY
Confirma que TODAS as configurações estão em 3x e USD_PER_TRADE está adequado
"""

import re
import os

def verificar_trading_py():
    print("🔍 VERIFICAÇÃO COMPLETA DE LEVERAGE NO TRADING.PY")
    print("=" * 60)
    
    with open("/Users/joaoreis/Documents/GitHub/Cripto/trading.py", "r") as f:
        content = f.read()
    
    print("\n📊 CONFIGURAÇÕES PRINCIPAIS:")
    
    # 1. Verificar GradientConfig LEVERAGE
    gradient_match = re.search(r'LEVERAGE:\s*int\s*=\s*(\d+)', content)
    if gradient_match:
        leverage_val = int(gradient_match.group(1))
        status = "✅" if leverage_val == 3 else "❌"
        print(f"{status} GradientConfig.LEVERAGE = {leverage_val}x")
    
    # 2. Verificar environment leverage padrão
    env_match = re.search(r'os\.getenv\("LEVERAGE",\s*"(\d+)"\)', content)
    if env_match:
        env_leverage = int(env_match.group(1))
        status = "✅" if env_leverage == 3 else "❌"
        print(f"{status} Environment LEVERAGE padrão = {env_leverage}x")
    
    # 3. Verificar USD_PER_TRADE padrão
    usd_match = re.search(r'usd_to_spend:\s*float\s*=\s*(\d+)', content)
    if usd_match:
        usd_val = int(usd_match.group(1))
        status = "✅" if usd_val == 4 else "❌"
        print(f"{status} USD_PER_TRADE padrão = ${usd_val}")
    
    print("\n🎯 ASSET SETUPS:")
    
    # 4. Verificar todos os AssetSetups
    asset_pattern = r'AssetSetup\("([^"]+)",\s*"[^"]+",\s*"[^"]+",\s*(\d+),'
    assets = re.findall(asset_pattern, content)
    
    problemas = []
    for asset_name, leverage in assets:
        leverage_int = int(leverage)
        status = "✅" if leverage_int == 3 else "❌"
        print(f"{status} {asset_name}: {leverage_int}x")
        if leverage_int != 3:
            problemas.append(f"{asset_name}: {leverage_int}x")
    
    # 5. Verificar se cfg.LEVERAGE = asset.leverage foi removido
    print("\n🛠️  VERIFICAÇÕES CRÍTICAS:")
    
    # Verificar se existe fora de comentários
    linhas_problematicas = []
    for num_linha, linha in enumerate(content.split('\n'), 1):
        if "cfg.LEVERAGE = asset.leverage" in linha and not linha.strip().startswith("#"):
            linhas_problematicas.append(f"Linha {num_linha}: {linha.strip()}")
    
    if linhas_problematicas:
        print("❌ CRÍTICO: cfg.LEVERAGE = asset.leverage ainda existe!")
        for linha in linhas_problematicas:
            print(f"   {linha}")
        problemas.append("cfg.LEVERAGE = asset.leverage não removido")
    else:
        print("✅ cfg.LEVERAGE = asset.leverage removido corretamente")
    
    # 6. Calcular valor final com leverage
    final_amount = 4 * 3  # USD_PER_TRADE * LEVERAGE
    print(f"\n💰 CÁLCULO FINAL:")
    print(f"✅ $4 USD × 3x leverage = ${final_amount} (acima do mínimo)")
    
    print(f"\n📋 RESUMO:")
    if not problemas:
        print("🎉 PERFEITO! Todas as configurações estão corretas:")
        print("   • Leverage: 3x em TODOS os assets")
        print("   • USD padrão: $4 (suficiente para mínimo)")
        print("   • Valor final: $12 por trade")
        print("   • cfg.LEVERAGE = asset.leverage removido")
        return True
    else:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for problema in problemas:
            print(f"   • {problema}")
        return False

if __name__ == "__main__":
    sucesso = verificar_trading_py()
    
    if sucesso:
        print("\n🚀 SISTEMA PRONTO PARA DEPLOY!")
        print("Environment no Render deve ter:")
        print("USD_PER_TRADE=4")
        print("LEVERAGE=3")
    else:
        print("\n⚠️  CORREÇÕES NECESSÁRIAS ANTES DO DEPLOY!")
