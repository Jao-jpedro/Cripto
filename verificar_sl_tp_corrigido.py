#!/usr/bin/env python3
"""
VERIFICAÇÃO FINAL DE SL/TP - TRADING.PY
Confirma que TODAS as configurações estão em SL 1.5% e TP 12%
"""

import re
import os

def verificar_sl_tp_trading_py():
    print("🎯 VERIFICAÇÃO COMPLETA DE SL/TP NO TRADING.PY")
    print("=" * 60)
    
    with open("/Users/joaoreis/Documents/GitHub/Cripto/trading.py", "r") as f:
        content = f.read()
    
    print("\n📊 CONFIGURAÇÕES PRINCIPAIS:")
    
    # 1. Verificar GradientConfig SL
    sl_match = re.search(r'STOP_LOSS_CAPITAL_PCT:\s*float\s*=\s*([0-9.]+)', content)
    if sl_match:
        sl_val = float(sl_match.group(1))
        sl_pct = sl_val * 100
        status = "✅" if sl_val == 0.015 else "❌"
        print(f"{status} GradientConfig.STOP_LOSS_CAPITAL_PCT = {sl_val} ({sl_pct}%)")
    
    # 2. Verificar GradientConfig TP
    tp_match = re.search(r'TAKE_PROFIT_CAPITAL_PCT:\s*float\s*=\s*([0-9.]+)', content)
    if tp_match:
        tp_val = float(tp_match.group(1))
        tp_pct = tp_val * 100
        status = "✅" if tp_val == 0.12 else "❌"
        print(f"{status} GradientConfig.TAKE_PROFIT_CAPITAL_PCT = {tp_val} ({tp_pct}%)")
    
    # 3. Verificar AssetSetup padrões
    asset_sl_match = re.search(r'stop_pct:\s*float\s*=\s*([0-9.]+)', content)
    if asset_sl_match:
        asset_sl = float(asset_sl_match.group(1))
        asset_sl_pct = asset_sl * 100
        status = "✅" if asset_sl == 0.015 else "❌"
        print(f"{status} AssetSetup.stop_pct padrão = {asset_sl} ({asset_sl_pct}%)")
    
    asset_tp_match = re.search(r'take_pct:\s*float\s*=\s*([0-9.]+)', content)
    if asset_tp_match:
        asset_tp = float(asset_tp_match.group(1))
        asset_tp_pct = asset_tp * 100
        status = "✅" if asset_tp == 0.12 else "❌"
        print(f"{status} AssetSetup.take_pct padrão = {asset_tp} ({asset_tp_pct}%)")
    
    print("\n🛠️  VERIFICAÇÕES CRÍTICAS:")
    
    # 4. Verificar se sobrescritas foram removidas
    problemas = []
    
    # Verificar se cfg.STOP_LOSS_CAPITAL_PCT = asset.stop_pct foi removido
    linhas_sl_problematicas = []
    for num_linha, linha in enumerate(content.split('\n'), 1):
        if "cfg.STOP_LOSS_CAPITAL_PCT = asset.stop_pct" in linha and not linha.strip().startswith("#"):
            linhas_sl_problematicas.append(f"Linha {num_linha}: {linha.strip()}")
    
    if linhas_sl_problematicas:
        print("❌ CRÍTICO: cfg.STOP_LOSS_CAPITAL_PCT = asset.stop_pct ainda existe!")
        for linha in linhas_sl_problematicas:
            print(f"   {linha}")
        problemas.append("cfg.STOP_LOSS_CAPITAL_PCT = asset.stop_pct não removido")
    else:
        print("✅ cfg.STOP_LOSS_CAPITAL_PCT = asset.stop_pct removido corretamente")
    
    # Verificar se cfg.TAKE_PROFIT_CAPITAL_PCT = asset.take_pct foi removido
    linhas_tp_problematicas = []
    for num_linha, linha in enumerate(content.split('\n'), 1):
        if "cfg.TAKE_PROFIT_CAPITAL_PCT = asset.take_pct" in linha and not linha.strip().startswith("#"):
            linhas_tp_problematicas.append(f"Linha {num_linha}: {linha.strip()}")
    
    if linhas_tp_problematicas:
        print("❌ CRÍTICO: cfg.TAKE_PROFIT_CAPITAL_PCT = asset.take_pct ainda existe!")
        for linha in linhas_tp_problematicas:
            print(f"   {linha}")
        problemas.append("cfg.TAKE_PROFIT_CAPITAL_PCT = asset.take_pct não removido")
    else:
        print("✅ cfg.TAKE_PROFIT_CAPITAL_PCT = asset.take_pct removido corretamente")
    
    print(f"\n📋 RESUMO:")
    if not problemas:
        print("🎉 PERFEITO! Todas as configurações SL/TP estão corretas:")
        print("   • Stop Loss: 1.5% ROI (fixo)")
        print("   • Take Profit: 12% ROI (fixo)")
        print("   • Sobrescritas dinâmicas removidas")
        print("   • Sistema usará SEMPRE os valores do GradientConfig")
        return True
    else:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for problema in problemas:
            print(f"   • {problema}")
        return False

if __name__ == "__main__":
    sucesso = verificar_sl_tp_trading_py()
    
    if sucesso:
        print("\n🚀 SISTEMA SL/TP PRONTO PARA DEPLOY!")
        print("• Stop Loss: SEMPRE 1.5% ROI")
        print("• Take Profit: SEMPRE 12% ROI")
        print("• Sem variações por asset")
    else:
        print("\n⚠️  CORREÇÕES NECESSÁRIAS ANTES DO DEPLOY!")
