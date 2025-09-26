#!/usr/bin/env python3
"""Demo do per-asset safety check - mostra como cada ativo é seguido imediatamente por verificação"""

import time

print("🚀 DEMO: Per-Asset Safety Check")
print("Nova abordagem:")
print("1. Processa BTC → Safety check immediate")
print("2. Processa SOL → Safety check immediate") 
print("3. Processa ETH → Safety check immediate")
print("etc...\n")

print("vs. Abordagem antiga:")
print("1. Processa TODOS (BTC, SOL, ETH, etc.) - demora 4-5 min")
print("2. Safety check de todos - pós 4-5 min")
print("3. Volta ao passo 1\n")

# Simulação do novo fluxo
assets = ["BTC", "SOL", "ETH", "XRP", "DOGE", "AVAX"]

print("=== SIMULAÇÃO DO NOVO FLUXO ===")
start_time = time.time()

for i, asset in enumerate(assets, 1):
    print(f"📊 [{time.strftime('%H:%M:%S')}] Processando {asset}...")
    time.sleep(2)  # Simula processamento do asset
    
    print(f"    ⚡ Safety check pós-{asset} (PnL/ROI emergency)")
    time.sleep(0.5)  # Simula safety check rápido
    
    if i == 2:  # Após SOL
        print(f"    🚨 ALERTA: {asset} detectou PnL -0.08 → Fechamento emergency!")
        print(f"    ✅ Posição fechada em 2.5s vs 4-5min do sistema antigo")
        break
        
    print()

total_time = time.time() - start_time
print(f"\n⏱️  Tempo para detectar emergência: {total_time:.1f}s")
print("🎯 Benefício: Detecção em ~2.5s por asset vs 4-5min para todos os assets")
print("🛡️  Proteção de capital drasticamente melhorada!")
