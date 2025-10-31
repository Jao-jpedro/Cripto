#!/usr/bin/env python3
"""
CONFIGURAÇÃO SEGUNDA CARTEIRA - CARTEIRA RAFA
==============================================

INFORMAÇÕES DA CONTA:
- Nome: CARTEIRA_RAFA
- Conta Mãe: 0x22C517A64769d8CEEFcF269B93d1117624604369
- Subconta (Vault): 0x61374a80c401b7448958f3c0f252734e9368a388
- Private Key: 0x802872a3c004dc79b990f3185e065b05d47c59c9bbb9802257b325f00bbe4c96

VARIÁVEIS DE AMBIENTE NECESSÁRIAS:
export HYPERLIQUID_PRIVATE_KEY_RAFA="0x802872a3c004dc79b990f3185e065b05d47c59c9bbb9802257b325f00bbe4c96"
export WALLET_ADDRESS_RAFA="0x22C517A64769d8CEEFcF269B93d1117624604369"
export VAULT_ADDRESS_RAFA="0x61374a80c401b7448958f3c0f252734e9368a388"

IMPLEMENTAÇÃO:
- Sempre que uma entrada for identificada, executa nas DUAS carteiras
- Carteiras são independentes (sem correlação)
- Cada carteira tem seus próprios logs de trades
- SL e TP criados para ambas as carteiras
- Safety checks aplicados em ambas

STATUS: ✅ Configuração preparada, aguardando implementação no TradingV4
"""

# Configuração para ser usada no TradingV4
CARTEIRA_RAFA_CONFIG = {
    "name": "CARTEIRA_RAFA",
    "wallet_address": "0x22C517A64769d8CEEFcF269B93d1117624604369", 
    "private_key_env": "HYPERLIQUID_PRIVATE_KEY_RAFA",
    "vault_address": "0x61374a80c401b7448958f3c0f252734e9368a388"
}

print("📋 Configuração da Segunda Carteira Preparada!")
print("🔧 Para usar no TradingV4, defina as variáveis de ambiente:")
print(f"   export HYPERLIQUID_PRIVATE_KEY_RAFA='{CARTEIRA_RAFA_CONFIG['vault_address'][:20]}...'")
print(f"   export WALLET_ADDRESS_RAFA='{CARTEIRA_RAFA_CONFIG['wallet_address']}'")
print(f"   export VAULT_ADDRESS_RAFA='{CARTEIRA_RAFA_CONFIG['vault_address']}'")
