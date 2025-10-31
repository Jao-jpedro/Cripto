#!/bin/bash
# Configuração das variáveis de ambiente para múltiplas carteiras

# Carteira Principal (já existente)
export WALLET_ADDRESS="0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
export HYPERLIQUID_PRIVATE_KEY="[SUA_CHAVE_PRIVADA_PRINCIPAL]"

# Carteira RAFA (nova)
export WALLET_ADDRESS_RAFA="0x22C517A64769d8CEEFcF269B93d1117624604369"
export HYPERLIQUID_PRIVATE_KEY_RAFA="0x802872a3c004dc79b990f3185e065b05d47c59c9bbb9802257b325f00bbe4c96"
export VAULT_ADDRESS_RAFA="0x61374a80c401b7448958f3c0f252734e9368a388"

echo "✅ Configuração de carteiras preparada!"
echo "📊 Para usar, execute: source setup_wallets.sh"
echo ""
echo "🔧 Carteiras configuradas:"
echo "   • CARTEIRA_PRINCIPAL: ${WALLET_ADDRESS:0:10}..."
echo "   • CARTEIRA_RAFA: ${WALLET_ADDRESS_RAFA:0:10}... (vault: ${VAULT_ADDRESS_RAFA:0:10}...)"
