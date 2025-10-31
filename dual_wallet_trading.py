#!/usr/bin/env python3
"""
Implementação para executar trades em duas carteiras simultaneamente
Abordagem simplificada que mantém compatibilidade com código existente
"""

import os
import ccxt
from dataclasses import dataclass
from typing import Optional

@dataclass
class WalletConfig:
    """Configuração de uma carteira de trading"""
    name: str
    wallet_address: str
    private_key_env: str
    vault_address: Optional[str] = None

# Configurações das carteiras
WALLET_CONFIGS = [
    WalletConfig(
        name="CARTEIRA_PRINCIPAL",
        wallet_address="0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
        private_key_env="HYPERLIQUID_PRIVATE_KEY",
        vault_address=None
    ),
    WalletConfig(
        name="CARTEIRA_RAFA", 
        wallet_address="0x22C517A64769d8CEEFcF269B93d1117624604369",
        private_key_env="HYPERLIQUID_PRIVATE_KEY_RAFA",
        vault_address="0x61374a80c401b7448958f3c0f252734e9368a388"
    )
]

def create_dual_wallet_dex_instances():
    """Cria instâncias DEX para ambas as carteiras"""
    dex_instances = {}
    
    for wallet_config in WALLET_CONFIGS:
        try:
            private_key = os.getenv(wallet_config.private_key_env)
            
            if not private_key:
                print(f"⚠️ Chave privada não encontrada para {wallet_config.name}: {wallet_config.private_key_env}")
                continue
            
            # Configuração base do DEX
            dex_config = {
                "walletAddress": wallet_config.wallet_address,
                "privateKey": private_key,
                "enableRateLimit": True,
                "timeout": 45000,
                "options": {"timeout": 45000},
            }
            
            # Adicionar vault address se for subconta
            if wallet_config.vault_address:
                dex_config["options"]["vaultAddress"] = wallet_config.vault_address
                print(f"🔧 {wallet_config.name} configurada com vault: {wallet_config.vault_address}")
            
            dex = ccxt.hyperliquid(dex_config)
            dex_instances[wallet_config.name] = dex
            
            print(f"✅ {wallet_config.name} inicializada | Wallet: {wallet_config.wallet_address[:10]}...")
            
            # Testar conexão
            try:
                balance = dex.fetch_balance()
                usdc_balance = balance.get('USDC', {}).get('total', 0)
                print(f"💰 {wallet_config.name} - Saldo USDC: ${usdc_balance:.2f}")
            except Exception as e:
                print(f"⚠️ {wallet_config.name} - Erro buscando saldo: {e}")
                
        except Exception as e:
            print(f"❌ Erro inicializando {wallet_config.name}: {e}")
    
    return dex_instances

def execute_trade_on_both_wallets(symbol, side, amount, price=None, order_type="market", params=None):
    """Executa trade nas duas carteiras simultaneamente"""
    
    dex_instances = create_dual_wallet_dex_instances()
    results = {}
    
    for wallet_name, dex in dex_instances.items():
        try:
            print(f"📊 Executando {side} {amount} {symbol} em {wallet_name}...")
            
            # Executar ordem
            if order_type == "market":
                order = dex.create_market_order(symbol, side, amount, price, params or {})
            else:
                order = dex.create_limit_order(symbol, side, amount, price, params or {})
            
            results[wallet_name] = {
                "success": True,
                "order": order,
                "order_id": order.get("id"),
                "status": order.get("status")
            }
            
            print(f"✅ {wallet_name}: Ordem {order.get('id')} criada com sucesso")
            
        except Exception as e:
            results[wallet_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ {wallet_name}: Erro na ordem - {e}")
    
    return results

def close_position_on_both_wallets(symbol):
    """Fecha posições do símbolo em ambas as carteiras"""
    
    dex_instances = create_dual_wallet_dex_instances()
    results = {}
    
    for wallet_name, dex in dex_instances.items():
        try:
            print(f"🔍 Verificando posição {symbol} em {wallet_name}...")
            
            # Verificar posição
            positions = dex.fetch_positions([symbol])
            if not positions or float(positions[0].get("contracts", 0)) == 0:
                results[wallet_name] = {"success": True, "message": "Nenhuma posição aberta"}
                print(f"ℹ️ {wallet_name}: Nenhuma posição aberta para {symbol}")
                continue
            
            pos = positions[0]
            side = pos.get("side")
            contracts = float(pos.get("contracts", 0))
            
            # Determinar lado de saída
            exit_side = "sell" if side in ("long", "buy") else "buy"
            
            # Executar fechamento
            order = dex.create_market_order(symbol, exit_side, abs(contracts), None, {"reduceOnly": True})
            
            results[wallet_name] = {
                "success": True,
                "order": order,
                "closed_position": {
                    "side": side,
                    "contracts": contracts
                }
            }
            
            print(f"✅ {wallet_name}: Posição {side} {contracts} fechada com sucesso")
            
        except Exception as e:
            results[wallet_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ {wallet_name}: Erro fechando posição - {e}")
    
    return results

def monitor_positions_both_wallets():
    """Monitora posições em ambas as carteiras"""
    
    dex_instances = create_dual_wallet_dex_instances()
    
    print("\n📊 MONITORAMENTO DE POSIÇÕES - AMBAS AS CARTEIRAS")
    print("=" * 60)
    
    for wallet_name, dex in dex_instances.items():
        try:
            print(f"\n🔍 {wallet_name}:")
            
            # Buscar todas as posições
            positions = dex.fetch_positions()
            open_positions = [pos for pos in positions if float(pos.get("contracts", 0)) != 0]
            
            if not open_positions:
                print(f"   ℹ️ Nenhuma posição aberta")
                continue
            
            for pos in open_positions:
                symbol = pos.get("symbol")
                side = pos.get("side")
                contracts = float(pos.get("contracts", 0))
                unrealized_pnl = float(pos.get("unrealizedPnl", 0))
                
                pnl_emoji = "📈" if unrealized_pnl > 0 else "📉" if unrealized_pnl < 0 else "➖"
                
                print(f"   {pnl_emoji} {symbol}: {side.upper()} {contracts:.4f} | PnL: ${unrealized_pnl:.2f}")
                
        except Exception as e:
            print(f"   ❌ Erro monitorando {wallet_name}: {e}")

def test_dual_wallet_setup():
    """Testa a configuração das duas carteiras"""
    
    print("🧪 TESTE DE CONFIGURAÇÃO - DUAS CARTEIRAS")
    print("=" * 50)
    
    # Verificar variáveis de ambiente
    for wallet_config in WALLET_CONFIGS:
        private_key = os.getenv(wallet_config.private_key_env)
        status = "✅ Configurada" if private_key else "❌ Chave ausente"
        vault_info = f" (vault: {wallet_config.vault_address[:10]}...)" if wallet_config.vault_address else ""
        
        print(f"{status} | {wallet_config.name}{vault_info}")
    
    print("\n📊 Testando conexões...")
    dex_instances = create_dual_wallet_dex_instances()
    
    print(f"\n✅ Total de carteiras ativas: {len(dex_instances)}")
    
    # Monitorar posições
    monitor_positions_both_wallets()

if __name__ == "__main__":
    # Teste da configuração
    test_dual_wallet_setup()
    
    print("\n💡 FUNÇÕES DISPONÍVEIS:")
    print("   • execute_trade_on_both_wallets(symbol, side, amount)")
    print("   • close_position_on_both_wallets(symbol)")
    print("   • monitor_positions_both_wallets()")
    print("   • test_dual_wallet_setup()")
