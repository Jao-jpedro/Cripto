#!/usr/bin/env python3
"""
Script para criar uma posição pequena e testar fechamento automático
"""

import ccxt
import time

# Configurar private key  
private_key = "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872"

def create_test_position():
    """Cria posição pequena para testar fechamento"""
    print("=== CRIANDO POSIÇÃO DE TESTE ===")
    
    try:
        # Inicializar DEX
        dex = ccxt.hyperliquid({
            "privateKey": private_key,
            "walletAddress": "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
            "enableRateLimit": True,
            "timeout": 30000,
        })
        
        # Escolher símbolo (use um volátil para testar rapidamente)
        symbol = "DOGE/USDC:USDC"
        
        # Buscar preço atual
        ticker = dex.fetch_ticker(symbol)
        current_price = float(ticker.get("last", 0) or 0)
        print(f"[PRICE] {symbol}: ${current_price}")
        
        # Criar posição tiny (muito pequena para minimizar risco)
        # Usar quantidade que resulte em ~$2-3 de exposição
        quantity = round(2.5 / current_price, 1)  # ~$2.50 de DOGE
        
        print(f"[ORDER] Criando posição long {quantity} {symbol} a ${current_price}")
        
        # Criar ordem market buy
        result = dex.create_order(
            symbol, 
            "market", 
            "buy", 
            quantity,
            current_price * 1.005,  # Ligeiramente acima para garantir execução
            {}
        )
        
        print(f"[SUCCESS] Posição criada: {result}")
        
        # Aguardar alguns segundos
        print("[WAIT] Aguardando 5s para posição se estabelecer...")
        time.sleep(5)
        
        # Verificar posição
        positions = dex.fetch_positions([symbol], {"user": "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"})
        for pos in positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts > 0:
                unrealized = float(pos.get("unrealizedPnl", 0) or 0)
                side = pos.get("side", "")
                entry = pos.get("entryPrice", 0)
                print(f"[POSITION] {symbol}: {contracts} {side} entry=${entry} pnl=${unrealized:.4f}")
                return True
        
        print("[ERROR] Posição não encontrada após criação")
        return False
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    create_test_position()
