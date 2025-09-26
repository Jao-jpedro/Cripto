#!/usr/bin/env python3
"""
Script para apenas verificar posições atuais sem executar trading.py
"""

import ccxt

# Configurar private key  
private_key = "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872"

def check_positions():
    """Verifica status das posições atuais"""
    print("=== VERIFICAÇÃO DE POSIÇÕES ===")
    
    try:
        # Inicializar DEX Hyperliquid diretamente
        print("[INIT] Inicializando DEX Hyperliquid...")
        dex = ccxt.hyperliquid({
            "privateKey": private_key,
            "walletAddress": "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
            "enableRateLimit": True,
            "timeout": 30000,
        })
        print("[INIT] DEX inicializado!")
        
        # Obter endereço da carteira
        wallet_address = dex.walletAddress or "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
        print(f"[INIT] Wallet: {wallet_address}")
        
        # Buscar todas as posições
        print(f"[CHECK] Buscando todas as posições...")
        positions = dex.fetch_positions(params={"user": wallet_address})
        
        active_positions = []
        for pos in positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts > 0:
                active_positions.append(pos)
        
        print(f"[RESULT] Encontradas {len(active_positions)} posições ativas:")
        
        for pos in active_positions:
            symbol = pos.get("symbol", "")
            qty = float(pos.get("contracts", 0) or 0)
            unrealized = float(pos.get("unrealizedPnl", 0) or 0)
            side = pos.get("side", "")
            entry_price = pos.get("entryPrice", 0)
            
            # Calcular ROI se possível
            roi_info = ""
            if entry_price and entry_price > 0:
                try:
                    ticker = dex.fetch_ticker(symbol)
                    current_price = float(ticker.get("last", 0) or 0)
                    
                    if current_price > 0:
                        if side.lower() in ("long", "buy"):
                            roi_raw = (current_price / entry_price) - 1.0
                        else:
                            roi_raw = (entry_price / current_price) - 1.0
                        
                        leverage = 10.0
                        roi_leveraged = roi_raw * leverage
                        roi_info = f" | ROI: {roi_leveraged:.1%}"
                except:
                    roi_info = " | ROI: N/A"
            
            print(f"  {symbol}: {qty} {side} | PnL: ${unrealized:.4f}{roi_info}")
            
            # Verificar critérios de fechamento
            should_close_pnl = unrealized <= -0.05
            should_close_roi = "roi_leveraged" in locals() and roi_leveraged <= -0.05
            
            if should_close_pnl or should_close_roi:
                reasons = []
                if should_close_pnl:
                    reasons.append(f"PnL ${unrealized:.4f} <= -$0.05")
                if should_close_roi:
                    reasons.append(f"ROI {roi_leveraged:.1%} <= -5%")
                print(f"    ⚠️  DEVERIA SER FECHADA: {' e '.join(reasons)}")
                
    except Exception as e:
        print(f"[FATAL] Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_positions()
