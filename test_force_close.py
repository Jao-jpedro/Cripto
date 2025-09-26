#!/usr/bin/env python3
"""
Script simplificado para forçar fechamento de posições AVAX/PUMP com prejuízo
"""

import os
import sys
import ccxt

# Configurar private key  
private_key = "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872"

def force_close_losing_positions():
    """Força fechamento de posições AVAX/PUMP com prejuízo"""
    print("=== FORÇA FECHAMENTO POSIÇÕES COM PREJUÍZO ===")
    
    try:
        # Inicializar DEX Hyperliquid diretamente
        print("[INIT] Inicializando DEX Hyperliquid...")
        dex = ccxt.hyperliquid({
            "privateKey": private_key,
            "walletAddress": "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"timeout": 30000},
        })
        print("[INIT] DEX inicializado!")
        
        # Obter endereço da carteira
        wallet_address = dex.walletAddress or "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
        print(f"[INIT] Wallet: {wallet_address}")
        
        # Símbolos para verificar (apenas para status)
        symbols_to_check = ["AVAX/USDC:USDC", "PUMP/USDC:USDC", "BNB/USDC:USDC", "NEAR/USDC:USDC"]
        
        for symbol in symbols_to_check:
            print(f"\n[CHECK] Verificando {symbol}...")
            
            try:
                # Buscar posições
                positions = dex.fetch_positions([symbol], {"user": wallet_address})
                pos = None
                for p in positions:
                    contracts = float(p.get("contracts", 0) or 0)
                    if contracts > 0:
                        pos = p
                        break
                
                if not pos:
                    print(f"[CHECK] {symbol}: Sem posição")
                    continue
                    
                # Extrair dados da posição
                qty = float(pos.get("contracts", 0) or 0)
                unrealized = float(pos.get("unrealizedPnl", 0) or 0)
                side = pos.get("side", "")
                entry_price = pos.get("entryPrice", 0)
                
                print(f"[POS] {symbol}: qty={qty} unrealized=${unrealized:.4f} side={side} entry=${entry_price}")
                
                # Calcular ROI manualmente para comparar
                if entry_price and entry_price > 0:
                    # Buscar preço atual
                    ticker = dex.fetch_ticker(symbol)
                    current_price = float(ticker.get("last", 0) or 0)
                    
                    if current_price > 0:
                        if side.lower() in ("long", "buy"):
                            roi_raw = (current_price / entry_price) - 1.0
                        else:
                            roi_raw = (entry_price / current_price) - 1.0
                        
                        # Assumir leverage 10 (padrão Hyperliquid)
                        leverage = 10.0
                        roi_leveraged = roi_raw * leverage
                        
                        print(f"[ROI_CALC] {symbol}: current=${current_price} roi_raw={roi_raw:.4f} roi_lev={roi_leveraged:.4f}")
                        
                        # Verificar se precisa fechar por ROI também (-5% = -0.05)
                        if roi_leveraged <= -0.05:
                            print(f"[ROI_BREACH] {symbol}: ROI {roi_leveraged:.4f} <= -0.05 - DEVE FECHAR!")
                
                # Não executar fechamento, apenas mostrar status
                if unrealized < -0.03 or (roi_leveraged <= -0.05 if 'roi_leveraged' in locals() else False):
                    print(f"[SHOULD_CLOSE] {symbol}: CRITÉRIOS ATINGIDOS - unrealized=${unrealized:.4f} ou roi={roi_leveraged:.4f}")
                else:
                    print(f"[OK] {symbol}: Critérios OK - unrealized=${unrealized:.4f}")
                    
            except Exception as e:
                print(f"[ERROR] {symbol}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"[FATAL] Erro geral: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    force_close_losing_positions()
