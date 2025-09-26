#!/usr/bin/env python3
"""
Script para testar as funções de fechamento corrigidas do trading.py
"""

import os
import sys
sys.path.append('.')

# Configurar private key
os.environ["HYPERLIQUID_PRIVATE_KEY"] = "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872"
os.environ["LIVE_TRADING"] = "1"

def test_closure_functions():
    """Testa as funções de fechamento corrigidas"""
    print("=== TESTE DAS FUNÇÕES DE FECHAMENTO ===")
    
    try:
        # Importar as funções corrigidas
        from trading import (
            _init_dex, close_if_roi_breaches, close_if_unrealized_pnl_breaches,
            close_if_hard_loss, close_if_liquidation_risk
        )
        
        print("[INIT] Inicializando DEX...")
        dex = _init_dex()
        print("[INIT] DEX inicializado!")
        
        # Símbolos para testar
        symbols_to_test = ["AVAX/USDC:USDC", "PUMP/USDC:USDC", "NEAR/USDC:USDC"]
        
        for symbol in symbols_to_test:
            print(f"\n[TEST] Testando funções para {symbol}...")
            
            # Testar close_if_unrealized_pnl_breaches (threshold baixo para não fechar de verdade)
            try:
                result1 = close_if_unrealized_pnl_breaches(dex, symbol, threshold=-1000.0)
                print(f"  close_if_unrealized_pnl_breaches: {result1}")
            except Exception as e:
                print(f"  close_if_unrealized_pnl_breaches: ERRO - {type(e).__name__}: {e}")
            
            # Testar close_if_roi_breaches (threshold baixo para não fechar de verdade)
            try:
                result2 = close_if_roi_breaches(dex, symbol, threshold=-1000.0)
                print(f"  close_if_roi_breaches: {result2}")
            except Exception as e:
                print(f"  close_if_roi_breaches: ERRO - {type(e).__name__}: {e}")
            
            # Testar close_if_hard_loss (threshold baixo para não fechar)
            try:
                ticker = dex.fetch_ticker(symbol)
                current_price = float(ticker.get("last", 0) or 0)
                result3 = close_if_hard_loss(dex, symbol, current_price)
                print(f"  close_if_hard_loss: {result3}")
            except Exception as e:
                print(f"  close_if_hard_loss: ERRO - {type(e).__name__}: {e}")
                
            # Testar close_if_liquidation_risk (buffer baixo para não fechar)
            try:
                result4 = close_if_liquidation_risk(dex, symbol, current_price, buffer_pct=0.99)
                print(f"  close_if_liquidation_risk: {result4}")
            except Exception as e:
                print(f"  close_if_liquidation_risk: ERRO - {type(e).__name__}: {e}")
        
    except Exception as e:
        print(f"[FATAL] Erro geral: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_closure_functions()
