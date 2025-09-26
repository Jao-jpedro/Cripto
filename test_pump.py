#!/usr/bin/env python3
"""
Teste específico para PUMP com trailing stop
"""
import os
import sys
import ccxt

# Configurar environment
os.environ["LIVE_TRADING"] = "1"

# Adicionar o caminho do trading.py
sys.path.append("/Users/joaoreis/Documents/GitHub/Cripto")

from trading import EMAGradientStrategy, AssetSetup, GradientConfig

def test_pump_trailing():
    print("=== TESTE PUMP TRAILING STOP ===")
    
    try:
        # Inicializar DEX (copiando do trading.py)
        dex_timeout = int(os.getenv("DEX_TIMEOUT_MS", "5000"))
        
        dex = ccxt.hyperliquid({
            "apiKey": os.getenv("HYPERLIQUID_API_KEY"),
            "secret": os.getenv("HYPERLIQUID_SECRET_KEY"),
            "sandbox": False,
            "timeout": dex_timeout,
        })
        
        print("DEX inicializado com sucesso")
        
        # Configurar apenas PUMP
        asset = AssetSetup("PUMP-USD", "PUMPUSDT", "PUMP/USDC:USDC", 5, usd_env="USD_PER_TRADE_PUMP")
        
        # Criar engine
        engine = EMAGradientStrategy(dex, asset.hl_symbol, GradientConfig(), None, debug=True)
        
        # Verificar posição atual diretamente
        try:
            positions = dex.fetch_positions([asset.hl_symbol])
            pos = positions[0] if positions else None
        except Exception as e:
            print(f"Erro ao buscar posição: {e}")
            pos = None
            
        print(f"Posição PUMP: {pos}")
        
        if pos and pos.get("contracts", 0) != 0:
            print("Posição encontrada! Testando proteções...")
            
            # Simular dados de proteção
            entry = float(pos.get("entryPrice", 0.005029))
            
            # Obter preço atual do ticker
            try:
                ticker = dex.fetch_ticker(asset.hl_symbol)
                current = float(ticker.get("last", entry))
            except Exception as e:
                print(f"Erro ao obter ticker: {e}")
                current = entry
            
            print(f"Entry price: {entry}")
            print(f"Current price: {current}")
            
            # Calcular ROI
            roi = (current / entry) - 1.0
            roi_alavancado = roi * 5.0
            print(f"ROI: {roi:.4f} ({roi*100:.2f}%)")
            print(f"ROI alavancado: {roi_alavancado:.4f} ({roi_alavancado*100:.2f}%)")
            
            # Testar cálculo de trailing stop diretamente
            trailing_px = engine._compute_trailing_stop(entry, current, "buy", 5.0)
            print(f"Trailing price calculado: {trailing_px}")
            
            # Executar proteções
            print("\n--- Executando _ensure_position_protections ---")
            engine._ensure_position_protections(pos)
        else:
            print("Nenhuma posição PUMP encontrada")
            
    except Exception as e:
        print(f"Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pump_trailing()
