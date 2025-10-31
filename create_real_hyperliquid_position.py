#!/usr/bin/env python3
"""
🚀 CRIADOR DE POSIÇÃO REAL NA HYPERLIQUID
Conecta diretamente à exchange usando o SDK oficial
"""

import os
import time
import json
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Configurações
WALLET_ADDRESS = "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
PRIVATE_KEY = "0xa524295ceec3e792d9aaa18b026dbc9ca74af350117631235ec62dcbe24bc405"

print("🚀 CRIADOR DE POSIÇÃO REAL NA HYPERLIQUID")
print("=" * 60)
print(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔑 Wallet: {WALLET_ADDRESS}")
print(f"💎 Using REAL Hyperliquid SDK")
print("=" * 60)

try:
    # Inicializar clientes Hyperliquid
    print("\n🔌 CONECTANDO À HYPERLIQUID...")
    
    # Info client (para consultas)
    info = Info(constants.MAINNET_API_URL)
    print("✅ Info client conectado")
    
    # Exchange client (para trading)
    exchange = Exchange(WALLET_ADDRESS, PRIVATE_KEY, constants.MAINNET_API_URL)
    print("✅ Exchange client conectado")
    
    # Verificar conexão
    print("\n📊 VERIFICANDO CONEXÃO...")
    try:
        user_state = info.user_state(WALLET_ADDRESS)
        print(f"✅ Conexão verificada - User state obtido")
        
        # Mostrar balance
        if 'marginSummary' in user_state:
            margin = user_state['marginSummary']
            account_value = float(margin.get('accountValue', 0))
            total_margin_used = float(margin.get('totalMarginUsed', 0))
            print(f"💰 Account Value: ${account_value:.2f}")
            print(f"💰 Margin Used: ${total_margin_used:.2f}")
            print(f"💰 Available: ${account_value - total_margin_used:.2f}")
        else:
            print("⚠️ Margin summary não encontrado")
            
    except Exception as e:
        print(f"❌ Erro verificando conexão: {e}")
        exit(1)
    
    # Verificar posições existentes
    print("\n📈 VERIFICANDO POSIÇÕES EXISTENTES...")
    try:
        positions = user_state.get('assetPositions', [])
        if positions:
            print(f"📊 {len(positions)} posição(ões) ativa(s):")
            for pos in positions:
                asset = pos['position']['coin']
                size = float(pos['position']['szi'])
                unrealized_pnl = float(pos['position']['unrealizedPnl'])
                print(f"   📈 {asset}: {size:.0f} contratos | PnL: ${unrealized_pnl:.2f}")
        else:
            print("📊 Nenhuma posição ativa")
    except Exception as e:
        print(f"⚠️ Erro verificando posições: {e}")
    
    # Configurar trade
    symbol = "PUMP"  # Hyperliquid usa apenas o nome base
    trade_size_usd = 5.0  # $5 USD
    leverage = 3  # 3x leverage
    side = "A"  # A = Long (Ask), B = Short (Bid)
    
    print(f"\n🎯 CONFIGURANDO TRADE...")
    print(f"💰 Símbolo: {symbol}")
    print(f"💵 Valor: ${trade_size_usd} USD")
    print(f"⚡ Leverage: {leverage}x")
    print(f"📊 Lado: {'LONG' if side == 'A' else 'SHORT'}")
    
    # Obter preço atual
    print(f"\n📊 OBTENDO PREÇO ATUAL...")
    try:
        all_mids = info.all_mids()
        
        if symbol in all_mids:
            current_price = float(all_mids[symbol])
            print(f"💰 Preço atual {symbol}: ${current_price:.6f}")
        else:
            print(f"❌ Símbolo {symbol} não encontrado no mercado")
            print(f"🔍 Símbolos disponíveis: {list(all_mids.keys())[:10]}...")
            exit(1)
            
    except Exception as e:
        print(f"❌ Erro obtendo preço: {e}")
        exit(1)
    
    # Calcular quantidade
    notional_value = trade_size_usd * leverage
    quantity = notional_value / current_price
    
    print(f"🔢 Valor nocional: ${notional_value:.2f}")
    print(f"🔢 Quantidade: {quantity:.0f} contratos")
    
    # Configurar leverage primeiro
    print(f"\n⚡ CONFIGURANDO LEVERAGE...")
    try:
        leverage_result = exchange.update_leverage(leverage, symbol)
        print(f"✅ Leverage configurada: {leverage_result}")
    except Exception as e:
        print(f"⚠️ Erro configurando leverage: {e}")
    
    # Criar ordem de mercado
    print(f"\n🔥 CRIANDO ORDEM DE MERCADO...")
    
    try:
        # Para ordem de mercado, usar market order
        order_result = exchange.market_order(
            coin=symbol,
            is_buy=(side == "A"),
            sz=quantity,
            px=None,  # Market order não precisa de preço
        )
        
        print(f"✅ ORDEM ENVIADA!")
        print(f"📝 Resultado: {json.dumps(order_result, indent=2)}")
        
        # Aguardar execução
        print(f"\n⏳ AGUARDANDO EXECUÇÃO...")
        time.sleep(3)
        
        # Verificar posições após ordem
        print(f"\n📊 VERIFICANDO POSIÇÕES APÓS ORDEM...")
        user_state_after = info.user_state(WALLET_ADDRESS)
        positions_after = user_state_after.get('assetPositions', [])
        
        pump_position = None
        for pos in positions_after:
            if pos['position']['coin'] == symbol:
                pump_position = pos
                break
        
        if pump_position:
            size = float(pump_position['position']['szi'])
            unrealized_pnl = float(pump_position['position']['unrealizedPnl'])
            entry_px = float(pump_position['position']['entryPx'])
            
            print(f"🎉 POSIÇÃO CRIADA COM SUCESSO!")
            print(f"📈 {symbol}: {size:.0f} contratos")
            print(f"💰 Preço de entrada: ${entry_px:.6f}")
            print(f"💰 PnL não realizado: ${unrealized_pnl:.2f}")
            print(f"📊 Lado: {'LONG' if size > 0 else 'SHORT'}")
            
        else:
            print(f"⚠️ Posição não encontrada - pode ter sido executada e fechada rapidamente")
        
        # Verificar trades recentes
        print(f"\n📋 VERIFICANDO TRADES RECENTES...")
        try:
            fills = info.user_fills(WALLET_ADDRESS)
            recent_fills = fills[:5]  # Últimos 5 trades
            
            if recent_fills:
                print(f"📊 Últimos {len(recent_fills)} trades:")
                for fill in recent_fills:
                    coin = fill.get('coin', 'Unknown')
                    side = fill.get('side', 'Unknown')
                    sz = fill.get('sz', '0')
                    px = fill.get('px', '0')
                    time_ms = fill.get('time', 0)
                    trade_time = datetime.fromtimestamp(int(time_ms) / 1000)
                    
                    print(f"   🔄 {trade_time.strftime('%H:%M:%S')} | "
                          f"{coin} {side} {sz} @ ${px}")
            else:
                print("📊 Nenhum trade recente encontrado")
                
        except Exception as e:
            print(f"⚠️ Erro verificando trades: {e}")
    
    except Exception as e:
        print(f"❌ ERRO CRIANDO ORDEM: {e}")
        print(f"🔍 Tipo do erro: {type(e)}")
        
        # Tentar entender o erro
        error_str = str(e)
        if "insufficient" in error_str.lower():
            print("💡 DIAGNÓSTICO: Saldo insuficiente")
        elif "invalid" in error_str.lower():
            print("💡 DIAGNÓSTICO: Parâmetros inválidos")
        elif "market" in error_str.lower():
            print("💡 DIAGNÓSTICO: Problema com mercado")
        else:
            print(f"💡 DIAGNÓSTICO: Erro desconhecido - {error_str}")

except Exception as e:
    print(f"❌ ERRO GERAL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 SCRIPT CONCLUÍDO")
print("=" * 60)
