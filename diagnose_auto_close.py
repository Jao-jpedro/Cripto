#!/usr/bin/env python3
"""
🔧 DIAGNÓSTICO: Forçar LONG e impedir fechamento automático
Objetivo: Abrir posição LONG e manter ela aberta para análise
"""

import os
import time
import sys
from datetime import datetime

# Configurar para trading real
os.environ['LIVE_TRADING'] = '1'
os.environ['HYPERLIQUID_PRIVATE_KEY'] = '0xa524295ceec3e792d9aaa18b026dbc9ca74af350117631235ec62dcbe24bc405'

print("🔧 DIAGNÓSTICO: FORÇAR LONG SEM FECHAMENTO AUTOMÁTICO")
print("=" * 70)
print(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Objetivo: Abrir LONG e MANTER ABERTO para análise")
print(f"⚡ LIVE_TRADING: {os.getenv('LIVE_TRADING')}")
print("=" * 70)

try:
    # Importar tradingv4
    from tradingv4 import (
        WALLET_CONFIGS, 
        _init_dex_if_needed,
        build_df
    )
    
    print("✅ Módulos importados com sucesso")
    
    # Usar carteira principal
    wallet_config = WALLET_CONFIGS[0]  # CARTEIRA_PRINCIPAL
    print(f"🏦 Usando: {wallet_config.name}")
    print(f"📱 Address: {wallet_config.wallet_address}")
    
    # Inicializar DEX
    print("\n🔌 INICIALIZANDO CONEXÃO...")
    dex = _init_dex_if_needed(wallet_config)
    print(f"✅ DEX inicializado: {type(dex)}")
    
    # Configurações do trade forçado
    symbol = 'PUMP/USDC:USDC'
    target_side = 'long'  # Forçar LONG
    trade_size = 2.0  # $2 USD
    leverage = 5  # 5x leverage
    
    print(f"\n🎯 CONFIGURAÇÃO DO TRADE FORÇADO:")
    print(f"📊 Símbolo: {symbol}")
    print(f"📈 Lado: {target_side.upper()}")
    print(f"💵 Valor: ${trade_size} USD")
    print(f"⚡ Leverage: {leverage}x")
    
    # Verificar preço atual
    print(f"\n📊 OBTENDO PREÇO ATUAL...")
    try:
        ticker = dex.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"💰 Preço atual {symbol}: ${current_price:.6f}")
    except Exception as e:
        print(f"❌ Erro obtendo preço: {e}")
        current_price = 0.004  # Fallback
    
    # Calcular quantidade
    notional_value = trade_size * leverage
    quantity = notional_value / current_price
    print(f"🔢 Valor nocional: ${notional_value:.2f}")
    print(f"🔢 Quantidade: {quantity:.0f} contratos")
    
    # Verificar posições antes
    print(f"\n📊 POSIÇÕES ANTES DA OPERAÇÃO:")
    try:
        positions_before = dex.fetch_positions([symbol])
        active_before = [p for p in positions_before if p['contracts'] != 0]
        if active_before:
            for pos in active_before:
                print(f"   📈 {pos['symbol']}: {pos['contracts']:.0f} contratos ({pos['side']})")
        else:
            print("   📊 Nenhuma posição ativa")
    except Exception as e:
        print(f"   ⚠️ Erro verificando posições: {e}")
    
    # Configurar leverage
    print(f"\n⚡ CONFIGURANDO LEVERAGE...")
    try:
        dex.set_leverage(symbol, leverage)
        print(f"✅ Leverage {leverage}x configurada")
    except Exception as e:
        print(f"⚠️ Erro configurando leverage: {e}")
    
    # FORÇAR ENTRADA LONG
    print(f"\n🔥 FORÇANDO ENTRADA {target_side.upper()}...")
    order_time = time.time()
    
    try:
        # Criar ordem de compra (LONG)
        order = dex.create_order(
            symbol=symbol,
            type='market',
            side='buy',  # buy = LONG
            amount=quantity,
            price=None
        )
        
        execution_time = time.time() - order_time
        print(f"✅ ORDEM EXECUTADA: {order}")
        print(f"⏱️ Tempo de execução: {execution_time:.3f}s")
        
        if 'id' in order:
            order_id = order['id']
            print(f"🆔 Order ID: {order_id}")
            
            # MONITORAMENTO INTENSIVO (evitar fechamento automático)
            print(f"\n👀 MONITORAMENTO INTENSIVO - IMPEDINDO FECHAMENTO...")
            print(f"🚨 ATENÇÃO: Monitorando por 60 segundos para evitar fechamento automático")
            
            for i in range(60):
                try:
                    # Verificar posições a cada segundo
                    positions = dex.fetch_positions([symbol])
                    active_positions = [p for p in positions if p['contracts'] != 0]
                    
                    if active_positions:
                        pos = active_positions[0]
                        size = pos['contracts']
                        side = pos['side']
                        pnl = pos['unrealizedPnl']
                        
                        print(f"📊 [{i+1:2d}s] {symbol}: {size:.0f} contratos "
                              f"({side}) | PnL: ${pnl:.2f}")
                        
                        # PROTEÇÃO: Se detectar tentativa de fechamento, alertar
                        if abs(size) < quantity * 0.9:  # Se posição diminuiu mais de 10%
                            print(f"🚨 ALERTA: POSIÇÃO DIMINUIU! Original: {quantity:.0f} → Atual: {size:.0f}")
                            print(f"🔍 POSSÍVEL FECHAMENTO AUTOMÁTICO DETECTADO!")
                        
                    else:
                        print(f"❌ [{i+1:2d}s] POSIÇÃO FECHADA! Investigando...")
                        
                        # Se posição foi fechada, verificar trades recentes
                        try:
                            trades = dex.fetch_my_trades(symbol, limit=5)
                            recent_trades = [t for t in trades if t['timestamp'] > (order_time * 1000)]
                            
                            if recent_trades:
                                print(f"📋 Trades recentes detectados:")
                                for trade in recent_trades:
                                    trade_time = datetime.fromtimestamp(trade['timestamp']/1000)
                                    print(f"   🔄 {trade_time.strftime('%H:%M:%S')} | "
                                          f"{trade['side']} {trade['amount']:.0f} @ ${trade['price']:.6f}")
                            
                            print(f"🔍 DIAGNÓSTICO: Posição foi fechada automaticamente!")
                            break
                            
                        except Exception as e:
                            print(f"⚠️ Erro verificando trades: {e}")
                
                except Exception as e:
                    print(f"⚠️ [{i+1:2d}s] Erro no monitoramento: {e}")
                
                time.sleep(1)
            
            # Verificação final
            print(f"\n📊 VERIFICAÇÃO FINAL...")
            try:
                final_positions = dex.fetch_positions([symbol])
                final_active = [p for p in final_positions if p['contracts'] != 0]
                
                if final_active:
                    pos = final_active[0]
                    print(f"✅ POSIÇÃO MANTIDA: {pos['symbol']}")
                    print(f"📊 Size: {pos['contracts']:.0f} contratos")
                    print(f"📊 Side: {pos['side']}")
                    print(f"💰 PnL: ${pos['unrealizedPnl']:.2f}")
                    print(f"💰 Entry: ${pos.get('entryPrice', 'N/A')}")
                    
                    print(f"\n🎉 SUCESSO: Posição mantida aberta por 60 segundos!")
                    
                else:
                    print(f"❌ PROBLEMA: Posição foi fechada durante o monitoramento")
                    print(f"🔍 CAUSA: Sistema está fechando posições automaticamente")
                    
                    # Sugestões de diagnóstico
                    print(f"\n💡 POSSÍVEIS CAUSAS DO FECHAMENTO AUTOMÁTICO:")
                    print(f"   1. Stop Loss automático")
                    print(f"   2. Take Profit automático") 
                    print(f"   3. Sistema de proteção de risco")
                    print(f"   4. Sinal de saída sendo gerado imediatamente")
                    print(f"   5. Configuração de tempo limite de posição")
                    
            except Exception as e:
                print(f"❌ Erro na verificação final: {e}")
        
    except Exception as e:
        print(f"❌ ERRO CRIANDO ORDEM: {e}")
        
        # Analisar tipo de erro
        error_str = str(e)
        if "Insufficient margin" in error_str:
            print("💡 DIAGNÓSTICO: Margem insuficiente")
        elif "Invalid symbol" in error_str:
            print("💡 DIAGNÓSTICO: Símbolo inválido")
        else:
            print(f"💡 DIAGNÓSTICO: {error_str}")

except Exception as e:
    print(f"❌ ERRO GERAL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("🏁 DIAGNÓSTICO CONCLUÍDO")
print("💡 Se a posição foi fechada automaticamente, precisamos investigar:")
print("   - Configurações de Stop Loss/Take Profit")
print("   - Lógica de detecção de sinais de saída")
print("   - Sistema de proteção de risco")
print("=" * 70)
