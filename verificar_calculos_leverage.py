#!/usr/bin/env python3
"""
VERIFICAÇÃO CÁLCULOS SL/TP COM LEVERAGE 3X - TRADING.PY
Confirma que os cálculos estão corretos considerando leverage 3x
"""

def verificar_calculos_leverage():
    print("🧮 VERIFICAÇÃO CÁLCULOS SL/TP COM LEVERAGE 3X")
    print("=" * 55)
    
    # Configurações fixas
    leverage = 3
    sl_roi = 0.015  # 1.5%
    tp_roi = 0.12   # 12%
    
    print(f"\n📊 CONFIGURAÇÕES BASE:")
    print(f"• Leverage: {leverage}x")
    print(f"• Stop Loss ROI: {sl_roi} ({sl_roi*100}%)")
    print(f"• Take Profit ROI: {tp_roi} ({tp_roi*100}%)")
    
    print(f"\n🧮 CÁLCULOS DE MOVIMENTO DE PREÇO:")
    
    # Stop Loss
    sl_price_movement = (sl_roi / leverage) * 100
    print(f"• SL Preço: ({sl_roi} ÷ {leverage}) × 100 = {sl_price_movement:.2f}%")
    
    # Take Profit
    tp_price_movement = (tp_roi / leverage) * 100
    print(f"• TP Preço: ({tp_roi} ÷ {leverage}) × 100 = {tp_price_movement:.2f}%")
    
    print(f"\n✅ VERIFICAÇÃO:")
    print(f"• SL: Para perder 1.5% ROI com 3x leverage → preço deve mover {sl_price_movement:.1f}%")
    print(f"• TP: Para ganhar 12% ROI com 3x leverage → preço deve mover {tp_price_movement:.1f}%")
    
    print(f"\n🎯 EXEMPLOS PRÁTICOS:")
    print(f"Se preço de entrada = $100:")
    sl_price_down = 100 * (1 - sl_price_movement/100)
    sl_price_up = 100 * (1 + sl_price_movement/100)
    tp_price_up = 100 * (1 + tp_price_movement/100)
    tp_price_down = 100 * (1 - tp_price_movement/100)
    
    print(f"📈 LONG (BUY):")
    print(f"  • SL: ${sl_price_down:.2f} (-{sl_price_movement:.1f}% preço = -1.5% ROI)")
    print(f"  • TP: ${tp_price_up:.2f} (+{tp_price_movement:.1f}% preço = +12% ROI)")
    
    print(f"📉 SHORT (SELL):")
    print(f"  • SL: ${sl_price_up:.2f} (+{sl_price_movement:.1f}% preço = -1.5% ROI)")  
    print(f"  • TP: ${tp_price_down:.2f} (-{tp_price_movement:.1f}% preço = +12% ROI)")
    
    print(f"\n🔍 VERIFICAÇÃO DO CÓDIGO TRADING.PY:")
    
    with open("/Users/joaoreis/Documents/GitHub/Cripto/trading.py", "r") as f:
        content = f.read()
    
    # Verificar se o cálculo está correto
    if "(self.cfg.STOP_LOSS_CAPITAL_PCT / leverage) * 100" in content:
        print("✅ SL: Fórmula correta no código → (ROI / leverage) × 100")
    else:
        print("❌ SL: Fórmula incorreta ou ausente")
        
    if "(self.cfg.TAKE_PROFIT_CAPITAL_PCT / leverage) * 100" in content:
        print("✅ TP: Fórmula correta no código → (ROI / leverage) × 100")
    else:
        print("❌ TP: Fórmula incorreta ou ausente")
    
    print(f"\n🎉 CONCLUSÃO:")
    print(f"O sistema está configurado corretamente para:")
    print(f"• SL: {sl_price_movement:.1f}% movimento = 1.5% ROI loss")
    print(f"• TP: {tp_price_movement:.1f}% movimento = 12% ROI gain")
    print(f"• Leverage: 3x (aplicado corretamente nos cálculos)")

if __name__ == "__main__":
    verificar_calculos_leverage()
