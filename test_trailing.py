#!/usr/bin/env python3
"""
Teste manual do trailing stop para verificar se está funcionando corretamente
"""

def test_compute_trailing_stop():
    # Dados REAIS da posição PUMP (conforme imagem)
    entry_price = 0.005029      # Entry Price da posição
    current_price = 0.005123    # Mark Price atual
    expected_sl = 0.005039      # SL Price que deveria ser
    
    leverage = 5.0              # alavancagem 5x
    margin = 0.10               # TRAILING_ROI_MARGIN = 10%
    norm_side = "buy"           # posição LONG
    
    print(f"=== TESTE TRAILING STOP (DADOS REAIS PUMP) ===")
    print(f"Entry price: {entry_price:.6f}")
    print(f"Current price (Mark): {current_price:.6f}")
    print(f"Expected SL price: {expected_sl:.6f}")
    print(f"Side: {norm_side}")
    print(f"Leverage: {leverage}x")
    print(f"Margin: {margin*100}%")
    
    # Calcula ROI base
    roi = (current_price / entry_price) - 1.0
    print(f"\nROI base: {roi:.4f} ({roi*100:.2f}%)")
    
    # ROI alavancado
    levered_roi = roi * leverage
    print(f"ROI alavancado: {levered_roi:.4f} ({levered_roi*100:.2f}%)")
    
    # Target ROI (ROI alavancado - margin)
    target_roi = levered_roi - margin
    print(f"Target ROI: {target_roi:.4f} ({target_roi*100:.2f}%)")
    
    # Calcular trailing price
    trailing_price = entry_price * (target_roi + 1.0)
    print(f"Trailing price calculado: {trailing_price:.6f}")
    
    # Para LONG: trailing stop deve estar abaixo do preço atual
    result = max(0.0, min(trailing_price, current_price))
    print(f"Resultado final: {result:.6f}")
    
    # Comparar com o valor esperado
    print(f"\n=== COMPARAÇÃO ===")
    print(f"Esperado: {expected_sl:.6f}")
    print(f"Calculado: {result:.6f}")
    print(f"Diferença: {abs(result - expected_sl):.6f}")
    print(f"Match? {abs(result - expected_sl) < 0.000001}")
    
    # Verificar se está acima do stop loss padrão
    # STOP_LOSS_CAPITAL_PCT = 6% / leverage = 6% / 5 = 1.2%
    stop_loss_ratio = 0.06 / leverage
    stop_px = entry_price * (1.0 - stop_loss_ratio)
    print(f"\nStop loss padrão: {stop_px:.6f} (-{stop_loss_ratio*100:.1f}%)")
    print(f"Trailing price: {result:.6f}")
    print(f"Trailing > Stop loss? {result > stop_px}")
    
    # Calcular qual ROI o trailing stop representa
    trailing_roi = (result / entry_price) - 1.0
    trailing_roi_levered = trailing_roi * leverage
    print(f"\nTrailing stop ROI: {trailing_roi:.4f} ({trailing_roi*100:.2f}%)")
    print(f"Trailing stop ROI alavancado: {trailing_roi_levered:.4f} ({trailing_roi_levered*100:.2f}%)")
    
    return result

if __name__ == "__main__":
    test_compute_trailing_stop()
