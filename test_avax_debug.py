#!/usr/bin/env python3

# Constantes
ROI_HARD_STOP = -0.05
UNREALIZED_PNL_HARD_STOP = -0.05

def _get_position_for_vault(dex, symbol, vault=None):
    return dex.fetch_positions([symbol])[0] if dex.fetch_positions([symbol]) else None

def _get_pos_size_and_leverage(dex, symbol, vault=None):
    pos = _get_position_for_vault(dex, symbol, vault)
    if not pos:
        return 0.0, 1.0, None, None
    qty = float(pos.get("contracts") or 0.0)
    lev = float(pos.get("leverage") or 1.0)
    entry = float(pos.get("entryPrice") or 0.0)
    side = pos.get("side")
    return qty, lev, entry, side

def _compute_roi_from_price(entry, side, price, leverage=1.0):
    if side.lower() in ("long", "buy"):
        return ((price - entry) / entry) * leverage
    return ((entry - price) / entry) * leverage

def close_if_unrealized_pnl_breaches(dex, symbol, *, threshold=-0.05):
    print(f"[PNL_CHECK] Verificando {symbol} - threshold={threshold}")
    pos = _get_position_for_vault(dex, symbol, None)
    if not pos:
        print(f"[PNL_CHECK] {symbol}: Sem posição")
        return False
    
    pnl = pos.get("unrealizedPnl")
    if pnl is None:
        pnl = pos.get("info", {}).get("position", {}).get("unrealizedPnl")
    
    if pnl is None:
        print(f"[PNL_CHECK] {symbol}: unrealizedPnl não encontrado")
        return False
        
    pnl_f = float(str(pnl).replace(",", "."))
    print(f"[PNL_CHECK] {symbol}: PnL={pnl_f:.4f} vs threshold={threshold:.4f}")
    
    if pnl_f <= threshold:
        print(f"[PNL_CHECK] {symbol}: PnL BREACH! {pnl_f:.4f} <= {threshold:.4f} - tentando fechar...")
        qty, _, _, side = _get_pos_size_and_leverage(dex, symbol)
        if not side or qty <= 0:
            print(f"[PNL_CHECK] {symbol}: Não pode fechar - side={side} qty={qty}")
            return False
        exit_side = "sell" if str(side).lower() in ("long", "buy") else "buy"
        print(f"[PNL_CHECK] {symbol}: Executando market {exit_side} qty={qty}")
        dex.create_order(symbol, "market", exit_side, float(qty), None, {"reduceOnly": True})
        print(f"[PNL_CHECK] {symbol}: Posição fechada com sucesso por PnL!")
        return True
    else:
        print(f"[PNL_CHECK] {symbol}: PnL OK ({pnl_f:.4f} > {threshold:.4f})")
    return False

def close_if_roi_breaches(dex, symbol, current_px, *, threshold=ROI_HARD_STOP):
    print(f"[ROI_CHECK] Verificando {symbol} - threshold={threshold}")
    pos = _get_position_for_vault(dex, symbol, None)
    if not pos:
        print(f"[ROI_CHECK] {symbol}: Sem posição")
        return False
    
    qty, lev, entry, side = _get_pos_size_and_leverage(dex, symbol)
    
    if not side or qty <= 0 or not entry:
        print(f"[ROI_CHECK] {symbol}: Dados insuficientes - side={side} qty={qty} entry={entry}")
        return False
    
    roi_f = _compute_roi_from_price(entry, side, current_px, leverage=lev)
    print(f"[ROI_CHECK] {symbol}: ROI calculado={roi_f:.4f} (entry={entry} current={current_px} side={side} lev={lev})")
    
    print(f"[ROI_CHECK] {symbol}: ROI={roi_f:.4f} vs threshold={threshold:.4f}")
    
    if roi_f > threshold:
        print(f"[ROI_CHECK] {symbol}: ROI OK ({roi_f:.4f} > {threshold:.4f})")
        return False

    print(f"[ROI_CHECK] {symbol}: ROI BREACH! {roi_f:.4f} <= {threshold:.4f} - tentando fechar...")
    exit_side = "sell" if str(side).lower() in ("long", "buy") else "buy"
    print(f"[ROI_CHECK] {symbol}: Executando market {exit_side} qty={qty}")
    dex.create_order(symbol, "market", exit_side, float(qty), None, {"reduceOnly": True})
    print(f"[ROI_CHECK] {symbol}: Posição fechada com sucesso!")
    return True

def guard_close_all(dex, symbol, current_px):
    print(f"[GUARD] Verificando {symbol}")
    # PRIORITÁRIO: unrealized PnL primeiro
    if close_if_unrealized_pnl_breaches(dex, symbol, threshold=UNREALIZED_PNL_HARD_STOP):
        print(f"[GUARD] {symbol} fechado por unrealized PnL")
        return True
    # Verificar ROI
    if close_if_roi_breaches(dex, symbol, current_px, threshold=ROI_HARD_STOP):
        print(f"[GUARD] {symbol} fechado por ROI")
        return True
    print(f"[GUARD] {symbol}: Nenhum critério de fechamento atingido")
    return False

# Mock DEX para teste
class MockDex:
    def fetch_positions(self, symbols):
        return [{
            'symbol': 'AVAX/USDC:USDC',
            'side': 'long',
            'contracts': 0.72,
            'entryPrice': 27.937,
            'unrealizedPnl': -0.13032,
            'leverage': 10.0,
            'info': {'position': {'unrealizedPnl': '-0.13032'}}
        }]
    
    def create_order(self, symbol, type_, side, amount, price, params):
        print(f'[MOCK] create_order executado: {symbol} {type_} {side} {amount}')
        return {'id': 'test123'}

if __name__ == "__main__":
    print("=== TESTE DEBUG AVAX ===")
    dex = MockDex()
    result = guard_close_all(dex, 'AVAX/USDC:USDC', 27.7425)
    print(f"Resultado final: {result}")
