#!/usr/bin/env python3
"""
Patch para corrigir as funções de safety check para múltiplas carteiras
"""

def create_fixed_safety_functions():
    """Cria versões corrigidas das funções de safety check"""
    
    return '''
    def check_all_trailing_stops_v4(asset_state) -> None:
        """Verifica e ajusta trailing stops dinâmicos para TODAS as posições em TODAS as carteiras."""
        for asset in ASSET_SETUPS:
            state = asset_state.get(asset.name)
            if state is None:
                continue  # Asset ainda não foi inicializado
                
            strategies = state.get("strategies", {})
            
            for wallet_name, strategy in strategies.items():
                try:
                    # Obter DEX específico da carteira
                    wallet_config = next((w for w in WALLET_CONFIGS if w.name == wallet_name), None)
                    if not wallet_config:
                        continue
                        
                    wallet_dex = _init_dex_if_needed(wallet_config)
                    
                    # Verificar se há posição aberta nesta carteira
                    positions = wallet_dex.fetch_positions([asset.hl_symbol])
                    if not positions or float(positions[0].get("contracts", 0)) == 0:
                        continue
                        
                    pos = positions[0]
                    
                    # Executar trailing stop dinâmico para esta posição
                    try:
                        # Criar um DataFrame dummy para o log
                        import pandas as pd
                        dummy_df = pd.DataFrame()
                        strategy._ensure_position_protections(pos, df_for_log=dummy_df)
                        _log_global("TRAILING_CHECK", f"{asset.name} ({wallet_name}): Trailing stop verificado")
                    except Exception as e:
                        _log_global("TRAILING_CHECK", f"{asset.name} ({wallet_name}): Erro no trailing stop - {e}", level="WARN")
                        
                except Exception as e:
                    _log_global("TRAILING_CHECK", f"Erro verificando {asset.name} ({wallet_name}): {type(e).__name__}: {e}", level="WARN")

    def fast_safety_check_v4(asset_state) -> None:
        """Executa verificações rápidas de segurança (PnL, ROI) para todos os ativos em TODAS as carteiras."""
        
        # Debug: verificar quantos assets estão no asset_state
        _log_global("FAST_SAFETY_V4", f"Asset_state contém {len(asset_state)} assets: {list(asset_state.keys())}", level="DEBUG")
        
        for wallet_config in WALLET_CONFIGS:
            _log_global("FAST_SAFETY_V4", f"Verificando segurança para {wallet_config.name}...")
            
            try:
                wallet_dex = _init_dex_if_needed(wallet_config)
                open_positions = []
                
                for asset in ASSET_SETUPS:
                    state = asset_state.get(asset.name)
                    
                    try:
                        # Verificar se há posição aberta nesta carteira
                        cache_key = f"positions_{asset.hl_symbol}_{wallet_config.name}"
                        positions = _get_cached_api_call(cache_key, wallet_dex.fetch_positions, [asset.hl_symbol])
                        if not positions or float(positions[0].get("contracts", 0)) == 0:
                            continue
                            
                        pos = positions[0]
                        emergency_closed = False
                        
                        # Se não tem strategy no asset_state, pular mas ainda mostrar no log
                        if state is None:
                            _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Posição encontrada mas asset não inicializado", level="DEBUG")
                            continue
                        
                        # Obter strategy específica da carteira
                        strategies = state.get("strategies", {})
                        strategy = strategies.get(wallet_config.name)
                        if not strategy:
                            _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Strategy não encontrada", level="DEBUG")
                            continue
                    
                        # Coletar informações da posição
                        side = pos.get("side") or pos.get("positionSide", "")
                        contracts = float(pos.get("contracts", 0))
                        unrealized_pnl = float(pos.get("unrealizedPnl", 0))
                        
                        # Calcular ROI real usando mesma fórmula do trailing stop
                        roi_pct = 0.0
                        try:
                            position_value = pos.get("positionValue") or pos.get("notional") or pos.get("size")
                            leverage = float(pos.get("leverage", 10))
                            
                            if position_value is None:
                                # Calcular position_value manualmente se necessário
                                current_px = 0
                                if strategy:
                                    current_px = strategy._preco_atual()
                                
                                if current_px == 0:
                                    # Fallback: usar ticker se não conseguimos do strategy
                                    try:
                                        cache_key = f"ticker_{asset.hl_symbol}_{wallet_config.name}"
                                        ticker = _get_cached_api_call(cache_key, wallet_dex.fetch_ticker, asset.hl_symbol)
                                        current_px = float(ticker.get("last", 0) or 0)
                                    except Exception:
                                        current_px = 0
                                
                                if contracts > 0 and current_px > 0:
                                    position_value = abs(contracts * current_px)
                        
                            if position_value and position_value > 0 and leverage > 0:
                                # Mesma fórmula: (PnL / (position_value / leverage)) * 100
                                capital_real = position_value / leverage
                                roi_pct = (unrealized_pnl / capital_real) * 100
                        except Exception as e:
                            _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Erro calculando ROI - {e}", level="WARN")
                        
                        # Adicionar à lista de posições abertas com status
                        status = "OK"
                        if unrealized_pnl <= UNREALIZED_PNL_HARD_STOP:
                            status = f"⚠️ PnL CRÍTICO: ${unrealized_pnl:.3f} (será fechado!)"
                        elif roi_pct <= ROI_HARD_STOP:
                            status = f"⚠️ ROI CRÍTICO: {roi_pct:.1f}% (será fechado!)"
                        elif unrealized_pnl < -0.01:  # Alertar perdas > -1 cent
                            status = f"📉 PnL: ${unrealized_pnl:.3f} ROI: {roi_pct:.1f}%"
                        elif unrealized_pnl > 0.01:   # Alertar lucros > +1 cent
                            status = f"📈 PnL: +${unrealized_pnl:.3f} ROI: +{roi_pct:.1f}%"
                        
                        open_positions.append(f"{asset.name} {side.upper()} ({wallet_config.name}): {status}")
                        
                        # PRIORITÁRIO: Verificar unrealized PnL primeiro
                        if unrealized_pnl <= UNREALIZED_PNL_HARD_STOP:
                            _log_global("FAST_SAFETY_V4", f"[DEBUG_CLOSE] 🚨 TESTE PnL: {unrealized_pnl:.4f} <= {UNREALIZED_PNL_HARD_STOP} = True", level="ERROR")
                            try:
                                qty = abs(contracts)
                                side_norm = strategy._norm_side(side)
                                exit_side = "sell" if side_norm in ("buy", "long") else "buy"
                                
                                # Buscar preço atual para ordem market
                                ticker = wallet_dex.fetch_ticker(asset.hl_symbol)
                                current_price = float(ticker.get("last", 0) or 0)
                                if current_price <= 0:
                                    continue
                                    
                                # Ajustar preço para garantir execução
                                if exit_side == "sell":
                                    order_price = current_price * 0.995  # Ligeiramente abaixo para long
                                else:
                                    order_price = current_price * 1.005  # Ligeiramente acima para short
                                
                                wallet_dex.create_order(asset.hl_symbol, "market", exit_side, qty, order_price, {"reduceOnly": True})
                                emergency_closed = True
                                _clear_high_water_mark(asset.name)  # Limpar HWM após fechamento de emergência
                                _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Emergência PnL ${unrealized_pnl:.4f} - posição fechada", level="ERROR")
                            except Exception as e:
                                _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Erro fechando por PnL - {e}", level="WARN")
                        else:
                            _log_global("FAST_SAFETY_V4", f"[DEBUG_CLOSE] ✅ PNL OK: {unrealized_pnl:.4f} > {UNREALIZED_PNL_HARD_STOP}", level="DEBUG")
                        
                        # Se não fechou por PnL, verificar ROI
                        if not emergency_closed and roi_pct <= ROI_HARD_STOP:
                            _log_global("FAST_SAFETY_V4", f"[DEBUG_CLOSE] 🚨 TESTE ROI: {roi_pct:.4f} <= {ROI_HARD_STOP} = True", level="ERROR")
                            try:
                                qty = abs(contracts)
                                side_norm = strategy._norm_side(side)
                                exit_side = "sell" if side_norm in ("buy", "long") else "buy"
                                
                                # Buscar preço atual para ordem market
                                ticker = wallet_dex.fetch_ticker(asset.hl_symbol)
                                current_price = float(ticker.get("last", 0) or 0)
                                if current_price <= 0:
                                    continue
                                    
                                # Ajustar preço para garantir execução
                                if exit_side == "sell":
                                    order_price = current_price * 0.995
                                else:
                                    order_price = current_price * 1.005
                                
                                wallet_dex.create_order(asset.hl_symbol, "market", exit_side, qty, order_price, {"reduceOnly": True})
                                emergency_closed = True
                                _clear_high_water_mark(asset.name)  # Limpar HWM após fechamento de emergência
                                _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Emergência ROI {roi_pct:.4f}% - posição fechada", level="ERROR")
                            except Exception as e:
                                _log_global("FAST_SAFETY_V4", f"{asset.name} ({wallet_config.name}): Erro fechando por ROI - {e}", level="WARN")
                        else:
                            if not emergency_closed:
                                _log_global("FAST_SAFETY_V4", f"[DEBUG_CLOSE] ✅ ROI OK: {roi_pct:.4f} > {ROI_HARD_STOP}", level="DEBUG")
                                
                    except Exception as e:
                        _log_global("FAST_SAFETY_V4", f"Erro no safety check {asset.name} ({wallet_config.name}): {type(e).__name__}: {e}", level="WARN")
                
                # Log resumo das posições abertas para esta carteira
                if open_positions:
                    _log_global("FAST_SAFETY_V4", f"{wallet_config.name} - Posições monitoradas: {' | '.join(open_positions)}", level="INFO")
                    
            except Exception as e:
                _log_global("FAST_SAFETY_V4", f"Erro verificando carteira {wallet_config.name}: {e}", level="ERROR")
        
        # Log final se nenhuma posição foi encontrada
        all_positions = []
        for wallet_config in WALLET_CONFIGS:
            try:
                wallet_dex = _init_dex_if_needed(wallet_config)
                for asset in ASSET_SETUPS:
                    positions = wallet_dex.fetch_positions([asset.hl_symbol])
                    if positions and float(positions[0].get("contracts", 0)) != 0:
                        all_positions.append(f"{asset.name} ({wallet_config.name})")
            except:
                pass
                
        if not all_positions:
            _log_global("FAST_SAFETY_V4", "Nenhuma posição aberta em nenhuma carteira", level="DEBUG")
    '''

if __name__ == "__main__":
    print("Script para corrigir as funções de safety check")
    print(create_fixed_safety_functions())
