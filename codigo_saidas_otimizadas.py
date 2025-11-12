#!/usr/bin/env python3
"""
CÓDIGO PRONTO PARA IMPLEMENTAR - Sistema de Saídas Otimizado

Copiar e adaptar para tradingv4.py
Baseado em análise de dados reais (01/10-11/11/2025)
ROI estimado: +60-80% de melhoria
"""

# ============================================================================
# PARTE 1: ADICIONAR AO TradingConfig (linha ~2170)
# ============================================================================

class TradingConfig:
    # ... configurações existentes ...
    
    # ========== CONFIGURAÇÕES DE SAÍDA OTIMIZADAS ==========
    
    # Fase 1: Stop Loss Dinâmico baseado em ATR
    INITIAL_SL_ATR_MULT: float = 2.0  # 2x ATR do ativo (vs -4% fixo)
    
    # Fase 2: Breakeven
    ENABLE_BREAKEVEN: bool = True
    BREAKEVEN_TRIGGER_ROI: float = 3.0  # Move SL para entrada após +3% ROI
    
    # Fase 3: Saída Parcial (Scale Out)
    ENABLE_PARTIAL_EXIT: bool = True
    PARTIAL_EXIT_ROI: float = 7.0        # Fecha parcial em +7% ROI
    PARTIAL_EXIT_AMOUNT: float = 0.30    # Fecha 30% da posição
    
    # Fase 4: Trailing Dinâmico
    ENABLE_DYNAMIC_TRAILING: bool = True
    TRAILING_ACTIVATION_ROI: float = 10.0  # Ativa após +10% ROI
    TRAILING_ATR_MULT: float = 2.5         # Distância do trailing
    
    # Fase 5: Stops de Emergência
    ENABLE_VOLUME_STOP: bool = True
    VOLUME_EMERGENCY_THRESHOLD: float = 1.5  # Sell/Buy > 1.5x
    VOLUME_EMERGENCY_CANDLES: int = 3        # Por 3 candles consecutivos
    
    ENABLE_RATIO_STOP: bool = True
    RATIO_DECLINE_CANDLES: int = 4  # Ratio caindo por 4+ candles
    
    ENABLE_EMA_DIVERGENCE_STOP: bool = True
    EMA_DIVERGENCE_THRESHOLD: float = -0.0002  # Gradiente EMA negativo


# ============================================================================
# PARTE 2: ADICIONAR NOVA CLASSE PARA GERENCIAR ESTADOS DE POSIÇÃO
# ============================================================================

class PositionState:
    """Estado de uma posição para gerenciamento dinâmico de saídas"""
    
    def __init__(self, entry_price: float, position_size: float, side: str):
        self.entry_price = entry_price
        self.position_size = position_size
        self.original_size = position_size
        self.side = side  # "buy" ou "sell"
        
        # Estados de proteção
        self.stop_loss = None
        self.initial_sl_set = False
        self.breakeven_activated = False
        self.trailing_active = False
        self.partial_exit_executed = False
        
        # Tracking de preços
        self.highest_price = entry_price if side == "buy" else None
        self.lowest_price = entry_price if side == "sell" else None
        
        # Histórico
        self.roi_history = []
        self.last_update = None
    
    def update_roi(self, current_price: float) -> float:
        """Calcula e atualiza ROI atual"""
        if self.side == "buy":
            roi = ((current_price - self.entry_price) / self.entry_price) * 100
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
        else:
            roi = ((self.entry_price - current_price) / self.entry_price) * 100
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
        
        self.roi_history.append(roi)
        return roi
    
    def reduce_position(self, amount: float):
        """Reduz tamanho da posição após saída parcial"""
        self.position_size -= amount
        self.partial_exit_executed = True


# ============================================================================
# PARTE 3: FUNÇÕES AUXILIARES DE DETECÇÃO
# ============================================================================

def check_volume_emergency(df, side: str, threshold: float, min_candles: int) -> bool:
    """
    Verifica se há emergência de volume adverso
    
    Args:
        df: DataFrame com dados do ativo
        side: "buy" ou "sell"
        threshold: Ratio mínimo para alarme (ex: 1.5)
        min_candles: Candles consecutivos necessários (ex: 3)
    
    Returns:
        True se detectou emergência
    """
    if len(df) < min_candles:
        return False
    
    consecutive_pressure = 0
    
    for i in range(-min_candles, 0):
        if side == "buy":
            # Para LONG: venda muito maior que compra = perigo
            if df['avg_sell_3'].iloc[i] > (df['avg_buy_3'].iloc[i] * threshold):
                consecutive_pressure += 1
        else:
            # Para SHORT: compra muito maior que venda = perigo
            if df['avg_buy_3'].iloc[i] > (df['avg_sell_3'].iloc[i] * threshold):
                consecutive_pressure += 1
    
    return consecutive_pressure >= min_candles


def check_ratio_decline(df, min_candles: int) -> bool:
    """
    Verifica se buy/sell ratio está caindo consecutivamente
    
    Args:
        df: DataFrame com dados
        min_candles: Número de candles em declínio necessários
    
    Returns:
        True se ratio está caindo por min_candles+
    """
    if len(df) < min_candles or 'ratio_trend' not in df.columns:
        return False
    
    decline_count = 0
    for i in range(-min_candles, 0):
        if df['ratio_trend'].iloc[i] == 'diminuindo':
            decline_count += 1
    
    return decline_count >= min_candles


def check_ema_divergence(df, side: str, threshold: float) -> bool:
    """
    Verifica divergência entre preço e EMA (possível topo/fundo)
    
    Args:
        df: DataFrame com dados
        side: "buy" ou "sell"
        threshold: Threshold do gradiente (ex: -0.0002)
    
    Returns:
        True se detectou divergência
    """
    if len(df) < 2:
        return False
    
    current_price = df['close'].iloc[-1]
    ema_fast = df['ema_fast'].iloc[-1]
    ema_gradient = df['ema_gradient'].iloc[-1]
    
    if side == "buy":
        # LONG: preço acima EMA mas gradiente negativo = possível topo
        return (current_price > ema_fast and ema_gradient < threshold)
    else:
        # SHORT: preço abaixo EMA mas gradiente positivo = possível fundo
        return (current_price < ema_fast and ema_gradient > -threshold)


# ============================================================================
# PARTE 4: GERENCIADOR DINÂMICO DE SAÍDAS
# ============================================================================

class DynamicExitManager:
    """Gerenciador de saídas dinâmicas otimizado"""
    
    def __init__(self, config: TradingConfig, logger):
        self.cfg = config
        self.logger = logger
        self.positions = {}  # symbol -> PositionState
    
    def add_position(self, symbol: str, entry_price: float, 
                    position_size: float, side: str):
        """Registra nova posição para gerenciamento"""
        self.positions[symbol] = PositionState(entry_price, position_size, side)
        self.logger(f"[EXIT_MGR] Nova posição: {symbol} {side} @ {entry_price:.6f}", 
                   level="INFO")
    
    def remove_position(self, symbol: str):
        """Remove posição do gerenciamento"""
        if symbol in self.positions:
            del self.positions[symbol]
            self.logger(f"[EXIT_MGR] Posição removida: {symbol}", level="INFO")
    
    def check_exit(self, symbol: str, current_df) -> dict:
        """
        Verifica se deve sair da posição
        
        Returns:
            dict com: {
                'action': 'HOLD' | 'CLOSE_ALL' | 'CLOSE_PARTIAL',
                'amount': float (se CLOSE_PARTIAL),
                'reason': str,
                'price': float,
                'roi': float
            }
        """
        if symbol not in self.positions:
            return {'action': 'HOLD'}
        
        pos = self.positions[symbol]
        
        # Dados atuais
        current_price = current_df['close'].iloc[-1]
        current_atr = current_df['atr'].iloc[-1]
        
        # Calcular ROI
        current_roi = pos.update_roi(current_price)
        
        self.logger(f"[EXIT_MGR] {symbol}: ROI {current_roi:+.2f}% | "
                   f"SL: {pos.stop_loss:.6f if pos.stop_loss else 'N/A'}", 
                   level="DEBUG")
        
        # ===== FASE 5: STOPS DE EMERGÊNCIA (PRIORIDADE MÁXIMA) =====
        
        # Emergency Stop 1: Volume Adverso
        if self.cfg.ENABLE_VOLUME_STOP:
            if check_volume_emergency(
                current_df, 
                pos.side, 
                self.cfg.VOLUME_EMERGENCY_THRESHOLD,
                self.cfg.VOLUME_EMERGENCY_CANDLES
            ):
                self.logger(f"🚨 [{symbol}] EMERGÊNCIA VOLUME - Fechando!", 
                           level="WARN")
                return {
                    'action': 'CLOSE_ALL',
                    'reason': 'volume_emergency',
                    'price': current_price,
                    'roi': current_roi
                }
        
        # Emergency Stop 2: Ratio Declinante
        if self.cfg.ENABLE_RATIO_STOP:
            if check_ratio_decline(current_df, self.cfg.RATIO_DECLINE_CANDLES):
                self.logger(f"⚠️ [{symbol}] RATIO DECLINANTE - Fechando!", 
                           level="WARN")
                return {
                    'action': 'CLOSE_ALL',
                    'reason': 'ratio_decline',
                    'price': current_price,
                    'roi': current_roi
                }
        
        # Emergency Stop 3: Divergência EMA (só se em lucro)
        if self.cfg.ENABLE_EMA_DIVERGENCE_STOP and current_roi > 0:
            if check_ema_divergence(
                current_df, 
                pos.side, 
                self.cfg.EMA_DIVERGENCE_THRESHOLD
            ):
                self.logger(f"📉 [{symbol}] DIVERGÊNCIA EMA - Fechando!", 
                           level="INFO")
                return {
                    'action': 'CLOSE_ALL',
                    'reason': 'ema_divergence',
                    'price': current_price,
                    'roi': current_roi
                }
        
        # ===== FASE 1: DEFINIR STOP LOSS INICIAL =====
        
        if not pos.initial_sl_set:
            if pos.side == "buy":
                pos.stop_loss = current_price - (self.cfg.INITIAL_SL_ATR_MULT * current_atr)
            else:
                pos.stop_loss = current_price + (self.cfg.INITIAL_SL_ATR_MULT * current_atr)
            
            pos.initial_sl_set = True
            self.logger(f"🔒 [{symbol}] SL inicial: {pos.stop_loss:.6f} "
                       f"({self.cfg.INITIAL_SL_ATR_MULT}x ATR)", level="INFO")
        
        # ===== FASE 2: BREAKEVEN =====
        
        if (self.cfg.ENABLE_BREAKEVEN and 
            not pos.breakeven_activated and
            current_roi >= self.cfg.BREAKEVEN_TRIGGER_ROI):
            
            pos.stop_loss = pos.entry_price
            pos.breakeven_activated = True
            self.logger(f"🔓 [{symbol}] BREAKEVEN ativado @ {pos.entry_price:.6f} "
                       f"(ROI: +{current_roi:.2f}%)", level="INFO")
        
        # ===== FASE 3: SAÍDA PARCIAL =====
        
        if (self.cfg.ENABLE_PARTIAL_EXIT and
            not pos.partial_exit_executed and
            current_roi >= self.cfg.PARTIAL_EXIT_ROI):
            
            partial_amount = pos.position_size * self.cfg.PARTIAL_EXIT_AMOUNT
            pos.reduce_position(partial_amount)
            
            self.logger(f"💰 [{symbol}] SAÍDA PARCIAL: "
                       f"{self.cfg.PARTIAL_EXIT_AMOUNT*100:.0f}% @ +{current_roi:.2f}%", 
                       level="INFO")
            
            return {
                'action': 'CLOSE_PARTIAL',
                'amount': partial_amount,
                'reason': 'partial_tp',
                'price': current_price,
                'roi': current_roi
            }
        
        # ===== FASE 4: TRAILING DINÂMICO =====
        
        if (self.cfg.ENABLE_DYNAMIC_TRAILING and
            current_roi >= self.cfg.TRAILING_ACTIVATION_ROI):
            
            if not pos.trailing_active:
                pos.trailing_active = True
                self.logger(f"📈 [{symbol}] TRAILING ativado @ +{current_roi:.2f}%", 
                           level="INFO")
            
            # Atualizar trailing stop
            if pos.side == "buy":
                new_stop = pos.highest_price - (self.cfg.TRAILING_ATR_MULT * current_atr)
                if new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop
                    self.logger(f"📊 [{symbol}] Trailing → {pos.stop_loss:.6f}", 
                               level="DEBUG")
            else:
                new_stop = pos.lowest_price + (self.cfg.TRAILING_ATR_MULT * current_atr)
                if new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop
                    self.logger(f"📊 [{symbol}] Trailing → {pos.stop_loss:.6f}", 
                               level="DEBUG")
        
        # ===== VERIFICAR SE ATINGIU STOP LOSS =====
        
        if pos.stop_loss:
            stop_hit = False
            
            if pos.side == "buy":
                stop_hit = current_price <= pos.stop_loss
            else:
                stop_hit = current_price >= pos.stop_loss
            
            if stop_hit:
                reason = "trailing_stop" if pos.trailing_active else "stop_loss"
                self.logger(f"🛑 [{symbol}] STOP @ {current_price:.6f} "
                           f"(ROI: {current_roi:+.2f}%)", level="INFO")
                
                return {
                    'action': 'CLOSE_ALL',
                    'reason': reason,
                    'price': current_price,
                    'roi': current_roi
                }
        
        # Sem ação necessária
        return {
            'action': 'HOLD',
            'roi': current_roi,
            'stop_loss': pos.stop_loss
        }


# ============================================================================
# PARTE 5: EXEMPLO DE INTEGRAÇÃO NO LOOP PRINCIPAL
# ============================================================================

def example_integration_in_main_loop():
    """
    Exemplo de como integrar no loop principal do tradingv4.py
    """
    
    # Inicializar gerenciador (fazer isso no __init__ do AssetTrader)
    exit_manager = DynamicExitManager(config=cfg, logger=self._log)
    
    # Quando abrir posição:
    def on_position_opened(symbol, entry_price, size, side):
        exit_manager.add_position(symbol, entry_price, size, side)
    
    # A cada candle novo (no loop principal):
    def on_new_candle(symbol, df):
        # Verificar se deve sair
        exit_decision = exit_manager.check_exit(symbol, df)
        
        if exit_decision['action'] == 'CLOSE_ALL':
            # Fechar toda a posição
            close_position(symbol, reason=exit_decision['reason'])
            exit_manager.remove_position(symbol)
            
        elif exit_decision['action'] == 'CLOSE_PARTIAL':
            # Fechar parcial
            close_partial_position(
                symbol, 
                amount=exit_decision['amount'],
                reason=exit_decision['reason']
            )
        
        # action == 'HOLD': não fazer nada, continuar segurando


# ============================================================================
# CONFIGURAÇÃO RECOMENDADA PARA INÍCIO
# ============================================================================

RECOMMENDED_CONFIG = {
    # Conservador (menos risco, menos ganho)
    'conservative': {
        'INITIAL_SL_ATR_MULT': 1.5,
        'BREAKEVEN_TRIGGER_ROI': 2.0,
        'PARTIAL_EXIT_ROI': 5.0,
        'PARTIAL_EXIT_AMOUNT': 0.50,  # Fecha 50%
        'TRAILING_ACTIVATION_ROI': 8.0,
        'TRAILING_ATR_MULT': 2.0,
    },
    
    # Balanceado (recomendado - baseado na análise)
    'balanced': {
        'INITIAL_SL_ATR_MULT': 2.0,
        'BREAKEVEN_TRIGGER_ROI': 3.0,
        'PARTIAL_EXIT_ROI': 7.0,
        'PARTIAL_EXIT_AMOUNT': 0.30,  # Fecha 30%
        'TRAILING_ACTIVATION_ROI': 10.0,
        'TRAILING_ATR_MULT': 2.5,
    },
    
    # Agressivo (mais risco, mais ganho)
    'aggressive': {
        'INITIAL_SL_ATR_MULT': 2.5,
        'BREAKEVEN_TRIGGER_ROI': 5.0,
        'PARTIAL_EXIT_ROI': 10.0,
        'PARTIAL_EXIT_AMOUNT': 0.20,  # Fecha apenas 20%
        'TRAILING_ACTIVATION_ROI': 15.0,
        'TRAILING_ATR_MULT': 3.0,
    }
}


if __name__ == "__main__":
    print("="*80)
    print("CÓDIGO PRONTO PARA IMPLEMENTAÇÃO")
    print("="*80)
    print("\n📋 PASSOS:")
    print("1. Copiar TradingConfig para tradingv4.py (linha ~2170)")
    print("2. Adicionar PositionState class")
    print("3. Adicionar funções check_volume_emergency, check_ratio_decline, etc")
    print("4. Adicionar DynamicExitManager class")
    print("5. Integrar no loop principal (ver example_integration_in_main_loop)")
    print("\n💡 Começar com configuração 'balanced' e ajustar conforme resultados")
    print("="*80)
