#!/usr/bin/env python3

"""
Sistema de Trading Simplificado - Apenas SimpleRatioStrategy
Removidas todas as funcionalidades desnecessárias: learner, fast, safat, etc.
"""

import os
import sys
import time
import sqlite3
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

# ===== CONFIGURAÇÃO DE TIMEZONE =====
UTC = timezone.utc

# ===== IMPORTS ESSENCIAIS =====
import ccxt

# ===== LOGGING GLOBAL =====
def _log_global(channel: str, message: str, level: str = "INFO"):
    """Sistema de log global simplificado"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] [{channel}] {message}", flush=True)

# ===== CACHE DE DADOS =====
class DataCache:
    def __init__(self, ttl_seconds: int = 30):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str):
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
        return None
    
    def set(self, key: str, data):
        self.cache[key] = (time.time(), data)
    
    def clear(self):
        self.cache.clear()

# Cache global
_data_cache = DataCache(ttl_seconds=30)

def _get_cached_api_call(cache_key: str, func, *args, **kwargs):
    """Wrapper para chamadas de API com cache"""
    cached = _data_cache.get(cache_key)
    if cached is not None:
        return cached
    
    result = func(*args, **kwargs)
    _data_cache.set(cache_key, result)
    return result

# ===== EXCEÇÃO PERSONALIZADA =====
class MarketDataUnavailable(Exception):
    """Sinaliza indisponibilidade temporária de candles para um ativo/timeframe."""
    pass

# ===== CONFIGURAÇÃO DE CARTEIRAS =====
@dataclass
class WalletConfig:
    name: str
    is_subconta: bool = False
    vault_address: Optional[str] = None
    
    def get_dex_instance(self):
        """Retorna instância do DEX para esta carteira"""
        if self.is_subconta and self.vault_address:
            return RealDataDex(vault_address=self.vault_address)
        else:
            return RealDataDex()

# Configurações de carteiras disponíveis
WALLET_CONFIGS = [
    WalletConfig(
        name="Carteira Principal",
        is_subconta=False,
        vault_address=None
    ),
    WalletConfig(
        name="Subconta Trading",
        is_subconta=True,
        vault_address="0x5ff0f14d577166f9ede3d9568a423166be61ea9d"
    )
]

# ===== DEX REAL =====
class RealDataDex:
    """Interface simplificada para Hyperliquid"""
    
    def __init__(self, vault_address: Optional[str] = None):
        # Configuração básica para Hyperliquid
        self.vault_address = vault_address
        self._setup_hyperliquid()
    
    def _setup_hyperliquid(self):
        """Configura conexão com Hyperliquid"""
        try:
            # Configuração básica do ccxt para Hyperliquid
            self.exchange = ccxt.hyperliquid({
                'sandbox': False,
                'options': {
                    'defaultType': 'swap',
                }
            })
            if self.vault_address:
                self.exchange.options['vault'] = self.vault_address
        except Exception as e:
            _log_global("DEX", f"Erro configurando Hyperliquid: {e}", "ERROR")
    
    def fetch_ticker(self, symbol: str):
        """Busca ticker do símbolo"""
        return self.exchange.fetch_ticker(symbol)
    
    def fetch_positions(self, symbols: List[str] = None):
        """Busca posições abertas"""
        return self.exchange.fetch_positions(symbols)
    
    def fetch_open_orders(self, symbol: str):
        """Busca ordens abertas"""
        return self.exchange.fetch_open_orders(symbol)
    
    def create_order(self, symbol: str, type_: str, side: str, amount: float, price: float, params: dict = None):
        """Cria ordem"""
        return self.exchange.create_order(symbol, type_, side, amount, price, params or {})
    
    def cancel_order(self, order_id: str, symbol: str):
        """Cancela ordem"""
        return self.exchange.cancel_order(order_id, symbol)
    
    def set_leverage(self, leverage: int, symbol: str, params: dict = None):
        """Define leverage"""
        return self.exchange.set_leverage(leverage, symbol, params or {})

# ===== LOGGER DE TRADES =====
class TradeLogger:
    def __init__(self, df_columns: pd.Index, 
                 use_local_csv: bool = True, 
                 use_sqlite_db: bool = False,
                 csv_path: str = None,
                 db_path: str = None):
        self.df_columns = df_columns
        self.use_local_csv = use_local_csv
        self.use_sqlite_db = use_sqlite_db
        self.csv_path = csv_path or "trading_log.csv"
        self.db_path = db_path or "trading.db"
    
    def log_trade(self, event_type: str, details: Dict[str, Any]):
        """Log de trade simplificado"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            **details
        }
        
        # Log apenas no console por simplicidade
        _log_global("TRADE", f"{event_type}: {details}", "INFO")

# ===== CONFIGURAÇÃO SIMPLIFICADA =====
class SimpleRatioConfig:
    """Configuração simples para estratégia baseada em avg_buy/sell ratio"""
    
    # Assets permitidos (apenas PUMP e AVNT para testes)
    def __post_init__(self):
        self.ASSETS = ["PUMP/USDT", "AVNT/USDT"]
    
    # Execução
    LEVERAGE: int           = 10          # Leverage moderado
    STOP_LOSS_PCT: float    = 0.20        # Stop loss fixo 20%
    TRADE_SIZE_USD: float   = 3.0         # Valor fixo por trade
    
    # Ratio tracking (para detectar inversões)
    RATIO_HISTORY_SIZE: int = 5           # Últimos N ratios para detectar inversão
    
    # Validação (mantemos alguns filtros básicos de segurança)
    USD_PER_TRADE: float    = 3.0         # Valor fixo por trade
    MIN_ORDER_USD: float    = 10.0
    STOP_LOSS_CAPITAL_PCT: float = 0.20  # 20% da margem como stop inicial (reduzido de 30% para 20%)
    TAKE_PROFIT_CAPITAL_PCT: float = 0.50   # take profit em 50% da margem (aumentado de 30% para 50%)
    MAX_LOSS_ABS_USD: float    = 50.00     # hard stop emergencial - limite absoluto de perda por posição (DESABILITADO TEMP)

    # down & anti-flip-flop
    COOLDOWN_BARS: int      = 0           # cooldown por velas desativado (usar tempo)
    POST_COOLDOWN_CONFIRM: int = 0        # confirmações pós-cooldown desativadas
    COOLDOWN_MINUTOS: int   = 120          # tempo mínimo entre entradas após saída

# ===== ASSET SETUP =====
class AssetSetup:
    def __init__(self):
        pass

# ===== ESTRATÉGIA PRINCIPAL =====
class SimpleRatioStrategy:
    """Estratégia simplificada baseada apenas em avg_buy/sell ratio"""
    
    def __init__(self, dex, symbol: str, cfg: SimpleRatioConfig = SimpleRatioConfig(), logger: "TradeLogger" = None, debug: bool = True, wallet_config: WalletConfig = None):
        self.dex = dex
        self.symbol = symbol
        self.cfg = cfg
        self.logger = logger
        self.debug = debug
        self.wallet_config = wallet_config or WALLET_CONFIGS[0]  # Default para carteira principal

        # Estado simples para nova estratégia de ratio
        self._last_pos_side: Optional[str] = None
        self._position_entry_time: Optional[float] = None
        self._last_ratio_value: Optional[float] = None  # Último valor do ratio avg_buy/sell
        self._ratio_history: List[float] = []           # Histórico de ratios para detectar inversões
        
        # Manter algumas variáveis essenciais para compatibilidade
        self._first_step_done: bool = False

        base = symbol.split("/")[0]
        self._df_symbol_hint = f"{base}USDT"

        # Buffer local (redundância) e flags
        self._local_events = []              # lista de eventos (fallback/espelho)
        self._local_events_count = 0         # contador de eventos locais
        self.force_local_log = False         # True => ignora logger externo
        self.duplicate_local_always = True   # True => sempre duplica no local

        # Estado para cooldown por barras e intenção pós-cooldown
        self._bars_since_last_close = 0     # contador de barras desde último fechamento
        self._pending_intent_after_cd = None # intenção pendente pós-cooldown
        
        # Configuração da carteira/subconta
        self.cfg.__post_init__()  # Inicializar ASSETS

    @property
    def _subconta_dex(self):
        """Retorna a instância do DEX configurada para subconta"""
        return self.wallet_config.get_dex_instance()

    def _log(self, message: str, level: str = "INFO"):
        """Log interno da estratégia"""
        prefix = f"[{self.symbol}]"
        _log_global("STRATEGY", f"{prefix} {message}", level)

    def step(self, df: pd.DataFrame):
        """Função principal da estratégia simplificada"""
        try:
            if len(df) < 30:  # Precisamos de dados suficientes
                return
            
            # 1. Calcular ratio avg_buy/sell atual
            current_ratio = self._calculate_avg_buy_sell_ratio(df)
            if current_ratio is None:
                return
            
            # 2. Atualizar histórico de ratios
            self._update_ratio_history(current_ratio)
            
            # 3. Debug: mostrar ratio atual
            self._log(f"📊 Ratio avg_buy/sell: {current_ratio:.3f}", level="DEBUG")
            
            # 4. Verificar se já temos posição aberta
            pos = self._posicao_aberta()
            
            if pos:
                # Temos posição: verificar condições de saída
                current_pos_side = self._norm_side(pos.get("side"))
                self._check_exit_conditions(pos, current_pos_side, current_ratio, df)
            else:
                # Sem posição: verificar entrada
                # Entrada LONG: ratio cruza de <1.0 para >1.0
                if self._detect_ratio_cross(current_ratio, direction="up"):
                    self._log(f"🔵 SINAL LONG: Ratio cruzou para cima {current_ratio:.3f}", level="INFO")
                    self._enter_position("buy", self.cfg.TRADE_SIZE_USD, df)
                
                # Entrada SHORT: ratio cruza de >1.0 para <1.0 
                elif self._detect_ratio_cross(current_ratio, direction="down"):
                    self._log(f"🔴 SINAL SHORT: Ratio cruzou para baixo {current_ratio:.3f}", level="INFO")
                    self._enter_position("sell", self.cfg.TRADE_SIZE_USD, df)
            
        except Exception as e:
            self._log(f"Erro na função step: {type(e).__name__}: {e}", level="ERROR")
            
    def _calculate_avg_buy_sell_ratio(self, df: pd.DataFrame) -> Optional[float]:
        """Calcula o ratio avg_buy/sell baseado no volume e movimento de preço"""
        try:
            if len(df) < 30:  # Precisamos de pelo menos 30 períodos para média
                return None
                
            last_row = df.iloc[-1]
            current_volume = float(last_row.get('volume', 0))
            current_close = float(last_row.get('valor_fechamento', 0))
            
            if current_volume <= 0 or len(df) < 2:
                return None
                
            # Estimar volume de compra/venda baseado na movimentação do preço
            prev_close = float(df.iloc[-2].get('valor_fechamento', current_close))
            price_change = current_close - prev_close
            
            # Estimativa simples: se preço subiu, mais volume de compra
            if price_change > 0:
                buy_volume_ratio = min(0.7, 0.5 + abs(price_change) / current_close * 10)
            elif price_change < 0:
                buy_volume_ratio = max(0.3, 0.5 - abs(price_change) / current_close * 10)
            else:
                buy_volume_ratio = 0.5
                
            current_buy_volume = current_volume * buy_volume_ratio
            current_sell_volume = current_volume * (1 - buy_volume_ratio)
            
            # Calcular médias dos últimos 30 períodos
            if len(df) >= 30:
                last_30_volume = df["volume"].tail(30)
                avg_total_vol_30 = float(last_30_volume.mean())
                
                # Estimar proporção histórica baseada na tendência
                price_trend = (current_close - float(df["valor_fechamento"].iloc[-30])) / float(df["valor_fechamento"].iloc[-30])
                if price_trend > 0:
                    avg_buy_ratio = 0.55  # Tendência de alta = mais compra
                elif price_trend < 0:
                    avg_buy_ratio = 0.45  # Tendência de baixa = mais venda
                else:
                    avg_buy_ratio = 0.5
                    
                avg_buy_volume_30 = avg_total_vol_30 * avg_buy_ratio
                avg_sell_volume_30 = avg_total_vol_30 * (1 - avg_buy_ratio)
            else:
                return None
                
            # Calcular ratio avg_buy/sell
            if avg_sell_volume_30 > 0:
                ratio = avg_buy_volume_30 / avg_sell_volume_30
                return ratio
            else:
                return None
                
        except Exception as e:
            self._log(f"Erro calculando ratio avg_buy/sell: {e}", level="WARN")
            return None
            
    def _detect_ratio_cross(self, current_ratio: float, direction: str) -> bool:
        """Detecta se houve cruzamento do ratio na direção especificada"""
        if len(self._ratio_history) < 2:
            return False
            
        # Pegar o penúltimo e último valor do histórico
        previous_ratio = self._ratio_history[-2]
        last_ratio = self._ratio_history[-1]
        
        if direction == "up":
            # Cruzamento para cima: anterior <1.0 e atual >1.0
            return previous_ratio < 1.0 and last_ratio > 1.0
        elif direction == "down":
            # Cruzamento para baixo: anterior >1.0 e atual <1.0
            return previous_ratio > 1.0 and last_ratio < 1.0
        else:
            return False
            
    def _check_exit_conditions(self, pos: dict, current_pos_side: str, current_ratio: float, df: pd.DataFrame):
        """Verifica condições de saída: inversão do ratio ou stop loss"""
        try:
            # 1. Verificar saída por inversão do ratio
            should_exit_by_ratio = False
            
            if current_pos_side == "buy":
                # Posição LONG: sair se ratio cruza para baixo (<1.0)
                should_exit_by_ratio = self._detect_ratio_cross(current_ratio, direction="down")
                exit_reason = "RATIO_CROSS_DOWN"
            elif current_pos_side == "sell":
                # Posição SHORT: sair se ratio cruza para cima (>1.0)
                should_exit_by_ratio = self._detect_ratio_cross(current_ratio, direction="up")
                exit_reason = "RATIO_CROSS_UP"
                
            if should_exit_by_ratio:
                self._log(f"🚪 SAÍDA POR RATIO: {exit_reason} - ratio={current_ratio:.3f}", level="INFO")
                self._close_position(df)
                return
                
            # 2. Verificar stop loss de 20%
            entry_price = float(pos.get("entryPrice", 0))
            current_price = float(df.iloc[-1].get('valor_fechamento', 0))
            
            if entry_price > 0 and current_price > 0:
                if current_pos_side == "buy":
                    # LONG: stop se preço caiu 20%
                    loss_pct = (entry_price - current_price) / entry_price
                    if loss_pct >= self.cfg.STOP_LOSS_PCT:
                        self._log(f"🛑 STOP LOSS LONG: {loss_pct*100:.1f}% - saindo", level="WARN")
                        self._close_position(df)
                        return
                elif current_pos_side == "sell":
                    # SHORT: stop se preço subiu 20%
                    loss_pct = (current_price - entry_price) / entry_price
                    if loss_pct >= self.cfg.STOP_LOSS_PCT:
                        self._log(f"🛑 STOP LOSS SHORT: {loss_pct*100:.1f}% - saindo", level="WARN")
                        self._close_position(df)
                        return
                        
            # 3. Verificar fechamento por tempo (4 horas) - mantido do sistema anterior
            if self._position_entry_time is not None:
                import time as _time
                current_time = _time.time()
                time_in_position = current_time - self._position_entry_time
                time_limit_4h = 4 * 60 * 60  # 4 horas em segundos
                
                if time_in_position >= time_limit_4h:
                    self._log(f"⏰ SAÍDA POR TEMPO: {time_in_position/3600:.1f}h - fechando posição", level="WARN")
                    self._close_position(df)
                    return
                    
        except Exception as e:
            self._log(f"Erro verificando condições de saída: {e}", level="ERROR")
            
    def _update_ratio_history(self, current_ratio: float):
        """Atualiza histórico de ratios mantendo apenas os últimos N valores"""
        self._ratio_history.append(current_ratio)
        
        # Manter apenas os últimos valores conforme configuração
        max_history = self.cfg.RATIO_HISTORY_SIZE
        if len(self._ratio_history) > max_history:
            self._ratio_history = self._ratio_history[-max_history:]
            
        self._last_ratio_value = current_ratio
        
    def _enter_position(self, side: str, usd_to_spend: float, df: pd.DataFrame):
        """Abre posição simples na subconta"""
        try:
            import time as _time
            
            # Configurar leverage
            self._subconta_dex.set_leverage(self.cfg.LEVERAGE, self.symbol, {"marginMode": "isolated"})
            
            # Calcular quantidade
            current_price = self._preco_atual()
            if not current_price or current_price <= 0:
                self._log("Preço inválido para entrada", level="ERROR")
                return
                
            amount = usd_to_spend * self.cfg.LEVERAGE / current_price
            
            # Criar ordem market
            order = self._subconta_dex.create_order(self.symbol, "market", side, amount, current_price)
            
            # Registrar tempo de entrada
            self._position_entry_time = _time.time()
            self._last_pos_side = self._norm_side(side)
            
            # Criar stop loss simples
            sl_price = self._calculate_stop_price(current_price, side)
            if sl_price:
                sl_side = "sell" if side == "buy" else "buy"
                try:
                    self._subconta_dex.create_order(self.symbol, "stop_market", sl_side, amount, sl_price, {"reduceOnly": True})
                    self._log(f"🛡️ Stop loss criado: {sl_side} @ {sl_price:.6f}", level="INFO")
                except Exception as e:
                    self._log(f"Erro criando stop loss: {e}", level="WARN")
            
            # Notificar
            self._notify_trade("open", side, current_price, amount, f"Entrada por ratio", include_hl=False)
            self._log(f"✅ POSIÇÃO ABERTA: {side.upper()} {amount:.2f} @ {current_price:.6f}", level="INFO")
            
        except Exception as e:
            self._log(f"Erro abrindo posição: {e}", level="ERROR")
            
    def _close_position(self, df: pd.DataFrame):
        """Fecha posição atual"""
        try:
            import time as _time
            
            pos = self._posicao_aberta()
            if not pos:
                return
                
            side = self._norm_side(pos.get("side"))
            amount = abs(float(pos.get("contracts", 0)))
            
            if amount <= 0:
                return
                
            # Determinar lado de fechamento
            close_side = "sell" if side == "buy" else "buy"
            current_price = self._preco_atual()
            
            # Fechar posição
            self._subconta_dex.create_order(self.symbol, "market", close_side, amount, current_price, {"reduceOnly": True})
            
            # Limpar estado
            self._position_entry_time = None
            self._last_pos_side = None
            
            # Notificar
            self._notify_trade("close", side, current_price, amount, "Fechamento", include_hl=False)
            self._log(f"🚪 POSIÇÃO FECHADA: {side.upper()} {amount:.2f} @ {current_price:.6f}", level="INFO")
            
        except Exception as e:
            self._log(f"Erro fechando posição: {e}", level="ERROR")
            
    def _calculate_stop_price(self, entry_price: float, side: str) -> Optional[float]:
        """Calcula preço do stop loss baseado em 20% de perda"""
        try:
            if side == "buy":
                # LONG: stop 20% abaixo
                return entry_price * (1 - self.cfg.STOP_LOSS_PCT)
            else:
                # SHORT: stop 20% acima
                return entry_price * (1 + self.cfg.STOP_LOSS_PCT)
        except:
            return None

    def _norm_side(self, raw: Optional[str]) -> Optional[str]:
        """Normaliza lado da posição"""
        if raw is None:
            return None
        raw_clean = str(raw).lower().strip()
        if raw_clean in ["buy", "long"]:
            return "buy"
        elif raw_clean in ["sell", "short"]:
            return "sell"
        else:
            return raw_clean

    def _notify_trade(self, kind: str, side: Optional[str], price: Optional[float], amount: Optional[float], note: str = "", include_hl: bool = False):
        """Notifica trade (simplificado)"""
        side_str = side or "?"
        price_str = f"{price:.6f}" if price else "?"
        amount_str = f"{amount:.2f}" if amount else "?"
        
        msg = f"🔔 {kind.upper()}: {side_str} {amount_str} @ {price_str}"
        if note:
            msg += f" ({note})"
            
        self._log(msg, level="INFO")

    def _preco_atual(self) -> float:
        """Obtém preço atual do ativo"""
        try:
            cache_key = f"ticker_{self.symbol}"
            t = _get_cached_api_call(cache_key, self._subconta_dex.fetch_ticker, self.symbol)
            return float(t.get('last', 0))
        except Exception as e:
            self._log(f"Erro obtendo preço atual: {e}", level="WARN")
            return 0.0

    def _posicao_aberta(self, force_fresh: bool = False) -> Optional[Dict[str, Any]]:
        """Verifica se há posição aberta"""
        try:
            cache_key = f"positions_{self.symbol}"
            
            if force_fresh:
                pos = self._subconta_dex.fetch_positions([self.symbol])  # Opera na subconta
            else:
                pos = _get_cached_api_call(cache_key, self._subconta_dex.fetch_positions, [self.symbol])  # Opera na subconta
            
            if pos and len(pos) > 0:
                return pos[0]  # Retorna primeira posição
            return None
        except Exception as e:
            self._log(f"Erro verificando posição: {e}", level="WARN")
            return None

# ===== FUNÇÃO PRINCIPAL DE BUILD DE DADOS =====
def build_df(symbol: str = "BTCUSDT", tf: str = "15m", limit: int = 260, source: str = "auto") -> pd.DataFrame:
    """Constrói DataFrame com dados de mercado (simplificado)"""
    try:
        _log_global("DATA", f"Iniciando build_df symbol={symbol} tf={tf} alvo={limit}", "INFO")
        
        # Usar Binance via ccxt como fonte principal
        exchange = ccxt.binance({
            'sandbox': False,
            'options': {'defaultType': 'spot'}
        })
        
        # Buscar dados
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        
        if not ohlcv:
            raise MarketDataUnavailable(f"Nenhum dado disponível para {symbol}")
        
        # Converter para DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Renomear colunas para compatibilidade
        df = df.rename(columns={
            'open': 'valor_abertura',
            'high': 'valor_alta', 
            'low': 'valor_baixa',
            'close': 'valor_fechamento',
            'volume': 'volume'
        })
        
        _log_global("DATA", f"Total candles retornados: {len(df)}", "INFO")
        return df
        
    except Exception as e:
        _log_global("DATA", f"Erro no build_df: {e}", "ERROR")
        raise MarketDataUnavailable(f"Erro obtendo dados para {symbol}: {e}")

# ===== FUNÇÃO PRINCIPAL DO SISTEMA =====
def main():
    """Função principal do sistema de trading simplificado"""
    print("🚀 SISTEMA DE TRADING SIMPLIFICADO - SimpleRatioStrategy")
    print("📊 Assets: PUMP/USDT, AVNT/USDT")
    print("💰 Trade size: $3 USD, Leverage: 10x, Stop: 20%")
    print("⚡ Estratégia: Entradas/saídas por inversão de ratio avg_buy/sell")
    print()
    
    # Configuração
    cfg = SimpleRatioConfig()
    cfg.__post_init__()
    
    # Configuração da subconta
    wallet_config = WALLET_CONFIGS[1]  # Subconta
    dex = wallet_config.get_dex_instance()
    
    # Loop principal para cada asset
    for symbol in cfg.ASSETS:
        try:
            print(f"🔍 Processando {symbol}...")
            
            # Buscar dados
            df = build_df(symbol, "15m", 100)
            
            # Criar estratégia
            strategy = SimpleRatioStrategy(dex, symbol, cfg, wallet_config=wallet_config)
            
            # Executar step
            strategy.step(df)
            
        except Exception as e:
            _log_global("MAIN", f"Erro processando {symbol}: {e}", "ERROR")
    
    print("✅ Ciclo completo!")

if __name__ == "__main__":
    main()
