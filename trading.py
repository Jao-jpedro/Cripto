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
import requests

# ===== LOGGING GLOBAL =====
def _log_global(channel: str, message: str, level: str = "INFO"):
    """Sistema de log global simplificado"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] [{channel}] {message}"
    print(log_line, flush=True)
    
    # Força flush do sistema para garantir saída imediata
    sys.stdout.flush()
    sys.stderr.flush()

# ===== NOTIFICAÇÕES DISCORD =====
class DiscordNotifier:
    """Sistema simplificado de notificações Discord"""
    
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.enabled = bool(self.webhook_url)
        self.last_notification_time = 0
        self.cooldown_seconds = 30  # Cooldown entre notificações
    
    def send_notification(self, title: str, message: str, color: int = 0x00ff00):
        """Envia notificação para Discord"""
        if not self.enabled:
            return False
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_notification_time < self.cooldown_seconds:
            return False
        
        try:
            embed = {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "SimpleRatio Trading Bot"
                }
            }
            
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 204:
                self.last_notification_time = current_time
                _log_global("DISCORD", f"Notificação enviada: {title}", "INFO")
                return True
            else:
                _log_global("DISCORD", f"Erro enviando notificação: {response.status_code}", "WARN")
                return False
                
        except Exception as e:
            _log_global("DISCORD", f"Erro no Discord: {e}", "ERROR")
            return False
    
    def notify_trade_open(self, symbol: str, side: str, price: float, amount: float, reason: str = ""):
        """Notifica abertura de trade"""
        side_emoji = "🟢" if side.lower() == "buy" else "🔴"
        title = f"{side_emoji} POSIÇÃO ABERTA"
        
        message = f"""
**Símbolo:** {symbol}
**Direção:** {side.upper()}
**Preço:** ${price:.6f}
**Quantidade:** {amount:.2f}
**Motivo:** {reason}
        """.strip()
        
        color = 0x00ff00 if side.lower() == "buy" else 0xff0000
        return self.send_notification(title, message, color)
    
    def notify_trade_close(self, symbol: str, side: str, price: float, amount: float, pnl_pct: float = None, reason: str = ""):
        """Notifica fechamento de trade"""
        if pnl_pct is not None:
            if pnl_pct > 0:
                title = "💰 POSIÇÃO FECHADA - LUCRO"
                color = 0x00ff00
                pnl_text = f"+{pnl_pct:.2f}%"
            else:
                title = "📉 POSIÇÃO FECHADA - PREJUÍZO"
                color = 0xff0000
                pnl_text = f"{pnl_pct:.2f}%"
        else:
            title = "🚪 POSIÇÃO FECHADA"
            color = 0xffff00
            pnl_text = "N/A"
        
        message = f"""
**Símbolo:** {symbol}
**Direção:** {side.upper()}
**Preço:** ${price:.6f}
**Quantidade:** {amount:.2f}
**P&L:** {pnl_text}
**Motivo:** {reason}
        """.strip()
        
        return self.send_notification(title, message, color)
    
    def notify_error(self, error_msg: str, symbol: str = ""):
        """Notifica erro crítico"""
        title = "⚠️ ERRO NO SISTEMA"
        
        message = f"""
**Símbolo:** {symbol if symbol else "Sistema"}
**Erro:** {error_msg}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return self.send_notification(title, message, 0xff0000)

# Instância global do notificador
discord_notifier = DiscordNotifier()

# ===== CALCULADORA DE INDICADORES TÉCNICOS =====
class TechnicalIndicators:
    """Calcula indicadores técnicos para monitoramento"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Calcula EMA (Exponential Moving Average)"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula ATR (Average True Range)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return ranges.rolling(window=period).mean()
    
    @staticmethod
    def volume_ma(volume: pd.Series, period: int = 30) -> pd.Series:
        """Calcula média móvel do volume"""
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def gradient_percentage(data: pd.Series, periods: int = 1) -> float:
        """Calcula gradiente percentual"""
        if len(data) < periods + 1:
            return 0.0
        current = data.iloc[-1]
        past = data.iloc[-(periods + 1)]
        if past == 0:
            return 0.0
        return ((current - past) / past) * 100
    
    @staticmethod
    def k_atr_ratio(close_price: float, atr_value: float) -> float:
        """Calcula ratio K-ATR"""
        if atr_value == 0:
            return 0.0
        return close_price / atr_value
    
    @staticmethod
    def estimate_buy_sell_volumes(df: pd.DataFrame) -> Dict[str, float]:
        """Estima volumes de compra e venda baseado no movimento de preços"""
        if len(df) < 2:
            return {"buy_vol": 0, "sell_vol": 0, "total_vol": 0}
        
        current_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        current_close = float(current_row.get('valor_fechamento', 0))
        prev_close = float(prev_row.get('valor_fechamento', current_close))
        current_volume = float(current_row.get('volume', 0))
        
        # Estimar proporção de compra/venda baseado no movimento de preço
        price_change = current_close - prev_close
        
        if price_change > 0:
            # Preço subiu - mais volume de compra
            buy_ratio = min(0.8, 0.5 + abs(price_change) / current_close * 20)
        elif price_change < 0:
            # Preço caiu - mais volume de venda
            buy_ratio = max(0.2, 0.5 - abs(price_change) / current_close * 20)
        else:
            # Sem mudança de preço
            buy_ratio = 0.5
        
        buy_vol = current_volume * buy_ratio
        sell_vol = current_volume * (1 - buy_ratio)
        
        return {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "total_vol": current_volume,
            "buy_ratio": buy_ratio
        }

# ===== MONITOR DE INDICADORES =====
class TradingMonitor:
    """Sistema de monitoramento de indicadores técnicos"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.last_snapshot_time = 0
        self.snapshot_interval = 30  # Snapshot a cada 30 segundos
    
    def should_take_snapshot(self) -> bool:
        """Verifica se deve tirar um snapshot dos indicadores"""
        current_time = time.time()
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            self.last_snapshot_time = current_time
            return True
        return False
    
    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calcula todos os indicadores técnicos"""
        if len(df) < 30:
            return {}
        
        try:
            # Preparar dados
            close = df['valor_fechamento'].astype(float)
            volume = df['volume'].astype(float)
            
            # Verificar se temos colunas de high/low
            if 'valor_alta' in df.columns and 'valor_baixa' in df.columns:
                high = df['valor_alta'].astype(float)
                low = df['valor_baixa'].astype(float)
            else:
                # Usar close como approximação
                high = close
                low = close
            
            current_close = close.iloc[-1]
            current_volume = volume.iloc[-1]
            
            # Calcular EMAs
            ema7 = self.indicators.ema(close, 7)
            ema21 = self.indicators.ema(close, 21)
            
            # Calcular ATR
            atr = self.indicators.atr(high, low, close, 14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0
            atr_pct = (current_atr / current_close * 100) if current_close > 0 else 0
            
            # Calcular Volume MA
            vol_ma = self.indicators.volume_ma(volume, 30)
            current_vol_ma = vol_ma.iloc[-1] if len(vol_ma) > 0 else 0
            
            # Calcular gradiente EMA7
            grad_ema7 = self.indicators.gradient_percentage(ema7, 1)
            
            # Calcular K-ATR
            k_atr = self.indicators.k_atr_ratio(current_close, current_atr)
            
            # Calcular médias de volume (30 candles)
            avg_30c = vol_ma.iloc[-1] if len(vol_ma) > 0 else current_volume
            vol_ratio = current_volume / avg_30c if avg_30c > 0 else 0
            
            # Estimar volumes de compra/venda
            buysell_data = self.indicators.estimate_buy_sell_volumes(df)
            
            # Calcular médias históricas de compra/venda (últimos 30 períodos)
            if len(df) >= 30:
                buy_volumes = []
                sell_volumes = []
                
                for i in range(-30, 0):
                    if abs(i) <= len(df):
                        temp_df = df.iloc[max(0, len(df) + i - 1):len(df) + i + 1]
                        if len(temp_df) >= 2:
                            temp_buysell = self.indicators.estimate_buy_sell_volumes(temp_df)
                            buy_volumes.append(temp_buysell["buy_vol"])
                            sell_volumes.append(temp_buysell["sell_vol"])
                
                buy_avg30 = np.mean(buy_volumes) if buy_volumes else buysell_data["buy_vol"]
                sell_avg30 = np.mean(sell_volumes) if sell_volumes else buysell_data["sell_vol"]
            else:
                buy_avg30 = buysell_data["buy_vol"]
                sell_avg30 = buysell_data["sell_vol"]
            
            # Calcular ratios
            buy_ratio = buysell_data["buy_vol"] / buy_avg30 if buy_avg30 > 0 else 0
            sell_ratio = buysell_data["sell_vol"] / sell_avg30 if sell_avg30 > 0 else 0
            buy_sell_ratio = buysell_data["buy_vol"] / buysell_data["sell_vol"] if buysell_data["sell_vol"] > 0 else 0
            avg_buy_sell_ratio = buy_avg30 / sell_avg30 if sell_avg30 > 0 else 0
            
            return {
                "symbol": symbol,
                "close": current_close,
                "ema7": ema7.iloc[-1],
                "ema21": ema21.iloc[-1],
                "atr": current_atr,
                "atr_pct": atr_pct,
                "volume": current_volume,
                "vol_ma": current_vol_ma,
                "grad_ema7": grad_ema7,
                "k_atr": k_atr,
                "trades_now": current_volume,
                "avg_30c": avg_30c,
                "vol_ratio": vol_ratio,
                "buy_vol": buysell_data["buy_vol"],
                "buy_avg30": buy_avg30,
                "buy_ratio": buy_ratio,
                "sell_vol": buysell_data["sell_vol"],
                "sell_avg30": sell_avg30,
                "sell_ratio": sell_ratio,
                "buy_sell_ratio": buy_sell_ratio,
                "avg_buy_sell_ratio": avg_buy_sell_ratio
            }
            
        except Exception as e:
            _log_global("MONITOR", f"Erro calculando indicadores: {e}", "ERROR")
            return {}
    
    def print_snapshot(self, indicators: Dict[str, Any]):
        """Imprime snapshot dos indicadores no formato solicitado"""
        if not indicators:
            return
        
        # Quebrar em linhas menores para evitar truncamento no terminal
        symbol = indicators['symbol']
        
        # Linha 1: Dados básicos de preço e indicadores
        line1 = (
            f"[DEBUG] [{symbol}] Trigger snapshot | "
            f"close={indicators['close']:.6f} "
            f"ema7={indicators['ema7']:.6f} "
            f"ema21={indicators['ema21']:.6f} "
            f"atr={indicators['atr']:.6f} "
            f"atr%={indicators['atr_pct']:.3f} "
            f"vol={indicators['volume']:.2f} "
            f"vol_ma={indicators['vol_ma']:.2f} "
            f"grad%_ema7={indicators['grad_ema7']:.4f}"
        )
        
        # Linha 2: Dados de volume e ratios
        line2 = (
            f"[DEBUG] [{symbol}] Volume data | "
            f"current_k_atr={indicators['k_atr']:.3f} | "
            f"trades_now={indicators['trades_now']:.0f} "
            f"avg_30c={indicators['avg_30c']:.0f} "
            f"ratio={indicators['vol_ratio']:.2f}x"
        )
        
        # Linha 3: Detalhes de compra e venda
        line3 = (
            f"[DEBUG] [{symbol}] Buy/Sell | "
            f"buy_vol={indicators['buy_vol']:.0f} "
            f"buy_avg30={indicators['buy_avg30']:.0f} "
            f"buy_ratio={indicators['buy_ratio']:.2f}x | "
            f"sell_vol={indicators['sell_vol']:.0f} "
            f"sell_avg30={indicators['sell_avg30']:.0f} "
            f"sell_ratio={indicators['sell_ratio']:.2f}x | "
            f"buy/sell={indicators['buy_sell_ratio']:.2f} "
            f"avg_buy/sell={indicators['avg_buy_sell_ratio']:.2f}"
        )
        
        # Imprimir as três linhas com flush forçado
        print(line1, flush=True)
        sys.stdout.flush()
        print(line2, flush=True)
        sys.stdout.flush()
        print(line3, flush=True)
        sys.stdout.flush()

# Instância global do monitor
trading_monitor = TradingMonitor()

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

# Configurações de carteiras usando variáveis de ambiente
def get_wallet_config():
    """Obtém configuração da carteira a partir das variáveis de ambiente"""
    wallet_address = os.getenv("WALLET_ADDRESS")
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY") 
    subaccount = os.getenv("HYPERLIQUID_SUBACCOUNT")
    
    # Se há subaccount especificada, usar como subconta
    if subaccount:
        return WalletConfig(
            name="Subconta Trading (ENV)",
            is_subconta=True,
            vault_address=subaccount
        )
    else:
        # Usar carteira principal
        return WalletConfig(
            name="Carteira Principal (ENV)",
            is_subconta=False,
            vault_address=None
        )

# Configurações de carteiras disponíveis (fallback)
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
        """Configura conexão com Hyperliquid usando variáveis de ambiente"""
        try:
            # Obter variáveis de ambiente
            wallet_address = os.getenv("WALLET_ADDRESS")
            private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
            subaccount = os.getenv("HYPERLIQUID_SUBACCOUNT")
            
            # Configuração básica do ccxt para Hyperliquid
            config = {
                'sandbox': False,
                'options': {
                    'defaultType': 'swap',
                }
            }
            
            # Adicionar credenciais se disponíveis
            if wallet_address and private_key:
                config['apiKey'] = wallet_address
                config['secret'] = private_key
                _log_global("DEX", f"🔐 Credenciais configuradas: {wallet_address[:10]}...", "INFO")
            
            self.exchange = ccxt.hyperliquid(config)
            
            # Configurar subconta se especificada
            if self.vault_address:
                self.exchange.options['vault'] = self.vault_address
                _log_global("DEX", f"🏦 Vault configurado: {self.vault_address}", "INFO")
            
            # Configurar subaccount se especificada
            if subaccount:
                self.exchange.options['subAccount'] = subaccount
                _log_global("DEX", f"📋 Subaccount configurado: {subaccount}", "INFO")
                
        except Exception as e:
            _log_global("DEX", f"Erro configurando Hyperliquid: {e}", "ERROR")
    
    def fetch_ticker(self, symbol: str):
        """Busca ticker do símbolo"""
        return self.exchange.fetch_ticker(symbol)
    
    def fetch_positions(self, symbols: List[str] = None):
        """Busca posições abertas"""
        # Para Hyperliquid, usar parâmetros específicos se disponível
        wallet_address = os.getenv("WALLET_ADDRESS")
        if wallet_address:
            params = {'user': wallet_address}
            return self.exchange.fetch_positions(symbols, params)
        else:
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
    
    # Assets permitidos (símbolos corretos da Binance para dados históricos)
    ASSETS: List[str] = ["PUMPUSDT", "AVNTUSDT"]
    
    # Mapeamento de símbolos: Binance (dados) -> Hyperliquid (trading)
    SYMBOL_MAPPING = {
        "PUMPUSDT": "PUMP/USDC:USDC",  # Binance -> Hyperliquid
        "AVNTUSDT": "AVNT/USDC:USDC"   # Binance -> Hyperliquid
    }
    
    @classmethod
    def get_trading_symbol(cls, data_symbol: str) -> str:
        """Converte símbolo de dados para símbolo de trading"""
        return cls.SYMBOL_MAPPING.get(data_symbol, data_symbol)
    
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
        self.symbol = symbol  # Símbolo para dados (Binance)
        self.trading_symbol = cfg.get_trading_symbol(symbol)  # Símbolo para trading (Hyperliquid)
        self.cfg = cfg
        self.logger = logger
        self.debug = debug
        self.wallet_config = wallet_config or WALLET_CONFIGS[0]  # Default para carteira principal

        # Estado simples para nova estratégia de ratio
        self._last_pos_side: Optional[str] = None
        self._position_entry_time: Optional[float] = None
        self._entry_price: Optional[float] = None       # Preço de entrada para cálculo de P&L
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
            
            # 1. SEMPRE calcular e mostrar snapshot de indicadores técnicos para cada ativo
            indicators = trading_monitor.calculate_indicators(df, self.symbol)
            if indicators:
                trading_monitor.print_snapshot(indicators)
            
            # 2. Usar o ratio avg_buy/sell calculado pelo TechnicalIndicators
            if not indicators or 'avg_buy_sell_ratio' not in indicators:
                return
                
            current_ratio = indicators['avg_buy_sell_ratio']
            if current_ratio is None or current_ratio <= 0:
                return
            
            # 3. Debug: mostrar ratio atual
            self._log(f"📊 Ratio avg_buy/sell: {current_ratio:.3f}", level="DEBUG")
            
            # 4. Verificar se já temos posição aberta
            pos = self._posicao_aberta()
            
            if pos:
                # Temos posição: verificar condições de saída
                current_pos_side = self._norm_side(pos.get("side"))
                self._check_exit_conditions(pos, current_pos_side, current_ratio, df)
            else:
                # Sem posição: verificar entrada ANTES de atualizar histórico
                # Entrada LONG: ratio cruza de <1.0 para >1.0
                if self._detect_ratio_cross(current_ratio, direction="up"):
                    self._log(f"🔵 SINAL LONG: Ratio cruzou para cima {current_ratio:.3f}", level="INFO")
                    self._enter_position("buy", self.cfg.TRADE_SIZE_USD, df)
                
                # Entrada SHORT: ratio cruza de >1.0 para <1.0 
                elif self._detect_ratio_cross(current_ratio, direction="down"):
                    self._log(f"🔴 SINAL SHORT: Ratio cruzou para baixo {current_ratio:.3f}", level="INFO")
                    self._enter_position("sell", self.cfg.TRADE_SIZE_USD, df)
            
            # 5. APÓS verificar sinais, atualizar histórico para próximo ciclo
            self._update_ratio_history(current_ratio)
            
            # Debug: mostrar histórico atualizado
            if len(self._ratio_history) >= 1:
                recent_ratios = self._ratio_history[-5:] if len(self._ratio_history) >= 5 else self._ratio_history
                history_str = " → ".join([f'{r:.3f}' for r in recent_ratios])
                self._log(f"📈 Histórico ratios (últimos {len(recent_ratios)}): {history_str}", level="DEBUG")
                
                # Se temos pelo menos 2 valores, mostrar a comparação
                if len(self._ratio_history) >= 2:
                    prev = self._ratio_history[-2]
                    curr = current_ratio
                    change = "↗️" if curr > prev else "↘️" if curr < prev else "➡️"
                    self._log(f"🔄 Mudança: {prev:.3f} → {curr:.3f} {change}", level="DEBUG")
            
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
        if len(self._ratio_history) < 1:
            return False
            
        # Usar o valor atual passado como parâmetro e o anterior do histórico
        previous_ratio = self._ratio_history[-1]  # O último valor do histórico é o anterior
        
        # Debug detalhado
        self._log(f"🔍 Debug Cross: previous={previous_ratio:.3f}, current={current_ratio:.3f}, direction={direction}", level="DEBUG")
        
        if direction == "up":
            # Cruzamento para cima: anterior <1.0 e atual >=1.0 (incluindo igualdade)
            cross_detected = previous_ratio < 1.0 and current_ratio >= 1.0
            if cross_detected:
                self._log(f"✅ CROSS UP detectado: {previous_ratio:.3f} → {current_ratio:.3f}", level="INFO")
            return cross_detected
        elif direction == "down":
            # Cruzamento para baixo: anterior >=1.0 e atual <1.0 (incluindo igualdade na origem)
            cross_detected = previous_ratio >= 1.0 and current_ratio < 1.0
            if cross_detected:
                self._log(f"✅ CROSS DOWN detectado: {previous_ratio:.3f} → {current_ratio:.3f}", level="INFO")
            return cross_detected
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
            order = self._subconta_dex.create_order(self.trading_symbol, "market", side, amount, current_price)
            
            # Registrar tempo de entrada
            self._position_entry_time = _time.time()
            self._last_pos_side = self._norm_side(side)
            self._entry_price = current_price  # Rastrear preço de entrada para P&L
            
            # Criar stop loss simples
            sl_price = self._calculate_stop_price(current_price, side)
            if sl_price:
                sl_side = "sell" if side == "buy" else "buy"
                try:
                    self._subconta_dex.create_order(self.trading_symbol, "stop_market", sl_side, amount, sl_price, {"reduceOnly": True})
                    self._log(f"🛡️ Stop loss criado: {sl_side} @ {sl_price:.6f}", level="INFO")
                except Exception as e:
                    self._log(f"Erro criando stop loss: {e}", level="WARN")
            
            # Notificar
            self._notify_trade("open", side, current_price, amount, f"Entrada por ratio", include_hl=False)
            self._log(f"✅ POSIÇÃO ABERTA: {side.upper()} {amount:.2f} @ {current_price:.6f}", level="INFO")
            
        except Exception as e:
            # Verificar se é erro de credenciais ou mercado não disponível
            error_msg = str(e).lower()
            if any(x in error_msg for x in ['user parameter', 'wallet address', 'authentication', 'credential']):
                # Modo demo - log discreto
                self._log(f"💤 Sinal detectado mas sem credenciais para operar", level="DEBUG")
            elif 'does not have market symbol' in error_msg:
                # Mercado não disponível - log discreto
                self._log(f"💤 Sinal detectado mas mercado {self.trading_symbol} não disponível", level="DEBUG")
            else:
                # Erros reais de execução - manter ERROR
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
            self._subconta_dex.create_order(self.trading_symbol, "market", close_side, amount, current_price, {"reduceOnly": True})
            
            # Limpar estado
            self._position_entry_time = None
            self._last_pos_side = None
            self._entry_price = None  # Limpar preço de entrada
            
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
        """Notifica trade com Discord e log local"""
        side_str = side or "?"
        price_str = f"{price:.6f}" if price else "?"
        amount_str = f"{amount:.2f}" if amount else "?"
        
        msg = f"🔔 {kind.upper()}: {side_str} {amount_str} @ {price_str}"
        if note:
            msg += f" ({note})"
            
        # Log local
        self._log(msg, level="INFO")
        
        # Notificação Discord
        if price and amount and side:
            try:
                if kind.lower() == "open":
                    discord_notifier.notify_trade_open(
                        symbol=self.symbol,
                        side=side,
                        price=price,
                        amount=amount,
                        reason=note
                    )
                elif kind.lower() == "close":
                    # Tentar calcular P&L se possível
                    pnl_pct = None
                    if hasattr(self, '_entry_price') and self._entry_price:
                        if side.lower() == "buy":
                            pnl_pct = (price - self._entry_price) / self._entry_price * 100
                        else:
                            pnl_pct = (self._entry_price - price) / self._entry_price * 100
                    
                    discord_notifier.notify_trade_close(
                        symbol=self.symbol,
                        side=side,
                        price=price,
                        amount=amount,
                        pnl_pct=pnl_pct,
                        reason=note
                    )
            except Exception as e:
                self._log(f"Erro enviando notificação Discord: {e}", level="WARN")

    def _preco_atual(self) -> float:
        """Obtém preço atual do ativo"""
        try:
            cache_key = f"ticker_{self.trading_symbol}"
            t = _get_cached_api_call(cache_key, self._subconta_dex.fetch_ticker, self.trading_symbol)
            return float(t.get('last', 0))
        except Exception as e:
            self._log(f"Erro obtendo preço atual: {e}", level="WARN")
            return 0.0

    def _posicao_aberta(self, force_fresh: bool = False) -> Optional[Dict[str, Any]]:
        """Verifica se há posição aberta"""
        try:
            cache_key = f"positions_{self.trading_symbol}"
            
            if force_fresh:
                pos = self._subconta_dex.fetch_positions([self.trading_symbol])  # Opera na subconta
            else:
                pos = _get_cached_api_call(cache_key, self._subconta_dex.fetch_positions, [self.trading_symbol])  # Opera na subconta
            
            if pos and len(pos) > 0:
                return pos[0]  # Retorna primeira posição
            return None
        except Exception as e:
            # Verificar se é erro de autenticação, credenciais ou mercado não disponível
            error_msg = str(e).lower()
            if any(x in error_msg for x in ['user parameter', 'wallet address', 'authentication', 'credential']):
                # Erro de credenciais - log discreto DEBUG
                self._log(f"💤 Sem credenciais para verificar posições", level="DEBUG")
            elif 'does not have market symbol' in error_msg:
                # Mercado não disponível - log discreto DEBUG  
                self._log(f"💤 Mercado {self.trading_symbol} não listado na exchange", level="DEBUG")
            else:
                # Outros erros realmente problemáticos - manter WARN
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
    print("📊 Assets: PUMPUSDT, AVNTUSDT (dados) → PUMP/USDC:USDC, AVNT/USDC:USDC (trading)")
    print("💰 Trade size: $3 USD, Leverage: 10x, Stop: 20%")
    print("⚡ Estratégia: Entradas/saídas por inversão de ratio avg_buy/sell")
    print("🔄 Execução contínua a cada 30 segundos")
    
    # Verificar variáveis de ambiente
    wallet_address = os.getenv("WALLET_ADDRESS")
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    subaccount = os.getenv("HYPERLIQUID_SUBACCOUNT")
    
    print("\n🔧 CONFIGURAÇÃO DE AMBIENTE:")
    print(f"   📋 Wallet Address: {'✅ Configurado' if wallet_address else '❌ Não configurado'}")
    print(f"   🔐 Private Key: {'✅ Configurado' if private_key else '❌ Não configurado'}")
    print(f"   🏦 Subaccount: {'✅ ' + subaccount if subaccount else '❌ Não configurado'}")
    
    if not wallet_address or not private_key:
        print("\n⚠️  MODO DEMONSTRAÇÃO: Algumas credenciais não estão configuradas")
        print("   Para operação completa, configure as variáveis de ambiente:")
        print("   - WALLET_ADDRESS")
        print("   - HYPERLIQUID_PRIVATE_KEY")
        print("   - HYPERLIQUID_SUBACCOUNT (opcional)")
    print()
    
    # Configuração
    cfg = SimpleRatioConfig()
    
    # Usar configuração baseada em variáveis de ambiente
    wallet_config = get_wallet_config()
    dex = wallet_config.get_dex_instance()
    
    print(f"🏦 Usando carteira: {wallet_config.name}")
    print()
    
    cycle_count = 0
    
    # Cache de estratégias para manter histórico entre ciclos
    strategies_cache = {}
    
    # Loop contínuo
    while True:
        try:
            cycle_count += 1
            print(f"\n🔄 CICLO #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Loop principal para cada asset
            for symbol in cfg.ASSETS:
                try:
                    print(f"🔍 Processando {symbol}...")
                    
                    # Buscar dados com timeout de debug
                    start_time = time.time()
                    print(f"    📊 Buscando dados de {symbol}...")
                    df = build_df(symbol, "15m", 100)
                    data_time = time.time() - start_time
                    print(f"    ✅ Dados obtidos em {data_time:.2f}s ({len(df)} candles)")
                    
                    # Usar estratégia cached ou criar nova
                    if symbol not in strategies_cache:
                        print(f"    🎯 Criando estratégia para {symbol}...")
                        strategy = SimpleRatioStrategy(dex, symbol, cfg, wallet_config=wallet_config)
                        strategies_cache[symbol] = strategy
                        print(f"    ✅ Estratégia criada e armazenada no cache")
                    else:
                        strategy = strategies_cache[symbol]
                        print(f"    🔄 Usando estratégia existente (histórico preservado)")
                    
                    # Executar step
                    print(f"    🚀 Executando step...")
                    step_start = time.time()
                    strategy.step(df)
                    step_time = time.time() - step_start
                    print(f"    ✅ Step executado em {step_time:.2f}s")
                    
                except Exception as e:
                    _log_global("MAIN", f"Erro processando {symbol}: {e}", "ERROR")
                    print(f"    ❌ Erro: {e}")
            
            print("✅ Ciclo completo!")
            print(f"⏰ Aguardando 30 segundos para próximo ciclo...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Trading interrompido pelo usuário!")
            break
        except Exception as e:
            _log_global("MAIN", f"Erro no loop principal: {e}", "ERROR")
            print(f"❌ Erro no ciclo, aguardando 30s antes de tentar novamente...")
            time.sleep(30)

if __name__ == "__main__":
    main()
