import os
import pandas as pd
import numpy as np
import time
import requests
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import hmac
import hashlib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv não instalado. Usando variáveis de ambiente do sistema.")

# =============================================================================
# CONFIGURAÇÕES DE AMBIENTE - VARIÁVEIS DO RENDER
# =============================================================================

# Variáveis obrigatórias para Hyperliquid
HYPERLIQUID_MAIN_WALLET = os.getenv('HYPERLIQUID_MAIN_WALLET', '')
HYPERLIQUID_SUBACCOUNT = os.getenv('HYPERLIQUID_SUBACCOUNT', '')
HYPERLIQUID_PRIVATE_KEY = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')
LIVE_TRADING = os.getenv('LIVE_TRADING', '0') == '1'  # 0 = Demo, 1 = Live

# Variáveis opcionais
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
USD_PER_TRADE = float(os.getenv('USD_PER_TRADE', '1.0'))

# =============================================================================
# CONFIGURAÇÕES E PARÂMETROS
# =============================================================================

@dataclass
class TradingConfig:
    # Parâmetros EMA
    EMA_SHORT_SPAN: int = 7
    EMA_LONG_SPAN: int = 21
    N_BARRAS_GRADIENTE: int = 3
    GRAD_CONSISTENCY: int = 3
    
    # Parâmetros ATR e Volatilidade
    ATR_PERIOD: int = 14
    VOL_MA_PERIOD: int = 20
    ATR_PCT_MIN: float = 0.7
    ATR_PCT_MAX: float = 5.0
    BREAKOUT_K_ATR: float = 3.0
    NO_TRADE_EPS_K_ATR: float = 0.07
    
    # Controles de Tempo
    COOLDOWN_MINUTOS: int = 120
    ANTI_SPAM_SECS: int = 3
    MIN_HOLD_BARS: int = 1
    
    # Gestão de Risco
    STOP_LOSS_PCT: float = 4.0  # 4% do preço com alavancagem
    USD_PER_TRADE: float = USD_PER_TRADE
    
    # Configurações API
    BINANCE_API_URL: str = "https://api.binance.com/api/v3"
    BYBIT_API_URL: str = "https://api.bybit.com/v2"
    HYPERLIQUID_API_URL: str = "https://api.hyperliquid.xyz"

@dataclass
class AssetSetup:
    name: str
    data_symbol: str  # Para buscar dados (Binance format)
    trading_symbol: str  # Para executar trades (Exchange format)
    leverage: int
    min_notional: float = 1.0  # Valor mínimo por trade na Hyperliquid
    usd_per_trade: float = USD_PER_TRADE

# Configuração dos ativos suportados
SUPPORTED_ASSETS = [
    AssetSetup("AVNT-USD", "AVNTUSDT", "AVNT", 5, min_notional=1.0),
    AssetSetup("ASTER-USD", "ASTERUSDT", "ASTER", 5, min_notional=1.0),
    AssetSetup("ETH-USD", "ETHUSDT", "ETH", 25, min_notional=1.0),
    AssetSetup("TAO-USD", "TAOUSDT", "TAO", 5, min_notional=1.0),
    AssetSetup("XRP-USD", "XRPUSDT", "XRP", 20, min_notional=1.0),
    AssetSetup("DOGE-USD", "DOGEUSDT", "DOGE", 10, min_notional=1.0),
    AssetSetup("AVAX-USD", "AVAXUSDT", "AVAX", 10, min_notional=1.0),
    AssetSetup("ENA-USD", "ENAUSDT", "ENA", 10, min_notional=1.0),
    AssetSetup("BNB-USD", "BNBUSDT", "BNB", 10, min_notional=1.0),
    AssetSetup("SUI-USD", "SUIUSDT", "SUI", 10, min_notional=1.0),
    AssetSetup("ADA-USD", "ADAUSDT", "ADA", 10, min_notional=1.0),
    AssetSetup("PUMP-USD", "PUMPUSDT", "PUMP", 10, min_notional=1.0),
    AssetSetup("LINK-USD", "LINKUSDT", "LINK", 10, min_notional=1.0),
    AssetSetup("WLD-USD", "WLDUSDT", "WLD", 10, min_notional=1.0),
    AssetSetup("AAVE-USD", "AAVEUSDT", "AAVE", 10, min_notional=1.0),
    AssetSetup("CRV-USD", "CRVUSDT", "CRV", 10, min_notional=1.0),
    AssetSetup("LTC-USD", "LTCUSDT", "LTC", 10, min_notional=1.0),
    AssetSetup("NEAR-USD", "NEARUSDT", "NEAR", 10, min_notional=1.0),
]

# =============================================================================
# CLIENTE HYPERLIQUID SIMPLIFICADO
# =============================================================================

class HyperliquidClient:
    def __init__(self, main_wallet: str, subaccount: str, private_key: str, live_trading: bool = False):
        self.main_wallet = main_wallet
        self.subaccount = subaccount
        self.private_key = private_key
        self.live_trading = live_trading
        self.base_url = "https://api.hyperliquid.xyz"
        
        # Log seguro
        pk_preview = f"{private_key[:6]}...{private_key[-4:]}" if private_key else "NOT_SET"
        logging.info(f"Hyperliquid Client: {'LIVE' if live_trading else 'DEMO'} | "
                    f"Main: {main_wallet[:8]}... | "
                    f"Subaccount: {subaccount} | "
                    f"PK: {pk_preview}")
        
        # Validar configuração se for live trading
        if live_trading:
            self._validate_live_config()
    
    def _validate_live_config(self):
        """Valida se todas as configurações para live trading estão presentes"""
        missing_configs = []
        
        if not self.main_wallet:
            missing_configs.append("HYPERLIQUID_MAIN_WALLET")
        if not self.subaccount:
            missing_configs.append("HYPERLIQUID_SUBACCOUNT") 
        if not self.private_key:
            missing_configs.append("HYPERLIQUID_PRIVATE_KEY")
            
        if missing_configs:
            error_msg = f"Configurações ausentes para LIVE TRADING: {', '.join(missing_configs)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    def _sign_request(self, data: Dict) -> str:
        """Assina requisição conforme documentação da Hyperliquid"""
        if not self.live_trading:
            return "demo_signature"
            
        try:
            # Converter dados para string e assinar
            data_str = json.dumps(data, separators=(',', ':'), sort_keys=True)
            signature = hmac.new(
                self.private_key.encode('utf-8'),
                data_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        except Exception as e:
            logging.error(f"Erro ao assinar requisição: {e}")
            return "error_signature"
    
    def place_order(self, symbol: str, side: str, order_type: str, size: float, price: float) -> Dict:
        """Coloca ordem na SUBCONTA especificada"""
        
        # Dados da ordem conforme documentação da Hyperliquid
        order_data = {
            "action": "placeOrder",
            "symbol": symbol,
            "side": side.upper(),
            "orderType": order_type.upper(),
            "size": size,
            "price": price,
            "subaccount": self.subaccount,
            "reduceOnly": False
        }
        
        if not self.live_trading:
            logging.info(f"🔶 DEMO ORDER [SUBCONTA: {self.subaccount}]: {side} {size} {symbol} @ {price}")
            return {"status": "demo", "order_id": f"demo_{int(time.time())}", "subaccount": self.subaccount}
        
        try:
            # Assinar requisição
            signature = self._sign_request(order_data)
            
            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": self.main_wallet,
                "X-API-SIGNATURE": signature
            }
            
            # Endpoint correto da Hyperliquid
            response = requests.post(
                f"{self.base_url}/exchange",
                json=order_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logging.info(f"🎯 LIVE ORDER [SUBCONTA: {self.subaccount}]: {side} {size} {symbol} @ {price}")
                    return {"status": "success", "order_id": result.get("orderId"), "subaccount": self.subaccount}
                else:
                    logging.error(f"Erro na ordem: {result}")
                    return {"status": "error", "error": result}
            else:
                logging.error(f"Erro API Hyperliquid: {response.status_code} - {response.text}")
                return {"status": "error", "error": response.text}
                
        except Exception as e:
            logging.error(f"Erro ao colocar ordem na subconta {self.subaccount}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_subaccount_position(self, symbol: str) -> Optional[Dict]:
        """Obtém posição atual na SUBCONTA para um símbolo - SIMPLIFICADO"""
        if not self.live_trading:
            return None
            
        try:
            # Para evitar erro 404, vamos usar uma abordagem mais simples
            # Em produção, implementar conforme documentação oficial
            logging.debug(f"Buscando posição para {symbol} na subconta {self.subaccount}")
            return None  # Retornar None por enquanto para evitar erros
            
        except Exception as e:
            logging.debug(f"Erro ao obter posição (pode ser normal): {e}")
            return None

# =============================================================================
# SISTEMA PRINCIPAL DE TRADING (MANTIDO IGUAL)
# =============================================================================

class TradingSystem:
    def __init__(self, config: TradingConfig, asset_setup: AssetSetup):
        self.config = config
        self.asset = asset_setup
        
        # Inicializar cliente focado na SUBCONTA
        self.exchange_client = HyperliquidClient(
            HYPERLIQUID_MAIN_WALLET,
            HYPERLIQUID_SUBACCOUNT,
            HYPERLIQUID_PRIVATE_KEY,
            LIVE_TRADING
        )
        
        self.data_cache = {}
        self._cooldown_until = None
        self._last_operation_at = None
        self.current_position = None
        
        self._setup_logging()
        
        logging.info(f"Sistema inicializado para {asset_setup.name} | "
                    f"SUBCONTA: {HYPERLIQUID_SUBACCOUNT} | "
                    f"Alavancagem: {asset_setup.leverage}x | "
                    f"Entrada: ${asset_setup.usd_per_trade} | "
                    f"Modo: {'LIVE' if LIVE_TRADING else 'DEMO'}")

    def _setup_logging(self):
        """Configura sistema de logging"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
            ]
        )

    def get_historical_data(self, symbol: str, interval: str = "15m", limit: int = 260) -> pd.DataFrame:
        """Obtém dados históricos da Binance com fallback para Bybit"""
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # Verificar cache (simplificado para Render)
        if cache_key in self.data_cache:
            cached_time, cached_data = self.data_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 30:  # Cache de 30 segundos
                return cached_data
        
        try:
            # Tentar Binance primeiro
            url = f"{self.config.BINANCE_API_URL}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Converter tipos
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                
                df = df.rename(columns={
                    'open': 'valor_abertura',
                    'high': 'valor_maximo', 
                    'low': 'valor_minimo',
                    'close': 'valor_fechamento',
                    'volume': 'volume'
                })
                
                # Atualizar cache
                self.data_cache[cache_key] = (datetime.now(), df)
                return df
                
        except Exception as e:
            logging.warning(f"Erro Binance para {symbol}: {e}")
                
        # Retornar DataFrame vazio em caso de falha
        return pd.DataFrame()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos necessários"""
        if len(df) < max(self.config.EMA_LONG_SPAN, self.config.ATR_PERIOD):
            return df
            
        try:
            # EMA 7 e 21
            df['ema_short'] = df['valor_fechamento'].ewm(span=self.config.EMA_SHORT_SPAN, adjust=False).mean()
            df['ema_long'] = df['valor_fechamento'].ewm(span=self.config.EMA_LONG_SPAN, adjust=False).mean()
            
            # ATR (Average True Range)
            df['atr'] = self._calculate_atr(df, period=self.config.ATR_PERIOD)
            df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
            
            # Volume MA
            df['vol_ma'] = df['volume'].rolling(window=self.config.VOL_MA_PERIOD).mean()
            
            # Gradiente EMA7 (taxa de mudança percentual)
            df['ema_short_grad_pct'] = df['ema_short'].pct_change(self.config.N_BARRAS_GRADIENTE) * 100
            
        except Exception as e:
            logging.error(f"Erro no cálculo de indicadores: {e}")
            
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula o Average True Range"""
        high_low = df['valor_maximo'] - df['valor_minimo']
        high_close_prev = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
        low_close_prev = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        true_range = true_range.max(axis=1)
        
        return true_range.rolling(window=period).mean()

    def _check_gradient_consistency(self, df: pd.DataFrame, direction: str) -> bool:
        """Verifica consistência do gradiente por N barras"""
        if len(df) < self.config.GRAD_CONSISTENCY + 1:
            return False
            
        grad_data = df['ema_short_grad_pct'].tail(self.config.GRAD_CONSISTENCY)
        
        if direction == "positive":
            return all(grad > 0 for grad in grad_data if not pd.isna(grad))
        else:  # negative
            return all(grad < 0 for grad in grad_data if not pd.isna(grad))

    def check_long_entry(self, df: pd.DataFrame) -> bool:
        """Verifica todas as condições para entrada LONG"""
        if len(df) < 2:
            return False
            
        last = df.iloc[-1]
        
        # CONDIÇÃO 1: EMA7 > EMA21 (Tendência de alta)
        ema_trend_ok = last.ema_short > last.ema_long
        
        # CONDIÇÃO 2: Gradiente EMA7 consistente (3 barras positivas)
        grad_pos_ok = self._check_gradient_consistency(df, direction="positive")
        
        # CONDIÇÃO 3: ATR% saudável (0.7% - 5.0%)
        atr_healthy = self.config.ATR_PCT_MIN <= last.atr_pct <= self.config.ATR_PCT_MAX
        
        # CONDIÇÃO 4: Breakout acima EMA7 + 3.0*ATR
        breakout_ok = last.valor_fechamento > (last.ema_short + self.config.BREAKOUT_K_ATR * last.atr)
        
        # CONDIÇÃO 5: Volume atual > Média móvel de volume
        volume_ok = last.volume > last.vol_ma
        
        conditions_met = all([ema_trend_ok, grad_pos_ok, atr_healthy, breakout_ok, volume_ok])
        
        if conditions_met:
            self._log_trade_metrics(df, "LONG")
            
        return conditions_met

    def check_short_entry(self, df: pd.DataFrame) -> bool:
        """Verifica todas as condições para entrada SHORT"""
        if len(df) < 2:
            return False
            
        last = df.iloc[-1]
        
        # CONDIÇÃO 1: EMA7 < EMA21 (Tendência de baixa)
        ema_trend_ok = last.ema_short < last.ema_long
        
        # CONDIÇÃO 2: Gradiente EMA7 consistente (3 barras negativas)
        grad_neg_ok = self._check_gradient_consistency(df, direction="negative")
        
        # CONDIÇÃO 3: ATR% saudável (0.7% - 5.0%)
        atr_healthy = self.config.ATR_PCT_MIN <= last.atr_pct <= self.config.ATR_PCT_MAX
        
        # CONDIÇÃO 4: Breakout abaixo EMA7 - 3.0*ATR
        breakout_ok = last.valor_fechamento < (last.ema_short - self.config.BREAKOUT_K_ATR * last.atr)
        
        # CONDIÇÃO 5: Volume atual > Média móvel de volume
        volume_ok = last.volume > last.vol_ma
        
        conditions_met = all([ema_trend_ok, grad_neg_ok, atr_healthy, breakout_ok, volume_ok])
        
        if conditions_met:
            self._log_trade_metrics(df, "SHORT")
            
        return conditions_met

    def _log_trade_metrics(self, df: pd.DataFrame, side: str):
        """Log detalhado das condições de mercado no momento do trigger"""
        last = df.iloc[-1]
        
        # Calcular métricas de volume
        current_trades = float(last.volume)
        last_30_volumes = df["volume"].tail(30)
        avg_trades_30 = float(last_30_volumes.mean()) if len(last_30_volumes) > 0 else 0.0
        trades_ratio = current_trades / avg_trades_30 if avg_trades_30 > 0 else 0.0
        
        # Calcular breakout atual
        current_breakout_k = abs(float(last.valor_fechamento) - float(last.ema_short)) / float(last.atr)
        
        # Gradiente EMA7
        g_last = float(last.ema_short_grad_pct) if pd.notna(last.ema_short_grad_pct) else 0.0
        
        logging.info(
            f"🚨 TRIGGER {side} | {self.asset.name} | "
            f"Preço: {last.valor_fechamento:.6f} | "
            f"EMA7: {last.ema_short:.6f} | "
            f"EMA21: {last.ema_long:.6f} | "
            f"ATR%: {last.atr_pct:.3f}% | "
            f"Volume: {trades_ratio:.2f}x | "
            f"Gradiente: {g_last:.4f}%"
        )

    def calculate_position_size(self, entry_price: float) -> float:
        """Calcula o tamanho da posição baseado no valor por trade"""
        # Verificar valor mínimo da Hyperliquid
        min_notional = self.asset.min_notional
        usd_per_trade = max(self.asset.usd_per_trade, min_notional)
        
        position_value = usd_per_trade * self.asset.leverage
        quantity = position_value / entry_price
        return round(quantity, 6)

    def calculate_dynamic_take_profit(self, entry_price: float, side: str, current_volatility: float) -> float:
        """Calcula take profit dinâmico baseado na volatilidade"""
        base_tp_pct = 2.5  # Base de 2.5%
        
        # Ajustar TP baseado na volatilidade (ATR%)
        if current_volatility < 1.0:
            # Baixa volatilidade - TP mais conservador
            tp_multiplier = 1.0
        elif current_volatility > 3.0:
            # Alta volatilidade - TP mais agressivo
            tp_multiplier = 2.0
        else:
            # Volatilidade média
            tp_multiplier = 1.5
        
        dynamic_tp_pct = base_tp_pct * tp_multiplier
        
        if side == "buy":
            return entry_price * (1.0 + dynamic_tp_pct / 100)
        else:
            return entry_price * (1.0 - dynamic_tp_pct / 100)

    def _cooldown_ativo(self) -> bool:
        """Verifica se está em período de cooldown"""
        if self._cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < self._cooldown_until

    def _marcar_cooldown(self):
        """Inicia período de cooldown após fechamento de posição"""
        self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.COOLDOWN_MINUTOS)
        logging.info(f"⏰ Cooldown ativado por {self.config.COOLDOWN_MINUTOS} minutos")

    def _anti_spam_ok(self) -> bool:
        """Previne operações muito frequentes"""
        now = datetime.now(timezone.utc)
        
        if self._last_operation_at is None:
            return True
            
        return (now - self._last_operation_at).total_seconds() >= self.config.ANTI_SPAM_SECS

    def _get_position(self) -> Optional[Dict]:
        """Obtém posição atual da SUBCONTA - SIMPLIFICADO"""
        # Por enquanto, usar posição local para evitar erros de API
        return self.current_position

    def _open_position(self, side: str, df: pd.DataFrame):
        """Abre uma nova posição na SUBCONTA"""
        if not self._anti_spam_ok():
            return
            
        last_price = df.iloc[-1]['valor_fechamento']
        current_volatility = df.iloc[-1]['atr_pct']
        quantity = self.calculate_position_size(last_price)
        
        # Calcular stops dinâmicos
        stop_loss = self._calculate_stop_loss(last_price, side)
        take_profit = self.calculate_dynamic_take_profit(last_price, side, current_volatility)
        
        # Verificar se quantity atende mínimo da exchange
        if quantity * last_price < self.asset.min_notional:
            logging.warning(f"Quantidade abaixo do mínimo para {self.asset.name} na subconta {HYPERLIQUID_SUBACCOUNT}")
            return

        # Colocar ordem na SUBCONTA
        order_result = self.exchange_client.place_order(
            symbol=self.asset.trading_symbol,
            side=side,
            order_type="market",
            size=quantity,
            price=last_price
        )
        
        if order_result.get('status') in ['success', 'demo']:
            self.current_position = {
                'side': side,
                'entry_price': last_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(timezone.utc),
                'usd_value': self.asset.usd_per_trade,
                'order_id': order_result.get('order_id'),
                'subaccount': HYPERLIQUID_SUBACCOUNT
            }
            
            self._last_operation_at = datetime.now(timezone.utc)
            
            logging.info(
                f"🎯 {'DEMO ' if not LIVE_TRADING else ''}ABERTURA {side} | "
                f"SUBCONTA: {HYPERLIQUID_SUBACCOUNT} | "
                f"Ativo: {self.asset.name} | "
                f"Preço: {last_price:.6f} | "
                f"Quantidade: {quantity:.6f} | "
                f"Valor: ${self.asset.usd_per_trade}"
            )

    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calcula stop loss fixo"""
        stop_pct = self.config.STOP_LOSS_PCT / 100
        
        if side == "buy":
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)

    def _close_position(self, reason: str, exit_price: float):
        """Fecha posição atual na SUBCONTA"""
        if not self.current_position:
            return
            
        position = self.current_position
        side = position['side']
        entry_price = position['entry_price']
        
        # Calcular P&L
        if side == "buy":
            pnl_pct = (exit_price - entry_price) / entry_price * 100 * self.asset.leverage
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100 * self.asset.leverage
            
        pnl_usd = (pnl_pct / 100) * position['usd_value']
        
        # Fechar posição na SUBCONTA
        close_side = "sell" if side == "buy" else "buy"
        self.exchange_client.place_order(
            symbol=self.asset.trading_symbol,
            side=close_side,
            order_type="market",
            size=position['quantity'],
            price=exit_price
        )
        
        logging.info(
            f"🔒 {'DEMO ' if not LIVE_TRADING else ''}FECHAMENTO {reason} | "
            f"SUBCONTA: {HYPERLIQUID_SUBACCOUNT} | "
            f"Ativo: {self.asset.name} | "
            f"Side: {side} | "
            f"P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})"
        )
        
        # Limpar posição e ativar cooldown
        self.current_position = None
        self._marcar_cooldown()

    def _check_exit_conditions(self, df: pd.DataFrame):
        """Verifica condições de saída dinâmicas"""
        if not self.current_position:
            return
            
        position = self.current_position
        current_price = df.iloc[-1]['valor_fechamento']
        side = position['side']
        
        # Verificar stop loss
        if (side == "buy" and current_price <= position['stop_loss']) or \
           (side == "sell" and current_price >= position['stop_loss']):
            self._close_position("STOP LOSS", current_price)
            return
            
        # Verificar take profit
        if (side == "buy" and current_price >= position['take_profit']) or \
           (side == "sell" and current_price <= position['take_profit']):
            self._close_position("TAKE PROFIT", current_price)
            return
            
        # Estratégia de saída por trailing ou condições técnicas
        self._check_technical_exit(df, current_price)

    def _check_technical_exit(self, df: pd.DataFrame, current_price: float):
        """Estratégia de saída por condições técnicas para maximizar lucro"""
        if not self.current_position:
            return
            
        position = self.current_position
        side = position['side']
        last = df.iloc[-1]
        
        # Condição 1: Inversão do gradiente
        if side == "buy":
            grad_negative = last.ema_short_grad_pct < -0.1
            ema_crossunder = last.ema_short < last.ema_long
            if grad_negative and ema_crossunder:
                self._close_position("INVERSÃO TÉCNICA", current_price)
                return
        else:
            grad_positive = last.ema_short_grad_pct > 0.1
            ema_crossover = last.ema_short > last.ema_long
            if grad_positive and ema_crossover:
                self._close_position("INVERSÃO TÉCNICA", current_price)
                return

    def run_strategy(self):
        """Loop principal da estratégia"""
        try:
            # Obter dados históricos
            df = self.get_historical_data(self.asset.data_symbol, "15m", 260)
            
            if df.empty:
                return
                
            # Calcular indicadores
            df = self.compute_indicators(df)
            
            if len(df) < self.config.EMA_LONG_SPAN:
                return
                
            # Verificar posição atual
            current_position = self._get_position()
            
            # Lógica de saída (se em posição)
            if current_position:
                self._check_exit_conditions(df)
                return
                
            # Verificar cooldown
            if self._cooldown_ativo():
                return
                
            # Lógica de entrada
            if self.check_long_entry(df):
                self._open_position("buy", df)
            elif self.check_short_entry(df):
                self._open_position("sell", df)
                
        except Exception as e:
            logging.error(f"Erro na execução da estratégia para {self.asset.name}: {e}")

# =============================================================================
# SISTEMA DE GERENCIAMENTO MULTI-ATIVOS
# =============================================================================

class MultiAssetTradingManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trading_systems = {}
        self.running = False
        
        # Inicializar sistemas para cada ativo
        for asset in SUPPORTED_ASSETS:
            self.trading_systems[asset.name] = TradingSystem(config, asset)
            
        logging.info(f"Gerenciador multi-ativos inicializado com {len(self.trading_systems)} ativos")

    def start_trading(self):
        """Inicia o loop de trading para todos os ativos"""
        self.running = True
        logging.info("🚀 INICIANDO SISTEMA DE TRADING AUTOMATIZADO")
        logging.info(f"💰 Valor por operação: ${USD_PER_TRADE}")
        logging.info(f"🔧 Modo: {'LIVE TRADING' if LIVE_TRADING else 'DEMO/MONITORAMENTO'}")
        
        cycle_count = 0
        while self.running:
            try:
                cycle_count += 1
                if cycle_count % 20 == 0:  # Log a cada 20 ciclos para não poluir
                    logging.info(f"🔁 Ciclo #{cycle_count} - Monitorando {len(self.trading_systems)} ativos")
                
                for asset_name, system in self.trading_systems.items():
                    system.run_strategy()
                    
                # Esperar 15 segundos entre ciclos
                time.sleep(15)
                
            except KeyboardInterrupt:
                logging.info("Parando sistema por interrupção do usuário...")
                self.stop_trading()
            except Exception as e:
                logging.error(f"Erro no loop principal: {e}")
                time.sleep(30)  # Esperar 30 segundos em caso de erro

    def stop_trading(self):
        """Para o sistema de trading"""
        self.running = False
        logging.info("Sistema de trading parado")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal para executar o sistema"""
    
    # Verificar variáveis de ambiente críticas
    if LIVE_TRADING:
        required_vars = ['HYPERLIQUID_MAIN_WALLET', 'HYPERLIQUID_SUBACCOUNT', 'HYPERLIQUID_PRIVATE_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logging.error(f"Variáveis de ambiente ausentes para LIVE TRADING: {missing_vars}")
            logging.error("Configure as variáveis no Render ou altere LIVE_TRADING para 0")
            return
    
    # Configuração global
    config = TradingConfig()
    
    # Inicializar gerenciador multi-ativos
    manager = MultiAssetTradingManager(config)
    
    try:
        # Iniciar trading
        manager.start_trading()
        
    except Exception as e:
        logging.error(f"Erro fatal: {e}")
    finally:
        manager.stop_trading()

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 SISTEMA DE TRADING AUTOMATIZADO - TRADINGV4")
    print("=" * 70)
    print(f"💰 Valor por operação: ${USD_PER_TRADE}")
    print(f"🏠 Main Wallet: {HYPERLIQUID_MAIN_WALLET[:8]}...{HYPERLIQUID_MAIN_WALLET[-6:]}" if HYPERLIQUID_MAIN_WALLET else "Não configurado")
    print(f"📊 Subaccount: {HYPERLIQUID_SUBACCOUNT}" if HYPERLIQUID_SUBACCOUNT else "Não configurado")
    print(f"🔑 Private Key: {'***CONFIGURADO***' if HYPERLIQUID_PRIVATE_KEY else 'Não configurado'}")
    print(f"🎯 Modo: {'LIVE TRADING' if LIVE_TRADING else 'DEMO/MONITORAMENTO'}")
    print(f"🎯 FOCO: OPERAÇÕES NA SUBCONTA: {HYPERLIQUID_SUBACCOUNT}")
    print(f"⏰ Timeframe: 15 minutos")
    print(f"📊 Ativos monitorados: {len(SUPPORTED_ASSETS)}")
    print(f"❄️ Cooldown: {TradingConfig.COOLDOWN_MINUTOS} minutos")
    print("=" * 70)
    
    main()