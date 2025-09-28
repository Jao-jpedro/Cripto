#!/usr/bin/env python3
"""
TradingFuturo100.py - Sistema de Trading para 100%+ ROI
======================================================

Sistema revolucionário de swing trading com estratégias de alta qualidade:
- Timeframes: 4h e 1d (sinais de qualidade)
- Leverage: 2-5x controlado
- Stop-loss e take-profit inteligentes
- Position sizing adaptativo
- Banca inicial: $10
- Meta: 100%+ ROI anual

Estratégias Implementadas:
1. Breakout Multi-Timeframe (4h/1d)
2. Trend Following com Confluence (4h)  
3. Support/Resistance Break + Volume (1d)
4. Mean Reversion com Divergência (4h)
5. Momentum Squeeze + Expansion (1d)
"""

import os
import sys
import pandas as pd
import numpy as np
import ccxt
import ta
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingFuturo100')

# === CONFIGURAÇÕES PARA 100%+ ROI ===
INITIAL_BALANCE = 10.0      # $10 de banca inicial
BASE_POSITION_SIZE = 3.0    # $3 base por trade (30%)
MAX_POSITION_SIZE = 8.0     # Máximo $8 em trades de alta confiança (80%)
COMMISSION_RATE = 0.0001    # 0.01% taxa baixíssima
MAX_LEVERAGE = 5.0          # Leverage máximo controlado
TARGET_ROI = 100.0          # Meta: 100% ROI
STOP_LOSS_PCT = 0.08        # Stop loss 8% (mais flexível)
TAKE_PROFIT_PCT = 0.25      # Take profit 25% (agressivo)

# Lista de ativos selecionados (alta volatilidade para 100%+ ROI)
ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'DOGE/USDT',
    'ADA/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT', 'NEAR/USDT'
]

@dataclass
class TradeResult:
    """Resultado de um trade individual"""
    strategy_name: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    side: str = 'buy'  # 'buy' or 'sell'
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    leverage: float = 1.0
    confidence: float = 0.0
    exit_reason: str = ""

class DataProvider:
    """Provedor de dados históricos otimizado para swing trading"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        self.cache = {}
    
    def get_historical_data(self, symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Busca dados históricos com foco em swing trading"""
        cache_key = f"{symbol}_{timeframe}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            since = self.exchange.parse8601(
                (datetime.now() - timedelta(days=days)).isoformat()
            )
            
            logger.info(f"Carregando dados históricos: {symbol} {timeframe} ({days} dias)")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache dos dados
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos para swing trading"""
        if df.empty:
            return df
        
        # === MÉDIAS MÓVEIS PARA SWING TRADING ===
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)  
        df['ema_100'] = ta.trend.ema_indicator(df['close'], window=100)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        
        # === OSCILADORES DE MOMENTUM ===
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        
        # === MACD PARA SWING TRADING ===
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # === BOLLINGER BANDS ===
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === VOLATILIDADE ===
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # === TREND INDICATORS ===
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['di_plus'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['di_minus'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        # === VOLUME INDICATORS ===
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        
        # === SUPPORT/RESISTANCE LEVELS ===
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_1'] = 2 * df['pivot'] - df['high']
        
        # === BREAKOUT INDICATORS ===
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['high_50'] = df['high'].rolling(window=50).max()
        df['low_50'] = df['low'].rolling(window=50).min()
        
        # === MOMENTUM SQUEEZE ===
        df['squeeze'] = (df['bb_width'] < df['bb_width'].rolling(window=20).mean()) & (df['atr_pct'] < 0.02)
        
        return df

class BaseStrategy(ABC):
    """Classe base para estratégias de swing trading"""
    
    def __init__(self, name: str):
        self.name = name
        self.trades = []
        self.balance = INITIAL_BALANCE
        self.positions = {}
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Gera sinais de alta qualidade para swing trading"""
        pass
    
    def calculate_position_size(self, price: float, confidence: float, volatility: float) -> Tuple[float, float]:
        """Calcula position size adaptativo com leverage baseado na confiança"""
        # Position size base aumenta com confiança
        if confidence > 0.9:
            base_size = MAX_POSITION_SIZE  # $8 para sinais de alta confiança
            leverage = min(5.0, 3.0 + confidence * 2)  # Leverage até 5x
        elif confidence > 0.75:
            base_size = BASE_POSITION_SIZE * 1.5  # $4.5
            leverage = min(3.0, 2.0 + confidence)
        else:
            base_size = BASE_POSITION_SIZE  # $3 base
            leverage = min(2.0, 1.5 + confidence)
        
        # Ajusta pela volatilidade (menor size se muito volátil)
        if volatility > 0.05:  # > 5% ATR
            base_size *= 0.7
        elif volatility < 0.02:  # < 2% ATR
            base_size *= 1.3
        
        quantity = (base_size * leverage) / price
        return quantity, leverage
    
    def execute_trade(self, signal: Dict, symbol: str, volatility: float) -> Optional[TradeResult]:
        """Executa trade com sizing adaptativo e leverage"""
        price = signal['price']
        confidence = signal['confidence']
        
        if signal['action'] == 'buy' and len(self.positions) < 3:  # Max 3 posições
            quantity, leverage = self.calculate_position_size(price, confidence, volatility)
            
            # Calcula comissão com leverage
            commission = quantity * price * COMMISSION_RATE / leverage
            
            if self.balance >= commission:
                self.balance -= commission
                
                trade = TradeResult(
                    strategy_name=self.name,
                    symbol=symbol,
                    entry_time=signal['time'],
                    entry_price=price,
                    side='buy',
                    quantity=quantity,
                    leverage=leverage,
                    confidence=confidence
                )
                
                self.positions[symbol] = trade
                logger.debug(f"{self.name}: OPEN {symbol} @ {price:.4f} | Size: {quantity:.4f} | Leverage: {leverage:.2f}x | Conf: {confidence:.2f}")
                return trade
                
        elif signal['action'] == 'sell' and symbol in self.positions:
            position = self.positions[symbol]
            
            # Calcula P&L com leverage
            price_change_pct = (price - position.entry_price) / position.entry_price
            pnl_gross = position.quantity * position.entry_price * price_change_pct * position.leverage
            commission = position.quantity * price * COMMISSION_RATE / position.leverage
            pnl_net = pnl_gross - commission
            
            # Finaliza trade
            trade = TradeResult(
                strategy_name=self.name,
                symbol=symbol,
                entry_time=position.entry_time,
                exit_time=signal['time'],
                entry_price=position.entry_price,
                exit_price=price,
                side=position.side,
                quantity=position.quantity,
                pnl=pnl_net,
                pnl_pct=price_change_pct * position.leverage * 100,
                leverage=position.leverage,
                confidence=position.confidence,
                exit_reason=signal.get('reason', 'signal')
            )
            
            self.balance += pnl_net
            self.trades.append(trade)
            del self.positions[symbol]
            
            logger.debug(f"{self.name}: CLOSE {symbol} @ {price:.4f} | PnL: ${pnl_net:.2f} ({trade.pnl_pct:.1f}%) | Leverage: {position.leverage:.2f}x")
            return trade
            
        return None

# === ESTRATÉGIAS PARA 100%+ ROI ===

class BreakoutMultiTimeframe(BaseStrategy):
    """Estratégia 1: Breakout Multi-Timeframe com Confluence"""
    
    def __init__(self):
        super().__init__("Breakout Multi-Timeframe")
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """🎯 VERSÃO SIMPLIFICADA PARA TESTES"""
        signals = []
        
        for i in range(50, len(df)):  # Reduzido para 50 períodos
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Sinais mais simples e permissivos
            bullish_signal = (current['close'] > current['ema_21'] and  # Preço acima EMA 21
                             current['rsi'] > 45 and current['rsi'] < 75 and  # RSI em range bom
                             current['volume_ratio'] > 1.2)  # Volume 20% acima da média
            
            bearish_signal = (current['close'] < current['ema_21'] and  # Preço abaixo EMA 21
                             current['rsi'] > 25 and current['rsi'] < 55 and  # RSI em range bom
                             current['volume_ratio'] > 1.2)  # Volume 20% acima da média
            
            # === ENTRADA LONGA ===
            if bullish_signal:
                confidence = min(0.85, 
                    (current['rsi'] - 45) / 30 +  # RSI strength
                    min(current['volume_ratio'] / 2, 1) * 0.3 +  # Volume factor
                    0.4  # Base confidence
                )
                
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'stop_loss': current['close'] * 0.95,  # SL 5%
                    'take_profit': current['close'] * 1.10,  # TP 10%
                    'reason': f'Bullish signal - RSI:{current["rsi"]:.1f} Vol:{current["volume_ratio"]:.2f}x'
                })
            
            # === ENTRADA CURTA ===
            elif bearish_signal:
                confidence = min(0.85,
                    (55 - current['rsi']) / 30 +  # RSI strength (inverted)
                    min(current['volume_ratio'] / 2, 1) * 0.3 +  # Volume factor
                    0.4  # Base confidence
                )
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': confidence,
                    'stop_loss': current['close'] * 1.05,  # SL 5%
                    'take_profit': current['close'] * 0.90,  # TP 10%
                    'reason': f'Bearish signal - RSI:{current["rsi"]:.1f} Vol:{current["volume_ratio"]:.2f}x'
                })
        
        logger.info(f"🎯 {self.name}: {len(signals)} sinais gerados para {symbol}")
        return signals
                
                logger.info(f"🎯 {self.name}: {len(signals)} sinais gerados para {symbol}")
        return signals
            
            # === SAÍDAS ===
            # Take Profit
            elif (current['rsi'] > 85 or 
                  current['close'] > current['bb_upper'] * 1.05 or
                  current['macd_histogram'] < prev['macd_histogram'] * 0.5):
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.8,
                    'reason': 'take_profit'
                })
            
            # Stop Loss
            elif (current['close'] < current['ema_21'] * 0.92 or  # 8% abaixo EMA21
                  current['rsi'] < 30):
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.9,
                    'reason': 'stop_loss'
                })
        
        return signals

class TrendFollowingConfluence(BaseStrategy):
    """Estratégia 2: Trend Following com Múltipla Confluência"""
    
    def __init__(self):
        super().__init__("Trend Following Confluence")
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        signals = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # === CONFLUÊNCIA DE TREND FOLLOWING ===
            
            # 1. EMA Alignment (todas alinhadas)
            ema_bullish = (current['ema_21'] > current['ema_50'] > current['ema_100'] > current['ema_200'])
            ema_cross_bull = (current['ema_21'] > current['ema_50'] and prev['ema_21'] <= prev['ema_50'])
            
            # 2. MACD Strong Signal
            macd_strong_bull = (current['macd'] > current['macd_signal'] and
                              current['macd_histogram'] > prev['macd_histogram'] > prev2['macd_histogram'] and
                              current['macd'] > 0)
            
            # 3. ADX Trend Strength
            strong_trend = current['adx'] > 30 and current['di_plus'] > current['di_minus']
            
            # 4. RSI in Trend Zone
            rsi_trending = 55 < current['rsi'] < 75  # RSI em zona de trend forte
            
            # 5. Volume Confirmation
            volume_confirm = current['volume_ratio'] > 1.3
            
            # 6. Price Action
            bullish_candle = current['close'] > current['open'] and current['close'] > prev['close'] * 1.01
            
            # === ENTRADA DE ALTA QUALIDADE ===
            if (ema_bullish and macd_strong_bull and strong_trend and 
                rsi_trending and volume_confirm and bullish_candle):
                
                confidence = 0.92
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'trend_confluence'
                })
            
            # === ENTRADA EM PULLBACK ===
            elif (ema_bullish and strong_trend and
                  current['close'] > current['ema_21'] * 0.98 and  # Próximo EMA21
                  current['close'] < prev['close'] and  # Pullback
                  current['rsi'] > 45 and current['rsi'] < 60):  # RSI não oversold
                
                confidence = 0.85
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'pullback_entry'
                })
            
            # === SAÍDAS ===
            elif (current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal'] or  # MACD cross down
                  current['rsi'] > 80 or  # Overbought extremo
                  current['close'] < current['ema_21'] * 0.95):  # 5% abaixo EMA21
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.85,
                    'reason': 'trend_exit'
                })
        
        return signals

class SupportResistanceBreak(BaseStrategy):
    """Estratégia 3: S/R Break + Volume Surge"""
    
    def __init__(self):
        super().__init__("Support/Resistance Break")
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        signals = []
        
        for i in range(100, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # === IDENTIFICAÇÃO DE S/R DINÂMICOS ===
            resistance_level = max(current['pivot'], current['resistance_1'], current['bb_upper'])
            support_level = min(current['pivot'], current['support_1'], current['bb_lower'])
            
            # === BREAKOUT COM VOLUME ===
            resistance_break = (current['close'] > resistance_level * 1.015 and  # 1.5% acima resistência
                              prev['close'] <= resistance_level * 1.01)
            
            support_break = (current['close'] < support_level * 0.985 and  # 1.5% abaixo suporte
                           prev['close'] >= support_level * 0.99)
            
            # Volume explosivo
            explosive_volume = current['volume_ratio'] > 3.0
            
            # Momentum confirmation
            momentum_up = current['rsi'] > 65 and current['macd_histogram'] > 0
            momentum_down = current['rsi'] < 35 and current['macd_histogram'] < 0
            
            # === ENTRADA BREAKOUT RESISTANCE ===
            if resistance_break and explosive_volume and momentum_up:
                confidence = 0.90
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'resistance_break'
                })
            
            # === ENTRADA SUPPORT BOUNCE ===
            elif (current['close'] > support_level and current['close'] < support_level * 1.02 and
                  prev['close'] < support_level and explosive_volume and
                  current['rsi'] < 40):  # Oversold bounce
                
                confidence = 0.88
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'support_bounce'
                })
            
            # === SAÍDAS ===
            elif (current['close'] > resistance_level * 1.25 or  # 25% profit target
                  current['close'] < current['ema_50'] * 0.92):  # Stop loss
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.85,
                    'reason': 'sr_exit'
                })
        
        return signals

class MeanReversionDivergence(BaseStrategy):
    """Estratégia 4: Mean Reversion com Divergência RSI"""
    
    def __init__(self):
        super().__init__("Mean Reversion Divergence")
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        signals = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev5 = df.iloc[i-5] if i >= 5 else prev
            
            # === DIVERGÊNCIA RSI ===
            # Bullish Divergence: Price lower low, RSI higher low
            price_lower_low = current['close'] < prev5['close']
            rsi_higher_low = current['rsi'] > prev5['rsi']
            bullish_divergence = price_lower_low and rsi_higher_low and current['rsi'] < 35
            
            # Bearish Divergence: Price higher high, RSI lower high  
            price_higher_high = current['close'] > prev5['close']
            rsi_lower_high = current['rsi'] < prev5['rsi']
            bearish_divergence = price_higher_high and rsi_lower_high and current['rsi'] > 65
            
            # === MEAN REVERSION SETUP ===
            oversold_extreme = current['rsi'] < 25 and current['bb_percent'] < 0.1
            overbought_extreme = current['rsi'] > 75 and current['bb_percent'] > 0.9
            
            # === ENTRADA BULLISH DIVERGENCE ===
            if (bullish_divergence and oversold_extreme and 
                current['volume_ratio'] > 1.5):
                
                confidence = 0.87
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'bullish_divergence'
                })
            
            # === ENTRADA OVERSOLD BOUNCE ===
            elif (oversold_extreme and current['close'] > prev['close'] and
                  current['rsi'] > prev['rsi']):  # RSI turning up
                
                confidence = 0.82
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'oversold_bounce'
                })
            
            # === SAÍDAS ===
            elif (current['rsi'] > 70 or overbought_extreme or
                  current['close'] > current['bb_upper']):
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.85,
                    'reason': 'mean_reversion_exit'
                })
        
        return signals

class MomentumSqueezeExpansion(BaseStrategy):
    """Estratégia 5: Momentum Squeeze + Expansion"""
    
    def __init__(self):
        super().__init__("Momentum Squeeze Expansion")
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        signals = []
        
        for i in range(30, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # === SQUEEZE DETECTION ===
            in_squeeze = current['squeeze']
            squeeze_release = not current['squeeze'] and prev['squeeze']
            
            # === MOMENTUM DIRECTION ===
            momentum_up = (current['close'] > current['ema_21'] and
                          current['macd_histogram'] > prev['macd_histogram'] and
                          current['rsi'] > 50)
            
            momentum_down = (current['close'] < current['ema_21'] and
                           current['macd_histogram'] < prev['macd_histogram'] and
                           current['rsi'] < 50)
            
            # === VOLATILITY EXPANSION ===
            volatility_expansion = current['atr_pct'] > prev['atr_pct'] * 1.3
            
            # === ENTRADA SQUEEZE RELEASE UP ===
            if (squeeze_release and momentum_up and volatility_expansion and
                current['volume_ratio'] > 2.0):
                
                confidence = 0.93
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'squeeze_release_up'
                })
            
            # === ENTRADA MOMENTUM CONTINUATION ===
            elif (not in_squeeze and momentum_up and 
                  current['adx'] > 25 and current['rsi'] > 60 and current['rsi'] < 75):
                
                confidence = 0.85
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': confidence,
                    'reason': 'momentum_continuation'
                })
            
            # === SAÍDAS ===
            elif (current['squeeze'] or  # Voltou ao squeeze
                  current['rsi'] > 85 or  # Overbought extremo
                  current['adx'] < 15):  # Perda de momentum
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.8,
                    'reason': 'momentum_exit'
                })
        
        return signals

class TradingEngine:
    """Engine principal para trading com meta de 100%+ ROI"""
    
    def __init__(self):
        self.data_provider = DataProvider()
        self.strategies = [
            BreakoutMultiTimeframe(),
            TrendFollowingConfluence(), 
            SupportResistanceBreak(),
            MeanReversionDivergence(),
            MomentumSqueezeExpansion()
        ]
        self.results = {}
    
    def run_backtest(self, symbol: str, timeframe: str, days: int = 365):
        """Executa backtest para uma estratégia e ativo"""
        df = self.data_provider.get_historical_data(symbol, timeframe, days)
        if df.empty:
            return
        
        df = self.data_provider.add_technical_indicators(df)
        
        for strategy in self.strategies:
            logger.info(f"Testando estratégia: {strategy.name}")
            
            # Reset strategy state
            strategy.balance = INITIAL_BALANCE
            strategy.positions = {}
            strategy.trades = []
            
            signals = strategy.generate_signals(df, symbol)
            
            for signal in signals:
                # Busca volatilidade no momento do sinal
                signal_time = signal['time']
                try:
                    signal_row = df.loc[signal_time]
                    volatility = signal_row['atr_pct']
                except:
                    volatility = 0.03  # Default 3%
                
                trade = strategy.execute_trade(signal, symbol, volatility)
            
            # Fecha posições abertas
            if strategy.positions:
                final_price = df['close'].iloc[-1]
                final_time = df.index[-1]
                for pos_symbol in list(strategy.positions.keys()):
                    final_signal = {
                        'time': final_time,
                        'action': 'sell',
                        'price': final_price,
                        'confidence': 0.5,
                        'reason': 'backtest_end'
                    }
                    strategy.execute_trade(final_signal, pos_symbol, 0.03)
        
        # Salva resultados
        if symbol not in self.results:
            self.results[symbol] = {}
        for strategy in self.strategies:
            self.results[symbol][strategy.name] = strategy.trades.copy()
    
    def run_full_backtest(self):
        """Executa backtest completo otimizado para 100%+ ROI"""
        # Timeframes otimizados para swing trading
        timeframes = {
            'Breakout Multi-Timeframe': '4h',
            'Trend Following Confluence': '4h', 
            'Support/Resistance Break': '1d',
            'Mean Reversion Divergence': '4h',
            'Momentum Squeeze Expansion': '1d'
        }
        
        # Testar todos os ativos
        for asset in ASSETS:
            for strategy in self.strategies:
                tf = timeframes[strategy.name]
                logger.info(f"\n=== Backtesting {asset} ({tf}) ===")
                self.run_backtest(asset, tf, days=365)  # 1 ano completo
    
    def calculate_metrics(self, trades: List[TradeResult]) -> Dict[str, float]:
        """Calcula métricas de performance"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'roi': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'final_balance': INITIAL_BALANCE,
                'avg_leverage': 0.0
            }
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in trades)
        final_balance = INITIAL_BALANCE + total_pnl
        roi = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        # Drawdown calculation
        cumulative_pnl = 0
        peak = INITIAL_BALANCE
        max_drawdown = 0
        for trade in trades:
            cumulative_pnl += trade.pnl
            current_balance = INITIAL_BALANCE + cumulative_pnl
            if current_balance > peak:
                peak = current_balance
            drawdown = (peak - current_balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe Ratio (simplified)
        if trades:
            returns = [t.pnl_pct for t in trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'total_pnl': total_pnl,
            'roi': roi,
            'profit_factor': sum(w.pnl for w in wins) / abs(sum(l.pnl for l in losses)) if losses else float('inf'),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': np.mean([w.pnl for w in wins]) if wins else 0,
            'avg_loss': np.mean([l.pnl for l in losses]) if losses else 0,
            'max_win': max([w.pnl for w in wins]) if wins else 0,
            'max_loss': min([l.pnl for l in losses]) if losses else 0,
            'final_balance': final_balance,
            'avg_leverage': np.mean([t.leverage for t in trades]) if trades else 0,
            'trades_per_month': len(trades) / 12.0  # Assumindo 1 ano
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Gera relatório otimizado para 100%+ ROI"""
        all_strategies = {}
        
        # Consolida todas as trades por estratégia
        for asset_results in self.results.values():
            for strategy_name, trades in asset_results.items():
                if strategy_name not in all_strategies:
                    all_strategies[strategy_name] = []
                all_strategies[strategy_name].extend(trades)
        
        # Calcula métricas por estratégia
        strategy_metrics = {}
        for strategy_name, trades in all_strategies.items():
            metrics = self.calculate_metrics(trades)
            # Garantir que todas as métricas existam
            required_metrics = ['trades_per_month', 'max_win', 'max_loss']
            for metric in required_metrics:
                if metric not in metrics:
                    metrics[metric] = 0.0
            strategy_metrics[strategy_name] = metrics
        
        # Encontra melhor estratégia por ROI
        best_strategy = max(strategy_metrics.items(), key=lambda x: x[1]['roi'])
        
        report = {
            'summary': {
                'best_strategy': best_strategy[0],
                'best_roi': best_strategy[1]['roi'],
                'target_roi': TARGET_ROI,
                'target_achieved': best_strategy[1]['roi'] >= TARGET_ROI,
                'total_strategies': len(strategy_metrics),
                'profitable_strategies': len([m for m in strategy_metrics.values() if m['roi'] > 0])
            },
            'strategies': strategy_metrics,
            'best_performance': best_strategy[1]
        }
        
        return report
    
    def print_report(self):
        """Imprime relatório detalhado focado em 100%+ ROI"""
        report = self.generate_report()
        
        print("=" * 80)
        print(" 🚀 TRADINGFUTURO100 - RELATÓRIO DE PERFORMANCE 🚀")
        print("=" * 80)
        print()
        
        # Resumo executivo
        summary = report['summary']
        print("📊 RESUMO EXECUTIVO:")
        print(f"├─ Meta ROI: {TARGET_ROI}%")
        print(f"├─ Melhor Estratégia: {summary['best_strategy']}")
        print(f"├─ Melhor ROI: {summary['best_roi']:.2f}%")
        print(f"├─ Meta Alcançada: {'✅ SIM' if summary['target_achieved'] else '❌ NÃO'}")
        print(f"├─ Estratégias Testadas: {summary['total_strategies']}")
        print(f"└─ Estratégias Lucrativas: {summary['profitable_strategies']}")
        print()
        
        # Top 3 estratégias
        sorted_strategies = sorted(report['strategies'].items(), key=lambda x: x[1]['roi'], reverse=True)
        
        print("🏆 TOP 3 ESTRATÉGIAS:")
        print()
        
        for i, (name, metrics) in enumerate(sorted_strategies[:3]):
            print(f"{i+1}. {name}")
            print(f"   ├─ ROI: {metrics['roi']:.2f}%")
            print(f"   ├─ Saldo Final: ${metrics['final_balance']:.2f}")
            print(f"   ├─ Win Rate: {metrics['win_rate']:.1f}%")
            print(f"   ├─ Total Trades: {metrics['total_trades']}")
            print(f"   ├─ Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   ├─ Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"   ├─ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   ├─ Leverage Médio: {metrics['avg_leverage']:.2f}x")
            print(f"   ├─ Trades/Mês: {metrics['trades_per_month']:.1f}")
            print(f"   ├─ Maior Ganho: ${metrics['max_win']:.2f}")
            print(f"   └─ Maior Perda: ${metrics['max_loss']:.2f}")
            print()
        
        # Análise detalhada da melhor estratégia
        best_name, best_metrics = sorted_strategies[0]
        print("📈 ANÁLISE DETALHADA - MELHOR ESTRATÉGIA:")
        print()
        print(f"────────────────────────────────────────────────────────────")
        print(f"📋 {best_name}")
        print(f"────────────────────────────────────────────────────────────")
        print("💰 Performance Financeira:")
        print(f"   ├─ ROI: {best_metrics['roi']:.2f}%")
        print(f"   ├─ PnL Total: ${best_metrics['total_pnl']:.2f}")
        print(f"   ├─ Saldo Final: ${best_metrics['final_balance']:.2f}")
        print(f"   └─ Profit Factor: {best_metrics['profit_factor']:.2f}")
        print("📊 Estatísticas de Trading:")
        print(f"   ├─ Total Trades: {best_metrics['total_trades']}")
        print(f"   ├─ Win Rate: {best_metrics['win_rate']:.1f}%")
        print(f"   ├─ Trades/Mês: {best_metrics['trades_per_month']:.1f}")
        print(f"   └─ Leverage Médio: {best_metrics['avg_leverage']:.2f}x")
        print("⚖️ Risk Management:")
        print(f"   ├─ Max Drawdown: {best_metrics['max_drawdown']:.2f}%")
        print(f"   ├─ Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        print(f"   ├─ Ganho Médio: ${best_metrics['avg_win']:.2f}")
        print(f"   ├─ Perda Média: ${best_metrics['avg_loss']:.2f}")
        print(f"   ├─ Maior Ganho: ${best_metrics['max_win']:.2f}")
        print(f"   └─ Maior Perda: ${best_metrics['max_loss']:.2f}")
        print()
        
        # Recomendações
        print("=" * 80)
        print(" 🎯 RECOMENDAÇÕES PARA 100%+ ROI:")
        if best_metrics['roi'] >= TARGET_ROI:
            print(f"✅ META ALCANÇADA! Usar estratégia: {best_name}")
            print(f"├─ ROI esperado: {best_metrics['roi']:.2f}% ao ano")
            print(f"├─ Leverage médio: {best_metrics['avg_leverage']:.2f}x")
            print(f"├─ Win Rate: {best_metrics['win_rate']:.1f}%")
            print(f"└─ Max Drawdown: {best_metrics['max_drawdown']:.2f}%")
        else:
            print(f"❌ Meta não alcançada. Melhor resultado: {best_metrics['roi']:.2f}%")
            print("🔧 Sugestões de otimização:")
            print("├─ Aumentar leverage em sinais de alta confiança")
            print("├─ Adicionar mais ativos de alta volatilidade")
            print("├─ Refinar filtros de qualidade dos sinais")
            print("└─ Implementar trailing stops para maximizar ganhos")
        print("=" * 80)
        
        # Salva relatório
        with open('trading_futuro100_results.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

def main():
    """Função principal para teste do sistema 100% ROI"""
    logger.info("🚀 Iniciando TradingFuturo100 - Sistema para 100%+ ROI")
    logger.info(f"📋 Configurações:")
    logger.info(f"   ├─ Banca Inicial: ${INITIAL_BALANCE}")
    logger.info(f"   ├─ Position Size Base: ${BASE_POSITION_SIZE}")
    logger.info(f"   ├─ Position Size Max: ${MAX_POSITION_SIZE}")
    logger.info(f"   ├─ Leverage Máximo: {MAX_LEVERAGE}x")
    logger.info(f"   ├─ Meta ROI: {TARGET_ROI}%")
    logger.info(f"   ├─ Ativos: {len(ASSETS)} criptomoedas")
    logger.info(f"   └─ Estratégias: 5 swing trading")
    
    try:
        engine = TradingEngine()
        logger.info("Iniciando backtest completo...")
        engine.run_full_backtest()
        
        logger.info("Gerando relatório...")
        engine.print_report()
        
        logger.info("✅ Backtest concluído com sucesso!")
        logger.info("📄 Relatório salvo em: trading_futuro100_results.json")
        
    except Exception as e:
        logger.error(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
