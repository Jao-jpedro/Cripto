#!/usr/bin/env python3
"""
🚀 TRADINGFUTURO100 - SISTEMA SIMPLIFICADO PARA TESTES 🚀
Focado em gerar trades para alcançar 100%+ ROI
"""

import sys
import logging
import pandas as pd
import numpy as np
import ccxt
import ta
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import json

# === CONFIGURAÇÕES ===
INITIAL_BALANCE = 10.0  # $10 inicial
BASE_POSITION_SIZE = 3.0  # $3 por posição
MAX_POSITION_SIZE = 8.0  # $8 max por posição
MAX_LEVERAGE = 5.0  # 5x leverage máximo
COMMISSION_RATE = 0.0001  # 0.01%
TARGET_ROI = 100.0  # Meta: 100%+

# Lista de ativos principais para teste com dados reais
ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'DOGE/USDT'
]

TIMEFRAMES = ['4h', '1d']  # Multi-timeframe

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingFuturo100')

@dataclass
class TradeResult:
    strategy_name: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    pnl: float
    pnl_pct: float
    leverage: float

@dataclass 
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    leverage: float

class DataManager:
    """Gerenciador de dados REAIS da Binance (1 ano)"""
    
    def __init__(self):
        # Configuração da Binance sem autenticação (dados públicos apenas)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'sandbox': False,  # Usar dados reais
        })
    
    def get_data(self, symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Busca dados REAIS da Binance dos últimos 365 dias"""
        try:
            logger.info(f"🔍 Buscando dados REAIS da Binance: {symbol} {timeframe} ({days} dias)")
            
            # Calcular timestamp de início (1 ano atrás)
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Buscar dados históricos reais da Binance
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not ohlcv:
                logger.error(f"❌ Nenhum dado retornado para {symbol}")
                return pd.DataFrame()
            
            # Converter para DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Converter preços para float
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float) 
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Calcular indicadores técnicos
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Remover NaN
            df = df.dropna()
            
            logger.info(f"✅ {symbol} {timeframe}: {len(df)} barras de dados REAIS carregadas")
            logger.info(f"📈 Período: {df.index[0]} até {df.index[-1]}")
            logger.info(f"💰 Preço inicial: ${df['close'].iloc[0]:.2f} | Final: ${df['close'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados REAIS para {symbol}: {e}")
            return pd.DataFrame()

class SimpleStrategy:
    """Estratégia simplificada para testes"""
    
    def __init__(self, name: str):
        self.name = name
        self.balance = INITIAL_BALANCE
        self.positions = {}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """🎯 Geração de sinais MUITO SIMPLES para garantir trades"""
        signals = []
        
        for i in range(20, len(df)):
            current = df.iloc[i]
            
            # SINAIS SUPER SIMPLES - apenas para gerar trades
            
            # Sinal LONG: preço sobe acima EMA21 + volume OK
            if (current['close'] > current['ema_21'] and
                current['rsi'] > 40 and current['rsi'] < 80 and
                current['volume_ratio'] > 1.0):
                
                signals.append({
                    'time': current.name,
                    'action': 'buy',
                    'price': current['close'],
                    'confidence': 0.7,
                    'stop_loss': current['close'] * 0.95,
                    'take_profit': current['close'] * 1.10
                })
            
            # Sinal SHORT: preço cai abaixo EMA21 + volume OK
            elif (current['close'] < current['ema_21'] and
                  current['rsi'] > 20 and current['rsi'] < 60 and
                  current['volume_ratio'] > 1.0):
                
                signals.append({
                    'time': current.name,
                    'action': 'sell',
                    'price': current['close'],
                    'confidence': 0.7,
                    'stop_loss': current['close'] * 1.05,
                    'take_profit': current['close'] * 0.90
                })
        
        logger.info(f"📊 {self.name}: {len(signals)} sinais gerados para {symbol}")
        return signals
    
    def backtest(self, df: pd.DataFrame, symbol: str) -> List[TradeResult]:
        """Backtest simples"""
        signals = self.generate_signals(df, symbol)
        trades = []
        
        for signal in signals:
            if signal['action'] == 'buy':
                # Simula entrada LONG
                entry_price = signal['price']
                confidence = signal['confidence']
                
                # Position sizing simples
                position_value = BASE_POSITION_SIZE * confidence
                leverage = min(3.0, 1 + confidence * 2)
                quantity = (position_value * leverage) / entry_price
                
                # Simula saída após algumas barras (take profit ou stop loss)
                entry_time = signal['time']
                
                # Encontra a próxima barra para simular saída
                try:
                    entry_idx = df.index.get_loc(entry_time)
                    if entry_idx + 10 < len(df):
                        exit_idx = entry_idx + 10  # Saída após 10 barras
                        exit_time = df.index[exit_idx]
                        exit_price = df.loc[exit_time, 'close']
                        
                        # Calcula P&L
                        price_change = (exit_price - entry_price) / entry_price
                        pnl = quantity * entry_price * price_change * leverage
                        
                        trades.append(TradeResult(
                            strategy_name=self.name,
                            symbol=symbol,
                            entry_time=entry_time,
                            exit_time=exit_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            side='LONG',
                            quantity=quantity,
                            pnl=pnl,
                            pnl_pct=price_change * leverage * 100,
                            leverage=leverage
                        ))
                except:
                    continue
                    
            elif signal['action'] == 'sell':
                # Simula entrada SHORT (similar à LONG mas invertida)
                entry_price = signal['price']
                confidence = signal['confidence']
                
                position_value = BASE_POSITION_SIZE * confidence
                leverage = min(3.0, 1 + confidence * 2)
                quantity = (position_value * leverage) / entry_price
                
                entry_time = signal['time']
                
                try:
                    entry_idx = df.index.get_loc(entry_time)
                    if entry_idx + 10 < len(df):
                        exit_idx = entry_idx + 10
                        exit_time = df.index[exit_idx]
                        exit_price = df.loc[exit_time, 'close']
                        
                        # P&L para SHORT (invertido)
                        price_change = (entry_price - exit_price) / entry_price
                        pnl = quantity * entry_price * price_change * leverage
                        
                        trades.append(TradeResult(
                            strategy_name=self.name,
                            symbol=symbol,
                            entry_time=entry_time,
                            exit_time=exit_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            side='SHORT',
                            quantity=quantity,
                            pnl=pnl,
                            pnl_pct=price_change * leverage * 100,
                            leverage=leverage
                        ))
                except:
                    continue
        
        return trades

class BacktestEngine:
    """Engine de backtest simplificado"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.results = {}
    
    def run_backtest(self):
        """Executa backtest em todos os ativos"""
        logger.info("🚀 Iniciando backtest SIMPLIFICADO...")
        
        strategy = SimpleStrategy("Estratégia Simples")
        
        for asset in ASSETS:
            logger.info(f"🔍 Processando {asset}...")
            
            for timeframe in TIMEFRAMES:
                try:
                    df = self.data_manager.get_data(asset, timeframe)
                    if df.empty:
                        continue
                    
                    trades = strategy.backtest(df, asset)
                    
                    if asset not in self.results:
                        self.results[asset] = {}
                    
                    self.results[asset][f"{timeframe}"] = trades
                    
                    logger.info(f"✅ {asset} {timeframe}: {len(trades)} trades")
                    
                except Exception as e:
                    logger.error(f"❌ Erro em {asset} {timeframe}: {e}")
                    continue
    
    def calculate_metrics(self, trades: List[TradeResult]) -> Dict:
        """Calcula métricas de performance"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'roi': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_leverage': 0,
                'final_balance': INITIAL_BALANCE,
                'trades_per_month': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        final_balance = INITIAL_BALANCE + total_pnl
        roi = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        # Drawdown
        balance_curve = [INITIAL_BALANCE]
        for trade in trades:
            balance_curve.append(balance_curve[-1] + trade.pnl)
        
        peak = balance_curve[0]
        max_dd = 0
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100,
            'total_pnl': total_pnl,
            'roi': roi,
            'profit_factor': sum(w.pnl for w in wins) / abs(sum(l.pnl for l in losses)) if losses else float('inf') if wins else 0,
            'max_drawdown': max_dd * 100,
            'sharpe_ratio': np.mean([t.pnl_pct for t in trades]) / np.std([t.pnl_pct for t in trades]) if len(trades) > 1 else 0,
            'avg_leverage': np.mean([t.leverage for t in trades]),
            'final_balance': final_balance,
            'trades_per_month': len(trades) / 12.0,
            'max_win': max([w.pnl for w in wins]) if wins else 0,
            'max_loss': min([l.pnl for l in losses]) if losses else 0
        }
    
    def generate_report(self):
        """Gera relatório consolidado"""
        all_trades = []
        
        # Consolida todos os trades
        for asset_results in self.results.values():
            for timeframe_trades in asset_results.values():
                all_trades.extend(timeframe_trades)
        
        if not all_trades:
            logger.error("❌ NENHUM TRADE ENCONTRADO!")
            return
        
        metrics = self.calculate_metrics(all_trades)
        
        print("=" * 80)
        print(" 🚀 TRADINGFUTURO100 - TESTE COM DADOS REAIS DA BINANCE")
        print("=" * 80)
        print()
        print("📊 PERFORMANCE GERAL:")
        print(f"   ├─ Banca Inicial: ${INITIAL_BALANCE:.2f}")
        print(f"   ├─ Saldo Final: ${metrics['final_balance']:.2f}")
        print(f"   ├─ ROI: {metrics['roi']:.2f}%")
        print(f"   ├─ P&L Total: ${metrics['total_pnl']:.2f}")
        print(f"   └─ Multiplicador: {metrics['final_balance']/INITIAL_BALANCE:.1f}x")
        print()
        print("📈 ESTATÍSTICAS DE TRADING:")
        print(f"   ├─ Total de Trades: {metrics['total_trades']}")
        print(f"   ├─ Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   ├─ Trades/Mês: {metrics['trades_per_month']:.1f}")
        print(f"   ├─ Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   └─ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print()
        print("⚖️ GESTÃO DE RISCO:")
        print(f"   ├─ Leverage Médio: {metrics['avg_leverage']:.2f}x")
        print(f"   ├─ Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   ├─ Maior Ganho: ${metrics['max_win']:.2f}")
        print(f"   └─ Maior Perda: ${metrics['max_loss']:.2f}")
        print()
        print("🎯 RESULTADO FINAL:")
        if metrics['roi'] >= TARGET_ROI:
            print(f"   🎉 META DE {TARGET_ROI}%+ ROI ALCANÇADA!")
            print(f"   ✅ Superação: {metrics['roi'] - TARGET_ROI:.2f}% acima da meta")
        else:
            print(f"   ❌ Meta não alcançada")
            print(f"   📉 Faltaram: {TARGET_ROI - metrics['roi']:.2f}% para atingir {TARGET_ROI}%")
        print("=" * 80)

def main():
    """Função principal"""
    try:
        logger.info("🚀 Iniciando TradingFuturo100 - TESTE SIMPLIFICADO")
        
        engine = BacktestEngine()
        engine.run_backtest()
        engine.generate_report()
        
        logger.info("✅ Teste concluído!")
        
    except Exception as e:
        logger.error(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
