#!/usr/bin/env python3
"""
TradingFuturo100 - Sistema de Trading Completo
Versão com todas as configurações do trading.py original
Executado com reinicialização automática para máxima estabilidade
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurações da Hyperliquid (baseadas no trading.py)
def get_hyperliquid_config():
    """Retorna configuração da Hyperliquid baseada no trading.py"""
    dex_timeout = int(os.getenv("DEX_TIMEOUT_MS", "5000"))
    _wallet_env = os.getenv("WALLET_ADDRESS")
    _priv_env = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    
    config = {
        "walletAddress": _wallet_env or "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
        "privateKey": _priv_env or "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872",
        "enableRateLimit": True,
        "timeout": dex_timeout,
        "options": {"timeout": dex_timeout},
        "sandbox": False
    }
    
    return config

class DataManager:
    """Gerenciador de dados - conecta à Hyperliquid mas opera em modo simulação"""
    
    def __init__(self, live_trading=False):
        self.live_trading = live_trading
        self.exchange = None
        self.connection_status = "OFFLINE"
        
        try:
            if live_trading:
                config = get_hyperliquid_config()
                self.exchange = ccxt.hyperliquid(config)
                print(f"🔗 Conectado à Hyperliquid - Wallet: {config['walletAddress'][:10]}...")
                self.connection_status = "HYPERLIQUID"
            else:
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'sandbox': False
                })
                print("📊 Modo simulação - usando Binance para dados históricos")
                self.connection_status = "BINANCE_SIM"
                
        except Exception as e:
            print(f"❌ Erro na conexão: {e}")
            self.connection_status = "ERROR"
    
    def test_connection(self) -> bool:
        """Testa conexão com a exchange"""
        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker('BTC/USDT')
                return ticker and 'last' in ticker
            return False
        except:
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str = '4h', limit: int = 1000, include_current: bool = False) -> pd.DataFrame:
        """Busca dados históricos, incluindo candle atual se especificado"""
        try:
            if self.live_trading:
                hl_symbol = symbol.replace('USDT', 'USD')
                ohlcv = self.exchange.fetch_ohlcv(hl_symbol, timeframe, limit=limit)
            else:
                fetch_limit = limit + 1 if include_current else limit
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao buscar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Busca preço atual"""
        try:
            if self.live_trading:
                hl_symbol = symbol.replace('USDT', 'USD')
                ticker = self.exchange.fetch_ticker(hl_symbol)
            else:
                ticker = self.exchange.fetch_ticker(symbol)
            
            return float(ticker['last'])
            
        except Exception as e:
            return 0.0

class SimpleStrategy:
    """Estratégia simplificada baseada em EMA21 + RSI + Volume"""
    
    def __init__(self):
        self.name = "EMA21_RSI_Volume_Enhanced"
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos"""
        df = df.copy()
        
        # EMA 21
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        
        # RSI 14
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Volume média 20 períodos
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # ATR para volatilidade
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais de compra/venda"""
        df = df.copy()
        df = self.calculate_indicators(df)
        
        # Condições mais permissivas para LONG
        long_conditions = (
            (df['close'] > df['ema21']) &  # Preço acima da EMA21
            (df['rsi'] < 75) &  # RSI não muito sobrecomprado
            (df['rsi'] > 25) &  # RSI não muito sobrevendido
            (df['volume_ratio'] > 0.8)  # Volume ligeiramente acima da média
        )
        
        # Condições mais permissivas para SHORT
        short_conditions = (
            (df['close'] < df['ema21']) &  # Preço abaixo da EMA21
            (df['rsi'] > 25) &  # RSI não muito sobrevendido
            (df['rsi'] < 75) &  # RSI não muito sobrecomprado
            (df['volume_ratio'] > 0.8)  # Volume ligeiramente acima da média
        )
        
        # Condições para SAÍDA (mais conservadoras)
        exit_long_conditions = (
            (df['close'] < df['ema21']) |  # Preço abaixo da EMA21
            (df['rsi'] > 80)  # RSI muito sobrecomprado
        )
        
        exit_short_conditions = (
            (df['close'] > df['ema21']) |  # Preço acima da EMA21
            (df['rsi'] < 20)  # RSI muito sobrevendido
        )
        
        df['signal'] = 0
        df.loc[long_conditions, 'signal'] = 1  # Compra
        df.loc[short_conditions, 'signal'] = -1  # Venda
        df.loc[exit_long_conditions, 'signal'] = -2  # Sair de LONG
        df.loc[exit_short_conditions, 'signal'] = 2  # Sair de SHORT
        
        return df

class Position:
    """Classe para gerenciar posições"""
    
    def __init__(self):
        self.side = None
        self.size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
        self.leverage = 10
    
    def open_position(self, side: str, size: float, price: float, timestamp):
        self.side = side
        self.size = size
        self.entry_price = price
        self.entry_time = timestamp
        self.unrealized_pnl = 0.0
    
    def close_position(self):
        self.side = None
        self.size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
    
    def update_pnl(self, current_price: float):
        if self.side and self.size > 0:
            price_diff = current_price - self.entry_price
            if self.side == 'short':
                price_diff = -price_diff
            
            self.unrealized_pnl = (price_diff / self.entry_price) * 100 * self.leverage
    
    def is_open(self) -> bool:
        return self.side is not None and self.size > 0

class BacktestEngine:
    """Engine de backtesting"""
    
    def __init__(self, initial_capital: float = 10.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = Position()
        self.trades = []
        self.equity_curve = []
        
    def execute_trade(self, signal: int, price: float, timestamp, symbol: str):
        """Executa uma operação baseada no sinal"""
        trade_size = 1.0
        
        if signal == 1 and not self.position.is_open():  # Abrir LONG
            self.position.open_position('long', trade_size, price, timestamp)
            print(f"🟢 LONG {symbol} @ {price:.6f} em {timestamp}")
            
        elif signal == -1 and not self.position.is_open():  # Abrir SHORT
            self.position.open_position('short', trade_size, price, timestamp)
            print(f"🔴 SHORT {symbol} @ {price:.6f} em {timestamp}")
            
        elif signal == -2 and self.position.is_open() and self.position.side == 'long':  # Fechar LONG
            pnl_pct = self.position.unrealized_pnl
            pnl_dollar = (pnl_pct / 100) * self.capital
            self.capital += pnl_dollar
            
            trade_record = {
                'symbol': symbol,
                'side': 'long',
                'entry_price': self.position.entry_price,
                'exit_price': price,
                'entry_time': self.position.entry_time,
                'exit_time': timestamp,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar
            }
            self.trades.append(trade_record)
            
            print(f"❌ Fechou LONG {symbol} @ {price:.6f} | PnL: {pnl_pct:.2f}% (${pnl_dollar:.2f})")
            self.position.close_position()
            
        elif signal == 2 and self.position.is_open() and self.position.side == 'short':  # Fechar SHORT
            pnl_pct = self.position.unrealized_pnl
            pnl_dollar = (pnl_pct / 100) * self.capital
            self.capital += pnl_dollar
            
            trade_record = {
                'symbol': symbol,
                'side': 'short',
                'entry_price': self.position.entry_price,
                'exit_price': price,
                'entry_time': self.position.entry_time,
                'exit_time': timestamp,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar
            }
            self.trades.append(trade_record)
            
            print(f"❌ Fechou SHORT {symbol} @ {price:.6f} | PnL: {pnl_pct:.2f}% (${pnl_dollar:.2f})")
            self.position.close_position()
    
    def update_equity(self, current_price: float, timestamp):
        current_equity = self.capital
        
        if self.position.is_open():
            self.position.update_pnl(current_price)
            unrealized_dollar = (self.position.unrealized_pnl / 100) * self.capital
            current_equity += unrealized_dollar
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'capital': self.capital,
            'unrealized_pnl': self.position.unrealized_pnl if self.position.is_open() else 0
        })
    
    def get_performance_stats(self) -> Dict:
        total_trades = len(self.trades)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        avg_win = 0
        avg_loss = 0
        
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            winning_trades = len(df_trades[df_trades['pnl_dollar'] > 0])
            losing_trades = len(df_trades[df_trades['pnl_dollar'] < 0])
            total_pnl = df_trades['pnl_dollar'].sum()
            avg_win = df_trades[df_trades['pnl_dollar'] > 0]['pnl_dollar'].mean() if winning_trades > 0 else 0
            avg_loss = df_trades[df_trades['pnl_dollar'] < 0]['pnl_dollar'].mean() if losing_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        roi = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'final_capital': self.capital,
            'initial_capital': self.initial_capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

class TradingFuturo100:
    """Sistema principal de trading"""
    
    def __init__(self, live_trading: bool = False):
        self.live_trading = live_trading
        self.data_manager = DataManager(live_trading)
        self.strategy = SimpleStrategy()
        self.backtest_engine = BacktestEngine()
        
        # Todos os símbolos do trading.py original
        self.symbols = [
            'BTC/USDT',   # Bitcoin
            'ETH/USDT',   # Ethereum  
            'SOL/USDT',   # Solana
            'XRP/USDT',   # Ripple
            'DOGE/USDT',  # Dogecoin
            'AVAX/USDT',  # Avalanche
            'ADA/USDT',   # Cardano
            'BNB/USDT',   # Binance Coin
            'SUI/USDT',   # Sui
            'ENA/USDT',   # Ethena
            'LINK/USDT',  # Chainlink
            'AAVE/USDT',  # Aave
            'CRV/USDT',   # Curve
            'LTC/USDT',   # Litecoin
            'NEAR/USDT',  # Near Protocol
            'WLD/USDT',   # Worldcoin
            'HYPE/USDT',  # Hype
            'PUMP/USDT',  # Pump
            'AVNT/USDT'   # Avant
        ]
        
        self.trades_file = f"trading_live_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.last_signals = {}
        
        print(f"🚀 TradingFuturo100 COMPLETO inicializado")
        print(f"📊 Modo: {'LIVE TRADING' if live_trading else 'SIMULAÇÃO'}")
        print(f"💰 Capital inicial: ${self.backtest_engine.initial_capital}")
        print(f"🎯 Meta: >100% ROI")
        print(f"📈 {len(self.symbols)} ativos monitorados")
        print(f"💾 Log de trades: {self.trades_file}")
    
    def save_trade_log(self, trade_data: Dict):
        """Salva trade individual em tempo real"""
        try:
            try:
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {
                    'start_time': datetime.now().isoformat(),
                    'trades': [],
                    'performance': {
                        'total_trades': 0,
                        'roi': 0.0,
                        'capital': self.backtest_engine.initial_capital
                    }
                }
            
            data['trades'].append(trade_data)
            data['performance']['total_trades'] = len(data['trades'])
            data['performance']['capital'] = self.backtest_engine.capital
            data['performance']['roi'] = ((self.backtest_engine.capital - self.backtest_engine.initial_capital) / self.backtest_engine.initial_capital) * 100
            data['last_update'] = datetime.now().isoformat()
            
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Erro ao salvar trade log: {e}")
    
    def check_signals_and_trade(self, symbol: str) -> bool:
        """Verifica sinais para um símbolo e executa trade se necessário"""
        try:
            check_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{check_time}] 🔍 {symbol:<12} ", end='')
            
            df = self.data_manager.get_historical_data(symbol, '4h', 100, include_current=True)
            if df.empty:
                print("❌ Sem dados")
                return False
            
            df_signals = self.strategy.generate_signals(df)
            
            latest_timestamp = df_signals.index[-1]
            latest_signal = df_signals['signal'].iloc[-1]
            latest_price = df_signals['close'].iloc[-1]
            latest_rsi = df_signals['rsi'].iloc[-1] if 'rsi' in df_signals.columns else 0
            
            print(f"${latest_price:>8.4f} | RSI:{latest_rsi:>5.1f} | Sinal:{latest_signal:>2d} ", end='')
            
            last_cached_signal = self.last_signals.get(symbol, 0)
            
            if latest_signal != 0 and latest_signal != last_cached_signal:
                print(f"🚨 TRADE!")
                print(f"    🎯 EXECUTANDO: {symbol} sinal {latest_signal} @ ${latest_price:.6f}")
                
                old_capital = self.backtest_engine.capital
                old_trade_count = len(self.backtest_engine.trades)
                
                self.backtest_engine.execute_trade(latest_signal, latest_price, latest_timestamp, symbol)
                
                if len(self.backtest_engine.trades) > old_trade_count:
                    new_trade = self.backtest_engine.trades[-1]
                    
                    trade_log_data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'signal': int(latest_signal),
                        'price': float(latest_price),
                        'candle_time': latest_timestamp.isoformat(),
                        'trade_data': new_trade,
                        'capital_before': float(old_capital),
                        'capital_after': float(self.backtest_engine.capital)
                    }
                    
                    self.save_trade_log(trade_log_data)
                    self.last_signals[symbol] = latest_signal
                    return True
            else:
                print("✅")
            
            self.last_signals[symbol] = latest_signal
            self.backtest_engine.update_equity(latest_price, latest_timestamp)
            
            return False
            
        except Exception as e:
            print(f"❌ Erro: {str(e)[:30]}")
            return False
    
    def run_live_monitoring(self):
        """Executa monitoramento contínuo com candle em aberto"""
        print("\n" + "="*80)
        print("🔴 MODO MONITORAMENTO CONTÍNUO - TODOS OS ATIVOS")
        print("="*80)
        print("📊 Verificando movimentações em tempo real (incluindo candle aberto)")
        print("💾 Salvando trades automaticamente quando ocorrerem")
        print("⚠️ Esta é uma simulação - nenhuma ordem real será executada")
        print(f"📝 Log de trades: {self.trades_file}")
        print(f"🎯 {len(self.symbols)} ativos monitorados")
        print("⏰ Execução limitada a 8 minutos (reiniciará automaticamente)")
        print("="*80)
        
        start_time = time.time()
        max_duration = 8 * 60  # 8 minutos
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                elapsed_time = time.time() - start_time
                
                if elapsed_time >= max_duration:
                    print(f"\n⏰ Limite de tempo atingido. Finalizando execução...")
                    break
                
                trades_this_cycle = 0
                print(f"\n{'='*80}")
                print(f"🔄 CICLO #{cycle_count} | {datetime.now().strftime('%H:%M:%S')} | ⏱️ {elapsed_time/60:.1f}min")
                print(f"{'='*80}")
                
                # Teste conexão
                if self.data_manager.test_connection():
                    print(f"🟢 Conexão: {self.data_manager.connection_status}")
                else:
                    print(f"🔴 Conexão: {self.data_manager.connection_status} - PROBLEMA")
                
                for i, symbol in enumerate(self.symbols, 1):
                    try:
                        print(f"[{i:2d}/{len(self.symbols)}] ", end='')
                        if self.check_signals_and_trade(symbol):
                            trades_this_cycle += 1
                    except Exception as e:
                        print(f"❌ Erro {symbol}: {e}")
                
                roi = ((self.backtest_engine.capital - self.backtest_engine.initial_capital) / self.backtest_engine.initial_capital) * 100
                total_trades = len(self.backtest_engine.trades)
                
                print(f"\n📊 RESUMO CICLO {cycle_count}:")
                print(f"💰 Capital: ${self.backtest_engine.capital:.2f} | 📈 ROI: {roi:.2f}% | 🔢 Trades: {total_trades}")
                
                if trades_this_cycle > 0:
                    print(f"🎉 {trades_this_cycle} NOVOS TRADES EXECUTADOS!")
                
                remaining_time = max_duration - elapsed_time
                print(f"⏰ Tempo restante: {remaining_time/60:.1f} min")
                
                sleep_time = min(120, remaining_time - 10)
                if sleep_time > 0:
                    print(f"💤 Aguardando {sleep_time/60:.1f} min...")
                    for i in range(int(sleep_time)):
                        if i % 30 == 0 and i > 0:
                            print(f"  ⏳ {(sleep_time-i)/60:.1f} min restantes...")
                        time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Interrompido após {cycle_count} ciclos")
        
        print(f"📊 Execução finalizada: {cycle_count} ciclos, {elapsed_time/60:.1f} min")
        self.print_results()
    
    def print_results(self):
        """Imprime resultados"""
        stats = self.backtest_engine.get_performance_stats()
        
        print("\n" + "="*60)
        print("📊 RESULTADOS FINAIS")
        print("="*60)
        
        print(f"💰 Capital inicial: ${stats['initial_capital']:.2f}")
        print(f"💰 Capital final: ${stats['final_capital']:.2f}")
        print(f"📈 ROI: {stats['roi']:.2f}%")
        print(f"💵 PnL total: ${stats['total_pnl']:.2f}")
        print()
        print(f"🔢 Total de trades: {stats['total_trades']}")
        print(f"✅ Trades vencedores: {stats['winning_trades']}")
        print(f"❌ Trades perdedores: {stats['losing_trades']}")
        print(f"🎯 Taxa de acerto: {stats['win_rate']:.1f}%")
        
        if stats['winning_trades'] > 0:
            print(f"💚 Ganho médio: ${stats['avg_win']:.2f}")
        if stats['losing_trades'] > 0:
            print(f"💔 Perda média: ${stats['avg_loss']:.2f}")
        
        print(f"\n🎯 Meta 100% ROI: {'✅ ATINGIDA!' if stats['roi'] >= 100 else '❌ Não atingida'}")
        
        if stats['roi'] >= 100:
            print(f"🚀 SUCESSO! ROI de {stats['roi']:.2f}% superou a meta!")
    
    def run_backtest(self):
        """Executa backtest histórico"""
        print("\n" + "="*60)
        print("🔄 INICIANDO BACKTEST HISTÓRICO")
        print("="*60)
        
        all_data = {}
        
        for symbol in self.symbols:
            print(f"📥 Buscando dados para {symbol}...")
            df = self.data_manager.get_historical_data(symbol, '4h', 1000)
            
            if not df.empty:
                all_data[symbol] = df
                print(f"✅ {symbol}: {len(df)} candles")
            else:
                print(f"❌ {symbol}: Falha")
        
        if not all_data:
            print("❌ Nenhum dado carregado")
            return
        
        common_start = max(df.index.min() for df in all_data.values())
        common_end = min(df.index.max() for df in all_data.values())
        
        print(f"📅 Período: {common_start} até {common_end}")
        print(f"⏱️ Duração: {(common_end - common_start).days} dias")
        
        for symbol in all_data:
            all_data[symbol] = all_data[symbol].loc[common_start:common_end]
        
        signals_data = {}
        for symbol, df in all_data.items():
            print(f"🔍 Gerando sinais para {symbol}...")
            df_with_signals = self.strategy.generate_signals(df)
            signals_data[symbol] = df_with_signals
        
        print(f"\n🎬 Executando backtest...")
        
        all_timestamps = set()
        for df in signals_data.values():
            all_timestamps.update(df.index)
        
        all_timestamps = sorted(all_timestamps)
        
        for timestamp in all_timestamps:
            for symbol, df in signals_data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    signal = row['signal']
                    price = row['close']
                    
                    if signal != 0:
                        self.backtest_engine.execute_trade(signal, price, timestamp, symbol)
                    
                    self.backtest_engine.update_equity(price, timestamp)
        
        self.print_results()

def main():
    """Função principal"""
    live_trading = int(os.getenv('LIVE_TRADING', '0')) == 1
    
    print("🚀 TradingFuturo100 COMPLETO - Sistema com TODOS os ativos")
    print("="*80)
    
    if live_trading:
        print("🔴 LIVE_TRADING=1 - MONITORAMENTO CONTÍNUO")
        print("📊 Incluindo candle em aberto da Binance")
        print("💾 Salvando trades automaticamente")
    else:
        print("📊 LIVE_TRADING=0 - BACKTEST HISTÓRICO")
    
    # Sempre em modo simulação (sem ordens reais)
    system = TradingFuturo100(live_trading=False)
    
    if live_trading:
        system.run_live_monitoring()
    else:
        system.run_backtest()

if __name__ == "__main__":
    main()
