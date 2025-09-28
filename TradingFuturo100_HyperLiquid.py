#!/usr/bin/env python3
"""
TradingFuturo100 - Sistema de Trading com configurações Hyperliquid
Versão adaptada para usar as configurações do trading.py com Hyperliquid
Executa em modo simulação local (LIVE_TRADING=0)
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

# Configurações da Hyperliquid (importadas do trading.py)
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
        "sandbox": False  # Usar testnet se necessário
    }
    
    return config

class DataManager:
    """Gerenciador de dados - conecta à Hyperliquid mas opera em modo simulação"""
    
    def __init__(self, live_trading=False):
        self.live_trading = live_trading
        self.exchange = None
        
        if live_trading:
            # Configuração para Hyperliquid
            config = get_hyperliquid_config()
            self.exchange = ccxt.hyperliquid(config)
            print(f"🔗 Conectado à Hyperliquid - Wallet: {config['walletAddress'][:10]}...")
        else:
            # Modo simulação - usar Binance para dados históricos
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'sandbox': False
            })
            print("📊 Modo simulação - usando Binance para dados históricos")
    
    def get_historical_data(self, symbol: str, timeframe: str = '4h', limit: int = 1000, include_current: bool = False) -> pd.DataFrame:
        """Busca dados históricos, incluindo candle atual se especificado"""
        try:
            if self.live_trading:
                # Para Hyperliquid, ajustar símbolo se necessário
                hl_symbol = symbol.replace('USDT', 'USD')
                ohlcv = self.exchange.fetch_ohlcv(hl_symbol, timeframe, limit=limit)
            else:
                # Modo simulação - usar Binance
                # Para incluir candle atual, buscar um a mais
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
            print(f"❌ Erro ao buscar preço de {symbol}: {e}")
            return 0.0

class SimpleStrategy:
    """Estratégia simplificada baseada em EMA21 + RSI + Volume"""
    
    def __init__(self):
        self.name = "EMA21_RSI_Volume"
    
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
        self.side = None  # 'long' ou 'short'
        self.size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
        self.leverage = 10  # Alavancagem padrão
    
    def open_position(self, side: str, size: float, price: float, timestamp):
        """Abre uma nova posição"""
        self.side = side
        self.size = size
        self.entry_price = price
        self.entry_time = timestamp
        self.unrealized_pnl = 0.0
    
    def close_position(self):
        """Fecha a posição atual"""
        self.side = None
        self.size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
    
    def update_pnl(self, current_price: float):
        """Atualiza PnL não realizado"""
        if self.side and self.size > 0:
            price_diff = current_price - self.entry_price
            if self.side == 'short':
                price_diff = -price_diff
            
            self.unrealized_pnl = (price_diff / self.entry_price) * 100 * self.leverage
    
    def is_open(self) -> bool:
        """Verifica se há posição aberta"""
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
        trade_size = 1.0  # Tamanho fixo por operação
        
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
        """Atualiza curva de equity"""
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
        """Calcula estatísticas de performance"""
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
        
        # Símbolos para trading (todos os ativos do trading.py)
        self.symbols = [
            'BTC/USDT',  # Bitcoin
            'ETH/USDT',  # Ethereum  
            'SOL/USDT',  # Solana
            'XRP/USDT',  # Ripple
            'DOGE/USDT', # Dogecoin
            'AVAX/USDT', # Avalanche
            'ADA/USDT',  # Cardano
            'BNB/USDT',  # Binance Coin
            'SUI/USDT',  # Sui
            'ENA/USDT',  # Ethena
            'LINK/USDT', # Chainlink
            'AAVE/USDT', # Aave
            'CRV/USDT',  # Curve
            'LTC/USDT',  # Litecoin
            'NEAR/USDT', # Near Protocol
            'WLD/USDT',  # Worldcoin
            'HYPE/USDT', # Hype
            'PUMP/USDT', # Pump
            'AVNT/USDT'  # Avant
        ]
        
        # Arquivo para salvar trades em tempo real
        self.trades_file = f"trading_live_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.last_signals = {}  # Cache dos últimos sinais por símbolo
        
        print(f"🚀 TradingFuturo100 inicializado")
        print(f"📊 Modo: {'LIVE TRADING' if live_trading else 'SIMULAÇÃO'}")
        print(f"💰 Capital inicial: ${self.backtest_engine.initial_capital}")
        print(f"🎯 Meta: >100% ROI")
        print(f"💾 Log de trades: {self.trades_file}")
    
    def run_backtest(self):
        """Executa backtest completo"""
        print("\n" + "="*60)
        print("🔄 INICIANDO BACKTEST")
        print("="*60)
        
        all_data = {}
        
        # Buscar dados para todos os símbolos
        for symbol in self.symbols:
            print(f"📥 Buscando dados para {symbol}...")
            df = self.data_manager.get_historical_data(symbol, '4h', 1000)
            
            if not df.empty:
                all_data[symbol] = df
                print(f"✅ {symbol}: {len(df)} candles carregados")
            else:
                print(f"❌ {symbol}: Falha ao carregar dados")
        
        if not all_data:
            print("❌ Nenhum dado carregado. Abortando backtest.")
            return
        
        # Encontrar período comum para todos os símbolos
        common_start = max(df.index.min() for df in all_data.values())
        common_end = min(df.index.max() for df in all_data.values())
        
        print(f"📅 Período do backtest: {common_start} até {common_end}")
        print(f"⏱️ Duração: {(common_end - common_start).days} dias")
        
        # Filtrar dados para período comum
        for symbol in all_data:
            all_data[symbol] = all_data[symbol].loc[common_start:common_end]
        
        # Gerar sinais para todos os símbolos
        signals_data = {}
        for symbol, df in all_data.items():
            print(f"🔍 Gerando sinais para {symbol}...")
            df_with_signals = self.strategy.generate_signals(df)
            signals_data[symbol] = df_with_signals
        
        # Execução do backtest
        print(f"\n🎬 Executando backtest...")
        
        # Criar um único DataFrame com todos os timestamps
        all_timestamps = set()
        for df in signals_data.values():
            all_timestamps.update(df.index)
        
        all_timestamps = sorted(all_timestamps)
        
        trade_count = 0
        for timestamp in all_timestamps:
            # Para cada timestamp, verificar sinais de todos os símbolos
            for symbol, df in signals_data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    signal = row['signal']
                    price = row['close']
                    
                    if signal != 0:
                        old_trade_count = len(self.backtest_engine.trades)
                        self.backtest_engine.execute_trade(signal, price, timestamp, symbol)
                        if len(self.backtest_engine.trades) > old_trade_count:
                            trade_count += 1
                    
                    # Atualizar equity com o último preço
                    self.backtest_engine.update_equity(price, timestamp)
        
        self.print_results()
    
    def print_results(self):
        """Imprime resultados do backtest"""
        stats = self.backtest_engine.get_performance_stats()
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DO BACKTEST")
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
        
        # Salvar resultados
        self.save_results(stats)
    
    def save_results(self, stats: Dict):
        """Salva resultados em arquivo JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy.name,
            'live_trading': self.live_trading,
            'symbols': self.symbols,
            'performance': stats,
            'trades': self.backtest_engine.trades
        }
        
        filename = f"trading_futuro100_hl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Resultados salvos em: {filename}")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")
    
    def save_trade_log(self, trade_data: Dict):
        """Salva trade individual em tempo real"""
        try:
            # Tentar carregar arquivo existente
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
            
            # Adicionar novo trade
            data['trades'].append(trade_data)
            data['performance']['total_trades'] = len(data['trades'])
            data['performance']['capital'] = self.backtest_engine.capital
            data['performance']['roi'] = ((self.backtest_engine.capital - self.backtest_engine.initial_capital) / self.backtest_engine.initial_capital) * 100
            data['last_update'] = datetime.now().isoformat()
            
            # Salvar arquivo atualizado
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Erro ao salvar trade log: {e}")
    
    def check_signals_and_trade(self, symbol: str) -> bool:
        """Verifica sinais para um símbolo e executa trade se necessário"""
        check_time = datetime.now().strftime('%H:%M:%S')
        print(f"🔍 [{check_time}] Verificando {symbol}...", end=' ')
        
        try:
            # Buscar dados incluindo candle atual
            df = self.data_manager.get_historical_data(symbol, '4h', 100, include_current=True)
            if df.empty:
                print("❌ Sem dados")
                return False
            
            print(f"✅ {len(df)} candles", end=' ')
            
            # Gerar sinais
            df_signals = self.strategy.generate_signals(df)
            
            # Pegar último sinal
            latest_timestamp = df_signals.index[-1]
            latest_signal = df_signals['signal'].iloc[-1]
            latest_price = df_signals['close'].iloc[-1]
            latest_rsi = df_signals['rsi'].iloc[-1] if 'rsi' in df_signals.columns else 0
            latest_ema = df_signals['ema21'].iloc[-1] if 'ema21' in df_signals.columns else 0
            
            print(f"| Preço: ${latest_price:.6f} | RSI: {latest_rsi:.1f} | EMA21: ${latest_ema:.6f} | Sinal: {latest_signal}")
            
            # Verificar se é um sinal novo (diferente do último cache)
            last_cached_signal = self.last_signals.get(symbol, 0)
            
            if latest_signal != 0 and latest_signal != last_cached_signal:
                print(f"🔔 NOVO SINAL {symbol}: {latest_signal} @ {latest_price:.6f} em {latest_timestamp}")
                
                # Executar trade
                old_capital = self.backtest_engine.capital
                old_trade_count = len(self.backtest_engine.trades)
                
                self.backtest_engine.execute_trade(latest_signal, latest_price, latest_timestamp, symbol)
                
                # Verificar se realmente executou um trade
                if len(self.backtest_engine.trades) > old_trade_count:
                    new_trade = self.backtest_engine.trades[-1]
                    
                    # Salvar trade em tempo real
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
                    
                    # Cache do último sinal
                    self.last_signals[symbol] = latest_signal
                    return True
            
            # Atualizar cache mesmo se não houve trade
            self.last_signals[symbol] = latest_signal
            
            # Atualizar equity
            self.backtest_engine.update_equity(latest_price, latest_timestamp)
            
            return False
            
        except Exception as e:
            print(f"❌ ERRO: {e}")
            return False
    
    def run_live_monitoring(self):
        """Executa monitoramento contínuo com candle em aberto"""
        print("\n" + "="*80)
        print("🔴 MODO MONITORAMENTO CONTÍNUO")
        print("="*80)
        print("📊 Verificando movimentações em tempo real (incluindo candle aberto)")
        print("💾 Salvando trades automaticamente quando ocorrerem")
        print("⚠️ Esta é uma simulação - nenhuma ordem real será executada")
        print(f"📝 Log de trades: {self.trades_file}")
        print("⏰ Execução limitada a 8 minutos (reiniciará automaticamente)")
        print("💓 Heartbeat: sinais de vida a cada verificação")
        print("="*80)
        
        start_time = time.time()
        max_duration = 8 * 60  # 8 minutos em segundos
        cycle_count = 0
        total_checks = 0
        
        try:
            while True:
                cycle_count += 1
                cycle_start = time.time()
                elapsed_time = time.time() - start_time
                
                # Verificar se deve parar por tempo limite
                if elapsed_time >= max_duration:
                    print(f"\n⏰ Limite de tempo atingido ({max_duration/60:.1f} min). Finalizando execução...")
                    break
                
                trades_this_cycle = 0
                
                print(f"\n{'='*80}")
                print(f"🔄 CICLO #{cycle_count} | ⏰ {datetime.now().strftime('%H:%M:%S')} | 📊 Tempo: {elapsed_time/60:.1f}min")
                print(f"{'='*80}")
                
                # Teste de conectividade
                try:
                    test_price = self.data_manager.get_current_price('BTC/USDT')
                    if test_price > 0:
                        print(f"🟢 Conexão API OK | BTC: ${test_price:,.0f}")
                    else:
                        print("🔴 Problema na conexão API")
                except Exception as e:
                    print(f"🔴 Erro de conectividade: {e}")
                
                print(f"\n📈 Verificando sinais em {len(self.symbols)} ativos:")
                print("-" * 80)
                
                for i, symbol in enumerate(self.symbols, 1):
                    total_checks += 1
                    try:
                        print(f"[{i}/{len(self.symbols)}] ", end='')
                        # Verificar sinais e executar trades se necessário
                        if self.check_signals_and_trade(symbol):
                            trades_this_cycle += 1
                            
                    except Exception as e:
                        print(f"❌ Erro processando {symbol}: {e}")
                    
                    # Mini pausa entre verificações para não sobrecarregar API
                    time.sleep(1)
                
                # Status do ciclo
                cycle_duration = time.time() - cycle_start
                roi = ((self.backtest_engine.capital - self.backtest_engine.initial_capital) / self.backtest_engine.initial_capital) * 100
                total_trades = len(self.backtest_engine.trades)
                
                print("-" * 80)
                print(f"📊 RESUMO DO CICLO #{cycle_count}:")
                print(f"   💰 Capital: ${self.backtest_engine.capital:.2f}")
                print(f"   📈 ROI: {roi:.2f}%")
                print(f"   🔢 Total de trades: {total_trades}")
                print(f"   🆕 Novos trades neste ciclo: {trades_this_cycle}")
                print(f"   ⏱️ Duração do ciclo: {cycle_duration:.1f}s")
                print(f"   🔍 Total de verificações: {total_checks}")
                
                if trades_this_cycle > 0:
                    print(f"   🎯 {trades_this_cycle} NOVOS TRADES EXECUTADOS!")
                
                # Tempo restante
                remaining_time = max_duration - elapsed_time
                print(f"   ⏰ Tempo restante da execução: {remaining_time/60:.1f} min")
                
                # Aguardar com heartbeat
                sleep_time = min(120, remaining_time - 10)  # Deixar 10s de margem
                if sleep_time > 0:
                    print(f"\n⏳ Aguardando {sleep_time/60:.1f} min até próximo ciclo...")
                    
                    # Heartbeat durante a espera
                    heartbeat_interval = 30  # Heartbeat a cada 30 segundos
                    elapsed_sleep = 0
                    
                    while elapsed_sleep < sleep_time:
                        actual_sleep = min(heartbeat_interval, sleep_time - elapsed_sleep)
                        time.sleep(actual_sleep)
                        elapsed_sleep += actual_sleep
                        
                        if elapsed_sleep < sleep_time:
                            remaining_wait = sleep_time - elapsed_sleep
                            print(f"💓 VIVO | {datetime.now().strftime('%H:%M:%S')} | Restam {remaining_wait/60:.1f} min para próximo ciclo")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoramento interrompido pelo usuário após {cycle_count} ciclos")
        
        print(f"\n{'='*80}")
        print(f"📊 EXECUÇÃO FINALIZADA")
        print(f"   🔄 Ciclos executados: {cycle_count}")
        print(f"   ⏱️ Duração total: {elapsed_time/60:.1f} minutos")
        print(f"   🔍 Total de verificações: {total_checks}")
        print(f"={'='*80}")
        self.print_results()

def main():
    """Função principal"""
    live_trading = int(os.getenv('LIVE_TRADING', '0')) == 1
    
    print("🚀 TradingFuturo100 - Sistema com configurações Hyperliquid")
    print("="*60)
    
    if live_trading:
        print("⚠️ ATENÇÃO: LIVE_TRADING=1 detectado")
        print("🔴 Executando MONITORAMENTO CONTÍNUO (com candle aberto)")
    else:
        print("📊 LIVE_TRADING=0 - Executando BACKTEST histórico")
    
    # Inicializar sistema
    system = TradingFuturo100(live_trading=False)  # Sempre em modo simulação
    
    if live_trading:
        system.run_live_monitoring()
    else:
        system.run_backtest()

if __name__ == "__main__":
    main()
