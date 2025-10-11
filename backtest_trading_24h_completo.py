#!/usr/bin/env python3
"""
Backtest Completo Trading.py - 24 Horas
========================================
Baixa histórico dos 18 assets das últimas 24 horas e aplica estratégia trading.py
com caixa de $36, entradas de $4 e leverage 3x ($12 por posição)

Configurações:
- Caixa inicial: $36
- Valor por entrada: $4
- Leverage: 3x (posição total: $12)
- ATR mínimo: 0.45%
- SL: 1.5% ROI / TP: 12% ROI
- Assets: 18 moedas do trading.py
"""

import pandas as pd
import numpy as np
import ccxt
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import json
import time

# Importar trading.py para usar as configurações
sys.path.append('.')
from trading import AssetSetup, GradientConfig

print("🚀 BACKTEST TRADING.PY - ÚLTIMAS 24 HORAS")
print("=" * 60)
print(f"📊 Configuração: Caixa $36 | Entrada $4 | Leverage 3x | Posição $12")
print(f"📈 SL: 1.5% ROI | TP: 12% ROI | ATR mín: 0.45%")
print(f"🎯 Assets: 18 moedas do trading.py")
print("=" * 60)

# Lista dos 18 assets do trading.py (exatos do código fonte)
ASSETS_TRADING = [
    {"name": "BTC-USD", "symbol": "BTCUSDT", "leverage": 3},
    {"name": "SOL-USD", "symbol": "SOLUSDT", "leverage": 3},
    {"name": "ETH-USD", "symbol": "ETHUSDT", "leverage": 3},
    {"name": "XRP-USD", "symbol": "XRPUSDT", "leverage": 3},
    {"name": "DOGE-USD", "symbol": "DOGEUSDT", "leverage": 3},
    {"name": "AVAX-USD", "symbol": "AVAXUSDT", "leverage": 3},
    {"name": "ENA-USD", "symbol": "ENAUSDT", "leverage": 3},
    {"name": "BNB-USD", "symbol": "BNBUSDT", "leverage": 3},
    {"name": "SUI-USD", "symbol": "SUIUSDT", "leverage": 3},
    {"name": "ADA-USD", "symbol": "ADAUSDT", "leverage": 3},
    {"name": "PUMP-USD", "symbol": "PUMPUSDT", "leverage": 3},
    {"name": "AVNT-USD", "symbol": "AVNTUSDT", "leverage": 3},
    {"name": "LINK-USD", "symbol": "LINKUSDT", "leverage": 3},
    {"name": "WLD-USD", "symbol": "WLDUSDT", "leverage": 3},
    {"name": "AAVE-USD", "symbol": "AAVEUSDT", "leverage": 3},
    {"name": "CRV-USD", "symbol": "CRVUSDT", "leverage": 3},
    {"name": "LTC-USD", "symbol": "LTCUSDT", "leverage": 3},
    {"name": "NEAR-USD", "symbol": "NEARUSDT", "leverage": 3},
]

# Configurações do backtest
CAIXA_INICIAL = 36.0
USD_PER_TRADE = 4.0
LEVERAGE = 3
POSICAO_TOTAL = USD_PER_TRADE * LEVERAGE  # $12
SL_ROI = -0.015  # -1.5%
TP_ROI = 0.12    # +12%
ATR_MIN_PCT = 0.45  # 0.45%

class BacktestTradingPy:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
        })
        self.caixa = CAIXA_INICIAL
        self.trades = []
        self.posicoes_abertas = {}
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        
    def baixar_dados_24h(self, symbol: str) -> Optional[pd.DataFrame]:
        """Baixa dados de 24 horas para um símbolo"""
        try:
            # Últimas 24 horas em timeframe 15m
            since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', since=since, limit=96)  # 24h / 15m = 96 candles
            
            if not ohlcv:
                print(f"❌ Sem dados para {symbol}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['data'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['valor_fechamento'] = df['close']
            df['criptomoeda'] = symbol
            
            # Calcular indicadores necessários
            df = self.calcular_indicadores(df)
            
            print(f"✅ {symbol}: {len(df)} candles baixados")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao baixar {symbol}: {e}")
            return None
    
    def calcular_indicadores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos necessários"""
        # EMAs genéticas
        df['ema_short'] = df['valor_fechamento'].ewm(span=3, adjust=False).mean()  # EMA3
        df['ema_long'] = df['valor_fechamento'].ewm(span=34, adjust=False).mean()  # EMA34
        
        # RSI 21
        delta = df['valor_fechamento'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
        
        # Volume MA
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def verificar_sinal_entrada(self, row: pd.Series) -> Optional[str]:
        """Verifica sinais de entrada baseado na estratégia trading.py"""
        # Verificar se temos dados suficientes
        if pd.isna(row['ema_short']) or pd.isna(row['ema_long']) or pd.isna(row['rsi']) or pd.isna(row['atr_pct']):
            return None
            
        # Condições genéticas DNA
        G1 = row['ema_short'] > row['ema_long']  # EMA3 > EMA34
        G2 = 20 < row['rsi'] < 85  # RSI21 dinâmico
        G3 = ATR_MIN_PCT < row['atr_pct'] < 8.0  # ATR calibrado
        G4 = row['volume'] > row['vol_ma'] * 1.3 if row['vol_ma'] > 0 else False  # Volume 1.3x
        G5 = row['valor_fechamento'] > row['ema_short']  # Preço acima EMA3
        
        # Critérios para LONG
        if G1 and G2 and G3 and G4 and G5:
            return "LONG"
            
        # Critérios para SHORT (inverso)
        G1_short = row['ema_short'] < row['ema_long']  # EMA3 < EMA34
        G2_short = 15 < row['rsi'] < 80  # RSI21 para SHORT
        G5_short = row['valor_fechamento'] < row['ema_short']  # Preço abaixo EMA3
        
        if G1_short and G2_short and G3 and G4 and G5_short:
            return "SHORT"
            
        return None
    
    def abrir_posicao(self, symbol: str, side: str, price: float, timestamp: datetime):
        """Abre uma posição"""
        if self.caixa < USD_PER_TRADE:
            return False  # Caixa insuficiente
            
        # Calcular preços de SL e TP
        if side == "LONG":
            sl_price = price * (1 + SL_ROI / LEVERAGE)  # SL em preço
            tp_price = price * (1 + TP_ROI / LEVERAGE)  # TP em preço
        else:  # SHORT
            sl_price = price * (1 - SL_ROI / LEVERAGE)
            tp_price = price * (1 - TP_ROI / LEVERAGE)
            
        posicao = {
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'entry_time': timestamp,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'amount': POSICAO_TOTAL,
            'usd_invested': USD_PER_TRADE
        }
        
        self.posicoes_abertas[symbol] = posicao
        self.caixa -= USD_PER_TRADE
        
        print(f"📈 {timestamp.strftime('%H:%M')} | ABERTO {side} {symbol} @ {price:.6f} | SL: {sl_price:.6f} | TP: {tp_price:.6f}")
        return True
    
    def verificar_fechamento(self, row: pd.Series):
        """Verifica se alguma posição deve ser fechada"""
        symbol = row['criptomoeda']
        current_price = row['valor_fechamento']
        timestamp = row['data']
        
        if symbol not in self.posicoes_abertas:
            return
            
        posicao = self.posicoes_abertas[symbol]
        
        # Verificar SL/TP
        fechado = False
        motivo = ""
        
        if posicao['side'] == "LONG":
            if current_price <= posicao['sl_price']:
                fechado = True
                motivo = "SL"
            elif current_price >= posicao['tp_price']:
                fechado = True
                motivo = "TP"
        else:  # SHORT
            if current_price >= posicao['sl_price']:
                fechado = True
                motivo = "SL"
            elif current_price <= posicao['tp_price']:
                fechado = True
                motivo = "TP"
        
        if fechado:
            self.fechar_posicao(symbol, current_price, timestamp, motivo)
    
    def fechar_posicao(self, symbol: str, exit_price: float, timestamp: datetime, motivo: str):
        """Fecha uma posição"""
        if symbol not in self.posicoes_abertas:
            return
            
        posicao = self.posicoes_abertas[symbol]
        
        # Calcular P&L
        if posicao['side'] == "LONG":
            roi = (exit_price - posicao['entry_price']) / posicao['entry_price']
        else:  # SHORT
            roi = (posicao['entry_price'] - exit_price) / posicao['entry_price']
            
        pnl_roi = roi * LEVERAGE  # P&L em ROI considerando leverage
        pnl_usd = pnl_roi * USD_PER_TRADE  # P&L em USD
        
        # Atualizar caixa
        self.caixa += USD_PER_TRADE + pnl_usd
        
        # Registrar trade
        trade = {
            'symbol': symbol,
            'side': posicao['side'],
            'entry_time': posicao['entry_time'],
            'exit_time': timestamp,
            'entry_price': posicao['entry_price'],
            'exit_price': exit_price,
            'duration_min': (timestamp - posicao['entry_time']).total_seconds() / 60,
            'roi_pct': pnl_roi * 100,
            'pnl_usd': pnl_usd,
            'motivo': motivo
        }
        
        self.trades.append(trade)
        self.total_trades += 1
        
        if pnl_usd > 0:
            self.wins += 1
            resultado = f"WIN +${pnl_usd:.2f}"
        else:
            self.losses += 1
            resultado = f"LOSS ${pnl_usd:.2f}"
            
        print(f"📉 {timestamp.strftime('%H:%M')} | FECHADO {posicao['side']} {symbol} @ {exit_price:.6f} | {motivo} | {resultado} | Caixa: ${self.caixa:.2f}")
        
        # Remover posição
        del self.posicoes_abertas[symbol]
    
    def executar_backtest(self):
        """Executa o backtest completo"""
        print("📥 Baixando dados dos 18 assets...")
        dados_assets = {}
        
        for asset in ASSETS_TRADING:
            symbol = asset['symbol']
            df = self.baixar_dados_24h(symbol)
            if df is not None:
                dados_assets[symbol] = df
            time.sleep(0.1)  # Rate limiting
        
        print(f"\n✅ Dados baixados para {len(dados_assets)} assets")
        print("🔄 Executando backtest...")
        
        # Combinar todos os dados e ordenar por timestamp
        all_rows = []
        for symbol, df in dados_assets.items():
            for _, row in df.iterrows():
                all_rows.append(row)
        
        # Ordenar por timestamp
        all_rows.sort(key=lambda x: x['data'])
        
        # Executar estratégia
        for i, row in enumerate(all_rows):
            # Verificar fechamentos primeiro
            self.verificar_fechamento(row)
            
            # Verificar novas entradas
            if row['criptomoeda'] not in self.posicoes_abertas:
                sinal = self.verificar_sinal_entrada(row)
                if sinal and self.caixa >= USD_PER_TRADE:
                    self.abrir_posicao(row['criptomoeda'], sinal, row['valor_fechamento'], row['data'])
        
        # Fechar posições ainda abertas (fim do período)
        for symbol, posicao in list(self.posicoes_abertas.items()):
            ultimo_preco = dados_assets[symbol]['valor_fechamento'].iloc[-1]
            ultimo_tempo = dados_assets[symbol]['data'].iloc[-1]
            self.fechar_posicao(symbol, ultimo_preco, ultimo_tempo, "TIMEOUT")
    
    def gerar_relatorio(self):
        """Gera relatório detalhado do backtest"""
        print("\n" + "=" * 80)
        print("📊 RELATÓRIO DETALHADO - BACKTEST TRADING.PY 24H")
        print("=" * 80)
        
        # Resumo geral
        total_pnl = sum(trade['pnl_usd'] for trade in self.trades)
        roi_total = (total_pnl / CAIXA_INICIAL) * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"💰 CAIXA INICIAL: ${CAIXA_INICIAL:.2f}")
        print(f"💰 CAIXA FINAL: ${self.caixa:.2f}")
        print(f"📈 P&L TOTAL: ${total_pnl:.2f}")
        print(f"📊 ROI TOTAL: {roi_total:.2f}%")
        print(f"🎯 TRADES TOTAIS: {self.total_trades}")
        print(f"✅ WINS: {self.wins}")
        print(f"❌ LOSSES: {self.losses}")
        print(f"📈 WIN RATE: {win_rate:.1f}%")
        
        if self.trades:
            avg_win = np.mean([t['pnl_usd'] for t in self.trades if t['pnl_usd'] > 0]) if self.wins > 0 else 0
            avg_loss = np.mean([t['pnl_usd'] for t in self.trades if t['pnl_usd'] < 0]) if self.losses > 0 else 0
            avg_duration = np.mean([t['duration_min'] for t in self.trades])
            
            print(f"💚 WIN MÉDIO: ${avg_win:.2f}")
            print(f"🔴 LOSS MÉDIO: ${avg_loss:.2f}")
            print(f"⏱️ DURAÇÃO MÉDIA: {avg_duration:.1f} min")
        
        print("\n" + "=" * 80)
        print("📋 DETALHES DOS TRADES")
        print("=" * 80)
        
        if not self.trades:
            print("❌ Nenhum trade executado")
            return
        
        # Tabela detalhada
        print(f"{'#':<3} {'Asset':<8} {'Side':<5} {'Entrada':<8} {'Saída':<8} {'Entry':<12} {'Exit':<12} {'Duração':<8} {'ROI':<8} {'P&L':<10} {'Motivo':<6}")
        print("-" * 120)
        
        for i, trade in enumerate(self.trades, 1):
            entrada_str = trade['entry_time'].strftime('%H:%M')
            saida_str = trade['exit_time'].strftime('%H:%M')
            roi_str = f"{trade['roi_pct']:+.1f}%"
            pnl_str = f"${trade['pnl_usd']:+.2f}"
            
            print(f"{i:<3} {trade['symbol']:<8} {trade['side']:<5} {entrada_str:<8} {saida_str:<8} "
                  f"{trade['entry_price']:<12.6f} {trade['exit_price']:<12.6f} {trade['duration_min']:<8.0f} "
                  f"{roi_str:<8} {pnl_str:<10} {trade['motivo']:<6}")
        
        # Análise por asset
        print("\n" + "=" * 80)
        print("📊 ANÁLISE POR ASSET")
        print("=" * 80)
        
        asset_stats = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in asset_stats:
                asset_stats[symbol] = {'trades': 0, 'wins': 0, 'pnl': 0}
            
            asset_stats[symbol]['trades'] += 1
            asset_stats[symbol]['pnl'] += trade['pnl_usd']
            if trade['pnl_usd'] > 0:
                asset_stats[symbol]['wins'] += 1
        
        print(f"{'Asset':<10} {'Trades':<7} {'Wins':<5} {'Win%':<6} {'P&L':<10}")
        print("-" * 45)
        
        for symbol, stats in sorted(asset_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            print(f"{symbol:<10} {stats['trades']:<7} {stats['wins']:<5} {win_rate:<6.1f} ${stats['pnl']:<9.2f}")
        
        # Salvar relatório em JSON
        relatorio = {
            'configuracao': {
                'caixa_inicial': CAIXA_INICIAL,
                'usd_per_trade': USD_PER_TRADE,
                'leverage': LEVERAGE,
                'sl_roi': SL_ROI,
                'tp_roi': TP_ROI,
                'atr_min_pct': ATR_MIN_PCT
            },
            'resultados': {
                'caixa_final': self.caixa,
                'pnl_total': total_pnl,
                'roi_total': roi_total,
                'total_trades': self.total_trades,
                'wins': self.wins,
                'losses': self.losses,
                'win_rate': win_rate
            },
            'trades': self.trades,
            'asset_stats': asset_stats
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'backtest_trading_24h_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(relatorio, f, indent=2, default=str)
        
        print(f"\n💾 Relatório salvo em: {filename}")

def main():
    """Função principal"""
    backtest = BacktestTradingPy()
    
    try:
        backtest.executar_backtest()
        backtest.gerar_relatorio()
        
    except KeyboardInterrupt:
        print("\n⏹️ Backtest interrompido pelo usuário")
        backtest.gerar_relatorio()
    except Exception as e:
        print(f"\n❌ Erro durante backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
