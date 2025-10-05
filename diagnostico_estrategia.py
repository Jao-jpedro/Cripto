#!/usr/bin/env python3
"""
Diagnóstico da Estratégia de Trading
Identifica problemas na lógica atual
"""

import os
import sys
import pandas as pd
import numpy as np

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_strategy():
    """Diagnostica problemas na estratégia"""
    
    print("🔍 DIAGNÓSTICO DA ESTRATÉGIA DE TRADING")
    print("🎯 Identificando problemas na lógica atual")
    print("="*60)
    
    # Testar com BTC primeiro
    filename = "dados_reais_btc_1ano.csv"
    if not os.path.exists(filename):
        print(f"❌ Arquivo {filename} não encontrado!")
        return
    
    df = load_data(filename)
    if df is None:
        print("❌ Erro ao carregar dados!")
        return
    
    print(f"📊 Dados carregados: {len(df)} barras")
    print(f"📅 Período: {df['timestamp'].min()} a {df['timestamp'].max()}" if 'timestamp' in df.columns else "📅 Período: dados históricos")
    
    # Calcular indicadores básicos
    df = calculate_basic_indicators(df)
    
    # Análise 1: Comportamento dos indicadores
    analyze_indicators(df)
    
    # Análise 2: Frequência de sinais
    analyze_signals(df)
    
    # Análise 3: Simulação simples
    simple_backtest(df)
    
    # Análise 4: Distribuição de retornos
    analyze_returns(df)

def load_data(filename):
    """Carrega dados com tratamento robusto"""
    try:
        df = pd.read_csv(filename)
        
        # Detectar e converter timestamp
        for col in ['timestamp', 'data', 'date', 'time']:
            if col in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df[col])
                    break
                except:
                    continue
        
        # Padronizar nomes das colunas
        column_mapping = {
            'open': 'valor_abertura',
            'high': 'valor_maximo',
            'low': 'valor_minimo',
            'close': 'valor_fechamento',
            'volume': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Verificar colunas essenciais
        required = ['valor_abertura', 'valor_maximo', 'valor_minimo', 'valor_fechamento', 'volume']
        if not all(col in df.columns for col in required):
            print(f"⚠️ Colunas faltando: {[col for col in required if col not in df.columns]}")
            return None
        
        # Limpar dados
        df = df.dropna()
        df = df[df['valor_fechamento'] > 0]
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None

def calculate_basic_indicators(df):
    """Calcula indicadores básicos"""
    df = df.copy()
    
    # EMAs
    df['ema_7'] = df['valor_fechamento'].ewm(span=7).mean()
    df['ema_21'] = df['valor_fechamento'].ewm(span=21).mean()
    
    # ATR
    high_low = df['valor_maximo'] - df['valor_minimo']
    high_close = abs(df['valor_maximo'] - df['valor_fechamento'].shift(1))
    low_close = abs(df['valor_minimo'] - df['valor_fechamento'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['valor_fechamento']) * 100
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    # RSI
    delta = df['valor_fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Retornos
    df['retorno_1d'] = df['valor_fechamento'].pct_change()
    df['retorno_1d_leverage'] = df['retorno_1d'] * 20  # Leverage 20x
    
    return df

def analyze_indicators(df):
    """Analisa comportamento dos indicadores"""
    
    print("\n📊 ANÁLISE DOS INDICADORES")
    print("-"*40)
    
    # Estatísticas básicas
    print(f"ATR %: {df['atr_pct'].describe()}")
    print(f"\nRSI: {df['rsi'].describe()}")
    print(f"\nVolume ratio: {(df['volume'] / df['vol_ma']).describe()}")
    
    # Filtros ATR
    atr_in_range = df[(df['atr_pct'] >= 0.5) & (df['atr_pct'] <= 3.0)]
    print(f"\n🎯 Barras com ATR 0.5-3.0%: {len(atr_in_range)}/{len(df)} ({len(atr_in_range)/len(df)*100:.1f}%)")
    
    # Volume alto
    high_volume = df[df['volume'] > df['vol_ma'] * 3.0]
    print(f"🎯 Barras com Volume > 3x: {len(high_volume)}/{len(df)} ({len(high_volume)/len(df)*100:.1f}%)")
    
    # RSI extremos
    rsi_oversold = df[df['rsi'] < 20]
    rsi_overbought = df[df['rsi'] > 80]
    print(f"🎯 RSI < 20 (oversold): {len(rsi_oversold)}/{len(df)} ({len(rsi_oversold)/len(df)*100:.1f}%)")
    print(f"🎯 RSI > 80 (overbought): {len(rsi_overbought)}/{len(df)} ({len(rsi_overbought)/len(df)*100:.1f}%)")

def analyze_signals(df):
    """Analisa frequência de sinais"""
    
    print("\n📡 ANÁLISE DE SINAIS")
    print("-"*40)
    
    # EMA Crosses
    ema_bullish = ((df['ema_7'].shift(1) <= df['ema_21'].shift(1)) & 
                   (df['ema_7'] > df['ema_21']))
    ema_bearish = ((df['ema_7'].shift(1) >= df['ema_21'].shift(1)) & 
                   (df['ema_7'] < df['ema_21']))
    
    print(f"📈 EMA Cross Bullish: {ema_bullish.sum()} sinais")
    print(f"📉 EMA Cross Bearish: {ema_bearish.sum()} sinais")
    
    # Filtros combinados
    valid_conditions = (
        (df['atr_pct'] >= 0.5) & 
        (df['atr_pct'] <= 3.0) &
        (df['volume'] > df['vol_ma'] * 3.0)
    )
    
    valid_bullish = ema_bullish & valid_conditions
    valid_bearish = ema_bearish & valid_conditions
    
    print(f"✅ Sinais Bullish válidos: {valid_bullish.sum()}")
    print(f"✅ Sinais Bearish válidos: {valid_bearish.sum()}")
    
    # Distribuição temporal
    if valid_bullish.sum() > 0:
        dias_entre_sinais = len(df) / (valid_bullish.sum() + valid_bearish.sum())
        print(f"⏰ Frequência: ~1 sinal a cada {dias_entre_sinais:.1f} barras")

def simple_backtest(df):
    """Backtest simples para identificar problemas"""
    
    print("\n🧪 BACKTEST SIMPLES")
    print("-"*40)
    
    balance = 1000.0
    positions = []
    leverage = 20
    
    # Parâmetros conservadores
    tp_pct = 0.15  # 15% TP
    sl_pct = 0.05  # 5% SL
    
    print(f"💰 Capital inicial: ${balance:.2f}")
    print(f"⚖️ Leverage: {leverage}x")
    print(f"🎯 TP/SL: {tp_pct*100:.0f}%/{sl_pct*100:.0f}%")
    
    position = None
    trades = []
    
    for i in range(21, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Verificar saída de posição
        if position:
            entry_price = position['entry_price']
            side = position['side']
            
            if side == 'long':
                pnl_pct = ((current['valor_fechamento'] - entry_price) / entry_price) * leverage
            else:
                pnl_pct = ((entry_price - current['valor_fechamento']) / entry_price) * leverage
            
            # Verificar TP/SL
            if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                balance_before = position['balance']
                balance = balance_before * (1 + pnl_pct)
                
                trades.append({
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': current['valor_fechamento'],
                    'pnl_pct': pnl_pct,
                    'balance_after': balance
                })
                
                position = None
                continue
        
        # Verificar entrada
        if not position:
            # EMA Cross
            ema_bullish = (prev['ema_7'] <= prev['ema_21'] and current['ema_7'] > current['ema_21'])
            ema_bearish = (prev['ema_7'] >= prev['ema_21'] and current['ema_7'] < current['ema_21'])
            
            # Filtros básicos
            atr_ok = 0.5 <= current['atr_pct'] <= 3.0
            volume_ok = current['volume'] > current['vol_ma'] * 2.0  # Mais permissivo
            
            if ema_bullish and atr_ok and volume_ok:
                position = {
                    'side': 'long',
                    'entry_price': current['valor_fechamento'],
                    'balance': balance
                }
            elif ema_bearish and atr_ok and volume_ok:
                position = {
                    'side': 'short',
                    'entry_price': current['valor_fechamento'],
                    'balance': balance
                }
    
    # Resultados
    if trades:
        total_return = ((balance - 1000) / 1000) * 100
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        win_rate = len(wins) / len(trades) * 100
        
        print(f"\n📊 RESULTADOS:")
        print(f"   💰 Balance final: ${balance:.2f}")
        print(f"   📈 Retorno total: {total_return:.1f}%")
        print(f"   📝 Total trades: {len(trades)}")
        print(f"   🎯 Win rate: {win_rate:.1f}%")
        print(f"   ✅ Trades ganhos: {len(wins)}")
        print(f"   ❌ Trades perdidos: {len(losses)}")
        
        if wins:
            avg_win = np.mean([t['pnl_pct'] for t in wins]) * 100
            print(f"   💚 Ganho médio: {avg_win:.1f}%")
        
        if losses:
            avg_loss = np.mean([t['pnl_pct'] for t in losses]) * 100
            print(f"   💔 Perda média: {avg_loss:.1f}%")
        
        # Mostrar últimos trades
        print(f"\n📋 ÚLTIMOS 5 TRADES:")
        for i, trade in enumerate(trades[-5:], 1):
            side = trade['side'].upper()
            pnl = trade['pnl_pct'] * 100
            status = "✅" if pnl > 0 else "❌"
            print(f"   {status} #{len(trades)-5+i}: {side} {pnl:+.1f}% (${trade['balance_after']:.2f})")
    else:
        print("\n❌ NENHUM TRADE EXECUTADO!")
        print("   Possíveis problemas:")
        print("   - Filtros muito restritivos")
        print("   - Dados insuficientes")
        print("   - Lógica de entrada incorreta")

def analyze_returns(df):
    """Analisa distribuição de retornos"""
    
    print("\n📈 ANÁLISE DE RETORNOS")
    print("-"*40)
    
    returns = df['retorno_1d'].dropna()
    leveraged_returns = returns * 20  # Leverage 20x
    
    print(f"Retorno médio diário: {returns.mean()*100:.3f}%")
    print(f"Volatilidade diária: {returns.std()*100:.2f}%")
    print(f"Com leverage 20x:")
    print(f"   Retorno médio: {leveraged_returns.mean()*100:.2f}%")
    print(f"   Volatilidade: {leveraged_returns.std()*100:.1f}%")
    
    # Distribuição de retornos leveraged
    positive_days = (leveraged_returns > 0).sum()
    negative_days = (leveraged_returns < 0).sum()
    
    print(f"\n📊 Distribuição com leverage:")
    print(f"   Dias positivos: {positive_days}/{len(leveraged_returns)} ({positive_days/len(leveraged_returns)*100:.1f}%)")
    print(f"   Dias negativos: {negative_days}/{len(leveraged_returns)} ({negative_days/len(leveraged_returns)*100:.1f}%)")
    
    # Extremos
    max_gain = leveraged_returns.max() * 100
    max_loss = leveraged_returns.min() * 100
    
    print(f"   Maior ganho diário: {max_gain:.1f}%")
    print(f"   Maior perda diária: {max_loss:.1f}%")
    
    # Perigos do leverage
    extreme_losses = (leveraged_returns < -0.1).sum()  # Perdas > 10%
    
    print(f"\n⚠️ RISCOS DO LEVERAGE:")
    print(f"   Dias com perda > 10%: {extreme_losses}")
    if extreme_losses > 0:
        print(f"   ⚠️ Com SL 10%, pode haver liquidação!")

def main():
    diagnose_strategy()
    
    print("\n" + "="*60)
    print("💡 CONCLUSÕES E PRÓXIMOS PASSOS")
    print("="*60)
    print("1. Verificar se os filtros não estão muito restritivos")
    print("2. Testar com TP/SL mais conservadores")
    print("3. Reduzir leverage se necessário")
    print("4. Validar qualidade dos dados históricos")
    print("5. Considerar estratégia de trend following mais simples")

if __name__ == "__main__":
    main()
