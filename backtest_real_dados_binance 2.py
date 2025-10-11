#!/usr/bin/env python3
"""
🔍 BACKTEST REAL COM DADOS BINANCE
=================================
✅ Usando dados reais de 1 ano da Binance
✅ Mesma lógica do trading.py
✅ Resultados exatos para gráfico PnL

Capital: $35 | Entradas: $4 | Leverage: 10x
Estratégia: Mais Permissiva (25% confluência)
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def configurar_estrategia_real():
    """Configuração exata da estratégia otimizada"""
    return {
        'capital_total': 35.0,
        'entry_size': 4.0,
        'leverage': 10,
        'max_positions': 8,
        'stop_loss_pct': 0.015,     # 1.5%
        'take_profit_pct': 0.12,    # 12%
        'min_confluence': 0.25,     # 25% (mais permissivo)
        'volume_multiplier': 1.2,
        'atr_breakout': 0.7,
        'fee_rate': 0.0008,         # 0.08% total por trade
        'ema_fast': 3,
        'ema_slow': 34,
        'rsi_period': 21,
        'atr_period': 14
    }

def carregar_dados_binance():
    """Carrega todos os dados reais da Binance"""
    assets = ['btc', 'eth', 'sol', 'xrp', 'doge', 'avax', 'ada', 'bnb', 'link', 'ltc', 'aave', 'crv', 'ena', 'near', 'sui', 'wld']
    
    dados_completos = {}
    
    print(f"📂 Carregando dados reais da Binance...")
    
    for asset in assets:
        try:
            filename = f"dados_reais_{asset}_1ano.csv"
            df = pd.read_csv(filename)
            
            # Converter data
            df['data'] = pd.to_datetime(df['data'])
            df = df.sort_values('data').reset_index(drop=True)
            
            # Verificar dados
            if len(df) > 8000:  # Pelo menos 8000 pontos (quase 1 ano)
                dados_completos[asset.upper()] = df
                print(f"   ✅ {asset.upper()}: {len(df)} registros ({df['data'].min().date()} → {df['data'].max().date()})")
            else:
                print(f"   ⚠️ {asset.upper()}: dados insuficientes ({len(df)} registros)")
                
        except Exception as e:
            print(f"   ❌ {asset.upper()}: erro - {e}")
    
    print(f"\n✅ Total assets carregados: {len(dados_completos)}")
    return dados_completos

def calcular_indicadores_reais(df):
    """Calcula indicadores técnicos com dados reais"""
    df = df.copy()
    
    # Renomear colunas para compatibilidade
    df['close'] = df['valor_fechamento']
    df['high'] = df['valor_maximo'] 
    df['low'] = df['valor_minimo']
    df['open'] = df['valor_abertura']
    
    # EMAs
    df['ema3'] = ta.trend.EMAIndicator(df['close'], window=3).ema_indicator()
    df['ema34'] = ta.trend.EMAIndicator(df['close'], window=34).ema_indicator()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    return df

def calcular_confluence_real(row, config):
    """Cálculo exato de confluência (igual ao trading.py)"""
    score = 0.0
    max_score = 10.0
    
    # 1. EMA Trend (peso 3)
    if pd.notna(row['ema3']) and pd.notna(row['ema34']) and row['ema3'] > row['ema34']:
        score += 3.0
    
    # 2. ATR saudável (peso 1)
    if pd.notna(row['atr_pct']) and 0.3 <= row['atr_pct'] <= 8.0:
        score += 1.0
    
    # 3. Volume (peso 2)
    if pd.notna(row['volume_ratio']):
        if row['volume_ratio'] >= config['volume_multiplier']:
            score += 2.0
        elif row['volume_ratio'] >= (config['volume_multiplier'] * 0.8):
            score += 1.0
    
    # 4. RSI não extremo (peso 1)
    if pd.notna(row['rsi']) and 25 <= row['rsi'] <= 75:
        score += 1.0
    
    # 5. Breakout ATR (peso 3)
    if pd.notna(row['atr_pct']) and row['atr_pct'] >= config['atr_breakout']:
        score += 3.0
    
    return (score / max_score) * 100

def executar_backtest_real(dados_completos, config):
    """Executa backtest com dados reais"""
    
    # Estado inicial
    capital_livre = config['capital_total']
    posicoes_ativas = []
    historico_trades = []
    historico_diario = []
    
    # Encontrar período mais amplo (usar união ao invés de interseção)
    todas_datas = set()
    for asset, df in dados_completos.items():
        todas_datas.update(df['data'])
    
    datas_ordenadas = sorted(list(todas_datas))
    periodo_inicio = datas_ordenadas[0]
    periodo_fim = datas_ordenadas[-1]
    
    print(f"\n📋 ASSETS DISPONÍVEIS: {list(dados_completos.keys())}")
    print(f"📊 Total assets: {len(dados_completos)}")
    
    print(f"\n🔄 EXECUTANDO BACKTEST REAL")
    print(f"📅 Período: {periodo_inicio.date()} → {periodo_fim.date()}")
    print(f"🏃 Total pontos: {len(datas_ordenadas)}")
    print(f"� Total assets: {len(dados_completos)}")
    print(f"�💰 Capital inicial: ${config['capital_total']}")
    print(f"🧮 Candles disponíveis: {sum(len(df) for df in dados_completos.values()):,}")
    print(f"🎯 Análises esperadas: {len(datas_ordenadas) * len(dados_completos):,}")
    
    total_analises_realizadas = 0
    
    # Processar cada timestamp
    for i, timestamp in enumerate(datas_ordenadas):
        
        # Progresso
        if i % 1000 == 0:
            progresso = (i / len(datas_ordenadas)) * 100
            print(f"   📊 Progresso: {progresso:.1f}% ({i}/{len(datas_ordenadas)}) | Análises: {total_analises_realizadas:,}")
        
        # Contar assets disponíveis neste timestamp
        assets_disponiveis = 0
        for asset, df_asset in dados_completos.items():
            if not df_asset[df_asset['data'] == timestamp].empty:
                assets_disponiveis += 1
                total_analises_realizadas += 1
        
        # Capital total no momento
        capital_total_atual = capital_livre
        
        # Verificar posições ativas (Stop Loss / Take Profit)
        posicoes_para_remover = []
        
        for j, pos in enumerate(posicoes_ativas):
            asset = pos['asset']
            
            # Buscar preço atual do asset
            df_asset = dados_completos[asset]
            row_atual = df_asset[df_asset['data'] == timestamp]
            
            if not row_atual.empty:
                preco_atual = row_atual.iloc[0]['valor_fechamento']
                
                # Calcular P&L não realizado
                if pos['tipo'] == 'LONG':
                    pnl_nao_realizado = (preco_atual - pos['preco_entrada']) / pos['preco_entrada']
                else:  # SHORT
                    pnl_nao_realizado = (pos['preco_entrada'] - preco_atual) / pos['preco_entrada']
                
                # Aplicar leverage
                pnl_nao_realizado *= pos['leverage']
                capital_posicao = pnl_nao_realizado * pos['size'] + pos['size']
                capital_total_atual += capital_posicao - pos['size']
                
                # Verificar Stop Loss
                if pnl_nao_realizado <= -config['stop_loss_pct']:
                    # Fechar posição por Stop Loss
                    # PnL = % movimento × SEU capital (não o notional)
                    pnl_realizado = -config['stop_loss_pct'] * pos['size']
                    fees = pos['size'] * pos['leverage'] * config['fee_rate']  # Taxa sobre notional
                    pnl_liquido = pnl_realizado - fees
                    
                    capital_livre += pos['size'] + pnl_liquido
                    
                    # Registrar trade
                    historico_trades.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'tipo': 'STOP_LOSS',
                        'pnl_bruto': pnl_realizado,
                        'fees': fees,
                        'pnl_liquido': pnl_liquido,
                        'size': pos['size'],
                        'leverage': pos['leverage']
                    })
                    
                    posicoes_para_remover.append(j)
                    
                # Verificar Take Profit
                elif pnl_nao_realizado >= config['take_profit_pct']:
                    # Fechar posição por Take Profit
                    # PnL = % movimento × SEU capital (não o notional)
                    pnl_realizado = config['take_profit_pct'] * pos['size']
                    fees = pos['size'] * pos['leverage'] * config['fee_rate']  # Taxa sobre notional
                    pnl_liquido = pnl_realizado - fees
                    
                    capital_livre += pos['size'] + pnl_liquido
                    
                    # Registrar trade
                    historico_trades.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'tipo': 'TAKE_PROFIT',
                        'pnl_bruto': pnl_realizado,
                        'fees': fees,
                        'pnl_liquido': pnl_liquido,
                        'size': pos['size'],
                        'leverage': pos['leverage']
                    })
                    
                    posicoes_para_remover.append(j)
        
        # Remover posições fechadas
        for j in reversed(posicoes_para_remover):
            posicoes_ativas.pop(j)
        
        # Verificar novas oportunidades (se temos capital livre)
        if (capital_livre >= config['entry_size'] and 
            len(posicoes_ativas) < config['max_positions']):
            
            # Analisar todos os assets disponíveis
            for asset, df_asset in dados_completos.items():
                
                # Verificar se já temos posição neste asset
                asset_ocupado = any(pos['asset'] == asset for pos in posicoes_ativas)
                if asset_ocupado:
                    continue
                
                # Buscar dados do timestamp atual (verificar se existe)
                row_atual = df_asset[df_asset['data'] == timestamp]
                if row_atual.empty:
                    continue  # Asset não tem dados para este timestamp
                
                # Calcular indicadores até este ponto
                df_ate_agora = df_asset[df_asset['data'] <= timestamp].copy()
                if len(df_ate_agora) < 50:  # Precisamos de histórico suficiente
                    continue
                
                df_indicators = calcular_indicadores_reais(df_ate_agora)
                row_analise = df_indicators.iloc[-1]  # Última linha com indicadores
                
                # Calcular confluência
                confluence = calcular_confluence_real(row_analise, config)
                
                # Verificar se atende critério mínimo
                if confluence >= config['min_confluence']:
                    
                    # Abrir posição LONG
                    preco_entrada = row_analise['close']
                    
                    # Taxa de abertura
                    fees_abertura = config['entry_size'] * config['leverage'] * config['fee_rate']
                    
                    # Criar posição
                    nova_posicao = {
                        'asset': asset,
                        'tipo': 'LONG',
                        'preco_entrada': preco_entrada,
                        'size': config['entry_size'],
                        'leverage': config['leverage'],
                        'timestamp_abertura': timestamp,
                        'confluence': confluence,
                        'fees_abertura': fees_abertura
                    }
                    
                    posicoes_ativas.append(nova_posicao)
                    capital_livre -= config['entry_size']
                    
                    # Registrar abertura
                    historico_trades.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'tipo': 'ABERTURA',
                        'pnl_bruto': 0,
                        'fees': fees_abertura,
                        'pnl_liquido': -fees_abertura,
                        'size': config['entry_size'],
                        'leverage': config['leverage'],
                        'confluence': confluence
                    })
                    
                    # Parar se atingimos limite de posições
                    if len(posicoes_ativas) >= config['max_positions']:
                        break
        
        # Registrar snapshot diário (apenas datas importantes)
        if timestamp.hour == 0:  # Uma vez por dia
            
            capital_total_final = capital_livre
            
            # Somar valor não realizado das posições ativas
            for pos in posicoes_ativas:
                asset = pos['asset']
                df_asset = dados_completos[asset]
                row_atual = df_asset[df_asset['data'] == timestamp]
                
                if not row_atual.empty:
                    preco_atual = row_atual.iloc[0]['valor_fechamento']
                    
                    if pos['tipo'] == 'LONG':
                        pnl_nao_realizado = (preco_atual - pos['preco_entrada']) / pos['preco_entrada']
                    else:
                        pnl_nao_realizado = (pos['preco_entrada'] - preco_atual) / pos['preco_entrada']
                    
                    pnl_nao_realizado *= pos['leverage']
                    valor_posicao = pos['size'] + (pnl_nao_realizado * pos['size'])
                    capital_total_final += valor_posicao - pos['size']
            
            # PnL do dia
            if len(historico_diario) > 0:
                capital_ontem = historico_diario[-1]['capital_total']
                daily_pnl = capital_total_final - capital_ontem
            else:
                daily_pnl = capital_total_final - config['capital_total']
            
            historico_diario.append({
                'data': timestamp.date(),
                'timestamp': timestamp,
                'capital_livre': capital_livre,
                'capital_total': capital_total_final,
                'posicoes_ativas': len(posicoes_ativas),
                'trades_do_dia': len([t for t in historico_trades if t['timestamp'].date() == timestamp.date()]),
                'daily_pnl': daily_pnl
            })
    
    print(f"\n✅ BACKTEST CONCLUÍDO!")
    print(f"📊 Total de trades: {len(historico_trades)}")
    print(f"📅 Dias analisados: {len(historico_diario)}")
    print(f"🧮 Análises de candles realizadas: {total_analises_realizadas:,}")
    print(f"📈 Cobertura: {(total_analises_realizadas / sum(len(df) for df in dados_completos.values())) * 100:.1f}% dos candles disponíveis")
    
    return historico_diario, historico_trades

def gerar_relatorio_final(historico_diario, historico_trades, config):
    """Gera relatório final dos resultados"""
    
    if not historico_diario:
        print("❌ Nenhum dado diário para análise")
        return None
    
    # Capital final
    capital_inicial = config['capital_total']
    capital_final = historico_diario[-1]['capital_total']
    roi_total = ((capital_final - capital_inicial) / capital_inicial) * 100
    
    # Análise de trades
    trades_vencedores = [t for t in historico_trades if t.get('pnl_liquido', 0) > 0]
    trades_perdedores = [t for t in historico_trades if t.get('pnl_liquido', 0) < 0]
    trades_abertura = [t for t in historico_trades if t['tipo'] == 'ABERTURA']
    
    win_rate = len(trades_vencedores) / max(len(trades_vencedores) + len(trades_perdedores), 1) * 100
    
    # PnL médio
    pnl_vencedores_medio = np.mean([t['pnl_liquido'] for t in trades_vencedores]) if trades_vencedores else 0
    pnl_perdedores_medio = np.mean([t['pnl_liquido'] for t in trades_perdedores]) if trades_perdedores else 0
    
    # Drawdown
    df_daily = pd.DataFrame(historico_diario)
    df_daily['capital_peak'] = df_daily['capital_total'].expanding().max()
    df_daily['drawdown_pct'] = (df_daily['capital_total'] - df_daily['capital_peak']) / df_daily['capital_peak'] * 100
    max_drawdown = df_daily['drawdown_pct'].min()
    
    # Período
    periodo_inicio = df_daily['data'].min()
    periodo_fim = df_daily['data'].max()
    dias_total = (periodo_fim - periodo_inicio).days + 1
    
    print(f"\n" + "="*60)
    print(f"📊 RELATÓRIO FINAL - BACKTEST REAL DADOS BINANCE")
    print(f"="*60)
    
    print(f"\n💰 PERFORMANCE:")
    print(f"   Capital Inicial: ${capital_inicial:.2f}")
    print(f"   Capital Final: ${capital_final:.2f}")
    print(f"   ROI Total: {roi_total:+.2f}%")
    print(f"   Lucro/Prejuízo: ${capital_final - capital_inicial:+.2f}")
    
    print(f"\n📅 PERÍODO:")
    print(f"   Início: {periodo_inicio}")
    print(f"   Fim: {periodo_fim}")
    print(f"   Total Dias: {dias_total}")
    print(f"   ROI por Dia: {roi_total/dias_total:+.3f}%")
    
    print(f"\n🎯 TRADES:")
    print(f"   Total Trades: {len(historico_trades)}")
    print(f"   Aberturas: {len(trades_abertura)}")
    print(f"   Vencedores: {len(trades_vencedores)} ({win_rate:.1f}%)")
    print(f"   Perdedores: {len(trades_perdedores)} ({100-win_rate:.1f}%)")
    print(f"   PnL Médio Ganho: ${pnl_vencedores_medio:.3f}")
    print(f"   PnL Médio Perda: ${pnl_perdedores_medio:.3f}")
    
    print(f"\n📉 RISCO:")
    print(f"   Max Drawdown: {max_drawdown:.2f}%")
    print(f"   Trades/Dia: {len(trades_abertura)/dias_total:.1f}")
    
    # Taxa de sucesso da estratégia
    if roi_total > 0:
        print(f"\n✅ ESTRATÉGIA LUCRATIVA!")
    else:
        print(f"\n❌ ESTRATÉGIA COM PREJUÍZO!")
    
    return {
        'capital_inicial': capital_inicial,
        'capital_final': capital_final,
        'roi_total': roi_total,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(historico_trades),
        'periodo_dias': dias_total,
        'df_daily': df_daily
    }

def main():
    """Função principal"""
    print("🚀 BACKTEST REAL COM DADOS BINANCE")
    print("="*50)
    
    # Configuração
    config = configurar_estrategia_real()
    
    # Carregar dados
    dados_completos = carregar_dados_binance()
    
    if len(dados_completos) < 5:
        print("❌ Dados insuficientes para backtest!")
        return
    
    # Executar backtest
    historico_diario, historico_trades = executar_backtest_real(dados_completos, config)
    
    # Gerar relatório
    resultado = gerar_relatorio_final(historico_diario, historico_trades, config)
    
    if resultado:
        # Salvar resultados para gráfico
        df_results = resultado['df_daily'].copy()
        
        # Adicionar campos necessários para gráfico
        df_results['capital_acumulado'] = df_results['capital_total']
        df_results['total_trades'] = df_results['trades_do_dia'].cumsum()
        df_results['date'] = pd.to_datetime(df_results['data'])
        
        # Salvar em CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_results = f"backtest_real_binance_{timestamp}.csv"
        df_results.to_csv(filename_results, index=False)
        
        filename_trades = f"trades_real_binance_{timestamp}.csv"
        pd.DataFrame(historico_trades).to_csv(filename_trades, index=False)
        
        print(f"\n💾 ARQUIVOS SALVOS:")
        print(f"   📊 Resultados Diários: {filename_results}")
        print(f"   📈 Histórico Trades: {filename_trades}")
        print(f"\n🎯 Use estes dados para gerar gráfico PnL real!")
        
        return df_results, historico_trades
    
    return None, None

if __name__ == "__main__":
    main()
