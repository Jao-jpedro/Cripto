#!/usr/bin/env python3
"""
🛡️ BACKTEST COM PROTEÇÕES CONTRA QUEDAS
=======================================
✅ Filtros de crash de mercado
✅ Limite de perdas consecutivas  
✅ Tendência do BTC
✅ Drawdown máximo
✅ Stop loss adaptativo

Testando se as proteções melhoram o resultado
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from backtest_real_dados_binance import *
from estrategias_protecao_mercado import *

def executar_backtest_protegido(dados_completos, config):
    """Executa backtest com proteções ativas"""
    
    # Estado inicial
    capital_livre = config['capital_total']
    posicoes_ativas = []
    historico_trades = []
    historico_diario = []
    capital_pico = config['capital_total']
    
    # Encontrar período mais amplo
    todas_datas = set()
    for asset, df in dados_completos.items():
        todas_datas.update(df['data'])
    
    datas_ordenadas = sorted(list(todas_datas))
    periodo_inicio = datas_ordenadas[0]
    periodo_fim = datas_ordenadas[-1]
    
    print(f"\n🛡️ EXECUTANDO BACKTEST COM PROTEÇÕES")
    print(f"📅 Período: {periodo_inicio.date()} → {periodo_fim.date()}")
    print(f"🏃 Total pontos: {len(datas_ordenadas)}")
    print(f"📊 Total assets: {len(dados_completos)}")
    print(f"💰 Capital inicial: ${config['capital_total']}")
    
    total_analises_realizadas = 0
    trades_bloqueados = 0
    dias_sem_operacao = 0
    
    # Processar cada timestamp
    for i, timestamp in enumerate(datas_ordenadas):
        
        # Progresso
        if i % 1000 == 0:
            progresso = (i / len(datas_ordenadas)) * 100
            print(f"   📊 {progresso:.1f}% | Bloqueados: {trades_bloqueados} | Dias pausados: {dias_sem_operacao}")
        
        # Contar assets disponíveis neste timestamp
        for asset, df_asset in dados_completos.items():
            if not df_asset[df_asset['data'] == timestamp].empty:
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
                else:
                    pnl_nao_realizado = (pos['preco_entrada'] - preco_atual) / pos['preco_entrada']
                
                # Aplicar leverage
                pnl_nao_realizado *= pos['leverage']
                capital_posicao = pnl_nao_realizado * pos['size'] + pos['size']
                capital_total_atual += capital_posicao - pos['size']
                
                # 🛡️ STOP LOSS ADAPTATIVO
                stop_loss_dinamico = pos.get('stop_loss_adaptativo', config['stop_loss_pct'])
                
                # Verificar Stop Loss
                if pnl_nao_realizado <= -stop_loss_dinamico:
                    pnl_realizado = -stop_loss_dinamico * pos['size']
                    fees = pos['size'] * pos['leverage'] * config['fee_rate']
                    pnl_liquido = pnl_realizado - fees
                    
                    capital_livre += pos['size'] + pnl_liquido
                    
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
                    pnl_realizado = config['take_profit_pct'] * pos['size']
                    fees = pos['size'] * pos['leverage'] * config['fee_rate']
                    pnl_liquido = pnl_realizado - fees
                    
                    capital_livre += pos['size'] + pnl_liquido
                    
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
        
        # Atualizar capital pico
        if capital_total_atual > capital_pico:
            capital_pico = capital_total_atual
        
        # 🛡️ APLICAR FILTROS DE PROTEÇÃO
        filtros = aplicar_filtros_protecao(
            config, dados_completos, timestamp, 
            historico_trades, capital_total_atual, capital_pico
        )
        
        # Contar dias sem operação
        if not filtros['pode_operar']:
            dias_sem_operacao += 1
        
        # Verificar novas oportunidades (COM PROTEÇÕES)
        if (filtros['pode_operar'] and 
            filtros['pode_abrir_posicao'] and
            capital_livre >= filtros['entry_size'] and 
            len(posicoes_ativas) < filtros['max_positions']):
            
            # Analisar todos os assets disponíveis
            for asset, df_asset in dados_completos.items():
                
                # Verificar se já temos posição neste asset
                asset_ocupado = any(pos['asset'] == asset for pos in posicoes_ativas)
                if asset_ocupado:
                    continue
                
                # Buscar dados do timestamp atual
                row_atual = df_asset[df_asset['data'] == timestamp]
                if row_atual.empty:
                    continue
                
                # Calcular indicadores até este ponto
                df_ate_agora = df_asset[df_asset['data'] <= timestamp].copy()
                if len(df_ate_agora) < 50:
                    continue
                
                df_indicators = calcular_indicadores_reais(df_ate_agora)
                row_analise = df_indicators.iloc[-1]
                
                # 🛡️ VERIFICAÇÕES ADICIONAIS DE VOLATILIDADE
                if pd.notna(row_analise.get('atr_pct')) and row_analise['atr_pct'] > 8.0:
                    trades_bloqueados += 1
                    continue
                
                # Calcular confluência
                confluence = calcular_confluence_real(row_analise, config)
                
                # Verificar se atende critério mínimo
                if confluence >= config['min_confluence']:
                    
                    # 🛡️ AJUSTAR TAMANHO DA POSIÇÃO POR VOLATILIDADE
                    entry_size_ajustado = ajustar_size_por_volatilidade(
                        filtros['entry_size'], 
                        row_analise.get('atr_pct', 3.0)
                    )
                    
                    # 🛡️ STOP LOSS ADAPTATIVO
                    if 'BTC' in dados_completos:
                        dados_btc = dados_completos['BTC'][dados_completos['BTC']['data'] <= timestamp]
                        _, trend_btc, _ = calcular_trend_btc(dados_btc)
                        stop_loss_adaptativo = ajustar_stop_loss_adaptativo(
                            filtros['stop_loss_pct'], 
                            trend_btc, 
                            row_analise.get('atr_pct', 3.0)
                        )
                    else:
                        stop_loss_adaptativo = filtros['stop_loss_pct']
                    
                    # Verificar se ainda temos capital suficiente
                    if capital_livre < entry_size_ajustado:
                        continue
                    
                    # Abrir posição LONG
                    preco_entrada = row_analise['close']
                    fees_abertura = entry_size_ajustado * config['leverage'] * config['fee_rate']
                    
                    nova_posicao = {
                        'asset': asset,
                        'tipo': 'LONG',
                        'preco_entrada': preco_entrada,
                        'size': entry_size_ajustado,
                        'leverage': config['leverage'],
                        'timestamp_abertura': timestamp,
                        'confluence': confluence,
                        'fees_abertura': fees_abertura,
                        'stop_loss_adaptativo': stop_loss_adaptativo
                    }
                    
                    posicoes_ativas.append(nova_posicao)
                    capital_livre -= entry_size_ajustado
                    
                    historico_trades.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'tipo': 'ABERTURA',
                        'pnl_bruto': 0,
                        'fees': fees_abertura,
                        'pnl_liquido': -fees_abertura,
                        'size': entry_size_ajustado,
                        'leverage': config['leverage'],
                        'confluence': confluence,
                        'protecoes': filtros['alertas']
                    })
                    
                    # Parar se atingimos limite de posições
                    if len(posicoes_ativas) >= filtros['max_positions']:
                        break
        
        elif not filtros['pode_abrir_posicao']:
            trades_bloqueados += 1
        
        # Registrar snapshot diário
        if timestamp.hour == 0:
            
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
                'daily_pnl': daily_pnl,
                'protecoes_ativas': len(filtros['alertas']) if 'alertas' in filtros else 0
            })
    
    print(f"\n✅ BACKTEST COM PROTEÇÕES CONCLUÍDO!")
    print(f"📊 Total de trades: {len(historico_trades)}")
    print(f"📅 Dias analisados: {len(historico_diario)}")
    print(f"🛡️ Trades bloqueados por proteções: {trades_bloqueados}")
    print(f"⏸️ Dias com operação pausada: {dias_sem_operacao}")
    print(f"🧮 Análises realizadas: {total_analises_realizadas:,}")
    
    return historico_diario, historico_trades

def main():
    """Função principal"""
    print("🛡️ BACKTEST COM PROTEÇÕES CONTRA QUEDAS")
    print("="*60)
    
    # Configuração
    config = configurar_estrategia_real()
    
    # Carregar dados
    dados_completos = carregar_dados_binance()
    
    if len(dados_completos) < 5:
        print("❌ Dados insuficientes para backtest!")
        return
    
    # Executar backtest protegido
    historico_diario, historico_trades = executar_backtest_protegido(dados_completos, config)
    
    # Gerar relatório
    resultado = gerar_relatorio_final(historico_diario, historico_trades, config)
    
    if resultado:
        # Comparar com backtest sem proteções
        print(f"\n🔍 COMPARAÇÃO COM BACKTEST SEM PROTEÇÕES:")
        print(f"="*50)
        print(f"📊 ROI com proteções: {resultado['roi_total']:+.2f}%")
        print(f"📉 Max Drawdown: {resultado['max_drawdown']:.2f}%")
        print(f"🎯 Win Rate: {resultado['win_rate']:.1f}%")
        print(f"📈 Total Trades: {resultado['total_trades']:,}")
        
        # Salvar resultados
        df_results = resultado['df_daily'].copy()
        df_results['capital_acumulado'] = df_results['capital_total']
        df_results['total_trades'] = df_results['trades_do_dia'].cumsum()
        df_results['date'] = pd.to_datetime(df_results['data'])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_results = f"backtest_protegido_binance_{timestamp}.csv"
        df_results.to_csv(filename_results, index=False)
        
        filename_trades = f"trades_protegido_binance_{timestamp}.csv"
        pd.DataFrame(historico_trades).to_csv(filename_trades, index=False)
        
        print(f"\n💾 ARQUIVOS SALVOS:")
        print(f"   📊 Resultados: {filename_results}")
        print(f"   📈 Trades: {filename_trades}")
        print(f"\n🎯 Compare os resultados com o backtest sem proteções!")
        
        return df_results, historico_trades
    
    return None, None

if __name__ == "__main__":
    main()
