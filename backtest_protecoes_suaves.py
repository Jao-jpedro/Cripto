#!/usr/bin/env python3
"""
🛡️ PROTEÇÕES SUAVES - ESTRATÉGIAS 2 & 4
=======================================
✅ Estratégia 2: Proteção só em crashes severos (>20% drawdown)
✅ Estratégia 4: Pausar 6h após 5 SL consecutivos
🎯 Objetivo: Proteger sem perder muitas oportunidades

Capital: $35 | Proteções calibradas
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from backtest_real_dados_binance import *
from estrategias_protecao_mercado import *
from datetime import timedelta

def configurar_protecoes_suaves():
    """Configurações de proteção mais suaves"""
    return {
        # Estratégia 2: Só parar em crashes severos
        'drawdown_critico': 0.20,           # Só parar se perder >20% do pico
        'crash_severo_btc': 0.15,           # BTC caindo >15% em 6h
        'volatilidade_extrema': 12.0,       # ATR >12% = extremo
        
        # Estratégia 4: Proteções temporais
        'max_sl_consecutivos': 5,           # Pausar após 5 SL seguidos
        'pausa_apos_sl': 6,                 # Pausar 6 horas
        'max_sl_por_hora': 3,               # Máx 3 SL por hora
        'pausa_volatilidade': 2,            # Pausar 2h se ATR >10%
        
        # Proteções mantidas
        'max_positions_crash': 4,           # Reduzir posições em crash
        'size_reduction_volatil': 0.5,     # Reduzir size 50% se volátil
    }

def verificar_sl_consecutivos_temporal(historico_trades, timestamp, config_suave):
    """Verifica SL consecutivos com controle temporal"""
    if len(historico_trades) < config_suave['max_sl_consecutivos']:
        return False, 0, None
    
    # Contar SL consecutivos recentes
    trades_recentes = [t for t in historico_trades[-20:] 
                      if t.get('tipo') in ['STOP_LOSS', 'TAKE_PROFIT']]
    
    sl_consecutivos = 0
    ultimo_sl_timestamp = None
    
    for trade in reversed(trades_recentes):
        if trade['tipo'] == 'STOP_LOSS':
            sl_consecutivos += 1
            if ultimo_sl_timestamp is None:
                ultimo_sl_timestamp = trade['timestamp']
        else:
            break
    
    # Verificar se deve pausar
    if sl_consecutivos >= config_suave['max_sl_consecutivos']:
        if ultimo_sl_timestamp:
            tempo_pausa = timedelta(hours=config_suave['pausa_apos_sl'])
            if timestamp < ultimo_sl_timestamp + tempo_pausa:
                return True, sl_consecutivos, ultimo_sl_timestamp
    
    return False, sl_consecutivos, ultimo_sl_timestamp

def verificar_sl_por_hora(historico_trades, timestamp):
    """Verifica quantos SL aconteceram na última hora"""
    uma_hora_atras = timestamp - timedelta(hours=1)
    
    sl_ultima_hora = len([
        t for t in historico_trades 
        if (t.get('tipo') == 'STOP_LOSS' and 
            t['timestamp'] >= uma_hora_atras and 
            t['timestamp'] <= timestamp)
    ])
    
    return sl_ultima_hora

def detectar_crash_severo(dados_btc, config_suave):
    """Detecta apenas crashes severos"""
    if len(dados_btc) < 6:
        return False, 0.0, 0.0
    
    ultimas_6h = dados_btc.tail(6)
    
    inicio = ultimas_6h.iloc[0]['valor_fechamento']
    fim = ultimas_6h.iloc[-1]['valor_fechamento']
    queda_pct = (fim - inicio) / inicio
    
    volatilidade_atual = ultimas_6h['valor_fechamento'].std() / ultimas_6h['valor_fechamento'].mean()
    
    # Crash SEVERO (mais restritivo)
    crash_severo = (
        queda_pct < -config_suave['crash_severo_btc'] or  # Queda >15%
        volatilidade_atual > (config_suave['volatilidade_extrema'] / 100)  # Volat >12%
    )
    
    return crash_severo, queda_pct, volatilidade_atual

def aplicar_protecoes_suaves(config, dados_completos, timestamp, historico_trades, capital_atual, capital_pico):
    """Aplica proteções suaves (estratégias 2 & 4)"""
    
    config_suave = configurar_protecoes_suaves()
    
    resultado = {
        'pode_operar': True,
        'pode_abrir_posicao': True,
        'max_positions': config['max_positions'],
        'entry_size': config['entry_size'],
        'stop_loss_pct': config['stop_loss_pct'],
        'alertas': []
    }
    
    # ESTRATÉGIA 2: Só parar em crashes severos (>20% drawdown)
    drawdown_atual = calcular_drawdown_atual(capital_atual, capital_pico)
    if drawdown_atual > config_suave['drawdown_critico']:
        resultado['pode_operar'] = False
        resultado['alertas'].append(f"🚨 DRAWDOWN CRÍTICO: {drawdown_atual*100:.1f}%")
        return resultado
    
    # Detectar crash severo do BTC
    if 'BTC' in dados_completos:
        dados_btc = dados_completos['BTC'][dados_completos['BTC']['data'] <= timestamp]
        crash_severo, queda_pct, volatilidade = detectar_crash_severo(dados_btc, config_suave)
        
        if crash_severo:
            resultado['pode_abrir_posicao'] = False
            resultado['max_positions'] = config_suave['max_positions_crash']
            resultado['alertas'].append(f"🌊 CRASH SEVERO: BTC {queda_pct*100:.1f}%")
    
    # ESTRATÉGIA 4: Proteções temporais
    # 4a. Pausar após 5 SL consecutivos
    pausar_sl, count_sl, ultimo_sl = verificar_sl_consecutivos_temporal(
        historico_trades, timestamp, config_suave
    )
    
    if pausar_sl:
        resultado['pode_abrir_posicao'] = False
        tempo_restante = (ultimo_sl + timedelta(hours=config_suave['pausa_apos_sl'])) - timestamp
        resultado['alertas'].append(f"⏸️ PAUSA: {count_sl} SL consecutivos (resta {tempo_restante})")
    
    # 4b. Limite de SL por hora
    sl_por_hora = verificar_sl_por_hora(historico_trades, timestamp)
    if sl_por_hora >= config_suave['max_sl_por_hora']:
        resultado['pode_abrir_posicao'] = False
        resultado['alertas'].append(f"⏰ LIMITE: {sl_por_hora} SL na última hora")
    
    # 4c. Reduzir size em alta volatilidade (sem parar completamente)
    if len(resultado['alertas']) == 0:  # Só se não há outras proteções ativas
        for asset, df_asset in dados_completos.items():
            row_atual = df_asset[df_asset['data'] == timestamp]
            if not row_atual.empty:
                df_ate_agora = df_asset[df_asset['data'] <= timestamp].copy()
                if len(df_ate_agora) >= 50:
                    df_indicators = calcular_indicadores_reais(df_ate_agora)
                    row_analise = df_indicators.iloc[-1]
                    
                    if pd.notna(row_analise.get('atr_pct')) and row_analise['atr_pct'] > 10.0:
                        resultado['entry_size'] = config['entry_size'] * config_suave['size_reduction_volatil']
                        resultado['alertas'].append(f"📉 SIZE REDUZIDO: Alta volatilidade")
                        break
    
    return resultado

def executar_backtest_protecoes_suaves(dados_completos, config):
    """Executa backtest com proteções suaves"""
    
    # Estado inicial
    capital_livre = config['capital_total']
    posicoes_ativas = []
    historico_trades = []
    historico_diario = []
    capital_pico = config['capital_total']
    
    # Encontrar período
    todas_datas = set()
    for asset, df in dados_completos.items():
        todas_datas.update(df['data'])
    
    datas_ordenadas = sorted(list(todas_datas))
    periodo_inicio = datas_ordenadas[0]
    periodo_fim = datas_ordenadas[-1]
    
    print(f"\n🛡️ BACKTEST COM PROTEÇÕES SUAVES (Estratégias 2 & 4)")
    print(f"📅 Período: {periodo_inicio.date()} → {periodo_fim.date()}")
    print(f"🏃 Total pontos: {len(datas_ordenadas)}")
    print(f"💰 Capital inicial: ${config['capital_total']}")
    
    config_suave = configurar_protecoes_suaves()
    print(f"🛡️ Drawdown crítico: {config_suave['drawdown_critico']*100}%")
    print(f"⏸️ Pausar após: {config_suave['max_sl_consecutivos']} SL consecutivos")
    print(f"⏰ Pausa por: {config_suave['pausa_apos_sl']} horas")
    
    total_analises_realizadas = 0
    trades_bloqueados = 0
    pausas_ativadas = 0
    size_reduzidos = 0
    
    # Processar cada timestamp
    for i, timestamp in enumerate(datas_ordenadas):
        
        # Progresso
        if i % 1000 == 0:
            progresso = (i / len(datas_ordenadas)) * 100
            print(f"   📊 {progresso:.1f}% | Bloqueados: {trades_bloqueados} | Pausas: {pausas_ativadas} | Size↓: {size_reduzidos}")
        
        # Contar análises
        for asset, df_asset in dados_completos.items():
            if not df_asset[df_asset['data'] == timestamp].empty:
                total_analises_realizadas += 1
        
        # Capital total atual
        capital_total_atual = capital_livre
        
        # Verificar posições ativas
        posicoes_para_remover = []
        
        for j, pos in enumerate(posicoes_ativas):
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
                capital_posicao = pnl_nao_realizado * pos['size'] + pos['size']
                capital_total_atual += capital_posicao - pos['size']
                
                # Stop Loss
                if pnl_nao_realizado <= -config['stop_loss_pct']:
                    pnl_realizado = -config['stop_loss_pct'] * pos['size']
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
                    
                # Take Profit
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
        
        # 🛡️ APLICAR PROTEÇÕES SUAVES
        filtros = aplicar_protecoes_suaves(
            config, dados_completos, timestamp, 
            historico_trades, capital_total_atual, capital_pico
        )
        
        # Contar ativações de proteção
        if not filtros['pode_operar']:
            pausas_ativadas += 1
        elif not filtros['pode_abrir_posicao']:
            trades_bloqueados += 1
        elif filtros['entry_size'] < config['entry_size']:
            size_reduzidos += 1
        
        # Verificar novas oportunidades
        if (filtros['pode_operar'] and 
            filtros['pode_abrir_posicao'] and
            capital_livre >= filtros['entry_size'] and 
            len(posicoes_ativas) < filtros['max_positions']):
            
            for asset, df_asset in dados_completos.items():
                
                asset_ocupado = any(pos['asset'] == asset for pos in posicoes_ativas)
                if asset_ocupado:
                    continue
                
                row_atual = df_asset[df_asset['data'] == timestamp]
                if row_atual.empty:
                    continue
                
                df_ate_agora = df_asset[df_asset['data'] <= timestamp].copy()
                if len(df_ate_agora) < 50:
                    continue
                
                df_indicators = calcular_indicadores_reais(df_ate_agora)
                row_analise = df_indicators.iloc[-1]
                
                confluence = calcular_confluence_real(row_analise, config)
                
                if confluence >= config['min_confluence']:
                    
                    entry_size_final = filtros['entry_size']
                    
                    if capital_livre < entry_size_final:
                        continue
                    
                    preco_entrada = row_analise['close']
                    fees_abertura = entry_size_final * config['leverage'] * config['fee_rate']
                    
                    nova_posicao = {
                        'asset': asset,
                        'tipo': 'LONG',
                        'preco_entrada': preco_entrada,
                        'size': entry_size_final,
                        'leverage': config['leverage'],
                        'timestamp_abertura': timestamp,
                        'confluence': confluence,
                        'fees_abertura': fees_abertura
                    }
                    
                    posicoes_ativas.append(nova_posicao)
                    capital_livre -= entry_size_final
                    
                    historico_trades.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'tipo': 'ABERTURA',
                        'pnl_bruto': 0,
                        'fees': fees_abertura,
                        'pnl_liquido': -fees_abertura,
                        'size': entry_size_final,
                        'leverage': config['leverage'],
                        'confluence': confluence,
                        'protecoes': filtros['alertas']
                    })
                    
                    if len(posicoes_ativas) >= filtros['max_positions']:
                        break
        
        # Registrar snapshot diário
        if timestamp.hour == 0:
            
            capital_total_final = capital_livre
            
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
    
    print(f"\n✅ BACKTEST PROTEÇÕES SUAVES CONCLUÍDO!")
    print(f"📊 Total trades: {len(historico_trades)}")
    print(f"🛡️ Trades bloqueados: {trades_bloqueados}")
    print(f"⏸️ Pausas ativadas: {pausas_ativadas}")
    print(f"📉 Size reduzidos: {size_reduzidos}")
    print(f"🧮 Análises: {total_analises_realizadas:,}")
    
    return historico_diario, historico_trades

def main():
    """Função principal"""
    print("🛡️ BACKTEST PROTEÇÕES SUAVES - ESTRATÉGIAS 2 & 4")
    print("="*60)
    
    config = configurar_estrategia_real()
    dados_completos = carregar_dados_binance()
    
    if len(dados_completos) < 5:
        print("❌ Dados insuficientes!")
        return
    
    historico_diario, historico_trades = executar_backtest_protecoes_suaves(dados_completos, config)
    resultado = gerar_relatorio_final(historico_diario, historico_trades, config)
    
    if resultado:
        print(f"\n📊 COMPARAÇÃO COM OUTROS BACKTESTS:")
        print(f"="*40)
        print(f"🔥 Sem proteções: +5.551% ROI | -92.64% drawdown")
        print(f"🛡️ Proteções totais: -2.10% ROI | -2.10% drawdown") 
        print(f"⚖️ Proteções suaves: {resultado['roi_total']:+.2f}% ROI | {resultado['max_drawdown']:.2f}% drawdown")
        
        df_results = resultado['df_daily'].copy()
        df_results['capital_acumulado'] = df_results['capital_total']
        df_results['date'] = pd.to_datetime(df_results['data'])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_protecoes_suaves_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        
        print(f"\n💾 Resultados salvos: {filename}")
        print(f"🎯 Estratégias 2 & 4 implementadas com sucesso!")

if __name__ == "__main__":
    main()
