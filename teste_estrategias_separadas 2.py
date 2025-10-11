#!/usr/bin/env python3
"""
🛡️ TESTE ESTRATÉGIAS SEPARADAS
=============================
✅ Estratégia 2: Só proteção em crashes severos (>20% drawdown)
✅ Estratégia 4: Proteções temporais (pausar após SL consecutivos)
🎯 Testar cada uma individualmente

Capital: $35 | Teste A/B das proteções
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from backtest_real_dados_binance import *
from estrategias_protecao_mercado import *
from datetime import timedelta

def executar_estrategia_2(dados_completos, config):
    """ESTRATÉGIA 2: Só proteção em crashes severos"""
    
    capital_livre = config['capital_total']
    posicoes_ativas = []
    historico_trades = []
    historico_diario = []
    capital_pico = config['capital_total']
    
    todas_datas = set()
    for asset, df in dados_completos.items():
        todas_datas.update(df['data'])
    
    datas_ordenadas = sorted(list(todas_datas))
    
    print(f"\n🛡️ ESTRATÉGIA 2: PROTEÇÃO EM CRASHES SEVEROS")
    print(f"📊 Só parar se drawdown >20% ou BTC cair >15% em 6h")
    
    total_analises = 0
    pausas_crash = 0
    drawdown_ativacoes = 0
    
    for i, timestamp in enumerate(datas_ordenadas):
        
        if i % 1000 == 0:
            progresso = (i / len(datas_ordenadas)) * 100
            print(f"   📊 {progresso:.1f}% | Pausas crash: {pausas_crash} | Drawdown ativado: {drawdown_ativacoes}")
        
        # Contar análises
        for asset, df_asset in dados_completos.items():
            if not df_asset[df_asset['data'] == timestamp].empty:
                total_analises += 1
        
        # Capital total atual
        capital_total_atual = capital_livre
        
        # Processar posições ativas
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
        
        # 🛡️ ESTRATÉGIA 2: Verificar apenas crashes severos
        pode_operar = True
        pode_abrir_posicao = True
        max_positions = config['max_positions']
        
        # Drawdown crítico (>20%)
        drawdown_atual = (capital_pico - capital_total_atual) / capital_pico if capital_pico > 0 else 0
        if drawdown_atual > 0.20:
            pode_operar = False
            drawdown_ativacoes += 1
        
        # Crash severo do BTC (>15% em 6h)
        if 'BTC' in dados_completos and pode_operar:
            dados_btc = dados_completos['BTC'][dados_completos['BTC']['data'] <= timestamp]
            if len(dados_btc) >= 6:
                ultimas_6h = dados_btc.tail(6)
                inicio = ultimas_6h.iloc[0]['valor_fechamento']
                fim = ultimas_6h.iloc[-1]['valor_fechamento']
                queda_pct = (fim - inicio) / inicio
                
                if queda_pct < -0.15:  # BTC caiu >15%
                    pode_abrir_posicao = False
                    max_positions = 4  # Reduzir posições
                    pausas_crash += 1
        
        # Verificar novas oportunidades
        if (pode_operar and pode_abrir_posicao and
            capital_livre >= config['entry_size'] and 
            len(posicoes_ativas) < max_positions):
            
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
                    
                    if capital_livre < config['entry_size']:
                        continue
                    
                    preco_entrada = row_analise['close']
                    fees_abertura = config['entry_size'] * config['leverage'] * config['fee_rate']
                    
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
                    
                    if len(posicoes_ativas) >= max_positions:
                        break
        
        # Snapshot diário
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
    
    print(f"\n✅ ESTRATÉGIA 2 CONCLUÍDA!")
    print(f"📊 Total trades: {len(historico_trades)}")
    print(f"🛡️ Pausas por crash: {pausas_crash}")
    print(f"📉 Ativações drawdown: {drawdown_ativacoes}")
    
    return historico_diario, historico_trades

def executar_estrategia_4(dados_completos, config):
    """ESTRATÉGIA 4: Proteções temporais"""
    
    capital_livre = config['capital_total']
    posicoes_ativas = []
    historico_trades = []
    historico_diario = []
    capital_pico = config['capital_total']
    
    todas_datas = set()
    for asset, df in dados_completos.items():
        todas_datas.update(df['data'])
    
    datas_ordenadas = sorted(list(todas_datas))
    
    print(f"\n🛡️ ESTRATÉGIA 4: PROTEÇÕES TEMPORAIS")
    print(f"⏸️ Pausar 6h após 5 SL consecutivos | Máx 3 SL/hora")
    
    total_analises = 0
    pausas_sl_consecutivos = 0
    pausas_sl_por_hora = 0
    
    for i, timestamp in enumerate(datas_ordenadas):
        
        if i % 1000 == 0:
            progresso = (i / len(datas_ordenadas)) * 100
            print(f"   📊 {progresso:.1f}% | Pausas SL consec: {pausas_sl_consecutivos} | Pausas SL/hora: {pausas_sl_por_hora}")
        
        # Contar análises
        for asset, df_asset in dados_completos.items():
            if not df_asset[df_asset['data'] == timestamp].empty:
                total_analises += 1
        
        # Capital total atual
        capital_total_atual = capital_livre
        
        # Processar posições ativas
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
        
        # 🛡️ ESTRATÉGIA 4: Proteções temporais
        pode_abrir_posicao = True
        
        # Verificar SL consecutivos
        if len(historico_trades) >= 5:
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
            
            # Pausar se 5+ SL consecutivos
            if sl_consecutivos >= 5:
                if ultimo_sl_timestamp:
                    tempo_pausa = timedelta(hours=6)
                    if timestamp < ultimo_sl_timestamp + tempo_pausa:
                        pode_abrir_posicao = False
                        pausas_sl_consecutivos += 1
        
        # Verificar SL por hora
        if pode_abrir_posicao:
            uma_hora_atras = timestamp - timedelta(hours=1)
            sl_ultima_hora = len([
                t for t in historico_trades 
                if (t.get('tipo') == 'STOP_LOSS' and 
                    t['timestamp'] >= uma_hora_atras and 
                    t['timestamp'] <= timestamp)
            ])
            
            if sl_ultima_hora >= 3:
                pode_abrir_posicao = False
                pausas_sl_por_hora += 1
        
        # Verificar novas oportunidades
        if (pode_abrir_posicao and
            capital_livre >= config['entry_size'] and 
            len(posicoes_ativas) < config['max_positions']):
            
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
                    
                    if capital_livre < config['entry_size']:
                        continue
                    
                    preco_entrada = row_analise['close']
                    fees_abertura = config['entry_size'] * config['leverage'] * config['fee_rate']
                    
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
                    
                    if len(posicoes_ativas) >= config['max_positions']:
                        break
        
        # Snapshot diário
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
    
    print(f"\n✅ ESTRATÉGIA 4 CONCLUÍDA!")
    print(f"📊 Total trades: {len(historico_trades)}")
    print(f"⏸️ Pausas SL consecutivos: {pausas_sl_consecutivos}")
    print(f"⏰ Pausas SL/hora: {pausas_sl_por_hora}")
    
    return historico_diario, historico_trades

def main():
    """Função principal"""
    print("🛡️ TESTE ESTRATÉGIAS SEPARADAS - 2 vs 4")
    print("="*50)
    
    config = configurar_estrategia_real()
    dados_completos = carregar_dados_binance()
    
    if len(dados_completos) < 5:
        print("❌ Dados insuficientes!")
        return
    
    # Testar Estratégia 2
    print("\n" + "="*60)
    print("🔥 TESTANDO ESTRATÉGIA 2: PROTEÇÃO EM CRASHES SEVEROS")
    print("="*60)
    
    historico_2, trades_2 = executar_estrategia_2(dados_completos, config)
    resultado_2 = gerar_relatorio_final(historico_2, trades_2, config)
    
    # Testar Estratégia 4
    print("\n" + "="*60)
    print("⏸️ TESTANDO ESTRATÉGIA 4: PROTEÇÕES TEMPORAIS")
    print("="*60)
    
    historico_4, trades_4 = executar_estrategia_4(dados_completos, config)
    resultado_4 = gerar_relatorio_final(historico_4, trades_4, config)
    
    # Comparação final
    print("\n" + "="*70)
    print("📊 COMPARAÇÃO FINAL: ESTRATÉGIA 2 vs 4")
    print("="*70)
    
    if resultado_2 and resultado_4:
        print(f"🔥 ESTRATÉGIA 2 (Crashes severos):")
        print(f"   📈 ROI: {resultado_2['roi_total']:+.2f}%")
        print(f"   📉 Max Drawdown: {resultado_2['max_drawdown']:.2f}%")
        print(f"   🎯 Win Rate: {resultado_2['win_rate']:.1f}%")
        print(f"   📊 Total Trades: {resultado_2['total_trades']:,}")
        
        print(f"\n⏸️ ESTRATÉGIA 4 (Proteções temporais):")
        print(f"   📈 ROI: {resultado_4['roi_total']:+.2f}%")
        print(f"   📉 Max Drawdown: {resultado_4['max_drawdown']:.2f}%")
        print(f"   🎯 Win Rate: {resultado_4['win_rate']:.1f}%")
        print(f"   📊 Total Trades: {resultado_4['total_trades']:,}")
        
        # Determinar vencedor
        if resultado_2['roi_total'] > resultado_4['roi_total']:
            print(f"\n🏆 VENCEDORA: ESTRATÉGIA 2")
            print(f"💰 Diferença: {resultado_2['roi_total'] - resultado_4['roi_total']:+.2f}% ROI")
        else:
            print(f"\n🏆 VENCEDORA: ESTRATÉGIA 4")
            print(f"💰 Diferença: {resultado_4['roi_total'] - resultado_2['roi_total']:+.2f}% ROI")
        
        # Salvar resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df_2 = resultado_2['df_daily'].copy()
        df_2['date'] = pd.to_datetime(df_2['data'])
        df_2.to_csv(f"estrategia_2_crashes_{timestamp}.csv", index=False)
        
        df_4 = resultado_4['df_daily'].copy()
        df_4['date'] = pd.to_datetime(df_4['data'])
        df_4.to_csv(f"estrategia_4_temporais_{timestamp}.csv", index=False)
        
        print(f"\n💾 Resultados salvos:")
        print(f"   📂 Estratégia 2: estrategia_2_crashes_{timestamp}.csv")
        print(f"   📂 Estratégia 4: estrategia_4_temporais_{timestamp}.csv")
        
        print(f"\n📋 REFERÊNCIA:")
        print(f"   🔥 Sem proteções: +5.551% ROI | -92.64% drawdown")
        print(f"   ⚖️ Ambas juntas: +1.589% ROI | -98.71% drawdown")

if __name__ == "__main__":
    main()
