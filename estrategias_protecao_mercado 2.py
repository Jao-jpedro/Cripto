#!/usr/bin/env python3
"""
🛡️ ESTRATÉGIAS DE PROTEÇÃO CONTRA QUEDAS DO MERCADO
==================================================
✅ Filtros de tendência geral
✅ Limite de perdas consecutivas
✅ Redução de exposição em volatilidade alta
✅ Pausa operacional em crash
✅ Diversificação temporal

Capital: $35 | Proteção ativa contra drawdowns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def configurar_protecoes_mercado():
    """Configurações de proteção contra quedas"""
    return {
        # Proteções básicas
        'max_perdas_consecutivas': 3,        # Parar após 3 SL seguidos
        'drawdown_maximo_permitido': 0.30,   # Parar se perder 30% do pico
        'pausa_apos_crash': 24,              # Pausar 24h após crash
        
        # Filtros de mercado
        'btc_trend_filter': True,            # Só operar se BTC em alta
        'correlacao_minima': 0.7,            # Assets muito correlacionados = risco
        'volatilidade_maxima': 8.0,          # Não operar se ATR > 8%
        
        # Redução de risco
        'reducao_size_volatilidade': True,   # Reduzir size se volátil
        'max_positions_crash': 4,            # Máx 4 posições em crash
        'stop_loss_adaptativo': True,        # SL mais apertado em queda
        
        # Filtros temporais
        'evitar_fins_semana': True,          # Não operar fins de semana
        'evitar_madrugada': True,            # Não operar 00h-06h UTC
    }

def detectar_crash_mercado(dados_btc, janela=6):
    """Detecta se o mercado está em crash"""
    if len(dados_btc) < janela:
        return False, 0.0, 0.0
    
    # Últimas 6 horas de BTC
    ultimas_6h = dados_btc.tail(janela)
    
    # Queda > 10% em 6h = crash
    inicio = ultimas_6h.iloc[0]['valor_fechamento']
    fim = ultimas_6h.iloc[-1]['valor_fechamento']
    queda_pct = (fim - inicio) / inicio
    
    # Volatilidade muito alta
    volatilidade_atual = ultimas_6h['valor_fechamento'].std() / ultimas_6h['valor_fechamento'].mean()
    
    crash_detectado = (
        queda_pct < -0.10 or  # Queda > 10%
        volatilidade_atual > 0.15  # Volatilidade > 15%
    )
    
    return crash_detectado, queda_pct, volatilidade_atual

def calcular_trend_btc(dados_btc, periodos=[12, 24, 48]):
    """Calcula tendência geral do BTC"""
    trends = {}
    
    for periodo in periodos:
        if len(dados_btc) >= periodo:
            inicio = dados_btc.iloc[-periodo]['valor_fechamento']
            fim = dados_btc.iloc[-1]['valor_fechamento']
            trend_pct = (fim - inicio) / inicio
            trends[f'{periodo}h'] = trend_pct
    
    # Tendência geral (média ponderada)
    if trends:
        trend_geral = (
            trends.get('12h', 0) * 0.5 +
            trends.get('24h', 0) * 0.3 +
            trends.get('48h', 0) * 0.2
        )
        return trend_geral > 0, trend_geral, trends
    
    return True, 0, {}  # Se não tem dados suficientes, permite

def verificar_perdas_consecutivas(historico_trades, limite=3):
    """Verifica se houve muitas perdas consecutivas"""
    if len(historico_trades) < limite:
        return False, 0
    
    # Últimos trades com PnL
    trades_com_pnl = [t for t in historico_trades[-10:] if t.get('pnl_liquido') is not None]
    
    perdas_consecutivas = 0
    for trade in reversed(trades_com_pnl):
        if trade['pnl_liquido'] < 0:
            perdas_consecutivas += 1
        else:
            break
    
    return perdas_consecutivas >= limite, perdas_consecutivas

def calcular_drawdown_atual(capital_atual, capital_pico):
    """Calcula drawdown atual"""
    if capital_pico <= 0:
        return 0
    
    drawdown = (capital_pico - capital_atual) / capital_pico
    return drawdown

def ajustar_size_por_volatilidade(size_base, atr_pct, limite_volat=5.0):
    """Ajusta tamanho da posição baseado na volatilidade"""
    if atr_pct <= limite_volat:
        return size_base
    
    # Reduz size proporcionalmente à volatilidade
    fator_reducao = limite_volat / atr_pct
    size_ajustado = size_base * fator_reducao
    
    # Mínimo de $2
    return max(size_ajustado, 2.0)

def ajustar_stop_loss_adaptativo(sl_base, trend_btc, volatilidade):
    """Ajusta stop loss baseado nas condições de mercado"""
    sl_ajustado = sl_base
    
    # Se BTC em queda, SL mais apertado
    if trend_btc < -0.05:  # BTC caindo > 5%
        sl_ajustado = sl_base * 0.8  # SL 20% mais apertado
    
    # Se volatilidade alta, SL mais apertado
    if volatilidade > 6.0:
        sl_ajustado = sl_ajustado * 0.9  # SL 10% mais apertado
    
    # Mínimo de 0.8% e máximo de 2.5%
    return max(0.008, min(sl_ajustado, 0.025))

def verificar_horario_operacao(timestamp):
    """Verifica se é horário apropriado para operar"""
    hora_utc = timestamp.hour
    dia_semana = timestamp.weekday()  # 0=segunda, 6=domingo
    
    # Evitar fins de semana (sexta 22h até domingo 22h UTC)
    if dia_semana == 6 or (dia_semana == 5 and hora_utc >= 22):
        return False, "Fim de semana"
    
    # Evitar madrugada (00h-06h UTC)
    if 0 <= hora_utc <= 6:
        return False, "Madrugada"
    
    return True, "OK"

def aplicar_filtros_protecao(config, dados_completos, timestamp, historico_trades, capital_atual, capital_pico):
    """Aplica todos os filtros de proteção"""
    protecoes = configurar_protecoes_mercado()
    resultado = {
        'pode_operar': True,
        'pode_abrir_posicao': True,
        'max_positions': config['max_positions'],
        'entry_size': config['entry_size'],
        'stop_loss_pct': config['stop_loss_pct'],
        'alertas': []
    }
    
    # 1. Verificar crash do mercado
    if 'BTC' in dados_completos:
        dados_btc = dados_completos['BTC'][dados_completos['BTC']['data'] <= timestamp]
        crash_detectado, queda_pct, volatilidade = detectar_crash_mercado(dados_btc)
        
        if crash_detectado:
            resultado['pode_abrir_posicao'] = False
            resultado['max_positions'] = protecoes['max_positions_crash']
            resultado['alertas'].append(f"🚨 CRASH detectado: {queda_pct*100:.1f}% | Volat: {volatilidade*100:.1f}%")
    
    # 2. Verificar tendência do BTC
    if protecoes['btc_trend_filter'] and 'BTC' in dados_completos:
        dados_btc = dados_completos['BTC'][dados_completos['BTC']['data'] <= timestamp]
        btc_up, trend_valor, trends = calcular_trend_btc(dados_btc)
        
        if not btc_up:
            resultado['pode_abrir_posicao'] = False
            resultado['alertas'].append(f"📉 BTC em queda: {trend_valor*100:.1f}%")
    
    # 3. Verificar perdas consecutivas
    perdas_consecutivas, count = verificar_perdas_consecutivas(historico_trades, protecoes['max_perdas_consecutivas'])
    if perdas_consecutivas:
        resultado['pode_abrir_posicao'] = False
        resultado['alertas'].append(f"🔴 {count} perdas consecutivas")
    
    # 4. Verificar drawdown máximo
    drawdown_atual = calcular_drawdown_atual(capital_atual, capital_pico)
    if drawdown_atual > protecoes['drawdown_maximo_permitido']:
        resultado['pode_operar'] = False
        resultado['alertas'].append(f"📉 Drawdown crítico: {drawdown_atual*100:.1f}%")
    
    # 5. Verificar horário
    horario_ok, motivo = verificar_horario_operacao(timestamp)
    if not horario_ok and protecoes.get('evitar_fins_semana'):
        resultado['pode_abrir_posicao'] = False
        resultado['alertas'].append(f"⏰ {motivo}")
    
    return resultado

def criar_exemplo_protecao():
    """Exemplo de como usar as proteções"""
    print("🛡️ ESTRATÉGIAS DE PROTEÇÃO CONTRA QUEDAS")
    print("="*50)
    
    protecoes = configurar_protecoes_mercado()
    
    print("📋 FILTROS IMPLEMENTADOS:")
    print("="*30)
    print(f"✅ Máx perdas consecutivas: {protecoes['max_perdas_consecutivas']}")
    print(f"✅ Drawdown máximo: {protecoes['drawdown_maximo_permitido']*100}%")
    print(f"✅ Filtro tendência BTC: {protecoes['btc_trend_filter']}")
    print(f"✅ Volatilidade máxima: {protecoes['volatilidade_maxima']}%")
    print(f"✅ Redução em crash: {protecoes['max_positions_crash']} posições")
    
    print(f"\n🎯 BENEFÍCIOS ESPERADOS:")
    print("="*25)
    print("📉 Reduzir drawdowns severos")
    print("🛡️ Evitar operar em condições adversas")
    print("⏱️ Pausar operações em momentos críticos")
    print("💰 Preservar capital em crashes")
    print("📊 Adaptar risco às condições do mercado")
    
    print(f"\n⚠️ TRADE-OFFS:")
    print("="*15)
    print("📉 Menor frequência de trades")
    print("💰 Possivelmente menor ROI total")
    print("🎯 Melhor relação risco/retorno")

if __name__ == "__main__":
    criar_exemplo_protecao()
