#!/usr/bin/env python3
"""
🛡️ SISTEMA DE PROTEÇÃO ESTRATÉGIA 2 - INTEGRAÇÃO TRADINGV4
==========================================================
✅ Proteção por Drawdown Crítico (>20%)
✅ Proteção por Crash do BTC (>15% em 6h)
🎯 Integração das proteções do backtest no sistema real

Capital: $35 | Proteções apenas em situações críticas
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import math
import requests
import time

class ProtecaoEstrategia2:
    """
    Sistema de proteção Estratégia 2: Crashes Severos
    - Bloqueia operações em drawdown >20%
    - Reduz posições em crash BTC >15% em 6h
    """
    
    def __init__(self, debug=True):
        self.debug = debug
        self.capital_pico = None
        self.historico_capital = []
        self.ultima_verificacao = None
        
    def _log(self, message: str, level: str = "INFO") -> None:
        """Log com prefixo de proteção"""
        print(f"[{level}] [PROTECAO_2] {message}", flush=True)
    
    def atualizar_capital(self, capital_atual: float) -> None:
        """Atualiza capital atual e pico histórico"""
        timestamp = datetime.now(timezone.utc)
        
        # Primeira vez
        if self.capital_pico is None:
            self.capital_pico = capital_atual
            self._log(f"Capital inicial registrado: ${capital_atual:.2f}", level="DEBUG")
        
        # Atualizar pico se necessário
        if capital_atual > self.capital_pico:
            old_pico = self.capital_pico
            self.capital_pico = capital_atual
            self._log(f"Novo pico de capital: ${capital_atual:.2f} (anterior: ${old_pico:.2f})", level="INFO")
        
        # Salvar histórico
        self.historico_capital.append({
            'timestamp': timestamp,
            'capital': capital_atual,
            'pico': self.capital_pico
        })
        
        # Manter apenas últimos 1000 registros
        if len(self.historico_capital) > 1000:
            self.historico_capital = self.historico_capital[-1000:]
        
        self.ultima_verificacao = timestamp
    
    def verificar_drawdown_critico(self, capital_atual: float) -> bool:
        """
        Verifica se drawdown atual >20%
        Retorna True se deve BLOQUEAR operações
        """
        if self.capital_pico is None or self.capital_pico <= 0:
            return False
        
        drawdown_atual = (self.capital_pico - capital_atual) / self.capital_pico
        
        if drawdown_atual > 0.20:  # >20% de drawdown
            self._log(
                f"🚨 DRAWDOWN CRÍTICO: {drawdown_atual*100:.2f}% "
                f"(${capital_atual:.2f} vs pico ${self.capital_pico:.2f}) - BLOQUEANDO OPERAÇÕES", 
                level="WARN"
            )
            return True
        
        if self.debug and drawdown_atual > 0.10:  # Log se >10%
            self._log(
                f"Drawdown atual: {drawdown_atual*100:.2f}% "
                f"(${capital_atual:.2f} vs pico ${self.capital_pico:.2f})", 
                level="INFO"
            )
        
        return False
    
    def obter_dados_btc_6h(self) -> pd.DataFrame:
        """Obtém dados das últimas 6 horas do BTC da Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'limit': 6  # Últimas 6 horas
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or len(data) < 2:
                self._log("Dados BTC insuficientes para análise de crash", level="WARN")
                return pd.DataFrame()
            
            # Converter para DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])
            
            df['close'] = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['timestamp', 'close']].copy()
            
        except Exception as e:
            self._log(f"Erro ao obter dados BTC: {e}", level="ERROR")
            return pd.DataFrame()
    
    def verificar_crash_btc(self) -> tuple[bool, bool]:
        """
        Verifica crash severo do BTC (>15% em 6h)
        Retorna: (deve_bloquear_novas, deve_reduzir_posicoes)
        """
        try:
            df_btc = self.obter_dados_btc_6h()
            
            if df_btc.empty or len(df_btc) < 2:
                return False, False
            
            preco_inicio = df_btc.iloc[0]['close']
            preco_fim = df_btc.iloc[-1]['close']
            
            if preco_inicio <= 0:
                return False, False
            
            queda_pct = (preco_fim - preco_inicio) / preco_inicio
            
            if queda_pct < -0.15:  # BTC caiu >15%
                self._log(
                    f"🔥 CRASH SEVERO BTC: {queda_pct*100:.2f}% em 6h "
                    f"(${preco_inicio:.0f} → ${preco_fim:.0f}) - PROTEÇÕES ATIVADAS", 
                    level="WARN"
                )
                return True, True  # Bloquear novas + reduzir posições
            
            elif queda_pct < -0.10:  # Log se >10%
                self._log(
                    f"BTC em queda: {queda_pct*100:.2f}% em 6h "
                    f"(${preco_inicio:.0f} → ${preco_fim:.0f})", 
                    level="INFO"
                )
            
            return False, False
            
        except Exception as e:
            self._log(f"Erro ao verificar crash BTC: {e}", level="ERROR")
            return False, False
    
    def pode_abrir_posicao(self, capital_atual: float, max_positions_atual: int = 8) -> tuple[bool, int]:
        """
        Verifica se pode abrir nova posição
        Retorna: (pode_abrir, max_positions_ajustado)
        """
        # Atualizar capital
        self.atualizar_capital(capital_atual)
        
        # 1. Verificar drawdown crítico
        if self.verificar_drawdown_critico(capital_atual):
            return False, 0  # BLOQUEIO TOTAL
        
        # 2. Verificar crash BTC
        bloquear_novas, reduzir_posicoes = self.verificar_crash_btc()
        
        if bloquear_novas:
            if reduzir_posicoes:
                # Crash severo: reduzir posições máximas
                max_positions_reduzido = max(1, max_positions_atual // 2)  # Reduzir pela metade
                self._log(
                    f"Crash BTC: reduzindo posições máximas de {max_positions_atual} para {max_positions_reduzido}", 
                    level="WARN"
                )
                return False, max_positions_reduzido  # Não abrir novas, mas manter as existentes
            else:
                return False, max_positions_atual  # Só bloquear novas
        
        # Tudo OK
        return True, max_positions_atual
    
    def relatorio_status(self) -> dict:
        """Gera relatório do status atual das proteções"""
        if not self.historico_capital:
            return {"status": "sem_dados"}
        
        ultimo = self.historico_capital[-1]
        capital_atual = ultimo['capital']
        capital_pico = ultimo['pico']
        
        drawdown_atual = (capital_pico - capital_atual) / capital_pico if capital_pico > 0 else 0
        
        # Verificar BTC
        bloquear_btc, reduzir_btc = self.verificar_crash_btc()
        
        return {
            "status": "ativo",
            "capital_atual": capital_atual,
            "capital_pico": capital_pico,
            "drawdown_pct": drawdown_atual * 100,
            "drawdown_critico": drawdown_atual > 0.20,
            "crash_btc_detectado": bloquear_btc,
            "deve_reduzir_posicoes": reduzir_btc,
            "pode_operar": not (drawdown_atual > 0.20 or bloquear_btc),
            "ultima_verificacao": self.ultima_verificacao.isoformat() if self.ultima_verificacao else None
        }

# Instância global do sistema de proteção
PROTECAO_SISTEMA = ProtecaoEstrategia2(debug=True)

def aplicar_protecoes_estrategia_2(capital_atual: float, max_positions: int = 8) -> tuple[bool, int]:
    """
    Função principal para aplicar proteções da Estratégia 2
    
    Args:
        capital_atual: Capital atual em USDC
        max_positions: Número máximo de posições normalmente
    
    Returns:
        (pode_abrir_posicao, max_positions_ajustado)
    """
    return PROTECAO_SISTEMA.pode_abrir_posicao(capital_atual, max_positions)

def obter_status_protecoes() -> dict:
    """Obtém status atual das proteções"""
    return PROTECAO_SISTEMA.relatorio_status()

def resetar_protecoes():
    """Reseta o sistema de proteções (usar com cuidado)"""
    global PROTECAO_SISTEMA
    PROTECAO_SISTEMA = ProtecaoEstrategia2(debug=True)
    print("[INFO] [PROTECAO_2] Sistema de proteções resetado", flush=True)

# Teste do sistema
if __name__ == "__main__":
    print("🛡️ TESTE SISTEMA PROTEÇÃO ESTRATÉGIA 2")
    print("="*50)
    
    # Simular cenários
    cenarios = [
        {"capital": 35.0, "desc": "Capital inicial"},
        {"capital": 45.0, "desc": "Capital +28%"},
        {"capital": 36.0, "desc": "Drawdown 20% (crítico)"},
        {"capital": 30.0, "desc": "Drawdown 33% (muito crítico)"},
        {"capital": 50.0, "desc": "Novo pico"},
        {"capital": 42.0, "desc": "Drawdown 16% (OK)"},
    ]
    
    for i, cenario in enumerate(cenarios):
        print(f"\n📊 Cenário {i+1}: {cenario['desc']}")
        
        pode_abrir, max_pos = aplicar_protecoes_estrategia_2(cenario['capital'])
        status = obter_status_protecoes()
        
        print(f"   💰 Capital: ${cenario['capital']:.2f}")
        print(f"   📈 Pico: ${status['capital_pico']:.2f}")
        print(f"   📉 Drawdown: {status['drawdown_pct']:.1f}%")
        print(f"   🛡️ Pode abrir: {'✅' if pode_abrir else '❌'}")
        print(f"   🎯 Max posições: {max_pos}")
        print(f"   🔥 Crash BTC: {'⚠️' if status['crash_btc_detectado'] else '✅'}")
        
        time.sleep(0.5)  # Simular passagem de tempo
    
    print(f"\n📋 Status Final:")
    status_final = obter_status_protecoes()
    for key, value in status_final.items():
        print(f"   {key}: {value}")
