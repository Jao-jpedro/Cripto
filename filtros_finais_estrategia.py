#!/usr/bin/env python3
"""
🔬 FILTROS FINAIS DA ESTRATÉGIA OTIMIZADA
=========================================
🎯 Capital: $35 | Entradas: $4 | ROI: +11.525%
💰 Baseado na análise "Mais Permissivo" com configuração ótima

CONFIGURAÇÃO DESCOBERTA VENCEDORA:
- Confluence: 25% (mais permissivo)
- Volume: 1.2x (mais oportunidades)
- ATR Breakout: 0.7 (menos restritivo)
- SL: 1.5% | TP: 12% | Leverage: 10x
"""

def get_estrategia_final_filtros():
    """Retorna todos os filtros finais da estratégia otimizada"""
    
    print("🔬 FILTROS FINAIS DA ESTRATÉGIA OTIMIZADA")
    print("=" * 55)
    print("🎯 Capital: $35 | Entradas: $4 | ROI Esperado: +11.525%")
    print()
    
    # ========================================
    # 🧬 CONFIGURAÇÃO GENÉTICA VENCEDORA
    # ========================================
    
    filtros_finais = {
        
        # 💰 GESTÃO DE CAPITAL E RISCO
        "CAPITAL_TOTAL": 35.0,
        "POSITION_SIZE": 4.0,                    # Entrada fixa de $4
        "MAX_LEVERAGE": 10,                      # Leverage 10x (ótimo para $4)
        "MAX_POSITIONS": 8,                      # Máximo 8 posições simultâneas
        "CAPITAL_RESERVED": 3.0,                 # $3 de margem de segurança
        
        # 🎯 STOP LOSS E TAKE PROFIT (DNA GENÉTICO)
        "STOP_LOSS_PCT": 0.015,                 # 1.5% ROI (DNA vencedor)
        "TAKE_PROFIT_PCT": 0.12,                # 12% ROI (DNA vencedor)
        "MAX_RISK_PER_TRADE": 0.60,             # $0.60 risco máximo
        "RISK_PCT_CAPITAL": 1.71,               # 1.71% do capital por trade
        
        # 📊 INDICADORES TÉCNICOS (DNA OTIMIZADO)
        "EMA_FAST": 3,                          # EMA rápida (DNA vencedor)
        "EMA_SLOW": 34,                         # EMA lenta (DNA vencedor)  
        "RSI_PERIOD": 21,                       # RSI período (DNA calibrado)
        "ATR_PERIOD": 14,                       # ATR padrão
        "VOLUME_MA_PERIOD": 20,                 # Volume média móvel
        
        # 🚀 CRITÉRIOS DE ENTRADA (MAIS PERMISSIVO)
        "CONFLUENCE_THRESHOLD": 0.25,           # 25% confluence (era 55%)
        "VOLUME_MULTIPLIER": 1.2,               # 1.2x volume médio (era 1.8x)
        "ATR_BREAKOUT_THRESHOLD": 0.7,          # 0.7 ATR breakout (era 1.0)
        "MIN_ATR_PCT": 0.45,                    # ATR mínimo 0.45%
        "MAX_ATR_PCT": 8.0,                     # ATR máximo 8%
        
        # 🎯 FILTROS DE QUALIDADE
        "RSI_OVERSOLD": 30,                     # RSI oversold
        "RSI_OVERBOUGHT": 70,                   # RSI overbought
        "RSI_NEUTRAL_MIN": 40,                  # RSI neutro mínimo
        "RSI_NEUTRAL_MAX": 60,                  # RSI neutro máximo
        
        # 📈 TREND E MOMENTUM
        "EMA_TREND_CONFIRMATION": True,         # EMA 3 > EMA 34 para long
        "PRICE_ABOVE_EMA": True,                # Preço acima de EMA para confirmação
        "VOLUME_CONFIRMATION": True,            # Volume acima da média
        "ATR_VOLATILITY_CHECK": True,           # Volatilidade dentro do range
        
        # ⏰ TIMING E COOLDOWN
        "COOLDOWN_MINUTES": 15,                 # 15min entre trades no mesmo ativo
        "ANTI_SPAM_SECONDS": 3,                 # 3s anti-spam
        "MIN_HOLD_BARS": 1,                     # Mínimo 1 barra de holding
        "ENABLE_COOLDOWN": True,                # Cooldown ativo
        
        # 💸 TAXAS E CUSTOS (HYPERLIQUID)
        "MAKER_FEE": 0.0002,                    # 0.02% maker fee
        "TAKER_FEE": 0.0005,                    # 0.05% taker fee  
        "FUNDING_FEE": 0.0001,                  # 0.01% funding fee
        "TOTAL_FEE_RATE": 0.0008,               # 0.08% total por trade
        "BREAK_EVEN_PCT": 0.08,                 # 0.08% break-even mínimo
        
        # 🎨 ASSETS PERMITIDOS (TOP PERFORMERS)
        "ALLOWED_ASSETS": [
            "BTC/USDC:USDC",   # BTC - base sólida
            "SOL/USDC:USDC",   # SOL - alta volatilidade
            "ETH/USDC:USDC",   # ETH - mercado grande
            "XRP/USDC:USDC",   # XRP - melhor eficiência (162.5% ROI/trade)
            "DOGE/USDC:USDC",  # DOGE - alta frequência
            "AVAX/USDC:USDC",  # AVAX - bom volume
        ],
        
        # 🧠 MACHINE LEARNING E PADRÕES
        "ENABLE_PATTERN_DETECTION": True,       # Detecção de padrões ativa
        "MIN_PATTERN_CONFIDENCE": 0.6,          # 60% confiança mínima
        "ADAPTIVE_THRESHOLDS": True,            # Thresholds adaptativos
        "MARKET_REGIME_DETECTION": True,        # Detecção de regime de mercado
        
        # 🛡️ PROTEÇÕES E SAFETY
        "MAX_DRAWDOWN_PCT": 10.0,               # 10% drawdown máximo
        "DAILY_LOSS_LIMIT": 2.0,                # $2 perda máxima diária
        "MAX_CONSECUTIVE_LOSSES": 5,            # 5 perdas consecutivas max
        "POSITION_SIZE_SCALING": False,         # Tamanho fixo $4
        
        # 📊 MONITORAMENTO E LOGS
        "ENABLE_DETAILED_LOGS": True,           # Logs detalhados
        "SAVE_TRADE_HISTORY": True,             # Salvar histórico
        "PERFORMANCE_TRACKING": True,           # Tracking de performance
        "REAL_TIME_METRICS": True,              # Métricas em tempo real
    }
    
    return filtros_finais

def print_filtros_organizados():
    """Imprime os filtros organizados por categoria"""
    
    filtros = get_estrategia_final_filtros()
    
    print("💰 GESTÃO DE CAPITAL:")
    print("=" * 25)
    print(f"   💵 Capital Total: ${filtros['CAPITAL_TOTAL']}")
    print(f"   📊 Entrada por Trade: ${filtros['POSITION_SIZE']}")
    print(f"   ⚡ Leverage Máximo: {filtros['MAX_LEVERAGE']}x")
    print(f"   🎯 Posições Simultâneas: {filtros['MAX_POSITIONS']}")
    print(f"   🛡️ Margem de Segurança: ${filtros['CAPITAL_RESERVED']}")
    
    print(f"\n🎯 RISCO E RETORNO:")
    print("=" * 22)
    print(f"   🔴 Stop Loss: {filtros['STOP_LOSS_PCT']*100:.1f}%")
    print(f"   🟢 Take Profit: {filtros['TAKE_PROFIT_PCT']*100:.1f}%")
    print(f"   💰 Risco por Trade: ${filtros['MAX_RISK_PER_TRADE']}")
    print(f"   📊 Risco % Capital: {filtros['RISK_PCT_CAPITAL']:.2f}%")
    
    print(f"\n📈 INDICADORES TÉCNICOS:")
    print("=" * 27)
    print(f"   📊 EMA Rápida: {filtros['EMA_FAST']}")
    print(f"   📊 EMA Lenta: {filtros['EMA_SLOW']}")
    print(f"   📊 RSI Período: {filtros['RSI_PERIOD']}")
    print(f"   📊 ATR Período: {filtros['ATR_PERIOD']}")
    print(f"   📊 Volume MA: {filtros['VOLUME_MA_PERIOD']}")
    
    print(f"\n🚀 CRITÉRIOS DE ENTRADA:")
    print("=" * 27)
    print(f"   🎯 Confluence: {filtros['CONFLUENCE_THRESHOLD']*100:.0f}%")
    print(f"   📊 Volume: {filtros['VOLUME_MULTIPLIER']:.1f}x média")
    print(f"   📈 ATR Breakout: {filtros['ATR_BREAKOUT_THRESHOLD']}")
    print(f"   📊 ATR Min: {filtros['MIN_ATR_PCT']:.2f}%")
    print(f"   📊 ATR Max: {filtros['MAX_ATR_PCT']:.1f}%")
    
    print(f"\n💸 CUSTOS E TAXAS:")
    print("=" * 20)
    print(f"   💰 Taxa Maker: {filtros['MAKER_FEE']*100:.3f}%")
    print(f"   💰 Taxa Taker: {filtros['TAKER_FEE']*100:.3f}%")
    print(f"   💰 Taxa Funding: {filtros['FUNDING_FEE']*100:.3f}%")
    print(f"   💰 Taxa Total: {filtros['TOTAL_FEE_RATE']*100:.3f}%")
    print(f"   ⚖️ Break-Even: {filtros['BREAK_EVEN_PCT']:.3f}%")
    
    print(f"\n🎨 ASSETS PERMITIDOS:")
    print("=" * 22)
    for i, asset in enumerate(filtros['ALLOWED_ASSETS'], 1):
        asset_name = asset.split('/')[0]
        print(f"   {i}. {asset_name}")
    
    print(f"\n⏰ TIMING E CONTROLE:")
    print("=" * 22)
    print(f"   🕐 Cooldown: {filtros['COOLDOWN_MINUTES']}min")
    print(f"   ⚡ Anti-spam: {filtros['ANTI_SPAM_SECONDS']}s")
    print(f"   📊 Hold Mínimo: {filtros['MIN_HOLD_BARS']} barra")
    
    print(f"\n🛡️ PROTEÇÕES:")
    print("=" * 15)
    print(f"   📉 Drawdown Máximo: {filtros['MAX_DRAWDOWN_PCT']:.1f}%")
    print(f"   💰 Perda Diária Max: ${filtros['DAILY_LOSS_LIMIT']}")
    print(f"   🔴 Perdas Consecutivas: {filtros['MAX_CONSECUTIVE_LOSSES']}")

def get_trading_py_config():
    """Retorna configuração específica para trading.py"""
    
    filtros = get_estrategia_final_filtros()
    
    config = f"""
# ========================================
# 🧬 CONFIGURAÇÃO OTIMIZADA PARA $35 CAPITAL
# ========================================
# 🎯 ROI Esperado: +11.525% anual
# 💰 Entradas: $4 | Leverage: 10x | 8 posições max

# GESTÃO DE CAPITAL
POSITION_SIZE = {filtros['POSITION_SIZE']}
MAX_LEVERAGE = {filtros['MAX_LEVERAGE']}
MAX_POSITIONS = {filtros['MAX_POSITIONS']}

# RISCO E RETORNO
STOP_LOSS_CAPITAL_PCT = {filtros['STOP_LOSS_PCT']}
TAKE_PROFIT_CAPITAL_PCT = {filtros['TAKE_PROFIT_PCT']}

# INDICADORES (DNA GENÉTICO)
EMA_SHORT_SPAN = {filtros['EMA_FAST']}
EMA_LONG_SPAN = {filtros['EMA_SLOW']}
RSI_PERIOD = {filtros['RSI_PERIOD']}
ATR_PERIOD = {filtros['ATR_PERIOD']}

# CRITÉRIOS DE ENTRADA (MAIS PERMISSIVO)
MIN_CONFLUENCIA = int({filtros['CONFLUENCE_THRESHOLD']} * 100)  # 25%
VOLUME_MULTIPLIER = {filtros['VOLUME_MULTIPLIER']}
BREAKOUT_K_ATR = {filtros['ATR_BREAKOUT_THRESHOLD']}
ATR_PCT_MIN = {filtros['MIN_ATR_PCT']}
ATR_PCT_MAX = {filtros['MAX_ATR_PCT']}

# TIMING
COOLDOWN_MINUTOS = {filtros['COOLDOWN_MINUTES']}
ANTI_SPAM_SECS = {filtros['ANTI_SPAM_SECONDS']}
"""
    
    return config

def compare_with_current():
    """Compara com configuração atual do trading.py"""
    
    print(f"\n🔍 COMPARAÇÃO COM CONFIGURAÇÃO ATUAL:")
    print("=" * 45)
    
    print("📊 ATUAL (trading.py):")
    print("   💰 Capital: $64")
    print("   ⚡ Leverage: 3x")
    print("   🎯 Confluence: 55%")
    print("   📊 Volume: 1.3x")
    print("   📈 ROI: +9.480%")
    
    print(f"\n🎯 OTIMIZADA ($35):")
    print("   💰 Capital: $35")
    print("   ⚡ Leverage: 10x")
    print("   🎯 Confluence: 25%")
    print("   📊 Volume: 1.2x")
    print("   📈 ROI: +11.525%")
    
    print(f"\n🏆 MELHORIAS:")
    print("   ✅ ROI 21.6% SUPERIOR")
    print("   ✅ Capital 45% MENOR necessário")
    print("   ✅ Mais oportunidades (confluence menor)")
    print("   ✅ Melhor diversificação (8 posições)")
    print("   ✅ Risco controlado (1.71% por trade)")

def main():
    """Executa análise completa dos filtros"""
    
    print("🔬 DEFININDO FILTROS FINAIS DA ESTRATÉGIA...")
    print()
    
    print_filtros_organizados()
    
    print(get_trading_py_config())
    
    compare_with_current()
    
    print(f"\n🎊 RESUMO FINAL:")
    print("=" * 20)
    print("✅ Estratégia otimizada para $35 capital")
    print("✅ Entradas fixas de $4 por trade") 
    print("✅ Leverage 10x para máximo ROI")
    print("✅ 8 posições simultâneas possíveis")
    print("✅ Filtros mais permissivos para mais oportunidades")
    print("✅ ROI esperado: +11.525% anual")
    print("✅ Superior à estratégia original em +21.6%")

if __name__ == "__main__":
    main()
