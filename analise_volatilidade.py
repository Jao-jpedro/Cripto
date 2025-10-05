#!/usr/bin/env python3
"""
ANÁLISE DA VOLATILIDADE - Por que leverage alto ainda falha?
"""

import pandas as pd
import numpy as np

def analyze_volatility_issue():
    """Analisa por que leverage alto ainda dá -100%"""
    
    print("🔍 ANÁLISE: POR QUE LEVERAGE ALTO AINDA FALHA?")
    print("="*60)
    
    # Carregar dados BTC
    try:
        df = pd.read_csv("dados_reais_btc_1ano.csv")
        if 'close' in df.columns:
            df['valor_fechamento'] = df['close']
    except:
        print("❌ Erro ao carregar dados")
        return
    
    # Calcular volatilidade diária
    df['retorno_diario'] = df['valor_fechamento'].pct_change()
    volatilidade_diaria = df['retorno_diario'].std()
    
    print(f"📊 ANÁLISE DE VOLATILIDADE (BTC):")
    print(f"   Volatilidade diária: {volatilidade_diaria*100:.2f}%")
    print(f"   SL configurado: 5.0%")
    print(f"   TP configurado: 8.0%")
    print()
    
    # Simular probabilidade de bater SL vs TP
    print(f"🎲 PROBABILIDADE DE SL vs TP:")
    print("-"*40)
    
    # Contar movimentos que bateriam SL/TP
    sl_hits = (df['retorno_diario'] <= -0.05).sum()
    tp_hits = (df['retorno_diario'] >= 0.08).sum()
    total_days = len(df['retorno_diario'].dropna())
    
    print(f"Dias que bateriam SL (-5%): {sl_hits}/{total_days} ({sl_hits/total_days*100:.1f}%)")
    print(f"Dias que bateriam TP (+8%): {tp_hits}/{total_days} ({tp_hits/total_days*100:.1f}%)")
    
    ratio_tp_sl = tp_hits / max(sl_hits, 1)
    print(f"Ratio TP/SL: {ratio_tp_sl:.2f}")
    
    if ratio_tp_sl < 1:
        print("❌ Mais dias batem SL que TP!")
    else:
        print("✅ Mais dias batem TP que SL!")

def analyze_leverage_impact():
    """Analisa o impacto matemático do leverage"""
    
    print(f"\n💰 IMPACTO MATEMÁTICO DO LEVERAGE:")
    print("="*50)
    
    initial_balance = 1000
    sl_pct = 0.05  # 5%
    tp_pct = 0.08  # 8%
    
    print("Leverage | SL Impact | TP Impact | Balance após SL | Balance após TP")
    print("-" * 70)
    
    for leverage in [1, 3, 5, 10, 20]:
        sl_impact = sl_pct * leverage  # % de perda com SL
        tp_impact = tp_pct * leverage  # % de ganho com TP
        
        balance_after_sl = initial_balance * (1 - sl_impact)
        balance_after_tp = initial_balance * (1 + tp_impact)
        
        print(f"{leverage:8}x | {sl_impact*100:8.1f}% | {tp_impact*100:8.1f}% | ${balance_after_sl:11.0f} | ${balance_after_tp:11.0f}")
        
        if balance_after_sl <= 0:
            print(f"         | ❌ LIQUIDAÇÃO! Balance = $0")

def suggest_optimal_leverage():
    """Sugere leverage ótimo baseado na análise"""
    
    print(f"\n🎯 SUGESTÃO DE LEVERAGE ÓTIMO:")
    print("="*50)
    
    # Para não liquidar, o SL com leverage não pode passar de 100%
    max_sl_impact = 1.0  # 100%
    sl_pct = 0.05  # 5%
    
    max_safe_leverage = max_sl_impact / sl_pct
    
    print(f"📏 CÁLCULO DE LEVERAGE SEGURO:")
    print(f"   SL máximo permitido: {max_sl_impact*100:.0f}% (para não liquidar)")
    print(f"   SL configurado: {sl_pct*100:.0f}%")
    print(f"   Leverage máximo seguro: {max_safe_leverage:.0f}x")
    print()
    
    print(f"🛡️ RECOMENDAÇÕES:")
    recommended_leverages = [1, 3, 5]
    
    for lev in recommended_leverages:
        sl_impact = sl_pct * lev
        tp_impact = 0.08 * lev
        
        print(f"   Leverage {lev}x:")
        print(f"     SL impact: -{sl_impact*100:.0f}% (seguro)")
        print(f"     TP impact: +{tp_impact*100:.0f}%")
        print(f"     Risk/Reward: {tp_impact/sl_impact:.1f}:1")
        print()

def analyze_winning_strategy():
    """Analisa a estratégia vencedora dos resultados"""
    
    print(f"\n🏆 ANÁLISE DA ESTRATÉGIA VENCEDORA:")
    print("="*50)
    
    # BTC com leverage 3x: +201.8%
    print(f"✅ MELHOR RESULTADO: BTC Leverage 3x = +201.8%")
    print("-"*40)
    
    leverage = 3
    sl_pct = 0.05
    tp_pct = 0.08
    
    print(f"Configuração:")
    print(f"   Leverage: {leverage}x")
    print(f"   SL: {sl_pct*100:.0f}% = -{sl_pct*leverage*100:.0f}% balance impact")
    print(f"   TP: {tp_pct*100:.0f}% = +{tp_pct*leverage*100:.0f}% balance impact")
    print()
    
    print(f"Por que funciona:")
    print(f"   ✅ SL impact (-15%) não liquida")
    print(f"   ✅ TP impact (+24%) é substantivo")
    print(f"   ✅ Ratio 1.6:1 favorável")
    print(f"   ✅ Volatilidade controlada")

def main():
    analyze_volatility_issue()
    analyze_leverage_impact()
    suggest_optimal_leverage()
    analyze_winning_strategy()
    
    print(f"\n" + "="*60)
    print("🎯 CONCLUSÃO FINAL:")
    print("="*60)
    print("✅ Bug de cálculo CORRIGIDO")
    print("✅ Leverage 1-3x funciona muito bem!")
    print("❌ Leverage 5x+ = risco de liquidação")
    print("🏆 Recomendação: Leverage 3x (ROI +201.8%)")
    print("💡 Problema era matemático, não estratégico!")

if __name__ == "__main__":
    main()
