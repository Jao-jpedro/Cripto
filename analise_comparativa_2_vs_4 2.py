#!/usr/bin/env python3
"""
📊 ANÁLISE COMPARATIVA: ESTRATÉGIAS 2 vs 4
==========================================
Comparação visual das estratégias testadas separadamente
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def gerar_comparacao_visual():
    """Gerar gráficos comparativos das estratégias"""
    
    print("📊 ANÁLISE COMPARATIVA DETALHADA")
    print("="*50)
    
    # Dados dos resultados
    resultados = {
        'Estratégia': ['Sem Proteções', 'Proteção 2 (Crashes)', 'Proteção 4 (Temporal)', 'Ambas (2+4)'],
        'ROI (%)': [5551, 5449, 1692, 1589],
        'Max Drawdown (%)': [-92.64, -78.14, -86.02, -98.71],
        'Total Trades': [52312, 51340, 21391, 20813],
        'Win Rate (%)': [15.5, 15.5, 13.7, 13.5],
        'Trades/Dia': [70.8, 69.4, 28.9, 28.1]
    }
    
    df = pd.DataFrame(resultados)
    
    # Configuração visual
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📊 COMPARAÇÃO ESTRATÉGIAS DE PROTEÇÃO\nTeste Individual: Estratégia 2 vs 4', 
                 fontsize=16, fontweight='bold', color='white')
    
    # 1. ROI Comparison
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    bars1 = axes[0,0].bar(df['Estratégia'], df['ROI (%)'], color=colors, alpha=0.8)
    axes[0,0].set_title('💰 ROI Total (%)', fontweight='bold', color='white')
    axes[0,0].set_ylabel('ROI (%)', color='white')
    axes[0,0].tick_params(axis='x', rotation=45, colors='white')
    axes[0,0].tick_params(axis='y', colors='white')
    axes[0,0].grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars1, df['ROI (%)']):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 100,
                      f'+{valor:,}%', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 2. Max Drawdown
    bars2 = axes[0,1].bar(df['Estratégia'], df['Max Drawdown (%)'], color=colors, alpha=0.8)
    axes[0,1].set_title('📉 Max Drawdown (%)', fontweight='bold', color='white')
    axes[0,1].set_ylabel('Drawdown (%)', color='white')
    axes[0,1].tick_params(axis='x', rotation=45, colors='white')
    axes[0,1].tick_params(axis='y', colors='white')
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, valor in zip(bars2, df['Max Drawdown (%)']):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height - 2,
                      f'{valor}%', ha='center', va='top', color='white', fontweight='bold')
    
    # 3. Total Trades
    bars3 = axes[0,2].bar(df['Estratégia'], df['Total Trades'], color=colors, alpha=0.8)
    axes[0,2].set_title('📊 Total de Trades', fontweight='bold', color='white')
    axes[0,2].set_ylabel('Trades', color='white')
    axes[0,2].tick_params(axis='x', rotation=45, colors='white')
    axes[0,2].tick_params(axis='y', colors='white')
    axes[0,2].grid(True, alpha=0.3)
    
    for bar, valor in zip(bars3, df['Total Trades']):
        height = bar.get_height()
        axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 500,
                      f'{valor:,}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 4. Win Rate
    bars4 = axes[1,0].bar(df['Estratégia'], df['Win Rate (%)'], color=colors, alpha=0.8)
    axes[1,0].set_title('🎯 Taxa de Acerto (%)', fontweight='bold', color='white')
    axes[1,0].set_ylabel('Win Rate (%)', color='white')
    axes[1,0].tick_params(axis='x', rotation=45, colors='white')
    axes[1,0].tick_params(axis='y', colors='white')
    axes[1,0].grid(True, alpha=0.3)
    
    for bar, valor in zip(bars4, df['Win Rate (%)']):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                      f'{valor}%', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 5. Trades por Dia
    bars5 = axes[1,1].bar(df['Estratégia'], df['Trades/Dia'], color=colors, alpha=0.8)
    axes[1,1].set_title('⚡ Frequência (Trades/Dia)', fontweight='bold', color='white')
    axes[1,1].set_ylabel('Trades/Dia', color='white')
    axes[1,1].tick_params(axis='x', rotation=45, colors='white')
    axes[1,1].tick_params(axis='y', colors='white')
    axes[1,1].grid(True, alpha=0.3)
    
    for bar, valor in zip(bars5, df['Trades/Dia']):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{valor}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 6. Risk-Return Scatter
    axes[1,2].scatter(df['Max Drawdown (%)'], df['ROI (%)'], 
                     c=colors, s=200, alpha=0.8, edgecolors='white', linewidth=2)
    axes[1,2].set_title('🎯 Risco vs Retorno', fontweight='bold', color='white')
    axes[1,2].set_xlabel('Max Drawdown (%)', color='white')
    axes[1,2].set_ylabel('ROI (%)', color='white')
    axes[1,2].tick_params(colors='white')
    axes[1,2].grid(True, alpha=0.3)
    
    # Adicionar labels no scatter
    for i, estrategia in enumerate(df['Estratégia']):
        axes[1,2].annotate(estrategia, 
                          (df['Max Drawdown (%)'].iloc[i], df['ROI (%)'].iloc[i]),
                          xytext=(5, 5), textcoords='offset points', 
                          color='white', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"comparacao_estrategias_2_vs_4_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"📊 Gráfico salvo: {filename}")
    
    # Tabela resumo
    print("\n" + "="*80)
    print("📋 TABELA COMPARATIVA COMPLETA")
    print("="*80)
    
    print(f"{'Estratégia':<20} {'ROI (%)':<12} {'Drawdown (%)':<14} {'Trades':<10} {'Win Rate':<10} {'Freq/Dia':<10}")
    print("-" * 80)
    
    for i, row in df.iterrows():
        print(f"{row['Estratégia']:<20} {row['ROI (%)']:>+8,.0f}%   {row['Max Drawdown (%)']:>+8.1f}%     {row['Total Trades']:>8,}  {row['Win Rate (%)']:>7.1f}%   {row['Trades/Dia']:>7.1f}")
    
    # Análise detalhada
    print("\n" + "="*80)
    print("🔍 ANÁLISE DETALHADA")
    print("="*80)
    
    print("🏆 RESULTADO INDIVIDUAL ESTRATÉGIAS:")
    print("   🔥 Estratégia 2 (Crashes): +5.449% ROI | -78.14% drawdown")
    print("   ⏸️ Estratégia 4 (Temporal): +1.692% ROI | -86.02% drawdown") 
    print("   💡 Estratégia 2 é 3.2x mais lucrativa que a 4")
    
    print("\n📊 IMPACTO DAS PROTEÇÕES:")
    print("   🚫 Sem proteções: 52.312 trades | ROI +5.551%")
    print("   🛡️ Proteção 2: 51.340 trades (-971) | ROI +5.449% (-102%)")
    print("   ⏸️ Proteção 4: 21.391 trades (-30.921) | ROI +1.692% (-3.859%)")
    print("   🤝 Ambas juntas: 20.813 trades (-31.499) | ROI +1.589% (-3.962%)")
    
    print("\n🎯 CONCLUSÕES:")
    print("   1️⃣ Estratégia 2 preserva quase toda lucratividade (-2% ROI)")
    print("   2️⃣ Estratégia 4 reduz muito a lucratividade (-69% ROI)")
    print("   3️⃣ Estratégia 2 melhora significativamente o drawdown (-78% vs -93%)")
    print("   4️⃣ Estratégia 4 bloqueia 58% dos trades (proteção excessiva)")
    print("   5️⃣ Para uso real: Estratégia 2 é muito superior!")
    
    print("\n💡 RECOMENDAÇÃO FINAL:")
    print("   🎯 USE APENAS A ESTRATÉGIA 2!")
    print("   ✅ Melhor equilíbrio risco-retorno")
    print("   ✅ Preserva 98% da lucratividade")
    print("   ✅ Reduz drawdown de -93% para -78%")
    print("   ✅ Mantém alta frequência de trades")

if __name__ == "__main__":
    gerar_comparacao_visual()
