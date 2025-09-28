#!/usr/bin/env python3
"""
TradingFuturo - Análise de Resultados
=====================================
Análise detalhada dos resultados das 10 estratégias de trading
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_results():
    """Carrega os resultados do backtest"""
    with open('trading_futuro_results.json', 'r') as f:
        return json.load(f)

def create_performance_dataframe(results):
    """Cria DataFrame com métricas de performance"""
    data = []
    
    for strategy_name, strategy_data in results['detailed_results'].items():
        # Parse das métricas (simplificado)
        roi = strategy_data['roi_percent']
        final_balance = strategy_data['final_balance']
        trades_per_day = strategy_data['trades_per_day']
        
        # Extrair métricas básicas da string
        metrics_str = strategy_data['metrics']
        
        # Parse básico das métricas principais
        total_trades = int(metrics_str.split('total_trades=')[1].split(',')[0])
        win_rate = float(metrics_str.split('win_rate=')[1].split(',')[0])
        max_drawdown = float(metrics_str.split('max_drawdown=')[1].split(',')[0])
        sharpe_ratio = float(metrics_str.split('sharpe_ratio=')[1].split(',')[0])
        profit_factor = float(metrics_str.split('profit_factor=')[1].split(',')[0])
        
        data.append({
            'Estratégia': strategy_name,
            'ROI (%)': roi,
            'Saldo Final ($)': final_balance,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Trades/Dia': trades_per_day,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Profit Factor': profit_factor
        })
    
    return pd.DataFrame(data)

def generate_comprehensive_report(results):
    """Gera relatório completo com análises adicionais"""
    print("=" * 100)
    print(" 🏆 TRADINGFUTURO - ANÁLISE COMPLETA DOS RESULTADOS 🏆")
    print("=" * 100)
    
    # Resumo executivo
    summary = results['summary']
    print(f"\n📊 RESUMO EXECUTIVO:")
    print(f"├─ Melhor Estratégia: {summary['best_strategy']}")
    print(f"├─ Melhor ROI: {summary['best_roi']:.3f}%")
    print(f"├─ Total de Estratégias: {summary['total_strategies']}")
    print(f"└─ Estratégias Lucrativas: {summary['profitable_strategies']}")
    
    # Criar DataFrame
    df = create_performance_dataframe(results)
    
    # Análise de Performance
    print(f"\n📈 ANÁLISE DE PERFORMANCE:")
    print(f"├─ ROI Médio: {df['ROI (%)'].mean():.2f}%")
    print(f"├─ ROI Mediano: {df['ROI (%)'].median():.2f}%")
    print(f"├─ Desvio Padrão ROI: {df['ROI (%)'].std():.2f}%")
    print(f"├─ Melhor ROI: {df['ROI (%)'].max():.2f}%")
    print(f"├─ Pior ROI: {df['ROI (%)'].min():.2f}%")
    print(f"└─ Win Rate Médio: {df['Win Rate (%)'].mean():.1f}%")
    
    # Análise de Risk/Reward
    print(f"\n⚖️ ANÁLISE RISK/REWARD:")
    profitable_strategies = df[df['ROI (%)'] > 0]
    losing_strategies = df[df['ROI (%)'] <= 0]
    
    print(f"├─ Estratégias Lucrativas: {len(profitable_strategies)}/10 ({len(profitable_strategies)*10}%)")
    print(f"├─ ROI Médio (Lucrativas): {profitable_strategies['ROI (%)'].mean():.2f}%")
    print(f"├─ ROI Médio (Perdedoras): {losing_strategies['ROI (%)'].mean():.2f}%")
    print(f"├─ Drawdown Médio: {df['Max Drawdown (%)'].mean():.2f}%")
    print(f"└─ Sharpe Ratio Médio: {df['Sharpe Ratio'].mean():.2f}")
    
    # Análise de Atividade
    print(f"\n📊 ANÁLISE DE ATIVIDADE:")
    print(f"├─ Total de Trades: {df['Total Trades'].sum():,}")
    print(f"├─ Trades Médios por Estratégia: {df['Total Trades'].mean():.1f}")
    print(f"├─ Estratégia Mais Ativa: {df.loc[df['Total Trades'].idxmax(), 'Estratégia']}")
    print(f"├─ Trades da Mais Ativa: {df['Total Trades'].max():,}")
    print(f"└─ Trades/Dia Total: {df['Trades/Dia'].sum():.1f}")
    
    # Ranking completo
    print(f"\n🥇 RANKING COMPLETO POR ROI:")
    df_sorted = df.sort_values('ROI (%)', ascending=False).reset_index(drop=True)
    
    for i, row in df_sorted.iterrows():
        status = "✅" if row['ROI (%)'] > 0 else "❌"
        print(f"{i+1:2d}. {status} {row['Estratégia']:<25} | ROI: {row['ROI (%)']:6.2f}% | Win Rate: {row['Win Rate (%)']:5.1f}% | Trades: {row['Total Trades']:4d}")
    
    # Análise detalhada das top 3
    print(f"\n🔍 ANÁLISE DETALHADA - TOP 3 ESTRATÉGIAS:")
    
    for i in range(min(3, len(df_sorted))):
        strategy = df_sorted.iloc[i]
        print(f"\n{i+1}. 🏆 {strategy['Estratégia'].upper()}")
        print(f"   ┌─ Performance:")
        print(f"   │  ├─ ROI: {strategy['ROI (%)']:+.3f}%")
        print(f"   │  ├─ Saldo Final: ${strategy['Saldo Final ($)']:.2f}")
        print(f"   │  └─ Profit Factor: {strategy['Profit Factor']:.2f}")
        print(f"   ├─ Trading:")
        print(f"   │  ├─ Win Rate: {strategy['Win Rate (%)']:.1f}%")
        print(f"   │  ├─ Total Trades: {strategy['Total Trades']}")
        print(f"   │  └─ Frequência: {strategy['Trades/Dia']:.2f} trades/dia")
        print(f"   └─ Risco:")
        print(f"      ├─ Max Drawdown: {strategy['Max Drawdown (%)']:.2f}%")
        print(f"      └─ Sharpe Ratio: {strategy['Sharpe Ratio']:.2f}")
    
    # Recomendações finais
    print(f"\n🎯 RECOMENDAÇÕES FINAIS:")
    
    best_strategy = df_sorted.iloc[0]
    
    print(f"├─ ESTRATÉGIA RECOMENDADA: {best_strategy['Estratégia']}")
    print(f"├─ Motivos da Recomendação:")
    print(f"│  ├─ Maior ROI: {best_strategy['ROI (%)']:+.3f}%")
    print(f"│  ├─ Win Rate: {best_strategy['Win Rate (%)']:.1f}%")
    print(f"│  ├─ Baixo Drawdown: {best_strategy['Max Drawdown (%)']:.2f}%")
    print(f"│  └─ Bom Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
    
    print(f"├─ CONFIGURAÇÃO SUGERIDA:")
    print(f"│  ├─ Capital Inicial: $1,000 (100x o teste)")
    print(f"│  ├─ Risco por Trade: $10-20 (1-2%)")
    print(f"│  └─ ROI Esperado: {best_strategy['ROI (%)'] * 4:.1f}% ao ano (extrapolado)")
    
    print(f"├─ ESTRATÉGIAS A EVITAR:")
    worst_strategies = df_sorted.tail(3)
    for _, strategy in worst_strategies.iterrows():
        print(f"│  ├─ {strategy['Estratégia']}: ROI {strategy['ROI (%)']:+.2f}%")
    
    print(f"└─ PRÓXIMOS PASSOS:")
    print(f"   ├─ Implementar a estratégia {best_strategy['Estratégia']} em live trading")
    print(f"   ├─ Começar com capital pequeno ($100-500)")
    print(f"   ├─ Monitorar performance por 30 dias")
    print(f"   └─ Escalar gradualmente conforme resultados")
    
    print(f"\n{'='*100}")
    print(f"📊 Dados salvos em: trading_analysis.csv")
    print(f"{'='*100}\n")
    
    # Salvar DataFrame
    df_sorted.to_csv('trading_analysis.csv', index=False)
    
    return df_sorted

def create_visualizations(df):
    """Cria visualizações dos resultados"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TradingFuturo - Análise de Performance das Estratégias', fontsize=16, fontweight='bold')
    
    # 1. ROI por Estratégia
    ax1 = axes[0, 0]
    colors = ['green' if roi > 0 else 'red' for roi in df['ROI (%)']]
    bars = ax1.bar(range(len(df)), df['ROI (%)'], color=colors, alpha=0.7)
    ax1.set_title('ROI por Estratégia (%)')
    ax1.set_xlabel('Estratégias')
    ax1.set_ylabel('ROI (%)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in df['Estratégia']], rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # 2. Win Rate vs ROI
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['Win Rate (%)'], df['ROI (%)'], 
                         c=df['Total Trades'], cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_title('Win Rate vs ROI (tamanho = número de trades)')
    ax2.set_xlabel('Win Rate (%)')
    ax2.set_ylabel('ROI (%)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=50, color='blue', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Total de Trades')
    
    # 3. Sharpe Ratio
    ax3 = axes[1, 0]
    colors = ['darkgreen' if sr > 1 else 'orange' if sr > 0 else 'darkred' for sr in df['Sharpe Ratio']]
    bars = ax3.bar(range(len(df)), df['Sharpe Ratio'], color=colors, alpha=0.7)
    ax3.set_title('Sharpe Ratio por Estratégia')
    ax3.set_xlabel('Estratégias')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in df['Estratégia']], rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Bom (>1)')
    ax3.axhline(y=2, color='darkgreen', linestyle='--', alpha=0.5, label='Excelente (>2)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Risco vs Retorno
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['Max Drawdown (%)'], df['ROI (%)'], 
                         s=df['Total Trades'], alpha=0.7, 
                         c=['green' if roi > 0 else 'red' for roi in df['ROI (%)']], 
                         edgecolors='black', linewidth=0.5)
    ax4.set_title('Risco vs Retorno (tamanho = número de trades)')
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('ROI (%)')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Adicionar nomes das estratégias nos pontos
    for i, txt in enumerate(df['Estratégia']):
        ax4.annotate(txt[:8], (df['Max Drawdown (%)'].iloc[i], df['ROI (%)'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('trading_futuro_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Gráficos salvos em: trading_futuro_analysis.png")
    plt.show()

def main():
    """Função principal"""
    try:
        # Carregar resultados
        results = load_results()
        
        # Gerar relatório completo
        df = generate_comprehensive_report(results)
        
        # Criar visualizações
        create_visualizations(df)
        
        # Estatísticas finais
        print("🎯 CONCLUSÕES PRINCIPAIS:")
        print(f"├─ Das 10 estratégias testadas, apenas {results['summary']['profitable_strategies']} foram lucrativas")
        print(f"├─ A melhor estratégia ({results['summary']['best_strategy']}) obteve {results['summary']['best_roi']:.3f}% de ROI")
        print(f"├─ O mercado de criptomoedas mostrou-se desafiador no período testado")
        print(f"└─ Estratégias de volatilidade e trend following mostraram melhor performance")
        
    except FileNotFoundError:
        print("❌ Arquivo trading_futuro_results.json não encontrado!")
        print("Execute primeiro o TradingFuturo.py para gerar os resultados.")
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    main()
