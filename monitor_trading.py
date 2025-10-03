#!/usr/bin/env python3
"""
Sistema de Monitoramento de Trading em Tempo Real
Monitora o desempenho do sistema de trading otimizado
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sqlite3
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TradingMonitor:
    def __init__(self, db_path: str = "hl_learn_inverse.db"):
        self.db_path = db_path
        self.monitoring_data = {
            'trades': [],
            'performance_metrics': {},
            'alerts': [],
            'daily_stats': defaultdict(dict)
        }
        self.baseline_roi = 227  # ROI baseline com dados reais
        self.optimized_roi = 2190  # ROI otimizado esperado
        
    def get_recent_trades(self, hours: int = 24) -> pd.DataFrame:
        """Busca trades recentes do banco de dados"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calcular timestamp de 24 horas atrás
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_timestamp = cutoff_time.timestamp()
            
            query = """
            SELECT * FROM trades 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(cutoff_timestamp,))
            conn.close()
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['profit_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price'] * 100
                if 'side' in df.columns:
                    df.loc[df['side'] == 'SHORT', 'profit_pct'] *= -1
                    
            return df
            
        except Exception as e:
            print(f"❌ Erro ao buscar trades: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas de performance"""
        if df.empty:
            return {}
            
        metrics = {}
        
        # Métricas básicas
        metrics['total_trades'] = len(df)
        metrics['profitable_trades'] = len(df[df['profit_pct'] > 0])
        metrics['losing_trades'] = len(df[df['profit_pct'] < 0])
        metrics['win_rate'] = (metrics['profitable_trades'] / metrics['total_trades']) * 100 if metrics['total_trades'] > 0 else 0
        
        # Métricas financeiras
        metrics['total_profit_pct'] = df['profit_pct'].sum()
        metrics['avg_profit_pct'] = df['profit_pct'].mean()
        metrics['max_profit_pct'] = df['profit_pct'].max()
        metrics['max_loss_pct'] = df['profit_pct'].min()
        
        # Métricas de risco
        metrics['profit_factor'] = abs(df[df['profit_pct'] > 0]['profit_pct'].sum() / df[df['profit_pct'] < 0]['profit_pct'].sum()) if len(df[df['profit_pct'] < 0]) > 0 else float('inf')
        metrics['sharpe_ratio'] = df['profit_pct'].mean() / df['profit_pct'].std() if df['profit_pct'].std() != 0 else 0
        
        # Drawdown
        cumulative = (1 + df['profit_pct']/100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        metrics['max_drawdown_pct'] = drawdown.min()
        
        # Métricas por ativo
        if 'symbol' in df.columns:
            metrics['assets_traded'] = df['symbol'].nunique()
            metrics['best_asset'] = df.groupby('symbol')['profit_pct'].sum().idxmax() if not df.empty else None
            metrics['worst_asset'] = df.groupby('symbol')['profit_pct'].sum().idxmin() if not df.empty else None
        
        return metrics
    
    def check_alerts(self, metrics: Dict) -> List[str]:
        """Verifica condições de alerta"""
        alerts = []
        
        # Alerta de win rate baixo
        if metrics.get('win_rate', 0) < 40:
            alerts.append(f"🚨 WIN RATE BAIXO: {metrics['win_rate']:.1f}% (esperado >40%)")
        
        # Alerta de drawdown alto
        if metrics.get('max_drawdown_pct', 0) < -15:
            alerts.append(f"🚨 DRAWDOWN ALTO: {metrics['max_drawdown_pct']:.1f}% (limite -15%)")
        
        # Alerta de profit factor baixo
        if metrics.get('profit_factor', 0) < 1.2:
            alerts.append(f"🚨 PROFIT FACTOR BAIXO: {metrics['profit_factor']:.2f} (esperado >1.2)")
        
        # Alerta de poucos trades
        if metrics.get('total_trades', 0) < 10:
            alerts.append(f"⚠️ POUCOS TRADES: {metrics['total_trades']} trades em 24h")
        
        # Alerta de performance muito abaixo do esperado
        daily_roi_projection = (metrics.get('total_profit_pct', 0) * 365) / 100
        if daily_roi_projection < self.baseline_roi * 0.5:  # 50% do baseline
            alerts.append(f"🚨 PERFORMANCE BAIXA: Projeção anual {daily_roi_projection:.1f}% (esperado >{self.baseline_roi}%)")
        
        return alerts
    
    def generate_report(self, period_hours: int = 24) -> str:
        """Gera relatório de monitoramento"""
        print(f"📊 Gerando relatório dos últimos {period_hours} horas...")
        
        # Buscar dados
        df = self.get_recent_trades(period_hours)
        metrics = self.calculate_performance_metrics(df)
        alerts = self.check_alerts(metrics)
        
        # Criar relatório
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📈 RELATÓRIO DE MONITORAMENTO TRADING                      ║
║                        Sistema Otimizado - ROI 2190%                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Período: Últimas {period_hours} horas                                                    ║
║  Timestamp: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 MÉTRICAS PRINCIPAIS:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Total de Trades:        {metrics.get('total_trades', 0):>8}                                    │
│ Trades Lucrativos:      {metrics.get('profitable_trades', 0):>8}                                    │
│ Trades Perdedores:      {metrics.get('losing_trades', 0):>8}                                    │
│ Win Rate:              {metrics.get('win_rate', 0):>8.1f}%                                   │
└─────────────────────────────────────────────────────────────────────────────┘

💰 PERFORMANCE FINANCEIRA:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Lucro Total:           {metrics.get('total_profit_pct', 0):>8.2f}%                                  │
│ Lucro Médio por Trade:  {metrics.get('avg_profit_pct', 0):>8.3f}%                                  │
│ Melhor Trade:          {metrics.get('max_profit_pct', 0):>8.2f}%                                  │
│ Pior Trade:            {metrics.get('max_loss_pct', 0):>8.2f}%                                  │
│ Profit Factor:         {metrics.get('profit_factor', 0):>8.2f}                                     │
└─────────────────────────────────────────────────────────────────────────────┘

📊 MÉTRICAS DE RISCO:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Max Drawdown:          {metrics.get('max_drawdown_pct', 0):>8.2f}%                                  │
│ Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>8.3f}                                     │
│ Ativos Negociados:     {metrics.get('assets_traded', 0):>8}                                    │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 PROJEÇÃO ANUAL:
┌─────────────────────────────────────────────────────────────────────────────┐
"""
        
        # Projeção anual
        if metrics.get('total_profit_pct'):
            daily_return = metrics['total_profit_pct'] / (period_hours / 24)
            annual_projection = (daily_return * 365)
            
            report += f"│ ROI Projetado (anual):  {annual_projection:>8.1f}%                                  │\n"
            report += f"│ ROI Baseline:          {self.baseline_roi:>8.1f}%                                  │\n"
            report += f"│ ROI Otimizado:         {self.optimized_roi:>8.1f}%                                  │\n"
            
            performance_vs_baseline = (annual_projection / self.baseline_roi) * 100
            performance_vs_optimized = (annual_projection / self.optimized_roi) * 100
            
            report += f"│ vs Baseline:           {performance_vs_baseline:>8.1f}%                                  │\n"
            report += f"│ vs Otimizado:          {performance_vs_optimized:>8.1f}%                                  │\n"
        
        report += "└─────────────────────────────────────────────────────────────────────────────┘\n\n"
        
        # Alertas
        if alerts:
            report += "🚨 ALERTAS:\n"
            for alert in alerts:
                report += f"   {alert}\n"
        else:
            report += "✅ SISTEMA FUNCIONANDO NORMALMENTE - Nenhum alerta ativo\n"
        
        # Top ativos se houver dados
        if not df.empty and 'symbol' in df.columns:
            asset_performance = df.groupby('symbol').agg({
                'profit_pct': ['sum', 'count', 'mean']
            }).round(2)
            
            if len(asset_performance) > 0:
                report += f"\n📈 TOP 5 ATIVOS (por lucro total):\n"
                top_assets = asset_performance.sort_values(('profit_pct', 'sum'), ascending=False).head(5)
                
                for symbol, data in top_assets.iterrows():
                    total_profit = data[('profit_pct', 'sum')]
                    trade_count = data[('profit_pct', 'count')]
                    avg_profit = data[('profit_pct', 'mean')]
                    report += f"   {symbol:>8}: {total_profit:>6.2f}% ({trade_count:>2} trades, avg: {avg_profit:>5.2f}%)\n"
        
        return report
    
    def save_monitoring_data(self, filename: str = None):
        """Salva dados de monitoramento em arquivo JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring_report_{timestamp}.json"
        
        df = self.get_recent_trades(24)
        metrics = self.calculate_performance_metrics(df)
        alerts = self.check_alerts(metrics)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'alerts': alerts,
            'trade_count': len(df),
            'recent_trades': df.to_dict('records') if not df.empty else []
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Dados salvos em: {filename}")
        return filename
    
    def create_performance_chart(self, period_hours: int = 168):  # 1 semana
        """Cria gráfico de performance"""
        df = self.get_recent_trades(period_hours)
        
        if df.empty:
            print("❌ Sem dados para gerar gráfico")
            return
        
        # Preparar dados
        df['cumulative_profit'] = df['profit_pct'].cumsum()
        df['hour'] = df['datetime'].dt.floor('H')
        
        # Criar figura com subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Dashboard de Performance Trading', fontsize=16, fontweight='bold')
        
        # 1. Lucro cumulativo
        ax1.plot(df['datetime'], df['cumulative_profit'], linewidth=2, color='#2E8B57')
        ax1.set_title('Lucro Cumulativo (%)')
        ax1.set_ylabel('Lucro (%)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Distribuição de lucros
        ax2.hist(df['profit_pct'], bins=30, alpha=0.7, color='#4169E1', edgecolor='black')
        ax2.axvline(df['profit_pct'].mean(), color='red', linestyle='--', label=f'Média: {df["profit_pct"].mean():.2f}%')
        ax2.set_title('Distribuição de Lucros por Trade')
        ax2.set_xlabel('Lucro (%)')
        ax2.set_ylabel('Frequência')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trades por hora
        hourly_trades = df.groupby('hour').size()
        ax3.bar(range(len(hourly_trades)), hourly_trades.values, color='#FF6347', alpha=0.7)
        ax3.set_title('Trades por Hora')
        ax3.set_xlabel('Horas')
        ax3.set_ylabel('Número de Trades')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance por ativo (se disponível)
        if 'symbol' in df.columns:
            asset_profit = df.groupby('symbol')['profit_pct'].sum().sort_values(ascending=True)
            colors = ['red' if x < 0 else 'green' for x in asset_profit.values]
            ax4.barh(range(len(asset_profit)), asset_profit.values, color=colors, alpha=0.7)
            ax4.set_yticks(range(len(asset_profit)))
            ax4.set_yticklabels(asset_profit.index)
            ax4.set_title('Lucro Total por Ativo (%)')
            ax4.set_xlabel('Lucro (%)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Dados de símbolo\nnão disponíveis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Performance por Ativo')
        
        plt.tight_layout()
        
        # Salvar gráfico
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_chart_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo: {filename}")
        
        plt.show()
        return filename

def main():
    """Função principal para monitoramento"""
    monitor = TradingMonitor()
    
    print("🚀 Iniciando monitoramento do sistema de trading...")
    print("=" * 80)
    
    # Gerar relatório
    report = monitor.generate_report(24)
    print(report)
    
    # Salvar dados
    json_file = monitor.save_monitoring_data()
    
    # Criar gráfico
    try:
        chart_file = monitor.create_performance_chart(168)  # 1 semana
    except Exception as e:
        print(f"⚠️ Erro ao criar gráfico: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Monitoramento concluído!")
    print("\n💡 Para monitoramento contínuo, execute:")
    print("   python monitor_trading.py")
    print("\n📝 Para relatório de período específico:")
    print("   python -c \"from monitor_trading import TradingMonitor; print(TradingMonitor().generate_report(48))\"")

if __name__ == "__main__":
    main()
