#!/usr/bin/env python3
"""
📋 GUIA RÁPIDO PARA VISUALIZAR LOGS NO RENDER
============================================

Como acessar os logs do seu serviço srv-d31n3pumcj7s738sddi0
"""

import webbrowser
import time
from datetime import datetime

def abrir_dashboard_render():
    """
    Abre o dashboard do Render diretamente no seu serviço
    """
    print("🚀 GUIA PARA VISUALIZAR LOGS NO RENDER")
    print("=" * 50)
    
    service_url = "https://dashboard.render.com/worker/srv-d31n3pumcj7s738sddi0"
    
    print(f"🎯 Seu serviço: srv-d31n3pumcj7s738sddi0")
    print(f"📊 URL Dashboard: {service_url}")
    print()
    
    print("📋 PASSO A PASSO:")
    print("-" * 20)
    print("1. ✅ Abrir dashboard no navegador")
    print("2. 🔑 Fazer login com GitHub")
    print("3. 📊 Clicar na aba 'Logs'")
    print("4. 🔍 Procurar por indicadores de funcionamento")
    print()
    
    # Tentar abrir automaticamente
    try:
        print("🌐 Tentando abrir o dashboard automaticamente...")
        webbrowser.open(service_url)
        print("✅ Dashboard aberto no navegador!")
    except Exception as e:
        print(f"❌ Erro ao abrir: {e}")
        print(f"💡 Acesse manualmente: {service_url}")
    
    print()
    print("🔍 O QUE PROCURAR NOS LOGS:")
    print("-" * 30)
    
    indicators = [
        "🎯 DNA/Genetic: Parâmetros do algoritmo genético",
        "💰 Trading: Trades sendo executados (buy/sell)",
        "🔗 Hyperliquid: Conexões com a exchange",
        "📊 Performance: ROI, profits, losses",
        "⚠️ Erros: Falhas ou problemas de conexão",
        "🔄 Sistema: Status de funcionamento"
    ]
    
    for indicator in indicators:
        print(f"   {indicator}")
    
    print()
    print("🎯 PALAVRAS-CHAVE IMPORTANTES:")
    print("-" * 30)
    keywords = [
        "DNA", "genetic", "SL: 1.5%", "TP: 12%", 
        "leverage: 3x", "trade", "position", "ROI",
        "BTC", "ETH", "profit", "Hyperliquid"
    ]
    
    for keyword in keywords:
        print(f"   • {keyword}")
    
    print()
    print("💡 DICAS PARA MONITORAMENTO:")
    print("-" * 30)
    print("   📱 Use Ctrl+F para buscar palavras-chave")
    print("   🔄 Logs são atualizados em tempo real")
    print("   📊 Observe timestamps para ver atividade recente")
    print("   ⚠️ Procure por erros em vermelho")
    print("   ✅ Sucesso geralmente aparece em verde")
    
    return service_url

def criar_checklist_verificacao():
    """
    Cria um checklist para verificação manual
    """
    checklist = {
        "timestamp": datetime.now().isoformat(),
        "service_id": "srv-d31n3pumcj7s738sddi0",
        "dashboard_url": "https://dashboard.render.com/worker/srv-d31n3pumcj7s738sddi0",
        "verificacoes": {
            "sistema_ativo": {
                "descricao": "Sistema está rodando e gerando logs",
                "como_verificar": "Logs recentes (últimos 5-10 minutos)",
                "status": "❓ Não verificado"
            },
            "parametros_dna": {
                "descricao": "Parâmetros DNA genético corretos",
                "valores_esperados": {
                    "SL": "1.5%",
                    "TP": "12%", 
                    "leverage": "3x",
                    "EMA": "3/34"
                },
                "como_verificar": "Procurar por 'DNA' ou 'genetic' nos logs",
                "status": "❓ Não verificado"
            },
            "conexao_hyperliquid": {
                "descricao": "Conexão ativa com Hyperliquid",
                "como_verificar": "Procurar por 'Hyperliquid', 'API', 'connected'",
                "status": "❓ Não verificado"
            },
            "trades_ativos": {
                "descricao": "Sistema executando trades",
                "como_verificar": "Procurar por 'trade', 'buy', 'sell', 'position'",
                "status": "❓ Não verificado"
            },
            "sem_erros_criticos": {
                "descricao": "Ausência de erros que impedem funcionamento",
                "como_verificar": "Procurar por 'error', 'exception', 'failed'",
                "status": "❓ Não verificado"
            }
        },
        "commits_implementados": {
            "662f8e1": "FIX CRÍTICO: Parâmetros DNA Genético",
            "56e3f06": "FIX MATEMÁTICO: TP/SL com Alavancagem"
        }
    }
    
    filename = f"checklist_verificacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(checklist, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Checklist criado: {filename}")
    print("💡 Use este arquivo para anotar suas verificações!")
    
    return filename

if __name__ == "__main__":
    print(f"🕐 Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Abrir dashboard
    dashboard_url = abrir_dashboard_render()
    
    # Criar checklist
    checklist_file = criar_checklist_verificacao()
    
    print()
    print("=" * 50)
    print("🎯 RESUMO:")
    print(f"   Dashboard: {dashboard_url}")
    print(f"   Checklist: {checklist_file}")
    print(f"   Service ID: srv-d31n3pumcj7s738sddi0")
    print()
    print("🚀 PRÓXIMOS PASSOS:")
    print("   1. Verificar se o dashboard foi aberto")
    print("   2. Fazer login com GitHub")
    print("   3. Navegar até a aba 'Logs'")
    print("   4. Usar o checklist para verificar funcionamento")
    print("   5. Reportar status encontrado")
