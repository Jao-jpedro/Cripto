#!/usr/bin/env python3
"""
📊 MONITOR DE LOGS DE EXECUÇÃO - RENDER
======================================

Script para capturar logs de funcionamento em tempo real
do sistema de trading no Render.
"""

import requests
import json
import time
import subprocess
import os
from datetime import datetime, timedelta
import threading

class RenderLogMonitor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.monitoring = False
    
    def method_1_render_cli_logs(self):
        """
        Método 1: Usar render CLI que já está instalado
        """
        print("🔧 MÉTODO 1: Render CLI - Logs em Tempo Real")
        print("-" * 50)
        
        try:
            # Primeiro, tentar listar serviços
            print("🔍 Listando serviços disponíveis...")
            
            result = subprocess.run(
                ['render', 'services', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Serviços encontrados:")
                print(result.stdout)
                
                # Tentar extrair service ID do output
                lines = result.stdout.split('\n')
                service_id = None
                
                for line in lines:
                    if 'srv-' in line:
                        # Extrair service ID
                        parts = line.split()
                        for part in parts:
                            if part.startswith('srv-'):
                                service_id = part
                                break
                        if service_id:
                            break
                
                if service_id:
                    print(f"🎯 Service ID encontrado: {service_id}")
                    return self._get_service_logs(service_id)
                else:
                    print("❌ Nenhum service ID encontrado no output")
                    
            else:
                print("❌ Erro ao listar serviços:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - comando demorou muito")
        except Exception as e:
            print(f"❌ Erro: {e}")
        
        return False
    
    def _get_service_logs(self, service_id):
        """
        Busca logs de um serviço específico
        """
        print(f"📊 Buscando logs do serviço: {service_id}")
        
        try:
            # Comando para buscar logs
            cmd = ['render', 'services', 'logs', service_id, '--tail', '100']
            
            print(f"🔍 Executando: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if result.returncode == 0:
                logs = result.stdout
                
                if logs.strip():
                    print("✅ LOGS ENCONTRADOS!")
                    print("=" * 60)
                    print(logs)
                    print("=" * 60)
                    
                    # Salvar logs
                    filename = f"render_execution_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Logs do serviço: {service_id}\n")
                        f.write(f"Data: {datetime.now()}\n")
                        f.write("=" * 60 + "\n")
                        f.write(logs)
                    
                    print(f"📁 Logs salvos em: {filename}")
                    
                    # Analisar logs para informações específicas
                    self._analyze_execution_logs(logs)
                    
                    return True
                else:
                    print("⚠️ Logs vazios - serviço pode não estar ativo")
            else:
                print("❌ Erro ao buscar logs:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Erro: {e}")
        
        return False
    
    def method_2_direct_service_monitoring(self):
        """
        Método 2: Monitoramento direto via URLs possíveis
        """
        print("🔧 MÉTODO 2: Monitoramento Direto de Serviços")
        print("-" * 50)
        
        # URLs mais prováveis baseadas no nome do repo
        possible_services = [
            "https://cripto.onrender.com",
            "https://trading-cripto.onrender.com",
            "https://cripto-trading.onrender.com",
            "https://tradingv4.onrender.com",
            "https://cripto-bot.onrender.com"
        ]
        
        active_services = []
        
        for url in possible_services:
            try:
                print(f"🔍 Testando: {url}")
                
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    print(f"✅ SERVIÇO ATIVO: {url}")
                    print(f"   Status: {response.status_code}")
                    
                    # Tentar extrair informações úteis
                    content = response.text
                    
                    if len(content) > 50:
                        print(f"   Resposta: {content[:200]}...")
                        
                        # Procurar por indicadores de trading
                        trading_indicators = [
                            'trading', 'position', 'profit', 'loss', 'balance',
                            'DNA', 'genetic', 'algorithm', 'ROI', 'BTC', 'ETH'
                        ]
                        
                        found_indicators = [ind for ind in trading_indicators if ind.lower() in content.lower()]
                        
                        if found_indicators:
                            print(f"   🎯 Indicadores encontrados: {', '.join(found_indicators)}")
                    
                    active_services.append(url)
                    
                    # Salvar resposta
                    filename = f"service_response_{url.split('//')[1].replace('.', '_')}_{datetime.now().strftime('%H%M%S')}.html"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"URL: {url}\n")
                        f.write(f"Status: {response.status_code}\n")
                        f.write(f"Headers: {dict(response.headers)}\n")
                        f.write(f"Timestamp: {datetime.now()}\n")
                        f.write("=" * 50 + "\n")
                        f.write(content)
                    
                    print(f"   📁 Resposta salva: {filename}")
                    
                else:
                    print(f"   Status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Erro: {str(e)[:50]}...")
            except Exception as e:
                print(f"   ❌ Erro inesperado: {e}")
        
        if active_services:
            print(f"\n✅ Serviços ativos encontrados: {len(active_services)}")
            return True
        else:
            print("\n❌ Nenhum serviço ativo encontrado")
            return False
    
    def method_3_webhook_monitoring(self):
        """
        Método 3: Tentar capturar webhooks ou logs via endpoints específicos
        """
        print("🔧 MÉTODO 3: Monitoramento via Webhooks/Endpoints")
        print("-" * 50)
        
        base_urls = [
            "https://cripto.onrender.com",
            "https://trading-cripto.onrender.com"
        ]
        
        log_endpoints = [
            "/logs", "/api/logs", "/status", "/health", 
            "/trades", "/positions", "/performance",
            "/debug", "/monitor", "/api/status"
        ]
        
        found_data = False
        
        for base_url in base_urls:
            for endpoint in log_endpoints:
                try:
                    url = base_url + endpoint
                    print(f"🔍 Tentando: {url}")
                    
                    response = self.session.get(url, timeout=8)
                    
                    if response.status_code == 200:
                        content = response.text
                        
                        # Verificar se parece com dados de trading
                        if any(keyword in content.lower() for keyword in 
                               ['trade', 'position', 'profit', 'dna', 'genetic', 'roi']):
                            
                            print(f"✅ DADOS DE TRADING ENCONTRADOS: {url}")
                            print(f"   Tamanho: {len(content)} bytes")
                            print(f"   Preview: {content[:150]}...")
                            
                            # Salvar dados
                            filename = f"trading_data_{endpoint.replace('/', '_')}_{datetime.now().strftime('%H%M%S')}.json"
                            
                            try:
                                # Tentar parsear como JSON
                                data = response.json()
                                with open(filename, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, indent=2, ensure_ascii=False)
                                print(f"   📁 JSON salvo: {filename}")
                            except:
                                # Salvar como texto
                                filename = filename.replace('.json', '.txt')
                                with open(filename, 'w', encoding='utf-8') as f:
                                    f.write(f"URL: {url}\n")
                                    f.write(f"Timestamp: {datetime.now()}\n")
                                    f.write("=" * 50 + "\n")
                                    f.write(content)
                                print(f"   📁 Texto salvo: {filename}")
                            
                            found_data = True
                            
                            # Analisar conteúdo
                            self._analyze_trading_data(content)
                            
                except Exception as e:
                    continue
        
        return found_data
    
    def _analyze_execution_logs(self, logs):
        """
        Analisa logs de execução para extrair informações úteis
        """
        print("\n🔍 ANÁLISE DOS LOGS DE EXECUÇÃO:")
        print("-" * 40)
        
        lines = logs.split('\n')
        
        # Procurar por padrões específicos
        patterns = {
            'trades': ['trade', 'buy', 'sell', 'position'],
            'errors': ['error', 'exception', 'failed', 'timeout'],
            'dna': ['dna', 'genetic', 'algorithm'],
            'performance': ['profit', 'loss', 'roi', 'pnl'],
            'connections': ['connected', 'disconnected', 'api'],
            'timestamps': ['2024', '2025', ':']
        }
        
        for category, keywords in patterns.items():
            matches = []
            for line in lines:
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    matches.append(line.strip())
            
            if matches:
                print(f"\n📊 {category.upper()} ({len(matches)} linhas):")
                for match in matches[-5:]:  # Últimas 5 linhas
                    print(f"   {match}")
    
    def _analyze_trading_data(self, content):
        """
        Analisa dados de trading encontrados
        """
        print("\n🔍 ANÁLISE DOS DADOS DE TRADING:")
        print("-" * 40)
        
        # Tentar extrair informações específicas
        if 'json' in content or '{' in content:
            try:
                data = json.loads(content)
                
                # Procurar por campos importantes
                important_fields = ['balance', 'positions', 'trades', 'roi', 'profit', 'dna']
                
                for field in important_fields:
                    if field in str(data).lower():
                        print(f"✅ Campo encontrado: {field}")
                        
            except:
                pass
        
        # Contar ocorrências de palavras-chave
        keywords = ['BTC', 'ETH', 'trade', 'position', 'profit', 'DNA', 'genetic']
        for keyword in keywords:
            count = content.upper().count(keyword.upper())
            if count > 0:
                print(f"📊 '{keyword}': {count} ocorrências")
    
    def run_monitoring(self):
        """
        Executa todos os métodos de monitoramento
        """
        print("📊 MONITOR DE LOGS DE EXECUÇÃO - RENDER")
        print("=" * 60)
        print("Buscando logs de funcionamento do sistema em tempo real...")
        print()
        
        methods = [
            ("Render CLI", self.method_1_render_cli_logs),
            ("Monitoramento Direto", self.method_2_direct_service_monitoring),  
            ("Webhooks/Endpoints", self.method_3_webhook_monitoring)
        ]
        
        success_count = 0
        
        for name, method in methods:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                if method():
                    success_count += 1
            except Exception as e:
                print(f"❌ Erro em {name}: {e}")
            print()
        
        print("=" * 60)
        print(f"📊 RESULTADO DO MONITORAMENTO:")
        print(f"   Métodos executados: {len(methods)}")
        print(f"   Sucessos: {success_count}")
        
        if success_count > 0:
            print(f"\n✅ LOGS CAPTURADOS COM SUCESSO!")
            print(f"   Verifique os arquivos gerados neste diretório")
        else:
            print(f"\n💡 ALTERNATIVAS:")
            print(f"   1. Acesse https://dashboard.render.com")
            print(f"   2. Vá até seu serviço")
            print(f"   3. Clique em 'Logs' para ver em tempo real")
            print(f"   4. Observe se o sistema está processando trades")

if __name__ == "__main__":
    monitor = RenderLogMonitor()
    monitor.run_monitoring()
