#!/usr/bin/env python3
"""
🎯 MONITOR ESPECÍFICO DO SERVIÇO RENDER
======================================

Monitora o serviço: srv-d31n3pumcj7s738sddi0
URL: https://dashboard.render.com/worker/srv-d31n3pumcj7s738sddi0
"""

import requests
import json
import subprocess
import os
import time
from datetime import datetime

class RenderServiceMonitor:
    def __init__(self):
        self.service_id = "srv-d31n3pumcj7s738sddi0"
        self.service_url = f"https://dashboard.render.com/worker/{self.service_id}"
        self.api_base = "https://api.render.com/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def method_1_render_cli_specific(self):
        """
        Método 1: Usar render CLI com service ID específico
        """
        print("🎯 MÉTODO 1: Render CLI - Serviço Específico")
        print(f"Service ID: {self.service_id}")
        print("-" * 50)
        
        commands_to_try = [
            # Comando principal para logs
            ['render', 'services', 'logs', self.service_id],
            ['render', 'services', 'logs', self.service_id, '--tail', '50'],
            ['render', 'services', 'logs', self.service_id, '--follow'],
            
            # Informações do serviço
            ['render', 'services', 'get', self.service_id],
            ['render', 'services', 'list'],
        ]
        
        for cmd in commands_to_try:
            try:
                print(f"🔍 Executando: {' '.join(cmd)}")
                
                # Para logs com follow, usar timeout menor
                timeout = 15 if '--follow' in cmd else 30
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    print("✅ SUCESSO!")
                    print("=" * 60)
                    print(result.stdout)
                    print("=" * 60)
                    
                    # Salvar resultado
                    cmd_name = '_'.join(cmd[2:])  # services_logs_srv-xxx
                    filename = f"render_{cmd_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Comando: {' '.join(cmd)}\n")
                        f.write(f"Service ID: {self.service_id}\n")
                        f.write(f"Timestamp: {datetime.now()}\n")
                        f.write("=" * 60 + "\n")
                        f.write(result.stdout)
                        
                        if result.stderr:
                            f.write("\n" + "=" * 60 + "\n")
                            f.write("STDERR:\n")
                            f.write(result.stderr)
                    
                    print(f"📁 Resultado salvo: {filename}")
                    
                    # Analisar logs se for comando de logs
                    if 'logs' in cmd:
                        self._analyze_trading_logs(result.stdout)
                    
                    return True
                    
                else:
                    print(f"⚠️ Return code: {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    print()
                    
            except subprocess.TimeoutExpired:
                print("⏰ Timeout - comando interrompido")
                print()
            except Exception as e:
                print(f"❌ Erro: {e}")
                print()
        
        return False
    
    def method_2_api_with_possible_keys(self):
        """
        Método 2: Tentar API com possíveis chaves de ambiente
        """
        print("🎯 MÉTODO 2: API Render com Chaves Possíveis")
        print("-" * 50)
        
        # Possíveis nomes de variáveis de ambiente
        possible_env_vars = [
            'RENDER_API_KEY',
            'RENDER_TOKEN', 
            'RENDER_KEY',
            'API_KEY',
            'RENDER_AUTH'
        ]
        
        api_key = None
        
        # Tentar encontrar API key
        for var_name in possible_env_vars:
            key = os.getenv(var_name)
            if key:
                print(f"✅ Chave encontrada em: {var_name}")
                api_key = key
                break
        
        if not api_key:
            print("❌ Nenhuma API key encontrada nas variáveis de ambiente")
            print("💡 Variáveis testadas:", ', '.join(possible_env_vars))
            print()
            print("Para obter uma API key:")
            print("1. Acesse: https://dashboard.render.com/api-keys")
            print("2. Crie uma nova API Key")
            print("3. Execute: export RENDER_API_KEY='sua_key_aqui'")
            return False
        
        # Tentar buscar logs via API
        try:
            print(f"🔍 Buscando logs via API...")
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            }
            
            # Endpoint para logs
            logs_url = f"{self.api_base}/services/{self.service_id}/logs"
            
            print(f"URL: {logs_url}")
            
            response = self.session.get(logs_url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                print("✅ LOGS OBTIDOS VIA API!")
                
                try:
                    logs_data = response.json()
                    
                    # Salvar como JSON
                    filename = f"render_api_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(logs_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"📁 Logs JSON salvos: {filename}")
                    
                    # Extrair e analisar logs de texto
                    if isinstance(logs_data, list):
                        log_text = '\n'.join([entry.get('message', str(entry)) for entry in logs_data])
                    else:
                        log_text = str(logs_data)
                    
                    self._analyze_trading_logs(log_text)
                    
                    return True
                    
                except json.JSONDecodeError:
                    # Não é JSON, salvar como texto
                    log_text = response.text
                    
                    filename = f"render_api_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Service ID: {self.service_id}\n")
                        f.write(f"Timestamp: {datetime.now()}\n")
                        f.write("=" * 60 + "\n")
                        f.write(log_text)
                    
                    print(f"📁 Logs texto salvos: {filename}")
                    self._analyze_trading_logs(log_text)
                    
                    return True
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Erro na API: {e}")
        
        return False
    
    def method_3_direct_service_check(self):
        """
        Método 3: Verificar se o serviço tem endpoint público
        """
        print("🎯 MÉTODO 3: Verificação Direta do Serviço")
        print("-" * 50)
        
        # O worker no Render geralmente não tem URL pública, mas vamos tentar
        possible_urls = [
            f"https://{self.service_id}.onrender.com",
            "https://cripto-trading.onrender.com",
            "https://trading-bot.onrender.com"
        ]
        
        for url in possible_urls:
            try:
                print(f"🔍 Testando: {url}")
                
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    print(f"✅ SERVIÇO RESPONDENDO: {url}")
                    content = response.text
                    
                    # Verificar se é relacionado ao trading
                    trading_indicators = [
                        'trading', 'crypto', 'btc', 'eth', 'position', 
                        'profit', 'dna', 'genetic', 'hyperliquid'
                    ]
                    
                    found = [ind for ind in trading_indicators if ind.lower() in content.lower()]
                    
                    if found:
                        print(f"🎯 Indicadores de trading: {', '.join(found)}")
                        
                        # Salvar resposta
                        filename = f"service_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Service ID: {self.service_id}\n")
                            f.write(f"Timestamp: {datetime.now()}\n")
                            f.write("=" * 60 + "\n")
                            f.write(content)
                        
                        print(f"📁 Resposta salva: {filename}")
                        return True
                    else:
                        print(f"ℹ️ Resposta não parece ser do sistema de trading")
                        
                else:
                    print(f"   Status: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Erro: {str(e)[:50]}...")
        
        return False
    
    def _analyze_trading_logs(self, log_content):
        """
        Analisa logs específicos do sistema de trading
        """
        print("\n🔍 ANÁLISE DOS LOGS DE TRADING:")
        print("-" * 40)
        
        if not log_content or len(log_content.strip()) == 0:
            print("❌ Logs vazios")
            return
        
        lines = log_content.split('\n')
        print(f"📊 Total de linhas: {len(lines)}")
        
        # Padrões específicos do sistema
        patterns = {
            '🎯 DNA/Genetic': ['dna', 'genetic', 'algoritmo', 'sl:', 'tp:', 'leverage'],
            '💰 Trading': ['trade', 'buy', 'sell', 'position', 'profit', 'loss'],
            '🔗 Hyperliquid': ['hyperliquid', 'api', 'fills', 'candles'],
            '📊 Performance': ['roi', 'pnl', '+%', '-%', 'balance'],
            '⚠️ Erros': ['error', 'exception', 'failed', 'timeout', 'erro'],
            '🔄 Sistema': ['started', 'stopped', 'connected', 'running']
        }
        
        analysis_results = {}
        
        for category, keywords in patterns.items():
            matches = []
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                if any(keyword.lower() in line_lower for keyword in keywords):
                    matches.append((line_num, line.strip()))
            
            analysis_results[category] = matches
            
            if matches:
                print(f"\n{category} ({len(matches)} ocorrências):")
                # Mostrar últimas 3 ocorrências
                for line_num, line in matches[-3:]:
                    print(f"   L{line_num}: {line[:80]}...")
        
        # Salvar análise detalhada
        analysis_file = f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            # Converter para formato serializável
            serializable_results = {}
            for category, matches in analysis_results.items():
                serializable_results[category] = [
                    {"line_number": line_num, "content": content} 
                    for line_num, content in matches
                ]
            
            json.dump({
                "service_id": self.service_id,
                "timestamp": datetime.now().isoformat(),
                "total_lines": len(lines),
                "analysis": serializable_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Análise detalhada salva: {analysis_file}")
        
        # Resumo rápido
        total_matches = sum(len(matches) for matches in analysis_results.values())
        if total_matches > 0:
            print(f"\n✅ Sistema parece estar ativo! ({total_matches} indicadores encontrados)")
        else:
            print(f"\n⚠️ Poucos indicadores de atividade encontrados")
    
    def run_monitoring(self):
        """
        Executa monitoramento completo do serviço específico
        """
        print("🎯 MONITOR ESPECÍFICO DO SERVIÇO RENDER")
        print("=" * 60)
        print(f"Service ID: {self.service_id}")
        print(f"Dashboard: {self.service_url}")
        print()
        
        methods = [
            ("Render CLI Específico", self.method_1_render_cli_specific),
            ("API com Chaves", self.method_2_api_with_possible_keys),
            ("Verificação Direta", self.method_3_direct_service_check)
        ]
        
        success_count = 0
        
        for name, method in methods:
            print(f"\n{'='*15} {name} {'='*15}")
            try:
                if method():
                    success_count += 1
                    print(f"✅ {name} - SUCESSO")
                else:
                    print(f"❌ {name} - SEM RESULTADOS")
            except Exception as e:
                print(f"❌ {name} - ERRO: {e}")
            print()
        
        print("=" * 60)
        print(f"📊 RESULTADO FINAL:")
        print(f"   Service ID: {self.service_id}")
        print(f"   Métodos testados: {len(methods)}")
        print(f"   Sucessos: {success_count}")
        
        if success_count > 0:
            print(f"\n✅ LOGS CAPTURADOS!")
            print(f"   Verifique os arquivos gerados neste diretório")
            print(f"   Procure por arquivos: render_*, log_analysis_*")
        else:
            print(f"\n💡 PRÓXIMO PASSO:")
            print(f"   Acesse: {self.service_url}")
            print(f"   Clique na aba 'Logs' para ver em tempo real")

if __name__ == "__main__":
    monitor = RenderServiceMonitor()
    monitor.run_monitoring()
