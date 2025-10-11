#!/usr/bin/env python3
"""
🚀 LOG DOWNLOADER RENDER - MÉTODOS MÚLTIPLOS
===========================================

Script que tenta diferentes métodos para baixar logs do Render
sem necessidade de autenticação complexa.
"""

import requests
import json
import subprocess
import os
from datetime import datetime, timedelta
import sys

class RenderLogDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def method_1_curl_with_auth(self):
        """
        Método 1: Usar curl diretamente se você tiver API key
        """
        print("🔧 MÉTODO 1: API Direta do Render")
        print("-" * 40)
        
        api_key = os.getenv('RENDER_API_KEY')
        service_id = os.getenv('RENDER_SERVICE_ID')
        
        if not api_key or not service_id:
            print("❌ Variáveis de ambiente não encontradas:")
            print("   RENDER_API_KEY - Sua API key do Render")
            print("   RENDER_SERVICE_ID - ID do seu serviço")
            print()
            print("💡 Para obter:")
            print("   1. Acesse: https://dashboard.render.com/api-keys")
            print("   2. Crie uma API Key")
            print("   3. export RENDER_API_KEY='sua_key_aqui'")
            print("   4. export RENDER_SERVICE_ID='srv-abc123'")
            return False
        
        try:
            print(f"🔍 Tentando baixar logs do serviço: {service_id}")
            
            cmd = [
                'curl', '-H', f'Authorization: Bearer {api_key}',
                f'https://api.render.com/v1/services/{service_id}/logs?limit=1000'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                filename = f"render_logs_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    f.write(result.stdout)
                
                print(f"✅ Logs salvos em: {filename}")
                self._search_commits_in_file(filename)
                return True
            else:
                print(f"❌ Erro: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro na execução: {e}")
            return False
    
    def method_2_github_actions_logs(self):
        """
        Método 2: Verificar logs do GitHub Actions (deploy automático)
        """
        print("🔧 MÉTODO 2: Logs do GitHub Actions")
        print("-" * 40)
        
        repo_owner = "Jao-jpedro"  # Baseado no repoContext
        repo_name = "Cripto"
        
        try:
            # API pública do GitHub para actions
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs"
            
            print(f"🔍 Buscando deploys recentes em: {repo_owner}/{repo_name}")
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                filename = f"github_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Actions salvos em: {filename}")
                
                # Procurar por runs recentes
                if 'workflow_runs' in data:
                    for run in data['workflow_runs'][:5]:
                        print(f"📊 Run: {run.get('display_title', 'N/A')}")
                        print(f"   Status: {run.get('status', 'N/A')}")
                        print(f"   Conclusão: {run.get('conclusion', 'N/A')}")
                        print(f"   Data: {run.get('created_at', 'N/A')}")
                        print()
                
                return True
            else:
                print(f"❌ Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def method_3_render_status_endpoint(self):
        """
        Método 3: Tentar endpoint público de status (se existir)
        """
        print("🔧 MÉTODO 3: Endpoints Públicos do Render")
        print("-" * 40)
        
        # URLs comuns para serviços no Render
        possible_urls = [
            "https://cripto-trading.onrender.com",
            "https://trading-cripto.onrender.com", 
            "https://tradingv4.onrender.com",
            "https://cripto.onrender.com"
        ]
        
        for base_url in possible_urls:
            print(f"🔍 Testando: {base_url}")
            
            endpoints = ["/health", "/status", "/logs", "/api/status", "/ping"]
            
            for endpoint in endpoints:
                try:
                    url = base_url + endpoint
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        print(f"✅ ENCONTRADO: {url}")
                        print(f"   Status: {response.status_code}")
                        print(f"   Resposta: {response.text[:200]}...")
                        
                        filename = f"render_endpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Status: {response.status_code}\n")
                            f.write(f"Headers: {dict(response.headers)}\n")
                            f.write(f"Content:\n{response.text}")
                        
                        print(f"📁 Resposta salva em: {filename}")
                        return True
                        
                except Exception as e:
                    continue
        
        print("❌ Nenhum endpoint público encontrado")
        return False
    
    def method_4_check_local_logs(self):
        """
        Método 4: Verificar se existem logs locais ou arquivos de configuração
        """
        print("🔧 MÉTODO 4: Verificação Local")
        print("-" * 40)
        
        current_dir = os.getcwd()
        log_patterns = [
            "*.log", 
            "render.yaml", 
            "render.json",
            ".render*",
            "deployment.log",
            "app.log"
        ]
        
        found_files = []
        
        for pattern in log_patterns:
            try:
                import glob
                matches = glob.glob(pattern)
                found_files.extend(matches)
            except:
                pass
        
        if found_files:
            print("✅ Arquivos relacionados encontrados:")
            for file in found_files:
                print(f"   📁 {file}")
                
                # Verificar se contém informações sobre deploy
                try:
                    with open(file, 'r') as f:
                        content = f.read()
                        if any(word in content.lower() for word in ['render', 'deploy', '662f8e1', '56e3f06']):
                            print(f"      🎯 Contém informações relevantes!")
                            self._search_commits_in_content(content, file)
                except:
                    pass
            return True
        else:
            print("❌ Nenhum arquivo de log local encontrado")
            return False
    
    def _search_commits_in_file(self, filename):
        """Procura pelos commits específicos no arquivo"""
        try:
            with open(filename, 'r') as f:
                content = f.read()
            self._search_commits_in_content(content, filename)
        except Exception as e:
            print(f"❌ Erro ao procurar commits: {e}")
    
    def _search_commits_in_content(self, content, source):
        """Procura pelos commits no conteúdo"""
        commits = ['662f8e1', '56e3f06']
        
        print(f"\n🔍 Procurando commits em {source}:")
        
        for commit in commits:
            if commit in content:
                print(f"✅ COMMIT {commit} ENCONTRADO!")
                
                # Tentar extrair contexto
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if commit in line:
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        print(f"   Contexto (linhas {start}-{end}):")
                        for j in range(start, end):
                            prefix = ">>> " if j == i else "    "
                            print(f"   {prefix}{lines[j]}")
                        print()
            else:
                print(f"❌ Commit {commit} não encontrado")
    
    def run_all_methods(self):
        """Executa todos os métodos disponíveis"""
        print("🚀 RENDER LOG DOWNLOADER")
        print("=" * 50)
        print("Tentando múltiplos métodos para baixar logs...")
        print()
        
        methods = [
            ("API Direta", self.method_1_curl_with_auth),
            ("GitHub Actions", self.method_2_github_actions_logs),
            ("Endpoints Públicos", self.method_3_render_status_endpoint),
            ("Verificação Local", self.method_4_check_local_logs)
        ]
        
        success_count = 0
        
        for name, method in methods:
            try:
                if method():
                    success_count += 1
                print()
            except Exception as e:
                print(f"❌ Erro em {name}: {e}")
                print()
        
        print("=" * 50)
        print(f"📊 RESULTADO FINAL:")
        print(f"   Métodos executados: {len(methods)}")
        print(f"   Sucessos: {success_count}")
        
        if success_count == 0:
            print("\n💡 SUGESTÕES:")
            print("1. Acesse https://dashboard.render.com diretamente")
            print("2. Faça login com GitHub")
            print("3. Vá até seu serviço de trading")
            print("4. Clique em 'Logs' para ver em tempo real")
            print("5. Use Ctrl+F para procurar '662f8e1' e '56e3f06'")

if __name__ == "__main__":
    downloader = RenderLogDownloader()
    downloader.run_all_methods()
