#!/usr/bin/env python3
"""
📥 DOWNLOAD FORÇADO DE LOGS - ÚLTIMAS 24H
========================================

Script para forçar download de logs das últimas 24 horas
e salvar em arquivo local para análise offline.
"""

import requests
import json
import subprocess
import time
import os
from datetime import datetime, timedelta
import base64

class ForceLogDownloader:
    def __init__(self):
        self.api_key = "rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
        self.service_id = "srv-d31n3pumcj7s738sddi0"
        
        # Período - últimas 24 horas
        self.end_time = datetime.now()
        self.start_time = self.end_time - timedelta(hours=24)
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Accept': '*/*',
            'User-Agent': 'LogDownloader/1.0'
        })
    
    def method_1_curl_raw_logs(self):
        """
        Método 1: Curl raw para download direto
        """
        print("📥 MÉTODO 1: Download Raw via Curl")
        print("-" * 40)
        
        # Timestamps para filtrar
        start_ts = int(self.start_time.timestamp())
        end_ts = int(self.end_time.timestamp())
        
        curl_commands = [
            # Comando principal
            [
                'curl', '-s', '-L', '-H', f'Authorization: Bearer {self.api_key}',
                '-H', 'Accept: text/plain',
                f'https://api.render.com/v1/services/{self.service_id}/logs?since={start_ts}&until={end_ts}&limit=10000'
            ],
            # Alternativo com diferentes parâmetros
            [
                'curl', '-s', '-L', '-H', f'Authorization: Bearer {self.api_key}',
                '-H', 'Accept: application/json',
                f'https://api.render.com/v1/services/{self.service_id}/logs?tail=5000'
            ],
            # Sem filtro temporal
            [
                'curl', '-s', '-L', '-H', f'Authorization: Bearer {self.api_key}',
                f'https://api.render.com/v1/services/{self.service_id}/logs'
            ]
        ]
        
        for i, cmd in enumerate(curl_commands, 1):
            try:
                print(f"🔄 Comando {i}: Baixando logs...")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    content = result.stdout.strip()
                    
                    if content and len(content) > 10:
                        # Verificar se não é erro
                        if not any(error in content.lower() for error in ['404', 'not found', 'error', 'unauthorized']):
                            print(f"✅ Logs baixados! Tamanho: {len(content)} chars")
                            
                            filename = f"render_logs_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"# LOGS DO SERVIÇO RENDER - ÚLTIMAS 24H\n")
                                f.write(f"# Service ID: {self.service_id}\n")
                                f.write(f"# Período: {self.start_time} até {self.end_time}\n")
                                f.write(f"# Baixado em: {datetime.now()}\n")
                                f.write(f"# Comando: {' '.join(cmd[:4])}...\n")
                                f.write("=" * 80 + "\n\n")
                                f.write(content)
                            
                            print(f"📁 Arquivo salvo: {filename}")
                            
                            # Analisar conteúdo
                            self._analyze_downloaded_logs(content, filename)
                            return True
                        else:
                            print(f"⚠️ Resposta contém erro: {content[:100]}...")
                    else:
                        print(f"⚠️ Resposta vazia ou muito pequena")
                else:
                    print(f"❌ Curl falhou: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Erro no comando {i}: {e}")
        
        return False
    
    def method_2_force_api_download(self):
        """
        Método 2: Forçar download via API com diferentes headers
        """
        print("\n📥 MÉTODO 2: Download Forçado via API")
        print("-" * 40)
        
        endpoints = [
            f'https://api.render.com/v1/services/{self.service_id}/logs',
            f'https://api-us-west.render.com/v1/services/{self.service_id}/logs',
            f'https://api-eu.render.com/v1/services/{self.service_id}/logs'
        ]
        
        headers_variations = [
            {'Accept': 'text/plain', 'Content-Type': 'text/plain'},
            {'Accept': 'application/json'},
            {'Accept': '*/*'},
            {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
        ]
        
        for endpoint in endpoints:
            for headers in headers_variations:
                try:
                    # Combinar headers
                    combined_headers = {
                        'Authorization': f'Bearer {self.api_key}',
                        'User-Agent': 'Mozilla/5.0 (compatible; RenderLogDownloader)',
                        **headers
                    }
                    
                    print(f"🔄 Testando {endpoint} com {headers.get('Accept', 'default')}...")
                    
                    response = requests.get(endpoint, headers=combined_headers, timeout=30)
                    
                    if response.status_code == 200:
                        content = response.text
                        
                        if len(content) > 100:
                            print(f"✅ Download bem-sucedido! Tamanho: {len(content)} chars")
                            
                            filename = f"render_logs_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"# LOGS VIA API - ÚLTIMAS 24H\n")
                                f.write(f"# Endpoint: {endpoint}\n")
                                f.write(f"# Headers: {headers}\n")
                                f.write(f"# Status: {response.status_code}\n")
                                f.write(f"# Baixado em: {datetime.now()}\n")
                                f.write("=" * 80 + "\n\n")
                                f.write(content)
                            
                            print(f"📁 Arquivo salvo: {filename}")
                            self._analyze_downloaded_logs(content, filename)
                            return True
                    
                except Exception as e:
                    continue
        
        return False
    
    def method_3_extract_from_dashboard(self):
        """
        Método 3: Extrair logs do dashboard HTML
        """
        print("\n📥 MÉTODO 3: Extração do Dashboard")
        print("-" * 40)
        
        dashboard_urls = [
            f'https://dashboard.render.com/worker/{self.service_id}',
            f'https://dashboard.render.com/api/services/{self.service_id}/logs',
            f'https://dashboard.render.com/services/{self.service_id}/logs'
        ]
        
        for url in dashboard_urls:
            try:
                print(f"🔄 Acessando: {url}")
                
                response = self.session.get(url, timeout=20)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Procurar por dados de logs no HTML/JavaScript
                    if 'log' in content.lower() or 'message' in content.lower():
                        print(f"✅ Dados encontrados no dashboard!")
                        
                        filename = f"dashboard_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"<!-- EXTRAÇÃO DO DASHBOARD RENDER -->\n")
                            f.write(f"<!-- URL: {url} -->\n")
                            f.write(f"<!-- Extraído em: {datetime.now()} -->\n")
                            f.write(f"<!-- Status: {response.status_code} -->\n")
                            f.write("<!-- " + "="*70 + " -->\n\n")
                            f.write(content)
                        
                        print(f"📁 Dashboard salvo: {filename}")
                        
                        # Tentar extrair dados estruturados
                        self._extract_logs_from_html(content)
                        return True
                        
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        return False
    
    def method_4_binary_log_download(self):
        """
        Método 4: Download binário forçado
        """
        print("\n📥 MÉTODO 4: Download Binário")
        print("-" * 40)
        
        try:
            # Tentar endpoint de stream
            stream_url = f'https://api.render.com/v1/services/{self.service_id}/logs/stream'
            
            print(f"🔄 Tentando stream: {stream_url}")
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache'
            }
            
            response = requests.get(stream_url, headers=headers, stream=True, timeout=30)
            
            if response.status_code == 200:
                print("✅ Stream conectado! Coletando dados...")
                
                collected_data = ""
                start_time = time.time()
                
                # Coletar por 10 segundos
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        collected_data += chunk
                    
                    # Parar após 10 segundos
                    if time.time() - start_time > 10:
                        break
                
                if collected_data:
                    filename = f"render_logs_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# LOGS VIA STREAM - ÚLTIMAS 24H\n")
                        f.write(f"# URL: {stream_url}\n")
                        f.write(f"# Coletado em: {datetime.now()}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(collected_data)
                    
                    print(f"📁 Stream salvo: {filename}")
                    self._analyze_downloaded_logs(collected_data, filename)
                    return True
                    
        except Exception as e:
            print(f"❌ Erro no stream: {e}")
        
        return False
    
    def _extract_logs_from_html(self, html_content):
        """
        Extrai logs do conteúdo HTML
        """
        print("🔍 Extraindo logs do HTML...")
        
        # Procurar por padrões comuns de logs em JavaScript
        import re
        
        patterns = [
            r'"logs":\s*(\[.*?\])',
            r'"messages":\s*(\[.*?\])',
            r'logs:\s*(\[.*?\])',
            r'window\.logs\s*=\s*(\[.*?\])'
        ]
        
        extracted_logs = []
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.DOTALL)
            for match in matches:
                try:
                    logs_data = json.loads(match)
                    extracted_logs.extend(logs_data)
                except:
                    continue
        
        if extracted_logs:
            filename = f"extracted_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(extracted_logs, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Logs extraídos salvos: {filename}")
            
            # Converter para texto legível
            text_logs = ""
            for log in extracted_logs:
                if isinstance(log, dict):
                    timestamp = log.get('timestamp', 'N/A')
                    message = log.get('message', str(log))
                    text_logs += f"[{timestamp}] {message}\n"
                else:
                    text_logs += f"{log}\n"
            
            if text_logs:
                text_filename = f"extracted_logs_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(text_filename, 'w', encoding='utf-8') as f:
                    f.write(text_logs)
                
                print(f"📁 Logs em texto: {text_filename}")
                self._analyze_downloaded_logs(text_logs, text_filename)
    
    def _analyze_downloaded_logs(self, content, filename):
        """
        Analisa os logs baixados
        """
        print(f"\n🔍 ANALISANDO ARQUIVO: {filename}")
        print("-" * 40)
        
        if not content:
            print("❌ Conteúdo vazio")
            return
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        print(f"📊 Total de linhas: {total_lines}")
        
        # Análise por palavras-chave
        keywords = {
            '🧬 DNA/Genetic': ['dna', 'genetic', 'algoritmo', 'sl:', 'tp:', 'leverage'],
            '💰 Trading': ['trade', 'buy', 'sell', 'position', 'profit', 'loss'],
            '🔗 Hyperliquid': ['hyperliquid', 'api', 'fills', 'candles'],
            '💎 Cryptos': ['btc', 'eth', 'ada', 'sol', 'avax', 'bnb'],
            '⚠️ Erros': ['error', 'exception', 'failed', 'timeout'],
            '✅ Sucesso': ['success', 'completed', 'connected', 'started']
        }
        
        analysis = {}
        
        for category, words in keywords.items():
            matches = []
            for i, line in enumerate(lines, 1):
                for word in words:
                    if word.lower() in line.lower():
                        matches.append(f"L{i}: {line.strip()[:80]}...")
                        break
            
            analysis[category] = matches
            
            if matches:
                print(f"\n{category} ({len(matches)} ocorrências):")
                for match in matches[-3:]:  # Últimas 3
                    print(f"   {match}")
        
        # Salvar análise
        analysis_file = filename.replace('.txt', '_analysis.json').replace('.html', '_analysis.json')
        
        analysis_data = {
            'arquivo': filename,
            'total_linhas': total_lines,
            'analise': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Análise salva: {analysis_file}")
        
        # Resumo
        total_matches = sum(len(matches) for matches in analysis.values())
        
        if total_matches > 0:
            print(f"\n✅ ARQUIVO ANALISADO: {total_matches} indicadores encontrados")
        else:
            print(f"\n⚠️ Poucos indicadores encontrados no arquivo")
    
    def force_download_all_methods(self):
        """
        Executa todos os métodos de download
        """
        print("📥 DOWNLOAD FORÇADO DE LOGS - ÚLTIMAS 24H")
        print("=" * 60)
        print(f"Service ID: {self.service_id}")
        print(f"Período: {self.start_time} até {self.end_time}")
        print()
        
        methods = [
            ("Curl Raw", self.method_1_curl_raw_logs),
            ("API Forçada", self.method_2_force_api_download),
            ("Dashboard Extract", self.method_3_extract_from_dashboard),
            ("Binary Download", self.method_4_binary_log_download)
        ]
        
        success_count = 0
        downloaded_files = []
        
        for name, method in methods:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                if method():
                    success_count += 1
                    # Coletar arquivos gerados
                    import glob
                    recent_files = glob.glob(f"*{datetime.now().strftime('%Y%m%d')}*.txt") + \
                                  glob.glob(f"*{datetime.now().strftime('%Y%m%d')}*.json") + \
                                  glob.glob(f"*{datetime.now().strftime('%Y%m%d')}*.html")
                    downloaded_files.extend(recent_files)
                    
            except Exception as e:
                print(f"❌ Erro em {name}: {e}")
        
        # Resumo final
        print("\n" + "=" * 60)
        print("📊 RESUMO DO DOWNLOAD:")
        print(f"   Métodos testados: {len(methods)}")
        print(f"   Sucessos: {success_count}")
        
        # Listar arquivos únicos
        unique_files = list(set(downloaded_files))
        
        if unique_files:
            print(f"\n📁 ARQUIVOS BAIXADOS ({len(unique_files)}):")
            for file in sorted(unique_files):
                if os.path.exists(file):
                    size = os.path.getsize(file)
                    print(f"   📄 {file} ({size} bytes)")
        
        if success_count > 0:
            print(f"\n✅ DOWNLOAD CONCLUÍDO!")
            print(f"   Verifique os arquivos gerados para análise offline")
        else:
            print(f"\n❌ FALHA NO DOWNLOAD")
            print(f"   Logs podem não estar disponíveis via API")

if __name__ == "__main__":
    downloader = ForceLogDownloader()
    downloader.force_download_all_methods()
