#!/usr/bin/env python3
"""
📥 RENDER LOGS API - VERSÃO PYTHON OFICIAL
==========================================

Baseado na documentação oficial da API do Render
para download de logs com paginação automática.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode

class RenderLogsAPI:
    def __init__(self, token, service_id):
        self.token = token
        self.service_id = service_id
        self.base_url = "https://api.render.com/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'User-Agent': 'RenderLogsDownloader/1.0'
        })
    
    def download_logs(self, start_time, end_time, output_file=None):
        """
        Download logs usando a API oficial do Render
        """
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"render_logs_oficial_{timestamp}.json"
        
        print("🚀 RENDER LOGS API - DOWNLOAD OFICIAL")
        print("=" * 50)
        print(f"Token: {self.token[:20]}...")
        print(f"Service: {self.service_id}")
        print(f"Período: {start_time} até {end_time}")
        print(f"Arquivo: {output_file}")
        print()
        
        # Preparar parâmetros
        params = {
            'startTime': start_time,
            'endTime': end_time,
            'resourceIds[]': self.service_id
        }
        
        all_logs = []
        page = 1
        
        try:
            while True:
                print(f"📥 Página {page}: Fazendo requisição...")
                
                # Fazer requisição
                url = f"{self.base_url}/logs"
                response = self.session.get(url, params=params, timeout=30)
                
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"   ✅ JSON recebido")
                        
                        # Processar logs desta página
                        logs = data.get('logs', [])
                        if logs:
                            all_logs.extend(logs)
                            print(f"   📊 {len(logs)} logs nesta página")
                        else:
                            print(f"   📊 Nenhum log nesta página")
                        
                        # Verificar paginação
                        has_more = data.get('hasMore', False)
                        
                        if has_more:
                            # Atualizar parâmetros para próxima página
                            next_start = data.get('nextStartTime')
                            if next_start:
                                params['startTime'] = next_start
                                page += 1
                                
                                if page > 10:  # Limite de segurança
                                    print("⚠️ Limite de 10 páginas atingido")
                                    break
                                    
                                time.sleep(0.5)  # Rate limiting
                            else:
                                print("❌ nextStartTime não encontrado")
                                break
                        else:
                            print("📄 Última página")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ Erro JSON: {e}")
                        print(f"Resposta: {response.text[:200]}...")
                        break
                        
                elif response.status_code == 404:
                    print("❌ Endpoint não encontrado (404)")
                    # Tentar endpoint alternativo
                    return self._try_alternative_endpoints(start_time, end_time, output_file)
                    
                else:
                    print(f"❌ Erro HTTP {response.status_code}")
                    print(f"Resposta: {response.text[:200]}...")
                    break
            
            # Salvar todos os logs
            if all_logs:
                result = {
                    'metadata': {
                        'service_id': self.service_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'total_logs': len(all_logs),
                        'pages_downloaded': page,
                        'downloaded_at': datetime.now().isoformat()
                    },
                    'logs': all_logs
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ DOWNLOAD CONCLUÍDO!")
                print(f"📁 Arquivo: {output_file}")
                print(f"📊 Total de logs: {len(all_logs)}")
                print(f"📄 Páginas: {page}")
                
                # Analisar logs
                self._analyze_logs(all_logs, output_file)
                
                return output_file
            else:
                print(f"\n⚠️ Nenhum log encontrado")
                return None
                
        except Exception as e:
            print(f"❌ Erro durante download: {e}")
            return None
    
    def _try_alternative_endpoints(self, start_time, end_time, output_file):
        """
        Tenta endpoints alternativos quando o principal falha
        """
        print("\n🔄 Tentando endpoints alternativos...")
        
        alternative_endpoints = [
            f"/services/{self.service_id}/logs",
            f"/services/{self.service_id}/events",
            "/logs",  # Endpoint genérico
        ]
        
        for endpoint in alternative_endpoints:
            try:
                print(f"🔍 Testando: {endpoint}")
                
                url = f"{self.base_url}{endpoint}"
                params = {
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }
                
                if 'services' not in endpoint:
                    params['resourceIds[]'] = self.service_id
                
                response = self.session.get(url, params=params, timeout=20)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Diferentes estruturas possíveis
                        logs = data.get('logs', data.get('events', data if isinstance(data, list) else []))
                        
                        if logs:
                            print(f"✅ Sucesso! {len(logs)} items encontrados")
                            
                            result = {
                                'metadata': {
                                    'service_id': self.service_id,
                                    'endpoint': endpoint,
                                    'total_items': len(logs),
                                    'downloaded_at': datetime.now().isoformat()
                                },
                                'data': logs
                            }
                            
                            alt_file = output_file.replace('.json', f'_alternative.json')
                            with open(alt_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            
                            print(f"📁 Salvo em: {alt_file}")
                            self._analyze_logs(logs, alt_file)
                            
                            return alt_file
                            
                    except json.JSONDecodeError:
                        print(f"❌ Resposta não é JSON válido")
                        
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        return None
    
    def _analyze_logs(self, logs, filename):
        """
        Analisa os logs baixados
        """
        print(f"\n🔍 ANALISANDO LOGS: {filename}")
        print("-" * 40)
        
        if not logs:
            print("❌ Nenhum log para analisar")
            return
        
        # Estatísticas básicas
        print(f"📊 Total de entries: {len(logs)}")
        
        # Análise por tipo
        log_types = {}
        keywords_found = {
            'trading': 0,
            'dna': 0,
            'error': 0,
            'success': 0,
            'hyperliquid': 0
        }
        
        for log_entry in logs:
            # Extrair tipo/level
            log_type = 'unknown'
            content = ''
            
            if isinstance(log_entry, dict):
                log_type = log_entry.get('level', log_entry.get('type', 'unknown'))
                content = str(log_entry.get('message', log_entry.get('content', str(log_entry))))
            else:
                content = str(log_entry)
            
            # Contar tipos
            if log_type not in log_types:
                log_types[log_type] = 0
            log_types[log_type] += 1
            
            # Procurar palavras-chave
            content_lower = content.lower()
            for keyword in keywords_found:
                if keyword in content_lower:
                    keywords_found[keyword] += 1
        
        # Mostrar estatísticas
        print(f"\n📈 Tipos de log:")
        for log_type, count in sorted(log_types.items()):
            print(f"   {log_type}: {count}")
        
        print(f"\n🎯 Palavras-chave encontradas:")
        for keyword, count in keywords_found.items():
            if count > 0:
                emoji = "✅" if count > 0 else "❌"
                print(f"   {emoji} {keyword}: {count} ocorrências")
        
        # Salvar análise
        analysis = {
            'total_logs': len(logs),
            'log_types': log_types,
            'keywords': keywords_found,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        analysis_file = filename.replace('.json', '_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Análise salva: {analysis_file}")
        
        # Mostrar exemplos
        print(f"\n📋 Primeiros 3 logs:")
        for i, log in enumerate(logs[:3]):
            print(f"   {i+1}: {str(log)[:100]}...")

def main():
    # Configuração
    RENDER_TOKEN = "rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
    SERVICE_ID = "srv-d31n3pumcj7s738sddi0"
    
    # Período - últimas 24+ horas
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=30)  # 30h para ter margem
    
    # Converter para formato ISO
    start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Criar downloader
    downloader = RenderLogsAPI(RENDER_TOKEN, SERVICE_ID)
    
    # Baixar logs
    result_file = downloader.download_logs(start_iso, end_iso)
    
    if result_file:
        print(f"\n🎯 DOWNLOAD CONCLUÍDO!")
        print(f"📁 Arquivo principal: {result_file}")
        print(f"\n💡 Para analisar:")
        print(f"   cat {result_file} | jq '.logs[] | select(.message | contains(\"trade\"))'")
        print(f"   cat {result_file} | jq '.logs[] | select(.message | contains(\"DNA\"))'")
    else:
        print(f"\n❌ FALHA NO DOWNLOAD")
        print(f"Os logs podem não estar disponíveis via API para este serviço")

if __name__ == "__main__":
    main()
