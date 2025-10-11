#!/usr/bin/env python3
"""
🎯 DOWNLOAD DIRETO DOS LOGS - API RENDER
=======================================

Usando API Key: rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR
Service ID: srv-d31n3pumcj7s738sddi0
"""

import requests
import json
import time
from datetime import datetime

class RenderAPILogDownloader:
    def __init__(self):
        self.api_key = "rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
        self.service_id = "srv-d31n3pumcj7s738sddi0"
        self.api_base = "https://api.render.com/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def get_service_info(self):
        """
        Busca informações básicas do serviço
        """
        print("🔍 BUSCANDO INFORMAÇÕES DO SERVIÇO")
        print("-" * 40)
        
        try:
            url = f"{self.api_base}/services/{self.service_id}"
            print(f"URL: {url}")
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                service_data = response.json()
                
                print("✅ INFORMAÇÕES DO SERVIÇO:")
                print(f"   Nome: {service_data.get('name', 'N/A')}")
                print(f"   Tipo: {service_data.get('type', 'N/A')}")
                print(f"   Status: {service_data.get('status', 'N/A')}")
                print(f"   Região: {service_data.get('region', 'N/A')}")
                print(f"   Criado: {service_data.get('createdAt', 'N/A')}")
                
                # Salvar info do serviço
                filename = f"service_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(service_data, f, indent=2, ensure_ascii=False)
                
                print(f"📁 Info salva: {filename}")
                return True
                
            else:
                print(f"❌ Erro: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def get_service_logs(self, limit=1000):
        """
        Busca logs do serviço
        """
        print(f"\n📊 BUSCANDO LOGS DO SERVIÇO (últimos {limit})")
        print("-" * 40)
        
        try:
            url = f"{self.api_base}/services/{self.service_id}/logs"
            params = {'limit': limit}
            
            print(f"URL: {url}")
            print(f"Params: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                print("✅ LOGS OBTIDOS COM SUCESSO!")
                
                try:
                    logs_data = response.json()
                    
                    # Salvar logs JSON
                    filename = f"render_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(logs_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"📁 Logs JSON salvos: {filename}")
                    
                    # Processar e analisar logs
                    if isinstance(logs_data, list):
                        print(f"📊 Total de entradas: {len(logs_data)}")
                        
                        # Converter para texto legível
                        log_text = ""
                        for entry in logs_data:
                            if isinstance(entry, dict):
                                timestamp = entry.get('timestamp', 'N/A')
                                message = entry.get('message', str(entry))
                                level = entry.get('level', 'INFO')
                                log_text += f"[{timestamp}] {level}: {message}\n"
                            else:
                                log_text += f"{entry}\n"
                        
                        # Salvar logs em formato texto
                        text_filename = f"render_logs_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        with open(text_filename, 'w', encoding='utf-8') as f:
                            f.write(f"Service ID: {self.service_id}\n")
                            f.write(f"Timestamp: {datetime.now()}\n")
                            f.write(f"Total entries: {len(logs_data)}\n")
                            f.write("=" * 60 + "\n")
                            f.write(log_text)
                        
                        print(f"📁 Logs texto salvos: {text_filename}")
                        
                        # Analisar logs
                        self._analyze_logs(log_text, logs_data)
                        
                        return True
                        
                    else:
                        print(f"⚠️ Formato inesperado dos logs: {type(logs_data)}")
                        return False
                        
                except json.JSONDecodeError:
                    # Resposta não é JSON
                    log_text = response.text
                    
                    filename = f"render_logs_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Service ID: {self.service_id}\n")
                        f.write(f"Timestamp: {datetime.now()}\n")
                        f.write("=" * 60 + "\n")
                        f.write(log_text)
                    
                    print(f"📁 Logs raw salvos: {filename}")
                    self._analyze_logs(log_text, None)
                    
                    return True
                    
            else:
                print(f"❌ Erro HTTP: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def get_service_events(self):
        """
        Busca eventos do serviço (deploys, builds, etc.)
        """
        print(f"\n🔄 BUSCANDO EVENTOS DO SERVIÇO")
        print("-" * 40)
        
        try:
            url = f"{self.api_base}/services/{self.service_id}/events"
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                events_data = response.json()
                
                print("✅ EVENTOS OBTIDOS:")
                
                if isinstance(events_data, list):
                    print(f"📊 Total de eventos: {len(events_data)}")
                    
                    # Mostrar últimos eventos
                    for event in events_data[:5]:
                        if isinstance(event, dict):
                            event_type = event.get('type', 'N/A')
                            status = event.get('status', 'N/A')
                            created_at = event.get('createdAt', 'N/A')
                            print(f"   📅 {created_at}: {event_type} - {status}")
                
                # Salvar eventos
                filename = f"service_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(events_data, f, indent=2, ensure_ascii=False)
                
                print(f"📁 Eventos salvos: {filename}")
                return True
                
            else:
                print(f"❌ Erro: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def _analyze_logs(self, log_text, raw_data):
        """
        Analisa logs de execução do sistema de trading
        """
        print(f"\n🔍 ANÁLISE DOS LOGS DE EXECUÇÃO")
        print("=" * 50)
        
        if not log_text:
            print("❌ Logs vazios")
            return
        
        lines = log_text.split('\n')
        print(f"📊 Total de linhas: {len(lines)}")
        
        # Análise por categorias
        categories = {
            '🎯 DNA/Algoritmo Genético': [
                'dna', 'genetic', 'algoritmo', 'sl:', 'tp:', 'leverage',
                '1.5%', '12%', '3x', 'ema'
            ],
            '💰 Trading Ativo': [
                'trade', 'buy', 'sell', 'position', 'profit', 'loss',
                'pnl', 'roi', 'balance'
            ],
            '🔗 Hyperliquid API': [
                'hyperliquid', 'api', 'fills', 'candles', 'connected',
                'market', 'order', 'wallet'
            ],
            '📊 Performance': [
                '+%', '-%', 'gain', 'profit', 'loss', 'roi',
                'performance', 'return'
            ],
            '⚠️ Erros/Problemas': [
                'error', 'exception', 'failed', 'timeout', 'erro',
                'warning', 'critical', 'crash'
            ],
            '🔄 Sistema/Status': [
                'started', 'stopped', 'running', 'ready', 'initialized',
                'status', 'health', 'alive'
            ],
            '💎 Criptomoedas': [
                'btc', 'eth', 'ada', 'sol', 'avax', 'bnb', 'xrp',
                'doge', 'link', 'ltc'
            ]
        }
        
        analysis_results = {}
        
        for category, keywords in categories.items():
            matches = []
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                for keyword in keywords:
                    if keyword.lower() in line_lower:
                        matches.append((line_num, line.strip()))
                        break
            
            analysis_results[category] = matches
            
            if matches:
                print(f"\n{category} ({len(matches)} ocorrências):")
                # Mostrar primeiras e últimas ocorrências
                to_show = matches[-3:] if len(matches) > 3 else matches
                for line_num, line in to_show:
                    preview = line[:100] + "..." if len(line) > 100 else line
                    print(f"   L{line_num}: {preview}")
        
        # Verificar commits específicos
        commits_to_check = ['662f8e1', '56e3f06']
        print(f"\n🔍 VERIFICAÇÃO DOS COMMITS:")
        print("-" * 30)
        
        for commit in commits_to_check:
            if commit in log_text:
                print(f"✅ Commit {commit} encontrado nos logs!")
            else:
                print(f"❌ Commit {commit} não encontrado")
        
        # Salvar análise
        analysis_file = f"log_analysis_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        serializable_results = {}
        for category, matches in analysis_results.items():
            serializable_results[category] = [
                {"line": line_num, "content": content}
                for line_num, content in matches
            ]
        
        analysis_summary = {
            "service_id": self.service_id,
            "timestamp": datetime.now().isoformat(),
            "total_lines": len(lines),
            "categories": serializable_results,
            "commits_found": {
                commit: commit in log_text for commit in commits_to_check
            },
            "summary": {
                "total_matches": sum(len(matches) for matches in analysis_results.values()),
                "active_categories": len([cat for cat, matches in analysis_results.items() if matches])
            }
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Análise detalhada salva: {analysis_file}")
        
        # Resumo executivo
        total_matches = analysis_summary["summary"]["total_matches"]
        active_cats = analysis_summary["summary"]["active_categories"]
        
        print(f"\n📊 RESUMO EXECUTIVO:")
        print("-" * 25)
        if total_matches > 0:
            print(f"✅ Sistema parece ATIVO! ({total_matches} indicadores)")
            print(f"📈 Categorias com atividade: {active_cats}/7")
            
            # Verificar saúde do sistema
            has_errors = len(analysis_results['⚠️ Erros/Problemas']) > 0
            has_trading = len(analysis_results['💰 Trading Ativo']) > 0
            has_dna = len(analysis_results['🎯 DNA/Algoritmo Genético']) > 0
            
            print(f"\n🏥 SAÚDE DO SISTEMA:")
            print(f"   💰 Trading ativo: {'✅' if has_trading else '❌'}")
            print(f"   🎯 DNA funcionando: {'✅' if has_dna else '❌'}")
            print(f"   ⚠️ Erros presentes: {'⚠️' if has_errors else '✅'}")
            
        else:
            print(f"⚠️ Poucos indicadores encontrados")
            print(f"💡 Sistema pode estar inativo ou com problemas")
    
    def run_complete_analysis(self):
        """
        Executa análise completa do serviço
        """
        print("🎯 ANÁLISE COMPLETA DO SERVIÇO RENDER")
        print("=" * 60)
        print(f"Service ID: {self.service_id}")
        print(f"API Key: {self.api_key[:20]}...")
        print()
        
        results = {
            "service_info": False,
            "service_logs": False,
            "service_events": False
        }
        
        # 1. Informações do serviço
        results["service_info"] = self.get_service_info()
        
        # 2. Logs do serviço
        results["service_logs"] = self.get_service_logs()
        
        # 3. Eventos do serviço
        results["service_events"] = self.get_service_events()
        
        # Resumo final
        print("\n" + "=" * 60)
        print("📊 RESULTADO FINAL:")
        success_count = sum(results.values())
        total_methods = len(results)
        
        print(f"   Métodos executados: {total_methods}")
        print(f"   Sucessos: {success_count}")
        
        for method, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {method.replace('_', ' ').title()}")
        
        if success_count > 0:
            print(f"\n✅ DADOS OBTIDOS COM SUCESSO!")
            print(f"   Verifique os arquivos gerados neste diretório")
        else:
            print(f"\n❌ Falha ao obter dados do serviço")

if __name__ == "__main__":
    downloader = RenderAPILogDownloader()
    downloader.run_complete_analysis()
