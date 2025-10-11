#!/usr/bin/env python3
"""
📊 ANÁLISE DE LOGS DAS ÚLTIMAS 24 HORAS
======================================

Script para buscar e analisar logs das últimas 24 horas do sistema de trading
"""

import requests
import json
import subprocess
from datetime import datetime, timedelta
import re

class RenderLogAnalyzer24h:
    def __init__(self):
        self.api_key = "rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
        self.service_id = "srv-d31n3pumcj7s738sddi0"
        self.api_base = "https://api.render.com/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        })
        
        # Período de análise - últimas 24 horas
        self.end_time = datetime.now()
        self.start_time = self.end_time - timedelta(hours=24)
    
    def get_recent_events_and_deploys(self):
        """
        Busca eventos e deploys das últimas 24 horas
        """
        print("🔍 BUSCANDO EVENTOS DAS ÚLTIMAS 24 HORAS")
        print("-" * 50)
        
        try:
            url = f"{self.api_base}/services/{self.service_id}/events"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                events = response.json()
                
                recent_events = []
                
                if isinstance(events, list):
                    for event_wrapper in events:
                        if 'event' in event_wrapper:
                            event = event_wrapper['event']
                            event_time_str = event.get('timestamp', '')
                            
                            try:
                                # Parse timestamp
                                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                                
                                # Verificar se está nas últimas 24 horas
                                if event_time.replace(tzinfo=None) >= self.start_time:
                                    recent_events.append(event)
                            except:
                                continue
                
                print(f"✅ Encontrados {len(recent_events)} eventos nas últimas 24h")
                
                # Analisar eventos
                self._analyze_recent_events(recent_events)
                
                return recent_events
            else:
                print(f"❌ Erro ao buscar eventos: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return []
    
    def try_multiple_log_endpoints(self):
        """
        Tenta múltiplos endpoints para obter logs
        """
        print("\n📊 TENTANDO MÚLTIPLOS ENDPOINTS PARA LOGS")
        print("-" * 50)
        
        # Diferentes estratégias para obter logs
        strategies = [
            self._try_direct_curl_with_params,
            self._try_alternative_api_versions,
            self._try_ssh_log_access,
            self._try_webhook_endpoints
        ]
        
        logs_found = False
        
        for strategy in strategies:
            try:
                if strategy():
                    logs_found = True
                    break
            except Exception as e:
                print(f"❌ Erro na estratégia: {e}")
                continue
        
        return logs_found
    
    def _try_direct_curl_with_params(self):
        """
        Tenta curl direto com diferentes parâmetros
        """
        print("🔧 Tentativa 1: curl direto com parâmetros temporais")
        
        # Calcular timestamps
        start_ts = int(self.start_time.timestamp())
        end_ts = int(self.end_time.timestamp())
        
        curl_commands = [
            [
                'curl', '-s', '-H', f'Authorization: Bearer {self.api_key}',
                f'https://api.render.com/v1/services/{self.service_id}/logs?since={start_ts}'
            ],
            [
                'curl', '-s', '-H', f'Authorization: Bearer {self.api_key}',
                f'https://api.render.com/v1/services/{self.service_id}/logs?from={start_ts}&to={end_ts}'
            ],
            [
                'curl', '-s', '-H', f'Authorization: Bearer {self.api_key}',
                f'https://api.render.com/v1/services/{self.service_id}/logs?tail=1000'
            ]
        ]
        
        for i, cmd in enumerate(curl_commands):
            try:
                print(f"   Comando {i+1}: curl com parâmetros temporais...")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and result.stdout.strip():
                    content = result.stdout.strip()
                    
                    # Verificar se não é apenas erro 404
                    if '404' not in content and 'not found' not in content.lower():
                        print(f"   ✅ Logs obtidos! Tamanho: {len(content)} chars")
                        
                        filename = f"logs_24h_curl_{datetime.now().strftime('%H%M%S')}.txt"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"Comando: {' '.join(cmd)}\n")
                            f.write(f"Período: {self.start_time} até {self.end_time}\n")
                            f.write("=" * 60 + "\n")
                            f.write(content)
                        
                        print(f"   📁 Salvos em: {filename}")
                        
                        # Analisar logs
                        self._analyze_trading_logs(content)
                        return True
                
            except Exception as e:
                print(f"   ❌ Erro comando {i+1}: {e}")
                continue
        
        return False
    
    def _try_alternative_api_versions(self):
        """
        Tenta versões alternativas da API
        """
        print("🔧 Tentativa 2: Versões alternativas da API")
        
        endpoints = [
            f'https://api.render.com/v2/services/{self.service_id}/logs',
            f'https://api.render.com/beta/services/{self.service_id}/logs',
            f'https://logs.render.com/services/{self.service_id}',
            f'https://dashboard.render.com/api/services/{self.service_id}/logs'
        ]
        
        for endpoint in endpoints:
            try:
                print(f"   Testando: {endpoint}")
                
                response = self.session.get(endpoint, timeout=15)
                
                if response.status_code == 200:
                    content = response.text
                    
                    if len(content) > 50:  # Conteúdo significativo
                        print(f"   ✅ Sucesso! Tamanho: {len(content)} chars")
                        
                        filename = f"logs_24h_api_{datetime.now().strftime('%H%M%S')}.txt"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"Endpoint: {endpoint}\n")
                            f.write(f"Status: {response.status_code}\n")
                            f.write("=" * 60 + "\n")
                            f.write(content)
                        
                        print(f"   📁 Salvos em: {filename}")
                        self._analyze_trading_logs(content)
                        return True
                
            except Exception as e:
                continue
        
        return False
    
    def _try_ssh_log_access(self):
        """
        Tenta acessar logs via SSH (se disponível)
        """
        print("🔧 Tentativa 3: Acesso SSH para logs")
        
        ssh_address = f"{self.service_id}@ssh.frankfurt.render.com"
        
        # Comandos SSH para tentar
        ssh_commands = [
            f'ssh -o ConnectTimeout=10 {ssh_address} "tail -n 1000 /var/log/*.log"',
            f'ssh -o ConnectTimeout=10 {ssh_address} "journalctl --since \\"24 hours ago\\""',
            f'ssh -o ConnectTimeout=10 {ssh_address} "find /var/log -name \\"*.log\\" -exec tail -n 100 {{}} \\;"'
        ]
        
        for cmd in ssh_commands:
            try:
                print(f"   Tentando SSH: {cmd[:50]}...")
                
                result = subprocess.run(
                    cmd.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=20
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    content = result.stdout
                    
                    print(f"   ✅ Logs SSH obtidos! Tamanho: {len(content)} chars")
                    
                    filename = f"logs_24h_ssh_{datetime.now().strftime('%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"SSH Command: {cmd}\n")
                        f.write("=" * 60 + "\n")
                        f.write(content)
                    
                    print(f"   📁 Salvos em: {filename}")
                    self._analyze_trading_logs(content)
                    return True
                
            except Exception as e:
                print(f"   ❌ SSH falhou: {e}")
                continue
        
        return False
    
    def _try_webhook_endpoints(self):
        """
        Tenta endpoints de webhook ou monitoring
        """
        print("🔧 Tentativa 4: Endpoints de webhook/monitoring")
        
        # Possíveis endpoints internos do Render
        webhook_endpoints = [
            f'https://api.render.com/v1/services/{self.service_id}/metrics',
            f'https://api.render.com/v1/services/{self.service_id}/health',
            f'https://api.render.com/v1/services/{self.service_id}/status',
            f'https://api.render.com/v1/services/{self.service_id}/deploys/latest/logs'
        ]
        
        for endpoint in webhook_endpoints:
            try:
                print(f"   Testando: {endpoint}")
                
                response = self.session.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if data and len(str(data)) > 50:
                            print(f"   ✅ Dados obtidos: {endpoint}")
                            
                            filename = f"monitoring_data_{datetime.now().strftime('%H%M%S')}.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                            
                            print(f"   📁 Dados salvos: {filename}")
                            
                            # Se contém logs ou informações úteis
                            if any(key in str(data).lower() for key in ['log', 'message', 'output', 'trade']):
                                self._analyze_monitoring_data(data)
                                return True
                            
                    except:
                        # Não é JSON, mas pode ser texto útil
                        content = response.text
                        if len(content) > 50:
                            filename = f"monitoring_text_{datetime.now().strftime('%H%M%S')}.txt"
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"Endpoint: {endpoint}\n")
                                f.write("=" * 60 + "\n")
                                f.write(content)
                            
                            print(f"   📁 Texto salvo: {filename}")
                            return True
                
            except Exception as e:
                continue
        
        return False
    
    def _analyze_recent_events(self, events):
        """
        Analisa eventos recentes
        """
        print("\n📊 ANÁLISE DE EVENTOS DAS ÚLTIMAS 24H")
        print("-" * 40)
        
        event_types = {}
        deploy_timeline = []
        
        for event in events:
            event_type = event.get('type', 'unknown')
            timestamp = event.get('timestamp', '')
            
            # Contar tipos de eventos
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
            
            # Timeline de deploys
            if 'deploy' in event_type:
                details = event.get('details', {})
                status = details.get('deployStatus', details.get('status', 'unknown'))
                
                deploy_timeline.append({
                    'timestamp': timestamp,
                    'type': event_type,
                    'status': status,
                    'details': details
                })
        
        print("📈 Tipos de eventos:")
        for event_type, count in event_types.items():
            print(f"   {event_type}: {count}")
        
        if deploy_timeline:
            print(f"\n🚀 Timeline de Deploys:")
            for deploy in sorted(deploy_timeline, key=lambda x: x['timestamp']):
                time_str = deploy['timestamp'][:19]  # YYYY-MM-DD HH:MM:SS
                print(f"   {time_str}: {deploy['type']} - {deploy['status']}")
                
                # Verificar se há commit hash
                details = deploy.get('details', {})
                if 'trigger' in details and 'newCommit' in details['trigger']:
                    commit = details['trigger']['newCommit'][:8]
                    print(f"      📝 Commit: {commit}")
        
        # Salvar análise de eventos
        events_analysis = {
            'period': f"{self.start_time} até {self.end_time}",
            'total_events': len(events),
            'event_types': event_types,
            'deploy_timeline': deploy_timeline,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        filename = f"events_analysis_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(events_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Análise de eventos salva: {filename}")
    
    def _analyze_trading_logs(self, log_content):
        """
        Análise específica dos logs de trading
        """
        print("\n🎯 ANÁLISE DOS LOGS DE TRADING (24H)")
        print("-" * 40)
        
        if not log_content:
            print("❌ Logs vazios")
            return
        
        lines = log_content.split('\n')
        total_lines = len(lines)
        
        print(f"📊 Total de linhas: {total_lines}")
        
        # Padrões para análise
        patterns = {
            '🧬 DNA/Genético': [
                r'dna|genetic|algoritmo',
                r'sl.*1\.5|tp.*12|leverage.*3x',
                r'ema.*3.*34'
            ],
            '💰 Trading': [
                r'trade|buy|sell|position',
                r'profit|loss|pnl|roi',
                r'balance|wallet'
            ],
            '🔗 Hyperliquid': [
                r'hyperliquid|api\.hyperliquid',
                r'fills|candles|market',
                r'connected|authenticated'
            ],
            '💎 Criptomoedas': [
                r'btc|eth|ada|sol|avax|bnb|xrp|doge|link|ltc',
                r'bitcoin|ethereum|cardano|solana'
            ],
            '⚠️ Erros': [
                r'error|exception|failed|timeout',
                r'warning|critical|crash'
            ],
            '✅ Sucesso': [
                r'success|completed|finished',
                r'connected|authenticated|started'
            ],
            '⏰ Timestamps': [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}:\d{2}:\d{2}'
            ]
        }
        
        analysis_results = {}
        
        for category, pattern_list in patterns.items():
            matches = []
            
            for line_num, line in enumerate(lines, 1):
                for pattern in pattern_list:
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append({
                            'line': line_num,
                            'content': line.strip()[:100],  # Primeiros 100 chars
                            'pattern': pattern
                        })
                        break
            
            analysis_results[category] = matches
            
            if matches:
                print(f"\n{category} ({len(matches)} ocorrências):")
                # Mostrar primeiras e últimas
                to_show = matches[-3:] if len(matches) > 3 else matches
                for match in to_show:
                    print(f"   L{match['line']}: {match['content']}...")
        
        # Análise temporal
        self._analyze_temporal_patterns(lines)
        
        # Análise de performance
        self._analyze_performance_indicators(lines)
        
        # Salvar análise completa
        complete_analysis = {
            'period': f"{self.start_time} até {self.end_time}",
            'total_lines': total_lines,
            'pattern_matches': analysis_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        filename = f"trading_logs_analysis_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(complete_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Análise completa salva: {filename}")
    
    def _analyze_temporal_patterns(self, lines):
        """
        Analisa padrões temporais nos logs
        """
        print(f"\n⏰ ANÁLISE TEMPORAL:")
        print("-" * 25)
        
        # Extrair timestamps
        timestamps = []
        for line in lines:
            # Procurar por padrões de timestamp
            timestamp_patterns = [
                r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
                r'\d{2}:\d{2}:\d{2}',
                r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]'
            ]
            
            for pattern in timestamp_patterns:
                match = re.search(pattern, line)
                if match:
                    timestamps.append(match.group())
                    break
        
        if timestamps:
            print(f"   📊 Timestamps encontrados: {len(timestamps)}")
            print(f"   🕐 Primeiro: {timestamps[0] if timestamps else 'N/A'}")
            print(f"   🕐 Último: {timestamps[-1] if timestamps else 'N/A'}")
            
            # Calcular intervalo médio entre logs
            if len(timestamps) > 1:
                print(f"   📈 Atividade: {len(timestamps)} entradas em 24h")
        else:
            print("   ⚠️ Nenhum timestamp encontrado nos logs")
    
    def _analyze_performance_indicators(self, lines):
        """
        Analisa indicadores de performance
        """
        print(f"\n📈 ANÁLISE DE PERFORMANCE:")
        print("-" * 30)
        
        # Procurar por indicadores específicos
        performance_patterns = {
            'Trades': r'trade.*\b(buy|sell)\b',
            'Profits': r'profit.*[\d\.]+[%$]?',
            'Losses': r'loss.*[\d\.]+[%$]?',
            'ROI': r'roi.*[\d\.]+%',
            'Balance': r'balance.*[\d\.]+'
        }
        
        performance_data = {}
        
        for indicator, pattern in performance_patterns.items():
            matches = []
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append(line.strip())
            
            performance_data[indicator] = matches
            
            if matches:
                print(f"   {indicator}: {len(matches)} ocorrências")
                if len(matches) <= 3:
                    for match in matches:
                        print(f"      {match[:80]}...")
        
        # Resumo de atividade
        total_activity = sum(len(matches) for matches in performance_data.values())
        
        if total_activity > 0:
            print(f"\n   📊 Total de atividade de trading: {total_activity} eventos")
            print(f"   💹 Sistema parece estar ATIVO!")
        else:
            print(f"\n   ⚠️ Pouca atividade de trading detectada")
    
    def _analyze_monitoring_data(self, data):
        """
        Analisa dados de monitoramento
        """
        print(f"\n📊 ANÁLISE DE DADOS DE MONITORAMENTO")
        print("-" * 40)
        
        # Procurar por campos importantes
        important_fields = [
            'status', 'health', 'uptime', 'errors', 'logs',
            'memory', 'cpu', 'trades', 'profit', 'balance'
        ]
        
        found_fields = {}
        
        def extract_fields(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    
                    if any(field.lower() in key.lower() for field in important_fields):
                        found_fields[full_key] = value
                    
                    if isinstance(value, (dict, list)):
                        extract_fields(value, full_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_fields(item, f"{prefix}[{i}]")
        
        extract_fields(data)
        
        if found_fields:
            print("📋 Campos importantes encontrados:")
            for field, value in found_fields.items():
                value_str = str(value)[:100]
                print(f"   {field}: {value_str}")
        else:
            print("⚠️ Nenhum campo de monitoramento importante encontrado")
    
    def generate_24h_report(self):
        """
        Gera relatório final das últimas 24 horas
        """
        print("\n📋 RELATÓRIO DAS ÚLTIMAS 24 HORAS")
        print("=" * 50)
        
        # Buscar dados
        recent_events = self.get_recent_events_and_deploys()
        logs_obtained = self.try_multiple_log_endpoints()
        
        # Gerar resumo
        report = {
            'periodo_analise': {
                'inicio': self.start_time.isoformat(),
                'fim': self.end_time.isoformat(),
                'duracao': '24 horas'
            },
            'eventos_encontrados': len(recent_events),
            'logs_obtidos': logs_obtained,
            'resumo_executivo': self._generate_executive_summary(recent_events, logs_obtained),
            'timestamp_relatorio': datetime.now().isoformat()
        }
        
        # Salvar relatório
        filename = f"relatorio_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Relatório final salvo: {filename}")
        
        # Exibir resumo
        print(f"\n🎯 RESUMO EXECUTIVO - ÚLTIMAS 24H:")
        print("-" * 35)
        for key, value in report['resumo_executivo'].items():
            print(f"   {key}: {value}")
        
        return report
    
    def _generate_executive_summary(self, events, logs_obtained):
        """
        Gera resumo executivo baseado nos dados coletados
        """
        summary = {}
        
        # Análise de eventos
        if events:
            deploy_events = [e for e in events if 'deploy' in e.get('type', '')]
            build_events = [e for e in events if 'build' in e.get('type', '')]
            
            summary['deploys_24h'] = len(deploy_events)
            summary['builds_24h'] = len(build_events)
            
            # Verificar sucesso dos deploys
            successful_deploys = [
                e for e in deploy_events 
                if e.get('details', {}).get('deployStatus') == 'succeeded'
            ]
            summary['deploys_sucessos'] = len(successful_deploys)
            
            # Último deploy
            if deploy_events:
                last_deploy = max(deploy_events, key=lambda x: x.get('timestamp', ''))
                summary['ultimo_deploy'] = last_deploy.get('timestamp', 'N/A')
                summary['ultimo_deploy_status'] = last_deploy.get('details', {}).get('deployStatus', 'N/A')
        else:
            summary['deploys_24h'] = 0
            summary['builds_24h'] = 0
            summary['deploys_sucessos'] = 0
            summary['ultimo_deploy'] = 'Nenhum'
        
        # Status dos logs
        summary['logs_acessiveis'] = 'Sim' if logs_obtained else 'Não'
        
        # Status geral do sistema
        if summary['deploys_sucessos'] > 0:
            summary['status_sistema'] = '✅ Ativo com deploys recentes'
        elif summary['deploys_24h'] > 0:
            summary['status_sistema'] = '⚠️ Ativo com alguns problemas de deploy'
        else:
            summary['status_sistema'] = '❓ Sem atividade de deploy detectada'
        
        return summary

if __name__ == "__main__":
    analyzer = RenderLogAnalyzer24h()
    analyzer.generate_24h_report()
