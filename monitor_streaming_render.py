#!/usr/bin/env python3
"""
📺 MONITOR STREAMING RENDER - LOGS EM TEMPO REAL
=============================================

Conecta ao streaming de logs do Render para capturar
dados em tempo real do trading system
"""

import requests
import json
import time
from datetime import datetime
import signal
import sys

class RenderStreamMonitor:
    def __init__(self, token, service_id):
        self.token = token
        self.service_id = service_id
        self.running = True
        self.logs_captured = []
        
        # Configurar handler para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print('\n🛑 Interrompendo captura...')
        self.running = False
        self.save_logs()
        sys.exit(0)
    
    def save_logs(self):
        if self.logs_captured:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"render_streaming_logs_{timestamp}.json"
            
            result = {
                'metadata': {
                    'service_id': self.service_id,
                    'captured_at': datetime.now().isoformat(),
                    'total_logs': len(self.logs_captured),
                    'duration': 'streaming'
                },
                'logs': self.logs_captured
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f'💾 Logs salvos: {filename}')
            print(f'📊 Total capturado: {len(self.logs_captured)} entradas')
        else:
            print('❌ Nenhum log capturado')
    
    def monitor_real_time(self):
        """
        Monitora logs em tempo real usando diferentes métodos
        """
        print("📺 MONITOR STREAMING RENDER")
        print("=" * 40)
        print(f"Service: {self.service_id}")
        print(f"Token: {self.token[:20]}...")
        print("Pressione Ctrl+C para parar e salvar")
        print()
        
        # Método 1: Pooling de eventos
        print("🔄 Iniciando monitoramento por polling...")
        
        last_event_id = None
        check_interval = 5  # segundos
        
        while self.running:
            try:
                # Buscar eventos recentes
                url = f"https://api.render.com/v1/services/{self.service_id}/events"
                params = {'limit': 50}
                
                headers = {
                    'Authorization': f'Bearer {self.token}',
                    'Accept': 'application/json'
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    events = response.json()
                    new_events = 0
                    
                    for event_wrapper in events:
                        event = event_wrapper.get('event', {})
                        event_id = event.get('id')
                        
                        # Só processar eventos novos
                        if last_event_id is None or event_id != last_event_id:
                            if self.is_relevant_event(event):
                                self.logs_captured.append({
                                    'timestamp': event.get('timestamp'),
                                    'type': event.get('type'),
                                    'details': event.get('details', {}),
                                    'captured_at': datetime.now().isoformat()
                                })
                                new_events += 1
                                
                                # Mostrar evento em tempo real
                                event_time = event.get('timestamp', 'N/A')
                                event_type = event.get('type', 'unknown')
                                print(f"🔔 {event_time}: {event_type}")
                                
                                # Mostrar detalhes se relevante
                                details = event.get('details', {})
                                if details:
                                    print(f"   💬 {str(details)[:100]}...")
                    
                    if events and not last_event_id:
                        last_event_id = events[0].get('event', {}).get('id')
                    
                    if new_events == 0:
                        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Aguardando novos eventos...")
                    
                else:
                    print(f"❌ Erro HTTP {response.status_code}")
                    if response.status_code == 429:
                        print("🚦 Rate limit - aumentando intervalo")
                        check_interval = min(check_interval * 2, 30)
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"💥 Erro: {e}")
                time.sleep(check_interval)
        
        self.save_logs()
    
    def is_relevant_event(self, event):
        """
        Determina se um evento é relevante para logging
        """
        event_type = event.get('type', '').lower()
        
        relevant_types = [
            'deploy_started',
            'deploy_ended', 
            'build_started',
            'build_ended',
            'service_started',
            'service_stopped',
            'service_error',
            'runtime_error',
            'log_message'
        ]
        
        return any(rtype in event_type for rtype in relevant_types)
    
    def check_current_status(self):
        """
        Verifica status atual do serviço
        """
        print("🔍 VERIFICANDO STATUS ATUAL")
        print("-" * 30)
        
        try:
            # Info do serviço
            url = f"https://api.render.com/v1/services/{self.service_id}"
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                service_info = response.json()
                
                print(f"📊 Nome: {service_info.get('serviceName', 'N/A')}")
                print(f"🏷️ Tipo: {service_info.get('type', 'N/A')}")
                print(f"🌐 Repo: {service_info.get('repo', 'N/A')}")
                print(f"🌿 Branch: {service_info.get('branch', 'N/A')}")
                print(f"🔄 Auto Deploy: {service_info.get('autoDeploy', 'N/A')}")
                
                # Último deploy
                service_details = service_info.get('serviceDetails', {})
                if service_details:
                    print(f"📦 Comando build: {service_details.get('buildCommand', 'N/A')}")
                    print(f"▶️ Comando start: {service_details.get('startCommand', 'N/A')}")
                
                # Verificar deploys recentes
                self.check_recent_deploys()
                
            else:
                print(f"❌ Erro ao buscar info: {response.status_code}")
                
        except Exception as e:
            print(f"💥 Erro: {e}")
    
    def check_recent_deploys(self):
        """
        Verifica deploys recentes
        """
        print(f"\n🚀 DEPLOYS RECENTES")
        print("-" * 20)
        
        try:
            url = f"https://api.render.com/v1/services/{self.service_id}/deploys"
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, params={'limit': 5}, timeout=10)
            
            if response.status_code == 200:
                deploys = response.json()
                
                for deploy_wrapper in deploys[:3]:  # Últimos 3
                    deploy = deploy_wrapper.get('deploy', {})
                    
                    deploy_id = deploy.get('id', 'N/A')
                    status = deploy.get('status', 'N/A')
                    created = deploy.get('createdAt', 'N/A')
                    finished = deploy.get('finishedAt', 'N/A')
                    
                    commit = deploy.get('commit', {})
                    commit_msg = commit.get('message', 'N/A')[:50] + "..."
                    
                    print(f"📦 {deploy_id}: {status}")
                    print(f"   ⏰ {created}")
                    print(f"   💬 {commit_msg}")
                    print()
                    
            else:
                print(f"❌ Erro ao buscar deploys: {response.status_code}")
                
        except Exception as e:
            print(f"💥 Erro: {e}")

def main():
    TOKEN = "rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
    SERVICE_ID = "srv-d31n3pumcj7s738sddi0"
    
    monitor = RenderStreamMonitor(TOKEN, SERVICE_ID)
    
    # Verificar status atual primeiro
    monitor.check_current_status()
    
    print(f"\n" + "="*50)
    print("MONITORAMENTO EM TEMPO REAL")
    print("="*50)
    
    # Iniciar monitoramento
    monitor.monitor_real_time()

if __name__ == "__main__":
    main()
