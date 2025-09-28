#!/usr/bin/env python3
"""
TradingFuturo100 - Sistema de execução em loop contínuo
Este script executa o sistema principal, espera terminar, e reinicia automaticamente
"""

import subprocess
import time
import os
import signal
import sys
from datetime import datetime

class TradingLoopManager:
    """Gerenciador de loop contínuo do sistema de trading"""
    
    def __init__(self):
        self.running = True
        self.execution_count = 0
        self.start_time = datetime.now()
        
        # Configurar handler para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handler para parar o loop graciosamente"""
        print(f"\n🔴 Sinal {signum} recebido. Parando o loop...")
        self.running = False
    
    def log_execution(self, execution_num, status, duration):
        """Log de cada execução"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('trading_loop_log.txt', 'a') as f:
            f.write(f"{timestamp} | Execução #{execution_num} | Status: {status} | Duração: {duration:.1f}s\n")
    
    def run_single_execution(self):
        """Executa uma única rodada do sistema"""
        self.execution_count += 1
        start_exec = time.time()
        
        print(f"\n{'='*60}")
        print(f"🚀 EXECUÇÃO #{self.execution_count}")
        print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Total desde início: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f}h")
        print(f"{'='*60}")
        
        try:
            # Configurar variáveis de ambiente
            env = os.environ.copy()
            env['LIVE_TRADING'] = '1'  # Sempre em modo live monitoring
            
            # Executar o sistema principal
            result = subprocess.run([
                'python3', 
                'TradingFuturo100_HyperLiquid.py'
            ], 
            env=env,
            cwd='/Users/joaoreis/Documents/GitHub/Cripto',
            capture_output=True,
            text=True,
            timeout=600  # Timeout de 10 minutos por execução
            )
            
            duration = time.time() - start_exec
            
            if result.returncode == 0:
                print("✅ Execução completada com sucesso")
                status = "SUCCESS"
            else:
                print(f"⚠️ Execução terminou com código {result.returncode}")
                status = f"EXIT_{result.returncode}"
                if result.stderr:
                    print(f"Erro: {result.stderr[:200]}...")
            
            # Mostrar parte da saída
            if result.stdout:
                lines = result.stdout.split('\n')
                if len(lines) > 10:
                    print("\n📄 Últimas linhas da saída:")
                    for line in lines[-10:]:
                        if line.strip():
                            print(f"  {line}")
            
            self.log_execution(self.execution_count, status, duration)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_exec
            print("⏰ Execução excedeu timeout de 10 minutos")
            self.log_execution(self.execution_count, "TIMEOUT", duration)
            
        except Exception as e:
            duration = time.time() - start_exec
            print(f"❌ Erro na execução: {e}")
            self.log_execution(self.execution_count, f"ERROR_{type(e).__name__}", duration)
        
        print(f"⏱️ Duração: {duration:.1f} segundos")
    
    def run_continuous_loop(self):
        """Executa o loop contínuo"""
        print("🔄 SISTEMA DE LOOP CONTÍNUO INICIADO")
        print("="*60)
        print("📋 Configurações:")
        print("  • Sistema: TradingFuturo100_HyperLiquid.py")
        print("  • Modo: LIVE_TRADING=1 (monitoramento contínuo)")
        print("  • Timeout por execução: 10 minutos")
        print("  • Intervalo entre execuções: 30 segundos")
        print("  • Log: trading_loop_log.txt")
        print("📝 Pressione Ctrl+C para parar o loop")
        print("="*60)
        
        while self.running:
            try:
                # Executar uma rodada
                self.run_single_execution()
                
                if not self.running:
                    break
                
                # Intervalo entre execuções
                print("\n⏳ Aguardando 30 segundos antes da próxima execução...")
                
                for i in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
                    if i % 10 == 0 and i > 0:
                        remaining = 30 - i
                        print(f"  ⏰ {remaining} segundos restantes...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Erro no loop principal: {e}")
                time.sleep(30)
        
        print(f"\n✋ Loop interrompido após {self.execution_count} execuções")
        print(f"⏰ Tempo total: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} horas")

def main():
    """Função principal"""
    print("🚀 TradingFuturo100 - Gerenciador de Loop Contínuo")
    print("="*60)
    
    # Verificar se o arquivo principal existe
    script_path = '/Users/joaoreis/Documents/GitHub/Cripto/TradingFuturo100_HyperLiquid.py'
    if not os.path.exists(script_path):
        print(f"❌ Arquivo não encontrado: {script_path}")
        sys.exit(1)
    
    # Inicializar e executar o loop
    manager = TradingLoopManager()
    manager.run_continuous_loop()

if __name__ == "__main__":
    main()
