#!/usr/bin/env python3
"""Demo do triple-loop para mostrar timing das verificações"""

import time

print("🚀 DEMO: Triple-Loop Trading Bot")
print("⚡ Fast Safety Loop: 5s (PnL/ROI/Emergency)")  
print("🎯 Trailing Stop Check: 15s (Trailing adjustments)")
print("🔍 Full Analysis Loop: 45s (Technical analysis + new entries)")
print("Rodando por 90 segundos...\n")

start_time = time.time()
last_full_analysis = 0
last_trailing_check = 0
fast_sleep = 5
trailing_sleep = 15
slow_sleep = 45

iteration = 0
while time.time() - start_time < 90:
    iteration += 1
    current_time = time.time()
    
    # SEMPRE executa fast safety
    print(f"⚡ [{time.strftime('%H:%M:%S')}] FAST SAFETY - PnL/ROI checks")
    
    # Trailing stop check (15s)
    time_since_trailing = current_time - last_trailing_check
    should_run_trailing = (time_since_trailing >= trailing_sleep) or (iteration == 1)
    
    if should_run_trailing:
        print(f"    🎯 Trailing check (último há {time_since_trailing:.1f}s)")
        last_trailing_check = current_time
    
    # Full analysis (45s)
    time_since_analysis = current_time - last_full_analysis
    should_run_full = (time_since_analysis >= slow_sleep) or (iteration == 1)
    
    if should_run_full:
        print(f"    🔍 Full analysis (último há {time_since_analysis:.1f}s)")
        last_full_analysis = current_time
        print()  # linha em branco após análise completa
    
    time.sleep(fast_sleep)

print("✅ Demo finalizada")
print(f"Total de iterações: {iteration}")
print("\nResultado esperado:")
print("- Fast Safety: executada a cada 5s (todas as iterações)")  
print("- Trailing Stop: executada a cada ~15s (a cada 3ª iteração)")
print("- Full Analysis: executada a cada ~45s (a cada 9ª iteração)")
print("\nVantagem: Trailing stops agora respondem 4x mais rápido (15s vs 60s)!")
