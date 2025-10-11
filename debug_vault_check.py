#!/usr/bin/env python3
"""
Script para verificar se todas as operações do tradingv4.py estão usando vault address
"""

import re

def check_vault_usage(filename):
    """Verifica se todas as operações de trading usam vault address"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    issues = []
    
    # Padrões que devem sempre incluir vault address
    trading_patterns = [
        r'\.fetch_positions\(',
        r'\.fetch_open_orders\(',
        r'\.create_order\(',
        r'\.cancel_order\(',
        r'\.fetch_balance\(',
        r'\.fetch_my_trades\(',
    ]
    
    for line_num, line in enumerate(lines, 1):
        for pattern in trading_patterns:
            if re.search(pattern, line):
                # Verificar se a linha contém vault ou é uma definição de função
                if 'def ' in line or 'vaultAddress' in line or 'VAULT_ADDRESS' in line:
                    continue
                if 'fetch_ticker' in line:  # fetch_ticker não precisa de vault (dados públicos)
                    continue
                if 'simulate' in line.lower() or 'mock' in line.lower() or 'dummy' in line.lower():
                    continue
                if 'return' in line and 'def ' in lines[max(0, line_num-10):line_num]:
                    continue  # Provavelmente retorno simulado
                
                # Se chegou aqui, pode ser problemático
                issues.append(f"Linha {line_num}: {line.strip()}")
    
    return issues

if __name__ == "__main__":
    issues = check_vault_usage('tradingv4.py')
    
    print("=== VERIFICAÇÃO DE USO DE VAULT ADDRESS ===")
    if issues:
        print(f"❌ Encontradas {len(issues)} possíveis operações sem vault address:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Todas as operações de trading parecem usar vault address")
