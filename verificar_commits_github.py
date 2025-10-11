#!/usr/bin/env python3
"""
🔍 VERIFICADOR DE COMMITS NO GITHUB
==================================

Script que verifica diretamente no GitHub se os commits estão presentes
e baixa informações sobre eles.
"""

import requests
import json
from datetime import datetime

def verificar_commits_github():
    """
    Verifica os commits específicos no repositório GitHub
    """
    print("🔍 VERIFICAÇÃO DE COMMITS NO GITHUB")
    print("=" * 50)
    
    repo_owner = "Jao-jpedro"
    repo_name = "Cripto" 
    commits_alvo = ["662f8e1", "56e3f06"]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; GitHubCommitChecker/1.0)',
        'Accept': 'application/vnd.github.v3+json'
    })
    
    results = {}
    
    for commit_hash in commits_alvo:
        print(f"\n🔍 Verificando commit: {commit_hash}")
        print("-" * 30)
        
        try:
            # API do GitHub para commit específico
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_hash}"
            
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                commit_data = response.json()
                
                print(f"✅ COMMIT ENCONTRADO!")
                print(f"   SHA: {commit_data['sha']}")
                print(f"   Autor: {commit_data['commit']['author']['name']}")
                print(f"   Data: {commit_data['commit']['author']['date']}")
                print(f"   Mensagem: {commit_data['commit']['message']}")
                
                # Verificar arquivos modificados
                if 'files' in commit_data:
                    print(f"   Arquivos modificados: {len(commit_data['files'])}")
                    for file_info in commit_data['files']:
                        print(f"     📄 {file_info['filename']} ({file_info['status']})")
                
                results[commit_hash] = {
                    'found': True,
                    'data': commit_data,
                    'message': commit_data['commit']['message'],
                    'date': commit_data['commit']['author']['date'],
                    'files': [f['filename'] for f in commit_data.get('files', [])]
                }
                
            elif response.status_code == 404:
                print(f"❌ Commit não encontrado no repositório")
                results[commit_hash] = {'found': False, 'error': 'Not found'}
                
            else:
                print(f"⚠️ Status: {response.status_code}")
                results[commit_hash] = {'found': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            results[commit_hash] = {'found': False, 'error': str(e)}
    
    # Verificar branch atual e commits recentes
    print(f"\n📊 VERIFICAÇÃO DE BRANCH E COMMITS RECENTES")
    print("-" * 50)
    
    try:
        # Últimos commits do main
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
        response = session.get(url, params={'per_page': 10}, timeout=15)
        
        if response.status_code == 200:
            recent_commits = response.json()
            
            print(f"✅ Últimos {len(recent_commits)} commits encontrados:")
            
            for i, commit in enumerate(recent_commits):
                short_sha = commit['sha'][:7]
                message = commit['commit']['message'].split('\n')[0][:60]
                date = commit['commit']['author']['date']
                
                # Destacar se é um dos commits que procuramos
                marker = "🎯" if short_sha in commits_alvo else "  "
                
                print(f"   {marker} {short_sha} - {message} ({date})")
                
                # Se encontrou um dos commits nos recentes
                if short_sha in commits_alvo:
                    print(f"      ✅ COMMIT ALVO ENCONTRADO NOS RECENTES!")
        
    except Exception as e:
        print(f"❌ Erro ao buscar commits recentes: {e}")
    
    # Salvar resultados
    filename = f"commit_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Resultados salvos em: {filename}")
    
    # Resumo final
    print(f"\n📊 RESUMO:")
    print("-" * 20)
    
    found_count = sum(1 for r in results.values() if r.get('found', False))
    total_count = len(commits_alvo)
    
    print(f"   Commits verificados: {total_count}")
    print(f"   Commits encontrados: {found_count}")
    
    if found_count == total_count:
        print(f"   Status: ✅ TODOS OS COMMITS ESTÃO NO REPOSITÓRIO")
        print(f"\n💡 ISSO SIGNIFICA:")
        print(f"   🔧 As correções estão commitadas")
        print(f"   🚀 Se o Render está conectado ao GitHub, elas foram deployadas")
        print(f"   📊 O sistema deve estar usando os parâmetros corretos")
    else:
        print(f"   Status: ⚠️ ALGUNS COMMITS NÃO ENCONTRADOS")
    
    return results

if __name__ == "__main__":
    verificar_commits_github()
