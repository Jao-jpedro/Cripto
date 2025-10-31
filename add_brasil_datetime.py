#!/usr/bin/env python3
"""
Script para adicionar colunas de data e hora no horário do Brasil ao arquivo CSV
"""

import pandas as pd
import pytz
from datetime import datetime

def add_brasil_datetime_columns(csv_file_path):
    """Adiciona colunas de data e hora no horário do Brasil"""
    
    print(f"📊 Carregando arquivo: {csv_file_path}")
    
    # Carregar o arquivo CSV
    df = pd.read_csv(csv_file_path)
    
    print(f"📋 Registros carregados: {len(df):,}")
    print(f"🕐 Convertendo timestamps para horário do Brasil...")
    
    # Converter datetime para timezone do Brasil (UTC-3)
    # Primeiro, garantir que a coluna datetime está em formato datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Definir timezone UTC para a coluna datetime (assumindo que está em UTC)
    df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    
    # Converter para timezone do Brasil (America/Sao_Paulo - considera horário de verão)
    brasil_tz = pytz.timezone('America/Sao_Paulo')
    df_brasil = df['datetime'].dt.tz_convert(brasil_tz)
    
    # Extrair data e hora separadamente
    df['data_brasil'] = df_brasil.dt.strftime('%Y-%m-%d')
    df['hora_brasil'] = df_brasil.dt.strftime('%H:%M:%S')
    df['datetime_brasil'] = df_brasil.dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Remover timezone info da coluna datetime original para evitar problemas
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    # Reorganizar colunas para ter data e hora logo após datetime
    cols = df.columns.tolist()
    
    # Encontrar posição da coluna datetime
    datetime_pos = cols.index('datetime')
    
    # Remover as novas colunas da lista
    cols.remove('data_brasil')
    cols.remove('hora_brasil') 
    cols.remove('datetime_brasil')
    
    # Inserir as novas colunas após datetime
    cols.insert(datetime_pos + 1, 'datetime_brasil')
    cols.insert(datetime_pos + 2, 'data_brasil')
    cols.insert(datetime_pos + 3, 'hora_brasil')
    
    # Reordenar DataFrame
    df = df[cols]
    
    # Criar novo nome de arquivo
    original_name = csv_file_path.replace('.csv', '')
    new_file_path = f"{original_name}_com_horario_brasil.csv"
    
    print(f"💾 Salvando arquivo atualizado: {new_file_path}")
    
    # Salvar arquivo atualizado
    df.to_csv(new_file_path, index=False)
    
    # Estatísticas
    file_size_mb = round(len(open(new_file_path, 'rb').read()) / (1024*1024), 1)
    
    print(f"""
📈 ARQUIVO ATUALIZADO COM SUCESSO!

📁 Arquivo original: {csv_file_path}
📁 Arquivo novo: {new_file_path}
📊 Total de registros: {len(df):,}
💾 Tamanho do arquivo: {file_size_mb} MB

🕐 NOVAS COLUNAS ADICIONADAS:
   • datetime_brasil: {df['datetime_brasil'].iloc[0]} (completa)
   • data_brasil: {df['data_brasil'].iloc[0]} (só data)
   • hora_brasil: {df['hora_brasil'].iloc[0]} (só hora)

🔍 PREVIEW DAS PRIMEIRAS LINHAS:
""")
    
    # Mostrar preview das colunas de tempo
    time_cols = ['datetime', 'datetime_brasil', 'data_brasil', 'hora_brasil', 'asset_name', 'valor_fechamento']
    print(df[time_cols].head(10).to_string(index=False))
    
    return new_file_path

if __name__ == "__main__":
    # Arquivo de entrada
    input_file = "tradingv4_historical_data_20251025_192058.csv"
    
    print("🇧🇷 Adicionando horário do Brasil ao arquivo de dados históricos")
    print("=" * 70)
    
    try:
        new_file = add_brasil_datetime_columns(input_file)
        print(f"\n✅ Processo concluído! Arquivo salvo: {new_file}")
        
    except Exception as e:
        print(f"\n❌ Erro ao processar arquivo: {e}")
        import traceback
        traceback.print_exc()
