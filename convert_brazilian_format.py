#!/usr/bin/env python3
"""
Script para converter arquivo CSV para formato brasileiro:
- Números: ponto como separador de milhares, vírgula como decimal
- Separador de colunas: ponto e vírgula (;)
"""

import pandas as pd
import locale
import numpy as np

def format_number_brazilian(value):
    """Converte número para formato brasileiro"""
    if pd.isna(value) or value == '':
        return ''
    
    try:
        # Converter para float se for string
        if isinstance(value, str):
            # Se já está em formato brasileiro, converter de volta para float primeiro
            if ',' in value and '.' in value:
                # Formato brasileiro -> float
                clean_value = value.replace('.', '').replace(',', '.')
                num_value = float(clean_value)
            elif ',' in value:
                # Apenas vírgula decimal
                num_value = float(value.replace(',', '.'))
            else:
                num_value = float(value)
        else:
            num_value = float(value)
        
        # Verificar se é um número inteiro
        if num_value == int(num_value):
            # Número inteiro - formatar com separador de milhares
            formatted = f"{int(num_value):,}".replace(',', '.')
        else:
            # Número decimal - formatar com 2-8 casas decimais dependendo do valor
            if abs(num_value) >= 1:
                # Valores >= 1: máximo 4 casas decimais
                formatted = f"{num_value:,.4f}".rstrip('0').rstrip('.')
            elif abs(num_value) >= 0.01:
                # Valores entre 0.01 e 1: máximo 6 casas decimais
                formatted = f"{num_value:,.6f}".rstrip('0').rstrip('.')
            else:
                # Valores muito pequenos: máximo 8 casas decimais
                formatted = f"{num_value:,.8f}".rstrip('0').rstrip('.')
            
            # Substituir separadores para formato brasileiro
            formatted = formatted.replace(',', '|').replace('.', ',').replace('|', '.')
        
        return formatted
        
    except (ValueError, TypeError):
        # Se não conseguir converter, retornar valor original
        return str(value)

def convert_to_brazilian_format(csv_file_path):
    """Converte arquivo CSV para formato brasileiro"""
    
    print(f"📊 Carregando arquivo: {csv_file_path}")
    
    # Carregar o arquivo CSV
    df = pd.read_csv(csv_file_path)
    
    print(f"📋 Registros carregados: {len(df):,}")
    print(f"🔢 Convertendo números para formato brasileiro...")
    
    # Identificar colunas numéricas (excluindo colunas de data/hora e texto)
    non_numeric_cols = ['datetime', 'datetime_brasil', 'data_brasil', 'hora_brasil', 'asset_name']
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    
    print(f"📊 Colunas numéricas identificadas: {len(numeric_cols)}")
    
    # Converter cada coluna numérica
    for col in numeric_cols:
        print(f"   Convertendo coluna: {col}")
        df[col] = df[col].apply(format_number_brazilian)
    
    # Criar novo nome de arquivo
    original_name = csv_file_path.replace('.csv', '')
    new_file_path = f"{original_name}_formato_brasileiro.csv"
    
    print(f"💾 Salvando arquivo no formato brasileiro: {new_file_path}")
    
    # Salvar arquivo com separador ponto e vírgula
    df.to_csv(new_file_path, index=False, sep=';', encoding='utf-8')
    
    # Estatísticas
    file_size_mb = round(len(open(new_file_path, 'rb').read()) / (1024*1024), 1)
    
    print(f"""
📈 ARQUIVO CONVERTIDO PARA FORMATO BRASILEIRO!

📁 Arquivo original: {csv_file_path}
📁 Arquivo brasileiro: {new_file_path}
📊 Total de registros: {len(df):,}
💾 Tamanho do arquivo: {file_size_mb} MB

🇧🇷 FORMATAÇÃO APLICADA:
   • Separador de colunas: ponto e vírgula (;)
   • Separador de milhares: ponto (.)
   • Separador decimal: vírgula (,)
   • Encoding: UTF-8

🔍 PREVIEW DAS PRIMEIRAS LINHAS:
""")
    
    # Mostrar preview de algumas colunas importantes
    preview_cols = ['datetime_brasil', 'asset_name', 'valor_fechamento', 'volume', 'atr_pct']
    available_cols = [col for col in preview_cols if col in df.columns]
    
    # Mostrar apenas primeiras 5 linhas para não sobrecarregar
    preview_df = df[available_cols].head(5)
    for i, row in preview_df.iterrows():
        print(f"Linha {i+1}:")
        for col in available_cols:
            print(f"   {col}: {row[col]}")
        print()
    
    return new_file_path

if __name__ == "__main__":
    # Arquivo de entrada
    input_file = "tradingv4_historical_data_20251025_192058_com_horario_brasil.csv"
    
    print("🇧🇷 Convertendo arquivo para formato numérico brasileiro")
    print("=" * 70)
    
    try:
        new_file = convert_to_brazilian_format(input_file)
        print(f"\n✅ Processo concluído! Arquivo brasileiro salvo: {new_file}")
        
    except Exception as e:
        print(f"\n❌ Erro ao processar arquivo: {e}")
        import traceback
        traceback.print_exc()
