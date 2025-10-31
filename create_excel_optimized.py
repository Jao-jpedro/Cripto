#!/usr/bin/env python3
"""
Script para criar arquivo CSV otimizado para Excel brasileiro
"""

import pandas as pd
import locale

def create_excel_optimized_csv():
    """Cria arquivo CSV otimizado para Excel brasileiro"""
    
    print("📊 Carregando arquivo original...")
    
    # Carregar arquivo original (com dados corretos)
    df = pd.read_csv("tradingv4_historical_data_20251025_192058_com_horario_brasil.csv")
    
    print(f"📋 Registros carregados: {len(df):,}")
    
    # Verificar dados do AVNT
    avnt_data = df[df['asset_name'] == 'AVNT-USD']
    print(f"🔍 Registros AVNT: {len(avnt_data)}")
    print(f"💰 AVNT - Min: ${avnt_data['valor_fechamento'].min():.6f}, Max: ${avnt_data['valor_fechamento'].max():.6f}")
    
    # Identificar colunas numéricas
    non_numeric_cols = ['datetime', 'datetime_brasil', 'data_brasil', 'hora_brasil', 'asset_name']
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    
    print(f"🔢 Convertendo {len(numeric_cols)} colunas numéricas...")
    
    # Converter números para formato brasileiro com mais cuidado
    for col in numeric_cols:
        print(f"   Processando: {col}")
        df[col] = df[col].apply(lambda x: format_number_excel_br(x))
    
    # Salvar com configurações específicas para Excel
    output_file = "tradingv4_dados_excel_brasileiro.csv"
    
    print(f"💾 Salvando: {output_file}")
    
    # Salvar com BOM UTF-8 para melhor compatibilidade com Excel
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        df.to_csv(f, index=False, sep=';', lineterminator='\r\n')
    
    # Verificar resultado
    file_size_mb = round(len(open(output_file, 'rb').read()) / (1024*1024), 1)
    
    print(f"""
✅ ARQUIVO EXCEL OTIMIZADO CRIADO!

📁 Arquivo: {output_file}
📊 Total de registros: {len(df):,}
💾 Tamanho: {file_size_mb} MB
🔧 Encoding: UTF-8 com BOM
📋 Separador: ponto e vírgula (;)
📈 Números: formato brasileiro

🔍 TESTE AVNT:
""")
    
    # Mostrar algumas linhas do AVNT para verificar
    avnt_sample = df[df['asset_name'] == 'AVNT-USD'].head(3)
    for i, row in avnt_sample.iterrows():
        print(f"   {row['datetime_brasil']} | AVNT | Fechamento: {row['valor_fechamento']}")

def format_number_excel_br(value):
    """Formata número para Excel brasileiro (mais rigoroso)"""
    if pd.isna(value) or value == '' or value is None:
        return ''
    
    try:
        # Converter para float
        if isinstance(value, str):
            if value.strip() == '':
                return ''
            # Se já tem vírgula, assumir que é brasileiro
            if ',' in value and '.' in value:
                clean_value = value.replace('.', '').replace(',', '.')
                num_value = float(clean_value)
            elif ',' in value:
                num_value = float(value.replace(',', '.'))
            else:
                num_value = float(value)
        else:
            num_value = float(value)
        
        # Formatação mais específica baseada no valor
        if abs(num_value) >= 10000:
            # Números grandes: separador de milhares
            if num_value == int(num_value):
                formatted = f"{int(num_value):,}".replace(',', '.')
            else:
                formatted = f"{num_value:,.2f}".replace(',', '|').replace('.', ',').replace('|', '.')
        elif abs(num_value) >= 1:
            # Números médios: até 4 casas decimais
            formatted = f"{num_value:.4f}".rstrip('0').rstrip('.')
            formatted = formatted.replace('.', ',')
        elif abs(num_value) >= 0.0001:
            # Números pequenos: até 6 casas decimais
            formatted = f"{num_value:.6f}".rstrip('0').rstrip('.')
            formatted = formatted.replace('.', ',')
        else:
            # Números muito pequenos: até 8 casas decimais
            formatted = f"{num_value:.8f}".rstrip('0').rstrip('.')
            formatted = formatted.replace('.', ',')
        
        return formatted
        
    except (ValueError, TypeError):
        return str(value)

if __name__ == "__main__":
    print("🇧🇷 Criando arquivo Excel otimizado para Brasil")
    print("=" * 60)
    
    try:
        create_excel_optimized_csv()
        print("\n✅ Arquivo criado com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
