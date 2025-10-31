#!/usr/bin/env python3
"""
Script para adicionar colunas de current_k_atr para preços mínimo e máximo
"""

import pandas as pd

def add_k_atr_minmax_columns():
    """Adiciona colunas current_k_atr_min e current_k_atr_max"""
    
    print("📊 Carregando arquivo de dados...")
    
    # Carregar arquivo
    input_file = "tradingv4_dados_excel_brasileiro.csv"
    df = pd.read_csv(input_file, sep=';', encoding='utf-8-sig')
    
    print(f"📋 Registros carregados: {len(df):,}")
    
    print("🔢 Calculando current_k_atr para preços mínimo e máximo...")
    
    # Função para converter número brasileiro para float
    def convert_br_to_float(value):
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        try:
            if isinstance(value, str):
                # Remover separadores de milhares (pontos) e converter vírgula decimal
                clean_value = value.replace('.', '').replace(',', '.')
                return float(clean_value)
            else:
                return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    # Função para converter float para formato brasileiro
    def convert_float_to_br(value):
        if pd.isna(value) or value == 0:
            return ''
        
        try:
            num_value = float(value)
            
            if abs(num_value) >= 1:
                formatted = f"{num_value:.4f}".rstrip('0').rstrip('.')
            elif abs(num_value) >= 0.0001:
                formatted = f"{num_value:.6f}".rstrip('0').rstrip('.')
            else:
                formatted = f"{num_value:.8f}".rstrip('0').rstrip('.')
            
            return formatted.replace('.', ',')
            
        except (ValueError, TypeError):
            return ''
    
    # Converter colunas necessárias para cálculo
    print("   Convertendo valores para cálculo...")
    
    df['valor_maximo_calc'] = df['valor_maximo'].apply(convert_br_to_float)
    df['valor_minimo_calc'] = df['valor_minimo'].apply(convert_br_to_float)
    df['ema7_calc'] = df['ema7'].apply(convert_br_to_float)
    df['atr_calc'] = df['atr'].apply(convert_br_to_float)
    
    # Calcular current_k_atr para preço máximo
    print("   Calculando current_k_atr_max (baseado no preço máximo)...")
    df['current_k_atr_max_calc'] = df.apply(
        lambda row: abs(row['valor_maximo_calc'] - row['ema7_calc']) / row['atr_calc'] 
        if row['atr_calc'] > 0 else 0, axis=1
    )
    
    # Calcular current_k_atr para preço mínimo
    print("   Calculando current_k_atr_min (baseado no preço mínimo)...")
    df['current_k_atr_min_calc'] = df.apply(
        lambda row: abs(row['valor_minimo_calc'] - row['ema7_calc']) / row['atr_calc'] 
        if row['atr_calc'] > 0 else 0, axis=1
    )
    
    # Converter para formato brasileiro
    print("   Convertendo para formato brasileiro...")
    df['current_k_atr_max'] = df['current_k_atr_max_calc'].apply(convert_float_to_br)
    df['current_k_atr_min'] = df['current_k_atr_min_calc'].apply(convert_float_to_br)
    
    # Remover colunas temporárias de cálculo
    cols_to_remove = ['valor_maximo_calc', 'valor_minimo_calc', 'ema7_calc', 'atr_calc', 
                      'current_k_atr_max_calc', 'current_k_atr_min_calc']
    df = df.drop(columns=cols_to_remove)
    
    # Reorganizar colunas - colocar as novas colunas após current_k_atr
    cols = df.columns.tolist()
    
    # Encontrar posição da coluna current_k_atr
    try:
        current_k_atr_pos = cols.index('current_k_atr')
        
        # Remover as novas colunas da lista
        cols.remove('current_k_atr_max')
        cols.remove('current_k_atr_min')
        
        # Inserir as novas colunas após current_k_atr
        cols.insert(current_k_atr_pos + 1, 'current_k_atr_max')
        cols.insert(current_k_atr_pos + 2, 'current_k_atr_min')
        
        # Reordenar DataFrame
        df = df[cols]
        
    except ValueError:
        print("   ⚠️  Coluna current_k_atr não encontrada, adicionando no final")
    
    # Salvar arquivo atualizado
    output_file = "tradingv4_dados_excel_brasileiro_com_k_atr_minmax.csv"
    
    print(f"💾 Salvando arquivo atualizado: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        df.to_csv(f, index=False, sep=';', lineterminator='\r\n')
    
    # Estatísticas
    file_size_mb = round(len(open(output_file, 'rb').read()) / (1024*1024), 1)
    
    print(f"""
✅ COLUNAS ADICIONADAS COM SUCESSO!

📁 Arquivo original: {input_file}
📁 Arquivo atualizado: {output_file}
📊 Total de registros: {len(df):,}
💾 Tamanho: {file_size_mb} MB

🆕 NOVAS COLUNAS:
   • current_k_atr_max: Distância do preço MÁXIMO à EMA7 em unidades de ATR
   • current_k_atr_min: Distância do preço MÍNIMO à EMA7 em unidades de ATR

📊 INTERPRETAÇÃO:
   • current_k_atr: Distância do fechamento (close)
   • current_k_atr_max: Máxima penetração da vela acima/abaixo da EMA7
   • current_k_atr_min: Mínima penetração da vela acima/abaixo da EMA7
   • Diferença entre max/min mostra a volatilidade intracandle

🔍 PREVIEW DOS DADOS (AVNT):
""")
    
    # Mostrar preview do AVNT
    avnt_sample = df[df['asset_name'] == 'AVNT-USD'].tail(5)
    
    preview_cols = ['datetime_brasil', 'asset_name', 'valor_fechamento', 'valor_maximo', 
                   'valor_minimo', 'current_k_atr', 'current_k_atr_max', 'current_k_atr_min']
    
    available_cols = [col for col in preview_cols if col in df.columns]
    
    for i, row in avnt_sample.iterrows():
        print(f"📈 {row['datetime_brasil']} | AVNT")
        print(f"   Fechamento: {row['valor_fechamento']} | k_atr: {row['current_k_atr']}")
        print(f"   Máximo: {row['valor_maximo']} | k_atr_max: {row['current_k_atr_max']}")
        print(f"   Mínimo: {row['valor_minimo']} | k_atr_min: {row['current_k_atr_min']}")
        print()
    
    return output_file

if __name__ == "__main__":
    print("📊 Adicionando colunas current_k_atr para preços mínimo e máximo")
    print("=" * 70)
    
    try:
        new_file = add_k_atr_minmax_columns()
        print(f"\n✅ Processo concluído! Arquivo salvo: {new_file}")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
