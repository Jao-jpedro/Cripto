#!/usr/bin/env bash

# =====================================
# 📥 SCRIPT OFICIAL RENDER LOGS API
# =====================================
# Baseado na documentação oficial do Render
# Adaptado para seu serviço específico

# — configuração —
RENDER_TOKEN="rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
SERVICE_ID="srv-d31n3pumcj7s738sddi0"         # Seu serviço de trading
START="2025-10-05T00:00:00Z"                  # Últimas 24h+ 
END="2025-10-06T23:59:59Z"                    # Até agora
OUTPUT_FILE="logs_render_oficial_$(date +%Y%m%d_%H%M%S).json"

echo "🚀 BAIXANDO LOGS OFICIAIS DO RENDER"
echo "=================================="
echo "Token: ${RENDER_TOKEN:0:20}..."
echo "Service: $SERVICE_ID"
echo "Período: $START até $END"
echo "Arquivo: $OUTPUT_FILE"
echo ""

# — requisição inicial —
echo "📥 Fazendo requisição inicial..."
HTTP_STATUS=$(curl -sG "https://api.render.com/v1/logs" \
  -H "Authorization: Bearer ${RENDER_TOKEN}" \
  -H "Accept: application/json" \
  --data-urlencode "startTime=${START}" \
  --data-urlencode "endTime=${END}" \
  --data-urlencode "resourceIds[]=${SERVICE_ID}" \
  -w "%{http_code}" \
  -o "${OUTPUT_FILE}")

echo "Status HTTP: $HTTP_STATUS"

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Requisição inicial bem-sucedida!"
    
    # Verificar se o arquivo tem conteúdo válido
    if [ -s "$OUTPUT_FILE" ]; then
        echo "📊 Arquivo criado com $(wc -c < "$OUTPUT_FILE") bytes"
        
        # Verificar se é JSON válido
        if jq empty "$OUTPUT_FILE" 2>/dev/null; then
            echo "✅ JSON válido recebido"
            
            # Mostrar estrutura do JSON
            echo ""
            echo "📋 Estrutura do JSON:"
            jq 'keys' "$OUTPUT_FILE" 2>/dev/null || echo "Não foi possível analisar estrutura"
            
            # — paginar se tiver mais —
            echo ""
            echo "� Verificando paginação..."
            
            has_more=$(jq -r '.hasMore // false' "$OUTPUT_FILE" 2>/dev/null)
            
            if [ "$has_more" = "true" ]; then
                echo "📄 Há mais páginas para baixar..."
                
                page_count=1
                while [ "$has_more" = "true" ] && [ $page_count -lt 10 ]; do  # Limite de 10 páginas
                    page_count=$((page_count + 1))
                    
                    # extrair nextStartTime e nextEndTime do JSON
                    NEXT_START=$(jq -r '.nextStartTime // empty' "${OUTPUT_FILE}")
                    NEXT_END=$(jq -r '.nextEndTime // empty' "${OUTPUT_FILE}")
                    
                    echo "� Página $page_count: $NEXT_START"
                    
                    if [ -n "$NEXT_START" ]; then
                        temp_file="temp_page_${page_count}.json"
                        
                        curl -sG "https://api.render.com/v1/logs" \
                          -H "Authorization: Bearer ${RENDER_TOKEN}" \
                          -H "Accept: application/json" \
                          --data-urlencode "startTime=${NEXT_START}" \
                          --data-urlencode "endTime=${END}" \
                          --data-urlencode "resourceIds[]=${SERVICE_ID}" \
                          -o "${temp_file}"
                        
                        # Combinar com arquivo principal (simplificado)
                        cat "${temp_file}" >> "${OUTPUT_FILE}"
                        rm "${temp_file}"
                        
                        has_more=$(jq -r '.hasMore // false' "${temp_file}" 2>/dev/null)
                    else
                        break
                    fi
                done
            else
                echo "📄 Apenas uma página de logs"
            fi
            
        else
            echo "❌ Resposta não é JSON válido"
            echo "Conteúdo recebido:"
            head -10 "$OUTPUT_FILE"
        fi
    else
        echo "❌ Arquivo vazio"
    fi
else
    echo "❌ Erro HTTP: $HTTP_STATUS"
    echo "Resposta:"
    cat "$OUTPUT_FILE"
fi

echo ""
echo "📊 RESULTADO FINAL:"
echo "=================="
echo "Arquivo: $OUTPUT_FILE"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Tamanho: $(wc -c < "$OUTPUT_FILE") bytes"
    echo "Linhas: $(wc -l < "$OUTPUT_FILE")"
fi

echo ""
echo "🔍 Para analisar os logs:"
echo "  cat $OUTPUT_FILE | jq ."
echo "  ou"
echo "  cat $OUTPUT_FILE | grep -i 'trade\\|dna\\|profit'"
