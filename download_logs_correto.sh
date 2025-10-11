#!/usr/bin/env bash
# Script corrigido para logs do Render com parâmetros corretos
set -e

# Configuração
TOKEN="rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
SERVICE_ID="srv-d31n3pumcj7s738sddi0"
OWNER_ID="tea-d21ug4u3jp1c738b3n80"
OUTPUT="logs_render_correto_$(date +%Y%m%d_%H%M%S).json"

echo "🚀 DOWNLOAD LOGS RENDER - VERSÃO CORRIGIDA"
echo "=========================================="
echo "Owner ID: $OWNER_ID"
echo "Service: $SERVICE_ID"
echo "Output: $OUTPUT"
echo ""

# Período das últimas 4 horas para teste
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_TIME=$(date -u -v-4H +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -d "4 hours ago" +"%Y-%m-%dT%H:%M:%SZ")

echo "📅 Período: $START_TIME → $END_TIME"
echo ""

# Primeira tentativa: com filtros de resource corretos
echo "🔍 Tentativa 1: Filtro por serviceId..."

# Usar POST para evitar problemas com encoding de arrays na URL
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"ownerId\": \"$OWNER_ID\",
    \"startTime\": \"$START_TIME\",
    \"endTime\": \"$END_TIME\",
    \"filters\": {
      \"services\": [\"$SERVICE_ID\"]
    }
  }" \
  "https://api.render.com/v1/logs" > "$OUTPUT" 2>&1

# Verificar resultado
if [ -s "$OUTPUT" ]; then
  if jq . "$OUTPUT" >/dev/null 2>&1; then
    echo "✅ Resposta JSON válida"
    
    # Verificar se há events
    if jq -e '.events' "$OUTPUT" >/dev/null 2>&1; then
      LOG_COUNT=$(jq '.events | length' "$OUTPUT")
      echo "📈 Logs encontrados: $LOG_COUNT"
      
      if [ "$LOG_COUNT" -gt 0 ]; then
        echo ""
        echo "🎯 PRIMEIROS LOGS:"
        jq -r '.events[0:3][] | "\(.timestamp // "N/A"): \(.message // .level // .type // "unknown")"' "$OUTPUT" | head -5
        
        echo ""
        echo "📊 ANÁLISE DOS LOGS:"
        echo "==================="
        
        # Contar por tipo/nível
        echo "🏷️ Tipos de evento:"
        jq -r '.events[].type // "unknown"' "$OUTPUT" | sort | uniq -c | sort -nr
        
        echo ""
        echo "📈 Níveis de log:"
        jq -r '.events[].level // "unknown"' "$OUTPUT" | sort | uniq -c | sort -nr
        
        # Buscar mensagens relevantes
        echo ""
        echo "🔍 Mensagens com 'trade', 'DNA', 'error':"
        jq -r '.events[] | select(.message | test("trade|DNA|error|Trade|ERROR"; "i")) | "\(.timestamp): \(.message)"' "$OUTPUT" | head -5
        
      else
        echo "⚠️ Resposta válida mas sem logs"
      fi
    else
      echo "❌ Resposta sem campo 'events':"
      jq . "$OUTPUT"
    fi
  else
    echo "❌ Resposta não é JSON:"
    cat "$OUTPUT"
  fi
else
  echo "❌ Arquivo vazio"
fi

echo ""
echo "🔄 Tentativa 2: Todos os logs do owner..."

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"ownerId\": \"$OWNER_ID\",
    \"startTime\": \"$START_TIME\",
    \"endTime\": \"$END_TIME\"
  }" \
  "https://api.render.com/v1/logs" > "${OUTPUT}_all" 2>&1

if [ -s "${OUTPUT}_all" ]; then
  if jq . "${OUTPUT}_all" >/dev/null 2>&1; then
    ALL_COUNT=$(jq '.events | length' "${OUTPUT}_all" 2>/dev/null || echo "0")
    echo "📊 Total de logs (todos os serviços): $ALL_COUNT"
    
    if [ "$ALL_COUNT" -gt 0 ]; then
      # Filtrar apenas nosso serviço
      SERVICE_COUNT=$(jq "[.events[] | select(.serviceId == \"$SERVICE_ID\")] | length" "${OUTPUT}_all" 2>/dev/null || echo "0")
      echo "🎯 Logs específicos do nosso serviço: $SERVICE_COUNT"
      
      if [ "$SERVICE_COUNT" -gt 0 ]; then
        echo ""
        echo "🎉 LOGS DO NOSSO SERVIÇO ENCONTRADOS!"
        jq -r ".events[] | select(.serviceId == \"$SERVICE_ID\") | \"\(.timestamp): \(.message // .type)\"" "${OUTPUT}_all" | head -10
      fi
    fi
  fi
fi

echo ""
echo "🏁 Análise concluída!"
echo "📁 Arquivos gerados:"
echo "   - $OUTPUT (filtrado)"
echo "   - ${OUTPUT}_all (todos os logs)"
