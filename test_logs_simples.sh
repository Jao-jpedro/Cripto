#!/usr/bin/env bash
# Script simplificado para logs do Render
set -e

# Configuração
TOKEN="rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"
SERVICE_ID="srv-d31n3pumcj7s738sddi0"
OUTPUT="logs_render_$(date +%Y%m%d_%H%M%S).json"

echo "🚀 BAIXANDO LOGS DO RENDER"
echo "=========================="
echo "Service: $SERVICE_ID"
echo "Output: $OUTPUT"
echo ""

# Período curto para teste (últimas 6 horas)
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_TIME=$(date -u -v-6H +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -d "6 hours ago" +"%Y-%m-%dT%H:%M:%SZ")

echo "📅 Período: $START_TIME → $END_TIME"
echo ""

# Tentar requisição simples primeiro
echo "🔍 Testando endpoint de logs..."

curl -s -H "Authorization: Bearer $TOKEN" \
  "https://api.render.com/v1/logs?startTime=${START_TIME}&endTime=${END_TIME}&resourceIds[]=${SERVICE_ID}" \
  > "$OUTPUT" 2>&1

# Verificar resultado
if [ -s "$OUTPUT" ]; then
  echo "✅ Arquivo criado: $OUTPUT"
  
  # Verificar se é JSON válido
  if jq . "$OUTPUT" >/dev/null 2>&1; then
    echo "✅ JSON válido"
    
    # Mostrar estrutura
    echo ""
    echo "📊 ESTRUTURA DOS DADOS:"
    jq 'keys' "$OUTPUT" 2>/dev/null || echo "❌ Erro ao ler estrutura"
    
    # Verificar se há logs
    LOG_COUNT=$(jq '.events | length' "$OUTPUT" 2>/dev/null || echo "0")
    echo "📈 Logs encontrados: $LOG_COUNT"
    
    if [ "$LOG_COUNT" -gt 0 ]; then
      echo ""
      echo "🎯 PRIMEIROS LOGS:"
      jq -r '.events[0:3][] | "\(.timestamp // "N/A"): \(.message // .type // "unknown")"' "$OUTPUT" | head -5
    fi
    
  else
    echo "❌ Resposta não é JSON válido:"
    cat "$OUTPUT"
  fi
else
  echo "❌ Arquivo vazio ou erro na requisição"
fi

echo ""
echo "🏁 Script concluído!"
