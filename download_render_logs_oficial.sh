#!/usr/bin/env bash
# Baixar logs do Render via API (histórico + paginação)
# Requisitos: curl, jq

# === CONFIG ===
RENDER_TOKEN="rnd_EGrZuK7lLtIrQtWmgv21Mi2OsbmR"              # API Key do Render
SERVICE_ID="srv-d31n3pumcj7s738sddi0"                         # ID direto do serviço
START_UTC="2025-10-05T00:00:00Z"                              # início (UTC) - últimas 48h
END_UTC="2025-10-06T23:59:59Z"                                # fim (UTC) - até agora
OUT="logs_trading_genetic_$(date +%Y%m%d_%H%M%S).jsonl"

set -euo pipefail

authH() { echo "Authorization: Bearer ${RENDER_TOKEN}"; }

echo "🚀 DOWNLOAD OFICIAL DE LOGS RENDER"
echo "=================================="
echo "Service ID: ${SERVICE_ID}"
echo "Período: ${START_UTC} → ${END_UTC}"
echo "Output: ${OUT}"
echo ""

echo ">> Testando autenticação..."
AUTH_TEST=$(curl -fsS "https://api.render.com/v1/services/${SERVICE_ID}" -H "$(authH)" | jq -r '.id // "ERRO"')

if [ "${AUTH_TEST}" != "${SERVICE_ID}" ]; then
  echo "❌ ERRO: Autenticação falhou ou serviço não encontrado"
  echo "   Resposta: ${AUTH_TEST}"
  exit 1
fi

echo "✅ Autenticação OK - Serviço encontrado"
echo ""

echo ">> Baixando logs de ${START_UTC} até ${END_UTC}..."
> "${OUT}"

NEXT_START="${START_UTC}"
NEXT_END="${END_UTC}"
PAGE=1
TOTAL_LOGS=0

while : ; do
  echo "📥 Página ${PAGE} | janela: ${NEXT_START} → ${NEXT_END}"
  
  # Fazer requisição para logs
  RESP=$(curl -fsS -G "https://api.render.com/v1/logs" -H "$(authH)" \
    --data-urlencode "startTime=${NEXT_START}" \
    --data-urlencode "endTime=${NEXT_END}" \
    --data-urlencode "resourceIds[]=${SERVICE_ID}" 2>/dev/null)

  # Verificar se a resposta é válida
  if ! echo "${RESP}" | jq . >/dev/null 2>&1; then
    echo "❌ Resposta inválida da API:"
    echo "${RESP}"
    break
  fi

  # Contar logs nesta página
  PAGE_LOGS=$(echo "${RESP}" | jq '.events | length' 2>/dev/null || echo "0")
  echo "   📊 ${PAGE_LOGS} logs nesta página"

  # Anexa eventos como JSON Lines
  if [ "${PAGE_LOGS}" -gt 0 ]; then
    echo "${RESP}" | jq -c '.events[]' >> "${OUT}"
    TOTAL_LOGS=$((TOTAL_LOGS + PAGE_LOGS))
  fi

  # Verificar se há mais páginas
  HAS_MORE=$(echo "${RESP}" | jq -r '.hasMore // false')
  echo "   🔄 Mais páginas: ${HAS_MORE}"
  
  if [ "${HAS_MORE}" != "true" ]; then
    echo "   📄 Última página alcançada"
    break
  fi

  # Para a próxima página, use os ponteiros de paginação
  NEXT_START=$(echo "${RESP}" | jq -r '.nextStartTime // empty')
  NEXT_END=$(echo "${RESP}" | jq -r '.nextEndTime // "'${END_UTC}'"')

  if [ -z "${NEXT_START}" ] || [ "${NEXT_START}" = "null" ]; then
    echo "   ⚠️ nextStartTime não disponível"
    break
  fi
  
  echo "   ➡️ Próxima janela: ${NEXT_START} → ${NEXT_END}"
  PAGE=$((PAGE+1))
  
  # Rate limiting
  sleep 0.5
done

echo ""
echo "✅ DOWNLOAD CONCLUÍDO!"
echo "====================="
echo "📁 Arquivo: ${OUT}"
echo "📊 Total de logs: ${TOTAL_LOGS}"
echo "📄 Páginas processadas: ${PAGE}"
echo ""

if [ "${TOTAL_LOGS}" -gt 0 ]; then
  echo "🔍 ANÁLISE RÁPIDA DOS LOGS:"
  echo "--------------------------"
  
  # Mostrar tipos de log encontrados
  echo "📈 Tipos de evento:"
  cat "${OUT}" | jq -r '.type' | sort | uniq -c | sort -nr | head -10
  
  echo ""
  echo "🕐 Primeiros 3 logs (cronológico):"
  cat "${OUT}" | head -3 | jq -r '"\(.timestamp // "N/A"): \(.message // .type // "unknown")"' | cut -c1-100
  
  echo ""
  echo "🕕 Últimos 3 logs (cronológico):"
  cat "${OUT}" | tail -3 | jq -r '"\(.timestamp // "N/A"): \(.message // .type // "unknown")"' | cut -c1-100
  
  echo ""
  echo "💡 COMANDOS ÚTEIS PARA ANÁLISE:"
  echo "==============================="
  echo "# Ver todos os logs com timestamp:"
  echo "cat ${OUT} | jq -r '\"\(.timestamp // \"N/A\"): \(.message // .type // \"unknown\")\"'"
  echo ""
  echo "# Filtrar apenas mensagens de aplicação:"
  echo "cat ${OUT} | jq -r 'select(.message) | \"\(.timestamp): \(.message)\"'"
  echo ""
  echo "# Buscar por palavras-chave específicas:"
  echo "cat ${OUT} | jq -r 'select(.message | contains(\"trade\") or contains(\"DNA\") or contains(\"error\")) | \"\(.timestamp): \(.message)\"'"
  echo ""
  echo "# Contar logs por nível:"
  echo "cat ${OUT} | jq -r '.level // \"unknown\"' | sort | uniq -c"
  
else
  echo "⚠️ NENHUM LOG ENCONTRADO"
  echo "========================"
  echo "Possíveis causas:"
  echo "1. 🕐 Período muito antigo (retenção expirou)"
  echo "2. 📝 App não está gerando logs em stdout/stderr"
  echo "3. 🚫 Background workers podem ter logs limitados"
  echo "4. ⏰ Fuso horário incorreto (use UTC)"
  echo ""
  echo "💡 Tente:"
  echo "- Reduzir o período para últimas 6-12h"
  echo "- Verificar se o app está rodando"
  echo "- Adicionar prints/logs no código Python"
fi

echo ""
echo "🎯 Script concluído!"
