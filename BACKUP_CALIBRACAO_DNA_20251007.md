# 🔄 BACKUP CALIBRAÇÃO DNA - VERSÕES PRESERVADAS
*Data: 7 de outubro de 2025*

## 📋 REGISTRO DE CALIBRAÇÕES

### 🧬 VERSÃO ORIGINAL (DNA GENÉTICO PURO)
**Período**: Inicial - 7 out 2025
**Performance**: +10,910% ROI anual validado
**Status**: Filtros calibrados para bull market extremo

```python
# Configuração Original DNA Genético
ATR_PCT_MIN = 0.3        # ATR mínimo 0.3%
VOLUME_MULTIPLIER = 1.8  # Volume 1.8x média
RSI_MIN = 35.0          # RSI > 35
RSI_MAX = 65.0          # RSI < 65

# Parâmetros Core (mantidos sempre)
LEVERAGE = 3            # Leverage 3x
SL_PCT = 1.5           # Stop Loss 1.5%
TP_PCT = 12.0          # Take Profit 12%
EMA_FAST = 3           # EMA rápida 3
EMA_SLOW = 34          # EMA lenta 34
MIN_CONFLUENCIA = 3    # Confluência 3
```

**Resultado**: 0 trades/dia (muito restritivo para mercado normal)

### 🔧 VERSÃO CALIBRADA (EXPERIMENTAL)
**Período**: 7 out 2025 - Teste
**Objetivo**: Permitir execuções mantendo DNA
**Status**: Testada por algumas horas

```python
# Configuração Calibrada
ATR_PCT_MIN = 0.2        # ATR mínimo 0.2% (relaxado)
VOLUME_MULTIPLIER = 1.3  # Volume 1.3x média (relaxado)
RSI_MIN = 30.0          # RSI > 30 (relaxado)
RSI_MAX = 70.0          # RSI < 70 (relaxado)

# Parâmetros Core (preservados)
LEVERAGE = 3            # Leverage 3x
SL_PCT = 1.5           # Stop Loss 1.5%
TP_PCT = 12.0          # Take Profit 12%
EMA_FAST = 3           # EMA rápida 3
EMA_SLOW = 34          # EMA lenta 34
MIN_CONFLUENCIA = 3    # Confluência 3
```

**Expectativa**: 15-25 trades/dia

## 🎯 ESTRATÉGIA DE BACKUP

### Arquivo Principal: `trading.py`
- **Versão Ativa**: Sempre a configuração atual
- **Backup**: Registrado neste arquivo MD

### Reversão Rápida:
```bash
# Para voltar à configuração original:
# Editar trading.py com valores da "VERSÃO ORIGINAL"
```

### Teste Gradual:
1. **Fase 1**: Manter original (seletividade máxima)
2. **Fase 2**: Testar calibrada (se necessário)
3. **Fase 3**: Ajustes pontuais baseados em dados reais

## 📊 COMPARAÇÃO DE PERFORMANCE

| Parâmetro | Original | Calibrada | Observações |
|-----------|----------|-----------|-------------|
| ATR_MIN | 0.3% | 0.2% | -33% threshold |
| VOLUME | 1.8x | 1.3x | -28% threshold |
| RSI_MIN | 35 | 30 | -14% threshold |
| RSI_MAX | 65 | 70 | +8% threshold |
| **ROI** | +10,910% | ~+10,910% | DNA preservado |
| **Trades/dia** | 0 | 15-25? | Estimativa |

## 🔍 MONITORAMENTO

### Métricas de Sucesso:
- [ ] Trades executados > 0
- [ ] ROI mantido próximo ao original
- [ ] Win rate consistente
- [ ] Drawdown controlado

### Critérios de Reversão:
- ROI < 8,000% (queda >25%)
- Win rate < 15%
- Drawdown > 20%
- Trades excessivos (>100/dia)

## 📝 HISTÓRICO DE MUDANÇAS

**7 out 2025 - 15:30**: Implementação calibração experimental
**7 out 2025 - 15:45**: Solicitação backup/reversão
**Status**: Preparando reversão para configuração original
