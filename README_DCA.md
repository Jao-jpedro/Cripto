# Sistema de Trading DCA (Dollar Cost Averaging) - SOL Long Only

Sistema automatizado de trading com estratégia de compra e venda em degraus baseados em porcentagens.

## 🎯 Estratégia

### Conceito
- **Asset:** SOL/USDC:USDC com **5x leverage**
- **Operações:** Apenas LONG (compra)
- **Dados:** Gráficos de 1 dia, últimos 30 dias da Binance
- **Execução:** Hyperliquid (subconta configurada)

### 📊 Degraus de COMPRA

Baseado no **% abaixo do preço máximo dos últimos 30 dias**:

| % Abaixo do Máximo | % do Capital a Investir | Exemplo (saldo $100) |
|-------------------|------------------------|---------------------|
| -10% | 15% | Investe $15 |
| -20% | 30% | Investe $30 |
| -30% | 50% | Investe $50 |

**Cooldown:** 5 dias entre compras **OU** avanço de degrau

**Lógica do Cooldown:**
- Se comprou em -10% e em menos de 5 dias o preço cai para -20%, pode comprar novamente (avanço de degrau)
- Se comprou em -10% e preço fica oscilando, só pode comprar novamente após 5 dias

### 💰 Degraus de VENDA

Baseado no **% de ganho da posição aberta**:

| % de Ganho | % da Posição a Vender | Exemplo (posição 10 SOL) |
|-----------|----------------------|-------------------------|
| +10% | 20% | Vende 2 SOL |
| +20% | 20% | Vende 2 SOL |
| +30% | 20% | Vende 2 SOL |
| +40% | 20% | Vende 2 SOL |
| +50% | 20% | Vende 2 SOL |

**Cooldown:** 3 dias entre vendas **OU** avanço de degrau

**Lógica do Cooldown:**
- Se vendeu em +10% e em menos de 3 dias atinge +20%, pode vender novamente (avanço de degrau)
- Não pode vender no mesmo degrau ou inferior dentro do cooldown (ex: vendeu +20%, não pode vender +10% nos próximos 3 dias)

## 🔧 Configuração

### 1. Instalar dependências

```bash
pip install ccxt pandas numpy python-dotenv requests
```

### 2. Configurar variáveis de ambiente

Copie o arquivo `.env.dca.example` para `.env`:

```bash
cp .env.dca.example .env
```

Edite o `.env` com suas credenciais:

```env
# Hyperliquid - Execução
WALLET_ADDRESS=0xYourWalletAddress
PRIVATE_KEY=0xYourPrivateKey
VAULT_ADDRESS=0xYourVaultAddress  # Subconta (obrigatório)

# Binance - Dados históricos
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Discord - Notificações (opcional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### 3. Executar

```bash
python trading_dca.py
```

## 📁 Arquivos

- `trading_dca.py` - Sistema principal
- `dca_state.json` - Estado persistente (criado automaticamente)
- `trading_dca_YYYYMMDD_HHMMSS.log` - Log de execução

## 🧠 Lógica de Funcionamento

### Ciclo Principal (a cada 1 hora)

1. **Análise de Mercado**
   - Busca dados históricos de 30 dias (1d) da Binance
   - Calcula o preço máximo dos últimos 30 dias
   - Verifica preço atual
   - Calcula % abaixo do máximo

2. **Verificação de Sinais de Compra**
   - Verifica se o preço atual está X% abaixo do máximo
   - Verifica se está dentro do cooldown ou avançou de degrau
   - Se condições OK → Executa compra

3. **Verificação de Sinais de Venda**
   - Calcula % de ganho da posição (baseado no preço médio de entrada)
   - Verifica se atingiu algum degrau de venda
   - Verifica cooldown de venda
   - Se condições OK → Executa venda parcial

4. **Persistência**
   - Salva timestamp e degrau de cada operação
   - Mantém histórico de entradas (preço e quantidade)
   - Calcula preço médio de entrada

### Exemplo de Funcionamento

**Cenário:**
- Saldo inicial: $100
- Preço máximo 30d: $200
- Preço atual: $180 (-10% do máximo)

**Ação:**
- ✅ Compra com 15% do saldo = $15
- Com 5x leverage = $75 de posição
- Registra entrada: preço $180

**Após 2 dias:**
- Preço cai para $160 (-20% do máximo)
- ✅ Pode comprar novamente (avanço de degrau)
- Compra com 30% do saldo restante

**Após 5 dias:**
- Preço sobe para $198 (+10% de ganho médio)
- ✅ Vende 20% da posição
- Registra venda no degrau +10%

**Após 2 dias:**
- Preço sobe para $218 (+21% de ganho)
- ✅ Pode vender novamente (avanço de degrau +20%)
- Vende mais 20% da posição

## ⚙️ Personalização

Edite a classe `DCAConfig` em `trading_dca.py`:

```python
@dataclass
class DCAConfig:
    # Asset
    SYMBOL: str = "SOL/USDC:USDC"
    LEVERAGE: int = 5
    
    # Dados históricos
    HISTORICAL_DAYS: int = 30
    TIMEFRAME: str = "1d"
    
    # Degraus de COMPRA (% abaixo máximo, % capital)
    BUY_STEPS: List[tuple] = [
        (10, 15),  # -10% → 15% do capital
        (20, 30),  # -20% → 30% do capital
        (30, 50),  # -30% → 50% do capital
    ]
    
    # Degraus de VENDA (% ganho, % posição)
    SELL_STEPS: List[tuple] = [
        (10, 20),  # +10% → vende 20%
        (20, 20),  # +20% → vende 20%
        (30, 20),  # +30% → vende 20%
    ]
    
    # Cooldowns
    BUY_COOLDOWN_DAYS: int = 5
    SELL_COOLDOWN_DAYS: int = 3
```

## 📊 Logs e Monitoramento

### Logs no Terminal

```
[2025-11-04 10:00:00] [INFO] 🔄 INICIANDO CICLO DCA
[2025-11-04 10:00:01] [INFO] 📊 Análise: Preço=$180.50 | Max 30d=$200.00 | Abaixo do max=9.75%
[2025-11-04 10:00:02] [INFO] 🚨 SINAL DE COMPRA: Degrau 0 ativado (9.75% >= 10%)
[2025-11-04 10:00:03] [INFO] 🟢 COMPRANDO: Degrau 0 | 15% do saldo ($15.00) | Leverage 5x
[2025-11-04 10:00:05] [INFO] ✅ Ordem criada
```

### Notificações Discord

Recebe notificações automáticas de:
- ✅ Compras executadas
- ✅ Vendas executadas
- ❌ Erros críticos

## 🛡️ Segurança

- **Cooldowns:** Evita overtrading
- **Degraus progressivos:** Compra mais quando preço cai mais
- **Venda escalonada:** Realiza lucros progressivamente
- **Estado persistente:** Não perde histórico em caso de reinicialização
- **Logs detalhados:** Auditoria completa de operações

## 🚨 Importante

- ⚠️ **Use por sua conta e risco**
- ⚠️ **Teste com valores pequenos primeiro**
- ⚠️ **Mantenha backup do arquivo `dca_state.json`**
- ⚠️ **Monitore regularmente os logs**
- ⚠️ **A estratégia assume mercado em tendência de alta no longo prazo**

## 📈 Vantagens da Estratégia

✅ **DCA Inteligente:** Compra mais quando preço está mais baixo  
✅ **Realização de Lucros:** Vende progressivamente quando sobe  
✅ **Proteção contra Overtrading:** Cooldowns obrigatórios  
✅ **Flexibilidade:** Permite avanço de degrau  
✅ **Long Only:** Focado em acumulação de longo prazo  
✅ **Leverage Controlado:** 5x para aumentar exposição com capital moderado  

## 🔄 Intervalo de Verificação

- **Padrão:** 1 hora (3600 segundos)
- **Recomendado para timeframe 1d:** 1-4 horas
- **Para ajustar:** Modifique `check_interval` na função `main()`

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `trading_dca_*.log`
2. Verifique o estado em `dca_state.json`
3. Teste conexões com Binance e Hyperliquid separadamente
