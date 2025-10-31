PROMPT PARA CRIAR SISTEMA DE TRADING AUTOMATIZADO BASEADO NO TRADINGV4

CONTEXTO E FONTE DAS INFORMACOES

Este prompt baseia-se no sistema TradingV4, um algoritmo de trading automatizado real desenvolvido para operar no mercado de criptomoedas com alta frequencia e precisao. O sistema utiliza dados de mercado da Binance para analise tecnica e executa operacoes na exchange Hyperliquid. O TradingV4 implementa uma estrategia de EMA Gradient com filtros rigorosos de volatilidade e volume, operando 24 horas por dia com controles avancados de risco.

OBJETIVO DO SISTEMA

Criar um sistema de trading automatizado que identifique oportunidades de entrada baseadas em convergencia de multiplos indicadores tecnicos, executando operacoes apenas quando todas as condicoes de entrada forem simultaneamente atendidas. O sistema deve ser conservador, seletivo e focado em qualidade sobre quantidade de operacoes.

ATIVOS SUPORTADOS E CONFIGURACOES DE ALAVANCAGEM

O sistema opera exclusivamente com contratos futuros perpetuos em USDC na Hyperliquid, utilizando as seguintes configuracoes por ativo:

AVNT-USD: simbolo de dados AVNTUSDT, simbolo de trading AVNT/USDC:USDC, alavancagem 5x
ASTER-USD: simbolo de dados ASTERUSDT, simbolo de trading ASTER/USDC:USDC, alavancagem 5x  
ETH-USD: simbolo de dados ETHUSDT, simbolo de trading ETH/USDC:USDC, alavancagem 25x
TAO-USD: simbolo de dados TAOUSDT, simbolo de trading TAO/USDC:USDC, alavancagem 5x
XRP-USD: simbolo de dados XRPUSDT, simbolo de trading XRP/USDC:USDC, alavancagem 20x
DOGE-USD: simbolo de dados DOGEUSDT, simbolo de trading DOGE/USDC:USDC, alavancagem 10x
AVAX-USD: simbolo de dados AVAXUSDT, simbolo de trading AVAX/USDC:USDC, alavancagem 10x
ENA-USD: simbolo de dados ENAUSDT, simbolo de trading ENA/USDC:USDC, alavancagem 10x
BNB-USD: simbolo de dados BNBUSDT, simbolo de trading BNB/USDC:USDC, alavancagem 10x
SUI-USD: simbolo de dados SUIUSDT, simbolo de trading SUI/USDC:USDC, alavancagem 10x
ADA-USD: simbolo de dados ADAUSDT, simbolo de trading ADA/USDC:USDC, alavancagem 10x
PUMP-USD: simbolo de dados PUMPUSDT, simbolo de trading PUMP/USDC:USDC, alavancagem 10x
LINK-USD: simbolo de dados LINKUSDT, simbolo de trading LINK/USDC:USDC, alavancagem 10x
WLD-USD: simbolo de dados WLDUSDT, simbolo de trading WLD/USDC:USDC, alavancagem 10x
AAVE-USD: simbolo de dados AAVEUSDT, simbolo de trading AAVE/USDC:USDC, alavancagem 10x
CRV-USD: simbolo de dados CRVUSDT, simbolo de trading CRV/USDC:USDC, alavancagem 10x
LTC-USD: simbolo de dados LTCUSDT, simbolo de trading LTC/USDC:USDC, alavancagem 10x
NEAR-USD: simbolo de dados NEARUSDT, simbolo de trading NEAR/USDC:USDC, alavancagem 10x

CONFIGURACAO TEMPORAL E DADOS

O sistema opera em timeframe de 15 minutos utilizando 260 candles historicos para calculo dos indicadores. Os dados sao obtidos da API da Binance atraves do endpoint de klines historicos, com fallback para Bybit em caso de indisponibilidade. O sistema processa dados em tempo real, analisando o candle atual nao fechado junto com o historico para identificacao de oportunidades.

PARAMETROS FUNDAMENTAIS DA ESTRATEGIA

EMA_SHORT_SPAN: 7 periodos para media movel exponencial rapida
EMA_LONG_SPAN: 21 periodos para media movel exponencial lenta  
N_BARRAS_GRADIENTE: 3 periodos para calculo do gradiente
GRAD_CONSISTENCY: 3 barras consecutivas de gradiente consistente
ATR_PERIOD: 14 periodos para Average True Range
VOL_MA_PERIOD: 20 periodos para media movel de volume
ATR_PCT_MIN: 0.7% volatilidade minima aceitavel
ATR_PCT_MAX: 5.0% volatilidade maxima aceitavel  
BREAKOUT_K_ATR: 3.0 multiplicador do ATR para deteccao de breakout
NO_TRADE_EPS_K_ATR: 0.07 multiplicador para zona neutra
COOLDOWN_MINUTOS: 120 minutos de pausa entre operacoes
ANTI_SPAM_SECS: 3 segundos entre verificacoes
MIN_HOLD_BARS: 1 barra minima de manutencao de posicao

CALCULO DETALHADO DOS INDICADORES TECNICOS

MEDIA MOVEL EXPONENCIAL EMA7 E EMA21

A EMA de 7 periodos e calculada aplicando a formula EMA = preco_atual * fator_suavizacao + EMA_anterior * um_menos_fator_suavizacao, onde fator_suavizacao = 2 dividido por periodo_mais_um. Para EMA7 o fator e 2 dividido por 8 igual a 0.25. Para EMA21 o fator e 2 dividido por 22 igual a 0.090909. As EMAs sao calculadas sobre os precos de fechamento dos candles de 15 minutos.

AVERAGE TRUE RANGE ATR E PERCENTUAL

O ATR e calculado usando a media movel simples de 14 periodos dos True Range valores. True Range e o maior valor entre: preco_maximo_menos_preco_minimo, valor_absoluto_de_preco_maximo_menos_fechamento_anterior, valor_absoluto_de_preco_minimo_menos_fechamento_anterior. O ATR percentual e calculado como ATR dividido por preco_de_fechamento multiplicado por 100.

MEDIA MOVEL DE VOLUME

Media movel simples de 20 periodos aplicada sobre os volumes de cada candle. Volume representa a quantidade total negociada no periodo de 15 minutos.

GRADIENTE DA EMA7

Calculado como a variacao percentual da EMA7 em relacao ao valor de 3 periodos atras. Formula: EMA7_atual menos EMA7_tres_periodos_atras dividido por EMA7_tres_periodos_atras multiplicado por 100.

CRITERIOS DE ENTRADA PARA POSICAO COMPRADA LONG

CONDICAO UM - TENDENCIA DE ALTA: EMA7 deve estar acima da EMA21. Esta condicao confirma que a tendencia de curto prazo esta acima da tendencia de medio prazo, indicando momentum de alta.

CONDICAO DOIS - GRADIENTE CONSISTENTE: As ultimas 3 barras devem apresentar gradiente positivo da EMA7. Isto significa que em cada uma das 3 ultimas barras, o gradiente percentual da EMA7 deve ser maior que zero, confirmando aceleracao consistente da tendencia de alta.

CONDICAO TRES - VOLATILIDADE SAUDAVEL: O ATR percentual deve estar entre 0.7% e 5.0%. Valores abaixo de 0.7% indicam mercado muito quieto com potencial limitado. Valores acima de 5.0% indicam volatilidade excessiva com risco elevado.

CONDICAO QUATRO - BREAKOUT CONFIRMADO: O preco de fechamento atual deve estar acima da EMA7 somada a 3.0 vezes o ATR. Esta condicao confirma um rompimento significativo acima da tendencia de curto prazo, sugerindo continuacao do movimento.

CONDICAO CINCO - VOLUME ELEVADO: O volume atual deve ser superior a media movel de volume de 20 periodos. Esta condicao confirma que o movimento tem participacao significativa do mercado.

CRITERIOS DE ENTRADA PARA POSICAO VENDIDA SHORT

CONDICAO UM - TENDENCIA DE BAIXA: EMA7 deve estar abaixo da EMA21. Esta condicao confirma que a tendencia de curto prazo esta abaixo da tendencia de medio prazo, indicando momentum de baixa.

CONDICAO DOIS - GRADIENTE CONSISTENTE: As ultimas 3 barras devem apresentar gradiente negativo da EMA7. Isto significa que em cada uma das 3 ultimas barras, o gradiente percentual da EMA7 deve ser menor que zero, confirmando aceleracao consistente da tendencia de baixa.

CONDICAO TRES - VOLATILIDADE SAUDAVEL: O ATR percentual deve estar entre 0.7% e 5.0%. Mesma logica da posicao comprada para garantir volatilidade adequada.

CONDICAO QUATRO - BREAKOUT CONFIRMADO: O preco de fechamento atual deve estar abaixo da EMA7 subtraida de 3.0 vezes o ATR. Esta condicao confirma um rompimento significativo abaixo da tendencia de curto prazo.

CONDICAO CINCO - VOLUME ELEVADO: O volume atual deve ser superior a media movel de volume de 20 periodos. Confirma participacao do mercado no movimento de baixa.

ZONA NEUTRA E FILTRO ANTI-RUIDO

Quando a diferenca absoluta entre EMA7 e EMA21 for menor que 0.07 vezes o ATR, o sistema considera estar em zona neutra e nao executa operacoes. Este filtro evita entradas durante periodos de consolidacao onde as EMAs estao muito proximas.

SISTEMA DE COOLDOWN E ANTI-SPAM

Apos o fechamento de qualquer posicao, o sistema entra em periodo de cooldown de 120 minutos antes de considerar novas entradas. Durante este periodo, nenhuma operacao sera executada independentemente das condicoes de mercado. Adicionalmente, o sistema possui filtro anti-spam que impede verificacoes de entrada mais frequentes que a cada 3 segundos.

DETECCAO DE CONSISTENCIA DO GRADIENTE

O sistema verifica se as ultimas 3 barras apresentam gradiente na mesma direcao. Para posicao comprada, todas as 3 barras devem ter gradiente positivo. Para posicao vendida, todas as 3 barras devem ter gradiente negativo. Esta verificacao garante que a aceleracao da tendencia seja sustentada e nao apenas um movimento isolado.

CALCULO DO MULTIPLICADOR DE BREAKOUT

O valor atual de 3.0 vezes o ATR foi otimizado atraves de backtesting extensivo para maximizar a relacao risco-retorno. Este multiplicador garante que apenas movimentos significativos sejam considerados como breakouts validos, filtrando ruidos de mercado e movimentos menores.

METRICAS DE VOLUME EM TEMPO REAL

O sistema calcula em tempo real: volume atual do candle, media de volume dos ultimos 30 candles, e a relacao entre volume atual e media. Esta informacao e utilizada tanto para validacao de entradas quanto para logging detalhado das condicoes de mercado.

SISTEMA DE LOGGING DE TRIGGERS

A cada verificacao de entrada, o sistema registra snapshot completo incluindo: preco de fechamento, valores das EMAs, ATR absoluto e percentual, volume atual e media, gradiente da EMA7, multiplicador de breakout atual, quantidade de trades e relacao com media de 30 periodos. Este logging permite analise posterior das condicoes que geraram ou nao geraram entradas.

CRITERIO DE SAIDA DAS POSICOES

O sistema utiliza criterio de saida baseado em inversao do gradiente. Se uma posicao comprada estiver aberta e 2 barras consecutivas apresentarem gradiente negativo da EMA7, a posicao e fechada. O mesmo se aplica para posicoes vendidas com 2 barras de gradiente positivo consecutivas.

ESTRUTURA DE MULTIPLAS CARTEIRAS

O sistema suporta operacao com duas carteiras: carteira principal com valor padrao de 3 dolares por operacao e carteira secundaria com 10 dolares por operacao. Cada carteira possui configuracao independente de chaves privadas e enderecos de wallet.

SISTEMA DE APRENDIZADO INTEGRADO

O sistema coleta metricas de cada entrada incluindo condicoes de mercado, indicadores tecnicos, sessao de trading, regime de volatilidade e outras caracteristicas. Estas informacoes sao armazenadas em banco SQLite para analise posterior de padroes e probabilidades de sucesso.

TOLERANCIA A FALHAS E RECONEXAO

O sistema implementa multiplos fallbacks para obtencao de dados, reconexao automatica em caso de falhas de rede, cache de dados para evitar requisicoes excessivas, e logging detalhado de todos os erros para debugging.

EXECUCAO EM TEMPO REAL

O sistema foi projetado para operar continuamente, processando novos dados a cada atualizacao de preco e verificando condicoes de entrada em tempo real. A latencia entre identificacao de oportunidade e execucao de ordem e minimizada atraves de codigo otimizado e conexoes diretas com as APIs.

VALIDACAO DE CONDICOES SIMULTANEAS

CRITICO: Todas as cinco condicoes de entrada devem ser verdadeiras simultaneamente para uma entrada ser executada. Nao existe logica de pelo menos X condicoes ou ponderacao entre condicoes. E uma operacao logica AND entre todas as condicoes.

TRATAMENTO DE POSICOES PREEXISTENTES

Antes de verificar condicoes de entrada, o sistema sempre verifica se ja existe posicao aberta. Se existir posicao, aplica logica de saida. Apenas quando nao ha posicao aberta e que as condicoes de entrada sao verificadas.

INTEGRACAO COM SISTEMA DE NOTIFICACOES

Todas as operacoes executadas sao notificadas em tempo real via webhook do Discord, incluindo tipo de operacao, ativo, direcao, preco de execucao e saldos atualizados das carteiras.
