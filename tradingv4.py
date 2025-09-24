# trading.py
# ------------------------------------------------------------
# Bot genérico com:
# - SL=5% e TP=10%
# - Inversão de sinais (LONG<->SHORT) mantendo a mesma lógica
# - Apenas carteira HL:0x5ff0f14d577166f9ede3d9568a423166be61ea9d
# - Usa HYPERLIQUID_PRIVATE_KEY do ambiente (Render)
# ------------------------------------------------------------

import os
import time
import math
from dataclasses import dataclass
from typing import Optional, Literal, Dict

# =============================
# Configurações principais
# =============================

TAKE_PROFIT_PCT = 0.10   # 10%
STOP_LOSS_PCT   = 0.05   # 5%

# Carteira única solicitada
HL_WALLET = "0x5ff0f14d577166f9ede3d9568a423166be61ea9d"

# Chave privada via variável de ambiente (Render)
HL_PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if not HL_PRIVATE_KEY:
    raise RuntimeError(
        "Variável de ambiente HYPERLIQUID_PRIVATE_KEY não encontrada. "
        "No Render, adicione-a em Environment > Environment Variables."
    )

# =============================
# Estruturas de dados
# =============================

Side = Literal["LONG", "SHORT"]

@dataclass
class Position:
    symbol: str
    side: Optional[Side] = None
    qty: float = 0.0
    entry_px: float = 0.0

    def is_open(self) -> bool:
        return self.side is not None and self.qty > 0

# =============================
# Camada de exchange (stub)
# Integre aqui seu SDK/HTTP real da Hyperliquid.
# =============================

class HyperliquidClient:
    def __init__(self, wallet: str, private_key: str):
        self.wallet = wallet
        self.private_key = private_key
        # TODO: inicializar SDK real aqui

    # ---- Mercado ----
    def get_price(self, symbol: str) -> float:
        # TODO: implemente com API real
        # Valor mock para fluxo local
        return 100.0

    # ---- Ordens ----
    def market_open(self, symbol: str, side: Side, qty: float) -> Dict:
        # TODO: implemente ordem a mercado na Hyperliquid
        print(f"[ORDER] OPEN {side} {qty} {symbol}")
        return {"status": "ok"}

    def market_close(self, symbol: str, side: Side, qty: float) -> Dict:
        # Fecha na direção oposta
        print(f"[ORDER] CLOSE {side} {qty} {symbol}")
        return {"status": "ok"}

# =============================
# Lógica de sinais (exemplo)
# ATENÇÃO: aqui fazemos a INVERSÃO pedida (LONG<->SHORT)
# Mantemos o mesmo raciocínio de entrada, mas trocamos o lado.
# =============================

def original_long_signal(df) -> bool:
    """
    Coloque aqui sua condição ORIGINAL de LONG (antes da inversão).
    Ex.: EMA curta > EMA longa, rompimento, etc.
    """
    # Placeholder: retorne False por padrão
    return False

def original_short_signal(df) -> bool:
    """
    Coloque aqui sua condição ORIGINAL de SHORT (antes da inversão).
    """
    # Placeholder: retorne False por padrão
    return False

def inverted_signals(df) -> Dict[str, bool]:
    """
    Mantém o MESMO raciocínio, mas inverte o lado:
      - Onde era LONG vira SHORT
      - Onde era SHORT vira LONG
    """
    was_long  = original_long_signal(df)   # lógica original de LONG
    was_short = original_short_signal(df)  # lógica original de SHORT

    # Inversão solicitada:
    long_now  = was_short  # agora entraria LONG onde antes era SHORT
    short_now = was_long   # agora entraria SHORT onde antes era LONG

    return {"long": long_now, "short": short_now}

# =============================
# Gestão de posição com SL/TP percentuais
# =============================

def compute_tp_sl(entry_px: float, side: Side) -> Dict[str, float]:
    """
    Define níveis absolutos de TP/SL a partir de percentuais.
    TP = 10%, SL = 5% (fixos conforme solicitado).
    """
    if side == "LONG":
        tp = entry_px * (1 + TAKE_PROFIT_PCT)
        sl = entry_px * (1 - STOP_LOSS_PCT)
    else:  # SHORT
        tp = entry_px * (1 - TAKE_PROFIT_PCT)
        sl = entry_px * (1 + STOP_LOSS_PCT)
    return {"tp": tp, "sl": sl}

def should_exit(price: float, side: Side, tp: float, sl: float) -> Optional[str]:
    """
    Retorna 'TP' ou 'SL' quando um dos níveis for atingido; caso contrário None.
    """
    if side == "LONG":
        if price >= tp:
            return "TP"
        if price <= sl:
            return "SL"
    else:  # SHORT
        if price <= tp:
            return "TP"
        if price >= sl:
            return "SL"
    return None

# =============================
# Estratégia (loop simplificado)
# =============================

class Strategy:
    def __init__(self, client: HyperliquidClient, symbol: str, qty: float):
        self.client = client
        self.symbol = symbol
        self.qty = qty
        self.pos = Position(symbol=symbol)

        # Armazena níveis de TP/SL atuais quando em posição
        self._tp: Optional[float] = None
        self._sl: Optional[float] = None

    def on_bar(self, df) -> None:
        """
        Chame este método a cada novo candle/barra com seu dataframe/estrutura 'df'.
        """
        price = self.client.get_price(self.symbol)

        # Se há posição aberta, verificar TP/SL
        if self.pos.is_open():
            exit_reason = should_exit(price, self.pos.side, self._tp, self._sl)
            if exit_reason:
                self.client.market_close(self.symbol, self.pos.side, self.pos.qty)
                print(f"[EXIT] {exit_reason} | side={self.pos.side} px={price:.4f} tp={self._tp:.4f} sl={self._sl:.4f}")
                self.pos = Position(symbol=self.symbol)  # zera
                self._tp = self._sl = None
            return

        # Sem posição: avaliar ENTRADA (com inversão aplicada)
        sig = inverted_signals(df)

        if sig["long"]:
            # Abrir LONG (note: isso era o SHORT original)
            self.client.market_open(self.symbol, "LONG", self.qty)
            self.pos = Position(symbol=self.symbol, side="LONG", qty=self.qty, entry_px=price)
            levels = compute_tp_sl(price, "LONG")
            self._tp, self._sl = levels["tp"], levels["sl"]
            print(f"[ENTER] LONG @ {price:.4f} | TP={self._tp:.4f} (+10%) SL={self._sl:.4f} (-5%)")
            return

        if sig["short"]:
            # Abrir SHORT (note: isso era o LONG original)
            self.client.market_open(self.symbol, "SHORT", self.qty)
            self.pos = Position(symbol=self.symbol, side="SHORT", qty=self.qty, entry_px=price)
            levels = compute_tp_sl(price, "SHORT")
            self._tp, self._sl = levels["tp"], levels["sl"]
            print(f"[ENTER] SHORT @ {price:.4f} | TP={self._tp:.4f} (+10%) SL={self._sl:.4f} (-5%)")
            return

        # Sem sinal
        print("[FLAT] Sem entrada nesta barra.")

# =============================
# Bootstrap
# =============================

def main():
    symbol = "WLD-USD"   # ajuste seu símbolo aqui (ex.: "SOL-USD", etc.)
    qty    = 1.0         # tamanho do lote

    client = HyperliquidClient(wallet=HL_WALLET, private_key=HL_PRIVATE_KEY)
    strat  = Strategy(client, symbol, qty)

    # Loop exemplo: substitua por seu feed/candles reais (df_atual)
    # Cada iteração representa chegada de uma nova barra.
    for _ in range(5):
        df_atual = None  # troque por seu dataframe/estrutura de indicadores
        strat.on_bar(df_atual)
        time.sleep(1)

if __name__ == "__main__":
    main()
