# joao bonito
# Databricks notebook source
# MAGIC %pip install ccxt

# COMMAND ----------
print("\n========== INÍCIO DO BLOCO: HISTÓRICO DE TRADES ==========")

# Silencia aviso visual do urllib3 sobre OpenSSL/LibreSSL (sem importar urllib3)
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL 1.1.1\+.*",
    category=Warning,
    module=r"urllib3.*",
)

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# --- Compat: stubs para ambiente local (sem Databricks) ---
try:  # display (Databricks) → no-op amigável
    display  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def display(x):
        try:
            # tenta imprimir DataFrame de forma compacta
            if isinstance(x, pd.DataFrame):
                with pd.option_context("display.max_columns", None, "display.width", 200):
                    print(x)
            else:
                print(x)
        except Exception:
            print(x)

try:  # displayHTML (Databricks) → apenas imprime o texto
    displayHTML  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def displayHTML(html: str):
        print(html)

# URL base da API da Binance
BASE_URL = "https://api.binance.com/api/v3/"

# Função para buscar todos os pares de criptomoedas disponíveis na Binance
def get_all_symbols():
    url = f"{BASE_URL}exchangeInfo"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        symbols = [symbol["symbol"] for symbol in data["symbols"] if "USDT" in symbol["symbol"]]
        return symbols
    else:
        print(f"Erro ao buscar pares de criptomoedas: {response.status_code}")
        return []

# Função para buscar os dados da criptomoeda
# Aceita datetime diretamente
def get_binance_data(symbol, interval, start_date, end_date):
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    all_data = []
    current_start = start_timestamp

    while current_start < end_timestamp:
        url = f"{BASE_URL}klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_timestamp,
            "limit": 1000  # Limite máximo permitido pela Binance API
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            current_start = int(data[-1][0]) + 1  # Atualizar o timestamp inicial para o próximo lote
        else:
            print(f"Erro ao buscar dados da API para {symbol}: {response.status_code}")
            break

    # Formatando os dados
    formatted_data = [{
        "data": item[0],  # Timestamp de abertura
        "valor_fechamento": round(float(item[4]), 7),  # Valor de fechamento com 7 casas decimais
        "criptomoeda": symbol,
        "volume_compra": float(item[5]),  # Volume de compra
        "volume_venda": float(item[7])  # Volume de venda
    } for item in all_data]
    return formatted_data

# Função para calcular o RSI para cada criptomoeda individualmente
def calcular_rsi_por_criptomoeda(df, window):
    df.sort_values(by=["criptomoeda", "data"], inplace=True)
    resultados = []

    for criptomoeda, grupo in df.groupby("criptomoeda"):
        grupo = grupo.copy()
        grupo["change"] = grupo["valor_fechamento"].diff()
        grupo["gain"] = grupo["change"].where(grupo["change"] > 0, 0)
        grupo["loss"] = -grupo["change"].where(grupo["change"] < 0, 0)

        grupo["avg_gain"] = np.nan
        grupo["avg_loss"] = np.nan

        if len(grupo) >= window:
            grupo.iloc[window - 1, grupo.columns.get_loc("avg_gain")] = grupo["gain"].iloc[:window].mean()
            grupo.iloc[window - 1, grupo.columns.get_loc("avg_loss")] = grupo["loss"].iloc[:window].mean()

        for i in range(window, len(grupo)):
            grupo.iloc[i, grupo.columns.get_loc("avg_gain")] = (
                (grupo.iloc[i - 1, grupo.columns.get_loc("avg_gain")] * (window - 1)) + grupo.iloc[i, grupo.columns.get_loc("gain")]
            ) / window
            grupo.iloc[i, grupo.columns.get_loc("avg_loss")] = (
                (grupo.iloc[i - 1, grupo.columns.get_loc("avg_loss")] * (window - 1)) + grupo.iloc[i, grupo.columns.get_loc("loss")]
            ) / window

        grupo["rs"] = grupo["avg_gain"] / grupo["avg_loss"]
        grupo["rsi"] = 100 - (100 / (1 + grupo["rs"]))

        resultados.append(grupo)

    return pd.concat(resultados, ignore_index=True)

# Função para calcular o MACD
def calcular_macd(df, short_window=7, long_window=40, signal_window=9):
    df["ema_short"] = df.groupby("criptomoeda")["valor_fechamento"].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
    df["ema_long"] = df.groupby("criptomoeda")["valor_fechamento"].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
    df["macd"] = df["ema_short"] - df["ema_long"]
    df["macd_signal"] = df.groupby("criptomoeda")["macd"].transform(lambda x: x.ewm(span=signal_window, adjust=False).mean())

    df["indicativo_macd"] = ""
    df.loc[df["macd"] > df["macd_signal"], "indicativo_macd"] = "Alta"
    df.loc[df["macd"] < df["macd_signal"], "indicativo_macd"] = "Baixa"
    df.loc[df["macd"] == df["macd_signal"], "indicativo_macd"] = "Neutro"

    return df

# Configurações do símbolo, intervalo e datas
start_date = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
end_date = datetime.now()  # hoje com horário atual
interval = "15m"

# Buscar todos os pares de criptomoedas disponíveis (ou usar apenas SOLUSDT)
all_symbols = ["SOLUSDT"]

if not all_symbols:
    print("Nenhum símbolo contendo 'USDT' foi encontrado.")
else:
    all_data = []
    for symbol in all_symbols:
        print(f"Buscando dados para {symbol}...")
        symbol_data = get_binance_data(symbol, interval, start_date, end_date)
        if symbol_data:
            all_data.extend(symbol_data)

    if all_data:
        df = pd.DataFrame(all_data)
        df["data"] = pd.to_datetime(df["data"], unit="ms")  # Converter timestamp para datetime
        df = calcular_rsi_por_criptomoeda(df, window=14)
        df["variacao_diaria"] = df.groupby("criptomoeda")["valor_fechamento"].pct_change() * 100
        df["total_linhas"] = df.groupby("criptomoeda")["criptomoeda"].transform("count")
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.month
        df = calcular_macd(df)

        # Gradientes do EMA curto para múltiplas janelas padronizadas (3,5,7,10)
        import numpy as _np
        import pandas as _pd
        def _slope_arr(arr):
            if arr.size < 2 or _np.isnan(arr).all():
                return _np.nan
            # remove NaN internos para estabilidade
            a = arr[~_np.isnan(arr)].astype(float)
            if a.size < 2:
                return _np.nan
            x = _np.arange(a.size, dtype=float)
            m, _b = _np.polyfit(x, a, 1)
            return float(m)
        def _rolling_slope(series: _pd.Series, n: int) -> _pd.Series:
            return series.rolling(window=n, min_periods=2).apply(_slope_arr, raw=True)
        for n in (3, 5, 7, 10):
            col = f"grad_n{n}"
            try:
                df[col] = (
                    df.groupby("criptomoeda", group_keys=False)["ema_short"]
                      .apply(lambda s: _rolling_slope(_pd.to_numeric(s, errors="coerce"), n))
                )
            except Exception:
                # fallback sem groupby
                df[col] = _rolling_slope(_pd.to_numeric(df["ema_short"], errors="coerce"), n)

        pd.set_option('display.float_format', lambda x: f'{x:.7f}')
        pd.set_option('display.max_columns', None)
    else:
        print("Nenhum dado disponível para processar.")

# COMMAND ----------

# Guarda contra falta de dados (ex.: erro 451/sem retorno da API)
if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
    max_date = df["data"].max()
    df_filtered = df[df["data"] == max_date]

    RSI_ATUAL = df_filtered["rsi"].values[0] if "rsi" in df_filtered.columns else None
    ma_short = df_filtered["ema_short"].values[0] if "ema_short" in df_filtered.columns else None
    ma_long = df_filtered["ema_long"].values[0] if "ema_long" in df_filtered.columns else None

    # COMMAND ----------

    if ma_long is not None: print(ma_long)
    if ma_short is not None: print(ma_short)
    if RSI_ATUAL is not None: print(RSI_ATUAL)
else:
    print("[INFO] Sem DF válido para métricas intradiárias (df vazio/não definido).")

# COMMAND ----------

"""
DEX (Hyperliquid via ccxt)
"""
import ccxt  # type: ignore

# ATENÇÃO: chaves privadas em código-fonte. Considere usar variáveis
# de ambiente em produção para evitar exposição acidental.
dex = ccxt.hyperliquid({
    "walletAddress": "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
    "privateKey": "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872",
})

# COMMAND ----------

if dex:
    try:
        dex.fetch_balance()
    except Exception as e:
        print(f"[WARN] Falha ao buscar saldo do DEX: {type(e).__name__}: {e}")

# COMMAND ----------

# COMMAND ----------
# =========================
# 🔔 LOGGER (CSV + XLSX em DBFS com workaround /tmp → dbutils.fs.cp)
# =========================
import os
import pandas as pd
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Py3.9+
    TZ_BRT = ZoneInfo("America/Sao_Paulo")
except Exception:
    TZ_BRT = None  # fallback sem timezone

# Stub seguro de dbutils para ambientes fora do Databricks
try:  # pragma: no cover
    dbutils  # type: ignore[name-defined]
except NameError:  # cria stub mínimo se não existir
    class _DBFSStub:
        def cp(self, src: str, dst: str, recurse: bool = False):
            try:
                import os as _os, shutil as _shutil
                _os.makedirs(_os.path.dirname(dst) or ".", exist_ok=True)
                _shutil.copy(src, dst)
            except Exception:
                pass
        def mkdirs(self, path: str):
            try:
                import os as _os
                _os.makedirs(path, exist_ok=True)
            except Exception:
                pass
    class _DbutilsStub:
        def __init__(self):
            self.fs = _DBFSStub()
    dbutils = _DbutilsStub()  # type: ignore

def _has_dbutils():
    try:
        _ = dbutils  # type: ignore[name-defined]
        return True
    except NameError:
        return False

class TradeLogger:
    def __init__(self, df_columns: pd.Index,
                 csv_path="trade_log.csv",
                 xlsx_path_dbfs="trade_log.xlsx"):
        # No ambiente local, use caminhos relativos
        self.csv_path = csv_path
        self.xlsx_path_dbfs = xlsx_path_dbfs
        self.xlsx_tmp = "/tmp/trade_log.xlsx"  # escreve local, depois copia

        self.meta_cols = [
            "trade_evento", "trade_tipo", "trade_op", "exec_price", "exec_amount",
            "order_id", "dt_evento_utc", "dt_evento_brt"
        ]
        self.all_cols = list(df_columns) + self.meta_cols

        # cria arquivos "vazios" se não existirem (ambiente local)
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.all_cols).to_csv(self.csv_path, index=False)
        else:
            # Se já existe, garante que novas colunas sejam adicionadas preservando dados
            try:
                _existing = pd.read_csv(self.csv_path)
                missing = [c for c in self.all_cols if c not in _existing.columns]
                if missing:
                    for c in missing:
                        _existing[c] = pd.NA
                    # Backfill de trade_op se possível
                    if "trade_op" in missing and {"trade_evento", "trade_tipo"}.issubset(set(_existing.columns)):
                        def _compose_op_row(row):
                            ev = str(row.get("trade_evento", "")).lower()
                            tp = str(row.get("trade_tipo", "")).lower()
                            if ev == "entrada":
                                return f"open_{tp}" if tp in ("long", "short") else "open"
                            if ev in ("saida", "fechado_externo"):
                                return f"close_{tp}" if tp in ("long", "short") else "close"
                            if ev == "ajuste_stop":
                                return f"adjust_stop_{tp}" if tp in ("long", "short") else "adjust_stop"
                            if ev == "preexistente":
                                return f"preexistente_{tp}" if tp in ("long", "short") else "preexistente"
                            return ev
                        _existing["trade_op"] = _existing.apply(_compose_op_row, axis=1)
                    _existing = _existing[self.all_cols]
                    _existing.to_csv(self.csv_path, index=False)
            except Exception:
                pass
        try:
            if not os.path.exists(self.xlsx_path_dbfs):
                pd.DataFrame(columns=self.all_cols).to_excel(self.xlsx_path_dbfs, index=False)
        except Exception:
            # Se não conseguir criar XLSX, seguimos apenas com CSV
            pass

    def _now_strings(self):
        now_utc = datetime.now(timezone.utc)
        dt_utc = now_utc.isoformat(timespec="seconds")
        dt_brt = now_utc.astimezone(TZ_BRT).isoformat(timespec="seconds") if TZ_BRT else ""
        return dt_utc, dt_brt

    def _save_xlsx_dbfs(self, df_all: pd.DataFrame):
        # Ambiente local: grava direto no caminho alvo; mantém assinatura para mínima alteração
        try:
            df_all.to_excel(self.xlsx_path_dbfs, index=False)
        except Exception:
            # fallback silencioso (CSV já é persistido)
            pass

    def append_event(self, df_snapshot: pd.DataFrame,
                     evento: str, tipo: str,
                     exec_price: float = None,
                     exec_amount: float = None,
                     order_id: str = None):
        # Garante que o snapshot possua todas as colunas do DF principal
        missing = [c for c in self.all_cols if c not in list(df_snapshot.columns) + self.meta_cols]
        for c in missing:
            df_snapshot[c] = pd.NA

        def _compose_op(ev: str, tp: str) -> str:
            ev = (ev or "").lower(); tp = (tp or "").lower()
            if ev == "entrada":
                return f"open_{tp}" if tp in ("long", "short") else "open"
            if ev in ("saida", "fechado_externo"):
                return f"close_{tp}" if tp in ("long", "short") else "close"
            if ev == "ajuste_stop":
                return f"adjust_stop_{tp}" if tp in ("long", "short") else "adjust_stop"
            if ev == "preexistente":
                return f"preexistente_{tp}" if tp in ("long", "short") else "preexistente"
            return ev

        dt_utc, dt_brt = self._now_strings()
        meta = {
            "trade_evento": evento,
            "trade_tipo": tipo,
            "trade_op": _compose_op(evento, tipo),
            "exec_price": exec_price,
            "exec_amount": exec_amount,
            "order_id": order_id,
            "dt_evento_utc": dt_utc,
            "dt_evento_brt": dt_brt,
        }

        row = df_snapshot.copy()
        for col in self.meta_cols:
            row[col] = meta[col]
        row = row[self.all_cols]

        if os.path.exists(self.csv_path):
            row.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            row.to_csv(self.csv_path, index=False)

        full = pd.read_csv(self.csv_path)
        try:
            self._save_xlsx_dbfs(full)
            print("✅ Histórico atualizado")
        except Exception as e:
            print(f"⚠️ XLSX não pôde ser atualizado ({type(e).__name__}: {e}). CSV salvo em {os.path.abspath(self.csv_path)}")

# =========================
# 📣 NOTIFICAÇÕES DISCORD
# =========================
import requests as _req
_DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
_HTTP_TIMEOUT = 10
_SESSION = _req.Session()
try:
    _ADAPTER = _req.adapters.HTTPAdapter(max_retries=3)
    _SESSION.mount("https://", _ADAPTER)
    _SESSION.mount("http://", _ADAPTER)
except Exception:
    pass

_HL_INFO_URL = "https://api.hyperliquid.xyz/info"

def _http_post_json(url: str, payload: dict, timeout: int = _HTTP_TIMEOUT):
    try:
        r = _SESSION.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # pragma: no cover
        print(f"[WARN] HTTP falhou: {type(e).__name__}: {e}")
        return None

def _notify_discord(message: str):
    if not _DISCORD_WEBHOOK or "discord.com/api/webhooks" not in _DISCORD_WEBHOOK:
        return
    try:
        resp = _SESSION.post(_DISCORD_WEBHOOK, json={"content": message}, timeout=_HTTP_TIMEOUT)
        if resp.status_code not in (200, 204):
            print(f"[WARN] Discord status {resp.status_code}: {resp.text}")
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Falha ao notificar Discord: {type(e).__name__}: {e}")

def _hl_get_latest_fill(wallet: str):
    if not wallet:
        return None
    return _http_post_json(_HL_INFO_URL, {"type": "userFills", "user": wallet})

def _hl_get_account_value(wallet: str) -> float:
    if not wallet:
        return 0.0
    data = _http_post_json(_HL_INFO_URL, {"type": "clearinghouseState", "user": wallet})
    try:
        return float(data["marginSummary"]["accountValue"]) if data else 0.0
    except Exception:
        return 0.0

# COMMAND ----------


# COMMAND ----------


# COMMAND ----------

# DBTITLE 1,Gatilho de entrada
# =========================
# 🧠 ESTRATÉGIA (HL + stop inicial 6% da margem + trailing BE±0,05% + logger com fallback + DEBUG)
# =========================
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

@dataclass
class GradientConfig:
    EMA_SHORT_SPAN: int     = 7
    EMA_LONG_SPAN: int      = 40
    N_BARRAS_GRADIENTE: int = 5

    SHORT_ENTER_MIN: float  = -0.25
    SHORT_ENTER_MAX: float  = -0.05
    SHORT_EXIT_GT: float    = +0.001    # encerrar SHORT se slope curto > +0.001

    LONG_ENTER_MIN: float   = +0.05
    LONG_ENTER_MAX: float   = +0.25
    LONG_EXIT_LT: float     = -0.001    # encerrar LONG se slope curto < -0.001

    LEVERAGE: int           = 20
    STOP_RISK_PCT: float    = 0.06      # 6% DA MARGEM (~0,3% do preço com 20x)
    MIN_ORDER_USD: float    = 10.0

    COOLDOWN_MINUTOS: int   = 2
    ANTI_SPAM_SECS: int     = 3

    BE_TRIGGER_PCT: float   = 0.005     # +0,5% a favor para acionar BE
    BE_OFFSET_PCT: float    = 0.0005    # BE +0,05% (long) / -0,05% (short)


class EMAGradientStrategy:
    def __init__(self, dex, symbol: str, cfg: GradientConfig = GradientConfig(), logger: "TradeLogger" = None, debug: bool = True):
        self.dex = dex
        self.symbol = symbol
        self.cfg = cfg
        self.logger = logger
        self.debug = debug

        self._cooldown_until: Optional[datetime] = None
        self._last_open_at: Optional[datetime] = None
        self._last_close_at: Optional[datetime] = None
        self._last_adjust_at: Optional[datetime] = None
        self._last_pos_side: Optional[str] = None
        self._first_step_done: bool = False

        base = symbol.split("/")[0]
        self._df_symbol_hint = f"{base}USDT"

        # Buffer local (redundância) e flags
        self._local_events = []              # lista de eventos (fallback/espelho)
        self._local_events_count = 0         # contador de eventos locais
        self.force_local_log = False         # True => ignora logger externo
        self.duplicate_local_always = True   # True => sempre duplica no local

    # ---------- util ----------
    def _norm_side(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        s = str(raw).lower()
        if s in ("buy", "long"):
            return "buy"
        if s in ("sell", "short"):
            return "sell"
        return None

    def _wallet_address(self) -> Optional[str]:
        # Busca carteira: env > dex attributes/options > None
        for key in ("WALLET_ADDRESS", "HYPERLIQUID_WALLET_ADDRESS"):
            val = os.getenv(key)
            if val:
                return val
        try:
            val = getattr(self.dex, "walletAddress", None)
            if val:
                return val
        except Exception:
            pass
        try:
            opts = getattr(self.dex, "options", {}) or {}
            val = opts.get("walletAddress")
            if val:
                return val
        except Exception:
            pass
        return None

    def _notify_trade(self, kind: str, side: Optional[str], price: Optional[float], amount: Optional[float], note: str = "", include_hl: bool = False):
        base = self.symbol.split("/")[0] if "/" in self.symbol else self.symbol
        side_map = {"buy": "LONG", "sell": "SHORT"}
        side_txt = side_map.get((side or "").lower(), "?") if side else "?"
        kind_map = {
            "open": "Abertura",
            "close": "Fechamento",
            "close_external": "Fechamento Externo (stop)",
        }
        kind_pt = kind_map.get(kind, kind.capitalize())
        parts = [
            "📢 Operação",
            f"• Tipo: {kind_pt}",
            f"• Par: {base}",
            f"• Lado: {side_txt}",
        ]
        if price is not None:
            parts.append(f"• Preço: {price:.6f}")
        if amount is not None:
            parts.append(f"• Quantidade: {amount}")
        if note:
            parts.append(f"• Obs: {note}")

        # Dados opcionais da Hyperliquid (Resultado/Valor da conta)
        if include_hl:
            wallet = self._wallet_address()
            fills = _hl_get_latest_fill(wallet)
            try:
                last = fills[0] if isinstance(fills, list) and fills else None
                if last:
                    pnl_raw = last.get("closedPnl")
                    try:
                        pnl = float(pnl_raw)
                        parts.append(f"• Resultado (PnL): {pnl:.2f} USDC")
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                acc_val = _hl_get_account_value(wallet)
                if acc_val:
                    parts.append(f"• Valor da Conta: {acc_val:.2f} USDC")
            except Exception:
                pass

        _notify_discord("\n".join(parts))

    # ---------- leitura de contexto para log ----------
    def _read_context(self):
        """
        Retorna contexto leve para log:
          - px_now: preço atual (float ou None se falhar)
          - pos_side: 'buy'|'sell'|None
          - qty: contratos/amount (float ou 0.0)
          - entry: preço de entrada (float ou None)
        """
        px_now = None
        pos_side = None
        qty = 0.0
        entry = None

        # tenta preço atual
        try:
            px_now = self._preco_atual()
        except Exception:
            pass

        # tenta posição
        try:
            pos = self._posicao_aberta()
            if pos:
                pos_side = self._norm_side(pos.get("side") or pos.get("positionSide"))
                qty = float(pos.get("contracts") or 0.0)
                ep = (pos.get("entryPrice") or pos.get("entryPx") or 0.0)
                entry = float(ep) if ep else None
        except Exception:
            pass

        return {"px_now": px_now, "pos_side": pos_side, "qty": qty, "entry": entry}

    # ---------- logging com redundância + fallback + auto-preenchimento ----------
    def _safe_log(self, evento: str, df_for_log: Optional[pd.DataFrame], **kwargs):
        """
        Log ultra-robusto + redundante:
          - Sempre grava no buffer local (duplicate_local_always=True).
          - Preenche exec_price/exec_amount a partir do contexto se vierem None.
          - Pode forçar somente local (force_local_log=True).
          - Logger externo: tenta com snapshot leve → sem snapshot → com stub vazio.
          - Aceita chaves: tipo, exec_price, exec_amount, order_id.
        """
        # (A) contexto
        ctx = self._read_context()
        tipo = kwargs.get("tipo") or "info"
        exec_price  = kwargs.get("exec_price")
        exec_amount = kwargs.get("exec_amount")
        order_id    = kwargs.get("order_id")

        # auto-fill
        if exec_price is None:
            exec_price = ctx["entry"] if (evento == "preexistente" and ctx["entry"]) else ctx["px_now"]
        if exec_amount is None:
            exec_amount = ctx["qty"] if ctx["qty"] else None  # mantém None se 0.0

        to_send = {"tipo": tipo}
        if exec_price  is not None:  to_send["exec_price"]  = exec_price
        if exec_amount is not None:  to_send["exec_amount"] = exec_amount
        if order_id    is not None:  to_send["order_id"]    = order_id

        # (B) snapshot COMPLETO da última linha do DF para log
        snap = None
        if isinstance(df_for_log, pd.DataFrame) and len(df_for_log) > 0:
            try:
                snap = df_for_log.tail(1)
            except Exception:
                snap = df_for_log.iloc[[-1]]

        # (C) SEMPRE grava no buffer local
        try:
            row_local = {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "evento": evento,
                "tipo": tipo,
                "exec_price": exec_price,
                "exec_amount": exec_amount,
                "order_id": order_id,
                "side_ctx": ctx["pos_side"],
                "entry_ctx": ctx["entry"],
                "px_now_ctx": ctx["px_now"],
                "snapshot": None
            }
            if snap is not None:
                try:
                    row_local["snapshot"] = snap.to_dict(orient="records")[0]
                except Exception:
                    row_local["snapshot"] = None

            self._local_events.append(row_local)
            self._local_events_count += 1
            print(f"📝 (local) '{evento}' registrado | total_local={self._local_events_count}")
        except Exception as e:
            print(f"❌ Falha ao registrar no buffer local: {type(e).__name__}: {e}")

        # (D) somente local?
        if self.force_local_log or self.logger is None:
            return

        # (E) tenta logger externo
        try:
            self.logger.append_event(df_snapshot=snap, evento=evento, **to_send)
            print(f"✅ Logger externo OK: '{evento}' (com snapshot)")
            return
        except Exception as e1:
            print(f"⚠️ Logger externo falhou (com snapshot): {type(e1).__name__}: {e1} → tentando sem snapshot...")

        try:
            self.logger.append_event(evento=evento, **to_send)
            print(f"✅ Logger externo OK: '{evento}' (sem snapshot)")
            return
        except Exception as e2:
            print(f"⚠️ Logger externo falhou (sem snapshot): {type(e2).__name__}: {e2} → tentando stub...")

        try:
            df_stub = pd.DataFrame({"ts": [datetime.utcnow()]})
            self.logger.append_event(df_snapshot=df_stub, evento=evento, **to_send)
            print(f"✅ Logger externo OK: '{evento}' (stub)")
            return
        except Exception as e3:
            print(f"⚠️ Logger externo falhou (stub): {type(e3).__name__}: {e3} → mantendo somente local.")

    # atalho para logar com contexto automaticamente
    def log_with_context(self, evento: str, df_for_log: Optional[pd.DataFrame] = None, tipo: str = "info"):
        return self._safe_log(evento, df_for_log=df_for_log, tipo=tipo)

    # ---------- helpers do buffer local ----------
    def local_log_tail(self, n: int = 10):
        """Retorna os últimos n eventos do buffer local (lista de dicts)."""
        if not self._local_events:
            return []
        return self._local_events[-n:]


    def clear_local_log(self):
        """Zera o buffer local."""
        n = len(self._local_events)
        self._local_events.clear()
        self._local_events_count = 0
        print(f"🧹 Buffer local limpo (removidos {n} eventos).")

    def export_local_log_csv(self, path: str = "trade_events_fallback.csv"):
        """Exporta o buffer local para CSV."""
        if not self._local_events:
            print("ℹ️ Nenhum evento no buffer local para exportar.")
            return None
        try:
            import json
            flat = []
            for ev in self._local_events:
                ev_copy = ev.copy()
                snap = ev_copy.pop("snapshot", None)
                ev_copy["snapshot_json"] = json.dumps(snap, ensure_ascii=False) if isinstance(snap, dict) else None
                flat.append(ev_copy)
            df = pd.DataFrame(flat)
            df.to_csv(path, index=False)
            print(f"✅ Exportado: {path} ({len(df)} eventos)")
            return path
        except Exception as e:
            print(f"❌ Falha ao exportar CSV: {type(e).__name__}: {e}")
            return None

    # ---------- numéricos ----------
    def _gradiente(self, serie, n=None) -> float:
        if n is None:
            n = self.cfg.N_BARRAS_GRADIENTE
        s = np.asarray(serie, dtype=float)
        if s.size < 2:
            return 0.0
        n = min(s.size, n)
        y = s[-n:]
        x = np.arange(n, dtype=float)
        a, _b = np.polyfit(x, y, 1)
        return float(a)

    def _ensure_emas_and_slopes(self, df: pd.DataFrame) -> pd.DataFrame:
        if "valor_fechamento" not in df.columns:
            raise ValueError("df precisa ter a coluna 'valor_fechamento'.")
        out = df.copy()
        if "data" in out.columns:
            out = out.sort_values("data")
        close = pd.to_numeric(out["valor_fechamento"], errors="coerce")
        if ("ema_short" not in out.columns) or out["ema_short"].isna().any():
            out.loc[:, "ema_short"] = close.ewm(span=self.cfg.EMA_SHORT_SPAN, adjust=False).mean()
        if ("ema_long" not in out.columns) or out["ema_long"].isna().any():
            out.loc[:, "ema_long"] = close.ewm(span=self.cfg.EMA_LONG_SPAN, adjust=False).mean()
        out.loc[:, "slope_short"] = np.nan
        out.loc[:, "slope_long"]  = np.nan

        if len(out) >= 2:
            def _slope_last(arr) -> float:
                valid = np.asarray(arr, dtype=float)
                valid = valid[~np.isnan(valid)]
                if valid.size < 2:
                    return 0.0
                w = min(valid.size, self.cfg.N_BARRAS_GRADIENTE)
                y = valid[-w:]
                x = np.arange(w, dtype=float)
                a, _b = np.polyfit(x, y, 1)
                return float(a)
            out.loc[out.index[-1], "slope_short"] = _slope_last(out["ema_short"])
            out.loc[out.index[-1], "slope_long"]  = _slope_last(out["ema_long"])
        return out

    # ---------- exchange ----------
    def _preco_atual(self) -> float:
        try:
            mkts = self.dex.load_markets()
            info = mkts[self.symbol]["info"]
            if info.get("midPx") is not None:
                return float(info["midPx"])
        except Exception:
            pass
        t = self.dex.fetch_ticker(self.symbol)
        if t and t.get("last"):
            return float(t["last"])
        raise RuntimeError("Não consegui obter preço atual (midPx/last).")

    def _posicao_aberta(self) -> Optional[Dict[str, Any]]:
        pos = self.dex.fetch_positions([self.symbol])
        if pos and float(pos[0].get("contracts", 0)) > 0:
            return pos[0]
        return None

    def _tem_ordem_de_entrada_pendente(self) -> bool:
        try:
            for o in self.dex.fetch_open_orders(self.symbol):
                ro = o.get("reduceOnly")
                if ro is None and isinstance(o.get("params"), dict):
                    ro = o["params"].get("reduceOnly")
                if not ro:
                    return True
        except Exception:
            pass
        return False

    def _cooldown_ativo(self) -> bool:
        return self._cooldown_until and datetime.utcnow() < self._cooldown_until

    def _marcar_cooldown(self):
        self._cooldown_until = datetime.utcnow() + timedelta(minutes=self.cfg.COOLDOWN_MINUTOS)

    def _anti_spam_ok(self, kind: str) -> bool:
        now = datetime.utcnow()
        if kind == "open":
            if self._last_open_at and (now - self._last_open_at).total_seconds() < self.cfg.ANTI_SPAM_SECS:
                return False
            self._last_open_at = now;  return True
        if kind == "close":
            if self._last_close_at and (now - self._last_close_at).total_seconds() < self.cfg.ANTI_SPAM_SECS:
                return False
            self._last_close_at = now; return True
        if kind == "adjust":
            if self._last_adjust_at and (now - self._last_adjust_at).total_seconds() < self.cfg.ANTI_SPAM_SECS:
                return False
            self._last_adjust_at = now; return True
        return True

    def _round_amount(self, amount: float) -> float:
        try:
            return float(self.dex.amount_to_precision(self.symbol, amount))
        except Exception:
            return float(amount)

    # ---------- stop reduceOnly ----------
    def _place_stop(self, side: str, amount: float, stop_price: float):
        amt = self._round_amount(amount)
        px  = float(stop_price)
        price_ref = self._preco_atual()
        params = {"reduceOnly": True, "stopLossPrice": px, "triggerPrice": px, "trigger": "mark"}
        try:
            if self.debug:
                print(f"🛑 Criando STOP {side.upper()} reduceOnly @ {px:.6f} (ref {price_ref:.6f})")
            return self.dex.create_order(self.symbol, "market", side, amt, price_ref, params)
        except Exception as e1:
            print(f"⚠️ stop formato1 falhou: {type(e1).__name__}: {e1}")
        try:
            return self.dex.create_order(self.symbol, "stop_market", side, amt, price_ref, params)
        except Exception as e2:
            print(f"❌ stop formato2 falhou: {type(e2).__name__}: {e2}")
            raise

    # ---------- ordens ----------
    def _abrir_posicao_com_stop(self, side: str, usd_to_spend: float, df_for_log: pd.DataFrame):
        if self._posicao_aberta():
            print("↪️ Já existe posição aberta. Abortando nova entrada."); return None, None
        if self._tem_ordem_de_entrada_pendente():
            print("↪️ Ordem de ENTRADA pendente detectada. Abortando nova entrada."); return None, None
        if not self._anti_spam_ok("open"):
            print("⏳ Anti-spam (open) acionado."); return None, None

        usd_to_spend = max(usd_to_spend, self.cfg.MIN_ORDER_USD / self.cfg.LEVERAGE)
        price  = self._preco_atual()
        amount = self._round_amount((usd_to_spend * self.cfg.LEVERAGE) / price)

        print(f"✅ Abrindo {side.upper()} | notional≈${usd_to_spend*self.cfg.LEVERAGE:.2f} | amount≈{amount:.6f} @ {price:.4f}")
        ordem_entrada = self.dex.create_order(self.symbol, "market", side, amount, price)
        print("↳ Entrada:", ordem_entrada)

        oid = None
        try:
            oid = (ordem_entrada.get("id")
                   or (ordem_entrada.get("info", {}).get("filled", {}) or {}).get("oid"))
        except Exception:
            pass

        self._safe_log(
            "entrada", df_for_log,
            tipo=("long" if self._norm_side(side) == "buy" else "short"),
            exec_price=price,
            exec_amount=amount,
            order_id=str(oid) if oid else None
        )

        # Notificação de abertura
        try:
            self._notify_trade(
                kind="open",
                side=self._norm_side(side),
                price=price,
                amount=amount,
                note="entrada executada",
                include_hl=False,
            )
        except Exception:
            pass

        # stop inicial = 6% da margem
        capital_loss  = usd_to_spend * self.cfg.STOP_RISK_PCT
        loss_per_unit = capital_loss / amount
        if self._norm_side(side) == "buy":
            sl_price = price - loss_per_unit
            sl_side  = "sell"
        else:
            sl_price = price + loss_per_unit
            sl_side  = "buy"

        if self.debug:
            print(f"🔎 Stop inicial calculado: margem={usd_to_spend:.2f}, risco={self.cfg.STOP_RISK_PCT*100:.2f}% "
                  f"⇒ capital_loss={capital_loss:.2f}, loss_per_unit={loss_per_unit:.6f}, stop @ {sl_price:.6f} ({sl_side.upper()})")

        ordem_stop = self._place_stop(sl_side, amount, sl_price)
        print(f"📉 Stop inicial @ {sl_price:.6f} "
              f"(≈{(loss_per_unit/price)*100:.3f}% do preço | {self.cfg.STOP_RISK_PCT*100:.2f}% da margem):", ordem_stop)

        self._safe_log(
            "stop_inicial", df_for_log,
            tipo=("long" if self._norm_side(side) == "buy" else "short"),
            exec_price=sl_price,
            exec_amount=amount
        )
        return ordem_entrada, ordem_stop

    # ---------- localizar/cancelar stop existente ----------
    def _find_existing_stop(self):
        try:
            for o in self.dex.fetch_open_orders(self.symbol):
                ro = o.get("reduceOnly")
                if ro is None and isinstance(o.get("params"), dict):
                    ro = o["params"].get("reduceOnly")
                if not ro:
                    continue
                stop_px = (
                    o.get("stopPrice")
                    or (o.get("info", {}).get("stopLossPrice"))
                    or (o.get("params", {}).get("stopLossPrice") if isinstance(o.get("params"), dict) else None)
                )
                if stop_px is None:
                    continue
                side_is_sell = self._norm_side(o.get("side")) == "sell"
                return o.get("id"), float(stop_px), side_is_sell
        except Exception:
            pass
        return None, None, None

    def _cancel_order_silent(self, order_id):
        try:
            if order_id:
                if self.debug:
                    print(f"🧹 Cancelando ordem existente id={order_id}")
                self.dex.cancel_order(order_id, self.symbol)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Falha ao cancelar ordem {order_id}: {e}")

    # ---------- fechar posição via market reduceOnly ----------
    def _market_reduce_only(self, side: str, amount: float):
        amt = self._round_amount(amount)
        px  = self._preco_atual()
        params = {"reduceOnly": True}
        if self.debug:
            print(f"🧲 Fechando com MARKET {side.upper()} reduceOnly qty={amt} ref={px:.6f}")
        return self.dex.create_order(self.symbol, "market", side, amt, px, params)

    def _fechar_posicao(self, df_for_log: pd.DataFrame):
        pos = self._posicao_aberta()
        if not pos or float(pos.get("contracts", 0)) == 0:
            print("↪️ Não há posição para fechar. Abortando."); return
        if not self._anti_spam_ok("close"):
            print("⏳ Anti-spam (close) acionado."); return

        lado_atual = self._norm_side(pos.get("side") or pos.get("positionSide"))
        qty        = float(pos.get("contracts") or 0.0)
        price_now  = self._preco_atual()
        if self.debug:
            print(f"🔎 Fechando posição {lado_atual.upper()} qty={qty} @ {price_now:.6f}")

        # cancela stops reduceOnly existentes
        try:
            oid, _cur_stop, _is_sell = self._find_existing_stop()
            if oid:
                self._cancel_order_silent(oid)
        except Exception:
            pass

        # fechamento via market reduceOnly (lado oposto)
        try:
            close_side = "sell" if lado_atual == "buy" else "buy"
            ret = self._market_reduce_only(close_side, qty)
            print("🔚 Posição encerrada (reduceOnly market):", ret)
            oid = ret.get("id") if isinstance(ret, dict) else None
        except Exception as e:
            print("❌ Erro ao fechar posição (reduceOnly):", e); oid = None
        finally:
            self._safe_log(
                "saida", df_for_log,
                tipo=("long" if lado_atual == "buy" else "short"),
                exec_price=price_now,
                exec_amount=qty,
                order_id=str(oid) if oid else None
            )
            self._marcar_cooldown()

            # Notificação de fechamento (inclui tentativa de PnL/valor conta)
            try:
                self._notify_trade(
                    kind="close",
                    side=lado_atual,
                    price=price_now,
                    amount=qty,
                    note="fechamento por decisão/trigger",
                    include_hl=True,
                )
            except Exception:
                pass

    # ---------- trailing BE± ----------
    def _maybe_trailing_breakeven_plus(self, pos: Dict[str, Any], df_for_log: pd.DataFrame):
        if not pos:
            return
        side  = self._norm_side(pos.get("side") or pos.get("positionSide"))
        entry = float(pos.get("entryPrice") or pos.get("entryPx") or 0.0)
        amt   = float(pos.get("contracts") or 0.0)
        if entry <= 0 or amt <= 0:
            return

        px_now = self._preco_atual()
        trg, off = self.cfg.BE_TRIGGER_PCT, self.cfg.BE_OFFSET_PCT

        if self.debug:
            trig_mult = (1.0 + trg) if side == "buy" else (1.0 - trg)
            off_mult  = (1.0 + off) if side == "buy" else (1.0 - off)
            print(f"🔎 Checando BE± | side={side.upper()} entry={entry:.6f} px_now={px_now:.6f} "
                  f"trigger_mult={trig_mult:.6f} off_mult={off_mult:.6f}")

        if side == "buy":
            if px_now < entry * (1.0 + trg):
                if self.debug:
                    print(f"… BE não ativado (LONG): px_now {px_now:.6f} < {entry*(1+trg):.6f}")
                return
            target_stop = entry * (1.0 + off)
            stop_side   = "sell"
            better      = lambda new, cur: (cur is None) or (new > cur)
        elif side == "sell":
            if px_now > entry * (1.0 - trg):
                if self.debug:
                    print(f"… BE não ativado (SHORT): px_now {px_now:.6f} > {entry*(1-trg):.6f}")
                return
            target_stop = entry * (1.0 - off)
            stop_side   = "buy"
            better      = lambda new, cur: (cur is None) or (new < cur)
        else:
            return

        oid, cur_stop, cur_is_sell = self._find_existing_stop()
        if self.debug:
            print(f"🔎 Stop atual: id={oid} px={cur_stop} lado_sell?={cur_is_sell} | target_stop={target_stop:.6f}")

        # stop do lado errado? remove
        if cur_stop is not None:
            if (side == "buy" and not cur_is_sell) or (side == "sell" and cur_is_sell):
                if self.debug:
                    print("🧹 Stop do lado errado ⇒ cancelando para recriar do lado correto.")
                self._cancel_order_silent(oid)
                cur_stop, oid = None, None

        if not better(target_stop, cur_stop):
            if self.debug:
                print("… Não melhora o stop atual ⇒ mantendo.")
            return
        if not self._anti_spam_ok("adjust"):
            if self.debug:
                print("⏳ Anti-spam (adjust) acionado ⇒ sem ajuste.")
            return

        if oid:
            self._cancel_order_silent(oid)
        ret = self._place_stop(stop_side, amt, target_stop)
        print(f"🔒 Trailing BE±: novo stop {stop_side.upper()} @ {target_stop:.6f} (entry {entry:.6f}, px_now {px_now:.6f})")

        self._safe_log(
            "ajuste_stop", df_for_log,
            tipo=("long" if side == "buy" else "short"),
            exec_price=px_now,
            exec_amount=amt
        )

    # ---------- loop principal ----------
    def step(self, df: pd.DataFrame, usd_to_spend: float):
        # filtra símbolo, se DF tiver múltiplos
        if "criptomoeda" in df.columns and (df["criptomoeda"] == self._df_symbol_hint).any():
            df = df.loc[df["criptomoeda"] == self._df_symbol_hint].copy()
        else:
            df = df.copy()

        df = self._ensure_emas_and_slopes(df)
        slope_short = float(df["slope_short"].iloc[-1]) if not pd.isna(df["slope_short"].iloc[-1]) else 0.0
        slope_long  = float(df["slope_long"].iloc[-1])  if not pd.isna(df["slope_long"].iloc[-1])  else 0.0

        print(f"📐 Gradiente EMA curto: {slope_short:.6f} | 📏 Gradiente EMA longo: {slope_long:.6f}")
        if self.debug:
            print(f"🔎 Regras: SHORT_ENTER ∈ [{self.cfg.SHORT_ENTER_MIN},{self.cfg.SHORT_ENTER_MAX}] | "
                  f"LONG_ENTER ∈ [{self.cfg.LONG_ENTER_MIN},{self.cfg.LONG_ENTER_MAX}] | "
                  f"SHORT_EXIT_GT {self.cfg.SHORT_EXIT_GT} | LONG_EXIT_LT {self.cfg.LONG_EXIT_LT}")

        # primeira execução: loga posição preexistente
        if not self._first_step_done:
            pos_now = self._posicao_aberta()
            if pos_now and float(pos_now.get("contracts", 0)) > 0:
                lado_atual = self._norm_side(pos_now.get("side") or pos_now.get("positionSide"))
                qty = float(pos_now.get("contracts") or 0.0)
                entry = float(pos_now.get("entryPrice") or pos_now.get("entryPx") or 0.0) or None
                self._safe_log(
                    "preexistente", df_for_log=df,
                    tipo=("long" if lado_atual == "buy" else "short"),
                    exec_price=entry,
                    exec_amount=qty
                )
                print("🧭 Posição preexistente detectada e logada.")
            self._first_step_done = True

        prev_side = self._last_pos_side
        pos = self._posicao_aberta()
        print("📊 Posição atual:", pos)

        # se havia posição e agora não há → stop/saída ocorreu fora
        if prev_side and not pos:
            print("📉 Detecção: posição fechada na exchange (provável stop/saída executada).")
            try:
                last_px = self._preco_atual()
            except Exception:
                last_px = None
            self._safe_log(
                "fechado_externo", df_for_log=df,
                tipo=("long" if prev_side == "buy" else "short"),
                exec_price=last_px
            )

            # Notificação de fechamento externo (provável stop)
            try:
                self._notify_trade(
                    kind="close_external",
                    side=prev_side,
                    price=last_px,
                    amount=None,
                    note="fechado externamente (possível stop)",
                    include_hl=True,
                )
            except Exception:
                pass

        if self._cooldown_ativo():
            print(f"⛔ Cooldown {self.cfg.COOLDOWN_MINUTOS} min ativo.")
            self._last_pos_side = (self._norm_side(pos.get("side")) if pos else None)
            return

        # trailing BE± se houver posição
        if pos:
            self._maybe_trailing_breakeven_plus(pos, df_for_log=df)

        # entradas (sem posição)
        if not pos:
            if self.cfg.SHORT_ENTER_MIN <= slope_short <= self.cfg.SHORT_ENTER_MAX:
                print("✅ Sinal SHORT (entrada)")
                self._abrir_posicao_com_stop("sell", usd_to_spend, df_for_log=df)
                pos_after = self._posicao_aberta()
                self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                return
            if self.cfg.LONG_ENTER_MIN <= slope_short <= self.cfg.LONG_ENTER_MAX:
                print("✅ Sinal LONG (entrada)")
                self._abrir_posicao_com_stop("buy", usd_to_spend, df_for_log=df)
                pos_after = self._posicao_aberta()
                self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                return
            print("⏸ Sem posição e sem sinal.")
            self._safe_log("decisao", df_for_log=df, tipo="info")
            self._last_pos_side = None
            return

        # saídas (com posição)
        lado = self._norm_side(pos.get("side") or pos.get("positionSide"))
        if lado == "sell":
            if slope_short > self.cfg.SHORT_EXIT_GT:
                if self.debug:
                    print(f"🔎 Regras de saída SHORT: slope_short({slope_short:.6f}) > SHORT_EXIT_GT({self.cfg.SHORT_EXIT_GT}) ⇒ fechar")
                self._fechar_posicao(df_for_log=df)
                self._last_pos_side = None
                return
        elif lado == "buy":
            if slope_short < self.cfg.LONG_EXIT_LT:
                if self.debug:
                    print(f"🔎 Regras de saída LONG: slope_short({slope_short:.6f}) < LONG_EXIT_LT({self.cfg.LONG_EXIT_LT}) ⇒ fechar")
                self._fechar_posicao(df_for_log=df)
                self._last_pos_side = None
                return

        print("🔄 Mantendo posição.")
        self._safe_log("decisao", df_for_log=df, tipo="info")
        self._last_pos_side = lado if lado in ("buy", "sell") else None


# COMMAND ----------

# DBTITLE 1,principal
# =========================
# 🔧 INSTÂNCIA E EXECUÇÃO
# =========================

if dex is not None and 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
    # 1) Logger com as colunas do DF final
    trade_logger = TradeLogger(df_columns=df.columns)

    # crie a estratégia:
    strategy = EMAGradientStrategy(dex, "SOL/USDC:USDC", GradientConfig(), logger=trade_logger)  # logger opcional

    # chame a cada atualização de candle:
    strategy.step(df, usd_to_spend=10)
else:
    print("[INFO] Sem dados ou DEX indisponível; pulando estratégia.")


# 4) (Opcional) Exibir histórico salvo com guard de vazio
csv_path = "trade_log.csv"
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    _hist = pd.read_csv(csv_path)
    if len(_hist) > 0:
        pass  # display removido
    else:
        print("Histórico criado, mas ainda sem linhas (nenhuma entrada/saída/ajuste).")
else:
    print("Histórico ainda não existe. Execute um trade para gerar registros.")


# COMMAND ----------

if dex is not None:
    # cria o objeto da estratégia
    bot = EMAGradientStrategy(
        dex, 
        "SOL/USDC:USDC",   # símbolo
        logger=None,       # ou seu TradeLogger se quiser testar
        debug=True
    )


# COMMAND ----------

if dex is not None:
    # 1) Confirme que está usando a MESMA instância
    print("bot id:", id(bot))

    # 2) Veja quantos eventos existem agora
    print("len(_local_events):", len(getattr(bot, "_local_events", [])))

    # 3) Forçe um evento local para testar o fluxo
    bot.force_local_log = True  # ignora logger externo
    bot._safe_log("teste_manual", df_for_log=None, tipo="info", exec_price=None, exec_amount=None)

if dex is not None:
    # 4) Cheque novamente
    print("len(_local_events) após teste:", len(bot._local_events))

# 5) Pré-visualize os últimos 5
def preview_local_events(bot, n: int = 10):
    if not hasattr(bot, "_local_events") or not bot._local_events:
        print("ℹ️ Nenhum evento no buffer local."); return
    for i, ev in enumerate(bot._local_events[-n:], 1):
        base = {k: ev.get(k) for k in ["ts","evento","tipo","exec_price","exec_amount","order_id"]}
        print(f"{i:02d}. {base}")

if dex is not None:
    preview_local_events(bot, 5)

if dex is not None:
    # 6) Exporte (se houver algo)
    bot.export_local_log_csv("meu_historico_local.csv")


# COMMAND ----------

import pandas as pd

try:
    df_log = pd.read_csv("meu_historico_local.csv")
except FileNotFoundError:
    print("ℹ️ 'meu_historico_local.csv' não encontrado; pulando pré-visualização.")


# COMMAND ----------

df["max_50"] = df.groupby("criptomoeda")["valor_fechamento"].transform(lambda x: x.rolling(window=50, min_periods=1).max())

# COMMAND ----------

import numpy as np

def gradiente_serie(y: np.ndarray) -> float:
    """Calcula o gradiente (inclinação) linear de uma série."""
    arr = np.asarray(y, dtype=float)
    if arr.size < 2 or np.isnan(arr).any():
        return np.nan
    x = np.arange(arr.size, dtype=float)
    a, _b = np.polyfit(x, arr, 1)
    return float(a)

# tamanho da janela para o cálculo
N_BARRAS_GRADIENTE = 5

df["gradiente_ema_curto"] = (
    df["ema_short"]
    .rolling(window=N_BARRAS_GRADIENTE, min_periods=2)
    .apply(gradiente_serie, raw=True)
)

df["gradiente_ema_longo"] = (
    df["ema_long"]
    .rolling(window=N_BARRAS_GRADIENTE, min_periods=2)
    .apply(gradiente_serie, raw=True)
)

# COMMAND ----------

# Converte o DataFrame PySpark para Pandas
df_pandas = df

# Salva como um único arquivo CSV local
df_pandas.to_csv("previsoes_df.csv", index=False)

# COMMAND ----------

# =========================
# 🔬 TESTE: ÚNICO GRADIENTE POR PERÍODO (sem gatilhos)
# =========================

# Série base para calcular o gradiente (troque para "ema_long" ou "valor_fechamento" se quiser)
SERIE_BASE = "ema_short"

# Garante EMAs caso ainda não existam
if "ema_short" not in df.columns:
    df = df.sort_values("data") if "data" in df.columns else df
    df["ema_short"] = df["valor_fechamento"].ewm(span=7, adjust=False).mean()
if "ema_long" not in df.columns:
    df["ema_long"]  = df["valor_fechamento"].ewm(span=40, adjust=False).mean()

# Copia base
df2 = df.copy()

# Função para gradiente (slope) via regressão linear na janela N
def _rolling_slope(series: pd.Series, n: int) -> pd.Series:
    def _slope(x):
        idx = np.arange(len(x), dtype=float)
        a, _b = np.polyfit(idx, x, 1)
        return a
    return series.rolling(window=n, min_periods=n).apply(_slope, raw=True)

# Períodos a testar (ajuste à vontade)
periodos = [3, 5, 7, 10, 12, 15, 20]

# Calcula UM conjunto de gradientes, todos sobre a mesma série base
if SERIE_BASE not in df2.columns:
    raise ValueError(f"Série base '{SERIE_BASE}' não encontrada nas colunas do DF.")

for n in periodos:
    df2[f"grad_n{n}"] = _rolling_slope(df2[SERIE_BASE].astype(float), n)

# Painel amigável
print("====================")
print("🗂️ Colunas incluídas:")
cols_fixas = ["data"] if "data" in df2.columns else []
cols_fixas += ["valor_fechamento", "ema_short", "ema_long"]

cols_view = [c for c in cols_fixas if c in df2.columns] + [f"grad_n{n}" for n in periodos]
with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.6f}".format):
    pass  # Removido o print da tabela

# Resumo da última barra
last_idx = df2.index[-1]
print(f"📍 RESUMO (última barra) | base={SERIE_BASE}")
print("====================")
vals = []
if "valor_fechamento" in df2.columns: vals.append(f"close={df2.loc[last_idx,'valor_fechamento']:.6f}")
if "ema_short" in df2.columns:        vals.append(f"ema_short={df2.loc[last_idx,'ema_short']:.6f}")
if "ema_long" in df2.columns:         vals.append(f"ema_long={df2.loc[last_idx,'ema_long']:.6f}")
print("  " + "  ".join(vals))

for n in periodos:
    g = df2.loc[last_idx, f"grad_n{n}"]
    print(f"  N={n:>2} → grad={g:.6f}")

# (opcional) exportar CSV:
df2.to_csv("gradiente_unico_por_periodo.csv", index=False)

print("========== FIM DO BLOCO: HISTÓRICO DE TRADES ==========\n")
