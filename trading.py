print("\n========== INÍCIO DO BLOCO: HISTÓRICO DE TRADES ==========", flush=True)

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
from datetime import datetime, timedelta, timezone
import os
import sys  # Adicione esta linha no topo do arquivo

BASE_URL = "https://api.binance.com/api/v3/"

# Variáveis globais padronizadas
try:
    UTC = datetime.UTC  # Python 3.11+
except Exception:
    UTC = timezone.utc

# Janela padrão e intervalo
START_DATE = datetime.now(UTC) - timedelta(hours=48)
END_DATE = datetime.now(UTC)
INTERVAL = "15m"
interval = INTERVAL  # compat com trechos legados

# df global (placeholder); será preenchido mais adiante
df: pd.DataFrame = pd.DataFrame()

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

# URL(s) base da API da Binance com failover
import time as _time

def _binance_bases():
    # Força o endpoint público (dados históricos) para evitar 451/403
    return ["https://data-api.binance.vision/api/v3/"]

def _binance_session():
    s = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=int(os.getenv("BINANCE_RETRIES", "3")),
            backoff_factor=float(os.getenv("BINANCE_BACKOFF", "0.5")),
            status_forcelist=[429, 451, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter); s.mount("http://", adapter)
    except Exception:
        pass
    s.headers.update({
        "User-Agent": os.getenv("BINANCE_UA", "Mozilla/5.0 (X11; Linux x86_64) PythonRequests/2.x"),
        "Accept": "application/json",
    })
    return s

# Função para buscar todos os pares de criptomoedas disponíveis na Binance
def get_all_symbols():
    session = _binance_session()
    timeout = int(os.getenv("BINANCE_TIMEOUT", "10"))
    last_err = None
    for base in _binance_bases():
        url = f"{base}exchangeInfo"
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                symbols = [symbol["symbol"] for symbol in data.get("symbols", []) if "USDT" in symbol.get("symbol", "")]
                if symbols:
                    return symbols
            else:
                last_err = response.status_code
        except Exception as e:
            last_err = e
        _time.sleep(0.2)
    print(f"[INFO] exchangeInfo falhou ({last_err})", flush=True)
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
            "limit": 1000
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            current_start = int(data[-1][0]) + 1
        else:
            print(f"Erro ao buscar dados da API para {symbol}: {response.status_code}")
            break
    formatted_data = [{
        "data": item[0],
        "valor_fechamento": round(float(item[4]), 7),
        "criptomoeda": symbol,
        "volume_compra": float(item[5]),
        "volume_venda": float(item[7])
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

# =========================
# Montagem do DF principal (48h, INTERVAL) com fallbacks
# =========================
def build_df(symbol: str = "SOLUSDT", tf: str = "15m",
             start: datetime = None, end: datetime = None,
             debug: bool = True,
             target_candles: int = None) -> pd.DataFrame:
    # Sempre prioriza um número alvo de candles (inclui o atual não fechado)
    n_target = int(os.getenv("TARGET_CANDLES", "0"))
    if target_candles is not None:
        n_target = int(target_candles)
    if n_target <= 0:
        n_target = 190  # padrão solicitado

    if debug:
        print(f"[INFO] Build DF Bybit {symbol} tf={tf} alvo={n_target} candles (inclui atual)", flush=True)

    # Calcula timestamp do início do candle atual (alinhado ao timeframe)
    def _tf_seconds(tf_str: str) -> int:
        tf_str = tf_str.lower()
        if tf_str.endswith('m'):
            return int(tf_str[:-1]) * 60
        if tf_str.endswith('h'):
            return int(tf_str[:-1]) * 3600
        if tf_str.endswith('d'):
            return int(tf_str[:-1]) * 86400
        # fallback: 60s
        return 60

    now_utc = datetime.now(UTC)
    secs = _tf_seconds(tf)
    epoch = int(now_utc.timestamp())
    cur_open_epoch = (epoch // secs) * secs
    cur_open_ms = cur_open_epoch * 1000

    symbol_bybit = symbol[:-4] + "/USDT" if symbol.endswith("USDT") else symbol
    data = []
    try:
        import ccxt  # type: ignore
        ex = ccxt.bybit({"enableRateLimit": True})
        # Busca até os últimos n_target candles (Bybit normalmente retorna fechados; alguns mercados incluem o em formação)
        lim = max(1, n_target)
        cc = ex.fetch_ohlcv(symbol_bybit, timeframe=tf, limit=lim) or []
        if cc:
            # Garante no máximo n_target candles
            if len(cc) > n_target:
                cc = cc[-n_target:]
            data = [{
                "data": o[0],
                "valor_fechamento": float(o[4]),
                "criptomoeda": symbol,
                "volume_compra": float(o[5] or 0.0),
                "volume_venda": float(o[5] or 0.0),
            } for o in cc]
            if debug:
                print(f"[INFO] Bybit: {len(data)} candles carregados (API)", flush=True)
        # Se o último candle não é o atual, adiciona o preço atual como candle em formação
        need_append_live = True
        if data:
            last_ts = int(data[-1]["data"])
            if last_ts == cur_open_ms:
                need_append_live = False
            elif last_ts > cur_open_ms:
                # Incomum: dados já no futuro do open atual; não adiciona
                need_append_live = False
        if need_append_live and len(data) < n_target:
            try:
                ticker = ex.fetch_ticker(symbol_bybit)
                data.append({
                    "data": cur_open_ms,
                    "valor_fechamento": float(ticker["last"]),
                    "criptomoeda": symbol,
                    "volume_compra": 0.0,
                    "volume_venda": 0.0,
                })
                if debug:
                    print(f"[INFO] Adicionado preço atual do ticker: {ticker['last']}", flush=True)
            except Exception as e:
                if debug:
                    print(f"[WARN] Não foi possível adicionar preço atual: {e}", flush=True)
        # Garante exatamente n_target no máximo (fechados + atual)
        if data and len(data) > n_target:
            data = data[-n_target:]
    except Exception as e:
        if debug:
            print(f"[WARN] Bybit falhou: {e}", flush=True)

    # Fallback: snapshot local
    if not data and os.path.exists("df_log.csv") and os.path.getsize("df_log.csv") > 0:
        try:
            df_local = pd.read_csv("df_log.csv")
            if "data" in df_local.columns:
                df_local["data"] = pd.to_datetime(df_local["data"])
            if debug:
                print("[INFO] Fallback: carregado df_log.csv", flush=True)
            return df_local
        except Exception as e:
            if debug:
                print(f"[WARN] Falha ao ler df_log.csv: {e}", flush=True)

    if not data:
        if debug:
            print(f"[ERR] Sem dados para {symbol} tf={tf}", flush=True)
        return pd.DataFrame()

    df_out = pd.DataFrame(data)
    df_out["data"] = pd.to_datetime(df_out["data"], unit="ms")
    try:
        df_out = calcular_rsi_por_criptomoeda(df_out, window=14)
        df_out = calcular_macd(df_out)
    except Exception as e:
        if debug:
            print(f"[WARN] Indicadores falharam: {e}", flush=True)
    if debug:
        try:
            print(f"[INFO] Total candles retornados: {len(df_out)}", flush=True)
        except Exception:
            pass
    return df_out
SYMBOL_BINANCE = "SOLUSDT"
# Constrói df global na carga, se estiver vazio
if isinstance(df, pd.DataFrame) and df.empty:
    try:
        df = build_df(SYMBOL_BINANCE, INTERVAL, START_DATE, END_DATE, debug=True)
    except Exception as _e:
        print(f"[WARN] build_df falhou: {_e}", flush=True)
        df = pd.DataFrame()


# COMMAND ----------

""" Bloco de métricas intradiárias (legado) removido. """

# COMMAND ----------

"""
DEX (Hyperliquid via ccxt)
"""
import ccxt  # type: ignore

# ATENÇÃO: chaves privadas em código-fonte. Considere usar variáveis
# de ambiente em produção para evitar exposição acidental.
dex_timeout = int(os.getenv("DEX_TIMEOUT_MS", "5000"))
dex = ccxt.hyperliquid({
    "walletAddress": "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
    "privateKey": "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872",
    "enableRateLimit": True,
    "timeout": dex_timeout,
    "options": {"timeout": dex_timeout},
})

# COMMAND ----------

if dex:
    print(f"[DEX] Inicializado | LIVE_TRADING={os.getenv('LIVE_TRADING','0')} | DEX_TIMEOUT_MS={dex_timeout}", flush=True)
    live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
    if live:
        print("[DEX] fetch_balance() iniciando…", flush=True)
        try:
            dex.fetch_balance()
            print("[DEX] fetch_balance() OK", flush=True)
        except Exception as e:
            print(f"[WARN] Falha ao buscar saldo do DEX: {type(e).__name__}: {e}", flush=True)
    else:
        print("[DEX] LIVE_TRADING=0 ⇒ ignorando fetch_balance()", flush=True)

# COMMAND ----------
# =========================
# 🔔 LOGGER (CSV + XLSX em DBFS com workaround /tmp → dbutils.fs.cp)
# =========================
import os
import pandas as pd
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
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
_DISCORD_WEBHOOK = os.getenv(
    "DISCORD_WEBHOOK",
    "https://discord.com/api/webhooks/1411808916316098571/m_qTenLaTMvyf2e1xNklxFP2PVIvrVD328TFyofY1ciCUlFdWetiC-y4OIGLV23sW9vM"
)
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
        print(f"[WARN] HTTP falhou: {type(e).__name__}: {e}", flush=True)
        return None

def _notify_discord(message: str):
    if not _DISCORD_WEBHOOK or "discord.com/api/webhooks" not in _DISCORD_WEBHOOK:
        return
    try:
        resp = _SESSION.post(_DISCORD_WEBHOOK, json={"content": message}, timeout=_HTTP_TIMEOUT)
        if resp.status_code not in (200, 204):
            print(f"[WARN] Discord status {resp.status_code}: {resp.text}", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Falha ao notificar Discord: {type(e).__name__}: {e}", flush=True)

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
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
import numpy as np
import pandas as pd

@dataclass
class GradientConfig:
    # Indicadores
    EMA_SHORT_SPAN: int     = 7
    EMA_LONG_SPAN: int      = 21
    N_BARRAS_GRADIENTE: int = 3           # janela para gradiente
    GRAD_CONSISTENCY: int   = 3           # nº velas com gradiente consistente
    ATR_PERIOD: int         = 14
    VOL_MA_PERIOD: int      = 20

    # Filtros de entrada
    ATR_PCT_MIN: float      = 0.15        # ATR% saudável (min)
    ATR_PCT_MAX: float      = 2.5         # ATR% saudável (max)
    BREAKOUT_K_ATR: float   = 0.25        # banda de rompimento: k*ATR
    NO_TRADE_EPS_K_ATR: float = 0.05      # zona neutra: |EMA7-EMA21| < eps*ATR

    # Saídas por gradiente
    INV_GRAD_BARS: int      = 2           # barras de gradiente oposto p/ sair

    # Execução
    LEVERAGE: int           = 20
    MIN_ORDER_USD: float    = 10.0
    STOP_RISK_PCT: float    = 0.06      # legado: risco % da margem para fallback

    # Cooldown & anti-flip-flop
    COOLDOWN_BARS: int      = 3           # cooldown em velas (prioritário)
    POST_COOLDOWN_CONFIRM: int = 1        # exigir +1 vela válida após cooldown
    COOLDOWN_MINUTOS: int   = 0           # legado; não usado se COOLDOWN_BARS>0
    ANTI_SPAM_SECS: int     = 3

    # Stops/TP
    STOP_ATR_MULT: float    = 1.5         # Stop = 1,5×ATR
    TAKEPROFIT_ATR_MULT: float = 0.0      # 0 desativa
    TRAILING_ATR_MULT: float   = 0.0      # 0 desativa

    # Breakeven trailing legado (mantido opcionalmente)
    BE_TRIGGER_PCT: float   = 0.005       # +0,5% a favor para acionar BE
    BE_OFFSET_PCT: float    = 0.0005      # BE +0,05% (long) / -0,05% (short)


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

        # Estado para cooldown por barras e intenção pós-cooldown
        self._cooldown_until_idx: Optional[int] = None
        self._pending_after_cd: Optional[Dict[str, Any]] = None  # {side, reason, created_idx}
        self._last_seen_bar_idx: Optional[int] = None

    # ---------- config → params (reuso dos cálculos do backtest) ----------
    def _cfg_to_btparams(self):
        try:
            return BacktestParams(
                ema_short=self.cfg.EMA_SHORT_SPAN,
                ema_long=self.cfg.EMA_LONG_SPAN,
                atr_period=self.cfg.ATR_PERIOD,
                vol_ma_period=self.cfg.VOL_MA_PERIOD,
                grad_window=self.cfg.N_BARRAS_GRADIENTE,
                grad_consistency=self.cfg.GRAD_CONSISTENCY,
                atr_pct_min=self.cfg.ATR_PCT_MIN,
                atr_pct_max=self.cfg.ATR_PCT_MAX,
                breakout_k_atr=self.cfg.BREAKOUT_K_ATR,
                no_trade_eps_k_atr=self.cfg.NO_TRADE_EPS_K_ATR,
                cooldown_bars=self.cfg.COOLDOWN_BARS,
                post_cooldown_confirm_bars=self.cfg.POST_COOLDOWN_CONFIRM,
                stop_atr_mult=self.cfg.STOP_ATR_MULT,
                takeprofit_atr_mult=(self.cfg.TAKEPROFIT_ATR_MULT or None),
                trailing_atr_mult=(self.cfg.TRAILING_ATR_MULT or None),
            )
        except Exception:
            # fallback seguro
            return BacktestParams()

    def _compute_indicators_live(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self._cfg_to_btparams()
        return compute_indicators(df, p)

    # ---------- cooldown por barras ----------
    def _bar_index(self, df: pd.DataFrame) -> int:
        return len(df) - 1

    def _cooldown_barras_ativo(self, df: pd.DataFrame) -> bool:
        if self._cooldown_until_idx is None:
            return False
        return self._bar_index(df) < self._cooldown_until_idx

    def _marcar_cooldown_barras(self, df: pd.DataFrame):
        bars = max(0, int(self.cfg.COOLDOWN_BARS or 0))
        if bars <= 0:
            self._cooldown_until_idx = None
            return
        self._cooldown_until_idx = self._bar_index(df) + bars

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
            live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
            if live:
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
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
            sys.stdout.flush()  # Troque _sys por sys

        try:
            self.logger.append_event(evento=evento, **to_send)
            print(f"✅ Logger externo OK: '{evento}' (sem snapshot)")
            return
        except Exception as e2:
            print(f"⚠️ Logger externo falhou (sem snapshot): {type(e2).__name__}: {e2} → tentando stub...")

        try:
            df_stub = pd.DataFrame({"ts": [datetime.now(timezone.utc)]})
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
        live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
        if not live:
            if self.debug:
                print("[DEX] LIVE_TRADING=0 ⇒ _preco_atual indisponível", flush=True)
            raise RuntimeError("LIVE_TRADING desativado")
        try:
            mkts = self.dex.load_markets()
            info = mkts[self.symbol]["info"]
            if info.get("midPx") is not None:
                return float(info["midPx"])
        except Exception:
            pass
        try:
            t = self.dex.fetch_ticker(self.symbol)
            if t and t.get("last"):
                return float(t["last"])
        except Exception as e:
            if self.debug:
                print(f"⚠️ fetch_ticker falhou: {type(e).__name__}: {e}")
        raise RuntimeError("Não consegui obter preço atual (midPx/last).")

    def _posicao_aberta(self) -> Optional[Dict[str, Any]]:
        # Permite desligar chamadas à exchange em ambientes restritos (default off)
        if os.getenv("LIVE_TRADING", "0") not in ("1", "true", "True"):
            return None
        try:
            pos = self.dex.fetch_positions([self.symbol])
            if pos and float(pos[0].get("contracts", 0)) > 0:
                return pos[0]
        except Exception as e:
            if self.debug:
                print(f"⚠️ fetch_positions falhou: {type(e).__name__}: {e}")
        return None

    def _tem_ordem_de_entrada_pendente(self) -> bool:
        try:
            if os.getenv("LIVE_TRADING", "0") not in ("1", "true", "True"):
                return False
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
        return self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until

    def _marcar_cooldown(self):
        self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.cfg.COOLDOWN_MINUTOS)

    def _anti_spam_ok(self, kind: str) -> bool:
        now = datetime.now(timezone.utc)
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
    def _abrir_posicao_com_stop(self, side: str, usd_to_spend: float, df_for_log: pd.DataFrame, atr_last: Optional[float] = None):
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

        # Stop inicial baseado em ATR (preferencial); fallback ao risco por margem legado
        use_atr = (atr_last is not None) and (self.cfg.STOP_ATR_MULT is not None) and (self.cfg.STOP_ATR_MULT > 0)
        if use_atr:
            if self._norm_side(side) == "buy":
                sl_price = price - (self.cfg.STOP_ATR_MULT * float(atr_last))
                sl_side  = "sell"
            else:
                sl_price = price + (self.cfg.STOP_ATR_MULT * float(atr_last))
                sl_side  = "buy"
            if self.debug:
                print(f"🔎 Stop ATR: ATR={atr_last:.6f}, mult={self.cfg.STOP_ATR_MULT} ⇒ stop @ {sl_price:.6f} ({sl_side.upper()})")
        else:
            capital_loss  = usd_to_spend * self.cfg.STOP_RISK_PCT
            loss_per_unit = capital_loss / amount
            if self._norm_side(side) == "buy":
                sl_price = price - loss_per_unit
                sl_side  = "sell"
            else:
                sl_price = price + loss_per_unit
                sl_side  = "buy"
            if self.debug:
                print(f"🔎 Stop risco/margem: margem={usd_to_spend:.2f}, risco={self.cfg.STOP_RISK_PCT*100:.2f}% ⇒ stop @ {sl_price:.6f} ({sl_side.upper()})")

        ordem_stop = self._place_stop(sl_side, amount, sl_price)
        try:
            extra = f"ATRx{self.cfg.STOP_ATR_MULT}" if use_atr else f"risk {self.cfg.STOP_RISK_PCT*100:.2f}%"
            print(f"📉 Stop inicial @ {sl_price:.6f} ({extra})", ordem_stop)
        except Exception:
            print(f"📉 Stop inicial @ {sl_price:.6f}", ordem_stop)

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
            if os.getenv("LIVE_TRADING", "0") not in ("1", "true", "True"):
                return None, None, None
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

        # indicadores e gradiente em %/barra
        df = self._compute_indicators_live(df)
        last = df.iloc[-1]
        last_idx = len(df) - 1
        self._last_seen_bar_idx = last_idx

        # helpers de consistência do gradiente
        g = df["ema_short_grad_pct"].tail(self.cfg.GRAD_CONSISTENCY)
        grad_pos_ok = g.notna().all() and (g > 0).all()
        grad_neg_ok = g.notna().all() and (g < 0).all()

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
            # aplica cooldown por barras para evitar reversão imediata
            self._marcar_cooldown_barras(df)
            self._last_pos_side = None

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

        # Cooldown por barras (prioritário); fallback por minutos (legado)
        if self._cooldown_barras_ativo(df):
            print(f"⛔ Em cooldown de barras ({self.cfg.COOLDOWN_BARS}). Sem novas entradas.")
            self._safe_log("cooldown", df_for_log=df, tipo="info")
            self._last_pos_side = (self._norm_side(pos.get("side")) if pos else None)
            # memoriza intenção durante cooldown
            if not pos:
                can_long = (
                    (last.ema_short > last.ema_long) and grad_pos_ok and
                    (self.cfg.ATR_PCT_MIN <= last.atr_pct <= self.cfg.ATR_PCT_MAX) and
                    (last.valor_fechamento > last.ema_short + self.cfg.BREAKOUT_K_ATR * last.atr) and
                    (last.volume > last.vol_ma)
                )
                can_short = (
                    (last.ema_short < last.ema_long) and grad_neg_ok and
                    (self.cfg.ATR_PCT_MIN <= last.atr_pct <= self.cfg.ATR_PCT_MAX) and
                    (last.valor_fechamento < last.ema_short - self.cfg.BREAKOUT_K_ATR * last.atr) and
                    (last.volume > last.vol_ma)
                )
                if can_long:
                    self._pending_after_cd = {"side": "LONG", "reason": "cooldown_intent_long", "created_idx": last_idx}
                elif can_short:
                    self._pending_after_cd = {"side": "SHORT", "reason": "cooldown_intent_short", "created_idx": last_idx}
            return

        # trailing (ATR opcional) e/ou BE± se houver posição
        if pos:
            # manter BE± legado
            self._maybe_trailing_breakeven_plus(pos, df_for_log=df)

        # entradas (sem posição), respeitando no-trade zone e intenção pós-cooldown
        if not pos:
            # evita qualquer tentativa de ordem se LIVE_TRADING=0
            live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
            if not live:
                print("🧪 LIVE_TRADING=0: avaliando sinais, mas sem abrir posições.")
                self._safe_log("paper_mode", df_for_log=df, tipo="info")
                self._last_pos_side = None
                return
            # no-trade zone
            if abs(float(last.ema_short - last.ema_long)) < (self.cfg.NO_TRADE_EPS_K_ATR * float(last.atr)) or \
               (last.atr_pct < self.cfg.ATR_PCT_MIN) or (last.atr_pct > self.cfg.ATR_PCT_MAX):
                print("🚫 No-Trade Zone: EMAs próximos ou ATR% fora da faixa.")
                self._safe_log("no_trade_zone", df_for_log=df, tipo="info")
                self._last_pos_side = None
                return

            # intenção pós-cooldown: exigir confirmação adicional
            if self._pending_after_cd is not None:
                intent = self._pending_after_cd
                if intent.get("side") == "LONG":
                    can_long = (
                        (last.ema_short > last.ema_long) and grad_pos_ok and
                        (self.cfg.ATR_PCT_MIN <= last.atr_pct <= self.cfg.ATR_PCT_MAX) and
                        (last.valor_fechamento > last.ema_short + self.cfg.BREAKOUT_K_ATR * last.atr) and
                        (last.volume > last.vol_ma)
                    )
                    if can_long:
                        print("✅ Confirmação pós-cooldown LONG")
                        self._abrir_posicao_com_stop("buy", usd_to_spend, df_for_log=df, atr_last=float(last.atr))
                        pos_after = self._posicao_aberta()
                        self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                        self._pending_after_cd = None
                        return
                else:
                    can_short = (
                        (last.ema_short < last.ema_long) and grad_neg_ok and
                        (self.cfg.ATR_PCT_MIN <= last.atr_pct <= self.cfg.ATR_PCT_MAX) and
                        (last.valor_fechamento < last.ema_short - self.cfg.BREAKOUT_K_ATR * last.atr) and
                        (last.volume > last.vol_ma)
                    )
                    if can_short:
                        print("✅ Confirmação pós-cooldown SHORT")
                        self._abrir_posicao_com_stop("sell", usd_to_spend, df_for_log=df, atr_last=float(last.atr))
                        pos_after = self._posicao_aberta()
                        self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                        self._pending_after_cd = None
                        return
                print("⏳ Sinal perdeu/sem confirmação pós-cooldown.")
                self._pending_after_cd = None
                self._last_pos_side = None
                return

            # Entradas normais
            can_long = (
                (last.ema_short > last.ema_long) and grad_pos_ok and
                (self.cfg.ATR_PCT_MIN <= last.atr_pct <= self.cfg.ATR_PCT_MAX) and
                (last.valor_fechamento > last.ema_short + self.cfg.BREAKOUT_K_ATR * last.atr) and
                (last.volume > last.vol_ma)
            )
            can_short = (
                (last.ema_short < last.ema_long) and grad_neg_ok and
                (self.cfg.ATR_PCT_MIN <= last.atr_pct <= self.cfg.ATR_PCT_MAX) and
                (last.valor_fechamento < last.ema_short - self.cfg.BREAKOUT_K_ATR * last.atr) and
                (last.volume > last.vol_ma)
            )
            if can_long:
                print("✅ Sinal LONG (entrada)")
                self._abrir_posicao_com_stop("buy", usd_to_spend, df_for_log=df, atr_last=float(last.atr))
                pos_after = self._posicao_aberta()
                self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                return
            if can_short:
                print("✅ Sinal SHORT (entrada)")
                self._abrir_posicao_com_stop("sell", usd_to_spend, df_for_log=df, atr_last=float(last.atr))
                pos_after = self._posicao_aberta()
                self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                return
            print("⏸ Sem posição e sem sinal.")
            self._safe_log("decisao", df_for_log=df, tipo="info")
            self._last_pos_side = None
            return

        # saídas (com posição)
        lado = self._norm_side(pos.get("side") or pos.get("positionSide"))
        # Saídas por cruzamento de EMA ou inversão sustentada do gradiente
        grad_recent = df["ema_short_grad_pct"].tail(max(2, self.cfg.INV_GRAD_BARS))
        inv_long = grad_recent.notna().all() and (grad_recent <= 0).sum() >= self.cfg.INV_GRAD_BARS
        inv_short = grad_recent.notna().all() and (grad_recent >= 0).sum() >= self.cfg.INV_GRAD_BARS
        if lado == "sell":
            if (last.ema_short > last.ema_long) or inv_short:
                if self.debug:
                    print("🔎 Saída SHORT: cruzamento EMA ou gradiente ≥ 0 por barras sustentadas.")
                self._fechar_posicao(df_for_log=df)
                self._last_pos_side = None
                # aplica cooldown em barras
                self._marcar_cooldown_barras(df)
                return
        elif lado == "buy":
            if (last.ema_short < last.ema_long) or inv_long:
                if self.debug:
                    print("🔎 Saída LONG: cruzamento EMA ou gradiente ≤ 0 por barras sustentadas.")
                self._fechar_posicao(df_for_log=df)
                self._last_pos_side = None
                # aplica cooldown em barras
                self._marcar_cooldown_barras(df)
                return

        print("🔄 Mantendo posição.")
        self._safe_log("decisao", df_for_log=df, tipo="info")
        self._last_pos_side = lado if lado in ("buy", "sell") else None


# COMMAND ----------

# =========================
# 📊 BACKTEST: EMA Gradiente com Máquina de Estados
# =========================
@dataclass
class BacktestParams:
    # Indicadores
    ema_short: int = 7
    ema_long: int = 21
    atr_period: int = 14
    vol_ma_period: int = 20
    grad_window: int = 3           # janelas para regressão linear do EMA curto
    grad_consistency: int = 3      # nº de velas consecutivas com gradiente consistente

    # Filtros
    atr_pct_min: float = 0.15      # em % (ATR% = 100*ATR/close)
    atr_pct_max: float = 2.5
    breakout_k_atr: float = 0.25   # banda de rompimento: k*ATR
    no_trade_eps_k_atr: float = 0.05  # ε = 0,05*ATR (zona neutra entre EMAs)

    # Execução e gerência
    cooldown_bars: int = 3
    post_cooldown_confirm_bars: int = 1  # exigir +1 barra válida após cooldown
    allow_pyramiding: bool = False

    # Saídas
    stop_atr_mult: float = 1.5
    takeprofit_atr_mult: Optional[float] = None  # ex.: 2.0; None desativa
    trailing_atr_mult: Optional[float] = None    # ex.: 1.0; None desativa


def _ensure_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "data" in df.columns:
        df = df.sort_values("data").reset_index(drop=True)
    if "valor_fechamento" not in df.columns:
        raise ValueError("DataFrame precisa ter a coluna 'valor_fechamento'.")
    # Volume: usa 'volume_compra' se existir; senão tenta 'volume'; senão soma compra+venda se disponíveis
    if "volume" not in df.columns:
        if "volume_compra" in df.columns and "volume_venda" in df.columns:
            df = df.copy()
            try:
                df["volume"] = pd.to_numeric(df["volume_compra"], errors="coerce").fillna(0) + \
                                pd.to_numeric(df["volume_venda"], errors="coerce").fillna(0)
            except Exception:
                df["volume"] = pd.to_numeric(df.get("volume_compra", 0), errors="coerce").fillna(0)
        elif "volume_compra" in df.columns:
            df = df.copy()
            df["volume"] = pd.to_numeric(df["volume_compra"], errors="coerce").fillna(0)
        else:
            df = df.copy()
            df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    return df


def compute_indicators(df: pd.DataFrame, p: BacktestParams) -> pd.DataFrame:
    df = _ensure_base_cols(df)
    out = df.copy()
    close = pd.to_numeric(out["valor_fechamento"], errors="coerce")

    # EMAs
    out["ema_short"] = close.ewm(span=p.ema_short, adjust=False).mean()
    out["ema_long"] = close.ewm(span=p.ema_long, adjust=False).mean()

    # ATR clássico
    # Se não houver OHLC, aproximamos TR via deslocamentos do fechamento
    if set(["high", "low", "open"]).issubset(out.columns):
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
    else:
        prev_close = close.shift(1)
        tr = (close - prev_close).abs()
    out["atr"] = tr.rolling(p.atr_period, min_periods=1).mean()
    out["atr_pct"] = (out["atr"] / close) * 100.0

    # Volume média
    out["vol_ma"] = out["volume"].rolling(p.vol_ma_period, min_periods=1).mean()

    # Gradiente EMA curto (slope % por barra via regressão sobre janela)
    def slope_pct(series: pd.Series, win: int) -> float:
        if series.notna().sum() < 2:
            return np.nan
        y = series.dropna().values
        n = min(len(y), win)
        x = np.arange(n, dtype=float)
        ywin = y[-n:]
        a, b = np.polyfit(x, ywin, 1)
        denom = ywin[-1] if ywin[-1] not in (0, np.nan) else (np.nan if ywin[-1] == 0 else np.nan)
        return (a / denom) * 100.0 if denom and not np.isnan(denom) else np.nan

    out["ema_short_grad_pct"] = out["ema_short"].rolling(p.grad_window, min_periods=2).apply(
        lambda s: slope_pct(s, p.grad_window), raw=False
    )
    return out


def _entry_long_condition(row, p: BacktestParams) -> Tuple[bool, str]:
    reasons = []
    conds = []
    # EMA short > EMA long
    c1 = row.ema_short > row.ema_long
    conds.append(c1);  reasons.append("EMA7>EMA21")
    # Gradiente positivo (consistência será checada fora por janelas)
    c2 = row.ema_short_grad_pct > 0
    conds.append(c2);  reasons.append("grad>0")
    # ATR% saudável
    c3 = (row.atr_pct >= p.atr_pct_min) and (row.atr_pct <= p.atr_pct_max)
    conds.append(c3);  reasons.append("ATR% saudável")
    # Rompimento
    c4 = row.valor_fechamento > (row.ema_short + p.breakout_k_atr * row.atr)
    conds.append(c4);  reasons.append("close>EMA7+k*ATR")
    # Volume
    c5 = row.volume > row.vol_ma
    conds.append(c5);  reasons.append("volume>média")
    ok = all(conds)
    return ok, "; ".join([r for r, c in zip(reasons, conds) if c]) if ok else "; ".join([r for r, c in zip(reasons, conds) if not c])


def _entry_short_condition(row, p: BacktestParams) -> Tuple[bool, str]:
    reasons = []
    conds = []
    c1 = row.ema_short < row.ema_long
    conds.append(c1);  reasons.append("EMA7<EMA21")
    c2 = row.ema_short_grad_pct < 0
    conds.append(c2);  reasons.append("grad<0")
    c3 = (row.atr_pct >= p.atr_pct_min) and (row.atr_pct <= p.atr_pct_max)
    conds.append(c3);  reasons.append("ATR% saudável")
    c4 = row.valor_fechamento < (row.ema_short - p.breakout_k_atr * row.atr)
    conds.append(c4);  reasons.append("close<EMA7-k*ATR")
    c5 = row.volume > row.vol_ma
    conds.append(c5);  reasons.append("volume>média")
    ok = all(conds)
    return ok, "; ".join([r for r, c in zip(reasons, conds) if c]) if ok else "; ".join([r for r, c in zip(reasons, conds) if not c])


def _no_trade_zone(row, p: BacktestParams) -> bool:
    return abs(row.ema_short - row.ema_long) < (p.no_trade_eps_k_atr * row.atr) or \
           (row.atr_pct < p.atr_pct_min) or (row.atr_pct > p.atr_pct_max)


def run_state_machine(df: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Executa a máquina de estados sobre o DF e retorna:
    - decisions: DataFrame com colunas [state, action, reason, cooldown]
    - trades: lista de trades com dicts {entry_idx, entry_dt, side, entry_px, atr_at_entry, exit_idx, exit_dt, exit_px, reason_exit}
    Garante exclusão mútua e bloqueia reversões diretas (aplica cooldown).
    """
    dfi = compute_indicators(df, p).reset_index(drop=True)

    states = []
    actions = []
    reasons = []
    cooldown = []

    state = "FLAT"
    cd = 0
    last_side = None  # "LONG" / "SHORT"
    consec_grad_pos = 0
    consec_grad_neg = 0
    pending_entry_after_cd = None  # None or (side, confirmed_bars)

    trades = []
    open_trade = None

    for i, row in dfi.iterrows():
        action = "HOLD"; reason = ""

        # atualizar consistência do gradiente
        g = row.ema_short_grad_pct
        if pd.isna(g):
            consec_grad_pos = 0; consec_grad_neg = 0
        else:
            if g > 0:
                consec_grad_pos += 1; consec_grad_neg = 0
            elif g < 0:
                consec_grad_neg += 1; consec_grad_pos = 0
            else:
                consec_grad_pos = 0; consec_grad_neg = 0

        # cooldown ticking
        if cd > 0:
            cd -= 1

        # No-Trade zone
        if _no_trade_zone(row, p):
            states.append(state); actions.append("NO_TRADE_ZONE"); reasons.append("no-trade zone"); cooldown.append(cd)
            continue

        # volume baixo apenas audita
        # (o filtro de volume já entra no _entry_*_condition)

        # Saídas por inversão sustentada/cross de EMA
        if state in ("LONG", "SHORT"):
            exit_signal = False
            exit_reason = []
            # cruzamento EMA
            if state == "LONG" and (row.ema_short < row.ema_long):
                exit_signal = True; exit_reason.append("EMA7<EMA21")
            if state == "SHORT" and (row.ema_short > row.ema_long):
                exit_signal = True; exit_reason.append("EMA7>EMA21")
            # inversão sustentada do gradiente
            if state == "LONG" and consec_grad_pos == 0 and consec_grad_neg >= 2:
                exit_signal = True; exit_reason.append("grad<=0 por 2+")
            if state == "SHORT" and consec_grad_neg == 0 and consec_grad_pos >= 2:
                exit_signal = True; exit_reason.append("grad>=0 por 2+")

            if exit_signal and open_trade is not None:
                open_trade["exit_idx"] = i
                open_trade["exit_dt"] = dfi["data"].iloc[i] if "data" in dfi.columns else i
                open_trade["exit_px"] = float(row.valor_fechamento)
                open_trade["reason_exit"] = ", ".join(exit_reason)
                trades.append(open_trade)
                open_trade = None
                state = "FLAT"; last_side = None; cd = p.cooldown_bars
                pending_entry_after_cd = None
                action = "EXIT"; reason = ", ".join(exit_reason)
                states.append(state); actions.append(action); reasons.append(reason); cooldown.append(cd)
                continue

        # Stop/TP/Trailing gerenciados no backtest runner (após trades serem montados)

        # Entradas
        if state == "FLAT":
            if cd > 0:
                # cooldown em curso: audita e opcionalmente exige sinal consistente pós-cooldown
                states.append(state); actions.append("COOLDOWN"); reasons.append("em cooldown"); cooldown.append(cd)
                # memoriza intenção de entrada durante cooldown
                if pending_entry_after_cd is None:
                    okL, rL = _entry_long_condition(row, p)
                    okS, rS = _entry_short_condition(row, p)
                    if okL and consec_grad_pos >= p.grad_consistency:
                        pending_entry_after_cd = ("LONG", 0, rL)
                    elif okS and consec_grad_neg >= p.grad_consistency:
                        pending_entry_after_cd = ("SHORT", 0, rS)
                continue

            # se havia intenção, exigir confirmação extra
            if pending_entry_after_cd is not None:
                side_intent, conf_bars, rIntent = pending_entry_after_cd
                if side_intent == "LONG":
                    ok, rr = _entry_long_condition(row, p)
                    ok = ok and (consec_grad_pos >= p.grad_consistency)
                else:
                    ok, rr = _entry_short_condition(row, p)
                    ok = ok and (consec_grad_neg >= p.grad_consistency)
                if ok:
                    conf_bars += 1
                    if conf_bars >= p.post_cooldown_confirm_bars:
                        # abre
                        state = side_intent
                        last_side = side_intent
                        open_trade = {
                            "entry_idx": i,
                            "entry_dt": dfi["data"].iloc[i] if "data" in dfi.columns else i,
                            "side": side_intent,
                            "entry_px": float(row.valor_fechamento),
                            "atr_at_entry": float(row.atr),
                            "reason_entry": f"cooldown_confirm: {rIntent}"
                        }
                        action = f"ENTER_{side_intent}"; reason = open_trade["reason_entry"]
                        pending_entry_after_cd = None
                    else:
                        pending_entry_after_cd = (side_intent, conf_bars, rIntent)
                        action = "WAIT_CONFIRM"; reason = f"confirmação {conf_bars}/{p.post_cooldown_confirm_bars}"
                else:
                    pending_entry_after_cd = None
                    action = "HOLD"; reason = "sinal perdeu validade pós-cooldown"
                states.append(state); actions.append(action); reasons.append(reason); cooldown.append(cd)
                continue

            # fluxos normais (sem cooldown)
            okL, rL = _entry_long_condition(row, p)
            okS, rS = _entry_short_condition(row, p)
            if okL and consec_grad_pos >= p.grad_consistency:
                state = "LONG"; last_side = "LONG"
                open_trade = {
                    "entry_idx": i,
                    "entry_dt": dfi["data"].iloc[i] if "data" in dfi.columns else i,
                    "side": "LONG",
                    "entry_px": float(row.valor_fechamento),
                    "atr_at_entry": float(row.atr),
                    "reason_entry": rL
                }
                action = "ENTER_LONG"; reason = rL
            elif okS and consec_grad_neg >= p.grad_consistency:
                state = "SHORT"; last_side = "SHORT"
                open_trade = {
                    "entry_idx": i,
                    "entry_dt": dfi["data"].iloc[i] if "data" in dfi.columns else i,
                    "side": "SHORT",
                    "entry_px": float(row.valor_fechamento),
                    "atr_at_entry": float(row.atr),
                    "reason_entry": rS
                }
                action = "ENTER_SHORT"; reason = rS
            else:
                # Motivos de invalidação detalhados
                inval = []
                if not okL:
                    inval.append(f"LONG inval: {rL}")
                if okL and consec_grad_pos < p.grad_consistency:
                    inval.append("LONG inval: consistência gradiente insuficiente")
                if not okS:
                    inval.append(f"SHORT inval: {rS}")
                if okS and consec_grad_neg < p.grad_consistency:
                    inval.append("SHORT inval: consistência gradiente insuficiente")
                action = "HOLD"; reason = "; ".join(inval) if inval else "regras não atendidas"

        # Ignorar sinais contrários quando em posição
        states.append(state); actions.append(action); reasons.append(reason); cooldown.append(cd)

    decisions = pd.DataFrame({
        "state": states, "action": actions, "reason": reasons, "cooldown": cooldown
    })

    return {"decisions": decisions, "trades": trades, "dfi": dfi}


def _apply_exits_and_equity(trades: list, dfi: pd.DataFrame, p: BacktestParams) -> pd.DataFrame:
    # Constrói DF de trades com SL/TP/Trailing e métricas por trade
    rows = []
    for t in trades:
        side = t["side"]
        e_idx = t["entry_idx"]
        e_px = t["entry_px"]
        atr0 = t["atr_at_entry"]
        stop = e_px - p.stop_atr_mult * atr0 if side == "LONG" else e_px + p.stop_atr_mult * atr0
        take = None
        if p.takeprofit_atr_mult is not None:
            take = e_px + p.takeprofit_atr_mult * atr0 if side == "LONG" else e_px - p.takeprofit_atr_mult * atr0

        # percorre barras até exit_idx se já setado (sinal inverso) ou até fim
        exit_idx = t.get("exit_idx", None)
        reason_exit = t.get("reason_exit", "")
        trail = None
        for j in range(e_idx + 1, (exit_idx if exit_idx is not None else len(dfi))):
            px = float(dfi["valor_fechamento"].iloc[j])
            atrj = float(dfi["atr"].iloc[j])
            # trailing
            if p.trailing_atr_mult is not None:
                if side == "LONG":
                    trail = max(trail or -np.inf, px - p.trailing_atr_mult * atrj)
                    stop = max(stop, trail)
                else:
                    trail = min(trail or np.inf, px + p.trailing_atr_mult * atrj)
                    stop = min(stop, trail)
            # Checa SL/TP a preço de fechamento (aprox)
            if side == "LONG" and px <= stop:
                exit_idx = j; reason_exit = (reason_exit + ", " if reason_exit else "") + "stop"
                break
            if side == "SHORT" and px >= stop:
                exit_idx = j; reason_exit = (reason_exit + ", " if reason_exit else "") + "stop"
                break
            if take is not None:
                if side == "LONG" and px >= take:
                    exit_idx = j; reason_exit = (reason_exit + ", " if reason_exit else "") + "take"
                    break
                if side == "SHORT" and px <= take:
                    exit_idx = j; reason_exit = (reason_exit + ", " if reason_exit else "") + "take"
                    break

        if exit_idx is None:
            exit_idx = len(dfi) - 1
            reason_exit = reason_exit or "eod"

        x_px = float(dfi["valor_fechamento"].iloc[exit_idx])
        ret = (x_px - e_px) / e_px if side == "LONG" else (e_px - x_px) / e_px
        rows.append({
            "entry_idx": e_idx,
            "exit_idx": exit_idx,
            "entry_dt": t.get("entry_dt"),
            "exit_dt": dfi["data"].iloc[exit_idx] if "data" in dfi.columns else exit_idx,
            "side": side,
            "entry_px": e_px,
            "exit_px": x_px,
            "atr_at_entry": atr0,
            "reason_entry": t.get("reason_entry", ""),
            "reason_exit": reason_exit,
            "ret": ret,
            "atr_pct_entry": float(dfi["atr_pct"].iloc[e_idx])
        })

    return pd.DataFrame(rows)


def _metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    r = trades_df["ret"].values
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = (wins / losses) if losses > 0 else np.inf
    win_rate = (r > 0).mean() * 100.0
    # equity curve
    eq = (1 + trades_df["ret"]).cumprod()
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    sharpe = (np.mean(r) / (np.std(r) + 1e-12)) * np.sqrt(len(r)) if len(r) > 1 else 0.0
    return {
        "trades": int(len(r)),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "max_dd": float(dd),
        "sharpe": float(sharpe),
    }


def backtest_ema_gradient(df: pd.DataFrame, params: Optional[BacktestParams] = None,
                          audit_csv_path: Optional[str] = None) -> Dict[str, Any]:
    p = params or BacktestParams()
    rs = run_state_machine(df, p)
    decisions, trades, dfi = rs["decisions"], rs["trades"], rs["dfi"]

    # Valida exclusão mútua e sem reversão direta
    # Reconstrói estado por actions garantindo que nunca haja LONG e SHORT simultâneos
    cur = "FLAT"; prev = None
    for i, a in enumerate(decisions["action"].tolist()):
        prev = cur
        if a == "ENTER_LONG":
            assert cur == "FLAT", f"Entrada LONG fora de FLAT na barra {i}"
            cur = "LONG"
        elif a == "ENTER_SHORT":
            assert cur == "FLAT", f"Entrada SHORT fora de FLAT na barra {i}"
            cur = "SHORT"
        elif a in ("EXIT",):
            cur = "FLAT"
        # proibição reversão direta é garantida por cooldown exigir FLAT e cd>0

    trades_df = _apply_exits_and_equity(trades, dfi, p)

    # Métricas globais
    metrics_all = _metrics(trades_df)

    # Métricas por regime de volatilidade: dentro vs fora da faixa saudável
    inside = trades_df[trades_df["atr_pct_entry"].between(p.atr_pct_min, p.atr_pct_max)]
    outside = trades_df[~trades_df.index.isin(inside.index)]
    metrics_inside = _metrics(inside)
    metrics_outside = _metrics(outside)

    # Auditoria opcional
    if audit_csv_path:
        aud = decisions.copy()
        if "data" in dfi.columns:
            aud["data"] = dfi["data"].values
        aud.to_csv(audit_csv_path, index=False)

    return {
        "decisions": decisions,
        "trades": trades_df,
        "metrics": {
            "all": metrics_all,
            "atr_inside": metrics_inside,
            "atr_outside": metrics_outside,
        },
        "params": p,
    }


# DBTITLE 1,principal
# =========================
# 🔧 INSTÂNCIA E EXECUÇÃO
# =========================

SYMBOL_HL = "SOL/USDC:USDC"  # Ajuste para o formato aceito pelo Hyperliquid

if __name__ == "__main__":
    # Compat: alias para versões antigas que esperam EMAGradientATRStrategy
    EMAGradientATRStrategy = EMAGradientStrategy  # type: ignore

    def executar_estrategia(df_in: pd.DataFrame, dex_in, trade_logger_in: TradeLogger,
                            usd_to_spend: float = 10.0, loop: bool = True, sleep_seconds: int = 60):
        """Compatível com chamadas antigas (Render.com)."""
        print(f"[MODE] LIVE_TRADING={os.getenv('LIVE_TRADING', '0')} | DEX_TIMEOUT_MS={os.getenv('DEX_TIMEOUT_MS', '5000')}")
        if not isinstance(df_in, pd.DataFrame) or df_in.empty:
            print("DataFrame de candles está vazio ou inválido."); return
        strat_local = EMAGradientStrategy(dex=dex_in, symbol=SYMBOL_HL, logger=trade_logger_in, debug=True)
        try:
            strat_local.step(df_in, usd_to_spend=usd_to_spend)
        except Exception as e:
            print(f"Erro ao executar a estratégia: {type(e).__name__}: {e}")
        if not loop:
            return
        import time as _t
        iter_count = 0
        while True:
            iter_count += 1
            # Heartbeat
            try:
                now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                last_ts = None
                if "data" in df_in.columns and len(df_in) > 0:
                    try:
                        last_ts = str(df_in["data"].iloc[-1])
                    except Exception:
                        last_ts = str(df_in.index[-1])
                live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
                print(f"[HB] {now_utc} | iter={iter_count} | df_len={len(df_in)} | last_ts={last_ts} | live={int(live)}", flush=True)
            except Exception:
                pass
            try:
                strat_local.step(df_in, usd_to_spend=usd_to_spend)
            except Exception as e:
                print(f"Erro ao executar a estratégia: {type(e).__name__}: {e}")
            _t.sleep(max(1, int(sleep_seconds)))

    # Instancia o logger de trades
    trade_logger = TradeLogger(df.columns if isinstance(df, pd.DataFrame) else [])
    # Chama o executor compatível (mantém comportamento anterior)
    executar_estrategia(df, dex, trade_logger)
