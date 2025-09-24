print("\n========== INÍCIO DO BLOCO: HISTÓRICO DE TRADES ==========", flush=True)


def _log_global(section: str, message: str, level: str = "INFO") -> None:
    """Formato padrão para logs fora das classes."""
    print(f"[{level}] [{section}] {message}", flush=True)

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
import math
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


class MarketDataUnavailable(Exception):
    """Sinaliza indisponibilidade temporária de candles para um ativo/timeframe."""
    pass

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
    _log_global("BINANCE", f"exchangeInfo falhou ({last_err})", level="WARN")
    return []

# Função para buscar os dados da criptomoeda
# Aceita datetime diretamente
def get_binance_data(symbol, interval, start_date, end_date):
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    all_data = []
    current_start = start_timestamp
    max_candles = 50  # Limite máximo de candles por ativo
    candles_fetched = 0
    while current_start < end_timestamp and candles_fetched < max_candles:
        url = f"{BASE_URL}klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_timestamp,
            "limit": min(1000, max_candles - candles_fetched)
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            candles_fetched += len(data)
            if candles_fetched >= max_candles:
                break
            current_start = int(data[-1][0]) + 1
        else:
            _log_global("BINANCE", f"Erro ao buscar dados da API para {symbol}: {response.status_code}", level="ERROR")
            break
    # Garante que retorna no máximo 50 candles
    all_data = all_data[-max_candles:]
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
    n_target = 20
    if target_candles is not None:
        n_target = max(1, int(target_candles))
    else:
        env_target = int(os.getenv("TARGET_CANDLES", "0"))
        if env_target > 0:
            n_target = max(1, env_target)

    if debug:
        _log_global("DATA", f"Iniciando build_df symbol={symbol} tf={tf} alvo={n_target}")

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
    ex = None

    # Primeiro tenta Binance
    try:
        candles_needed = n_target
        start_dt = datetime.fromtimestamp(cur_open_epoch - (candles_needed - 1) * secs, UTC)
        end_dt = now_utc
        if debug:
            _log_global("BINANCE_VISION", "Buscando candles recentes (prioridade)")
        bdata = get_binance_data(symbol, tf, start_dt, end_dt)
        if bdata:
            data = bdata[-n_target:]
            if debug:
                _log_global("BINANCE_VISION", f"{len(data)} candles carregados (prioridade)")
    except Exception as e:
        if debug:
            _log_global("BINANCE_VISION", f"Falhou ao buscar prioridade: {type(e).__name__}: {e}", level="WARN")

    # Fallback: Bybit
    if not data:
        try:
            import ccxt  # type: ignore
            ex = ccxt.bybit({
                "enableRateLimit": True,
                "timeout": int(os.getenv("BYBIT_TIMEOUT_MS", "5000")),
                "options": {"timeout": int(os.getenv("BYBIT_TIMEOUT_MS", "5000"))},
            })
            lim = max(1, n_target)
            cc = []
            last_err = None
            for attempt in range(2):
                try:
                    cc = ex.fetch_ohlcv(symbol_bybit, timeframe=tf, limit=lim) or []
                    break
                except Exception as e:
                    last_err = e
                    if debug:
                        _log_global("BYBIT", f"fetch_ohlcv tentativa {attempt+1} falhou: {type(e).__name__}: {e}", level="WARN")
                    _time.sleep(0.3)
            if cc:
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
                    _log_global("BYBIT", f"{len(data)} candles carregados (fallback)")
            else:
                if debug:
                    _log_global("BYBIT", f"Nenhum candle retornado (último erro: {last_err})", level="WARN")
        except Exception as e:
            if debug:
                _log_global("BYBIT", f"Exceção geral: {type(e).__name__}: {e}", level="WARN")

    if data:
        last_ts = int(data[-1]["data"])
        if last_ts != cur_open_ms:
            live_price = None
            if ex is not None:
                try:
                    ticker = ex.fetch_ticker(symbol_bybit)
                    if ticker and ticker.get("last") is not None:
                        live_price = float(ticker["last"])
                        if debug:
                            _log_global("BYBIT", f"Candle em formação anexado via ticker price={live_price}")
                except Exception as e:
                    if debug:
                        _log_global("BYBIT", f"Ticker Bybit indisponível para candle em formação: {type(e).__name__}: {e}", level="DEBUG")
            if live_price is None:
                try:
                    resp = requests.get(
                        f"{BASE_URL}ticker/price",
                        params={"symbol": symbol},
                        timeout=int(os.getenv("BINANCE_TIMEOUT", "10")),
                    )
                    if resp.status_code == 200:
                        payload = resp.json()
                        price_val = payload.get("price") if isinstance(payload, dict) else None
                        if price_val is not None:
                            live_price = float(price_val)
                            if debug:
                                _log_global("BINANCE", f"Candle em formação anexado via ticker price={live_price}")
                except Exception as e:
                    if debug:
                        _log_global("BINANCE", f"Falha ao buscar ticker atual: {type(e).__name__}: {e}", level="DEBUG")
            if live_price is not None:
                data.append({
                    "data": cur_open_ms,
                    "valor_fechamento": float(live_price),
                    "criptomoeda": symbol,
                    "volume_compra": 0.0,
                    "volume_venda": 0.0,
                })
                if len(data) > n_target:
                    data = data[-n_target:]
    if not data:
        if debug:
            _log_global("DATA", f"Nenhum dado encontrado para {symbol} tf={tf}", level="ERROR")
        raise MarketDataUnavailable(f"sem dados para {symbol} tf={tf}")

    df_out = pd.DataFrame(data)
    df_out["data"] = pd.to_datetime(df_out["data"], unit="ms")
    try:
        df_out = calcular_rsi_por_criptomoeda(df_out, window=14)
        df_out = calcular_macd(df_out)
    except Exception as e:
        if debug:
            _log_global("INDICATORS", f"Falha ao calcular indicadores: {e}", level="WARN")
    if debug:
        try:
            _log_global("DATA", f"Total candles retornados: {len(df_out)}")
        except Exception:
            pass
    return df_out
SYMBOL_BINANCE = "BTCUSDT"
# Constrói df global na carga, se estiver vazio
if isinstance(df, pd.DataFrame) and df.empty:
    try:
        df = build_df(SYMBOL_BINANCE, INTERVAL, START_DATE, END_DATE, debug=True)
    except Exception as _e:
        _log_global("DATA", f"build_df falhou: {_e}", level="WARN")
        df = pd.DataFrame()


# COMMAND ----------

""" Bloco de métricas intradiárias (legado) removido. """

# COMMAND ----------

"""
DEX (Hyperliquid via ccxt)
"""
import ccxt  # type: ignore

dex_timeout = int(os.getenv("DEX_TIMEOUT_MS", "5000"))
# Mantém apenas a carteira fixa
WALLET_TRADINGV4 = "0x5ff0f14d577166f9ede3d9568a423166be61ea9d"
_priv_env = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if not _priv_env:
    msg = (
        "Credenciais da Hyperliquid ausentes: HYPERLIQUID_PRIVATE_KEY. "
        "Defina a variável de ambiente obrigatória antes de executar."
    )
    _log_global("DEX", msg, level="ERROR")
    raise RuntimeError(msg)

dex = ccxt.hyperliquid({
    "walletAddress": WALLET_TRADINGV4,
    "privateKey": _priv_env,
    "enableRateLimit": True,
    "timeout": dex_timeout,
    "options": {"timeout": dex_timeout},
})

# Segundo DEX (racional inverso) com credenciais distintas
if dex:
    _log_global("DEX", f"Inicializado | LIVE_TRADING={os.getenv('LIVE_TRADING','0')} | TIMEOUT_MS={dex_timeout}")
    live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
    if live:
        _log_global("DEX", "fetch_balance() iniciando…")
        try:
            dex.fetch_balance()
            _log_global("DEX", "fetch_balance() OK")
        except Exception as e:
            _log_global("DEX", f"Falha ao buscar saldo: {type(e).__name__}: {e}", level="WARN")
    else:
        _log_global("DEX", "LIVE_TRADING=0 ⇒ ignorando fetch_balance()", level="DEBUG")

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
            # Suprime print barulhento "Histórico atualizado" a cada evento
        except Exception as e:
            _log_global(
                "LOGGER",
                f"XLSX não atualizado ({type(e).__name__}: {e}). CSV disponível em {os.path.abspath(self.csv_path)}",
                level="WARN",
            )

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
        _log_global("HTTP", f"Requisição falhou: {type(e).__name__}: {e}", level="WARN")
        return None

def _notify_discord(message: str):
    if not _DISCORD_WEBHOOK or "discord.com/api/webhooks" not in _DISCORD_WEBHOOK:
        return
    try:
        resp = _SESSION.post(_DISCORD_WEBHOOK, json={"content": message}, timeout=_HTTP_TIMEOUT)
        if resp.status_code not in (200, 204):
            _log_global("DISCORD", f"Status {resp.status_code}: {resp.text}", level="WARN")
    except Exception as e:  # pragma: no cover
        _log_global("DISCORD", f"Falha ao notificar: {type(e).__name__}: {e}", level="WARN")

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
from typing import Optional, Dict, Any, Tuple, List
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
    STOP_LOSS_CAPITAL_PCT: float = 0.05  # 5% da margem como stop
    TAKE_PROFIT_CAPITAL_PCT: float = 0.10   # take profit fixo em 10% da margem
    MAX_LOSS_ABS_USD: float    = 0.05     # limite absoluto de perda por posição

    # down & anti-flip-flop
    COOLDOWN_BARS: int      = 0           # cooldown por velas desativado (usar tempo)
    POST_COOLDOWN_CONFIRM: int = 0        # confirmações pós-cooldown desativadas
    COOLDOWN_MINUTOS: int   = 30          # tempo mínimo entre entradas após saída
    ANTI_SPAM_SECS: int     = 3
    MIN_HOLD_BARS: int      = 1           # não sair na mesma vela da entrada

    # Stops/TP
    STOP_ATR_MULT: float    = 0.0         # desativado (uso por % da margem)
    TAKEPROFIT_ATR_MULT: float = 0.0      # desativado
    TRAILING_ATR_MULT: float   = 0.0      # desativado
    ENABLE_TRAILING_STOP: bool = False    # trailing stop desativado

    # Breakeven trailing legado (mantido opcionalmente)
    BE_TRIGGER_PCT: float   = 0.0
    BE_OFFSET_PCT: float    = 0.0


@dataclass
class AssetSetup:
    name: str
    data_symbol: str
    hl_symbol: str
    leverage: int
    stop_pct: float
    take_pct: float
    usd_env: Optional[str] = None


ASSET_SETUPS: List[AssetSetup] = [
    AssetSetup("BTC-USD", "BTCUSDT", "BTC/USDC:USDC", 40, 0.05, 0.10, usd_env="USD_PER_TRADE_BTC"),
    AssetSetup("SOL-USD", "SOLUSDT", "SOL/USDC:USDC", 20, 0.05, 0.10, usd_env="USD_PER_TRADE_SOL"),
    AssetSetup("ETH-USD", "ETHUSDT", "ETH/USDC:USDC", 25, 0.05, 0.10, usd_env="USD_PER_TRADE_ETH"),
    AssetSetup("HYPE-USD", "HYPEUSDT", "HYPE/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_HYPE"),
    AssetSetup("XRP-USD", "XRPUSDT", "XRP/USDC:USDC", 20, 0.05, 0.10, usd_env="USD_PER_TRADE_XRP"),
    AssetSetup("DOGE-USD", "DOGEUSDT", "DOGE/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_DOGE"),
    AssetSetup("AVAX-USD", "AVAXUSDT", "AVAX/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_AVAX"),
    AssetSetup("ENA-USD", "ENAUSDT", "ENA/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_ENA"),
    AssetSetup("BNB-USD", "BNBUSDT", "BNB/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_BNB"),
    AssetSetup("SUI-USD", "SUIUSDT", "SUI/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_SUI"),
    AssetSetup("ADA-USD", "ADAUSDT", "ADA/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_ADA"),
    AssetSetup("PUMP-USD", "PUMPUSDT", "PUMP/USDC:USDC", 5, 0.05, 0.10, usd_env="USD_PER_TRADE_PUMP"),
    AssetSetup("AVNT-USD", "AVNTUSDT", "AVNT/USDC:USDC", 5, 0.05, 0.10, usd_env="USD_PER_TRADE_AVNT"),
    AssetSetup("LINK-USD", "LINKUSDT", "LINK/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_LINK"),
    AssetSetup("WLD-USD", "WLDUSDT", "WLD/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_WLD"),
    AssetSetup("AAVE-USD", "AAVEUSDT", "AAVE/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_AAVE"),
    AssetSetup("CRV-USD", "CRVUSDT", "CRV/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_CRV"),
    AssetSetup("LTC-USD", "LTCUSDT", "LTC/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_LTC"),
    AssetSetup("NEAR-USD", "NEARUSDT", "NEAR/USDC:USDC", 10, 0.05, 0.10, usd_env="USD_PER_TRADE_NEAR"),
]


try:
    from trading import BacktestParams
    def compute_indicators(df, params): return df  # TODO: substitua por implementação real se necessário
except ImportError:
    class BacktestParams:
        def __init__(self, **kwargs): pass
    def compute_indicators(df, params): return df

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
        self._entry_bar_idx: Optional[int] = None
        self._entry_bar_time: Optional[pd.Timestamp] = None

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
        # Cooldown por barras (robusto a janela deslizante)
        self._cd_bars_left: Optional[int] = None
        self._cd_last_bar_time: Optional[pd.Timestamp] = None
        self._cd_last_seen_idx: Optional[int] = None

        # Controle das ordens de proteção
        self._last_stop_order_id: Optional[str] = None
        self._last_take_order_id: Optional[str] = None
        self._trail_max_gain_pct: Optional[float] = None
        self._last_stop_order_px: Optional[float] = None
        self._last_take_order_px: Optional[float] = None
        self._last_price_snapshot: Optional[float] = None

    def _log(self, message: str, level: str = "INFO") -> None:
        prefix = f"{self.symbol}" if self.symbol else "STRAT"
        print(f"[{level}] [{prefix}] {message}", flush=True)

    def _protection_prices(self, entry_price: float, side: str) -> Tuple[float, float]:
        if entry_price <= 0:
            raise ValueError("entry_price deve ser positivo")
        norm_side = self._norm_side(side)
        if norm_side not in ("buy", "sell"):
            raise ValueError("side inválido para proteção")
        risk_ratio = float(self.cfg.STOP_LOSS_CAPITAL_PCT) / float(self.cfg.LEVERAGE)
        reward_ratio = float(self.cfg.TAKE_PROFIT_CAPITAL_PCT) / float(self.cfg.LEVERAGE)
        if norm_side == "buy":
            stop_px = entry_price * (1.0 - risk_ratio)
            take_px = entry_price * (1.0 + reward_ratio)
        else:
            stop_px = entry_price * (1.0 + risk_ratio)
            take_px = entry_price * (1.0 - reward_ratio)
        return stop_px, take_px


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

    def _get_last_bar_time(self, df: pd.DataFrame):
        try:
            if "data" in df.columns and len(df) > 0:
                return pd.to_datetime(df["data"].iloc[-1])
        except Exception:
            pass
        return None

    def _tick_cooldown_barras(self, df: pd.DataFrame):
        # Decrementa cooldown somente quando detecta avanço de barra (timestamp muda)
        if (self._cd_bars_left is None) or (self._cd_bars_left <= 0):
            return
        cur_ts = self._get_last_bar_time(df)
        cur_idx = self._bar_index(df)
        if self._cd_last_bar_time is None:
            # Primeiro tick após iniciar o cooldown: apenas memoriza referência
            self._cd_last_bar_time = cur_ts
            self._cd_last_seen_idx = cur_idx
            return

        bars_adv = 0
        try:
            last_ts_val = None
            cur_ts_val = None
            if self._cd_last_bar_time is not None:
                last_ts_val = pd.Timestamp(self._cd_last_bar_time).value
            if cur_ts is not None:
                cur_ts_val = pd.Timestamp(cur_ts).value

            if (last_ts_val is not None) and ("data" in df.columns):
                series_dt = pd.to_datetime(df["data"], errors="coerce", utc=True)
                if hasattr(series_dt, "asi8"):
                    newer_raw = series_dt.asi8
                else:
                    newer_raw = np.asarray(series_dt, dtype="datetime64[ns]").astype("int64", copy=False)
                newer_mask = newer_raw > last_ts_val
                if newer_mask.any():
                    bars_adv = int(np.unique(newer_raw[newer_mask]).size)

            # Se não conseguimos contar via timestamp mas detectamos avanço, conta pelo menos 1
            if bars_adv == 0 and cur_ts_val is not None and last_ts_val is not None and cur_ts_val > last_ts_val:
                bars_adv = 1

            # Fallback por índice quando sem coluna 'data'
            if bars_adv == 0 and self._cd_last_seen_idx is not None and (cur_idx is not None) and (cur_idx > self._cd_last_seen_idx):
                bars_adv = int(cur_idx - self._cd_last_seen_idx)
        except Exception:
            bars_adv = 0

        if bars_adv <= 0:
            return

        old_left = int(self._cd_bars_left)
        dec = min(old_left, bars_adv)
        self._cd_bars_left = max(0, old_left - dec)
        self._cd_last_bar_time = cur_ts
        self._cd_last_seen_idx = cur_idx
        try:
            if dec > 1:
                self._log(f"Cooldown avançou {dec} barras ({old_left}→{self._cd_bars_left}) última={cur_ts}", level="DEBUG")
            else:
                self._log(f"Cooldown avançou 1 barra ({old_left}→{self._cd_bars_left}) última={cur_ts}", level="DEBUG")
        except Exception:
            pass
        if self._cd_bars_left == 0:
            try:
                self._log("Cooldown de barras concluído.", level="DEBUG")
            except Exception:
                pass
            self._cd_bars_left = None

    def _cooldown_barras_ativo(self, df: pd.DataFrame) -> bool:
        # Novo método: baseado em avanço real de barras por timestamp
        self._tick_cooldown_barras(df)
        if self._cd_bars_left is not None and self._cd_bars_left > 0:
            return True
        # Compatibilidade: se ainda houver estado legado por índice, tenta liberar
        if self._cooldown_until_idx is not None:
            if self._bar_index(df) >= self._cooldown_until_idx:
                self._cooldown_until_idx = None
                return False
            return True
        return False

    def _marcar_cooldown_barras(self, df: pd.DataFrame):
        # Sempre registra cooldown temporal, independente das barras
        if int(self.cfg.COOLDOWN_MINUTOS or 0) > 0:
            self._marcar_cooldown()
        bars = max(0, int(self.cfg.COOLDOWN_BARS or 0))
        if bars <= 0:
            # limpa ambos os modos
            self._cooldown_until_idx = None
            self._cd_bars_left = None
            self._cd_last_bar_time = None
            self._cd_last_seen_idx = None
            return
        # Novo modo: contar por avanço real de barras
        self._cd_bars_left = bars
        self._cd_last_bar_time = self._get_last_bar_time(df)
        self._cd_last_seen_idx = self._bar_index(df)
        try:
            self._log(f"Cooldown iniciado por {bars} barra(s). última_barra={self._cd_last_bar_time}", level="DEBUG")
        except Exception:
            pass
        # Legado: mantém índice apenas como fallback (não confiável com janela deslizante)
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
        fixed = "0x5ff0f14d577166f9ede3d9568a423166be61ea9d"
        for key in ("WALLET_TRADINGV4", "WALLET_ADDRESS", "HYPERLIQUID_WALLET_ADDRESS"):
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
        return fixed

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
                qty = self._position_quantity(pos)
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
            self._log(f"Evento local registrado: {evento} total_local={self._local_events_count}", level="DEBUG")
        except Exception as e:
            self._log(f"Falha ao registrar no buffer local: {type(e).__name__}: {e}", level="ERROR")

        # (D) somente local?
        if self.force_local_log or self.logger is None:
            return

        # (E) tenta logger externo
        try:
            self.logger.append_event(df_snapshot=snap, evento=evento, **to_send)
            if (evento or "").lower() != "decisao":
                self._log(f"Logger externo OK: {evento} (com snapshot)", level="DEBUG")
            return
        except Exception as e1:
            self._log(f"Logger externo falhou (com snapshot): {type(e1).__name__}: {e1}. Retentando sem snapshot.", level="WARN")
            sys.stdout.flush()  # Troque _sys por sys

        try:
            self.logger.append_event(evento=evento, **to_send)
            if (evento or "").lower() != "decisao":
                self._log(f"Logger externo OK: {evento} (sem snapshot)", level="DEBUG")
            return
        except Exception as e2:
            self._log(f"Logger externo falhou (sem snapshot): {type(e2).__name__}: {e2}. Tentando stub.", level="WARN")

        try:
            df_stub = pd.DataFrame({"ts": [datetime.now(timezone.utc)]})
            self.logger.append_event(df_snapshot=df_stub, evento=evento, **to_send)
            self._log(f"Logger externo OK: {evento} (stub)", level="DEBUG")
            return
        except Exception as e3:
            self._log(f"Logger externo falhou (stub): {type(e3).__name__}: {e3}. Mantendo apenas log local.", level="WARN")

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
        self._log(f"Buffer local limpo. Eventos removidos={n}", level="DEBUG")

    def export_local_log_csv(self, path: str = "trade_events_fallback.csv"):
        """Exporta o buffer local para CSV."""
        if not self._local_events:
            self._log("Nenhum evento local disponível para exportar.", level="DEBUG")
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
            self._log(f"Buffer local exportado para {path} ({len(df)} eventos)", level="DEBUG")
            return path
        except Exception as e:
            self._log(f"Falha ao exportar buffer local: {type(e).__name__}: {e}", level="ERROR")
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
                self._log("_preco_atual não disponível com LIVE_TRADING=0", level="DEBUG")
            raise RuntimeError("LIVE_TRADING desativado")
        try:
            t = self.dex.fetch_ticker(self.symbol)
            if t and t.get("last"):
                price = float(t["last"])
                self._last_price_snapshot = price
                return price
            if t and t.get("info"):
                info = t["info"] if isinstance(t["info"], dict) else {}
                px = info.get("indexPx") or info.get("markPx") or info.get("midPx")
                if px is not None:
                    price = float(px)
                    self._last_price_snapshot = price
                    return price
        except Exception as e:
            if self.debug:
                self._log(f"fetch_ticker falhou: {type(e).__name__}: {e}", level="WARN")
        try:
            mkts = self.dex.load_markets(reload=True)
            info = mkts[self.symbol]["info"]
            if info.get("midPx") is not None:
                price = float(info["midPx"])
                self._last_price_snapshot = price
                return price
        except Exception:
            pass
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
                self._log(f"fetch_positions falhou: {type(e).__name__}: {e}", level="WARN")
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

    def _extract_order_id(self, order: Any) -> Optional[str]:
        if not isinstance(order, dict):
            return None
        try:
            oid = order.get("id") or order.get("orderId")
            info = order.get("info") if isinstance(order.get("info"), dict) else {}
            if not oid and info:
                oid = info.get("orderId") or info.get("oid")
                filled = info.get("filled") if isinstance(info.get("filled"), dict) else {}
                if not oid and filled:
                    oid = filled.get("oid")
            return str(oid) if oid else None
        except Exception:
            return None

    def _norm_order_side(self, order: Dict[str, Any]) -> Optional[str]:
        side = order.get("side")
        info = order.get("info") or {}
        params = order.get("params") or {}
        if side is None and isinstance(info, dict):
            side = info.get("side") or info.get("orderSide")
            resting = info.get("resting") or info.get("restingOrder")
            if isinstance(resting, dict):
                side = resting.get("side") or resting.get("b")
        if side is None and isinstance(params, dict):
            side = params.get("side")
        if isinstance(side, bool):
            side = "buy" if side else "sell"
        return self._norm_side(side)

    def _parse_reduce_only_kind_price(self, order: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        info = order.get("info") or {}
        params = order.get("params") or {}
        trigger_candidates = [
            order.get("triggerPrice"), order.get("stopPrice"), order.get("stopLossPrice"),
            info.get("triggerPrice"), info.get("stopPrice"), info.get("stopLossPrice"),
            params.get("triggerPrice") if isinstance(params, dict) else None,
            params.get("stopLossPrice") if isinstance(params, dict) else None,
        ]
        trigger = next((t for t in trigger_candidates if t is not None), None)
        if trigger is None and isinstance(info, dict):
            trigger_info = info.get("trigger") or {}
            trigger = trigger_info.get("triggerPx")
        if trigger is not None:
            try:
                return "stop", float(trigger)
            except (TypeError, ValueError):
                return "stop", None

        price_candidates = [
            order.get("price"),
            info.get("price") if isinstance(info, dict) else None,
            info.get("px") if isinstance(info, dict) else None,
        ]
        if isinstance(info, dict):
            resting = info.get("resting") or {}
            if isinstance(resting, dict):
                price_candidates.append(resting.get("px"))
        if isinstance(params, dict):
            price_candidates.append(params.get("price"))

        for candidate in price_candidates:
            if candidate is None:
                continue
            try:
                return "take", float(candidate)
            except (TypeError, ValueError):
                continue
        return "take", None

    def _is_reduce_only(self, order: Dict[str, Any]) -> bool:
        if not isinstance(order, dict):
            return False
        candidates = [order.get("reduceOnly")]
        info = order.get("info")
        if isinstance(info, dict):
            candidates.append(info.get("reduceOnly"))
            resting = info.get("resting") or {}
            if isinstance(resting, dict):
                candidates.append(resting.get("reduceOnly"))
        params = order.get("params")
        if isinstance(params, dict):
            candidates.append(params.get("reduceOnly"))
        return any(bool(c) for c in candidates)

    def _fetch_reduce_only_orders(self) -> List[Dict[str, Any]]:
        if os.getenv("LIVE_TRADING", "0") not in ("1", "true", "True"):
            return []
        try:
            orders = self.dex.fetch_open_orders(self.symbol)
        except Exception as e:
            if self.debug:
                self._log(f"Falha ao obter open_orders para verificação de proteções: {type(e).__name__}: {e}", level="WARN")
            return []
        result = []
        for order in orders or []:
            if self._is_reduce_only(order):
                result.append(order)
        return result

    def _find_matching_protection(self, kind: str, side: str, price: float) -> Optional[Dict[str, Any]]:
        return self._find_matching_protection_in_orders(kind, side, price, self._fetch_reduce_only_orders())

    def _find_matching_protection_in_orders(self, kind: str, side: str, price: float, orders: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        target_side = self._norm_side(side)
        if not orders:
            return None
        tol = max(1e-8, abs(price) * 1e-5)
        for order in orders:
            order_kind, order_price = self._parse_reduce_only_kind_price(order)
            if order_kind != kind:
                continue
            oside = self._norm_order_side(order)
            if target_side and oside and target_side != oside:
                continue
            if order_price is None:
                continue
            if abs(order_price - price) <= tol:
                return order
        return None

    def _order_effective_price(self, order: Dict[str, Any]) -> Optional[float]:
        if not isinstance(order, dict):
            return None
        _, price = self._parse_reduce_only_kind_price(order)
        if price is not None:
            return price
        candidates = [
            order.get("price"),
        ]
        info = order.get("info") or {}
        if isinstance(info, dict):
            candidates.append(info.get("price"))
            candidates.append(info.get("px"))
        for cand in candidates:
            if cand is None:
                continue
            try:
                return float(cand)
            except (TypeError, ValueError):
                continue
        return None

    def _classify_protection_price(self, order: Dict[str, Any], price: float, entry: float, norm_side: str) -> str:
        info = order.get("info") if isinstance(order, dict) else {}
        if isinstance(info, dict):
            trigger_meta = info.get("trigger") or {}
            if isinstance(trigger_meta, dict):
                tpsl = str(trigger_meta.get("tpsl") or "").lower()
                if tpsl == "sl":
                    return "stop"
                if tpsl == "tp":
                    return "take"
        oid = self._extract_order_id(order)
        if oid and oid == self._last_stop_order_id:
            return "stop"
        if oid and oid == self._last_take_order_id:
            return "take"
        if norm_side == "buy":
            return "stop" if price <= entry else "take"
        else:
            return "stop" if price >= entry else "take"

    def _cancel_protective_orders(self, fetch_backup: bool = False):
        for attr in ("_last_stop_order_id", "_last_take_order_id"):
            oid = getattr(self, attr)
            if oid:
                self._cancel_order_silent(oid)
                setattr(self, attr, None)
        self._last_stop_order_px = None
        self._last_take_order_px = None

        if not fetch_backup:
            return

        try:
            if os.getenv("LIVE_TRADING", "0") not in ("1", "true", "True"):
                return
            for o in self.dex.fetch_open_orders(self.symbol):
                ro = o.get("reduceOnly")
                if ro is None and isinstance(o.get("params"), dict):
                    ro = o["params"].get("reduceOnly")
                if not ro:
                    continue
                typ = (o.get("type") or "").lower()
                has_stop = (
                    o.get("stopPrice")
                    or (o.get("info", {}).get("stopLossPrice"))
                    or (o.get("params", {}).get("stopLossPrice") if isinstance(o.get("params"), dict) else None)
                )
                if typ not in ("limit", "stop", "stop_market") and not has_stop:
                    continue
                oid = o.get("id") or (o.get("info", {}).get("oid"))
                if oid:
                    self._cancel_order_silent(oid)
        except Exception as e:
            if self.debug:
                self._log(f"Falha ao cancelar ordens de proteção remanescentes: {e}", level="WARN")

    # ---------- stop reduceOnly ----------
    def _place_stop(self, side: str, amount: float, stop_price: float,
                    df_for_log: Optional[pd.DataFrame] = None,
                    existing_orders: Optional[List[Dict[str, Any]]] = None):
        amt = self._round_amount(amount)
        px  = float(stop_price)
        # Apenas ordem de gatilho (stop), nunca market
        params = {
            "reduceOnly": True,
            "triggerPrice": px,
            "stopLossPrice": px,
            "trigger": "mark",
        }
        if self.debug:
            self._log(f"Criando STOP gatilho {side.upper()} reduceOnly @ {px:.6f}", level="DEBUG")
        if existing_orders is None:
            existing = self._find_matching_protection("stop", side, px)
        else:
            existing = self._find_matching_protection_in_orders("stop", side, px, existing_orders)
        if existing is not None:
            self._last_stop_order_id = self._extract_order_id(existing)
            self._last_stop_order_px = px
            if self.debug:
                self._log(
                    f"Stop existente reutilizado id={self._last_stop_order_id} price≈{px:.6f}",
                    level="DEBUG",
                )
            return existing
        try:
            # Hyperliquid exige especificar preço base mesmo para stop_market
            ret = self.dex.create_order(self.symbol, "stop_market", side, amt, px, params)
        except Exception as e:
            msg = f"Falha ao criar STOP gatilho: {type(e).__name__}: {e}"
            text = str(e).lower()
            if any(flag in text for flag in ("insufficient", "not enough", "margin", "balance")):
                self._log(msg + " (ignorando por saldo insuficiente)", level="WARN")
                return None
            self._log(msg, level="ERROR")
            raise

        # Diagnóstico do stop criado
        try:
            info = ret if isinstance(ret, dict) else {}
            oid = info.get("id") or info.get("orderId") or (info.get("info", {}) or {}).get("oid")
            typ = info.get("type") or (info.get("info", {}) or {}).get("type")
            inf = info.get("info", {}) or {}
            ro = inf.get("reduceOnly") if isinstance(inf, dict) else None
            sl = inf.get("stopLossPrice") if isinstance(inf, dict) else None
            tp = inf.get("triggerPrice") if isinstance(inf, dict) else None
            self._log(f"STOP criado id={oid} type={typ} reduceOnly={ro} stopLoss={sl} trigger={tp}", level="DEBUG")
            self._last_stop_order_id = str(oid) if oid else None
            self._last_stop_order_px = px
            # Logger opcional
            try:
                self._safe_log("stop_criado", df_for_log, tipo="info", exec_price=px, exec_amount=amt, order_id=str(oid) if oid else None)
            except Exception:
                pass
        except Exception:
            pass
        return ret

    def _place_take_profit(self, side: str, amount: float, target_price: float,
                           df_for_log: Optional[pd.DataFrame] = None,
                           existing_orders: Optional[List[Dict[str, Any]]] = None):
        amt = self._round_amount(amount)
        px = float(target_price)
        params = {"reduceOnly": True}
        if self.debug:
            self._log(f"Criando TAKE PROFIT {side.upper()} reduceOnly @ {px:.6f}", level="DEBUG")
        if existing_orders is None:
            existing = self._find_matching_protection("take", side, px)
        else:
            existing = self._find_matching_protection_in_orders("take", side, px, existing_orders)
        if existing is not None:
            self._last_take_order_id = self._extract_order_id(existing)
            self._last_take_order_px = px
            if self.debug:
                self._log(
                    f"Take profit existente reutilizado id={self._last_take_order_id} price≈{px:.6f}",
                    level="DEBUG",
                )
            return existing
        try:
            ret = self.dex.create_order(self.symbol, "limit", side, amt, px, params)
        except Exception as e:
            msg = f"Falha ao criar TAKE PROFIT: {type(e).__name__}: {e}"
            text = str(e).lower()
            if any(flag in text for flag in ("insufficient", "not enough", "margin", "balance")):
                self._log(msg + " (ignorando por saldo insuficiente)", level="WARN")

                return None
            self._log(msg, level="ERROR")
            raise

        try:
            info = ret if isinstance(ret, dict) else {}

            oid = info.get("id") or info.get("orderId") or (info.get("info", {}) or {}).get("oid")
            typ = info.get("type") or (info.get("info", {}) or {}).get("type")
            inf = info.get("info", {}) or {}
            ro = inf.get("reduceOnly") if isinstance(inf, dict) else None
            sl = inf.get("stopLossPrice") if isinstance(inf, dict) else None
            tp = inf.get("triggerPrice") if isinstance(inf, dict) else None
            self._log(f"TAKE PROFIT criado id={oid} type={typ} reduceOnly={ro} stopLoss={sl} trigger={tp}", level="DEBUG")
            self._last_take_order_id = str(oid) if oid else None
            self._last_take_order_px = px
            # Logger opcional
            try:
                self._safe_log("take_profit_criado", df_for_log, tipo="info", exec_price=px, exec_amount=amt, order_id=str(oid) if oid else None)
            except Exception:
                pass
        except Exception:
            pass
        return ret

    # ---------- execução da estratégia ----------
    def _step(self, df: pd.DataFrame) -> None:
        raise NotImplementedError("Método _step deve ser implementado na estratégia.")

    def step(self, df: pd.DataFrame) -> None:
        try:
            self._step(df)
        except Exception as e:
            self._log(f"Erro na execução da estratégia: {type(e).__name__}: {e}", level="ERROR")
            raise

    # ---------- gerenciamento de posições ----------
    def _position_quantity(self, position: Dict[str, Any]) -> float:
        if not isinstance(position, dict):
            return 0.0
        try:
            return float(position.get("contracts") or position.get("qty") or 0.0)
        except Exception:
            return 0.0

    def _ajustar_stop(self, df: pd.DataFrame, side: str, entry_price: float, amount: float):
        if self.debug:
            self._log(f"ajustar_stop() {side} entry_price={entry_price} amount={amount}", level="DEBUG")
        try:
            if side == "buy":
                self._place_stop("sell", amount, entry_price * (1 - self.cfg.STOP_LOSS_CAPITAL_PCT))
            else:
                self._place_stop("buy", amount, entry_price * (1 + self.cfg.STOP_LOSS_CAPITAL_PCT))
        except Exception as e:
            self._log(f"Erro ao ajustar stop: {type(e).__name__}: {e}", level="ERROR")

    def _trailing_stop(self, df: pd.DataFrame, side: str, entry_price: float, amount: float):
        if self.debug:
            self._log(f"trailing_stop() {side} entry_price={entry_price} amount={amount}", level="DEBUG")
        try:
            if side == "buy":
                # Stop segue o preço com um offset
                self._place_stop("sell", amount, entry_price * (1 - self.cfg.TRAILING_ATR_MULT))
            else:
                # Stop segue o preço com um offset
                self._place_stop("buy", amount, entry_price * (1 + self.cfg.TRAILING_ATR_MULT))
        except Exception as e:
            self._log(f"Erro ao ajustar trailing stop: {type(e).__name__}: {e}", level="ERROR")

    # ---------- lógica da estratégia (exemplo) ----------
    def _step(self, df: pd.DataFrame) -> None:
        # Exemplo: lógica simplificada de entrada/saída
        if df.empty:
            return

        # Última vela fechada
        last_candle = df.iloc[-1]

        # Condições de entrada (exemplo: cruzamento de médias móveis)
        if last_candle["slope_short"] > 0 and last_candle["slope_long"] > 0:
            # Ambas as médias subindo: possível entrada comprada
            if self._last_pos_side != "buy" and not self._tem_ordem_de_entrada_pendente():
                # Executa ordem de compra (exemplo simplificado)
                self._log("Condição de compra atendida.", level="INFO")
                self._last_pos_side = "buy"
                # Aqui você chamaria a função para executar a ordem de compra

        elif last_candle["slope_short"] < 0 and last_candle["slope_long"] < 0:
            # Ambas as médias descendo: possível entrada vendida
            if self._last_pos_side != "sell" and not self._tem_ordem_de_entrada_pendente():
                # Executa ordem de venda (exemplo simplificado)
                self._log("Condição de venda atendida.", level="INFO")
                self._last_pos_side = "sell"
                # Aqui você chamaria a função para executar a ordem de venda

        # Ajuste de stop (exemplo)
        if self._last_pos_side == "buy":
            self._ajustar_stop(df, "buy", last_candle["valor_fechamento"], 0.1)
        elif self._last_pos_side == "sell":
            self._ajustar_stop(df, "sell", last_candle["valor_fechamento"], 0.1)

        # Trailing stop (exemplo)
        if self._last_pos_side == "buy":
            self._trailing_stop(df, "buy", last_candle["valor_fechamento"], 0.1)
        elif self._last_pos_side == "sell":
            self._trailing_stop(df, "sell", last_candle["valor_fechamento"], 0.1)
