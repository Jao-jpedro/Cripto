

# ---- Compat helper to call strategy "step" safely ----
def _safe_strategy_step(strategy_obj, *args, **kwargs):
    """Call the appropriate stepping method on a strategy instance.
    Tries .step(), then .run(), then .process(); falls back to iterate/tick/next/__call__.
    Adds a concise diagnostic log about the class origin.
    """
    try:
        cls = type(strategy_obj)
        mod = getattr(cls, "__module__", "?")
        file_hint = None
        try:
            import sys, inspect
            mod_obj = sys.modules.get(mod)
            if mod_obj is not None:
                try:
                    file_hint = inspect.getsourcefile(mod_obj) or inspect.getfile(mod_obj)
                except Exception:
                    file_hint = getattr(mod_obj, "__file__", None)
        except Exception:
            pass
        print(f"[DEBUG] [STRATEGY] Using {cls.__name__} from module={mod} file={file_hint}", flush=True)
    except Exception:
        pass

    # Preferred names
    for name in ("step", "run", "process", "iterate", "tick", "next", "__call__"):
        meth = getattr(strategy_obj, name, None)
        if callable(meth):
            return meth(*args, **kwargs)

    raise AttributeError(f"{type(strategy_obj).__name__} does not expose a step/run/process method")


# ---- Instance-level compat: ensure the created strategy object has .step() ----
def _ensure_strategy_step_instance(strategy_obj):
    """Attach a 'step' method to the instance if the class doesn't expose one."""
    for name in ("step", "run", "process"):
        if callable(getattr(strategy_obj, name, None)):
            return strategy_obj

    def _inst_step(self, *args, **kwargs):  # pragma: no cover
        for name in ("run", "process", "iterate", "tick", "next", "__call__"):
            m = getattr(self, name, None)
            if callable(m):
                return m(*args, **kwargs)
        raise AttributeError(f"{type(self).__name__} does not expose a step/run/process method")

    try:
        import types
        strategy_obj.step = types.MethodType(_inst_step, strategy_obj)  # bind to instance
    except Exception:
        try:
            strategy_obj.step = _inst_step.__get__(strategy_obj, type(strategy_obj))
        except Exception:
            pass

    try:
        print("[WARN] [COMPAT] Attached instance-level .step() to strategy", flush=True)
    except Exception:
        pass
    return strategy_obj


# ---- Compat helper to call strategy "step" safely ----
def _safe_strategy_step(strategy_obj, *args, **kwargs):
    """Call the appropriate stepping method on a strategy instance.
    Tries .step(), then .run(), then .process(). Adds rich diagnostics to logs.
    """
    try:
        cls = type(strategy_obj)
        mod = getattr(cls, "__module__", "?")
        file_hint = getattr(__import__(mod), "__file__", None) if isinstance(mod, str) and mod in globals() or mod in locals() else None
        print(f"[DEBUG] [STRATEGY] Using {cls.__name__} from module={mod} file={file_hint}", flush=True)
    except Exception:
        pass

    if hasattr(strategy_obj, "step"):
        return strategy_obj.step(*args, **kwargs)
    if hasattr(strategy_obj, "run"):
        return strategy_obj.run(*args, **kwargs)
    if hasattr(strategy_obj, "process"):
        return strategy_obj.process(*args, **kwargs)
    # As last resort, look for a single-callable method
    for alt in ["iterate", "tick", "next", "__call__"]:
        if hasattr(strategy_obj, alt) and callable(getattr(strategy_obj, alt)):
            return getattr(strategy_obj, alt)(*args, **kwargs)
    raise AttributeError(f"{type(strategy_obj).__name__} does not expose a step/run/process method")
# Fixed build: ensures EMAGradientStrategy.step exists and is invoked by the runner.
#codigo com [all] trades=70 win_rate=35.71% PF=1.378 maxDD=-6.593% Sharpe=0.872 

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
            _log_global("BINANCE", f"Erro ao buscar dados da API para {symbol}: {response.status_code}", level="ERROR")
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

# ATENÇÃO: chaves privadas em código-fonte. Considere usar variáveis
# de ambiente em produção para evitar exposição acidental.
dex_timeout = int(os.getenv("DEX_TIMEOUT_MS", "5000"))
# Lê credenciais da env (recomendado) com fallback seguro para dev local
_wallet_env = os.getenv("WALLET_ADDRESS")
_priv_env = os.getenv("HYPERLIQUID_PRIVATE_KEY")
dex = ccxt.hyperliquid({
    "walletAddress": _wallet_env or "0x08183aa09eF03Cf8475D909F507606F5044cBdAB",
    "privateKey": _priv_env or "0x5d0d62a9eff697dd31e491ec34597b06021f88de31f56372ae549231545f0872",
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
    STOP_LOSS_CAPITAL_PCT: float = 0.10  # 10% da margem como stop
    TAKE_PROFIT_CAPITAL_PCT: float = 0.0   # standby (usar trailing para ganhos)
    MAX_LOSS_ABS_USD: float    = 0.10     # limite absoluto de perda por posição
    ENTRY_EPS_HARD: float      = 0.15     # histerese para zona neutra/força mínima
    ENTRY_ALPHA_ATR: float     = 0.15     # força mínima via ATR
    ENTRY_GRAD_PERSIST: int    = 4        # barras para persistência do gradiente
    ENTRY_DEBOUNCE_BARS: int   = 2        # barras de debounce
    ENTRY_RETRY_BARS: int      = 10       # bloqueio após tentativa falha
    ENTRY_SCORE_MIN: int       = 3        # score mínimo
    ENTRY_K_LOW: float         = 1.4      # k quando ATR% baixo
    ENTRY_K_MID: float         = 1.1      # k quando ATR% médio
    ENTRY_K_HIGH: float        = 0.9      # k quando ATR% alto

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

    # Breakeven trailing legado (mantido opcionalmente)
    BE_TRIGGER_PCT: float   = 0.0
    BE_OFFSET_PCT: float    = 0.0


@dataclass
class AssetSetup:
    name: str
    data_symbol: str
    hl_symbol: str
    leverage: int
    stop_pct: float = 0.10
    take_pct: float = 0.30
    usd_env: Optional[str] = None


ASSET_SETUPS: List[AssetSetup] = [
    AssetSetup("BTC-USD", "BTCUSDT", "BTC/USDC:USDC", 40, usd_env="USD_PER_TRADE_BTC"),
    AssetSetup("SOL-USD", "SOLUSDT", "SOL/USDC:USDC", 20, usd_env="USD_PER_TRADE_SOL"),
    AssetSetup("ETH-USD", "ETHUSDT", "ETH/USDC:USDC", 25, usd_env="USD_PER_TRADE_ETH"),
    AssetSetup("HYPE-USD", "HYPEUSDT", "HYPE/USDC:USDC", 10, usd_env="USD_PER_TRADE_HYPE"),
    AssetSetup("XRP-USD", "XRPUSDT", "XRP/USDC:USDC", 20, usd_env="USD_PER_TRADE_XRP"),
    AssetSetup("DOGE-USD", "DOGEUSDT", "DOGE/USDC:USDC", 10, usd_env="USD_PER_TRADE_DOGE"),
    AssetSetup("AVAX-USD", "AVAXUSDT", "AVAX/USDC:USDC", 10, usd_env="USD_PER_TRADE_AVAX"),
    AssetSetup("ENA-USD", "ENAUSDT", "ENA/USDC:USDC", 10, usd_env="USD_PER_TRADE_ENA"),
    AssetSetup("BNB-USD", "BNBUSDT", "BNB/USDC:USDC", 10, usd_env="USD_PER_TRADE_BNB"),
    AssetSetup("SUI-USD", "SUIUSDT", "SUI/USDC:USDC", 10, usd_env="USD_PER_TRADE_SUI"),
    AssetSetup("ADA-USD", "ADAUSDT", "ADA/USDC:USDC", 10, usd_env="USD_PER_TRADE_ADA"),
    AssetSetup("PUMP-USD", "PUMPUSDT", "PUMP/USDC:USDC", 5, usd_env="USD_PER_TRADE_PUMP"),
    AssetSetup("AVNT-USD", "AVNTUSDT", "AVNT/USDC:USDC", 5, usd_env="USD_PER_TRADE_AVNT"),
    AssetSetup("LINK-USD", "LINKUSDT", "LINK/USDC:USDC", 10, usd_env="USD_PER_TRADE_LINK"),
    AssetSetup("WLD-USD", "WLDUSDT", "WLD/USDC:USDC", 10, usd_env="USD_PER_TRADE_WLD"),
    AssetSetup("AAVE-USD", "AAVEUSDT", "AAVE/USDC:USDC", 10, usd_env="USD_PER_TRADE_AAVE"),
    AssetSetup("CRV-USD", "CRVUSDT", "CRV/USDC:USDC", 10, usd_env="USD_PER_TRADE_CRV"),
    AssetSetup("LTC-USD", "LTCUSDT", "LTC/USDC:USDC", 10, usd_env="USD_PER_TRADE_LTC"),
    AssetSetup("NEAR-USD", "NEARUSDT", "NEAR/USDC:USDC", 10, usd_env="USD_PER_TRADE_NEAR"),
]


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
        self._entry_pending_signal: Optional[Dict[str, Any]] = None
        self._entry_last_block_idx: Dict[str, int] = {"LONG": -10**9, "SHORT": -10**9}
        self._entry_hysteresis: Dict[str, bool] = {"LONG": False, "SHORT": False}
        self._cooldown_recent_release: bool = False
        self._last_cooldown_active: bool = False

        def _enforce_max_loss(self, pos: Dict[str, Any], df_for_log: pd.DataFrame = None) -> None:
            """Força saída market se perda > max(10%, $0.10) do preço de entrada."""
            try:
                entry = float(pos.get("entryPrice") or pos.get("entryPx") or 0.0)
                if entry <= 0:
                    return
                side = self._norm_side(pos.get("side") or pos.get("positionSide"))
                px_now = float(self._preco_atual() or 0.0)
                if px_now <= 0:
                    return
                loss_abs = (entry - px_now) if side == "LONG" else (px_now - entry)
                loss_abs = max(0.0, loss_abs)
                loss_pct = (loss_abs / entry) if entry > 0 else 0.0
                threshold_abs = max(0.10, 0.10 * entry)
                if (loss_abs >= threshold_abs) or (loss_pct >= 0.10):
                    self._log(f"MaxLoss atingido: loss_abs={loss_abs:.6f} loss_pct={loss_pct:.3%} => FORÇAR SAÍDA MARKET", level="WARN")
                    self._fechar_posicao_market(df_for_log=df_for_log, reason="max_loss")
            except Exception:
                pass

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

    # ---------- entrada avançada: helpers ----------
    def _volume_robust_flag(self, df: pd.DataFrame, window: int = 50) -> Tuple[bool, float, float, float]:
        try:
            vol_series = pd.to_numeric(df["volume"].tail(window), errors="coerce").dropna()
        except Exception:
            vol_series = pd.Series([], dtype=float)
        if vol_series.empty:
            return False, float('nan'), float('nan'), float('nan')
        median = float(vol_series.median())
        mad = float((vol_series - median).abs().median())
        threshold = median + mad if mad and math.isfinite(mad) else median
        try:
            current = float(df["volume"].iloc[-1])
        except Exception:
            current = float('nan')
        flag = math.isfinite(current) and current > threshold
        return flag, current, median, mad

    def _grad_persistence(self, df: pd.DataFrame, side: str, m: int) -> Tuple[bool, int]:
        try:
            grads = pd.to_numeric(df["ema_short_grad_pct"].tail(m), errors="coerce")
        except Exception:
            grads = pd.Series([], dtype=float)
        if grads.empty or grads.isna().all():
            return False, 0
        if side == "LONG":
            count = int((grads > 0).sum())
        else:
            count = int((grads < 0).sum())
        persist = count >= max(1, m - 1)
        return persist, count

    def _update_hysteresis(self, spread: float, atr: float) -> None:
        alpha = float(self.cfg.ENTRY_ALPHA_ATR or 0.0)
        if atr <= 0:
            self._entry_hysteresis = {"LONG": False, "SHORT": False}
            return
        long_on = self._entry_hysteresis.get("LONG", False)
        short_on = self._entry_hysteresis.get("SHORT", False)
        if spread > alpha * atr:
            long_on = True
        elif spread < -alpha * atr:
            long_on = False
        if -spread > alpha * atr:
            short_on = True
        elif -spread < -alpha * atr:
            short_on = False
        self._entry_hysteresis = {"LONG": long_on, "SHORT": short_on}

    def _adaptive_k(self, atr_pct: float) -> float:
        atr_min = float(self.cfg.ATR_PCT_MIN)
        atr_max = float(self.cfg.ATR_PCT_MAX)
        span = max(1e-6, atr_max - atr_min)
        low_cut = atr_min + 0.25 * span
        high_cut = atr_min + 0.75 * span
        if atr_pct <= low_cut:
            return float(self.cfg.ENTRY_K_LOW)
        if atr_pct >= high_cut:
            return float(self.cfg.ENTRY_K_HIGH)
        return float(self.cfg.ENTRY_K_MID)

    def _entry_side_label(self, side: str) -> str:
        return "LONG" if side.lower() == "buy" else "SHORT"

    def _record_entry_block(self, side: str, idx: int) -> None:
        label = self._entry_side_label(side)
        self._entry_last_block_idx[label] = idx
        if self._entry_pending_signal and self._entry_pending_signal.get("side") == label:
            self._entry_pending_signal = None

    def _entry_one_shot_blocked(self, label: str, idx: int) -> bool:
        last_idx = self._entry_last_block_idx.get(label, -10**9)
        return (idx - last_idx) < int(self.cfg.ENTRY_RETRY_BARS or 0)

    def _log_entry_telemetry(self, side: str, telemetry: Dict[str, Any]) -> None:
        parts = [f"{k}={telemetry[k]}" for k in (
            "ema_spread_in_atr", "grad_persist", "atr_pct", "k_used", "rsi",
            "vol_flag", "score"
        ) if k in telemetry and telemetry[k] is not None]
        reason = telemetry.get("reason")
        msg = f"Entrada {side}: " + ", ".join(parts)
        if reason:
            msg += f" | reason={reason}"
        self._log(msg, level="INFO")

    def _current_rsi(self, df: pd.DataFrame, rsi_df_hourly: Optional[pd.DataFrame]) -> float:
        rsi_val = float('nan')
        try:
            if isinstance(rsi_df_hourly, pd.DataFrame) and not rsi_df_hourly.empty and ("rsi" in rsi_df_hourly.columns):
                df_rsi = rsi_df_hourly
                if "criptomoeda" in df_rsi.columns:
                    df_rsi = df_rsi.loc[df_rsi["criptomoeda"] == self._df_symbol_hint]
                if not df_rsi.empty:
                    val = df_rsi["rsi"].dropna().iloc[-1]
                    rsi_val = float(val)
        except Exception:
            rsi_val = float('nan')
        if math.isnan(rsi_val):
            try:
                if "rsi" in df.columns:
                    rsi_val = float(df["rsi"].dropna().iloc[-1])
            except Exception:
                rsi_val = float('nan')
        if math.isnan(rsi_val):
            try:
                last = df.iloc[-1]
                if hasattr(last, "rsi") and pd.notna(last.rsi):
                    rsi_val = float(last.rsi)
            except Exception:
                rsi_val = float('nan')
        return rsi_val
def _assess_entry(self,
                  metrics: Dict[str, Any],
                  side: str,
                  last_idx: int,
                  rsi_val: float,
                  just_released_cooldown: bool) -> Dict[str, Any]:
    """
    Avalia entrada usando APENAS os racionais do tradingv4_ajustado (L1..L5 / S1..S5):
      LONG:  EMA7>EMA21, grad>0 persistente, ATR% saudável, close>EMA7+k*ATR, volume>média
      SHORT: EMA7<EMA21, grad<0 persistente, ATR% saudável, close<EMA7-k*ATR, volume>média
    Demais controles (debounce, cooldown, pendências, saídas etc.) permanecem como no trading.py.
    """
    ema7 = float(metrics["ema7"])
    ema21 = float(metrics["ema21"])
    spread = float(metrics["spread"])
    atr = float(metrics["atr"])
    atr_pct = float(metrics["atr_pct"])
    price = float(metrics["price"])
    vol_flag = bool(metrics.get("vol_flag", True))  # proxy para (volume > vol_ma)
    grad_persist_flag = bool(metrics["grad_persist_flags"][side])
    grad_count = int(metrics["grad_counts"][side])
    k_used = float(metrics.get("k_used", self.cfg.BREAKOUT_K_ATR))

    atr_min = float(self.cfg.ATR_PCT_MIN)
    atr_max = float(self.cfg.ATR_PCT_MAX)
    eps = float(self.cfg.NO_TRADE_EPS_K_ATR) * atr  # zona neutra

    # Telemetria
    telemetry = {
        "ema7": round(ema7, 6),
        "ema21": round(ema21, 6),
        "spread": round(spread, 6),
        "atr": round(atr, 6),
        "atr_pct": round(atr_pct, 4),
        "price": round(price, 6),
        "ema_spread_in_atr": round(spread / atr, 4) if atr else None,
        "grad_persist": grad_count,
        "k_used": round(k_used, 4),
        "rsi": round(rsi_val, 2) if math.isfinite(rsi_val) else None,
        "vol_flag": vol_flag,
    }

    # Zona neutra (não operar)
    zona_neutra = abs(ema7 - ema21) < eps

    # Breakout triggers
    thr_long = ema7 + k_used * atr
    thr_short = ema7 - k_used * atr
    trigger_long = price > thr_long
    trigger_short = price < thr_short

    reasons: List[str] = []
    if side == "LONG":
        L1 = ema7 > ema21
        L2 = grad_persist_flag
        L3 = atr_min <= atr_pct <= atr_max
        L4 = trigger_long
        L5 = vol_flag
        if not L1: reasons.append("EMA7<=EMA21")
        if not L2: reasons.append(f"gradiente não >0 por {int(self.cfg.ENTRY_GRAD_PERSIST)} velas")
        if not L3: reasons.append(f"ATR% fora [{atr_min},{atr_max}]")
        if not L4: reasons.append(f"close<=EMA7+{k_used}*ATR")
        if not L5: reasons.append("volume<=média")
        tendency = L1
        grad_ok = L2
        force_ok = L3  # mapeado p/ filtro de ATR%
        trigger = trigger_long
        satisfied = int(L1) + int(L2) + int(L3) + int(L4) + int(L5)
    else:
        S1 = ema7 < ema21
        S2 = grad_persist_flag
        S3 = atr_min <= atr_pct <= atr_max
        S4 = trigger_short
        S5 = vol_flag
        if not S1: reasons.append("EMA7>=EMA21")
        if not S2: reasons.append(f"gradiente não <0 por {int(self.cfg.ENTRY_GRAD_PERSIST)} velas")
        if not S3: reasons.append(f"ATR% fora [{atr_min},{atr_max}]")
        if not S4: reasons.append(f"close>=EMA7-{k_used}*ATR")
        if not S5: reasons.append("volume<=média")
        tendency = S1
        grad_ok = S2
        force_ok = S3
        trigger = trigger_short
        satisfied = int(S1) + int(S2) + int(S3) + int(S4) + int(S5)

    if zona_neutra:
        reasons.append("zona_neutra(|EMA7-EMA21|<eps)")

    score = satisfied  # 0..5, compatível com uso corrente
    approved = (len(reasons) == 0)

    reason_text = None if approved else ",".join(reasons)
    telemetry["reason"] = reason_text

    decision = {
        "side": side,
        "approved": approved,
        "score": score,
        "telemetry": telemetry,
        "reason": reason_text,
        "k_used": k_used,
        "tendency": tendency,
        "grad_ok": grad_ok,
        "force_ok": force_ok,
        "rsi_ok": True,   # v4 base não exige RSI
        "trigger": trigger,
        "vol_flag": vol_flag,
    }
    candidate_ready = trigger or (tendency and grad_ok and force_ok)
    if not approved and candidate_ready:
        self._entry_last_block_idx[side] = last_idx
    return decision


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
        # TP desativado
        return None

    def _abrir_posicao_com_stop(self, side: str, usd_to_spend: float, df_for_log: pd.DataFrame, atr_last: Optional[float] = None):
        if self._posicao_aberta():
            self._log("Entrada ignorada: posição já aberta.", level="DEBUG"); return None, None
        if self._tem_ordem_de_entrada_pendente():
            self._log("Entrada ignorada: ordem pendente detectada.", level="WARN"); return None, None
        if not self._anti_spam_ok("open"):
            self._log("Entrada bloqueada pelo anti-spam.", level="DEBUG"); return None, None

        try:
            lev_int = int(self.cfg.LEVERAGE)
        except Exception:
            lev_int = None
        if lev_int and lev_int > 0:
            try:
                self.dex.set_leverage(lev_int, self.symbol, {"marginMode": "isolated"})
                if self.debug:
                    self._log(f"Leverage ajustada para {lev_int}x (isolated)", level="DEBUG")
            except Exception as e:
                self._log(f"Falha ao ajustar leverage isolada: {type(e).__name__}: {e}", level="WARN")

        usd_to_spend = max(usd_to_spend, self.cfg.MIN_ORDER_USD / self.cfg.LEVERAGE)
        price  = self._preco_atual()
        amount = self._round_amount((usd_to_spend * self.cfg.LEVERAGE) / price)

        # Ao abrir nova posição, limpa cooldown temporal
        self._cooldown_until = None

        self._log(
            f"Abrindo {side.upper()} | notional≈${usd_to_spend*self.cfg.LEVERAGE:.2f} amount≈{amount:.6f} px≈{price:.4f}",
            level="INFO",
        )
        ordem_entrada = self.dex.create_order(self.symbol, "market", side, amount, price)
        self._log(f"Resposta create_order: {ordem_entrada}", level="DEBUG")

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

        # Atualiza dados da posição após execução
        fill_price = None
        fill_amount = None
        try:
            if isinstance(ordem_entrada, dict):
                if ordem_entrada.get("average"):
                    fill_price = float(ordem_entrada["average"])
                info_resp = ordem_entrada.get("info") or {}
                if isinstance(info_resp, dict):
                    if info_resp.get("average"):
                        fill_price = float(info_resp["average"])
                    filled = info_resp.get("filled") or {}
                    if isinstance(filled, dict):
                        if filled.get("avgPx"):
                            fill_price = float(filled["avgPx"])
                        if filled.get("totalSz"):
                            fill_amount = float(filled["totalSz"])
                if ordem_entrada.get("amount"):
                    fill_amount = float(ordem_entrada["amount"])
        except Exception:
            pass

        try:
            pos_after_exec = self._posicao_aberta()
        except Exception:
            pos_after_exec = None
        if pos_after_exec:
            try:
                entry_px_cb = float(pos_after_exec.get("entryPrice") or pos_after_exec.get("entryPx") or 0.0)
                if entry_px_cb > 0:
                    fill_price = entry_px_cb
                filled_cb = float(pos_after_exec.get("contracts") or 0.0)
                if filled_cb > 0:
                    fill_amount = filled_cb
            except Exception:
                pass
        if fill_price is None or fill_price <= 0:
            fill_price = price
        if fill_amount is None or fill_amount <= 0:
            fill_amount = amount

        # Guarda índice/tempo da barra de entrada (para hold mínimo)
        try:
            self._entry_bar_idx = (len(df_for_log) - 1) if isinstance(df_for_log, pd.DataFrame) else None
            if isinstance(df_for_log, pd.DataFrame) and "data" in df_for_log.columns and len(df_for_log) > 0:
                self._entry_bar_time = pd.to_datetime(df_for_log["data"].iloc[-1])
        except Exception:
            self._entry_bar_idx = None; self._entry_bar_time = None

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

        self._last_stop_order_id = None
        self._last_stop_order_px = None
        self._last_take_order_id = None
        self._last_take_order_px = None
        self._trail_max_gain_pct = 0.0

        norm_side = self._norm_side(side)
        sl_price, tp_price = self._protection_prices(fill_price, norm_side)
        manage_take = float(self.cfg.TAKE_PROFIT_CAPITAL_PCT or 0.0) > 0.0
        if not manage_take:
            tp_price = None
        sl_side = "sell" if norm_side == "buy" else "buy"
        tp_side = sl_side

        if self.debug:
            if manage_take and tp_price is not None:
                self._log(
                    f"Proteções configuradas | stop={sl_price:.6f} (-{self.cfg.STOP_LOSS_CAPITAL_PCT*100:.1f}% margem) "
                    f"take={tp_price:.6f} (+{self.cfg.TAKE_PROFIT_CAPITAL_PCT*100:.1f}% margem)",
                    level="DEBUG",
                )
            else:
                self._log(
                    f"Proteções configuradas | stop={sl_price:.6f} (-{self.cfg.STOP_LOSS_CAPITAL_PCT*100:.1f}% margem) | take=standby",
                    level="DEBUG",
                )

        ordem_stop = self._place_stop(sl_side, fill_amount, sl_price, df_for_log=df_for_log)
        self._last_stop_order_id = self._extract_order_id(ordem_stop)

        self._last_take_order_id = None
        if manage_take and tp_price is not None:
            ordem_take = self._place_take_profit(tp_side, fill_amount, tp_price, df_for_log=df_for_log)
            self._last_take_order_id = self._extract_order_id(ordem_take)

        self._safe_log(
            "stop_inicial", df_for_log,
            tipo=("long" if norm_side == "buy" else "short"),
            exec_price=sl_price,
            exec_amount=amount
        )

        if manage_take and tp_price is not None:
            self._safe_log(
                "take_profit_inicial", df_for_log,
                tipo=("long" if norm_side == "buy" else "short"),
                exec_price=tp_price,
                exec_amount=amount
            )

        # Diagnóstico: listar ordens abertas reduceOnly
        try:
            if os.getenv("LIVE_TRADING", "0") in ("1", "true", "True"):
                open_orders = self.dex.fetch_open_orders(self.symbol)
                if open_orders:
                    self._log("Ordens reduceOnly ativas:", level="DEBUG")
                    for o in open_orders:
                        ro = o.get("reduceOnly")
                        if ro is None and isinstance(o.get("params"), dict):
                            ro = o["params"].get("reduceOnly")
                        if not ro:
                            continue
                        info = o.get("info", {}) or {}
                        self._log(
                            f"id={o.get('id')} type={o.get('type')} side={o.get('side')} reduceOnly={ro} "
                            f"stopLossPrice={info.get('stopLossPrice')} triggerPrice={info.get('triggerPrice')}",
                            level="DEBUG",
                        )
        except Exception as e:
            self._log(f"Falha ao listar open_orders: {type(e).__name__}: {e}", level="WARN")
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
                    self._log(f"Cancelando ordem reduceOnly id={order_id}", level="DEBUG")
                self.dex.cancel_order(order_id, self.symbol)
        except Exception as e:
            if self.debug:
                self._log(f"Falha ao cancelar ordem {order_id}: {e}", level="WARN")

    # ---------- fechar posição via market reduceOnly ----------
    def _market_reduce_only(self, side: str, amount: float):
        amt = self._round_amount(amount)
        px  = self._preco_atual()
        params = {"reduceOnly": True}
        if self.debug:
            self._log(f"Fechando posição via MARKET reduceOnly {side.upper()} qty={amt} px_ref={px:.6f}", level="DEBUG")
        return self.dex.create_order(self.symbol, "market", side, amt, px, params)

    def _fechar_posicao(self, df_for_log: pd.DataFrame):
        pos = self._posicao_aberta()
        if not pos or float(pos.get("contracts", 0)) == 0:
            self._log("Fechamento ignorado: posição ausente.", level="DEBUG"); return
        if not self._anti_spam_ok("close"):
            self._log("Fechamento bloqueado pelo anti-spam.", level="DEBUG"); return

        lado_atual = self._norm_side(pos.get("side") or pos.get("positionSide"))
        qty        = float(pos.get("contracts") or 0.0)
        price_now  = self._preco_atual()
        if self.debug:
            self._log(f"Fechando posição {lado_atual.upper()} qty={qty} px={price_now:.6f}", level="DEBUG")

        self._cancel_protective_orders(fetch_backup=True)

        # fechamento via market reduceOnly (lado oposto)
        try:
            close_side = "sell" if lado_atual == "buy" else "buy"
            ret = self._market_reduce_only(close_side, qty)
            self._log(f"Posição encerrada (reduceOnly): {ret}", level="INFO")
            oid = ret.get("id") if isinstance(ret, dict) else None
        except Exception as e:
            self._log(f"Erro ao fechar posição reduceOnly: {e}", level="ERROR"); oid = None
        finally:
            self._safe_log(
                "saida", df_for_log,
                tipo=("long" if lado_atual == "buy" else "short"),
                exec_price=price_now,
                exec_amount=qty,
                order_id=str(oid) if oid else None
            )
            # Cooldown por barras (anti-flip)
            try:
                self._marcar_cooldown_barras(df_for_log)
            except Exception:
                pass
            self._trail_max_gain_pct = None

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
        if not pos or self.cfg.STOP_LOSS_CAPITAL_PCT <= 0:
            return
        side = self._norm_side(pos.get("side") or pos.get("positionSide"))
        entry = float(pos.get("entryPrice") or pos.get("entryPx") or 0.0)
        amt = float(pos.get("contracts") or 0.0)
        if side not in ("buy", "sell") or entry <= 0 or amt <= 0:
            return

        try:
            px_now = self._preco_atual()
        except Exception:
            return
        if px_now <= 0:
            return

        lev_meta = ((pos.get("info") or {}).get("position") or {}).get("leverage") or {}
        try:
            lev_val = float(lev_meta.get("value") or pos.get("leverage") or self.cfg.LEVERAGE)
        except Exception:
            lev_val = float(self.cfg.LEVERAGE)
        if lev_val <= 0:
            lev_val = float(self.cfg.LEVERAGE)
        if lev_val == 0:
            return

        if side == "buy":
            gain_pct_inst = ((px_now - entry) / entry) * lev_val * 100.0
        else:
            gain_pct_inst = ((entry - px_now) / entry) * lev_val * 100.0
        if not math.isfinite(gain_pct_inst):
            return

        if self._trail_max_gain_pct is None:
            self._trail_max_gain_pct = max(0.0, gain_pct_inst)
        else:
            self._trail_max_gain_pct = max(self._trail_max_gain_pct, gain_pct_inst)
        max_gain = self._trail_max_gain_pct

        tol = max(1e-8, entry * 1e-5)
        risk_ratio = float(self.cfg.STOP_LOSS_CAPITAL_PCT) / float(lev_val)

        if side == "buy":
            # Cálculo solicitado: ((preço atual / preço entrada) - 1) * alavancagem, ajustado em -10%,
            # normalizado pela alavancagem e convertido novamente para preço.
            variation = (px_now / entry) - 1.0
            leveraged_variation = variation * lev_val
            adjusted_leveraged = leveraged_variation - float(self.cfg.STOP_LOSS_CAPITAL_PCT)
            normalized_adjusted = adjusted_leveraged / lev_val if lev_val != 0 else 0.0
            target_stop = entry * (1.0 + normalized_adjusted)
            stop_side = "sell"
        else:
            base_loss_pct = self.cfg.STOP_LOSS_CAPITAL_PCT * 100.0
            stop_roi = max(-base_loss_pct, max_gain - base_loss_pct)
            target_stop = entry * (1.0 - (stop_roi / (lev_val * 100.0)))
            stop_side = "buy"

        if target_stop <= 0:
            return

        existing_stop_id = self._last_stop_order_id
        existing_stop_px = self._last_stop_order_px
        if existing_stop_px is None or existing_stop_id is None:
            found_id, found_px, found_is_sell = self._find_existing_stop()
            if found_px is not None:
                existing_stop_id = found_id
                existing_stop_px = found_px

        baseline_stop = None
        if side == "buy":
            baseline_stop = entry * (1.0 - risk_ratio)
        else:
            baseline_stop = entry * (1.0 + risk_ratio)

        reference_stop = existing_stop_px if existing_stop_px is not None else baseline_stop
        if reference_stop is not None:
            if side == "buy" and target_stop <= reference_stop + tol:
                return
            if side == "sell" and target_stop >= reference_stop - tol:
                return

        if not self._anti_spam_ok("adjust"):
            return

        ret = self._place_stop(stop_side, amt, target_stop, df_for_log=df_for_log)
        if ret is not None:
            new_stop_id = self._last_stop_order_id
            if existing_stop_id and existing_stop_id != new_stop_id:
                self._cancel_order_silent(existing_stop_id)
            self._last_stop_order_px = target_stop
            self._log(
                f"Trailing capital: novo stop {stop_side.upper()} @ {target_stop:.6f} (entry {entry:.6f}, px_now {px_now:.6f}, max_gain={max_gain:.2f}%)",
                level="INFO",
            )
            self._safe_log(
                "ajuste_stop", df_for_log,
                tipo=("long" if side == "buy" else "short"),
                exec_price=px_now,
                exec_amount=amt
            )

    # ---------- loop principal ----------
    def step(self, df: pd.DataFrame, usd_to_spend: float, rsi_df_hourly: Optional[pd.DataFrame] = None):
        # filtra símbolo, se DF tiver múltiplos
        if "criptomoeda" in df.columns and (df["criptomoeda"] == self._df_symbol_hint).any():
            df = df.loc[df["criptomoeda"] == self._df_symbol_hint].copy()
        else:
            df = df.copy()

        self._last_price_snapshot = None

        # indicadores e gradiente em %/barra
        df = self._compute_indicators_live(df)
        last = df.iloc[-1]
        last_idx = len(df) - 1
        self._last_seen_bar_idx = last_idx

        price_snapshot = None
        live_enabled = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
        if live_enabled:
            try:
                price_snapshot = self._preco_atual()
            except Exception:
                price_snapshot = None
        if price_snapshot is None:
            fallback_price = None
            try:
                fallback_price = float(last.valor_fechamento)
            except Exception:
                fallback_price = None
            if fallback_price is not None and math.isfinite(fallback_price):
                self._last_price_snapshot = fallback_price

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
                self._log("Posição preexistente detectada ao iniciar ciclo.", level="DEBUG")
            self._first_step_done = True

        prev_side = self._last_pos_side
        pos = self._posicao_aberta()
        self._log(f"Snapshot posição atual: {pos}", level="DEBUG")

        # se havia posição e agora não há → stop/saída ocorreu fora
        if prev_side and not pos:
            self._log("Posição fechada externamente detectada (provável stop).", level="INFO")
            try:
                last_px = self._preco_atual()
            except Exception:
                last_px = None
            self._safe_log(
                "fechado_externo", df_for_log=df,
                tipo=("long" if prev_side == "buy" else "short"),
                exec_price=last_px
            )
            self._cancel_protective_orders(fetch_backup=True)
            # aplica cooldown por barras para evitar reversão imediata
            self._marcar_cooldown_barras(df)
            self._last_pos_side = None
            self._last_stop_order_id = None
            self._last_take_order_id = None
            self._trail_max_gain_pct = None
            self._last_stop_order_px = None
            self._last_take_order_px = None

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

        # Cooldown temporal (tempo fixo pós-saída)
        if not pos and self._cooldown_ativo():
            now = datetime.now(timezone.utc)
            remaining_sec = (self._cooldown_until - now).total_seconds() if self._cooldown_until else 0
            if remaining_sec <= 0:
                self._cooldown_until = None
            else:
                remaining_min = remaining_sec / 60.0
                self._log(
                    f"Cooldown temporal ativo: novas entradas liberadas em {remaining_min:.1f} minuto(s).",
                    level="INFO",
                )
                self._safe_log("cooldown_temporal", df_for_log=df, tipo="info")
                self._last_pos_side = None
                return

        # Cooldown por barras (legado; mantido para compatibilidade)
        just_released_cooldown = False
        if self._cooldown_barras_ativo(df):
            self._last_cooldown_active = True
            try:
                cd_left = None
                if self._cd_bars_left is not None:
                    cd_left = int(self._cd_bars_left)
                elif self._cooldown_until_idx is not None:
                    cd_left = max(0, int(self._cooldown_until_idx - self._bar_index(df)))
                if cd_left is not None:
                    self._log(f"Cooldown ativo: faltam {cd_left} barra(s) para liberar entradas.", level="INFO")
                else:
                    self._log(f"Cooldown ativo ({self.cfg.COOLDOWN_BARS} barras).", level="INFO")
            except Exception:
                self._log("Cooldown ativo (fallback).", level="INFO")
            self._safe_log("cooldown", df_for_log=df, tipo="info")
            self._last_pos_side = (self._norm_side(pos.get("side")) if pos else None)
            return
        else:
            if self._last_cooldown_active:
                just_released_cooldown = True
            self._last_cooldown_active = False

        if pos:
            lado = self._norm_side(pos.get("side") or pos.get("positionSide"))
            self._ensure_position_protections(pos, df_for_log=df)
            self._enforce_max_loss(pos, df_for_log=df)
            self._maybe_trailing_breakeven_plus(pos, df_for_log=df)
            # Contenção adicional: fecha se perda > limite configurado
        try:
            entry_px = float(pos.get("entryPrice") or pos.get("entryPx") or 0.0)
            qty_pos = float(pos.get("contracts") or 0.0)
            contract_sz = float(pos.get("contractSize") or 1.0)
            px_now = self._preco_atual()
            lev_meta = ((pos.get("info") or {}).get("position") or {}).get("leverage") or {}
            lev_val = float(lev_meta.get("value") or pos.get("leverage") or self.cfg.LEVERAGE)
        except Exception:
            entry_px = 0.0; qty_pos = 0.0; contract_sz = 1.0; px_now = 0.0; lev_val = float(self.cfg.LEVERAGE)
        if lev_val <= 0:
            lev_val = float(self.cfg.LEVERAGE)
        loss_trigger_pct = -abs(self.cfg.STOP_LOSS_CAPITAL_PCT * 100.0)
        pnl_abs = None
        if pos:
            raw_abs = pos.get("unrealizedPnl")
            if raw_abs is None:
                raw_abs = ((pos.get("info") or {}).get("position") or {}).get("unrealizedPnl")
            try:
                pnl_abs = float(raw_abs)
            except Exception:
                pnl_abs = None
            if pnl_abs is None or not math.isfinite(pnl_abs):
                if entry_px > 0 and qty_pos > 0 and px_now > 0:
                    qvalue = qty_pos * contract_sz
                    if qvalue > 0:
                        if lado == "buy":
                            pnl_abs = (px_now - entry_px) * qvalue
                        else:
                            pnl_abs = (entry_px - px_now) * qvalue
        if entry_px > 0 and qty_pos > 0 and px_now > 0:
            if lado == "buy":
                pnl_pct = ((px_now - entry_px) / entry_px) * lev_val * 100.0
                pnl_abs = pnl_abs if pnl_abs is not None else (px_now - entry_px) * qty_pos * contract_sz
            else:
                pnl_pct = ((entry_px - px_now) / entry_px) * lev_val * 100.0
                pnl_abs = pnl_abs if pnl_abs is not None else (entry_px - px_now) * qty_pos * contract_sz
            if self.debug:
                self._log(f"Drawdown atual={pnl_pct:.2f}% | limite={loss_trigger_pct:.2f}%", level="DEBUG")
            max_loss_abs = float(getattr(self.cfg, "MAX_LOSS_ABS_USD", 0.0) or 0.0)
            if max_loss_abs > 0 and pnl_abs is not None and math.isfinite(pnl_abs):
                if pnl_abs <= -abs(max_loss_abs):
                    self._log(
                        f"Perda de {pnl_abs:.4f} USDC excedeu limite -{abs(max_loss_abs):.2f}. Fechando posição imediatamente.",
                        level="WARN",
                    )
                    self._fechar_posicao(df_for_log=df)
                    return
            if pnl_pct <= loss_trigger_pct:
                self._log(
                    f"Perda de {pnl_pct:.2f}% excedeu limite {loss_trigger_pct:.2f}%. Fechando posição imediatamente.",
                    level="WARN",
                )
                self._fechar_posicao(df_for_log=df)
                return
            self._log("Posição aberta: aguardando execução de TP/SL.", level="DEBUG")
            self._safe_log("decisao", df_for_log=df, tipo="info")
            self._last_pos_side = lado if lado in ("buy", "sell") else None
            return

        if not pos:
            live = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
            if not live:
                self._log("LIVE_TRADING=0: avaliando sinais sem enviar ordens.", level="INFO")
                self._safe_log("paper_mode", df_for_log=df, tipo="info")
                self._last_pos_side = None
                self._entry_pending_signal = None
                return

            ema7 = float(last.ema_short)
            ema21 = float(last.ema_long)
            atr = float(last.atr)
            atr_pct = float(last.atr_pct)
            price = float(last.valor_fechamento)
            spread = ema7 - ema21
            self._update_hysteresis(spread, atr)

            vol_flag, vol_current, vol_median, vol_mad = self._volume_robust_flag(df)
            m_persist = max(1, int(self.cfg.ENTRY_GRAD_PERSIST or 1))
            persist_long, count_long = self._grad_persistence(df, "LONG", m_persist)
            persist_short, count_short = self._grad_persistence(df, "SHORT", m_persist)
            k_val = self._adaptive_k(atr_pct)
            rsi_val = self._current_rsi(df, rsi_df_hourly)

            metrics = {
                "ema7": ema7,
                "ema21": ema21,
                "spread": spread,
                "atr": atr,
                "atr_pct": atr_pct,
                "price": price,
                "vol_flag": vol_flag,
                "grad_persist_flags": {"LONG": persist_long, "SHORT": persist_short},
                "grad_counts": {"LONG": count_long, "SHORT": count_short},
                "k_values": {"LONG": k_val, "SHORT": k_val},
            }

            decisions = [
                self._assess_entry(metrics, "LONG", last_idx, rsi_val, just_released_cooldown),
                self._assess_entry(metrics, "SHORT", last_idx, rsi_val, just_released_cooldown),
            ]

            approved = [d for d in decisions if d["approved"]]
            best = max(approved, key=lambda d: d["score"]) if approved else None

            if best:
                pending = self._entry_pending_signal
                wait_required = max(0, int(self.cfg.ENTRY_DEBOUNCE_BARS or 0))
                if wait_required == 0:
                    order_side = "buy" if best["side"] == "LONG" else "sell"
                    self._entry_pending_signal = None
                    self._abrir_posicao_com_stop(order_side, usd_to_spend, df_for_log=df, atr_last=float(last.atr))
                    pos_after = self._posicao_aberta()
                    self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                    return
                if pending and pending.get("side") == best["side"]:
                    pending["info"] = best
                    waited = last_idx - pending["start_idx"]
                    if waited >= wait_required:
                        order_side = "buy" if best["side"] == "LONG" else "sell"
                        self._entry_pending_signal = None
                        self._abrir_posicao_com_stop(order_side, usd_to_spend, df_for_log=df, atr_last=float(last.atr))
                        pos_after = self._posicao_aberta()
                        self._last_pos_side = self._norm_side(pos_after.get("side")) if pos_after else None
                        return
                    else:
                        remaining = max(0, wait_required - waited)
                        self._log(
                            f"Entrada pendente ({best['side']}) aguardando debounce: faltam {remaining} barra(s)",
                            level="INFO",
                        )
                        self._last_pos_side = None
                        return
                else:
                    self._entry_pending_signal = {"side": best["side"], "start_idx": last_idx, "info": best}
                    self._log(
                        f"Entrada pendente ({best['side']}) iniciada (score={best['score']})",
                        level="INFO",
                    )
                    self._last_pos_side = None
                    return

            # Nenhum candidato aprovado: limpa pendente e mantém bloqueio
            if self._entry_pending_signal is not None:
                self._log(
                    f"Entrada pendente cancelada ({self._entry_pending_signal['side']}) por perda de contexto.",
                    level="INFO",
                )
                self._entry_last_block_idx[self._entry_pending_signal["side"]] = last_idx
                self._entry_pending_signal = None
            self._last_pos_side = None
            return


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

if __name__ == "__main__":
    # Compat: alias para versões antigas que esperam EMAGradientATRStrategy
    EMAGradientATRStrategy = EMAGradientStrategy  # type: ignore

    def executar_estrategia(
        df_in: pd.DataFrame,
        dex_in,
        trade_logger_in: TradeLogger | None,
        usd_to_spend: float = 1,
        loop: bool = True,
        sleep_seconds: int = 60,
    ):
        """Executa a estratégia sequencialmente para cada ativo configurado."""
        _log_global(
            "ENGINE",
            f"LIVE_TRADING={os.getenv('LIVE_TRADING', '0')} | DEX_TIMEOUT_MS={os.getenv('DEX_TIMEOUT_MS', '5000')} | assets={len(ASSET_SETUPS)}",
        )

        if trade_logger_in is not None:
            _log_global("ENGINE", "Logger externo fornecido será ignorado no modo multiativo.", level="DEBUG")

        asset_state: Dict[str, Dict[str, Any]] = {}
        default_cols = df_in.columns if isinstance(df_in, pd.DataFrame) else pd.Index([])

        iter_count = 0
        while True:
            iter_count += 1
            try:
                live_flag = os.getenv("LIVE_TRADING", "0") in ("1", "true", "True")
                _log_global("HEARTBEAT", f"iter={iter_count} live={int(live_flag)}")
            except Exception:
                pass

            for asset in ASSET_SETUPS:
                _log_global("ASSET", f"Processando {asset.name}")
                try:
                    df_asset = build_df(asset.data_symbol, INTERVAL, debug=True)
                except MarketDataUnavailable as e:
                    _log_global(
                        "ASSET",
                        f"Sem dados recentes para {asset.name} ({asset.data_symbol}) {INTERVAL}: {e}",
                        level="WARN",
                    )
                    continue
                except Exception as e:
                    _log_global("ASSET", f"Falha ao atualizar DF {asset.name}: {type(e).__name__}: {e}", level="WARN")
                    continue

                try:
                    df_asset_hour = build_df(asset.data_symbol, "1h", debug=False)
                except MarketDataUnavailable:
                    _log_global(
                        "ASSET",
                        f"Sem dados 1h para {asset.name} ({asset.data_symbol}); seguindo sem rsi_aux.",
                        level="WARN",
                    )
                    df_asset_hour = pd.DataFrame()
                except Exception as e:
                    _log_global("ASSET", f"Falha ao atualizar DF 1h {asset.name}: {type(e).__name__}: {e}", level="WARN")
                    df_asset_hour = pd.DataFrame()

                if not isinstance(df_asset, pd.DataFrame) or df_asset.empty:
                    _log_global("ASSET", f"DataFrame vazio para {asset.name}; pulando.", level="WARN")
                    continue

                state = asset_state.get(asset.name)
                if state is None:
                    cfg = GradientConfig()
                    cfg.LEVERAGE = asset.leverage
                    cfg.STOP_LOSS_CAPITAL_PCT = asset.stop_pct
                    cfg.TAKE_PROFIT_CAPITAL_PCT = asset.take_pct
                    safe_suffix = asset.name.lower().replace("-", "_").replace("/", "_")
                    csv_path = f"trade_log_{safe_suffix}.csv"
                    xlsx_path = f"trade_log_{safe_suffix}.xlsx"
                    cols = df_asset.columns if isinstance(df_asset, pd.DataFrame) else default_cols
                    logger = TradeLogger(cols, csv_path=csv_path, xlsx_path_dbfs=xlsx_path)
                    strategy = EMAGradientStrategy(
                        dex=dex_in,
                        symbol=asset.hl_symbol,
                        cfg=cfg,
                        logger=logger,
                        debug=True,
                    )
                    strategy = _ensure_strategy_step_instance(strategy)
                    asset_state[asset.name] = {"strategy": strategy, "logger": logger}
                strategy: EMAGradientStrategy = asset_state[asset.name]["strategy"]

                usd_asset = usd_to_spend
                try:
                    global_env = os.getenv("USD_PER_TRADE")
                    if global_env:
                        usd_asset = float(global_env)
                    if asset.usd_env:
                        specific_env = os.getenv(asset.usd_env)
                        if specific_env:
                            usd_asset = float(specific_env)
                except Exception:
                    pass

                try:
                    _safe_strategy_step(strategy, df_asset, usd_to_spend=usd_asset, rsi_df_hourly=df_asset_hour)
                    price_seen = getattr(strategy, "_last_price_snapshot", None)
                    if price_seen is not None and math.isfinite(price_seen):
                        try:
                            strategy._log(f"Preço atual: {price_seen:.6f}", level="INFO")
                        except Exception:
                            pass
                except Exception as e:
                    _log_global("ASSET", f"Erro executando {asset.name}: {type(e).__name__}: {e}", level="ERROR")
                _time.sleep(0.25)

            if not loop:
                break

            try:
                env_sleep = os.getenv("SLEEP_SECONDS")
                if env_sleep:
                    sleep_seconds = int(env_sleep)
            except Exception:
                pass
            _time.sleep(max(1, int(sleep_seconds)))

    base_df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    executar_estrategia(base_df, dex, None)

# --- Compat shim: garante que EMAGradientStrategy tenha método `step` ---
try:
    _cls = EMAGradientStrategy
    if not hasattr(_cls, "step"):
        def _shim_step(self, df: pd.DataFrame, usd_to_spend: float, rsi_df_hourly: Optional[pd.DataFrame] = None):
            if hasattr(self, "run"):
                return self.run(df, usd_to_spend, rsi_df_hourly)
            if hasattr(self, "execute"):
                return self.execute(df, usd_to_spend, rsi_df_hourly)
            if hasattr(self, "tick"):
                return self.tick(df, usd_to_spend, rsi_df_hourly)
            raise AttributeError("EMAGradientStrategy não possui método step/aliases")
        EMAGradientStrategy.step = _shim_step
except Exception:
    pass

# ---- Ensure EMAGradientStrategy exposes .step even if an older build is imported ----
try:
    _EMAG = EMAGradientStrategy  # type: ignore[name-defined]
    if not hasattr(_EMAG, "step"):
        def _emag_step(self, *args, **kwargs):  # pragma: no cover
            if hasattr(self, "run"):
                return self.run(*args, **kwargs)
            if hasattr(self, "process"):
                return self.process(*args, **kwargs)
            raise AttributeError("EMAGradientStrategy does not expose step/run/process")
        setattr(_EMAG, "step", _emag_step)
        print("[WARN] [COMPAT] Injected .step() into EMAGradientStrategy at runtime for compatibility.", flush=True)
except Exception as _compat_err:
    try:
        print(f"[WARN] [COMPAT] Could not ensure .step on EMAGradientStrategy: {_compat_err}", flush=True)
    except Exception:
        pass



