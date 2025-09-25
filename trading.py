def close_if_abs_loss_exceeds_5c(dex, symbol, current_px: float, *, vault) -> bool: ...
def close_if_breached_leveraged(dex, symbol, current_px: float, *, vault) -> bool: ...

def close_if_unrealized_pnl_breaches(dex, symbol, *, vault, threshold: float = -0.10) -> bool:
    """
    Fecha imediatamente se unrealizedPnl <= threshold (ex.: threshold=-0.10 para -5 cents).
    Se unrealizedPnl não estiver disponível, não faz nada (fallbacks separados cuidam do resto).
    """
    try:
        pos = _get_position_for_vault(dex, symbol, vault)
    except Exception:
        pos = None
    if not pos:
        return False
    # Tenta extrair unrealizedPnl em vários formatos
    pnl = None
    try:
        pnl = pos.get("unrealizedPnl")
        if pnl is None:
            pnl = (pos.get("info") or {}).get("unrealizedPnl")
        if pnl is None:
            pnl = ((pos.get("info") or {}).get("position") or {}).get("unrealizedPnl")
    except Exception:
        pnl = None
    if pnl is None:
        return False
    try:
        pnl_f = float(pnl)
    except Exception:
        return False

    if pnl_f <= float(threshold):
        # Fecha a posição inteira no lado de saída
        try:
            qty, _, _, side = _get_pos_size_and_leverage(dex, symbol, vault=vault)
            if not side or qty <= 0: 
                return False
            exit_side = "sell" if (str(side).lower() in ("long", "buy")) else "buy"
            dex.create_order(symbol, "market", exit_side, float(qty), None, {"reduceOnly": True })
            return True
        except Exception:
            return False
    return False

#codigo com [all] trades=70 win_rate=35.71% PF=1.378 maxDD=-6.593% Sharpe=0.872 


# ====== Wallet Config (main account only) ======
MAIN_ACCOUNT_ADDRESS = "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
print(f"[WALLET] Operando SOMENTE na carteira principal: {MAIN_ACCOUNT_ADDRESS}", flush=True)
print("\n========== INÍCIO DO BLOCO: HISTÓRICO DE TRADES ==========", flush=True)


def _log_global(section: str, message: str, level: str = "INFO") -> None:
    """Formato padrão para logs fora das classes."""
    print(f"[{level}] [{section}] {message}", flush=True)

# Silencia aviso visual do urllib3 sobre OpenSSL/LibreSSL (sem importar urllib3)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import requests  # type: ignore
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from urllib3.util.retry import Retry  # type: ignore
else:
    try:
        import requests  # type: ignore
    except Exception:
        requests = None  # type: ignore
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    try:
        from requests.adapters import HTTPAdapter  # type: ignore
        from urllib3.util.retry import Retry  # type: ignore
    except Exception:
        HTTPAdapter = object  # type: ignore
        class Retry:  # type: ignore
            def __init__(self, *args, **kwargs): pass

import warnings as _warnings
# ===== Hyperliquid accounts / vault / signer =====
HL_MAIN_ACCOUNT = "0x08183aa09eF03Cf8475D909F507606F5044cBdAB"
HL_SUBACCOUNT_VAULT = "0x5ff0f14d577166f9ede3d9568a423166be61ea9d"
HL_API_WALLET = "0x95cf910f947a5be26bc7c18f8b8048185126b4e9"

_warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL 1.1.1\+.*",
    category=Warning,
    module=r"urllib3.*",
)


def _binance_bases():
    # Força o endpoint público (dados históricos) para evitar 451/403
    return ["https://data-api.binance.vision/api/v3/"]

def _binance_session():
    s = requests.Session()
    try:

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
        "Accept": "application/json" })
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
    n_target = int(os.getenv("TARGET_CANDLES", "0"))
    if target_candles is not None:
        n_target = int(target_candles)
    if n_target <= 0:
        n_target = 50  # padrão solicitado
    n_target = min(n_target, 50)

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
    try:
        import ccxt  # type: ignore

        ex = ccxt.bybit({
            "enableRateLimit": True,
            "timeout": int(os.getenv("BYBIT_TIMEOUT_MS", "5000")),
            "options": {"timeout": int(os.getenv("BYBIT_TIMEOUT_MS", "5000"))} })
        # Busca até os últimos n_target candles (Bybit normalmente retorna fechados; alguns mercados incluem o em formação)
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
            # Garante no máximo n_target candles
            if len(cc) > n_target:
                cc = cc[-n_target:]
            data = [{
                "data": o[0],
                "valor_fechamento": float(o[4]),
                "criptomoeda": symbol,
                "volume_compra": float(o[5] or 0.0),
                "volume_venda": float(o[5] or 0.0) } for o in cc]
            if debug:
                _log_global("BYBIT", f"{len(data)} candles carregados (API)")
        else:
            if debug:
                _log_global("BYBIT", f"Nenhum candle retornado (último erro: {last_err})", level="WARN")
        # Se o último candle não é o atual, adiciona o preço atual como candle em formação
        if data:
            need_append_live = True
            last_ts = int(data[-1]["data"])
            if last_ts == cur_open_ms:
                need_append_live = False
            elif last_ts > cur_open_ms:
                need_append_live = False
            if need_append_live and len(data) < n_target:
                try:
                    ticker = ex.fetch_ticker(symbol_bybit)
                    if ticker and (ticker.get("last") is not None):
                        data.append({
                            "data": cur_open_ms,
                            "valor_fechamento": float(ticker["last"]),
                            "criptomoeda": symbol,
                            "volume_compra": 0.0,
                            "volume_venda": 0.0 })
                        if debug:
                            _log_global("BYBIT", f"Ticker adicionou candle em formação price={ticker['last']}")
                except Exception as e:
                    if debug:
                        _log_global("BYBIT", f"Não foi possível adicionar preço atual: {type(e).__name__}: {e}", level="WARN")
        # Garante exatamente n_target no máximo (fechados + atual)
        if data and len(data) > n_target:
            data = data[-n_target:]
    except Exception as e:
        if debug:
            _log_global("BYBIT", f"Exceção geral: {type(e).__name__}: {e}", level="WARN")

    # Fallback 1: tentar Binance Vision pública se Bybit vazio (sem bloquear)
    if not data:
        try:
            candles_needed = n_target
            start_dt = datetime.fromtimestamp(cur_open_epoch - (candles_needed - 1) * secs, UTC)
            end_dt = now_utc
            if debug:
                _log_global("BINANCE_VISION", "Ativando fallback público")
            bdata = get_binance_data(symbol, tf, start_dt, end_dt)
            if bdata:
                data = bdata[-n_target:]
                if debug:
                    _log_global("BINANCE_VISION", f"{len(data)} candles carregados")
        except Exception as e:
            if debug:
                _log_global("BINANCE_VISION", f"Falhou: {type(e).__name__}: {e}", level="WARN")

    # Fallback: snapshot local
    if not data and os.path.exists("df_log.csv") and os.path.getsize("df_log.csv") > 0:
        try:
            df_local = pd.read_csv("df_log.csv")
            if "data" in df_local.columns:
                df_local["data"] = pd.to_datetime(df_local["data"])
            if debug:
                _log_global("DATA", "Fallback local df_log.csv carregado")
            return df_local
        except Exception as e:
            if debug:
                _log_global("DATA", f"Falha ao ler df_log.csv: {e}", level="WARN")

    if not data:
        if debug:
            _log_global("DATA", f"Sem dados retornados para {symbol} tf={tf}", level="ERROR")
        return pd.DataFrame()

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


def guard_close_all(dex, symbol, current_px: float, *, vault) -> bool:
    try:
        if close_if_unrealized_pnl_breaches(dex, symbol, vault=vault, threshold=-0.10):
            return True
    except Exception:
        pass
    return False

def _patch_method(name):
    if hasattr(dex, name):
        orig = getattr(dex, name)
        def _fn(*args, **kwargs):
            if args and isinstance(args[-1], dict):
                args = (*args[:-1], _with_vault(args[-1]))
            else:
                kwargs["params"] = _with_vault(kwargs.get("params"))
            return orig(*args, **kwargs)
        setattr(dex, name, _fn)

for _meth in ("cancel_order", "cancel_orders", "fetch_open_orders", "fetch_orders", "fetch_positions",
              "set_leverage", "set_margin_mode", "set_position_mode"):
    _patch_method(_meth)
# Segundo DEX (racional inverso) com credenciais distintas
# COMMAND ----------
# =========================
# 🔔 LOGGER (CSV + XLSX em DBFS com workaround /tmp → dbutils.fs.cp)
# =========================
import os

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
            "dt_evento_brt": dt_brt }

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
    STOP_LOSS_CAPITAL_PCT: float = 0.05  # 10% da margem como stop
    TAKE_PROFIT_CAPITAL_PCT: float = 0.10  # 30% da margem como alvo

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
    usd_env: Optional[str] = None

ASSET_SETUPS: list[AssetSetup] = [
    AssetSetup('BTC-USD', 'BTCUSDT', 'BTC/USDC:USDC', 40, usd_env='USD_PER_TRADE_BTC'),
    AssetSetup('SOL-USD', 'SOLUSDT', 'SOL/USDC:USDC', 20, usd_env='USD_PER_TRADE_SOL'),
    AssetSetup('ETH-USD', 'ETHUSDT', 'ETH/USDC:USDC', 25, usd_env='USD_PER_TRADE_ETH'),
    AssetSetup('HYPE-USD', 'HYPEUSDT', 'HYPE/USDC:USDC', 10, usd_env='USD_PER_TRADE_HYPE'),
    AssetSetup('XRP-USD', 'XRPUSDT', 'XRP/USDC:USDC', 20, usd_env='USD_PER_TRADE_XRP'),
    AssetSetup('DOGE-USD', 'DOGEUSDT', 'DOGE/USDC:USDC', 10, usd_env='USD_PER_TRADE_DOGE'),
    AssetSetup('AVAX-USD', 'AVAXUSDT', 'AVAX/USDC:USDC', 10, usd_env='USD_PER_TRADE_AVAX'),
    AssetSetup('ENA-USD', 'ENAUSDT', 'ENA/USDC:USDC', 10, usd_env='USD_PER_TRADE_ENA'),
    AssetSetup('BNB-USD', 'BNBUSDT', 'BNB/USDC:USDC', 10, usd_env='USD_PER_TRADE_BNB'),
    AssetSetup('SUI-USD', 'SUIUSDT', 'SUI/USDC:USDC', 10, usd_env='USD_PER_TRADE_SUI'),
    AssetSetup('ADA-USD', 'ADAUSDT', 'ADA/USDC:USDC', 10, usd_env='USD_PER_TRADE_ADA'),
    AssetSetup('PUMP-USD', 'PUMPUSDT', 'PUMP/USDC:USDC', 5, usd_env='USD_PER_TRADE_PUMP'),
    AssetSetup('AVNT-USD', 'AVNTUSDT', 'AVNT/USDC:USDC', 5, usd_env='USD_PER_TRADE_AVNT'),
    AssetSetup('XPL-USD', 'XPLUSDT', 'XPL/USDC:USDC', 3, usd_env='USD_PER_TRADE_XPL'),
    AssetSetup('KPEPE-USD', 'KPEPEUSDT', 'KPEPE/USDC:USDC', 10, usd_env='USD_PER_TRADE_KPEPE'),
    AssetSetup('LINK-USD', 'LINKUSDT', 'LINK/USDC:USDC', 10, usd_env='USD_PER_TRADE_LINK'),
    AssetSetup('WLD-USD', 'WLDUSDT', 'WLD/USDC:USDC', 10, usd_env='USD_PER_TRADE_WLD'),
    AssetSetup('AAVE-USD', 'AAVEUSDT', 'AAVE/USDC:USDC', 10, usd_env='USD_PER_TRADE_AAVE'),
    AssetSetup('CRV-USD', 'CRVUSDT', 'CRV/USDC:USDC', 10, usd_env='USD_PER_TRADE_CRV'),
    AssetSetup('LTC-USD', 'LTCUSDT', 'LTC/USDC:USDC', 10, usd_env='USD_PER_TRADE_LTC'),
    AssetSetup('NEAR-USD', 'NEARUSDT', 'NEAR/USDC:USDC', 10, usd_env='USD_PER_TRADE_NEAR'),
]


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
    AssetSetup("XPL-USD", "XPLUSDT", "XPL/USDC:USDC", 3, usd_env="USD_PER_TRADE_XPL"),
    AssetSetup("KPEPE-USD", "KPEPEUSDT", "KPEPE/USDC:USDC", 10, usd_env="USD_PER_TRADE_KPEPE"),
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
        self._pending_after_cd: Optional[Dict[str, Any]] = None  # {side, reason, created_idx}
        self._last_seen_bar_idx: Optional[int] = None
        # Cooldown por barras (robusto a janela deslizante)
        self._cd_bars_left: Optional[int] = None
        self._cd_last_bar_time: Optional[pd.Timestamp] = None
        self._cd_last_seen_idx: Optional[int] = None

        # Controle das ordens de proteção
        self._last_stop_order_id: Optional[str] = None
        self._last_take_order_id: Optional[str] = None

    def _log(self, message: str, level: str = "INFO") -> None:
        prefix = f"{self.symbol}" if self.symbol else "STRAT"
        print(f"[{level}] [{prefix}] {message}", flush=True)

    
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
        "sharpe": float(sharpe) }


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
            "atr_outside": metrics_outside },
        "params": p }


# DBTITLE 1,principal
# =========================
# 🔧 INSTÂNCIA E EXECUÇÃO
# =========================



# ===== SAFETY UTILS (inseridos antes do __main__ para evitar NameError) =====

def _get_position_for_vault(dex, symbol, vault):
    """
    Tenta obter a posição atual (da subconta/vault) para o símbolo.
    Retorna dict com chaves padrão: contracts, entryPrice, side, leverage, info
    """
    try:
        poss = dex.fetch_positions([symbol]) or []
    except Exception as e:
        print(f"[POS][{symbol}] fetch_positions falhou: {type(e).__name__}: {e}")
        poss = []

    best = None
    for p in poss:
        try:
            contracts = float(p.get("contracts") or p.get("amount") or 0.0)
            if contracts > 0:
                best = p
                break
            # alguns conectores reportam posição com side mesmo zerada
            if p.get("side"):
                best = p
        except Exception:
            continue

    if not best and poss:
        best = poss[0]

    if not isinstance(best, dict):
        return None

    # normaliza campos
    try:
        if "contracts" not in best:
            c = best.get("amount") or 0.0
            best["contracts"] = float(c)
    except Exception:
        pass
    try:
        if "entryPrice" not in best:
            ep = (best.get("entry") or best.get("avgEntryPrice") or 0.0)
            if ep: best["entryPrice"] = float(ep)
    except Exception:
        pass
    return best


def _get_pos_size_and_leverage(dex, symbol, *, vault):
    p = _get_position_for_vault(dex, symbol, vault)
    if not p:
        return 0.0, 1.0, None, None
    qty = float(p.get("contracts") or 0.0)
    entry = float(p.get("entryPrice") or 0.0) or None
    side = p.get("side") or None
    lev = p.get("leverage")
    if isinstance(lev, dict):
        lev = lev.get("value")
    if lev is None:
        info = p.get("info") or {}
        lev = ((info.get("position") or {}).get("leverage") or {}).get("value")
    if lev is None:
        lev = 1.0
    return float(qty), float(lev), entry, side



def _approx_equal(a: float, b: float, tol_abs: float = None, tol_pct: float = 0.001) -> bool:
    if a is None or b is None:
        return False
    ref = max(abs(a), abs(b), 1e-12)
    bound = max((tol_abs or 0.0), ref * tol_pct)
    return abs(a - b) <= bound

def _get_current_contracts(dex, symbol, *, vault) -> float:
    try:
        poss = dex.fetch_positions([symbol]) or []
        for p in poss:
            contracts = p.get("contracts") or p.get("amount") or 0
            if contracts and float(contracts) > 0:
                return float(contracts)
    except Exception:
        pass
    return 0.0


def _get_position_for_vault(dex, symbol, vault):
    try:
        poss = dex.fetch_positions([symbol]) or []
        for p in poss:
            qty = float(p.get("contracts") or 0.0)
            side = (p.get("side") or "").lower()
            if qty > 0 and side in ("long", "short"):
                return p
    except Exception:
        pass
    return None

def _get_pos_size_and_leverage(dex, symbol, *, vault):
    p = _get_position_for_vault(dex, symbol, vault)
    if not p:
        return 0.0, 1.0, None, None
    qty = float(p.get("contracts") or 0.0)
    entry = float(p.get("entryPrice") or 0.0) or None
    side = p.get("side") or None
    lev = p.get("leverage")
    if isinstance(lev, dict):
        lev = lev.get("value")
    if lev is None:
        info = p.get("info") or {}
        lev = ((info.get("position") or {}).get("leverage") or {}).get("value")
    if lev is None:
        lev = 1.0
    return float(qty), float(lev), entry, side

def _approx(a, b, tol=0.001):
    try:
        a, b = float(a), float(b)
        return abs(a - b) <= max(1e-12, tol * max(abs(a), abs(b), 1.0))
    except Exception:
        return False




def safety_close_if_loss_exceeds_5c(dex, symbol, current_px: float, *, vault) -> bool:
    """
    Se a perda não realizada > $0.05, fecha a posição imediatamente (market reduceOnly).
    """
    qty, _, entry, side = _get_pos_size_and_leverage(dex, symbol, vault=vault)
    if qty <= 0 or not entry or not side:
        return False

    s = side.lower()
    if s in ("long", "buy"):
        loss = max(0.0, (entry - float(current_px)) * qty)
        exit_side = "sell"
    else:
        loss = max(0.0, (float(current_px) - entry) * qty)
        exit_side = "buy"

    if loss > 0.05:
        try:
            dex.create_order(symbol, "market", exit_side, qty, None,
                             {"reduceOnly": True})
            return True
        except Exception:
            return False
    return False

