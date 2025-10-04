import pandas as pd
from tradingv4 import ASSET_SETUPS, GradientConfig, EMAGradientStrategy

# Simulação de contexto/dex para teste (substitua por seu objeto real)
class DummyDex:
    def fetch_ticker(self, symbol):
        return {"last": 1.0}
    def fetch_positions(self, symbols, opts):
        return [{"contracts": 0}]
    def fetch_open_orders(self, symbol, a, b, opts):
        return []
    def load_markets(self, reload=True):
        return {symbol: {"info": {"midPx": 1.0}} for symbol in [a.hl_symbol for a in ASSET_SETUPS]}
    def set_leverage(self, lev, symbol, opts):
        pass
    def amount_to_precision(self, symbol, amount):
        return amount


def verifica_criterios_ativos():
    resultados = []
    dex = DummyDex()
    cfg = GradientConfig()
    for asset in ASSET_SETUPS:
        strat = EMAGradientStrategy(dex, asset.hl_symbol, cfg, debug=False)
        try:
            # Simule um DataFrame de candles para o ativo
            df = pd.DataFrame({
                "valor_fechamento": [1.0, 1.1, 1.2, 1.3, 1.4],
                "data": pd.date_range("2025-10-01", periods=5, freq="H")
            })
            df_ind = strat._compute_indicators_live(df)
            # Aqui você pode adicionar verificações dos critérios de entrada
            grad = strat._gradiente(df_ind["ema_short"])
            atr = 0.8  # Simule ATR
            dentro_atr = cfg.ATR_PCT_MIN <= atr <= cfg.ATR_PCT_MAX
            criterios = {
                "ativo": asset.name,
                "gradiente": grad,
                "ATR": atr,
                "ATR_OK": dentro_atr,
                "leverage": asset.leverage,
                "stop_pct": asset.stop_pct,
                "take_pct": asset.take_pct,
            }
            resultados.append(criterios)
        except Exception as e:
            resultados.append({"ativo": asset.name, "erro": str(e)})
    df_result = pd.DataFrame(resultados)
    print(df_result)
    return df_result

if __name__ == "__main__":
    verifica_criterios_ativos()
