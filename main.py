import requests
import time
import os
import threading
from flask import Flask

wallet_address = os.getenv("WALLET")
discord_webhook = os.getenv("DISCORD_WEBHOOK")
open_positions = {}

# === BOT DE MONITORAMENTO ===
def get_positions(wallet):
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "userState", "user": wallet}
    res = requests.post(url, json=payload)
    return {
        pos["coin"]: pos
        for pos in res.json()["assetPositions"]
        if float(pos["szi"]) != 0
    }

def send_discord_alert(message):
    payload = {"content": message}
    requests.post(discord_webhook, json=payload)

def monitor():
    print("🔄 Bot de monitoramento iniciado...")
    while True:
        try:
            current_positions = get_positions(wallet_address)
            for coin in open_positions:
                if coin not in current_positions:
                    send_discord_alert(f"🚨 Posição encerrada: `{coin}`")
            open_positions.clear()
            open_positions.update(current_positions)
            time.sleep(10)
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(15)

# === INICIA O BOT EM UMA THREAD ===
threading.Thread(target=monitor).start()

# === CRIA UM FLASK SERVER SIMPLES PARA MANTER A PORTA VIVA ===
app = Flask(__name__)

@app.route('/')
def home():
    return 'Bot está rodando!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Porta qualquer
