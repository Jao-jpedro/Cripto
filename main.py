import requests
import time
import os

wallet_address = os.getenv("WALLET")
discord_webhook = os.getenv("DISCORD_WEBHOOK")

open_positions = {}

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

print("🔄 Bot rodando...")

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
