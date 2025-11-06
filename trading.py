#!/usr/bin/env python3

"""
Sistema de Trading DCA (Dollar Cost Averaging) - SOL Long Only
Estratégia de degraus de compra e venda baseada em % do preço máximo/ganho
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# ===== IMPORTS ESSENCIAIS =====
import ccxt
import requests

# ===== CONFIGURAÇÃO DE TIMEZONE =====
UTC = timezone.utc

# ===== HYPERLIQUID API =====
_HL_INFO_URL = "https://api.hyperliquid.xyz/info"
_HTTP_TIMEOUT = 10
_SESSION = requests.Session()

def _http_post_json(url: str, payload: dict, timeout: int = _HTTP_TIMEOUT):
    """Helper para fazer requisições POST JSON"""
    try:
        r = _SESSION.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Requisição falhou: {e}")
        return None

def _hl_get_account_value(wallet: str) -> float:
    """Busca o saldo DISPONÍVEL de uma conta/vault específica via API Hyperliquid"""
    if not wallet:
        return 0.0
    data = _http_post_json(_HL_INFO_URL, {"type": "clearinghouseState", "user": wallet})
    try:
        if not data or "marginSummary" not in data:
            return 0.0
        
        margin_summary = data["marginSummary"]
        
        # Saldo disponível = accountValue - totalMarginUsed
        account_value = float(margin_summary.get("accountValue", 0))
        margin_used = float(margin_summary.get("totalMarginUsed", 0))
        
        available = account_value - margin_used
        
        return max(available, 0.0)  # Garantir que não seja negativo
    except Exception as e:
        print(f"[ERROR] Erro parseando saldo: {e}")
        return 0.0

def _hl_get_latest_fill(wallet: str, symbol: str = None):
    """Busca último preenchimento (fill) de ordem via API Hyperliquid"""
    if not wallet:
        return None
    data = _http_post_json(_HL_INFO_URL, {"type": "userFills", "user": wallet})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    
    fills = data
    if symbol:
        fills = [f for f in fills if f.get('coin') == symbol.replace('/USDC:USDC', '')]
    
    if not fills:
        return None
    
    # Retornar fill mais recente
    return fills[0]

def _hl_get_user_state(wallet: str):
    """Busca estado completo do usuário incluindo posições abertas"""
    if not wallet:
        return None
    data = _http_post_json(_HL_INFO_URL, {"type": "clearinghouseState", "user": wallet})
    return data

def _hl_get_user_fills(wallet: str, limit: int = 100):
    """
    Busca histórico de trades (fills) do usuário via API Hyperliquid.
    Retorna lista de fills ordenados por timestamp (mais recente primeiro).
    """
    if not wallet:
        return []
    
    data = _http_post_json(_HL_INFO_URL, {
        "type": "userFills",
        "user": wallet
    })
    
    if not data or not isinstance(data, list):
        return []
    
    # Limitar quantidade de fills retornados
    return data[:limit]

# ===== LOGGING =====
_LOG_FILE = None

def setup_log_file():
    """Configura arquivo de log baseado na data/hora atual"""
    global _LOG_FILE
    if _LOG_FILE is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_FILE = f"trading_dca_{timestamp}.log"
        print(f"📝 Log será salvo em: {_LOG_FILE}")

def log(message: str, level: str = "INFO"):
    """Sistema de log simplificado"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    
    print(log_line, flush=True)
    
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')
                f.flush()
        except Exception as e:
            print(f"[ERROR] Erro salvando log: {e}")

# ===== NOTIFICAÇÕES DISCORD =====
class DiscordNotifier:
    """Sistema de notificações Discord"""
    
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.enabled = bool(self.webhook_url)
        self.last_notification_time = 0
        self.cooldown_seconds = 30
    
    def send(self, title: str, message: str, color: int = 0x00ff00):
        """Envia notificação para Discord"""
        if not self.enabled:
            return False
        
        current_time = time.time()
        if current_time - self.last_notification_time < self.cooldown_seconds:
            return False
        
        try:
            embed = {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
            
            if response.status_code == 204:
                self.last_notification_time = current_time
                return True
            return False
                
        except Exception as e:
            log(f"Erro no Discord: {e}", "ERROR")
            return False

discord = DiscordNotifier()

# ===== CONFIGURAÇÃO DCA =====
@dataclass
class DCAConfig:
    """Configuração da estratégia DCA"""
    
    # Asset
    SYMBOL: str = "SOL/USDC:USDC"
    LEVERAGE: int = 5
    
    # Dados históricos
    HISTORICAL_DAYS: int = 30  # Últimos 30 dias
    TIMEFRAME: str = "1d"       # Gráfico de 1 dia
    
    # Degraus de COMPRA (% abaixo do máximo 30 dias)
    # Formato: (% abaixo do máximo, % do capital disponível a investir)
    BUY_STEPS: List[tuple] = None
    
    # Degraus de VENDA (% de ganho da posição)
    # Formato: (% de ganho, % da posição a vender)
    SELL_STEPS: List[tuple] = None
    
    # Cooldowns
    BUY_COOLDOWN_DAYS: int = 5   # Cooldown entre compras (ou até próximo degrau)
    SELL_COOLDOWN_DAYS: int = 3  # Cooldown entre vendas (ou até próximo degrau)
    
    # Gestão de capital
    MIN_ORDER_USD: float = 10.0  # Mínimo $10 para ordem Hyperliquid
    
    def __post_init__(self):
        if self.BUY_STEPS is None:
            # (% abaixo do max, % do capital)
            self.BUY_STEPS = [
                (10, 15),  # -10% do máximo → investe 15% do capital
                (20, 30),  # -20% do máximo → investe 30% do capital
                (30, 50),  # -30% do máximo → investe 50% do capital
                (40, 50),  # -40% do máximo → investe 50% do capital
                (50, 50),  # -50% do máximo → investe 50% do capital
            ]
        
        if self.SELL_STEPS is None:
            # (% de ganho, % da posição a vender)
            self.SELL_STEPS = [
                (10, 20),  # +10% de ganho → vende 20% da posição
                (20, 20),  # +20% de ganho → vende 20% da posição
                (30, 20),  # +30% de ganho → vende 20% da posição
                (40, 20),  # +40% de ganho → vende 20% da posição
                (50, 20),  # +50% de ganho → vende 20% da posição
            ]

# ===== GERENCIADOR DE ESTADO =====
class StateManager:
    """Gerencia o estado da estratégia (últimas operações, cooldowns, etc)"""
    
    def __init__(self, state_file: str = "dca_state.json", exchange=None):
        self.state_file = state_file
        self.state = self.load_state()
        self.exchange = exchange
        
        # Auto-reconstruir estado se necessário
        if exchange and self.needs_reconstruction():
            log("⚠️ Detectado estado inconsistente - iniciando reconstrução...", "WARN")
            self.reconstruct_from_hyperliquid(exchange)
    
    def needs_reconstruction(self) -> bool:
        """Verifica se precisa reconstruir estado a partir da API"""
        try:
            if not self.exchange:
                return False
            
            # Buscar posição atual
            position = self.exchange.get_position()
            
            # Se tem posição aberta mas sem entries no estado, precisa reconstruir
            if position and abs(float(position.get("contracts", 0))) > 0:
                has_entries = len(self.state.get("position_entries", [])) > 0
                if not has_entries:
                    log(f"🔍 Posição detectada ({abs(float(position.get('contracts', 0))):.4f} {position.get('symbol', 'SOL')}) mas estado vazio", "WARN")
                    return True
            
            return False
            
        except Exception as e:
            log(f"Erro verificando necessidade de reconstrução: {e}", "DEBUG")
            return False
    
    def load_state(self) -> dict:
        """Carrega estado do arquivo JSON"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                log(f"Erro carregando estado: {e}", "WARN")
        
        # Estado inicial
        return {
            "last_buy_timestamp": None,
            "last_buy_step": None,  # Último degrau de compra executado
            "last_sell_timestamp": None,
            "last_sell_step": None,  # Último degrau de venda executado
            "position_entries": [],  # Lista de entradas com preço e quantidade
        }
    
    def reconstruct_from_hyperliquid(self, exchange):
        """
        Reconstrói o estado a partir do histórico de trades da Hyperliquid.
        Chamado quando detecta posição aberta mas estado vazio.
        """
        try:
            log("🔍 Reconstruindo estado do histórico Hyperliquid...", "INFO")
            
            # Buscar fills recentes da Hyperliquid
            vault_address = os.getenv("HYPERLIQUID_SUBACCOUNT")
            if not vault_address:
                log("❌ HYPERLIQUID_SUBACCOUNT não configurado", "ERROR")
                return False
            
            # Chamar a função helper global
            fills = _hl_get_user_fills(vault_address, limit=100)
            
            if not fills:
                log("❌ Nenhum fill encontrado no histórico", "WARN")
                return False
            
            log(f"📊 Encontrados {len(fills)} fills no histórico", "INFO")
            
            # Reconstruir position_entries a partir dos fills de compra (Open Long)
            entries = []
            last_buy_time = None
            last_buy_step = None
            last_sell_time = None
            last_sell_step = None
            
            # Processar do mais antigo para o mais recente
            for fill in reversed(fills):
                try:
                    coin = fill.get("coin", "")
                    direction = fill.get("dir", "")
                    px = float(fill.get("px", 0))
                    sz = float(fill.get("sz", 0))
                    time_ms = int(fill.get("time", 0))
                    
                    # Filtrar apenas SOL
                    if coin != "SOL":
                        continue
                    
                    # Converter timestamp
                    timestamp = datetime.fromtimestamp(time_ms / 1000.0)
                    
                    # Detectar COMPRAS: "Open Long" ou "Open Short" (que seria fechar uma short anterior)
                    if "Open Long" in direction or direction == "Open Long":
                        # É uma compra
                        entries.append({
                            "price": px,
                            "amount": sz,
                            "timestamp": timestamp.isoformat()
                        })
                        last_buy_time = timestamp
                        log(f"   🟢 Compra: {sz:.4f} SOL @ ${px:.4f} em {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
                    
                    # Detectar VENDAS: "Close Long" (fechar posição long) ou "Open Short" (vender a descoberto)
                    elif "Close Long" in direction or direction == "Close Long":
                        # É uma venda (fechando long)
                        last_sell_time = timestamp
                        
                        # Remover proporcionalmente dos entries (FIFO - first in, first out)
                        remaining_to_sell = sz
                        i = 0
                        while i < len(entries) and remaining_to_sell > 0:
                            if entries[i]["amount"] <= remaining_to_sell:
                                # Remove entrada completamente
                                remaining_to_sell -= entries[i]["amount"]
                                entries.pop(i)
                            else:
                                # Remove parcialmente
                                entries[i]["amount"] -= remaining_to_sell
                                remaining_to_sell = 0
                                i += 1
                        
                        log(f"   🔴 Venda: {sz:.4f} SOL @ ${px:.4f} em {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
                        
                except Exception as e:
                    log(f"Erro processando fill: {e}", "DEBUG")
                    continue
            
            # Atualizar estado
            if entries:
                self.state["position_entries"] = entries
                avg_price = sum(e["price"] * e["amount"] for e in entries) / sum(e["amount"] for e in entries)
                total_amount = sum(e["amount"] for e in entries)
                log(f"✅ Reconstruído: {len(entries)} entradas, {total_amount:.4f} SOL @ preço médio ${avg_price:.4f}", "INFO")
            
            if last_buy_time:
                self.state["last_buy_timestamp"] = last_buy_time.isoformat()
                # Tentar inferir o degrau de compra baseado no % abaixo do máximo
                if last_buy_step is not None:
                    self.state["last_buy_step"] = last_buy_step
                log(f"✅ Última compra: {last_buy_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
                
            if last_sell_time:
                self.state["last_sell_timestamp"] = last_sell_time.isoformat()
                # Tentar inferir o degrau de venda baseado no ganho obtido
                if last_sell_step is not None:
                    self.state["last_sell_step"] = last_sell_step
                log(f"✅ Última venda: {last_sell_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
            
            self.save_state()
            return True
            
        except Exception as e:
            log(f"❌ Erro reconstruindo estado: {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "DEBUG")
            return False
    
    def save_state(self):
        """Salva estado no arquivo JSON"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            log(f"Erro salvando estado: {e}", "ERROR")
    
    def can_buy(self, current_step: int, cooldown_days: int) -> bool:
        """Verifica se pode comprar (respeita cooldown ou avanço de degrau)"""
        if self.state["last_buy_timestamp"] is None:
            return True
        
        last_timestamp = datetime.fromisoformat(self.state["last_buy_timestamp"])
        time_diff = datetime.now() - last_timestamp
        
        # Se avançou para um degrau maior (pior), pode comprar imediatamente
        last_step = self.state["last_buy_step"]
        if last_step is not None and current_step > last_step:
            log(f"✅ Avanço de degrau: {last_step} → {current_step}, pode comprar", "INFO")
            return True
        
        # Senão, respeita cooldown
        cooldown_passed = time_diff.days >= cooldown_days
        if not cooldown_passed:
            log(f"⏳ Cooldown de compra: {time_diff.days}/{cooldown_days} dias", "DEBUG")
        return cooldown_passed
    
    def can_sell(self, current_step: int, cooldown_days: int) -> bool:
        """Verifica se pode vender (respeita cooldown ou avanço de degrau)"""
        if self.state["last_sell_timestamp"] is None:
            return True
        
        last_timestamp = datetime.fromisoformat(self.state["last_sell_timestamp"])
        time_diff = datetime.now() - last_timestamp
        
        # Se avançou para um degrau maior (melhor lucro), pode vender imediatamente
        last_step = self.state["last_sell_step"]
        if last_step is not None and current_step > last_step:
            log(f"✅ Avanço de degrau: {last_step} → {current_step}, pode vender", "INFO")
            return True
        
        # Não pode vender no mesmo degrau ou inferior dentro do cooldown
        if last_step is not None and current_step <= last_step:
            if time_diff.days < cooldown_days:
                log(f"⏳ Cooldown de venda: degrau {current_step} <= {last_step}, aguardar {cooldown_days - time_diff.days} dias", "DEBUG")
                return False
        
        cooldown_passed = time_diff.days >= cooldown_days
        if not cooldown_passed:
            log(f"⏳ Cooldown de venda: {time_diff.days}/{cooldown_days} dias", "DEBUG")
        return cooldown_passed
    
    def record_buy(self, step: int, price: float, amount: float):
        """Registra uma compra"""
        now = datetime.now()
        self.state["last_buy_timestamp"] = now.isoformat()
        self.state["last_buy_step"] = step
        self.state["position_entries"].append({
            "timestamp": now.isoformat(),
            "step": step,
            "price": price,
            "amount": amount
        })
        self.save_state()
        
        log(f"💾 COMPRA REGISTRADA NO ESTADO:", "INFO")
        log(f"   📅 Data/Hora: {now.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
        log(f"   🎯 Degrau: {step}", "INFO")
        log(f"   💰 Preço: ${price:.4f}", "INFO")
        log(f"   🪙 Quantidade: {amount:.4f} SOL", "INFO")
        log(f"   ⏰ Próxima compra: após {now + timedelta(days=5)} (ou avanço de degrau)", "INFO")
    
    def record_sell(self, step: int):
        """Registra uma venda"""
        self.state["last_sell_timestamp"] = datetime.now().isoformat()
        self.state["last_sell_step"] = step
        self.save_state()
    
    def get_average_entry_price(self) -> float:
        """
        Calcula preço médio de entrada baseado nas entradas registradas.
        Se não houver entries mas houver posição, tenta buscar da API como fallback.
        """
        entries = self.state["position_entries"]
        if not entries:
            # Fallback: tentar buscar da posição atual se exchange está disponível
            if self.exchange:
                try:
                    position = self.exchange.get_position()
                    if position and position.get("size", 0) > 0:
                        entry_price = position.get("entryPrice", 0)
                        if entry_price > 0:
                            log(f"⚠️ Usando entry_price da posição como fallback: ${entry_price:.4f}", "WARN")
                            return float(entry_price)
                except Exception as e:
                    log(f"Erro buscando entry_price da posição: {e}", "DEBUG")
            
            return 0.0
        
        total_value = sum(e["price"] * e["amount"] for e in entries)
        total_amount = sum(e["amount"] for e in entries)
        
        return total_value / total_amount if total_amount > 0 else 0.0
    
    def show_state_summary(self):
        """Mostra resumo do estado atual"""
        log("", "INFO")
        log("📋 ESTADO ATUAL DO SISTEMA:", "INFO")
        
        # Última compra
        if self.state["last_buy_timestamp"]:
            last_buy = datetime.fromisoformat(self.state["last_buy_timestamp"])
            days_since_buy = (datetime.now() - last_buy).days
            log(f"   🟢 Última compra: {last_buy.strftime('%Y-%m-%d %H:%M:%S')} ({days_since_buy} dias atrás)", "INFO")
            log(f"      Degrau: {self.state['last_buy_step']}", "INFO")
        else:
            log(f"   🟢 Última compra: Nenhuma", "INFO")
        
        # Última venda
        if self.state["last_sell_timestamp"]:
            last_sell = datetime.fromisoformat(self.state["last_sell_timestamp"])
            days_since_sell = (datetime.now() - last_sell).days
            log(f"   🔴 Última venda: {last_sell.strftime('%Y-%m-%d %H:%M:%S')} ({days_since_sell} dias atrás)", "INFO")
            log(f"      Degrau: {self.state['last_sell_step']}", "INFO")
        else:
            log(f"   🔴 Última venda: Nenhuma", "INFO")
        
        # Entradas registradas
        entries = self.state["position_entries"]
        if entries:
            log(f"   📊 Total de entradas: {len(entries)}", "INFO")
            for i, entry in enumerate(entries, 1):
                entry_time = datetime.fromisoformat(entry["timestamp"])
                log(f"      {i}. {entry_time.strftime('%Y-%m-%d %H:%M')} | Degrau {entry.get('step', '?')} | ${entry['price']:.4f} | {entry['amount']:.4f} SOL", "INFO")
        else:
            log(f"   📊 Total de entradas: 0", "INFO")
        
        log("", "INFO")

# ===== CONEXÃO COM EXCHANGES =====
class ExchangeConnector:
    """Gerencia conexões com Binance (dados) e Hyperliquid (execução)"""
    
    def __init__(self, cfg: DCAConfig):
        self.cfg = cfg
        
        # Binance para dados históricos
        self.binance = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY', ''),
            'secret': os.getenv('BINANCE_API_SECRET', ''),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Hyperliquid para execução
        wallet_address = os.getenv("WALLET_ADDRESS", "")
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY") or os.getenv("PRIVATE_KEY", "")
        vault_address = os.getenv("HYPERLIQUID_SUBACCOUNT") or os.getenv("VAULT_ADDRESS", "")  # Subconta
        
        if not wallet_address or not private_key:
            raise ValueError("WALLET_ADDRESS e HYPERLIQUID_PRIVATE_KEY devem estar configurados")
        
        self.hyperliquid = ccxt.hyperliquid({
            'walletAddress': wallet_address,
            'privateKey': private_key,
            'enableRateLimit': True,
        })
        
        # Usar subconta se configurada
        if vault_address:
            self.hyperliquid.options['vaultAddress'] = vault_address
            log(f"🏦 Usando subconta (vault): {vault_address}", "INFO")
        
        self.wallet_address = vault_address if vault_address else wallet_address
        
        log("✅ Conexões estabelecidas: Binance (dados) + Hyperliquid (execução)", "INFO")
    
    def fetch_historical_data(self, days: int) -> pd.DataFrame:
        """Busca dados históricos da Binance"""
        try:
            # Binance usa SOLUSDT para futuros
            symbol_binance = "SOL/USDT:USDT"
            
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            ohlcv = self.binance.fetch_ohlcv(
                symbol_binance,
                timeframe=self.cfg.TIMEFRAME,
                since=since,
                limit=days + 5  # Alguns dias extras para garantir
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            log(f"📊 Dados históricos: {len(df)} candles, {df['timestamp'].min()} até {df['timestamp'].max()}", "INFO")
            
            return df
            
        except Exception as e:
            log(f"❌ Erro buscando dados históricos: {e}", "ERROR")
            return pd.DataFrame()
    
    def get_current_price(self) -> float:
        """Busca preço atual do SOL"""
        try:
            ticker = self.binance.fetch_ticker("SOL/USDT:USDT")
            price = ticker['last']
            log(f"💰 Preço atual SOL: ${price:.4f}", "DEBUG")
            return price
        except Exception as e:
            log(f"❌ Erro buscando preço: {e}", "ERROR")
            return 0.0
    
    def get_balance(self) -> float:
        """Retorna saldo disponível na conta Hyperliquid"""
        try:
            balance = _hl_get_account_value(self.wallet_address) if self.wallet_address else 0.0
            log(f"💵 Saldo disponível: ${balance:.2f}", "DEBUG")
            return balance
        except Exception as e:
            log(f"❌ Erro buscando saldo: {e}", "ERROR")
            return 0.0
    
    def get_position(self) -> Optional[Dict]:
        """Retorna posição aberta (se houver) via API direta Hyperliquid"""
        try:
            # Usar API direta para obter estado do usuário
            user_state = _hl_get_user_state(self.wallet_address)
            
            if not user_state or "assetPositions" not in user_state:
                log("⚠️ Nenhuma posição encontrada na resposta da API", "DEBUG")
                return None
            
            asset_positions = user_state["assetPositions"]
            
            # Procurar posição do SOL
            coin_name = self.cfg.SYMBOL.replace('/USDC:USDC', '')  # "SOL"
            
            for pos in asset_positions:
                position_coin = pos.get("position", {}).get("coin", "")
                size = float(pos.get("position", {}).get("szi", 0))
                
                if position_coin == coin_name and abs(size) > 0:
                    # Converter para formato compatível
                    entry_px = float(pos.get("position", {}).get("entryPx", 0))
                    
                    log(f"✅ Posição encontrada: {size} {coin_name} @ ${entry_px:.4f}", "DEBUG")
                    
                    return {
                        "symbol": self.cfg.SYMBOL,
                        "contracts": size,
                        "entryPrice": entry_px,
                        "side": "long" if size > 0 else "short",
                        "unrealizedPnl": float(pos.get("position", {}).get("unrealizedPnl", 0)),
                        "marginUsed": float(pos.get("position", {}).get("marginUsed", 0)),
                    }
            
            log("⚠️ Nenhuma posição de SOL encontrada", "DEBUG")
            return None
            
        except Exception as e:
            log(f"❌ Erro buscando posição: {e}", "ERROR")
            return None
    
    def create_market_order(self, side: str, amount_usd: float, leverage: int) -> bool:
        """Cria ordem market na Hyperliquid"""
        try:
            # Configurar leverage primeiro
            log(f"🔧 Configurando leverage {leverage}x para {self.cfg.SYMBOL}", "DEBUG")
            self.hyperliquid.set_leverage(leverage, self.cfg.SYMBOL, {"marginMode": "isolated"})
            
            # Buscar preço atual
            current_price = self.get_current_price()
            if current_price <= 0:
                log("❌ Preço inválido, não é possível criar ordem", "ERROR")
                return False
            
            # Calcular quantidade de SOL com alavancagem
            # Fórmula: amount = (USD_a_gastar * leverage) / preço
            notional = amount_usd * leverage
            amount = notional / current_price
            
            # Arredondar quantidade conforme precisão do mercado
            amount = float(self.hyperliquid.amount_to_precision(self.cfg.SYMBOL, amount))
            
            # Verificar valor mínimo
            notional_value = amount * current_price
            if notional_value < self.cfg.MIN_ORDER_USD:
                log(f"❌ Valor nocional muito baixo: ${notional_value:.2f} < ${self.cfg.MIN_ORDER_USD}", "ERROR")
                return False
            
            log(f"📤 Criando ordem: {side} {amount:.4f} SOL", "INFO")
            log(f"   💰 USD investidos: ${amount_usd:.2f}", "INFO")
            log(f"   📊 Leverage: {leverage}x", "INFO")
            log(f"   💵 Valor nocional: ${notional:.2f}", "INFO")
            log(f"   📈 Preço: ${current_price:.4f}", "INFO")
            
            # Criar ordem market (Hyperliquid exige price para calcular slippage)
            order = self.hyperliquid.create_order(
                symbol=self.cfg.SYMBOL,
                type='market',
                side=side,
                amount=amount,
                price=current_price,  # Necessário para Hyperliquid calcular slippage
                params={}
            )
            
            log(f"✅ Ordem criada com sucesso: {order}", "INFO")
            return True
            
        except Exception as e:
            log(f"❌ Erro criando ordem: {e}", "ERROR")
            return False
    
    def close_position_partial(self, percentage: float) -> bool:
        """Fecha parcialmente a posição (percentage = 0-100)"""
        try:
            pos = self.get_position()
            if not pos:
                log("⚠️ Nenhuma posição aberta para fechar", "WARN")
                return False
            
            current_amount = abs(float(pos.get('contracts', 0)))
            amount_to_close = current_amount * (percentage / 100.0)
            
            # Arredondar
            amount_to_close = float(self.hyperliquid.amount_to_precision(self.cfg.SYMBOL, amount_to_close))
            
            # Verificar mínimo
            current_price = self.get_current_price()
            notional = amount_to_close * current_price
            
            if notional < self.cfg.MIN_ORDER_USD:
                log(f"⚠️ Ordem muito pequena (${notional:.2f} < ${self.cfg.MIN_ORDER_USD}), pulando", "WARN")
                return False
            
            log(f"📤 Fechando {percentage:.0f}% da posição ({amount_to_close:.4f} SOL)", "INFO")
            
            # Fechar posição (ordem market com reduceOnly - Hyperliquid exige price)
            order = self.hyperliquid.create_order(
                symbol=self.cfg.SYMBOL,
                type='market',
                side='sell',  # Sempre sell para fechar LONG
                amount=amount_to_close,
                price=current_price,  # Necessário para Hyperliquid calcular slippage
                params={'reduceOnly': True}
            )
            
            log(f"✅ Posição fechada parcialmente: {order}", "INFO")
            return True
            
        except Exception as e:
            log(f"❌ Erro fechando posição: {e}", "ERROR")
            return False

# ===== ESTRATÉGIA DCA =====
class DCAStrategy:
    """Estratégia DCA com degraus de compra e venda"""
    
    def __init__(self, cfg: DCAConfig):
        self.cfg = cfg
        self.exchange = ExchangeConnector(cfg)
        # StateManager precisa do exchange para reconstruir estado
        self.state = StateManager(exchange=self.exchange)
    
    def analyze_market(self) -> Dict[str, Any]:
        """Analisa o mercado e retorna informações"""
        # Buscar dados históricos
        df = self.exchange.fetch_historical_data(self.cfg.HISTORICAL_DAYS)
        
        if df.empty:
            log("❌ Sem dados históricos disponíveis", "ERROR")
            return {}
        
        # Calcular máximo dos últimos 30 dias
        max_price_30d = df['high'].max()
        
        # Preço atual
        current_price = self.exchange.get_current_price()
        
        if current_price <= 0 or max_price_30d <= 0:
            log("❌ Preços inválidos", "ERROR")
            return {}
        
        # Calcular % abaixo do máximo
        pct_below_max = ((max_price_30d - current_price) / max_price_30d) * 100
        
        # Posição atual
        position = self.exchange.get_position()
        
        # Saldo disponível
        balance = self.exchange.get_balance()
        
        analysis = {
            "current_price": current_price,
            "max_price_30d": max_price_30d,
            "pct_below_max": pct_below_max,
            "position": position,
            "balance": balance,
            "has_position": position is not None,
        }
        
        # Se tem posição, calcular % de ganho REAL (considerando alavancagem)
        if position:
            entry_price = self.state.get_average_entry_price()
            position_size = abs(float(position.get('contracts', 0)))
            position_value = position_size * current_price
            
            # PNL real da posição (já considera alavancagem)
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            margin_used = float(position.get('marginUsed', 0))
            
            # % de ganho baseado no PNL real sobre a margem usada
            # Isso já reflete o impacto da alavancagem!
            if margin_used > 0:
                pct_gain_real = (unrealized_pnl / margin_used) * 100
            else:
                # Fallback: calcular manualmente com alavancagem
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                pct_gain_real = price_change_pct * self.cfg.LEVERAGE
            
            # Variação de preço (para referência)
            price_diff = current_price - entry_price
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            analysis["entry_price"] = entry_price
            analysis["pct_gain"] = pct_gain_real  # Usar o PNL real!
            analysis["position_size"] = position_size
            analysis["position_value"] = position_value
            analysis["unrealized_pnl"] = unrealized_pnl
            
            # Log detalhado da posição
            log(f"", "INFO")
            log(f"📈 POSIÇÃO ABERTA:", "INFO")
            log(f"   🪙 Quantidade: {position_size:.4f} SOL", "INFO")
            log(f"   💰 Preço de entrada: ${entry_price:.4f}", "INFO")
            log(f"   📊 Preço atual: ${current_price:.4f}", "INFO")
            log(f"    Variação de preço: ${price_diff:+.4f} ({price_change_pct:+.2f}%)", "INFO")
            log(f"   💵 Valor atual da posição: ${position_value:.2f}", "INFO")
            log(f"   {'📈' if pct_gain_real >= 0 else '📉'} **PNL REAL (c/ {self.cfg.LEVERAGE}x leverage): {pct_gain_real:+.2f}%** | ${unrealized_pnl:+.2f}", "INFO")
            
            # Mostrar próximos degraus de venda
            next_sell_step = None
            for i, (threshold, _) in enumerate(self.cfg.SELL_STEPS):
                if pct_gain_real < threshold:
                    next_sell_step = threshold
                    break
            
            if next_sell_step:
                points_to_next = next_sell_step - pct_gain_real
                log(f"   🎯 Próximo degrau de venda: +{next_sell_step}% PNL (faltam {points_to_next:.2f}%)", "INFO")
            else:
                log(f"   ✅ Acima de todos os degraus de venda!", "INFO")
        
        log(f"📊 Análise: Preço=${current_price:.4f} | Max 30d=${max_price_30d:.4f} | "
            f"Abaixo do max={pct_below_max:.2f}% | Posição={'SIM' if position else 'NÃO'}", "INFO")
        
        return analysis
    
    def check_buy_signals(self, analysis: Dict) -> Optional[int]:
        """Verifica se deve comprar e retorna o degrau"""
        pct_below_max = analysis.get("pct_below_max", 0)
        
        # Verificar cada degrau de compra (do maior para o menor threshold)
        # Precisamos manter o índice original para identificar corretamente o degrau
        for step_idx in range(len(self.cfg.BUY_STEPS) - 1, -1, -1):  # De trás pra frente
            threshold, capital_pct = self.cfg.BUY_STEPS[step_idx]
            
            if pct_below_max >= threshold:
                # Verificar se pode comprar (cooldown)
                if self.state.can_buy(step_idx, self.cfg.BUY_COOLDOWN_DAYS):
                    log(f"🚨 SINAL DE COMPRA: Degrau {step_idx} ativado ({pct_below_max:.2f}% >= {threshold}%) → {capital_pct}% do capital", "INFO")
                    return step_idx
                else:
                    log(f"⏳ Degrau {step_idx} ativado mas em cooldown", "DEBUG")
        
        return None
    
    def check_sell_signals(self, analysis: Dict) -> Optional[int]:
        """Verifica se deve vender e retorna o degrau"""
        if not analysis.get("has_position"):
            return None
        
        pct_gain = analysis.get("pct_gain", 0)
        
        # Verificar cada degrau de venda (do maior para o menor threshold)
        # Precisamos manter o índice original para identificar corretamente o degrau
        for step_idx in range(len(self.cfg.SELL_STEPS) - 1, -1, -1):  # De trás pra frente
            threshold, position_pct = self.cfg.SELL_STEPS[step_idx]
            
            if pct_gain >= threshold:
                # Verificar se pode vender (cooldown)
                if self.state.can_sell(step_idx, self.cfg.SELL_COOLDOWN_DAYS):
                    log(f"🚨 SINAL DE VENDA: Degrau {step_idx} ativado ({pct_gain:.2f}% >= {threshold}%) → {position_pct}% da posição", "INFO")
                    return step_idx
                else:
                    log(f"⏳ Degrau {step_idx} ativado mas em cooldown", "DEBUG")
        
        return None
    
    def execute_buy(self, step: int, analysis: Dict) -> bool:
        """Executa compra no degrau especificado"""
        threshold, capital_pct = self.cfg.BUY_STEPS[step]
        
        balance = analysis["balance"]
        current_price = analysis["current_price"]
        
        # Calcular quanto investir (em USD da carteira)
        amount_usd = balance * (capital_pct / 100.0)
        
        # Com leverage 5x: se investe $4, valor nocional = $20
        notional_value = amount_usd * self.cfg.LEVERAGE
        amount_coins = notional_value / current_price
        
        # IMPORTANTE: Verificar mínimo usando valor NOCIONAL (alavancado), não valor da carteira
        if notional_value < self.cfg.MIN_ORDER_USD:
            log(f"⚠️ Valor nocional muito baixo: ${notional_value:.2f} < ${self.cfg.MIN_ORDER_USD}", "WARN")
            log(f"   (${amount_usd:.2f} da carteira × {self.cfg.LEVERAGE}x leverage = ${notional_value:.2f})", "WARN")
            return False
        
        log(f"🟢 COMPRANDO: Degrau {step} ({threshold}% abaixo do máximo)", "INFO")
        log(f"   💰 Capital da carteira: ${balance:.2f}", "INFO")
        log(f"   📊 {capital_pct}% do capital = ${amount_usd:.2f}", "INFO")
        log(f"   🔧 Leverage {self.cfg.LEVERAGE}x → Valor nocional: ${notional_value:.2f}", "INFO")
        log(f"   🪙 Quantidade SOL: {amount_coins:.4f} @ ${current_price:.4f}", "INFO")
        
        # Executar ordem
        success = self.exchange.create_market_order("buy", amount_usd, self.cfg.LEVERAGE)
        
        if success:
            # Registrar compra (usar valor nocional para cálculo correto)
            self.state.record_buy(step, current_price, amount_coins)
            
            # Notificar Discord
            discord.send(
                "🟢 COMPRA EXECUTADA",
                f"**Degrau:** {step} ({threshold}% abaixo do máximo)\n"
                f"**Preço:** ${current_price:.4f}\n"
                f"**Capital usado:** ${amount_usd:.2f} ({capital_pct}% do saldo)\n"
                f"**Leverage:** {self.cfg.LEVERAGE}x\n"
                f"**Valor nocional:** ${notional_value:.2f}\n"
                f"**Quantidade SOL:** {amount_coins:.4f}",
                0x00ff00
            )
        
        return success
    
    def execute_sell(self, step: int, analysis: Dict) -> bool:
        """Executa venda no degrau especificado"""
        threshold, position_pct = self.cfg.SELL_STEPS[step]
        
        pct_gain = analysis.get("pct_gain", 0)
        current_price = analysis["current_price"]
        
        log(f"🔴 VENDENDO: Degrau {step} | {position_pct}% da posição | Ganho {pct_gain:.2f}%", "INFO")
        
        # Executar ordem
        success = self.exchange.close_position_partial(position_pct)
        
        if success:
            # Registrar venda
            self.state.record_sell(step)
            
            # Notificar Discord
            discord.send(
                "🔴 VENDA EXECUTADA",
                f"**Degrau:** {step} ({threshold}% de ganho)\n"
                f"**Preço:** ${current_price:.4f}\n"
                f"**Ganho:** +{pct_gain:.2f}%\n"
                f"**Posição vendida:** {position_pct}%",
                0xff9900
            )
        
        return success
    
    def run_cycle(self):
        """Executa um ciclo da estratégia"""
        log("=" * 80, "INFO")
        log("🔄 INICIANDO CICLO DCA", "INFO")
        log("=" * 80, "INFO")
        
        # Mostrar estado atual
        self.state.show_state_summary()
        
        try:
            # Analisar mercado
            analysis = self.analyze_market()
            
            if not analysis:
                log("⚠️ Análise falhou, pulando ciclo", "WARN")
                return
            
            # Verificar sinais de compra
            buy_step = self.check_buy_signals(analysis)
            if buy_step is not None:
                self.execute_buy(buy_step, analysis)
            
            # Verificar sinais de venda
            sell_step = self.check_sell_signals(analysis)
            if sell_step is not None:
                self.execute_sell(sell_step, analysis)
            
            log("✅ Ciclo concluído", "INFO")
            
        except Exception as e:
            log(f"❌ Erro no ciclo: {e}", "ERROR")
            import traceback
            traceback.print_exc()

# ===== MAIN =====
def main():
    """Função principal"""
    setup_log_file()
    
    log("=" * 80, "INFO")
    log("🚀 INICIANDO SISTEMA DE TRADING DCA - SOL LONG ONLY", "INFO")
    log("=" * 80, "INFO")
    
    # Carregar variáveis de ambiente
    try:
        from dotenv import load_dotenv
        # Tentar carregar .env.dca primeiro, senão .env padrão
        if os.path.exists('.env.dca'):
            load_dotenv('.env.dca')
            log("✅ Carregado .env.dca", "INFO")
        else:
            load_dotenv()
            log("✅ Carregado .env", "INFO")
    except ImportError:
        log("⚠️ python-dotenv não instalado, usando variáveis de ambiente do sistema", "WARN")
    
    # Configuração
    cfg = DCAConfig()
    
    log(f"⚙️  Configuração:", "INFO")
    log(f"   Asset: {cfg.SYMBOL} ({cfg.LEVERAGE}x leverage)", "INFO")
    log(f"   Histórico: {cfg.HISTORICAL_DAYS} dias ({cfg.TIMEFRAME})", "INFO")
    log(f"   Degraus de compra: {cfg.BUY_STEPS}", "INFO")
    log(f"   Degraus de venda: {cfg.SELL_STEPS}", "INFO")
    log(f"   Cooldown compra: {cfg.BUY_COOLDOWN_DAYS} dias", "INFO")
    log(f"   Cooldown venda: {cfg.SELL_COOLDOWN_DAYS} dias", "INFO")
    
    # Criar estratégia
    strategy = DCAStrategy(cfg)
    
    # Loop principal
    log("🔁 Entrando no loop principal (Ctrl+C para parar)", "INFO")
    
    # Intervalo de verificação (a cada 60 segundos para monitoramento rápido)
    check_interval = 60  # 60 segundos
    
    try:
        while True:
            strategy.run_cycle()
            
            log(f"⏰ Próximo ciclo em {check_interval} segundos...", "INFO")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        log("🛑 Sistema interrompido pelo usuário", "INFO")
    except Exception as e:
        log(f"❌ Erro fatal: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    
    log("👋 Sistema encerrado", "INFO")

if __name__ == "__main__":
    main()