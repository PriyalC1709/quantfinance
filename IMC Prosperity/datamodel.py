"""
Local stub for Prosperity's datamodel module.
This file should sit alongside trader.py for local development/testing.
Do NOT upload this file — Prosperity injects the real one at runtime.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


UserId = str


@dataclass
class Order:
    symbol: str
    price: int
    quantity: int

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({side} {abs(self.quantity)}x {self.symbol} @ {self.price})"


class OrderDepth:
    def __init__(self):
        self.buy_orders:  Dict[int, int] = {}   # price -> positive qty
        self.sell_orders: Dict[int, int] = {}   # price -> negative qty


@dataclass
class Trade:
    symbol: str
    price: int
    quantity: int
    buyer:  str = ""
    seller: str = ""


class TradingState:
    def __init__(
        self,
        traderData: str,
        timestamp: int,
        listings: Dict,
        order_depths: Dict[str, OrderDepth],
        own_trades: Dict[str, List[Trade]],
        market_trades: Dict[str, List[Trade]],
        position: Dict[str, int],
        observations: Any,
    ):
        self.traderData    = traderData
        self.timestamp     = timestamp
        self.listings      = listings
        self.order_depths  = order_depths
        self.own_trades    = own_trades
        self.market_trades = market_trades
        self.position      = position
        self.observations  = observations


class Observation:
    def __init__(self):
        self.plainValueObservations: Dict = {}
        self.conversionObservations: Dict = {}