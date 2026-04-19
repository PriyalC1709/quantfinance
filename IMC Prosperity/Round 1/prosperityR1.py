from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json


class Trader:

    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    POSITION_LIMIT = 80

    OSMIUM_FAIR = 10_000

    # Mean reversion thresholds
    # Buy when ask is at or below this (dislocated low)
    OSMIUM_BUY_BELOW  = 10_000
    # Sell when bid is at or above this (dislocated high)
    OSMIUM_SELL_ABOVE = 10_000

    # Passive MM as fallback (±3 from fair — proven best in v1/v2)
    OSMIUM_BID_OFF = 3
    OSMIUM_ASK_OFF = 3

    PEPPER_TARGET    = 79
    PEPPER_MAX_CHASE = 8

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        od = state.order_depths

        pos_o = state.position.get(self.OSMIUM, 0) if state.position else 0
        pos_p = state.position.get(self.PEPPER,  0) if state.position else 0

        if self.OSMIUM in od:
            result[self.OSMIUM] = self._trade_osmium(od[self.OSMIUM], pos_o)
        if self.PEPPER in od:
            result[self.PEPPER] = self._trade_pepper(od[self.PEPPER], pos_p)

        return result, 0, json.dumps({"o": pos_o, "p": pos_p})

    # ------------------------------------------------------------------
    # Osmium: mean reversion taking + passive MM fallback
    # ------------------------------------------------------------------

    def _trade_osmium(self, depth: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMIT

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        fair = (best_bid + best_ask) / 2 if (best_bid and best_ask) else self.OSMIUM_FAIR

        # 1. MEAN REVERSION TAKING
        # If ask is at or below fair (dislocated low) → buy everything available
        if best_ask and best_ask <= self.OSMIUM_BUY_BELOW:
            for ask_px in sorted(depth.sell_orders.keys()):
                if ask_px > self.OSMIUM_BUY_BELOW:
                    break
                vol = min(-depth.sell_orders[ask_px], limit - pos)
                if vol <= 0:
                    break
                orders.append(Order(self.OSMIUM, ask_px, vol))
                pos += vol

        # If bid is at or above fair (dislocated high) → sell everything available
        if best_bid and best_bid >= self.OSMIUM_SELL_ABOVE:
            for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_px < self.OSMIUM_SELL_ABOVE:
                    break
                vol = min(depth.buy_orders[bid_px], pos + limit)
                if vol <= 0:
                    break
                orders.append(Order(self.OSMIUM, bid_px, -vol))
                pos -= vol

        # 2. PASSIVE MM FALLBACK — catches residual flow when not dislocated
        # Only post if we have capacity and book is in normal range
        bid_px = int(fair - self.OSMIUM_BID_OFF)
        ask_px = int(fair + self.OSMIUM_ASK_OFF)

        buy_cap  = limit - pos
        sell_cap = pos + limit

        # Taper passive quotes based on how much position we've built
        # If long (from reversion buys), reduce passive bids
        # If short (from reversion sells), reduce passive asks
        passive_bid_cap = min(10, buy_cap) if pos < 40 else min(5, buy_cap)
        passive_ask_cap = min(10, sell_cap) if pos > -40 else min(5, sell_cap)

        if passive_bid_cap > 0:
            orders.append(Order(self.OSMIUM, bid_px, passive_bid_cap))
        if passive_ask_cap > 0:
            orders.append(Order(self.OSMIUM, ask_px, -passive_ask_cap))

        return orders

    # ------------------------------------------------------------------
    # Pepper: hold 79 units, don't chase expensive asks
    # ------------------------------------------------------------------

    def _trade_pepper(self, depth: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMIT

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        if not best_bid and not best_ask:
            return orders

        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else (best_bid or best_ask)

        # Buy to target, only if ask within range of mid
        if pos < self.PEPPER_TARGET and depth.sell_orders:
            for ask_px in sorted(depth.sell_orders.keys()):
                if ask_px > mid + self.PEPPER_MAX_CHASE:
                    break
                want      = self.PEPPER_TARGET - pos
                available = -depth.sell_orders[ask_px]
                vol = min(available, want, limit - pos)
                if vol <= 0:
                    break
                orders.append(Order(self.PEPPER, ask_px, vol))
                pos += vol
                if pos >= self.PEPPER_TARGET:
                    break

        # Passive bid at best_bid if still below target
        if pos < self.PEPPER_TARGET and best_bid:
            orders.append(Order(self.PEPPER, best_bid, min(15, limit - pos)))

        # Trim only at hard limit
        if pos >= limit and depth.buy_orders:
            for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
                want = pos - self.PEPPER_TARGET
                vol  = min(depth.buy_orders[bid_px], want)
                if vol <= 0:
                    break
                orders.append(Order(self.PEPPER, bid_px, -vol))
                pos -= vol
                if pos <= self.PEPPER_TARGET:
                    break

        return orders