from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional
import json

"""
IMC Prosperity Round 2 — FINAL v2
===================================
Changes from v4 (291455):
- bid() returns 0 (was 3500): saves 3500 XIRECs, MAF not worth it since
  we are position-limited not flow-limited on IPR
- IPR entry: kept aggressive take (reverted passive attempt from final v1)
  Passive entry failed in 291736 — no fills until ts=5000, avg cost rose
  by 3.6 ticks × 80 units = -288 XIRECs, worse than the spread saved.
  Aggressive take at fair+9 is more reliable across random seeds.

ACO unchanged from v4 (best ACO result so far):
- One-sided quoting when |pos| > 8
- Skew factor 0.10
- Drift stable at ~-18 across runs
"""

ACO = "ASH_COATED_OSMIUM"
IPR = "INTARIAN_PEPPER_ROOT"
POS_LIMIT = 80

# IPR parameters
IPR_TAKE_BUFFER = 9       # take any ask at or below fair+9 (~83% of asks)
IPR_PASSIVE_VOL = 20      # passive bid size after aggressive fill

# ACO parameters
ACO_FAIR = 10_000
ACO_MAKE_VOL = 15
ACO_ONE_SIDE_THRESHOLD = 8
ACO_SKEW_FACTOR = 0.10


def ipr_fair_value(day: int, timestamp: int) -> float:
    return (day + 2) * 1000 + 10_000 + timestamp * 0.001


def infer_day(mid: float, timestamp: int) -> int:
    day_float = (mid - 10_000 - timestamp * 0.001) / 1000.0 - 2.0
    return max(-1, min(1, int(round(day_float))))


class Trader:

    def bid(self) -> int:
        """
        MAF = 0. Extra 25% flow worth ~1089 XIRECs over 3 days vs 3500 cost.
        We fill IPR to +80 by ts~800 regardless — position-limited not flow-limited.
        """
        return 0

    def run(self, state: TradingState):
        inferred_day: Optional[int] = None
        if state.traderData:
            try:
                d = json.loads(state.traderData)
                inferred_day = d.get("day", None)
            except Exception:
                pass

        result: Dict[str, List[Order]] = {}

        if ACO in state.order_depths:
            result[ACO] = self._trade_aco(state)

        if IPR in state.order_depths:
            ipr_orders, inferred_day = self._trade_ipr(state, inferred_day)
            result[IPR] = ipr_orders

        traderData = json.dumps({"day": inferred_day})
        return result, 0, traderData

    # ──────────────────────────────────────────────────────────────────────
    # ACO: one-sided quoting based on inventory position
    # When short: only post bids to get back to flat
    # When long:  only post asks to get back to flat
    # When near flat: post both sides with inventory skew
    # ──────────────────────────────────────────────────────────────────────
    def _trade_aco(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        od: OrderDepth = state.order_depths[ACO]
        pos = state.position.get(ACO, 0)
        buy_capacity  = POS_LIMIT - pos
        sell_capacity = POS_LIMIT + pos
        fair = ACO_FAIR

        if not od.buy_orders and not od.sell_orders:
            return orders

        best_bid_p = max(od.buy_orders.keys()) if od.buy_orders else fair - 8
        best_ask_p = min(od.sell_orders.keys()) if od.sell_orders else fair + 8

        skew = int(pos * ACO_SKEW_FACTOR)

        if pos > ACO_ONE_SIDE_THRESHOLD:
            our_ask = max(best_ask_p - 1 - skew, fair + 1)
            if sell_capacity > 0:
                orders.append(Order(ACO, our_ask, -min(sell_capacity, ACO_MAKE_VOL)))

        elif pos < -ACO_ONE_SIDE_THRESHOLD:
            our_bid = min(best_bid_p + 1 - skew, fair - 1)
            if buy_capacity > 0:
                orders.append(Order(ACO, our_bid, min(buy_capacity, ACO_MAKE_VOL)))

        else:
            our_bid = min(best_bid_p + 1 - skew, fair - 1)
            our_ask = max(best_ask_p - 1 - skew, fair + 1)

            if our_bid >= our_ask:
                our_bid = fair - 2
                our_ask = fair + 2

            if buy_capacity > 0:
                orders.append(Order(ACO, our_bid, min(buy_capacity, ACO_MAKE_VOL)))
            if sell_capacity > 0:
                orders.append(Order(ACO, our_ask, -min(sell_capacity, ACO_MAKE_VOL)))

        return orders

    # ──────────────────────────────────────────────────────────────────────
    # IPR: trend following — get to max long as fast as possible
    # Aggressive take: hit any ask at or below fair+9 from ts=0
    # Passive bid: also post best_bid+1 to catch any remaining flow
    # Only sell if someone bids wildly above fair (>fair+12)
    # ──────────────────────────────────────────────────────────────────────
    def _trade_ipr(self, state: TradingState, inferred_day: Optional[int]):
        orders: List[Order] = []
        od: OrderDepth = state.order_depths[IPR]
        pos = state.position.get(IPR, 0)
        buy_capacity  = POS_LIMIT - pos
        sell_capacity = POS_LIMIT + pos
        ts = state.timestamp

        if od.buy_orders and od.sell_orders:
            best_bid_p = max(od.buy_orders.keys())
            best_ask_p = min(od.sell_orders.keys())
            mid = (best_bid_p + best_ask_p) / 2.0
            inferred_day = infer_day(mid, ts)

        if inferred_day is None:
            return orders, inferred_day

        fair = ipr_fair_value(inferred_day, ts)

        # Aggressive take: lift any ask at or below fair+9
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price <= fair + IPR_TAKE_BUFFER:
                    vol = min(-od.sell_orders[ask_price], buy_capacity)
                    if vol > 0:
                        orders.append(Order(IPR, ask_price, vol))
                        buy_capacity -= vol
                        if buy_capacity == 0:
                            break
                else:
                    break

        # Passive bid to catch any remaining flow
        if buy_capacity > 0 and od.buy_orders:
            best_bid_p = max(od.buy_orders.keys())
            orders.append(Order(IPR, best_bid_p + 1,
                                min(buy_capacity, IPR_PASSIVE_VOL)))

        # Only sell if bid is wildly above fair
        if od.buy_orders and pos > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair + 12:
                    vol = min(od.buy_orders[bid_price], sell_capacity)
                    if vol > 0:
                        orders.append(Order(IPR, bid_price, -vol))
                        sell_capacity -= vol
                else:
                    break

        return orders, inferred_day