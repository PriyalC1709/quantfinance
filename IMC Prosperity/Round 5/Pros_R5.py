from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

POS_LIMIT = 10

DIRECTIONAL_TARGETS = {
    # PEBBLES — XS/S short, XL long confirmed. L and M neutral.
    "PEBBLES_XS":   -10,
    "PEBBLES_S":    -10,
    "PEBBLES_M":      0,
    "PEBBLES_L":      0,   # cut: lost 662 on day 4
    "PEBBLES_XL":   +10,
 
    # SLEEP_POD — 4 of 5 confirmed uptrend. LAMB_WOOL cut.
    "SLEEP_POD_SUEDE":      +10,
    "SLEEP_POD_LAMB_WOOL":    0,   # cut: lost 5,954 on day 4
    "SLEEP_POD_POLYESTER":  +10,
    "SLEEP_POD_NYLON":      +10,
    "SLEEP_POD_COTTON":     +10,
 
    # UV_VISOR — RED and AMBER confirmed. MAGENTA cut.
    "UV_VISOR_AMBER":    -10,
    "UV_VISOR_RED":      +10,
    "UV_VISOR_MAGENTA":    0,   # cut: lost 3,767 on day 4
    "UV_VISOR_YELLOW":     0,
    "UV_VISOR_ORANGE":     0,
 
    # OXYGEN_SHAKE — GARLIC confirmed strong uptrend
    "OXYGEN_SHAKE_GARLIC":         +10,
    "OXYGEN_SHAKE_MORNING_BREATH":   0,
    "OXYGEN_SHAKE_EVENING_BREATH":   0,
    "OXYGEN_SHAKE_MINT":             0,
    "OXYGEN_SHAKE_CHOCOLATE":        0,
 
    # MICROCHIP — only OVAL short confirmed
    "MICROCHIP_CIRCLE":     0,   # cut: lost 1,507 on day 4
    "MICROCHIP_OVAL":      -10,
    "MICROCHIP_SQUARE":     0,
    "MICROCHIP_RECTANGLE":  0,
    "MICROCHIP_TRIANGLE":   0,   # cut: lost 1,403 on day 4
 
    # TRANSLATOR — both confirmed, keep
    "TRANSLATOR_VOID_BLUE":        +7,
    "TRANSLATOR_SPACE_GRAY":       -7,
    "TRANSLATOR_ASTRO_BLACK":       0,
    "TRANSLATOR_ECLIPSE_CHARCOAL":  0,
    "TRANSLATOR_GRAPHITE_MIST":     0,
 
    # ROBOT — all cut, not worth the risk
    "ROBOT_VACUUMING":  0,
    "ROBOT_MOPPING":    0,
    "ROBOT_DISHES":     0,
    "ROBOT_LAUNDRY":    0,
    "ROBOT_IRONING":    0,   # cut: lost 2,565 on day 4
 
    # GALAXY_SOUNDS — pure market make
    "GALAXY_SOUNDS_DARK_MATTER":     0,
    "GALAXY_SOUNDS_BLACK_HOLES":     0,
    "GALAXY_SOUNDS_PLANETARY_RINGS": 0,
    "GALAXY_SOUNDS_SOLAR_WINDS":     0,
    "GALAXY_SOUNDS_SOLAR_FLAMES":    0,
 
    # PANEL — pure market make
    "PANEL_1X2":  0,
    "PANEL_2X2":  0,
    "PANEL_1X4":  0,
    "PANEL_2X4":  0,
    "PANEL_4X4":  0,
 
    # SNACKPACK — pure market make (signal too weak)
    "SNACKPACK_CHOCOLATE":  0,
    "SNACKPACK_VANILLA":    0,
    "SNACKPACK_PISTACHIO":  0,
    "SNACKPACK_STRAWBERRY": 0,
    "SNACKPACK_RASPBERRY":  0,
}


def get_best_bid(order_depth: OrderDepth):
    if order_depth.buy_orders:
        price = max(order_depth.buy_orders.keys())
        return price, order_depth.buy_orders[price]
    return None, None

def get_best_ask(order_depth: OrderDepth):
    if order_depth.sell_orders:
        price = min(order_depth.sell_orders.keys())
        return price, order_depth.sell_orders[price]
    return None, None


def mid_price(order_depth: OrderDepth):
    bb, _ = get_best_bid(order_depth)
    ba, _ = get_best_ask(order_depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


class Trader:
 
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
 
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}
 
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders:      List[Order] = []
            pos    = state.position.get(product, 0)
            target = DIRECTIONAL_TARGETS.get(product, 0)
 
            best_bid, best_bid_vol = get_best_bid(order_depth)
            best_ask, best_ask_vol = get_best_ask(order_depth)
            mid = mid_price(order_depth)
 
            if mid is None:
                result[product] = orders
                continue
 
            if target != 0:
                orders += self._directional(
                    product, pos, target, order_depth,
                    best_bid, best_ask
                )
            else:
                orders += self._market_make(
                    product, pos, order_depth,
                    best_bid, best_ask, mid
                )
 
            result[product] = orders
 
        return result, conversions, json.dumps(trader_data)
 
    # ─────────────────────────────────────────────────────────────────────────
 
    def _directional(self, product, pos, target, order_depth,
                     best_bid, best_ask) -> List[Order]:
        """
        Aggressively sweep the book toward target, then post passive
        order for any remaining gap.
        """
        orders = []
        gap = target - pos
 
        if gap > 0:
            # Need to BUY — sweep asks
            remaining = gap
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                ask_vol = -order_depth.sell_orders[ask_px]
                qty = min(remaining, ask_vol, POS_LIMIT - pos)
                if qty > 0:
                    orders.append(Order(product, ask_px, qty))
                    remaining -= qty
                    pos += qty
            # Passive bid for remainder
            if remaining > 0 and best_bid is not None:
                qty = min(remaining, POS_LIMIT - pos)
                if qty > 0:
                    orders.append(Order(product, best_bid + 1, qty))
 
        elif gap < 0:
            # Need to SELL — sweep bids
            remaining = -gap
            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining <= 0:
                    break
                bid_vol = order_depth.buy_orders[bid_px]
                qty = min(remaining, bid_vol, pos + POS_LIMIT)
                if qty > 0:
                    orders.append(Order(product, bid_px, -qty))
                    remaining -= qty
                    pos -= qty
            # Passive ask for remainder
            if remaining > 0 and best_ask is not None:
                qty = min(remaining, pos + POS_LIMIT)
                if qty > 0:
                    orders.append(Order(product, best_ask - 1, -qty))
 
        return orders
 
    def _market_make(self, product, pos, order_depth,
                     best_bid, best_ask, mid) -> List[Order]:
        """
        Post 1 tick inside spread with inventory skew.
        Skip if spread < 2 (no edge).
        """
        orders = []
        if best_bid is None or best_ask is None:
            return orders
 
        spread = best_ask - best_bid
        if spread < 2:
            return orders
 
        # Skew quotes toward flat: -1 tick per 3 units of inventory
        skew    = max(-2, min(2, -round(pos / 3)))
        our_bid = best_bid + 1 + skew
        our_ask = best_ask - 1 + skew
 
        # Safety: never cross quotes
        if our_bid >= our_ask:
            our_bid = round(mid) - 1
            our_ask = round(mid) + 1
 
        bid_qty = min(3, POS_LIMIT - pos)
        ask_qty = min(3, pos + POS_LIMIT)
 
        if bid_qty > 0:
            orders.append(Order(product, our_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order(product, our_ask, -ask_qty))
 
        return orders

