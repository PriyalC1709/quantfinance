from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

"""
IMC Prosperity Round 5 — Cherry Picking Winners
================================================
Position limit: ±10 for ALL products

Strategy per group (ranked by expected alpha):
----------------------------------------------
PEBBLES:      Momentum divergence — XS/S falling, XL/L rising.
              Short XS & S at -10, Long XL & L at +10.
              M: market make around mid.

SLEEP_POD:    All 5 products in persistent uptrend across 3 days.
              Max long +10 all 5. Aggressive take on any ask.

UV_VISOR:     AMBER crashing, RED+MAGENTA rising.
              Short AMBER at -10, Long RED+MAGENTA at +10.
              YELLOW+ORANGE: market make.

OXYGEN_SHAKE: GARLIC surging monotonically to 14k+.
              Max long GARLIC +10. Short MORNING_BREATH.
              Others: market make.

MICROCHIP:    CIRCLE trending up, OVAL crashing.
              Long CIRCLE, Short OVAL. Others: cautious market make.

TRANSLATOR:   VOID_BLUE outperforming, SPACE_GRAY underperforming.
              Mild long VOID_BLUE, short SPACE_GRAY.
              Others: market make.

ALL OTHERS:   Pure market making — post inside spread, earn half-spread.
"""

POS_LIMIT = 10

# ── DIRECTIONAL TARGETS ──────────────────────────────────────────────────────
# +10 = max long, -10 = max short, 0 = neutral (market make)
DIRECTIONAL_TARGETS = {
    # PEBBLES
    "PEBBLES_XS":   -10,
    "PEBBLES_S":    -10,
    "PEBBLES_M":      0,   # market make
    "PEBBLES_L":    +10,
    "PEBBLES_XL":   +10,

    # SLEEP_POD — all trend up
    "SLEEP_POD_SUEDE":      +10,
    "SLEEP_POD_LAMB_WOOL":  +10,
    "SLEEP_POD_POLYESTER":  +10,
    "SLEEP_POD_NYLON":      +10,
    "SLEEP_POD_COTTON":     +10,

    # UV_VISOR
    "UV_VISOR_AMBER":    -10,
    "UV_VISOR_RED":      +10,
    "UV_VISOR_MAGENTA":  +10,
    "UV_VISOR_YELLOW":     0,   # market make
    "UV_VISOR_ORANGE":     0,   # market make

    # OXYGEN_SHAKE
    "OXYGEN_SHAKE_GARLIC":          +10,
    "OXYGEN_SHAKE_MORNING_BREATH":   -5,   # mild short
    "OXYGEN_SHAKE_EVENING_BREATH":    0,
    "OXYGEN_SHAKE_MINT":              0,
    "OXYGEN_SHAKE_CHOCOLATE":         0,

    # MICROCHIP
    "MICROCHIP_CIRCLE":     +10,
    "MICROCHIP_OVAL":       -10,
    "MICROCHIP_SQUARE":       0,
    "MICROCHIP_RECTANGLE":    0,
    "MICROCHIP_TRIANGLE":   -5,

    # TRANSLATOR
    "TRANSLATOR_VOID_BLUE":       +7,
    "TRANSLATOR_SPACE_GRAY":      -7,
    "TRANSLATOR_ASTRO_BLACK":      0,
    "TRANSLATOR_ECLIPSE_CHARCOAL": 0,
    "TRANSLATOR_GRAPHITE_MIST":    0,

    # GALAXY_SOUNDS — all neutral, market make
    "GALAXY_SOUNDS_DARK_MATTER":     0,
    "GALAXY_SOUNDS_BLACK_HOLES":     0,
    "GALAXY_SOUNDS_PLANETARY_RINGS": 0,
    "GALAXY_SOUNDS_SOLAR_WINDS":     0,
    "GALAXY_SOUNDS_SOLAR_FLAMES":    0,

    # PANEL — no clear signal
    "PANEL_1X2":  0,
    "PANEL_2X2":  0,
    "PANEL_1X4":  0,
    "PANEL_2X4":  0,
    "PANEL_4X4":  0,

    # SNACKPACK
    "SNACKPACK_CHOCOLATE":   +3,
    "SNACKPACK_VANILLA":     -3,
    "SNACKPACK_PISTACHIO":   -3,
    "SNACKPACK_STRAWBERRY":  +3,
    "SNACKPACK_RASPBERRY":    0,

    # ROBOT
    "ROBOT_DISHES":    +5,
    "ROBOT_IRONING":   -5,
    "ROBOT_VACUUMING":  0,
    "ROBOT_MOPPING":    0,
    "ROBOT_LAUNDRY":    0,
}


def get_best_bid(order_depth: OrderDepth):
    """Returns (price, volume) of best bid, or (None, None)."""
    if order_depth.buy_orders:
        price = max(order_depth.buy_orders.keys())
        return price, order_depth.buy_orders[price]
    return None, None

def get_best_ask(order_depth: OrderDepth):
    """Returns (price, volume) of best ask, or (None, None).
    Note: sell order volumes are stored as negative in IMC datamodel."""
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

        # Load persistent state (for tracking trend context if needed later)
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)
            target = DIRECTIONAL_TARGETS.get(product, 0)

            best_bid, best_bid_vol = get_best_bid(order_depth)
            best_ask, best_ask_vol = get_best_ask(order_depth)
            mid = mid_price(order_depth)

            if mid is None:
                result[product] = orders
                continue

            if target != 0:
                # ── DIRECTIONAL MODE ────────────────────────────────────────
                orders += self._directional_orders(
                    product, pos, target, order_depth,
                    best_bid, best_bid_vol, best_ask, best_ask_vol, mid
                )
            else:
                # ── MARKET MAKING MODE ──────────────────────────────────────
                orders += self._market_make(
                    product, pos, order_depth,
                    best_bid, best_ask, mid
                )

            result[product] = orders

        return result, conversions, json.dumps(trader_data)

    # ─────────────────────────────────────────────────────────────────────────

    def _directional_orders(
        self, product, pos, target,
        order_depth, best_bid, best_bid_vol, best_ask, best_ask_vol, mid
    ) -> List[Order]:
        """
        Drive position toward `target` (±10).
        If we need to BUY: aggressively lift asks, then place passive bid.
        If we need to SELL: aggressively hit bids, then place passive ask.
        """
        orders = []
        gap = target - pos  # how many units we still need to trade

        if gap > 0:
            # Need to BUY
            # 1. Aggressively sweep asks up to the gap
            remaining = gap
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                ask_vol = -order_depth.sell_orders[ask_px]  # make positive
                buy_qty = min(remaining, ask_vol)
                buy_qty = min(buy_qty, POS_LIMIT - pos)     # respect limit
                if buy_qty > 0:
                    orders.append(Order(product, ask_px, buy_qty))
                    remaining -= buy_qty
                    pos += buy_qty

            # 2. Passive bid one tick above best_bid if still short of target
            if remaining > 0 and best_bid is not None:
                passive_px = best_bid + 1
                passive_qty = min(remaining, POS_LIMIT - pos)
                if passive_qty > 0:
                    orders.append(Order(product, passive_px, passive_qty))

        elif gap < 0:
            # Need to SELL
            remaining = -gap
            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining <= 0:
                    break
                bid_vol = order_depth.buy_orders[bid_px]
                sell_qty = min(remaining, bid_vol)
                sell_qty = min(sell_qty, pos + POS_LIMIT)  # respect short limit
                if sell_qty > 0:
                    orders.append(Order(product, bid_px, -sell_qty))
                    remaining -= sell_qty
                    pos -= sell_qty

            # Passive ask one tick below best_ask if still above target
            if remaining > 0 and best_ask is not None:
                passive_px = best_ask - 1
                passive_qty = min(remaining, pos + POS_LIMIT)
                if passive_qty > 0:
                    orders.append(Order(product, passive_px, -passive_qty))

        return orders

    def _market_make(
        self, product, pos,
        order_depth, best_bid, best_ask, mid
    ) -> List[Order]:
        """
        Post passive quotes inside the spread.
        Inventory skew: shift both quotes toward flat when position is large.
        """
        orders = []
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        if spread < 2:
            # Spread too tight — no edge, skip
            return orders

        # Inventory skew: -1 tick per 3 units of position
        skew = -round(pos / 3)
        skew = max(-2, min(2, skew))   # clamp skew to ±2 ticks

        our_bid = best_bid + 1 + skew
        our_ask = best_ask - 1 + skew

        # Don't cross our own quotes
        if our_bid >= our_ask:
            our_bid = round(mid) - 1
            our_ask = round(mid) + 1

        # Volume: limited by how close we are to position limits
        bid_qty = min(3, POS_LIMIT - pos)    # buy up to 3 units
        ask_qty = min(3, pos + POS_LIMIT)    # sell up to 3 units

        if bid_qty > 0:
            orders.append(Order(product, our_bid, bid_qty))
        if ask_qty > 0:
            orders.append(Order(product, our_ask, -ask_qty))

        return orders