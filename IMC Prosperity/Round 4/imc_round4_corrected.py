from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json

class Trader:
    # ── Constants ─────────────────────────────────────────────────────────────
    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100,
        "VEV_5200": 5200, "VEV_5300": 5300,
        "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }
    VEV_SELL_LIMITS = {
        "VEV_5200": 100, "VEV_5300": 100,
        "VEV_5400":  50, "VEV_5500":  50,
    }
    VEV_BUY_ARB = {"VEV_4000", "VEV_4500"}

    HP_EDGE       = 3     # half-spread for HYDROGEL_PACK market making
    HP_SIZE       = 20    # units per side per tick
    HP_LIMIT      = 200

    VFE_LIMIT     = 200
    VFE_SIGNAL_SIZE = 0   # units to buy on Mark 67 signal
    VFE_HOLD_TICKS  = 500   # hold duration after signal
    VFE_STOP_LOSS   = 4     # exit if price drops this many below entry

    VEV_LIMIT     = 300   # hard position limit per voucher (IMC rule)

    def run(self, state: TradingState):
        result  = {}
        t       = state.timestamp

        # ── Persistent state via traderData ───────────────────────────────────
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        m67_signal_t = data.get("m67_signal_t", -999999)
        vfe_entry    = data.get("vfe_entry", None)

        # ── Detect Mark 67 in this tick's market trades ───────────────────────
        mark67_seen = False
        if state.market_trades and "VELVETFRUIT_EXTRACT" in state.market_trades:
            for trade in state.market_trades["VELVETFRUIT_EXTRACT"]:
                if trade.buyer == "Mark 67":
                    mark67_seen = True
                    break

        # ── Get current VFE mid for VEV intrinsic calculations ────────────────
        vfe_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vfe_od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if vfe_od.buy_orders and vfe_od.sell_orders:
                vfe_mid = (max(vfe_od.buy_orders) + min(vfe_od.sell_orders)) / 2

        # ══════════════════════════════════════════════════════════════════════
        # 1. HYDROGEL_PACK — market making
        # ══════════════════════════════════════════════════════════════════════
        hp_orders = []
        if "HYDROGEL_PACK" in state.order_depths:
            od  = state.order_depths["HYDROGEL_PACK"]
            pos = state.position.get("HYDROGEL_PACK", 0)

            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders)
                best_ask = min(od.sell_orders)
                mid      = (best_bid + best_ask) / 2

                buy_qty  = min(self.HP_SIZE, self.HP_LIMIT - pos)
                sell_qty = min(self.HP_SIZE, self.HP_LIMIT + pos)

                if buy_qty  > 0:
                    hp_orders.append(Order("HYDROGEL_PACK", int(mid - self.HP_EDGE), buy_qty))
                if sell_qty > 0:
                    hp_orders.append(Order("HYDROGEL_PACK", int(mid + self.HP_EDGE), -sell_qty))

        result["HYDROGEL_PACK"] = hp_orders

        # ══════════════════════════════════════════════════════════════════════════════
        # 2. VELVETFRUIT_EXTRACT — Mark 67 signal ONLY, no passive MM
        # ══════════════════════════════════════════════════════════════════════════════
        vfe_orders = []
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od  = state.order_depths["VELVETFRUIT_EXTRACT"]
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders)
                best_ask = min(od.sell_orders)
                mid      = (best_bid + best_ask) / 2

                if mark67_seen:
                    data["m67_signal_t"] = t
                    data["vfe_entry"]    = best_ask
                    qty = min(self.VFE_SIGNAL_SIZE, self.VFE_LIMIT - pos)
                    if qty > 0:
                        vfe_orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, qty))

                elif pos > 0:
                    ticks_since = t - m67_signal_t
                    stop_hit    = vfe_entry and mid < vfe_entry - self.VFE_STOP_LOSS
                    hold_done   = ticks_since >= self.VFE_HOLD_TICKS

                    # Only exit once — don't fire every tick while holding
                    already_exiting = data.get("vfe_exiting", False)
                    if (stop_hit or hold_done) and not already_exiting:
                        data["vfe_exiting"] = True
                        vfe_orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -pos))

                # Reset exit flag when flat
                if pos == 0:
                    data["vfe_exiting"] = False
                    data["m67_signal_t"] = -999999
                    data["vfe_entry"] = None

        result["VELVETFRUIT_EXTRACT"] = []
                  

        # ══════════════════════════════════════════════════════════════════════
        # 3. VEV — sell premium on OTM vouchers + arb buy on deep ITM
        # ══════════════════════════════════════════════════════════════════════
        for sym, K in self.STRIKES.items():
            if sym not in state.order_depths:
                continue

            od  = state.order_depths[sym]
            pos = state.position.get(sym, 0)
            orders = []

            intrinsic = max(vfe_mid - K, 0) if vfe_mid else 0

            # ── Sell premium: OTM vouchers where bid >> intrinsic ──────────
            if sym in self.VEV_SELL_LIMITS and od.buy_orders:
                best_bid   = max(od.buy_orders)
                max_short  = self.VEV_SELL_LIMITS[sym]
                premium    = best_bid - intrinsic

                # Only sell if premium is meaningful (> 1 tick)
                if premium > 1:
                    # How much more can we short?
                    sell_qty = min(
                        abs(od.buy_orders[best_bid]),  # available at bid
                        max_short + pos,               # room to short more
                        20                             # max per tick
                    )
                    if sell_qty > 0 and pos > -max_short:
                        orders.append(Order(sym, best_bid, -sell_qty))

            # ── Arb buy: deep ITM where ask < intrinsic ────────────────────
            elif sym in self.VEV_BUY_ARB and od.sell_orders and vfe_mid:
                best_ask = min(od.sell_orders)
                if best_ask < intrinsic - 1:   # ask is below intrinsic value
                    buy_qty = min(
                        abs(od.sell_orders[best_ask]),
                        self.VEV_LIMIT - pos,
                        10
                    )
                    if buy_qty > 0:
                        orders.append(Order(sym, best_ask, buy_qty))

            result[sym] = orders

        return result, 0, json.dumps(data)