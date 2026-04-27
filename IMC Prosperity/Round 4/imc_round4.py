"""
IMC Prosperity Log Parser (JSON format)
Usage: python parse_log.py your_log_file.log
"""

import sys
import json
import pandas as pd
from io import StringIO

log_file = sys.argv[1] if len(sys.argv) > 1 else "submission.log"

with open(log_file, "r") as f:
    raw = f.read().strip()

data = json.loads(raw)
print("TOP-LEVEL KEYS:", list(data.keys()))

# ── 1. Sandbox logs ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("YOUR PRINT() OUTPUT")
print("=" * 70)
sandbox = data.get("sandboxLogs", data.get("sandbox_logs", ""))
print(sandbox if sandbox else "(none)")

# ── 2. Activities log — PnL ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PnL AND POSITION BREAKDOWN")
print("=" * 70)
act_raw = data.get("activitiesLog", data.get("activities_log", ""))
if act_raw:
    df = pd.read_csv(StringIO(act_raw), sep=";")
    df.columns = df.columns.str.strip()
    print(f"Days: {sorted(df['day'].unique())}")
    print(f"Products: {sorted(df['product'].unique())}")

    final = df.sort_values(["day","timestamp"]).groupby("product").last()["profit_and_loss"]
    print("\nFINAL PnL PER PRODUCT:")
    for prod, pnl in final.sort_values(ascending=False).items():
        print(f"  {prod:45s}  {pnl:>12.2f}")
    print(f"  {'TOTAL':45s}  {final.sum():>12.2f}")

    print("\nPnL OVER TIME (sampled every 100k ticks):")
    for prod in sorted(df['product'].unique()):
        sub = df[df["product"] == prod].sort_values(["day","timestamp"])
        sub = sub.copy()
        sub["global_t"] = (sub["day"] - 1) * 1_000_000 + sub["timestamp"]
        sample = sub[sub["global_t"] % 100000 == 0][["global_t","profit_and_loss"]]
        if len(sample):
            print(f"\n{prod}:")
            print(sample.to_string(index=False))
else:
    print("No activitiesLog found")

# ── 3. Trade history ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRADE HISTORY SUMMARY")
print("=" * 70)
trade_raw = data.get("tradeHistory", data.get("trade_history", ""))
if trade_raw:
    tdf = pd.read_csv(StringIO(trade_raw), sep=";") if isinstance(trade_raw, str) else pd.DataFrame(trade_raw)
    tdf.columns = tdf.columns.str.strip()
    print(f"Total trades: {len(tdf)}  |  Columns: {list(tdf.columns)}")

    if "symbol" in tdf.columns:
        print("\nTRADE COUNT + VOLUME PER SYMBOL:")
        print(tdf.groupby("symbol")["quantity"].agg(["count","sum"]).to_string())

    if "buyer" in tdf.columns:
        our_buys  = tdf[tdf["buyer"]  == "SUBMISSION"]
        our_sells = tdf[tdf["seller"] == "SUBMISSION"]
        print(f"\nOur buys:  {len(our_buys)} trades")
        print(f"Our sells: {len(our_sells)} trades")
        if len(our_buys):
            print("\nOUR BUY FILLS:")
            print(our_buys.groupby("symbol")["price"].agg(["count","mean","min","max"]).to_string())
        if len(our_sells):
            print("\nOUR SELL FILLS:")
            print(our_sells.groupby("symbol")["price"].agg(["count","mean","min","max"]).to_string())

        m67 = tdf[tdf["buyer"] == "Mark 67"]
        print(f"\nMark 67 buy trades in log: {len(m67)}")
        if len(m67):
            print(m67[["timestamp","symbol","price","quantity"]].head(20).to_string(index=False))
else:
    print("No tradeHistory found. Keys:", list(data.keys()))