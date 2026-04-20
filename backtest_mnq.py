"""
MNQ Backtest — ICT Sweep & Reverse Strategy
Replica a lógica do strategy_MNQ_v3.pine em Python.

Dados: yfinance (MNQ=F, 15-min, últimos 60 dias — grátis)
Execução: python backtest_mnq.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate
from datetime import datetime
import pytz

# ─── Parâmetros (mesmos do Pine Script) ───────────────────────────────────────
LKBK             = 10      # lookback de barras para detectar swing
RR_MIN           = 2.0     # R:R mínimo
RR_MAX           = 5.0     # R:R máximo (cap do TP)
A_PLUS           = False   # True = 2 contratos (setup A+), False = 1 contrato
USE_KZ           = True    # Filtrar pelo Kill Zone NY AM
DAILY_LOSS_LIMIT = 150.0   # Limite de perda diária ($)
MAX_DAILY_TRADES = 3       # Máximo de trades por dia
STOP_AFTER_LOSSES = 2      # Para após N losses no dia

# MNQ: 1 ponto = $2 | 1 tick = 0.25pt = $0.50
POINT_VALUE      = 2.0
COMMISSION       = 0.50    # por contrato
SLIPPAGE_TICKS   = 1       # 1 tick de slippage na entrada
TICK_SIZE        = 0.25

ET = pytz.timezone("America/New_York")

# ─── Download de dados ────────────────────────────────────────────────────────
print("Baixando dados MNQ 15-min (últimos 60 dias)...")
raw = yf.download("MNQ=F", period="60d", interval="15m", auto_adjust=True)

if raw.empty:
    print("Erro: sem dados. Verifique conexão ou ticker.")
    exit(1)

df = raw.copy()
df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index)

# Converter para ET e filtrar sessão regular (sem weekend)
if df.index.tzinfo is None:
    df.index = df.index.tz_localize("UTC")
df.index = df.index.tz_convert(ET)
df = df[df.index.dayofweek < 5]   # remove sábado/domingo

print(f"Dados: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')} | {len(df)} barras\n")

# ─── Indicadores ──────────────────────────────────────────────────────────────
# Pine: ref_low = ta.lowest(low, lkbk)[2]  → rolling min do lkbk barras, shifted 2
df["ref_low"]  = df["Low"].shift(2).rolling(LKBK).min()
df["ref_high"] = df["High"].shift(2).rolling(LKBK).max()

# Sweep: barra anterior furou, barra atual reverteu
df["bull_sweep"] = (
    (df["Low"].shift(1) < df["ref_low"]) &
    (df["Close"] > df["ref_low"]) &
    (df["Close"] > df["Open"])
)
df["bear_sweep"] = (
    (df["High"].shift(1) > df["ref_high"]) &
    (df["Close"] < df["ref_high"]) &
    (df["Close"] < df["Open"])
)

# Kill Zone: 9:30–11:30 ET
df["hour"]  = df.index.hour
df["minute"] = df.index.minute
df["in_kz"] = (
    ((df["hour"] == 9)  & (df["minute"] >= 30)) |
    ((df["hour"] == 10)) |
    ((df["hour"] == 11) & (df["minute"] < 30))
)
if not USE_KZ:
    df["in_kz"] = True

df.dropna(inplace=True)

# ─── Simulação trade a trade ──────────────────────────────────────────────────
contracts = 2 if A_PLUS else 1
trades = []

# Agrupa por dia
df["date"] = df.index.date

for day, day_df in df.groupby("date"):
    d_trades = 0
    d_losses = 0
    d_pnl    = 0.0
    position = None   # {"side": "long"/"short", "entry": float, "sl": float, "tp": float}

    for i, (ts, row) in enumerate(day_df.iterrows()):
        # Se há posição aberta, checar SL/TP usando High/Low da barra
        if position:
            if position["side"] == "long":
                if row["Low"] <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) * POINT_VALUE * contracts - COMMISSION * contracts
                    d_pnl += pnl
                    d_losses += 1
                    trades.append({"data": ts, "tipo": "LONG", "resultado": "LOSS",
                                   "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                                   "pnl": round(pnl, 2)})
                    position = None
                elif row["High"] >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * POINT_VALUE * contracts - COMMISSION * contracts
                    d_pnl += pnl
                    trades.append({"data": ts, "tipo": "LONG", "resultado": "WIN",
                                   "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                                   "pnl": round(pnl, 2)})
                    position = None
            else:  # short
                if row["High"] >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * POINT_VALUE * contracts - COMMISSION * contracts
                    d_pnl += pnl
                    d_losses += 1
                    trades.append({"data": ts, "tipo": "SHORT", "resultado": "LOSS",
                                   "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                                   "pnl": round(pnl, 2)})
                    position = None
                elif row["Low"] <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * POINT_VALUE * contracts - COMMISSION * contracts
                    d_pnl += pnl
                    trades.append({"data": ts, "tipo": "SHORT", "resultado": "WIN",
                                   "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                                   "pnl": round(pnl, 2)})
                    position = None

        # Condições para nova entrada
        can_trade = (
            row["in_kz"] and
            position is None and
            d_trades < MAX_DAILY_TRADES and
            d_losses < STOP_AFTER_LOSSES and
            d_pnl > -DAILY_LOSS_LIMIT
        )

        if not can_trade:
            continue

        slippage = SLIPPAGE_TICKS * TICK_SIZE

        if row["bull_sweep"]:
            entry = row["Close"] + slippage
            sl    = row["Low"] - TICK_SIZE * 2
            risk  = entry - sl
            if risk <= 0:
                continue
            tp = entry + risk * RR_MIN
            tp = min(tp, entry + risk * RR_MAX)
            position = {"side": "long", "entry": entry, "sl": sl, "tp": tp}
            d_trades += 1

        elif row["bear_sweep"]:
            entry = row["Close"] - slippage
            sl    = row["High"] + TICK_SIZE * 2
            risk  = sl - entry
            if risk <= 0:
                continue
            tp = entry - risk * RR_MIN
            tp = max(tp, entry - risk * RR_MAX)
            position = {"side": "short", "entry": entry, "sl": sl, "tp": tp}
            d_trades += 1

    # Fim do dia: fechar posição aberta no último preço
    if position and len(day_df) > 0:
        last_close = day_df["Close"].iloc[-1]
        if position["side"] == "long":
            pnl = (last_close - position["entry"]) * POINT_VALUE * contracts - COMMISSION * contracts
        else:
            pnl = (position["entry"] - last_close) * POINT_VALUE * contracts - COMMISSION * contracts
        if pnl < 0:
            d_losses += 1
        d_pnl += pnl
        trades.append({"data": day_df.index[-1], "tipo": position["side"].upper(),
                       "resultado": "WIN" if pnl >= 0 else "LOSS",
                       "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                       "pnl": round(pnl, 2)})
        position = None

# ─── Relatório ────────────────────────────────────────────────────────────────
if not trades:
    print("Nenhum trade encontrado no período. Verifique os parâmetros.")
    exit(0)

results = pd.DataFrame(trades)
results["data"] = pd.to_datetime(results["data"]).dt.strftime("%d/%m %H:%M")

total       = len(results)
wins        = (results["resultado"] == "WIN").sum()
losses      = (results["resultado"] == "LOSS").sum()
win_rate    = wins / total * 100
net_pnl     = results["pnl"].sum()
gross_win   = results[results["pnl"] > 0]["pnl"].sum()
gross_loss  = abs(results[results["pnl"] < 0]["pnl"].sum())
pf          = gross_win / gross_loss if gross_loss > 0 else float("inf")
avg_win     = results[results["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
avg_loss    = results[results["pnl"] < 0]["pnl"].mean() if losses > 0 else 0

# Drawdown máximo
equity = [25000]
for p in results["pnl"]:
    equity.append(equity[-1] + p)
peak = pd.Series(equity).cummax()
dd = (pd.Series(equity) - peak)
max_dd = dd.min()

print("=" * 55)
print("  MNQ BACKTEST — ICT Sweep & Reverse (15-min)")
print("=" * 55)
summary = [
    ["Total de trades",     total],
    ["Wins / Losses",       f"{wins} / {losses}"],
    ["Win Rate",            f"{win_rate:.1f}%"],
    ["P&L Líquido",         f"${net_pnl:,.2f}"],
    ["Profit Factor",       f"{pf:.2f}"],
    ["Média Win",           f"${avg_win:.2f}"],
    ["Média Loss",          f"${avg_loss:.2f}"],
    ["Max Drawdown",        f"${max_dd:,.2f}"],
    ["Contratos",           contracts],
]
print(tabulate(summary, tablefmt="simple"))
print()

print("--- Lista de Trades ---")
cols = ["data", "tipo", "resultado", "entry", "sl", "tp", "pnl"]
print(tabulate(results[cols].values, headers=cols, tablefmt="simple", floatfmt=".2f"))
print()
print(f"Capital final estimado: ${25000 + net_pnl:,.2f}")
