"""
MNQ Backtest - ICT Sweep & Reverse Strategy
Replica a logica do strategy_MNQ_v3.pine em Python.

Dados: yfinance (MNQ=F, 15-min, ultimos 60 dias)
Execucao: python backtest_mnq.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate
import pytz

# --- Parametros -----------------------------------------------------------
LKBK              = 10      # lookback de barras para detectar swing
RR_MIN            = 2.0     # R:R minimo
RR_MAX            = 5.0     # R:R maximo (cap do TP)
A_PLUS            = False   # True = 2 contratos (A+), False = 1 contrato
USE_KZ            = True    # Filtrar pelo Kill Zone NY AM
DAILY_LOSS_LIMIT  = 150.0   # Limite de perda diaria ($)
MAX_DAILY_TRADES  = 3       # Maximo trades por dia
STOP_AFTER_LOSSES = 2       # Para apos N losses no dia
MAX_RISK_PTS      = 75.0    # Stop maximo em pontos (evita trades com risco absurdo)

# MNQ: 1 ponto = $2 | 1 tick = 0.25pt = $0.50
POINT_VALUE  = 2.0
COMMISSION   = 0.50         # por contrato por lado
SLIPPAGE_PTS = 0.25         # 1 tick de slippage

ET = pytz.timezone("America/New_York")

# --- Download -------------------------------------------------------------
print("Baixando dados MNQ 15-min (ultimos 60 dias)...")
raw = yf.download("MNQ=F", period="60d", interval="15m", auto_adjust=True)

if raw.empty:
    print("Erro: sem dados.")
    exit(1)

df = raw.copy()
df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index).tz_convert(ET)
df = df[df.index.dayofweek < 5]  # remove final de semana

print(f"Dados: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')} | {len(df)} barras\n")

# --- Indicadores ----------------------------------------------------------
df["ref_low"]  = df["Low"].shift(2).rolling(LKBK).min()
df["ref_high"] = df["High"].shift(2).rolling(LKBK).max()

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

df["hour"]   = df.index.hour
df["minute"] = df.index.minute

def in_kz(hour, minute):
    return (
        (hour == 9  and minute >= 30) or
        (hour == 10) or
        (hour == 11 and minute < 30)
    )

df["in_kz"] = df.apply(lambda r: in_kz(r["hour"], r["minute"]), axis=1)
if not USE_KZ:
    df["in_kz"] = True

# Kill Zone ends at 11:30 ET
def kz_ended(hour, minute):
    return (hour > 11) or (hour == 11 and minute >= 30)

df["kz_ended"] = df.apply(lambda r: kz_ended(r["hour"], r["minute"]), axis=1)

df.dropna(inplace=True)

# --- Simulacao -----------------------------------------------------------
contracts = 2 if A_PLUS else 1
trades = []

# Agrupar por DATA de trading (sessao CME comeca 18h ET, usa proximo dia calendario)
df["trade_date"] = df.index.date

for day, day_df in df.groupby("trade_date"):
    d_trades = 0
    d_losses = 0
    d_pnl    = 0.0
    position = None  # dict com info da posicao aberta

    for ts, row in day_df.iterrows():

        # 1) Se posicao aberta: verificar SL/TP ou fim do Kill Zone
        if position:
            closed = False

            if position["side"] == "long":
                if row["Low"] <= position["sl"]:
                    exit_price = position["sl"]
                    pnl = (exit_price - position["entry"]) * POINT_VALUE * contracts - COMMISSION * contracts
                    result = "LOSS"
                    closed = True
                elif row["High"] >= position["tp"]:
                    exit_price = position["tp"]
                    pnl = (exit_price - position["entry"]) * POINT_VALUE * contracts - COMMISSION * contracts
                    result = "WIN"
                    closed = True
            else:  # short
                if row["High"] >= position["sl"]:
                    exit_price = position["sl"]
                    pnl = (position["entry"] - exit_price) * POINT_VALUE * contracts - COMMISSION * contracts
                    result = "LOSS"
                    closed = True
                elif row["Low"] <= position["tp"]:
                    exit_price = position["tp"]
                    pnl = (position["entry"] - exit_price) * POINT_VALUE * contracts - COMMISSION * contracts
                    result = "WIN"
                    closed = True

            # Fechar no fim do Kill Zone se ainda aberta
            if not closed and row["kz_ended"]:
                exit_price = row["Open"]  # abre a barra seguinte ao KZ
                if position["side"] == "long":
                    pnl = (exit_price - position["entry"]) * POINT_VALUE * contracts - COMMISSION * contracts
                else:
                    pnl = (position["entry"] - exit_price) * POINT_VALUE * contracts - COMMISSION * contracts
                result = "WIN" if pnl >= 0 else "LOSS"
                closed = True

            if closed:
                if result == "LOSS":
                    d_losses += 1
                d_pnl += pnl
                trades.append({
                    "entrada": position["ts"].strftime("%d/%m %H:%M"),
                    "saida":   ts.strftime("%d/%m %H:%M"),
                    "tipo":    position["side"].upper(),
                    "result":  result,
                    "entry":   round(position["entry"], 2),
                    "sl":      round(position["sl"], 2),
                    "tp":      round(position["tp"], 2),
                    "pnl":     round(pnl, 2),
                })
                position = None

        # 2) Verificar se pode entrar
        can_trade = (
            row["in_kz"] and
            position is None and
            d_trades < MAX_DAILY_TRADES and
            d_losses < STOP_AFTER_LOSSES and
            d_pnl > -DAILY_LOSS_LIMIT
        )

        if not can_trade:
            continue

        # 3) Sinal de entrada
        if row["bull_sweep"]:
            entry = row["Close"] + SLIPPAGE_PTS
            sl    = row["Low"] - SLIPPAGE_PTS
            risk  = entry - sl
            if risk <= 0 or risk > MAX_RISK_PTS:
                continue
            tp = min(entry + risk * RR_MAX, entry + risk * RR_MIN)
            tp = entry + risk * RR_MIN  # usa RR_MIN como alvo
            position = {"side": "long",  "entry": entry, "sl": sl, "tp": tp, "ts": ts}
            d_trades += 1

        elif row["bear_sweep"]:
            entry = row["Close"] - SLIPPAGE_PTS
            sl    = row["High"] + SLIPPAGE_PTS
            risk  = sl - entry
            if risk <= 0 or risk > MAX_RISK_PTS:
                continue
            tp = entry - risk * RR_MIN
            position = {"side": "short", "entry": entry, "sl": sl, "tp": tp, "ts": ts}
            d_trades += 1

# --- Relatorio -----------------------------------------------------------
if not trades:
    print("Nenhum trade encontrado. Tente ajustar os parametros.")
    exit(0)

results = pd.DataFrame(trades)
total      = len(results)
wins       = (results["result"] == "WIN").sum()
losses     = (results["result"] == "LOSS").sum()
win_rate   = wins / total * 100
net_pnl    = results["pnl"].sum()
gross_win  = results[results["pnl"] > 0]["pnl"].sum()
gross_loss = abs(results[results["pnl"] < 0]["pnl"].sum())
pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")
avg_win    = results[results["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
avg_loss   = results[results["pnl"] < 0]["pnl"].mean() if losses > 0 else 0

equity = [25000.0]
for p in results["pnl"]:
    equity.append(equity[-1] + p)
peak   = pd.Series(equity).cummax()
max_dd = (pd.Series(equity) - peak).min()

print("=" * 56)
print("  MNQ BACKTEST - ICT Sweep & Reverse (15-min, KZ)")
print("=" * 56)
summary = [
    ["Total de trades",      total],
    ["Wins / Losses",        f"{wins} / {losses}"],
    ["Win Rate",             f"{win_rate:.1f}%"],
    ["P&L Liquido",          f"${net_pnl:,.2f}"],
    ["Profit Factor",        f"{pf:.2f}"],
    ["Media Win",            f"${avg_win:.2f}"],
    ["Media Loss",           f"${avg_loss:.2f}"],
    ["Max Drawdown",         f"${max_dd:,.2f}"],
    ["Max risco/trade",      f"{MAX_RISK_PTS}pts = ${MAX_RISK_PTS*POINT_VALUE:.0f}"],
    ["Contratos",            contracts],
]
print(tabulate(summary, tablefmt="simple"))
print()
print("--- Lista de Trades ---")
cols = ["entrada", "saida", "tipo", "result", "entry", "sl", "tp", "pnl"]
print(tabulate(results[cols].values, headers=cols, tablefmt="simple", floatfmt=".2f"))
print()
print(f"Capital final estimado: ${equity[-1]:,.2f}")
