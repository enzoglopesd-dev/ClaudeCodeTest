# Workflow de Análise — MNQ1!

## Indicadores fixos (todos os timeframes)

Usar SEMPRE estes 6 — sem alternância, sem grupos:

| Indicador | Entidade | Função |
|---|---|---|
| SR Breaks and Retests [ChartPrime] | VJwBLy | Suporte/resistência e breaks |
| EMA 20/50/100/200 | Yhyq0x | Tendência e dinâmica de médias |
| FVG/iFVG (Nephew_Sam_) | s7F9bY | Gaps de valor justo |
| MACD | g33Wyz | Momentum e divergências |
| Volume | 48r50X | Confirmação de movimentos |
| Volume Profile with Node Detection [LuxAlgo] | 6nGufQ | POC, HVN, LVN |

## Fluxo top-down

### 1. Diário
- Tendência macro via EMAs e SR
- FVGs diários abertos
- Volume Profile: POC e HVN relevantes
- MACD: bias direcional geral

### 2. 4 horas
- Confirmar ou refutar bias do diário
- FVGs 4H não preenchidos
- Breaks de SR e retestes com volume

### 3. 15 minutos
- Estrutura intraday (HH/HL ou LH/LL)
- FVGs de curto prazo e pontos de interesse
- SR intraday para entrada

### 4. 1 minuto
- Timing de entrada
- Confirmação de momentum via MACD e Volume

## Output esperado

1. **Bias do dia** — bullish / bearish / neutro com justificativa
2. **Níveis-chave** — SR, FVGs abertos, POC/HVN do Volume Profile
3. **Zonas desenhadas no gráfico** — rectangles e linhas horizontais no 1min
4. **Setup sugerido** — entrada com SL/TP respeitando o gerenciamento de risco

## Gerenciamento de risco (My Trading Futures $25K)

- Max loss diário: $150 | Daily target: $150–200
- SL máx por operação: $50 (= 25 pts MNQ a $2/pt)
- Contratos: 1 por operação
- TP1: +20pts | TP2: +40pts | TP3: estrutural
- Sessão operacional: 08h–15h horário de São Paulo
