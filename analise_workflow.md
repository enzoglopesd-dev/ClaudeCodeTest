# Workflow de Análise — MNQ1! (Modelo Completo)

## Instrumento padrão
MNQ (Micro E-mini Nasdaq-100 Futures) — $2/pt | 1 tick = $0.50

---

## Indicadores fixos (todos os timeframes)

| Indicador | ID | Função |
|---|---|---|
| SR Breaks and Retests [ChartPrime] | VJwBLy | Suporte/resistência e breaks |
| EMA 20/50/100/200 | Yhyq0x | Tendência e dinâmica de médias |
| FVG/iFVG (Nephew_Sam_) | s7F9bY | Gaps de valor justo |
| MACD | g33Wyz | Momentum e divergências |
| Volume | 48r50X | Confirmação de movimentos |
| Volume Profile with Node Detection [LuxAlgo] | 6nGufQ | POC, HVN, LVN |

---

## Fluxo de análise top-down

### STEP 0 — Intermarket (via WebSearch, toda análise começa aqui)

Buscar valores atuais de:
| Ativo | Relação com NQ | Leitura |
|---|---|---|
| DXY (US Dollar Index) | Inversa — DXY sobe = NQ cai | 🔴 se subindo / 🟢 se caindo |
| VIX (Fear Index) | Inversa — VIX >20 = evitar longs | 🔴 >20 / 🟡 15-20 / 🟢 <15 |
| US10Y (Yield 10 anos) | Inversa — yields sobem = NQ cai | 🔴 se subindo forte |
| Gold (XAU) | Risk-off — Gold sobe = fuga do risco | 🔴 se subindo muito |

Composição: 3-4 vermelhos = 🔴 bearish | 2 vermelhos = 🟡 neutro | 0-1 vermelho = 🟢 bullish

Checar calendário econômico: **NÃO operar** em CPI, FOMC, NFP, GDP, PCE, PPI.

---

### STEP 1 — Diário

- Tendência macro via EMAs (20/50/100/200) e SR
- FVGs diários abertos (zonas de demanda/oferta macro)
- Volume Profile: POC e HVN relevantes
- MACD: bias direcional geral
- Ferramentas: `data_get_ohlcv(summary=true, count=20)` + `data_get_study_values` + `data_get_pine_boxes(study_filter="FVG")`

### STEP 2 — 4 Horas

- Confirmar ou refutar bias do diário
- FVGs 4H não preenchidos (resistência/suporte de médio prazo)
- Breaks de SR e retestes com volume
- VWAP: preço acima ou abaixo define pressão direcional

### STEP 3 — 15 Minutos

- Estrutura intraday: HH/HL (bullish) ou LH/LL (bearish)
- FVGs de curto prazo e pontos de interesse para entrada
- SR intraday e volume nos breaks

### STEP 4 — 5 Minutos (situacional)

- Confirmar timing da entrada
- FVGs 5m para SL/TP ajuste fino
- MACD: momentum de curto prazo

### STEP 5 — 1 Minuto

- Timing de execução
- Confirmação via Rule of 3 (ver abaixo)
- `capture_screenshot(filename, region="chart")`

---

## Rule of 3 — Confirmação de entrada (substitui CVD)

Precisa de **2 de 3** critérios para entrar. 3/3 = Setup A+ (2 contratos), 2/3 = Setup A (1 contrato), 1/3 = PASS.

### Critério 1 — Volume + Corpo do Candle
- Volume alto (2-3× média) + corpo **pequeno** = absorção (reversal signal)
- Volume alto + corpo **grande** na direção = pressão direcional confirmada
- Volume normal = critério não confirmado

### Critério 2 — Divergência MACD
- **Bearish:** preço faz Higher High mas histograma MACD faz pico menor → exaustão compradores
- **Bullish:** preço faz Lower Low mas histograma MACD faz vale menor (menos negativo) → exaustão vendedores
- Verificar no 5m ou 1m

### Critério 3 — Candle Pattern no nível-chave
- **Bearish:** shooting star, bearish engulf, inside bar de teto
- **Bullish:** hammer, bullish engulf, pin bar de fundo, inside bar de suporte

---

## Output esperado (daily briefing)

1. **Tabela intermarket** — DXY, VIX, US10Y, Gold com leitura e composição
2. **Bias do dia** — 🔴 bearish / 🟡 neutro / 🟢 bullish com justificativa por timeframe
3. **Mapa de níveis** — tabela com preço, tipo (FVG/SR/VWAP/OB/Liquidez) e relevância
4. **Setup primário** — entrada com Rule of 3 explicado (o que ver no gráfico)
5. **Setup secundário** — alternativa oportunista (se houver)

Para cada setup entregar:
- Entrada, SL (em pts e $), TP1/TP2/TP3 (em pts e $), RR
- Instrução visual: o que observar em cada critério do Rule of 3

---

## Kill Zones (horários prioritários — ET)

| Sessão | Horário ET | Prioridade |
|---|---|---|
| London Open | 03:00–05:00 | Média |
| NY AM | **09:30–11:30** | **Alta** |
| NY PM | 13:30–15:00 | Baixa/Opcional |

---

## Gerenciamento de risco (My Trading Futures $25K)

- **Max loss diário:** $100
- **SL máximo por operação:** $50 (= 25 pts MNQ)
- **Modelo:** melhor de 3 (geralmente 2 operações)
- **Contratos:** 1 por operação padrão; 2 apenas em Setup A+ (3/3 Rule of 3)
- **TP1:** mínimo RR 2:1 | **TP2:** RR 4:1 | **TP3:** estrutural/RR 6:1+
- **Floor da conta:** $24,000 (EOD trailing drawdown)

---

## Comando "zona"

Quando o usuário digitar **"zona"**, significa que o preço acabou de entrar na zona de entrada que foi passada no setup anterior. Executar imediatamente:

1. Confirmar que está no timeframe 1 minuto
2. Coletar: `data_get_ohlcv(count=15)` + `data_get_study_values` + `capture_screenshot`
3. Avaliar **Rule of 3** para o setup ativo:
   - Critério 1: Volume da última barra vs média + tamanho do corpo
   - Critério 2: MACD histograma — divergência em formação?
   - Critério 3: Padrão de candle no nível (shooting star / hammer / engulf)
4. Veredicto rápido:
   - ✅ 3/3 → ENTRAR (2 contratos, A+) — confirmar entrada, SL e TP
   - ✅ 2/3 → ENTRAR (1 contrato, A) — confirmar entrada, SL e TP
   - ❌ 1/3 ou 0/3 → AGUARDAR ou PASS — explicar o que falta
