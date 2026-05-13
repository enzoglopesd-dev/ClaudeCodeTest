# Instruções para Claude — Projeto MNQ Trading

## Arquivo principal
O workflow completo de análise está em `analise_workflow.md`. Ler esse arquivo antes de qualquer análise de mercado.

## Comando "zona"

Quando o usuário digitar **"zona"** (sozinho ou em frase curta como "estamos na zona", "zona atingida"):

1. Mudar para timeframe 1 minuto se necessário
2. Executar análise rápida:
   - `data_get_ohlcv(count=15, summary=true)` — últimas barras
   - `data_get_study_values` — MACD e Volume atuais
   - `capture_screenshot(region="chart")`
3. Avaliar Rule of 3 para o **último setup ativo** comunicado nessa sessão:
   - **Critério 1 (Volume + Corpo):** comparar volume da última barra com a média; corpo pequeno = absorção, corpo grande = pressão direcional
   - **Critério 2 (MACD Divergência):** preço fez novo extremo? histograma acompanhou ou divergiu?
   - **Critério 3 (Candle Pattern):** shooting star / bearish engulf / hammer / bullish engulf no nível
4. Entregar veredicto direto:
   - ✅ 3/3 → **ENTRAR — Setup A+** (2 contratos) | Entrada: X | SL: X | TP1: X
   - ✅ 2/3 → **ENTRAR — Setup A** (1 contrato) | Entrada: X | SL: X | TP1: X
   - ❌ 1/3 → **AGUARDAR** — falta [critério X]
   - ❌ 0/3 → **PASS** — setup não confirmado

Resposta máxima de 20 linhas. Sem introdução, direto ao veredicto.

## Comando "análise" / "daily briefing"

Quando o usuário pedir análise do mercado ou daily briefing, seguir o fluxo completo de `analise_workflow.md`:
- Step 0: Intermarket via WebSearch
- Steps 1-5: Daily → 4H → 15m → 5m → 1m
- Entregar todas as seções do output esperado

## Instrumento padrão
MNQ (CME_MINI:MNQ1!) salvo indicação contrária.

## Gerenciamento de risco
- SL máximo: $50 por operação (25 pts MNQ)
- Max loss diário: $100
- Nunca sugerir SL acima de $50 sem avisar explicitamente
