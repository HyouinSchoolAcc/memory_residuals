# Memory Residuals Data Audit

Tokenizer: `Qwen/Qwen3-0.6B`

## Stage 1

| Source | Split | Books/Shows | Sessions | Chars | Qwen tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| PG-19 | train | 500 | 24,627 | 180,742,266 | 44,452,040 |
| PG-19 | validation | 50 | 2,151 | 14,658,895 | 3,586,601 |
| PG-19 | test | 100 | 3,665 | 35,208,615 | 8,708,710 |
| TV | train | 30 | 2,189 | 61,040,901 | 16,293,769 |

## Stage 2

| Source | Shows | Episodes | Turns | Chars | Qwen tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| TV SFT | 30 | 2,189 | 1,274,960 | 72,940,305 | 19,554,780 |

## Eval Sets

| Benchmark | Split | Rows/items | Turns | Chars | Qwen tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSC | train | 17,940 | 236,987 | 19,525,293 | 4,731,924 |
| MSC | validation | 3,000 | 39,257 | 3,699,116 | 886,632 |
| MSC | test | 2,505 | 30,320 | 3,464,720 | 822,675 |
| LoCoMo-MC10 | all | 1,986 | - | - | - |
