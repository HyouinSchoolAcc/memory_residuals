
# Per-recipe corpus-level summary

| recipe | scale | n_seeds | Δ_cb (mean ± std) | Δ_sh random | Δ_sh same-category |
|---|---|---|---|---|---|
| v24a | 0p6b | 3 | +0.223 ± 0.143 | +0.0064 ± 0.0159 | +0.0150 ± 0.0057 |
| v27b | 0p6b | 4 | +1.318 ± 0.522 | -0.0005 ± 0.0095 | -0.0003 ± 0.0048 |
| v28a | 1p7b | 1 | +0.880 ± 0.000 | -0.0245 ± 0.0000 | -0.0057 ± 0.0000 |
| v28b | 1p7b | 1 | +0.945 ± 0.000 | -0.0085 ± 0.0000 | -0.0066 ± 0.0000 |
| v28c | 1p7b | 1 | +1.090 ± 0.000 | -0.0087 ± 0.0000 | -0.0222 ± 0.0000 |

# v27b (0p6b, n_seeds=4) — per-category

| category | n_chains | Δ_cb mean | Δ_sh rand | Δ_sh same-cat | Δ_sh hidden? (Δsamecat − Δrand) |
|---|---|---|---|---|---|
| knowledge-update | 6 | +2.613 | +0.0124 | +0.0081 | -0.0043 |
| multi-session | 13 | +1.112 | -0.0078 | -0.0077 | +0.0001 |
| single-session-assistant | 8 | +0.775 | +0.0106 | -0.0113 | -0.0219 **★** |
| single-session-preference | 4 | +0.330 | -0.0007 | +0.0008 | +0.0016 |
| single-session-user | 13 | +1.704 | -0.0121 | +0.0024 | +0.0145 |
| temporal-reasoning | 6 | +1.014 | +0.0129 | +0.0153 | +0.0024 |

# v28a (1p7b, n_seeds=1) — per-category

| category | n_chains | Δ_cb mean | Δ_sh rand | Δ_sh same-cat | Δ_sh hidden? (Δsamecat − Δrand) |
|---|---|---|---|---|---|
| knowledge-update | 6 | +1.939 | -0.0753 | -0.0214 | +0.0539 **★** |
| multi-session | 13 | +1.069 | -0.0026 | -0.0054 | -0.0028 |
| single-session-assistant | 8 | +0.536 | -0.0374 | -0.0152 | +0.0221 **★** |
| single-session-preference | 4 | +0.443 | -0.0067 | -0.0052 | +0.0015 |
| single-session-user | 13 | +0.832 | -0.0289 | -0.0026 | +0.0263 **★** |
| temporal-reasoning | 6 | +0.268 | -0.0063 | +0.0149 | +0.0212 **★** |

# v24a (0p6b, n_seeds=3) — per-category

| category | n_chains | Δ_cb mean | Δ_sh rand | Δ_sh same-cat | Δ_sh hidden? (Δsamecat − Δrand) |
|---|---|---|---|---|---|
| knowledge-update | 6 | +0.406 | -0.0436 | +0.0436 | +0.0872 **★** |
| multi-session | 13 | +0.300 | -0.0156 | -0.0127 | +0.0029 |
| single-session-assistant | 8 | +0.096 | -0.0048 | -0.0057 | -0.0009 |
| single-session-preference | 4 | +0.005 | -0.0006 | +0.0040 | +0.0046 |
| single-session-user | 13 | +0.248 | +0.0575 | +0.0476 | -0.0099 |
| temporal-reasoning | 6 | +0.134 | +0.0131 | +0.0106 | -0.0025 |

# v27b 0.6B per-seed

| seed | Δ_cb | Δ_sh rand | Δ_sh same-cat |
|---|---|---|---|
| 1 | +0.802 | -0.0011 | +0.0040 |
| 2 | +0.940 | +0.0132 | +0.0036 |
| 3 | +1.829 | -0.0080 | -0.0039 |
| 4 | +1.700 | -0.0060 | -0.0050 |

# v28 1.7B per-seed

| recipe | seed | Δ_cb | Δ_sh rand | Δ_sh same-cat |
|---|---|---|---|---|
| v28a | 1 | +0.880 | -0.0245 | -0.0057 |
| v28b | 2 | +0.945 | -0.0085 | -0.0066 |
| v28c | 3 | +1.090 | -0.0087 | -0.0222 |
