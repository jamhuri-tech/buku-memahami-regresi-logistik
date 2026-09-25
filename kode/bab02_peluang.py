"""Bab 2: peluang bersyarat, rumus Bayes, dan sebaran Bernoulli.

Tes penyakit: prevalensi 1%, sensitivitas 90%, spesifisitas 95%.
Rumus Bayes dihitung langsung, dalam bentuk odds, lalu diperiksa
dengan satu juta orang rekaan.
"""
import numpy as np

BENIH = 20260924
prev, sens, spes = 0.01, 0.90, 0.95

p_pos = sens * prev + (1 - spes) * (1 - prev)
p_sakit = sens * prev / p_pos
odds_awal = prev / (1 - prev)
rasio = sens / (1 - spes)
odds_akhir = odds_awal * rasio

print("(1) tes penyakit, rumus Bayes:")
print(f"    P(positif)          = {p_pos:.4f}")
print(f"    P(sakit | positif)  = {p_sakit:.4f}")
print(f"    odds awal {odds_awal:.4f} x rasio {rasio:.1f}"
      f" = odds akhir {odds_akhir:.4f}")
print(f"    peluang dari odds akhir = {odds_akhir / (1 + odds_akhir):.4f}")

rng = np.random.default_rng(BENIH)
n = 1_000_000
sakit = rng.random(n) < prev
positif = np.where(sakit, rng.random(n) < sens, rng.random(n) > spes)
sp = (sakit & positif).sum()
print(f"    satu juta orang: {positif.sum()} positif,")
print(f"    {sp} di antaranya sakit ({sp / positif.sum():.4f})")

print("(2) Bernoulli dengan p = 0,3, 100 000 undian:")
y = (rng.random(100_000) < 0.3).astype(float)
print(f"    rata-rata {y.mean():.4f}   (harapan p = 0.3)")
print(f"    varians   {y.var():.4f}   (p(1 - p) = 0.21)")
