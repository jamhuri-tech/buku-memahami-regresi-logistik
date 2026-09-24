"""Bab 5: log-loss sebagai fungsi margin, dan aturan skor yang jujur.

(1) Untuk label y di {0, 1} dan s = 2y - 1 di {-1, +1},
    log(1 + e^z) - y z = log(1 + e^(-s z)).
(2) Nilai beberapa loss pada margin m = s z tertentu.
(3) Harapan loss bila label sebenarnya Bernoulli(p*) dan model
    menjawab q: log-loss dan kuadrat galat minimum di q = p*,
    galat mutlak tidak.
"""
import numpy as np

from bab04_data import BENIH

if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    z = rng.normal(scale=5, size=100_000)
    y = rng.integers(0, 2, size=z.size)
    s = 2 * y - 1
    kiri = np.logaddexp(0, z) - y * z
    kanan = np.logaddexp(0, -s * z)
    print(f"(1) maks selisih kedua bentuk = "
          f"{np.max(np.abs(kiri - kanan)):.1e}")
    print("(2)    m    0-1  logistik  logistik/log2   hinge  eksponen")
    for m in (-2.0, -1.0, 0.0, 1.0, 2.0, 5.0):
        print(f"    {m:4.0f}  {float(m <= 0):5.0f}  "
              f"{np.logaddexp(0, -m):8.4f}  "
              f"{np.logaddexp(0, -m) / np.log(2):13.4f}  "
              f"{max(0.0, 1 - m):6.2f}  {np.exp(-m):8.4f}")
    q = np.linspace(0.0005, 0.9995, 1999)
    ps = 0.3
    harap = {
        "log-loss": -(ps * np.log(q) + (1 - ps) * np.log(1 - q)),
        "kuadrat ": ps * (1 - q) ** 2 + (1 - ps) * q ** 2,
        "mutlak  ": ps * (1 - q) + (1 - ps) * q,
    }
    print(f"(3) p* = {ps}: q yang meminimumkan harapan loss")
    for nama, h in harap.items():
        print(f"    {nama}: q = {q[np.argmin(h)]:.4f}, "
              f"harapan minimum {h.min():.4f}")
    ent = -(ps * np.log(ps) + (1 - ps) * np.log(1 - ps))
    print(f"    entropi H(0.3) = {ent:.4f}")
