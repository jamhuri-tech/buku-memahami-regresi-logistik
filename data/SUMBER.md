# Sumber data

Setiap data yang dipakai buku ini dicatat di sini: nama, alamat
unduhan, tanggal pengambilan, dan berkas kode yang memakainya. Data
lain di buku ini dibangkitkan sendiri oleh kode (benih 20260924).

## Heart Disease (Cleveland) — Bab 19

| | |
|---|---|
| Berkas | `processed.cleveland.data` (303 baris, 14 kolom, tanpa judul kolom; nilai hilang ditulis `?`) |
| Alamat | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data |
| Laman data | https://archive.ics.uci.edu/dataset/45/heart+disease (DOI 10.24432/C52P4X) |
| Diambil | 25 September 2026 |
| SHA-256 | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| Lisensi | CC BY 4.0 (UCI Machine Learning Repository) |
| Rujukan | Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. Detrano, R. dkk. (1989). *American Journal of Cardiology*, 64(5), 304–310. |
| Dipakai oleh | `kode/bab19_data.py` (kolom diberi nama Indonesia; 6 baris ber-`?` dibuang; `sakit` = num > 0) |

## Data yang tidak kita pakai

Titanic (Kaggle) dan Iris sudah dipakai buku *Gauss–Newton* untuk
regresi logistik dan softmax, sehingga tidak dipakai di sini.
