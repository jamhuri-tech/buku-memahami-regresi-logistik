# Kode — Memahami Regresi Logistik

Repositori pendamping buku **_Memahami Regresi Logistik: Menaksir
peluang, bukan sekadar menebak kelas_** (Edisi Pertama, 2026) oleh
Mohammad Jamhuri, seri *Memahami*.

Berisi seluruh kode Python yang dipakai buku, per bab, beserta data dan
pembangkit gambarnya. **Setiap angka keluaran yang tercetak di buku
dihasilkan oleh kode di sini**, dan `periksa.py` membuktikannya: skrip
itu menjalankan ulang kode setiap bab dan mencocokkan hasilnya dengan
blok keluaran yang tercetak di buku.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)).

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-regresi-logistik.git
cd buku-memahami-regresi-logistik
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab04_model.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
selalu memakai benih tetap (20260924), sehingga keluarannya sama setiap
kali dijalankan.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab
.venv/bin/python periksa.py 04     # Bab 4 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. Pemeriksaan yang sama berjalan otomatis di GitHub
Actions setiap kali kode berubah.

## Struktur

```
kode/
  bab04_data.py      data contoh berjalan (Bab 4-7)
  bab04_*.py         kode Bab 4, dan seterusnya per bab
  bab03_versi.py     mencetak versi Python dan pustaka
data/SUMBER.md       asal setiap data: alamat, tanggal pengambilan
keluaran/babNN.txt   blok keluaran yang tercetak di Bab NN
gen_gambar.py        membangkitkan semua gambar buku ke gbr/
periksa.py           mencocokkan kode dengan keluaran/
requirements.txt     versi pustaka yang dipakai buku
```

Nama berkas kode di buku sama dengan nama di sini: listing yang
merujuk `kode/bab04_sigmoid.py` berasal dari berkas itu.

`python gen_gambar.py bab04` membangkitkan gambar Bab 4 saja.

## Versi

Tag `edisi-1` menandai kode yang tepat dipakai untuk mencetak Edisi
Pertama. Cabang `main` dapat memuat perbaikan sesudahnya; setiap
perbaikan yang mengubah angka di buku dicatat di daftar errata.

`requirements.txt` mematok versi yang dipakai buku (Python 3.13.5,
NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, statsmodels 0.14.4,
Matplotlib 3.10.0). Versi lain biasanya berjalan, tetapi digit terakhir
sebagian angka dapat berbeda.

## Salah ketik, galat, dan saran

Silakan buka [issue](../../issues) di repositori ini: sebutkan bab,
halaman, dan apa yang keliru. Saran juga dapat dikirim ke
m.jamhuri@live.com.
