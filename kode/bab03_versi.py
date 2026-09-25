"""Bab 3: versi Python dan pustaka yang dipakai seluruh buku.

Python hanya dicetak sampai versi minor, karena versi patch-nya
berbeda di setiap komputer; pustaka dikunci di requirements.txt.

Keluarannya disalin ke halaman hak cipta (depan.tex).
"""
import platform

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import statsmodels

print("Python      ", ".".join(platform.python_version_tuple()[:2]))
print("NumPy       ", numpy.__version__)
print("SciPy       ", scipy.__version__)
print("scikit-learn", sklearn.__version__)
print("statsmodels ", statsmodels.__version__)
print("pandas      ", pandas.__version__)
print("Matplotlib  ", matplotlib.__version__)
