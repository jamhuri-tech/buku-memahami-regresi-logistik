"""Mencetak versi Python dan pustaka yang dipakai seluruh buku.

Keluarannya disalin ke halaman hak cipta (depan.tex).
"""
import platform

import matplotlib
import numpy
import scipy
import sklearn
import statsmodels

print("Python      ", platform.python_version())
print("NumPy       ", numpy.__version__)
print("SciPy       ", scipy.__version__)
print("scikit-learn", sklearn.__version__)
print("statsmodels ", statsmodels.__version__)
print("Matplotlib  ", matplotlib.__version__)
