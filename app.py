import streamlit as st
import numpy as np
from scipy.optimize import linear_sum_assignment

def maximize_assignment(cost_matrix):
    """
    Menghitung penugasan maksimal dengan mengonversi nilai menjadi minimasi.
    """
    profit_matrix = np.max(cost_matrix) - cost_matrix
    row_ind, col_ind = linear_sum_assignment(profit_matrix)
    total_profit = cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_profit

st.title("Kalkulator Metode Penugasan Maksimal")
st.write("Masukkan keuntungan untuk setiap pekerja dan tugas dalam tabel berikut:")

# Dinamis form untuk matriks keuntungan
matrix_input = st.experimental_data_editor(
    np.zeros((3, 3)),  # Matriks default 3x3
    num_rows="dynamic",
    num_cols="dynamic",
    key="profit_matrix"
)

# Pastikan input matriks valid
try:
    cost_matrix = np.array(matrix_input, dtype=float)

    # Tombol hitung
    if st.button("Hitung Penugasan Maksimal"):
        try:
            row_ind, col_ind, total_profit = maximize_assignment(cost_matrix)
            
            # Menampilkan hasil
            st.subheader("Hasil Penugasan:")
            for worker, task in zip(row_ind, col_ind):
                st.write(f"Pekerja {worker + 1} → Tugas {task + 1} (Keuntungan: {cost_matrix[worker, task]})")
            st.write(f"**Total Keuntungan Maksimal:** {total_profit}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
except ValueError:
    st.error("Pastikan semua elemen dalam tabel adalah angka valid.")
