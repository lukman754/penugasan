import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def maximize_assignment(cost_matrix):
    """
    Menghitung penugasan maksimal dengan mengonversi nilai menjadi minimasi.
    """
    profit_matrix = np.max(cost_matrix) - cost_matrix
    row_ind, col_ind = linear_sum_assignment(profit_matrix)
    total_profit = cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_profit

def main():
    st.title("📊 Kalkulator Penugasan Maksimal")
    
    # Input dimensi matriks
    num_workers = st.number_input("Jumlah Pekerja", min_value=2, max_value=10, value=3)
    num_tasks = st.number_input("Jumlah Tugas", min_value=2, max_value=10, value=3)
    
    # Membuat matriks input
    st.subheader("Masukkan Matriks Keuntungan")
    
    # Inisialisasi matriks kosong
    matrix = []
    for i in range(num_workers):
        row = st.columns(num_tasks)
        matrix_row = []
        for j in range(num_tasks):
            with row[j]:
                value = st.number_input(
                    f'Pekerja {i+1} - Tugas {j+1}', 
                    min_value=0.0, 
                    value=0.0, 
                    step=0.1,
                    key=f'input_{i}_{j}'
                )
                matrix_row.append(value)
        matrix.append(matrix_row)
    
    # Konversi ke numpy array
    cost_matrix = np.array(matrix)
    
    # Tombol hitung
    if st.button("Hitung Penugasan Maksimal"):
        try:
            # Jalankan algoritma penugasan
            row_ind, col_ind, total_profit = maximize_assignment(cost_matrix)
            
            # Tampilkan matriks keuntungan
            st.subheader("Matriks Keuntungan")
            df = pd.DataFrame(
                cost_matrix, 
                columns=[f'Tugas {j+1}' for j in range(num_tasks)],
                index=[f'Pekerja {i+1}' for i in range(num_workers)]
            )
            st.dataframe(df)
            
            # Tampilkan hasil
            st.subheader("Hasil Penugasan")
            results = []
            for worker, task in zip(row_ind, col_ind):
                results.append({
                    'Pekerja': f'Pekerja {worker + 1}', 
                    'Tugas': f'Tugas {task + 1}', 
                    'Keuntungan': cost_matrix[worker, task]
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Total keuntungan
            st.metric("Total Keuntungan Maksimal", f"{total_profit:.2f}")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
