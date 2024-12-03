import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

def maximize_assignment(cost_matrix):
    """
    Menghitung penugasan maksimal dengan mengonversi nilai menjadi minimasi.
    """
    profit_matrix = np.max(cost_matrix) - cost_matrix
    row_ind, col_ind = linear_sum_assignment(profit_matrix)
    total_profit = cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_profit

def plot_optimal_assignment(cost_matrix, row_ind, col_ind):
    """
    Membuat visualisasi matriks dengan garis optimal
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cost_matrix, annot=True, cmap='YlGnBu', fmt='.2f', 
                xticklabels=[f'Tugas {j+1}' for j in range(cost_matrix.shape[1])],
                yticklabels=[f'Pekerja {i+1}' for i in range(cost_matrix.shape[0])])
    
    # Tambahkan garis untuk penugasan optimal
    for worker, task in zip(row_ind, col_ind):
        plt.plot([task, task+1], [worker, worker+1], color='red', linewidth=2)
    
    plt.title('Matriks Penugasan Optimal')
    plt.tight_layout()
    return plt

def calculate_reduced_matrix(cost_matrix):
    """
    Menghitung matriks reduksi dengan mengurangkan minimum dari setiap baris dan kolom
    """
    # Kurangi minimum dari setiap baris
    row_reduced_matrix = cost_matrix.copy()
    for i in range(row_reduced_matrix.shape[0]):
        row_min = np.min(row_reduced_matrix[i, :])
        row_reduced_matrix[i, :] -= row_min
    
    # Kurangi minimum dari setiap kolom
    col_reduced_matrix = row_reduced_matrix.copy()
    for j in range(col_reduced_matrix.shape[1]):
        col_min = np.min(col_reduced_matrix[:, j])
        col_reduced_matrix[:, j] -= col_min
    
    return col_reduced_matrix

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
            
            # Hitung matriks reduksi
            reduced_matrix = calculate_reduced_matrix(cost_matrix)
            
            # Tampilkan matriks reduksi
            st.subheader("Matriks Reduksi")
            reduced_df = pd.DataFrame(
                reduced_matrix, 
                columns=[f'Tugas {j+1}' for j in range(num_tasks)],
                index=[f'Pekerja {i+1}' for i in range(num_workers)]
            )
            st.dataframe(reduced_df)
            
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
            
            # Visualisasi matriks dengan garis optimal
            st.subheader("Visualisasi Penugasan Optimal")
            optimal_plot = plot_optimal_assignment(cost_matrix, row_ind, col_ind)
            st.pyplot(optimal_plot)
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
