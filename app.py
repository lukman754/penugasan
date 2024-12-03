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

def calculate_reductions(cost_matrix):
    """
    Menghitung pengurangan per baris dan kolom
    """
    row_reductions = np.max(cost_matrix, axis=1) - np.min(cost_matrix, axis=1)
    col_reductions = np.max(cost_matrix, axis=0) - np.min(cost_matrix, axis=0)
    
    return row_reductions, col_reductions

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
            
            # Hitung dan tampilkan pengurangan
            st.subheader("Pengurangan Nilai")
            row_reductions, col_reductions = calculate_reductions(cost_matrix)
            
            # Tampilkan pengurangan per baris
            st.write("Pengurangan per Baris:")
            row_reduction_df = pd.DataFrame({
                'Pekerja': [f'Pekerja {i+1}' for i in range(num_workers)],
                'Pengurangan': row_reductions
            })
            st.dataframe(row_reduction_df)
            
            # Tampilkan pengurangan per kolom
            st.write("Pengurangan per Kolom:")
            col_reduction_df = pd.DataFrame({
                'Tugas': [f'Tugas {j+1}' for j in range(num_tasks)],
                'Pengurangan': col_reductions
            })
            st.dataframe(col_reduction_df)
            
            # Visualisasi matriks dengan garis optimal
            st.subheader("Visualisasi Penugasan Optimal")
            optimal_plot = plot_optimal_assignment(cost_matrix, row_ind, col_ind)
            st.pyplot(optimal_plot)
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
