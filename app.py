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
    st.title("📊 Kalkulator Metode Penugasan Maksimal")
    st.write("Aplikasi ini membantu Anda menentukan penugasan optimal dengan memaksimalkan keuntungan.")
    
    # Pilihan jumlah pekerja dan tugas
    col1, col2 = st.columns(2)
    with col1:
        num_workers = st.slider("Jumlah Pekerja", min_value=2, max_value=10, value=3)
    with col2:
        num_tasks = st.slider("Jumlah Tugas", min_value=2, max_value=10, value=3)
    
    # Membuat matriks keuntungan
    st.subheader("1. Masukkan Matriks Keuntungan")
    st.write("Isi tabel dengan keuntungan masing-masing pekerja untuk setiap tugas.")
    
    # Inisialisasi DataFrame
    df = pd.DataFrame(
        np.zeros((num_workers, num_tasks)), 
        columns=[f'Tugas {i+1}' for i in range(num_tasks)],
        index=[f'Pekerja {i+1}' for i in range(num_workers)]
    )
    
    # Gunakan session state untuk menyimpan input
    if 'matrix_input' not in st.session_state:
        st.session_state.matrix_input = df
    
    # Edit DataFrame
    edited_df = st.data_editor(
        st.session_state.matrix_input, 
        num_rows="fixed", 
        num_cols="fixed",
        key="profit_matrix"
    )
    
    # Perbarui session state
    st.session_state.matrix_input = edited_df
    
    # Tombol untuk menghitung
    if st.button("🔢 Hitung Penugasan Maksimal"):
        # Konversi ke numpy array
        try:
            cost_matrix = edited_df.values
            
            # Validasi input
            if np.any(np.isnan(cost_matrix)):
                st.error("Pastikan semua sel telah diisi dengan angka!")
                return
            
            # Jalankan algoritma penugasan
            row_ind, col_ind, total_profit = maximize_assignment(cost_matrix)
            
            # Tampilkan langkah-langkah
            st.subheader("2. Proses Perhitungan")
            
            # Tabel matriks keuntungan asli
            st.write("Matriks Keuntungan Awal:")
            st.dataframe(edited_df)
            
            # Tampilkan matriks profit (setelah konversi)
            profit_matrix = np.max(cost_matrix) - cost_matrix
            st.write("Matriks Biaya (Konversi untuk Algoritma):")
            st.dataframe(pd.DataFrame(
                profit_matrix, 
                columns=edited_df.columns,
                index=edited_df.index
            ))
            
            # Hasil penugasan
            st.subheader("3. Hasil Penugasan Optimal")
            
            # Buat tabel hasil
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
