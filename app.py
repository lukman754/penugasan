import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def step_by_step_assignment(cost_matrix):
    """
    Menampilkan langkah-langkah perhitungan penugasan maksimal
    """
    steps = []
    
    # Langkah 1: Matriks Keuntungan Awal
    steps.append({
        'judul': 'Matriks Keuntungan Awal',
        'matriks': cost_matrix.copy()
    })
    
    # Langkah 2: Temukan nilai maksimum
    max_value = np.max(cost_matrix)
    steps.append({
        'judul': 'Nilai Maksimum Matriks',
        'keterangan': f'Nilai maksimum dalam matriks: {max_value}',
        'matriks': cost_matrix.copy()
    })
    
    # Langkah 3: Konversi ke matriks biaya
    profit_matrix = max_value - cost_matrix
    steps.append({
        'judul': 'Matriks Biaya (Konversi)',
        'keterangan': 'Matriks dikonversi dengan mengurangkan setiap elemen dari nilai maksimum',
        'matriks': profit_matrix
    })
    
    # Langkah 4: Penugasan menggunakan algoritma Hungarian
    row_ind, col_ind = linear_sum_assignment(profit_matrix)
    total_profit = cost_matrix[row_ind, col_ind].sum()
    
    steps.append({
        'judul': 'Hasil Penugasan Optimal',
        'keterangan': f'Total Keuntungan: {total_profit}',
        'penugasan': list(zip(row_ind, col_ind))
    })
    
    return steps, row_ind, col_ind, total_profit

def main():
    st.title("📊 Kalkulator Penugasan Maksimal dengan Langkah Detail")
    
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
            # Jalankan algoritma penugasan dengan langkah detail
            steps, row_ind, col_ind, total_profit = step_by_step_assignment(cost_matrix)
            
            # Tampilkan setiap langkah
            for i, step in enumerate(steps, 1):
                st.subheader(f"Langkah {i}: {step['judul']}")
                
                # Tampilkan keterangan jika ada
                if 'keterangan' in step:
                    st.write(step['keterangan'])
                
                # Tampilkan matriks jika ada
                if 'matriks' in step:
                    df = pd.DataFrame(
                        step['matriks'], 
                        columns=[f'Tugas {j+1}' for j in range(num_tasks)],
                        index=[f'Pekerja {i+1}' for i in range(num_workers)]
                    )
                    st.dataframe(df)
                
                # Tampilkan penugasan jika ada
                if 'penugasan' in step:
                    results = []
                    for worker, task in step['penugasan']:
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
