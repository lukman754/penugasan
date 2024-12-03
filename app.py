import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def step_by_step_assignment(cost_matrix):
    """
    Menampilkan langkah-langkah perhitungan penugasan maksimal secara detail
    """
    steps = []
    num_workers, num_tasks = cost_matrix.shape
    
    # Langkah 1: Matriks Keuntungan Awal
    steps.append({
        'judul': 'Matriks Keuntungan Awal',
        'matriks': cost_matrix.copy(),
        'keterangan': 'Matriks keuntungan awal yang diinputkan'
    })
    
    # Langkah 2: Temukan nilai maksimum dan minimum
    max_value = np.max(cost_matrix)
    min_value = np.min(cost_matrix)
    steps.append({
        'judul': 'Analisis Nilai Matriks',
        'keterangan': f'Nilai Maksimum: {max_value}\nNilai Minimum: {min_value}'
    })
    
    # Langkah 3: Kurangi setiap baris dengan nilai minimum baris
    row_reduced_matrix = cost_matrix.copy()
    row_reductions = []
    for i in range(num_workers):
        row_min = np.min(row_reduced_matrix[i])
        row_reduced_matrix[i] -= row_min
        row_reductions.append(row_min)
    
    steps.append({
        'judul': 'Pengurangan Nilai per Baris',
        'matriks': row_reduced_matrix,
        'keterangan': f'Pengurangan per Baris: {row_reductions}',
        'detail_reduksi': row_reductions
    })
    
    # Langkah 4: Kurangi setiap kolom dengan nilai minimum kolom
    col_reduced_matrix = row_reduced_matrix.copy()
    col_reductions = []
    for j in range(num_tasks):
        col_min = np.min(col_reduced_matrix[:, j])
        col_reduced_matrix[:, j] -= col_min
        col_reductions.append(col_min)
    
    steps.append({
        'judul': 'Pengurangan Nilai per Kolom',
        'matriks': col_reduced_matrix,
        'keterangan': f'Pengurangan per Kolom: {col_reductions}',
        'detail_reduksi': col_reductions
    })
    
    # Langkah 5: Algoritma Penugasan Hungarian
    row_ind, col_ind = linear_sum_assignment(col_reduced_matrix)
    total_profit = cost_matrix[row_ind, col_ind].sum()
    
    steps.append({
        'judul': 'Hasil Penugasan Optimal',
        'keterangan': f'Total Keuntungan: {total_profit}',
        'penugasan': list(zip(row_ind, col_ind))
    })
    
    return steps, row_ind, col_ind, total_profit, col_reduced_matrix

def plot_assignment(cost_matrix, row_ind, col_ind):
    """
    Membuat visualisasi garis penugasan optimal
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(cost_matrix, cmap='YlGnBu')
    plt.title('Visualisasi Penugasan Optimal')
    plt.xlabel('Tugas')
    plt.ylabel('Pekerja')
    
    # Tambahkan label
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            plt.text(j, i, f'{cost_matrix[i,j]:.1f}', 
                     ha='center', va='center', color='black')
    
    # Gambar garis untuk penugasan optimal
    for worker, task in zip(row_ind, col_ind):
        plt.plot(task, worker, 'ro')  # Titik merah untuk penugasan
    
    plt.colorbar(label='Keuntungan')
    return plt

def main():
    st.title("📊 Kalkulator Penugasan Maksimal dengan Analisis Detail")
    
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
            steps, row_ind, col_ind, total_profit, final_matrix = step_by_step_assignment(cost_matrix)
            
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
                
                # Tampilkan detail reduksi jika ada
                if 'detail_reduksi' in step:
                    st.write("Nilai Reduksi:", step['detail_reduksi'])
                
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
            
            # Visualisasi penugasan
            st.subheader("Visualisasi Penugasan Optimal")
            fig = plot_assignment(cost_matrix, row_ind, col_ind)
            st.pyplot(fig)
            
            # Total keuntungan
            st.metric("Total Keuntungan Maksimal", f"{total_profit:.2f}")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
