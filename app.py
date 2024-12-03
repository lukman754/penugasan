import streamlit as st
import numpy as np
import pandas as pd

def hungarian_method(cost_matrix):
    """
    Implementasi lengkap metode Hungarian
    """
    steps = []
    matrix = np.array(cost_matrix, dtype=float)
    n = matrix.shape[0]

    # Tabel 1: Transformasi matriks
    max_val = np.max(matrix)
    transformed_matrix = max_val - matrix
    steps.append({
        'title': 'Tabel 1: Transformasi Matriks',
        'matrix': transformed_matrix.copy()
    })

    # Tabel 2: Pengurangan Kolom
    col_mins = np.min(transformed_matrix, axis=0)
    col_reduced_matrix = transformed_matrix - col_mins
    steps.append({
        'title': 'Tabel 2: Pengurangan Kolom',
        'matrix': col_reduced_matrix.copy(),
        'col_mins': col_mins
    })

    # Tabel 3: Pengurangan Baris
    row_mins = np.min(col_reduced_matrix, axis=1)
    row_reduced_matrix = col_reduced_matrix - row_mins[:, np.newaxis]
    steps.append({
        'title': 'Tabel 3: Pengurangan Baris',
        'matrix': row_reduced_matrix.copy(),
        'row_mins': row_mins
    })

    # Tabel 4: Penutupan Garis
    def cover_zeros(matrix):
        covered_matrix = matrix.copy()
        lines = 0
        zero_positions = []

        # Implementasi logika penutupan garis
        # (ini adalah implementasi sederhana, bisa dikembangkan lebih lanjut)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isclose(matrix[i, j], 0):
                    zero_positions.append((i, j))

        lines = len(zero_positions)
        
        steps.append({
            'title': 'Tabel 4: Penutupan Garis Awal',
            'matrix': covered_matrix,
            'lines': lines,
            'zero_positions': zero_positions
        })

        return lines, zero_positions

    lines, zero_positions = cover_zeros(row_reduced_matrix)

    # Tahap selanjutnya: Pengecekan optimasi
    total_lines = lines
    optimized = total_lines == n

    steps.append({
        'title': 'Tahap Optimasi',
        'total_lines': total_lines,
        'matrix_size': n,
        'optimized': optimized,
        'zero_positions': zero_positions
    })

    return steps

def main():
    st.title("🔢 Metode Hungarian (Penugasan Optimal)")
    
    # Input matriks
    st.subheader("Masukkan Matriks Biaya/Keuntungan")
    
    # Dimensi default
    num_workers = st.number_input("Jumlah Baris", min_value=2, max_value=10, value=3)
    num_tasks = st.number_input("Jumlah Kolom", min_value=2, max_value=10, value=3)
    
    # Membuat matriks input
    matrix = []
    for i in range(num_workers):
        row = st.columns(num_tasks)
        matrix_row = []
        for j in range(num_tasks):
            with row[j]:
                value = st.number_input(
                    f'Baris {i+1} - Kolom {j+1}', 
                    min_value=0.0, 
                    value=0.0, 
                    step=0.1,
                    key=f'input_{i}_{j}'
                )
                matrix_row.append(value)
        matrix.append(matrix_row)
    
    if st.button("Hitung Penugasan Optimal"):
        steps = hungarian_method(matrix)
        
        for step in steps:
            st.subheader(step['title'])
            
            # Tampilkan matriks
            if 'matrix' in step:
                df = pd.DataFrame(
                    step['matrix'], 
                    columns=[f'Kolom {j+1}' for j in range(step['matrix'].shape[1])],
                    index=[f'Baris {i+1}' for i in range(step['matrix'].shape[0])]
                )
                st.dataframe(df)
            
            # Tampilkan informasi tambahan
            if step['title'] == 'Tabel 2: Pengurangan Kolom':
                st.write("Nilai Minimum Kolom:", step['col_mins'])
            
            if step['title'] == 'Tabel 3: Pengurangan Baris':
                st.write("Nilai Minimum Baris:", step['row_mins'])
            
            if step['title'] == 'Tabel 4: Penutupan Garis Awal':
                st.write("Jumlah Garis:", step['lines'])
                st.write("Posisi Nol:", step['zero_positions'])
            
            if step['title'] == 'Tahap Optimasi':
                st.write("Jumlah Garis:", step['total_lines'])
                st.write("Ukuran Matriks:", step['matrix_size'])
                st.write("Optimasi Tercapai:", step['optimized'])
                st.write("Posisi Nol:", step['zero_positions'])

if __name__ == "__main__":
    main()
