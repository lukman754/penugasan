import streamlit as st
import numpy as np
import pandas as pd


def hungarian_method(cost_matrix):
    """
    Implementasi metode Hungarian untuk penugasan optimal (tanpa garis penutup).
    """
    steps = []
    matrix = np.array(cost_matrix, dtype=float)
    n = matrix.shape[0]

    # Tabel 1: Transformasi Matriks (untuk maksimisasi)
    max_val = np.max(matrix)
    transformed_matrix = max_val - matrix

    steps.append({
        'title': 'Tabel 1: Transformasi Matriks',
        'matrix': transformed_matrix.copy(),
        'max_value': max_val
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

    # Penugasan Optimal
    def find_optimal_assignment(matrix):
        n = matrix.shape[0]
        assignment = []
        used_rows = set()
        used_cols = set()

        # Cari zero yang unik
        zero_positions = np.argwhere(np.isclose(matrix, 0))
        for row, col in zero_positions:
            if row not in used_rows and col not in used_cols:
                assignment.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        
        return assignment

    optimal_assignment = find_optimal_assignment(row_reduced_matrix)
    
    steps.append({
        'title': 'Tabel 4: Penugasan Optimal',
        'matrix': row_reduced_matrix,
        'assignment': optimal_assignment,
        'total_value': sum(cost_matrix[row][col] for row, col in optimal_assignment)
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
            if step['title'] == 'Tabel 1: Transformasi Matriks':
                st.write(f"Nilai Maksimum: {step['max_value']}")
            
            if step['title'] == 'Tabel 2: Pengurangan Kolom':
                st.write("Nilai Minimum Kolom:", step['col_mins'])
            
            if step['title'] == 'Tabel 3: Pengurangan Baris':
                st.write("Nilai Minimum Baris:", step['row_mins'])
            
            if step['title'] == 'Tabel 4: Penugasan Optimal':
                # Tampilkan penugasan optimal
                st.subheader("Hasil Penugasan Optimal")
                assignment_data = []
                for row, col in step['assignment']:
                    assignment_data.append({
                        "Baris": row + 1,
                        "Kolom": col + 1,
                        "Nilai": matrix[row][col]
                    })
                assignment_df = pd.DataFrame(assignment_data)
                st.table(assignment_df)
                
                # Tampilkan total nilai
                st.subheader("Ringkasan")
                st.write(f"Total Nilai: {step['total_value']}")
                st.success("Penugasan Optimal Telah Dihitung")

if __name__ == "__main__":
    main()
