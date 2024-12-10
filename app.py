import streamlit as st
import numpy as np
import pandas as pd


def hungarian_method(cost_matrix):
    """
    Implementasi metode Hungarian untuk penugasan optimal dengan penyesuaian nol bentrok.
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

    # Penyesuaian nol bentrok
    def adjust_matrix(matrix):
        n = matrix.shape[0]
        covered_rows = set()
        covered_cols = set()
        while True:
            zero_positions = np.argwhere(np.isclose(matrix, 0))
            row_zeros = {r: 0 for r in range(n)}
            col_zeros = {c: 0 for c in range(n)}

            # Hitung nol di setiap baris dan kolom
            for r, c in zero_positions:
                if r not in covered_rows and c not in covered_cols:
                    row_zeros[r] += 1
                    col_zeros[c] += 1

            # Periksa apakah ada bentrok (lebih dari 1 nol di baris/kolom)
            adjustments_needed = False
            for r, count in row_zeros.items():
                if count > 1:
                    adjustments_needed = True
            for c, count in col_zeros.items():
                if count > 1:
                    adjustments_needed = True

            if not adjustments_needed:
                break

            # Tambahkan garis penutup untuk menyelesaikan bentrok
            min_uncovered = np.min(matrix[np.isclose(matrix, 0) == False])
            matrix -= min_uncovered
            for r in covered_rows:
                matrix[r, :] += min_uncovered
            for c in covered_cols:
                matrix[:, c] -= min_uncovered

        return matrix

    adjusted_matrix = adjust_matrix(row_reduced_matrix)

    steps.append({
        'title': 'Tabel 4: Penyesuaian Nol Bentrok',
        'matrix': adjusted_matrix.copy()
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

    optimal_assignment = find_optimal_assignment(adjusted_matrix)
    
    steps.append({
        'title': 'Tabel 5: Penugasan Optimal',
        'matrix': adjusted_matrix,
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
            
            if step['title'] == 'Tabel 5: Penugasan Optimal':
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
