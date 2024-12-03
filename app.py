import streamlit as st
import numpy as np
import pandas as pd

def hungarian_method(cost_matrix):
    """
    Implementasi Hungarian Method dengan langkah penutupan garis yang optimal
    untuk menyelesaikan masalah penugasan.
    """
    steps = []
    matrix = np.array(cost_matrix, dtype=float)
    n, m = matrix.shape

    # Step 1: Transformasi Matriks (jika perlu, untuk maksimisasi)
    max_val = np.max(matrix)
    transformed_matrix = max_val - matrix

    steps.append({
        'title': 'Tabel 1: Transformasi Matriks',
        'matrix': transformed_matrix.copy(),
        'max_value': max_val
    })

    # Step 2: Pengurangan Kolom
    col_mins = np.min(transformed_matrix, axis=0)
    transformed_matrix -= col_mins

    steps.append({
        'title': 'Tabel 2: Pengurangan Kolom',
        'matrix': transformed_matrix.copy(),
        'col_mins': col_mins
    })

    # Step 3: Pengurangan Baris
    row_mins = np.min(transformed_matrix, axis=1)
    transformed_matrix -= row_mins[:, np.newaxis]

    steps.append({
        'title': 'Tabel 3: Pengurangan Baris',
        'matrix': transformed_matrix.copy(),
        'row_mins': row_mins
    })

    # Fungsi untuk menemukan penugasan optimal
    def find_optimal_assignment(matrix):
        n = matrix.shape[0]
        assignment = []
        used_rows = set()
        used_cols = set()
        
        zero_positions = np.argwhere(np.isclose(matrix, 0))
        for row, col in zero_positions:
            if row not in used_rows and col not in used_cols:
                assignment.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        return assignment

    # Langkah Penutupan Garis
    def cover_zeros(matrix):
        n, m = matrix.shape
        covered_rows = set()
        covered_cols = set()
        while True:
            # Cari jumlah nol di setiap baris dan kolom
            row_zero_counts = [np.sum(np.isclose(matrix[row, :], 0)) for row in range(n)]
            col_zero_counts = [np.sum(np.isclose(matrix[:, col], 0)) for col in range(m)]

            # Pilih baris/kolom dengan jumlah nol terbanyak
            if max(row_zero_counts) >= max(col_zero_counts):
                row = row_zero_counts.index(max(row_zero_counts))
                covered_rows.add(row)
                matrix[row, :] = np.inf  # Tandai sebagai tertutup
            else:
                col = col_zero_counts.index(max(col_zero_counts))
                covered_cols.add(col)
                matrix[:, col] = np.inf  # Tandai sebagai tertutup

            # Berhenti jika semua nol tertutup
            if all(np.isinf(matrix[row, :]).all() for row in range(n)) and \
               all(np.isinf(matrix[:, col]).all() for col in range(m)):
                break

        return covered_rows, covered_cols

    while True:
        # Lakukan langkah penutupan
        temp_matrix = transformed_matrix.copy()
        covered_rows, covered_cols = cover_zeros(temp_matrix)

        # Jika jumlah garis sama dengan jumlah baris atau kolom, selesai
        if len(covered_rows) + len(covered_cols) >= n:
            break

        # Jika tidak, modifikasi matriks
        uncovered_min = np.min(transformed_matrix[~np.isinf(temp_matrix)])
        for row in range(n):
            for col in range(m):
                if row not in covered_rows and col not in covered_cols:
                    transformed_matrix[row, col] -= uncovered_min
                elif row in covered_rows and col in covered_cols:
                    transformed_matrix[row, col] += uncovered_min

        steps.append({
            'title': 'Modifikasi Matriks (Kurangi Nilai Terkecil Tak Tertutup)',
            'matrix': transformed_matrix.copy(),
            'min_uncovered': uncovered_min
        })

    # Penugasan Optimal
    optimal_assignment = find_optimal_assignment(transformed_matrix)

    steps.append({
        'title': 'Tabel 4: Penugasan Optimal',
        'matrix': transformed_matrix,
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
            if 'min_uncovered' in step:
                st.write(f"Nilai Terkecil Tak Tertutup: {step['min_uncovered']}")
            
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
