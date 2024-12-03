import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hungarian_method(cost_matrix):
    """
    Implementasi lengkap metode Hungarian untuk penugasan optimal dengan perhitungan maksimal
    """
    steps = []
    matrix = np.array(cost_matrix, dtype=float)
    n = matrix.shape[0]

    # Tentukan apakah mencari min atau max
    is_maximization = st.radio("Pilih Tipe Optimasi", ["Minimasi", "Maksimasi"], index=0) == "Maksimasi"

    # Tabel 1: Transformasi matriks
    if is_maximization:
        max_val = np.max(matrix)
        transformed_matrix = max_val - matrix
    else:
        transformed_matrix = matrix.copy()

    steps.append({
        'title': 'Tabel 1: Transformasi Matriks',
        'matrix': transformed_matrix.copy(),
        'optimization_type': 'Maksimasi' if is_maximization else 'Minimasi'
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

    # Tabel 4: Penutupan Garis Optimal dengan algoritma yang lebih canggih
    def advanced_cover_zeros(matrix):
        n = matrix.shape[0]
        zero_positions = np.argwhere(np.isclose(matrix, 0))
        
        # Inisialisasi
        row_covered = np.zeros(n, dtype=bool)
        col_covered = np.zeros(n, dtype=bool)
        lines = []

        # Algoritma penugasan
        def find_independent_zeros():
            independent_zeros = []
            marked_rows = set()
            marked_cols = set()
            
            for pos in zero_positions:
                row, col = pos
                if row not in marked_rows and col not in marked_cols:
                    independent_zeros.append((row, col))
                    marked_rows.add(row)
                    marked_cols.add(col)
            
            return independent_zeros

        independent_zeros = find_independent_zeros()
        
        # Konversi penugasan ke garis
        for row, col in independent_zeros:
            lines.append(('row', row))
            lines.append(('col', col))
            row_covered[row] = True
            col_covered[col] = True

        return lines, row_covered, col_covered

    # Eksekusi penutupan garis
    lines, row_covered, col_covered = advanced_cover_zeros(row_reduced_matrix)
    
    steps.append({
        'title': 'Tabel 4: Penutupan Garis Optimal',
        'matrix': row_reduced_matrix,
        'lines': lines,
        'row_covered': row_covered,
        'col_covered': col_covered
    })

    # Tabel 5: Penyesuaian Matriks
    def precise_matrix_adjustment(matrix, row_covered, col_covered):
        n = matrix.shape[0]
        uncovered_values = matrix[~row_covered][:, ~col_covered]
        
        if uncovered_values.size > 0:
            uncovered_min = np.min(uncovered_values)
        else:
            uncovered_min = 0

        adjusted_matrix = matrix.copy()
        for r in range(n):
            for c in range(n):
                if not row_covered[r] and not col_covered[c]:
                    adjusted_matrix[r, c] -= uncovered_min
                elif row_covered[r] and col_covered[c]:
                    adjusted_matrix[r, c] += uncovered_min
        
        return adjusted_matrix, uncovered_min

    adjusted_matrix, adjustment_value = precise_matrix_adjustment(
        row_reduced_matrix, row_covered, col_covered
    )
    
    steps.append({
        'title': 'Tabel 5: Penyesuaian Matriks',
        'matrix': adjusted_matrix,
        'adjustment_value': adjustment_value,
        'row_covered': row_covered,
        'col_covered': col_covered
    })

    # Tabel 6: Pencarian Penugasan Optimal
    def precise_optimal_assignment(matrix):
        n = matrix.shape[0]
        assignment = []
        used_rows = set()
        used_cols = set()

        # Tambahkan prioritas pada zero dengan posisi unik
        zero_positions = np.argwhere(np.isclose(matrix, 0))
        zero_counts_rows = np.sum(np.isclose(matrix, 0), axis=1)
        zero_counts_cols = np.sum(np.isclose(matrix, 0), axis=0)

        # Urutkan posisi zero berdasarkan keunikan
        zero_positions = sorted(
            zero_positions, 
            key=lambda pos: zero_counts_rows[pos[0]] + zero_counts_cols[pos[1]]
        )

        for row, col in zero_positions:
            if row not in used_rows and col not in used_cols:
                assignment.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        
        return assignment

    optimal_assignment = precise_optimal_assignment(adjusted_matrix)
    
    steps.append({
        'title': 'Tabel 6: Penugasan Optimal',
        'matrix': adjusted_matrix,
        'assignment': optimal_assignment
    })

    # Fungsi untuk menghitung total berdasarkan tipe optimasi
    def calculate_total(original_matrix, assignment, is_maximization):
        if is_maximization:
            total = sum(original_matrix[row][col] for row, col in assignment)
        else:
            total = sum(original_matrix[row][col] for row, col in assignment)
        return total

    total_value = calculate_total(cost_matrix, optimal_assignment, is_maximization)
    steps[-1]['total_value'] = total_value
    steps[-1]['optimization_type'] = 'Maksimasi' if is_maximization else 'Minimasi'

    return steps

def visualize_matrix_with_lines(matrix, lines, row_covered, col_covered):
    """
    Visualisasi matriks dengan garis penutup
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    
    # Gambar garis horizontal
    for line_type, idx in lines:
        if line_type == 'row':
            plt.axhline(y=idx+1, color='red', linestyle='--')
        else:
            plt.axvline(x=idx+1, color='red', linestyle='--')
    
    plt.title('Matriks dengan Garis Penutup')
    return plt

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
                st.write(f"Tipe Optimasi: {step['optimization_type']}")
            
            if step['title'] == 'Tabel 2: Pengurangan Kolom':
                st.write("Nilai Minimum Kolom:", step['col_mins'])
            
            if step['title'] == 'Tabel 3: Pengurangan Baris':
                st.write("Nilai Minimum Baris:", step['row_mins'])
            
            if step['title'] == 'Tabel 4: Penutupan Garis Optimal':
                # Tampilkan detail garis
                st.write("Garis Penutup:")
                for line_type, idx in step['lines']:
                    st.write(f"{line_type.capitalize()} {idx+1}")
                
                # Visualisasi
                st.subheader("Visualisasi Garis Penutup")
                optimal_plot = visualize_matrix_with_lines(
                    step['matrix'], 
                    step['lines'], 
                    step['row_covered'], 
                    step['col_covered']
                )
                st.pyplot(optimal_plot)
            
            if step['title'] == 'Tabel 5: Penyesuaian Matriks':
                st.write(f"Nilai Penyesuaian: {step['adjustment_value']}")
            
            if step['title'] == 'Tabel 6: Penugasan Optimal':
                # Tampilkan penugasan optimal
                st.subheader("Hasil Penugasan Optimal")
                assignment_text = []
                for row, col in step['assignment']:
                    assignment_text.append(f"Baris {row+1} → Kolom {col+1}")
                    st.write(f"Baris {row+1} ditugaskan ke Kolom {col+1} (Nilai: {matrix[row][col]})")
                
                # Tampilkan total nilai
                st.subheader("Ringkasan")
                st.write(f"Tipe Optimasi: {step['optimization_type']}")
                st.write(f"Total Nilai: {step['total_value']}")
                st.success(" | ".join(assignment_text))

if __name__ == "__main__":
    main()
