import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hungarian_method(cost_matrix):
    """
    Implementasi lengkap metode Hungarian untuk penugasan optimal.
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

    # Tabel 4: Penutupan Garis Optimal
    def cover_zeros_optimally(matrix):
        n = matrix.shape[0]
        zero_positions = np.argwhere(np.isclose(matrix, 0))
        
        # Inisialisasi variabel untuk pelacakan
        row_covered = np.zeros(n, dtype=bool)
        col_covered = np.zeros(n, dtype=bool)
        lines = []

        # Coba tutupi zero dengan garis horizontal dan vertikal
        while len(lines) < n:
            # Cari baris dengan zero terbanyak yang belum tertutup
            best_row = -1
            max_zeros = -1
            for row in range(n):
                if not row_covered[row]:
                    zero_count = np.sum((zero_positions[:, 0] == row) & 
                                        (~col_covered[zero_positions[:, 1]]))
                    if zero_count > max_zeros:
                        max_zeros = zero_count
                        best_row = row

            if best_row == -1:
                break

            # Tandai baris dan kolom yang sesuai dengan zero
            row_zeros = zero_positions[
                (zero_positions[:, 0] == best_row) & 
                (~col_covered[zero_positions[:, 1]])
            ]

            if len(row_zeros) > 0:
                # Pilih kolom pertama
                col = row_zeros[0, 1]
                lines.append(('row', best_row))
                lines.append(('col', col))
                row_covered[best_row] = True
                col_covered[col] = True

        return lines, row_covered, col_covered

    # Eksekusi penutupan garis
    lines, row_covered, col_covered = cover_zeros_optimally(row_reduced_matrix)
    
    steps.append({
        'title': 'Tabel 4: Penutupan Garis Optimal',
        'matrix': row_reduced_matrix,
        'lines': lines,
        'row_covered': row_covered,
        'col_covered': col_covered
    })

    # Tabel 5: Penyesuaian Matriks
    def adjust_matrix(matrix, row_covered, col_covered):
        n = matrix.shape[0]
        # Temukan nilai terkecil yang tidak tertutup
        uncovered_min = np.inf
        for r in range(n):
            for c in range(n):
                if not row_covered[r] and not col_covered[c]:
                    uncovered_min = min(uncovered_min, matrix[r, c])
        
        # Kurangi nilai yang tidak tertutup
        adjusted_matrix = matrix.copy()
        for r in range(n):
            for c in range(n):
                if not row_covered[r] and not col_covered[c]:
                    adjusted_matrix[r, c] -= uncovered_min
                elif row_covered[r] and col_covered[c]:
                    adjusted_matrix[r, c] += uncovered_min
        
        return adjusted_matrix, uncovered_min

    adjusted_matrix, adjustment_value = adjust_matrix(row_reduced_matrix, row_covered, col_covered)
    
    steps.append({
        'title': 'Tabel 5: Penyesuaian Matriks',
        'matrix': adjusted_matrix,
        'adjustment_value': adjustment_value,
        'row_covered': row_covered,
        'col_covered': col_covered
    })

    # Tabel 6: Pencarian Penugasan Optimal
    def find_optimal_assignment(matrix):
        n = matrix.shape[0]
        assignment = []
        used_rows = set()
        used_cols = set()

        # Cari penugasan optimal
        for _ in range(n):
            zero_positions = np.argwhere(np.isclose(matrix, 0))
            for pos in zero_positions:
                row, col = pos
                if row not in used_rows and col not in used_cols:
                    assignment.append((row, col))
                    used_rows.add(row)
                    used_cols.add(col)
                    break
        
        return assignment

    optimal_assignment = find_optimal_assignment(adjusted_matrix)
    
    steps.append({
        'title': 'Tabel 6: Penugasan Optimal',
        'matrix': adjusted_matrix,
        'assignment': optimal_assignment
    })

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

def calculate_total_cost(original_matrix, assignment):
    """
    Menghitung total biaya/keuntungan dari penugasan optimal
    """
    total_cost = sum(original_matrix[row][col] for row, col in assignment)
    return total_cost

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
                
                # Hitung total biaya/keuntungan
                total_cost = calculate_total_cost(matrix, step['assignment'])
                st.subheader("Ringkasan")
                st.write("Total Biaya/Keuntungan:", total_cost)
                st.success(" | ".join(assignment_text))

if __name__ == "__main__":
    main()
