import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hungarian_method(cost_matrix):
    """
    Implementasi lengkap metode Hungarian untuk penugasan optimal
    """
    steps = []
    matrix = np.array(cost_matrix, dtype=float)
    n = matrix.shape[0]

    # Tabel 1: Transformasi Matriks
    max_val = np.max(matrix)
    transformed_matrix = max_val - matrix

    steps.append({
        'title': 'Tabel 1: Transformasi Matriks',
        'matrix': transformed_matrix.copy(),
        'max_value': max_val
    })

    # Tabel 2: Transformasi Seluruh Tabel
    steps.append({
        'title': 'Tabel 2: Transformasi Seluruh Tabel',
        'matrix': transformed_matrix.copy()
    })

    # Tabel 3: Pengurangan Kolom
    col_mins = np.min(transformed_matrix, axis=0)
    col_reduced_matrix = transformed_matrix - col_mins

    steps.append({
        'title': 'Tabel 3: Pengurangan Kolom',
        'matrix': col_reduced_matrix.copy(),
        'col_mins': col_mins
    })

    # Tabel 4: Pengurangan Baris
    row_mins = np.min(col_reduced_matrix, axis=1)
    row_reduced_matrix = col_reduced_matrix - row_mins[:, np.newaxis]

    steps.append({
        'title': 'Tabel 4: Pengurangan Baris',
        'matrix': row_reduced_matrix.copy(),
        'row_mins': row_mins
    })

    # Tabel 5: Penutupan Garis Optimal
    def cover_zeros(matrix):
        n = matrix.shape[0]
        zero_positions = np.argwhere(np.isclose(matrix, 0))
        
        # Inisialisasi
        row_covered = np.zeros(n, dtype=bool)
        col_covered = np.zeros(n, dtype=bool)
        lines = []

        # Hitung zero di setiap baris dan kolom
        zero_counts_rows = np.sum(np.isclose(matrix, 0), axis=1)
        zero_counts_cols = np.sum(np.isclose(matrix, 0), axis=0)

        # Prioritaskan baris/kolom dengan zero
        while len(lines) < n:
            # Cari baris atau kolom dengan zero yang belum tertutup
            uncovered_zero_rows = np.where((~row_covered) & (zero_counts_rows > 0))[0]
            uncovered_zero_cols = np.where((~col_covered) & (zero_counts_cols > 0))[0]

            if uncovered_zero_rows.size > 0:
                row = uncovered_zero_rows[0]
                lines.append(('row', row))
                row_covered[row] = True
            elif uncovered_zero_cols.size > 0:
                col = uncovered_zero_cols[0]
                lines.append(('col', col))
                col_covered[col] = True
            else:
                # Jika tidak ada zero lagi, pilih baris/kolom yang belum tertutup
                if not np.all(row_covered):
                    row = np.where(~row_covered)[0][0]
                    lines.append(('row', row))
                    row_covered[row] = True
                elif not np.all(col_covered):
                    col = np.where(~col_covered)[0][0]
                    lines.append(('col', col))
                    col_covered[col] = True

        return lines, row_covered, col_covered

    lines, row_covered, col_covered = cover_zeros(row_reduced_matrix)
    
    steps.append({
        'title': 'Tabel 5: Penutupan Garis Optimal',
        'matrix': row_reduced_matrix,
        'lines': lines,
        'row_covered': row_covered,
        'col_covered': col_covered
    })

    # Tambahkan perhitungan total nilai
    total_value = sum(cost_matrix[row][col] for row, col in [(l[1], l[1]) for l in lines if l[0] == 'row'])

    steps.append({
        'title': 'Total Penugasan',
        'total_value': total_value,
        'lines': lines
    })

    return steps



def visualize_matrix_with_lines(matrix, lines, row_covered, col_covered):
    """
    Visualisasi matriks dengan garis penutup
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    
    # Gambar garis horizontal dan vertikal
    for line_type, idx in lines:
        if line_type == 'row':
            plt.axhline(y=idx+0.5, color='red', linestyle='--')
        else:
            plt.axvline(x=idx+0.5, color='red', linestyle='--')
    
    plt.title('Matriks dengan Garis Penutup')
    return plt

def main():
    st.title("🔢 Metode Harian (Penugasan Optimal)")
    
    # Input matriks
    st.subheader("Masukkan Matriks Biaya/Keuntungan")
    
    # Dimensi default
    num_workers = st.number_input("Jumlah Baris", min_value=2, max_value=10, value=3)
    num_tasks = st.number_input("Jumlah Kolom", min_value=2, max_value=10, value=3)
    
    # Tipe optimasi di awal
    is_maximization = st.radio("Pilih Tipe Optimasi", ["Minimasi", "Maksimasi"], index=0) == "Maksimasi"
    
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
            
            if step['title'] == 'Tabel 3: Pengurangan Kolom':
                st.write("Nilai Minimum Kolom:", step['col_mins'])
            
            if step['title'] == 'Tabel 4: Pengurangan Baris':
                st.write("Nilai Minimum Baris:", step['row_mins'])
            
            if step['title'] == 'Tabel 5: Penutupan Garis Optimal':
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
            
            if step['title'] == 'Tabel 6: Penyesuaian Matriks':
                st.write(f"Nilai Terkecil Tidak Tertutup: {step['uncovered_min']}")
            
            if step['title'] == 'Tabel 7: Penugasan Optimal':
                # Tampilkan penugasan optimal
                st.subheader("Hasil Penugasan Optimal")
                assignment_text = []
                for row, col in step['assignment']:
                    assignment_text.append(f"Baris {row+1} → Kolom {col+1}")
                    st.write(f"Baris {row+1} ditugaskan ke Kolom {col+1} (Nilai: {matrix[row][col]})")
                
                # Tampilkan total nilai
                st.subheader("Ringkasan")
                st.write(f"Total Nilai: {step['total_value']}")
                st.success(" | ".join(assignment_text))

if __name__ == "__main__":
    main()
