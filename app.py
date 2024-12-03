import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations

def calculate_total_benefit(matrix, assignment):
    """
    Menghitung total keuntungan dari penugasan
    """
    return sum(matrix[row][col] for row, col in assignment)

def hungarian_method_maximize(cost_matrix):
    """
    Implementasi Metode Hungarian untuk Maksimalisasi Keuntungan
    """
    matrix = np.array(cost_matrix)
    n = matrix.shape[0]
    
    # Langkah 1: Cari semua kemungkinan penugasan
    all_assignments = list(permutations(range(n)))
    
    # Simpan penugasan terbaik
    best_assignment = None
    max_benefit = float('-inf')
    
    # Evaluasi setiap kemungkinan penugasan
    for assignment in all_assignments:
        # Hitung keuntungan untuk penugasan ini
        current_benefit = sum(matrix[row][col] for row, col in enumerate(assignment))
        
        # Update jika keuntungan lebih besar
        if current_benefit > max_benefit:
            max_benefit = current_benefit
            best_assignment = list(enumerate(assignment))
    
    return {
        'best_assignment': best_assignment,
        'max_benefit': max_benefit
    }

def visualize_assignment(matrix, best_assignment):
    """
    Visualisasi penugasan terbaik
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    
    # Tandai sel yang dipilih
    for row, col in best_assignment:
        plt.text(col+0.5, row+0.5, '✓', 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 color='red', 
                 fontsize=15, 
                 fontweight='bold')
    
    plt.title('Visualisasi Penugasan Optimal')
    return plt

def main():
    st.title("🚀 Optimasi Penugasan untuk Maksimalisasi Keuntungan")
    
    # Input matriks
    st.subheader("Masukkan Matriks Keuntungan")
    
    # Dimensi default
    num_workers = st.number_input("Jumlah Baris (Pekerja/Sumber)", min_value=2, max_value=6, value=3)
    num_tasks = st.number_input("Jumlah Kolom (Tugas/Tujuan)", min_value=2, max_value=6, value=3)
    
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
        # Konversi ke numpy array
        np_matrix = np.array(matrix)
        
        # Jalankan optimasi
        result = hungarian_method_maximize(np_matrix)
        
        # Tampilkan hasil
        st.subheader("📊 Hasil Optimasi Penugasan")
        
        # Tabel ringkasan
        summary_data = []
        for row, col in result['best_assignment']:
            summary_data.append({
                'Pekerja/Sumber': f'Baris {row+1}', 
                'Tugas/Tujuan': f'Kolom {col+1}', 
                'Keuntungan': matrix[row][col]
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # Total keuntungan
        st.metric("Total Keuntungan Maksimal", f"{result['max_benefit']:.2f}")
        
        # Detail penugasan
        st.subheader("📝 Detail Penugasan")
        for row, col in result['best_assignment']:
            st.write(f"• Baris {row+1} ditugaskan ke Kolom {col+1} dengan keuntungan {matrix[row][col]}")
        
        # Visualisasi
        st.subheader("🖼️ Visualisasi Penugasan")
        assignment_plot = visualize_assignment(np_matrix, result['best_assignment'])
        st.pyplot(assignment_plot)
        
        # Penjelasan detail
        st.subheader("🔍 Analisis Rinci")
        st.write("Metode ini mengevaluasi seluruh kemungkinan kombinasi penugasan untuk memaksimalkan total keuntungan.")
        st.write("Algoritma memeriksa setiap kemungkinan penugasan dan memilih kombinasi dengan keuntungan tertinggi.")

if __name__ == "__main__":
    main()
