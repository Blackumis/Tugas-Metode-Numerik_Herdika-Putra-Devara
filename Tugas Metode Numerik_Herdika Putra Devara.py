import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Nama : Herdika Putra Devara
# NIM : 21120123140112
# Kelas : Metode Numerik B

class NonLinearSystemSolver:
    """
    Solver untuk sistem persamaan non-linear:
    f1(x,y) = x² + xy - 10 = 0
    f2(x,y) = y + 3xy² - 57 = 0
    
    NIM = 21120123140112 (mod 4 = 0) → Menggunakan g1A dan g2A untuk IT
    """
    
    def __init__(self, x0: float, y0: float, epsilon: float = 0.000001, max_iter: int = 1000):
        self.x0 = x0
        self.y0 = y0
        self.epsilon = epsilon
        self.max_iter = max_iter
        
    def f1(self, x: float, y: float) -> float:
        """f1(x,y) = x² + xy - 10"""
        return x**2 + x*y - 10
    
    def f2(self, x: float, y: float) -> float:
        """f2(x,y) = y + 3xy² - 57"""
        return y + 3*x*y**2 - 57
    
    # Fungsi iterasi g1A dan g2A (NIM mod 4 = 0)
    def g1A(self, x: float, y: float) -> float:
        """g1A: x = √(10 - xy)"""
        val = 10 - x*y
        if val < 0:
            return np.nan
        return np.sqrt(val)
    
    def g2A(self, x: float, y: float) -> float:
        """g2A: y = 57 - 3xy²"""
        return 57 - 3*x*y**2
    
    # Turunan parsial untuk Newton-Raphson
    def df1_dx(self, x: float, y: float) -> float:
        """∂f1/∂x = 2x + y"""
        return 2*x + y
    
    def df1_dy(self, x: float, y: float) -> float:
        """∂f1/∂y = x"""
        return x
    
    def df2_dx(self, x: float, y: float) -> float:
        """∂f2/∂x = 3y²"""
        return 3*y**2
    
    def df2_dy(self, x: float, y: float) -> float:
        """∂f2/∂y = 1 + 6xy"""
        return 1 + 6*x*y
    
    def jacobian(self, x: float, y: float) -> np.ndarray:
        """Matriks Jacobian"""
        return np.array([
            [self.df1_dx(x, y), self.df1_dy(x, y)],
            [self.df2_dx(x, y), self.df2_dy(x, y)]
        ])
    
    def fixed_point_jacobi(self) -> Dict:
        """Metode Iterasi Titik Tetap - Jacobi (g1A, g2A)"""
        x, y = self.x0, self.y0
        iterations = []
        
        for i in range(self.max_iter):
            x_old, y_old = x, y
            
            # Jacobi: hitung semua nilai baru berdasarkan nilai lama
            x_new = self.g1A(x_old, y_old)
            y_new = self.g2A(x_old, y_old)
            
            # Check for NaN
            if np.isnan(x_new) or np.isnan(y_new):
                return {
                    'method': 'Fixed Point - Jacobi (g1A, g2A)',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x_old,
                    'final_y': y_old,
                    'num_iterations': i+1,
                    'reason': 'Nilai NaN terdeteksi (domain tidak valid)'
                }
            
            error_x = abs(x_new - x_old)
            error_y = abs(y_new - y_old)
            error = max(error_x, error_y)
            
            iterations.append({
                'iter': i+1,
                'x': x_new,
                'y': y_new,
                'f1': self.f1(x_new, y_new),
                'f2': self.f2(x_new, y_new),
                'error': error
            })
            
            x, y = x_new, y_new
            
            if error < self.epsilon:
                return {
                    'method': 'Fixed Point - Jacobi (g1A, g2A)',
                    'converged': True,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1
                }
            
            # Deteksi divergensi
            if abs(x) > 1e10 or abs(y) > 1e10:
                return {
                    'method': 'Fixed Point - Jacobi (g1A, g2A)',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1,
                    'reason': 'Divergensi terdeteksi (nilai terlalu besar)'
                }
        
        return {
            'method': 'Fixed Point - Jacobi (g1A, g2A)',
            'converged': False,
            'iterations': iterations,
            'final_x': x,
            'final_y': y,
            'num_iterations': self.max_iter,
            'reason': 'Maksimum iterasi tercapai'
        }
    
    def fixed_point_seidel(self) -> Dict:
        """Metode Iterasi Titik Tetap - Gauss-Seidel (g1A, g2A)"""
        x, y = self.x0, self.y0
        iterations = []
        
        for i in range(self.max_iter):
            x_old, y_old = x, y
            
            # Seidel: gunakan nilai terbaru yang sudah dihitung
            x_new = self.g1A(x_old, y_old)
            
            # Check for NaN in x_new
            if np.isnan(x_new):
                return {
                    'method': 'Fixed Point - Gauss-Seidel (g1A, g2A)',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x_old,
                    'final_y': y_old,
                    'num_iterations': i+1,
                    'reason': 'Nilai NaN terdeteksi pada x (domain tidak valid)'
                }
            
            y_new = self.g2A(x_new, y_old)  # Gunakan x_new (nilai terbaru)
            
            # Check for NaN in y_new
            if np.isnan(y_new):
                return {
                    'method': 'Fixed Point - Gauss-Seidel (g1A, g2A)',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x_new,
                    'final_y': y_old,
                    'num_iterations': i+1,
                    'reason': 'Nilai NaN terdeteksi pada y (domain tidak valid)'
                }
            
            error_x = abs(x_new - x_old)
            error_y = abs(y_new - y_old)
            error = max(error_x, error_y)
            
            iterations.append({
                'iter': i+1,
                'x': x_new,
                'y': y_new,
                'f1': self.f1(x_new, y_new),
                'f2': self.f2(x_new, y_new),
                'error': error
            })
            
            x, y = x_new, y_new
            
            if error < self.epsilon:
                return {
                    'method': 'Fixed Point - Gauss-Seidel (g1A, g2A)',
                    'converged': True,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1
                }
            
            # Deteksi divergensi
            if abs(x) > 1e10 or abs(y) > 1e10:
                return {
                    'method': 'Fixed Point - Gauss-Seidel (g1A, g2A)',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1,
                    'reason': 'Divergensi terdeteksi (nilai terlalu besar)'
                }
        
        return {
            'method': 'Fixed Point - Gauss-Seidel (g1A, g2A)',
            'converged': False,
            'iterations': iterations,
            'final_x': x,
            'final_y': y,
            'num_iterations': self.max_iter,
            'reason': 'Maksimum iterasi tercapai'
        }
    
    def newton_raphson(self) -> Dict:
        """Metode Newton-Raphson"""
        x, y = self.x0, self.y0
        iterations = []
        
        for i in range(self.max_iter):
            f_vec = np.array([self.f1(x, y), self.f2(x, y)])
            J = self.jacobian(x, y)
            
            # Cek determinan
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-10:
                return {
                    'method': 'Newton-Raphson',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1,
                    'reason': 'Jacobian singular (det ≈ 0)'
                }
            
            # Solve: J * delta = -F
            delta = np.linalg.solve(J, -f_vec)
            
            x_new = x + delta[0]
            y_new = y + delta[1]
            
            error = np.linalg.norm(delta)
            
            iterations.append({
                'iter': i+1,
                'x': x_new,
                'y': y_new,
                'f1': self.f1(x_new, y_new),
                'f2': self.f2(x_new, y_new),
                'error': error
            })
            
            x, y = x_new, y_new
            
            if error < self.epsilon:
                return {
                    'method': 'Newton-Raphson',
                    'converged': True,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1
                }
            
            # Deteksi divergensi
            if abs(x) > 1e10 or abs(y) > 1e10 or np.isnan(x) or np.isnan(y):
                return {
                    'method': 'Newton-Raphson',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x,
                    'final_y': y,
                    'num_iterations': i+1,
                    'reason': 'Divergensi terdeteksi'
                }
        
        return {
            'method': 'Newton-Raphson',
            'converged': False,
            'iterations': iterations,
            'final_x': x,
            'final_y': y,
            'num_iterations': self.max_iter,
            'reason': 'Maksimum iterasi tercapai'
        }
    
    def secant(self) -> Dict:
        """Metode Secant (Multi-dimensi)"""
        # Inisialisasi dengan 2 tebakan awal
        x0, y0 = self.x0, self.y0
        x1, y1 = x0 + 0.1, y0 + 0.1  # Perturbasi kecil
        
        iterations = []
        
        for i in range(self.max_iter):
            f0_1 = self.f1(x0, y0)
            f0_2 = self.f2(x0, y0)
            f1_1 = self.f1(x1, y1)
            f1_2 = self.f2(x1, y1)
            
            # Approximate Jacobian
            dx = x1 - x0
            dy = y1 - y0
            
            if abs(dx) < 1e-10 or abs(dy) < 1e-10:
                return {
                    'method': 'Secant',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x1,
                    'final_y': y1,
                    'num_iterations': i+1,
                    'reason': 'Denominator terlalu kecil'
                }
            
            # Pendekatan Jacobian dengan beda hingga
            J_approx = np.array([
                [(f1_1 - f0_1)/dx, (f1_1 - f0_1)/dy],
                [(f1_2 - f0_2)/dx, (f1_2 - f0_2)/dy]
            ])
            
            det_J = np.linalg.det(J_approx)
            if abs(det_J) < 1e-10:
                return {
                    'method': 'Secant',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x1,
                    'final_y': y1,
                    'num_iterations': i+1,
                    'reason': 'Approximate Jacobian singular'
                }
            
            f_vec = np.array([f1_1, f1_2])
            delta = np.linalg.solve(J_approx, -f_vec)
            
            x2 = x1 + delta[0]
            y2 = y1 + delta[1]
            
            error = np.linalg.norm(delta)
            
            iterations.append({
                'iter': i+1,
                'x': x2,
                'y': y2,
                'f1': self.f1(x2, y2),
                'f2': self.f2(x2, y2),
                'error': error
            })
            
            if error < self.epsilon:
                return {
                    'method': 'Secant',
                    'converged': True,
                    'iterations': iterations,
                    'final_x': x2,
                    'final_y': y2,
                    'num_iterations': i+1
                }
            
            # Deteksi divergensi
            if abs(x2) > 1e10 or abs(y2) > 1e10 or np.isnan(x2) or np.isnan(y2):
                return {
                    'method': 'Secant',
                    'converged': False,
                    'iterations': iterations,
                    'final_x': x2,
                    'final_y': y2,
                    'num_iterations': i+1,
                    'reason': 'Divergensi terdeteksi'
                }
            
            # Update untuk iterasi berikutnya
            x0, y0 = x1, y1
            x1, y1 = x2, y2
        
        return {
            'method': 'Secant',
            'converged': False,
            'iterations': iterations,
            'final_x': x1,
            'final_y': y1,
            'num_iterations': self.max_iter,
            'reason': 'Maksimum iterasi tercapai'
        }
    
    def print_results(self, result: Dict, max_display: int = 10):
        """Cetak hasil dengan format yang rapi"""
        print(f"\n{'='*80}")
        print(f"METODE: {result['method']}")
        print(f"{'='*80}")
        print(f"Tebakan Awal: x0 = {self.x0}, y0 = {self.y0}")
        print(f"Toleransi: ε = {self.epsilon}")
        
        df = pd.DataFrame(result['iterations'])
        
        # Tampilkan beberapa iterasi pertama dan terakhir
        if len(df) > max_display:
            print(f"\n--- {max_display//2} Iterasi Pertama ---")
            print(df.head(max_display//2).to_string(index=False))
            print(f"\n... ({len(df) - max_display} iterasi lainnya) ...\n")
            print(f"--- {max_display//2} Iterasi Terakhir ---")
            print(df.tail(max_display//2).to_string(index=False))
        else:
            print("\n--- Semua Iterasi ---")
            print(df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print(f"STATUS: {'✓ KONVERGEN' if result['converged'] else '✗ TIDAK KONVERGEN'}")
        
        if result['converged']:
            print(f"Jumlah Iterasi: {result['num_iterations']}")
            print(f"Solusi: x = {result['final_x']:.10f}, y = {result['final_y']:.10f}")
            print(f"f1(x,y) = {self.f1(result['final_x'], result['final_y']):.2e}")
            print(f"f2(x,y) = {self.f2(result['final_x'], result['final_y']):.2e}")
        else:
            print(f"Alasan: {result.get('reason', 'Unknown')}")
            print(f"Iterasi terakhir: {result['num_iterations']}")
            if not np.isnan(result['final_x']) and not np.isnan(result['final_y']):
                print(f"Nilai terakhir: x = {result['final_x']:.6f}, y = {result['final_y']:.6f}")
        
        print(f"{'='*80}\n")
        
        return df

def main():
    print("="*80)
    print("PENYELESAIAN SISTEM PERSAMAAN NON-LINEAR")
    print("="*80)
    print("Sistem Persamaan:")
    print("  f₁(x,y) = x² + xy - 10 = 0")
    print("  f₂(x,y) = y + 3xy² - 57 = 0")
    print("\nParameter:")
    print("  NIM: 21120123140112")
    print("  NIM mod 4 = 0")
    print("  Fungsi iterasi: g1A dan g2A")
    print("    g1A(x,y) = √(10 - xy)")
    print("    g2A(x,y) = 57 - 3xy²")
    print("  Tebakan awal: x0 = 1.5, y0 = 3.5")
    print("  Epsilon: 0.000001")
    print("="*80)
    
    # Inisialisasi solver
    solver = NonLinearSystemSolver(x0=1.5, y0=3.5, epsilon=0.000001)
    
    # Jalankan semua metode
    results = []
    
    # 1. Fixed Point - Jacobi
    print("\n[1/4] Menjalankan metode Fixed Point - Jacobi...")
    result_jacobi = solver.fixed_point_jacobi()
    df_jacobi = solver.print_results(result_jacobi)
    results.append(result_jacobi)
    
    # 2. Fixed Point - Gauss-Seidel
    print("\n[2/4] Menjalankan metode Fixed Point - Gauss-Seidel...")
    result_seidel = solver.fixed_point_seidel()
    df_seidel = solver.print_results(result_seidel)
    results.append(result_seidel)
    
    # 3. Newton-Raphson
    print("\n[3/4] Menjalankan metode Newton-Raphson...")
    result_newton = solver.newton_raphson()
    df_newton = solver.print_results(result_newton)
    results.append(result_newton)
    
    # 4. Secant
    print("\n[4/4] Menjalankan metode Secant...")
    result_secant = solver.secant()
    df_secant = solver.print_results(result_secant)
    results.append(result_secant)
    
    # Ringkasan Perbandingan
    print("\n" + "="*80)
    print("RINGKASAN PERBANDINGAN METODE")
    print("="*80)
    
    comparison = []
    for r in results:
        comparison.append({
            'Metode': r['method'],
            'Konvergen': '✓ Ya' if r['converged'] else '✗ Tidak',
            'Iterasi': r['num_iterations'] if r['converged'] else f"{r['num_iterations']} (max)",
            'x': f"{r['final_x']:.6f}" if r['converged'] else 'N/A',
            'y': f"{r['final_y']:.6f}" if r['converged'] else 'N/A'
        })
    
    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))
    
    # Analisis Konvergensi
    print("\n" + "="*80)
    print("ANALISIS KONVERGENSI")
    print("="*80)
    
    print("\n1. METODE ITERASI TITIK TETAP (g1A, g2A):")
    print("   - g1A: x = √(10 - xy)")
    print("   - g2A: y = 57 - 3xy²")
    
    if result_jacobi['converged']:
        print(f"\n   • Jacobi: KONVERGEN dalam {result_jacobi['num_iterations']} iterasi")
        print("     Kecepatan: Lambat (konvergensi linear)")
    else:
        print(f"\n   • Jacobi: TIDAK KONVERGEN - {result_jacobi.get('reason', 'Unknown')}")
    
    if result_seidel['converged']:
        print(f"\n   • Gauss-Seidel: KONVERGEN dalam {result_seidel['num_iterations']} iterasi")
        print("     Kecepatan: Lebih cepat dari Jacobi (menggunakan nilai terbaru)")
    else:
        print(f"\n   • Gauss-Seidel: TIDAK KONVERGEN - {result_seidel.get('reason', 'Unknown')}")
    
    print("\n2. METODE NEWTON-RAPHSON:")
    if result_newton['converged']:
        print(f"   • KONVERGEN dalam {result_newton['num_iterations']} iterasi")
        print("     Kecepatan: SANGAT CEPAT (konvergensi kuadratik)")
        print("     Keunggulan: Konvergensi paling cepat jika tebakan awal dekat solusi")
    else:
        print(f"   • TIDAK KONVERGEN - {result_newton.get('reason', 'Unknown')}")
    
    print("\n3. METODE SECANT:")
    if result_secant['converged']:
        print(f"   • KONVERGEN dalam {result_secant['num_iterations']} iterasi")
        print("     Kecepatan: CEPAT (konvergensi superlinear)")
        print("     Keunggulan: Tidak perlu turunan analitik")
    else:
        print(f"   • TIDAK KONVERGEN - {result_secant.get('reason', 'Unknown')}")
    
    print("\n" + "="*80)
    print("KESIMPULAN")
    print("="*80)
    
    converged_methods = [r for r in results if r['converged']]
    if converged_methods:
        fastest = min(converged_methods, key=lambda x: x['num_iterations'])
        print(f"\nMetode tercepat: {fastest['method']}")
        print(f"Jumlah iterasi: {fastest['num_iterations']}")
        print(f"Solusi: x = {fastest['final_x']:.10f}, y = {fastest['final_y']:.10f}")
        
        print("\nUrutan kecepatan konvergensi (teori):")
        print("1. Newton-Raphson (kuadratik) - tercepat")
        print("2. Secant (superlinear)")
        print("3. Gauss-Seidel (linear)")
        print("4. Jacobi (linear) - terlambat")
    else:
        print("\nTIDAK ADA metode yang konvergen dengan parameter yang diberikan.")
        print("Saran: Coba tebakan awal yang berbeda atau fungsi iterasi yang lain.")
    
    print("="*80)

if __name__ == "__main__":
    main()