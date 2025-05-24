# Main.py
import numpy as np
import matplotlib.pyplot as plt

from Modulo_Puntos import generar_puntos_base, obtener_valores, target_function
from Matriz import construir_matriz, multiquadric_rbf, inverse_multiquadric_rbf
from Error_Graficas import calcular_error_local, graficar_errores_distribucion, plot_surface, plot_error_surface


def run_interpolation_scenario(rbf_func, rbf_name, c_param):
    print(f"\n{'='*80}")
    print(f"🔵 Iniciando Interpolación con RBF: {rbf_name}  |  Parámetro c = {c_param}")
    print(f"{'='*80}")

    # 1. Generar puntos de entrenamiento
    puntos_base = generar_puntos_base()
    valores_reales = obtener_valores(puntos_base)

    # Filtrar puntos con NaN
    indices_validos = ~np.isnan(valores_reales)
    puntos_validos = puntos_base[indices_validos]
    valores_validos = valores_reales[indices_validos]

    n_puntos = len(puntos_validos)
    if n_puntos == 0:
        print("❌ ERROR: Todos los puntos presentan discontinuidades (x=0 o y=0). Interpolación cancelada.")
        return
    elif n_puntos < len(puntos_base):
        print(f"⚠️  Advertencia: Algunos puntos fueron descartados por NaNs. Matriz resultante: {n_puntos}x{n_puntos}")

    print(f"✔️  Puntos válidos utilizados: {n_puntos}")
    print(puntos_validos)

    # 2. Construcción de matriz A
    print("\n🔧 Construyendo matriz A de interpolación...")
    A = construir_matriz(puntos_validos, rbf_func, c_param)
    print("✔️  Matriz A construida:")
    print(A)

    # 3. Resolver sistema lineal A * lambda = b
    print("\n🔍 Resolviendo sistema A * lambda = b...")
    try:
        lambdas = np.linalg.solve(A, valores_validos)
        print("✔️  Coeficientes lambda obtenidos.")
    except np.linalg.LinAlgError as e:
        print(f"❌ ERROR al resolver el sistema lineal: {e}")
        print("🛠️  Sugerencias:")
        print("   - Verifica que los puntos no sean colineales.")
        print("   - Ajusta el parámetro c.")
        return

    # 4. Evaluación de la superficie
    x_min, x_max = np.min(puntos_validos[:, 0]) - 0.5, np.max(puntos_validos[:, 0]) + 0.5
    y_min, y_max = np.min(puntos_validos[:, 1]) - 0.5, np.max(puntos_validos[:, 1]) + 0.5
    x_min = max(0.01, x_min)
    y_min = max(0.01, y_min)

    x_eval = np.linspace(x_min, x_max, 100)
    y_eval = np.linspace(y_min, y_max, 100)
    X_grid, Y_grid = np.meshgrid(x_eval, y_eval)
    puntos_eval = np.c_[X_grid.ravel(), Y_grid.ravel()]

    Z_interp_flat = np.array([
        sum(lambdas[j] * rbf_func(np.linalg.norm(p_eval - puntos_validos[j]), c_param)
            for j in range(n_puntos))
        for p_eval in puntos_eval
    ])
    Z_interp = Z_interp_flat.reshape(X_grid.shape)

    Z_real = np.array([[target_function(x, y) for y in y_eval] for x in x_eval]).T

    # 5. Gráficas
    plot_surface(X_grid, Y_grid, Z_real, f'Superficie Real f(x,y) - {rbf_name}')
    plot_surface(X_grid, Y_grid, Z_interp, f'Superficie Interpolada RBF - {rbf_name}')

    # 6. Cálculo de errores
    Z_real_flat = Z_real.ravel()
    Z_interp_flat = Z_interp.ravel()
    indices_validos_eval = ~np.isnan(Z_real_flat)

    if np.any(indices_validos_eval):
        errores = calcular_error_local(Z_real_flat[indices_validos_eval], Z_interp_flat[indices_validos_eval])
        print(f"\n📊 Métricas de error:")
        print(f"   • Error absoluto medio  : {np.mean(errores):.4f}")
        print(f"   • Error absoluto máximo : {np.max(errores):.4f}")

        graficar_errores_distribucion(errores, f'Errores Absolutos - {rbf_name}')
        plot_error_surface(X_grid, Y_grid, Z_real, Z_interp, f'Superficie de Error Absoluto - {rbf_name}')
    else:
        print("⚠️  No se encontraron puntos válidos para calcular errores.")


if __name__ == "__main__":
    print("="*80)
    print("Interpolación de Superficies con Funciones de Base Radial (RBF)")
    print("="*80)
    print("Este programa aproxima una función f(x,y) mediante interpolación RBF.")
    print("Tipos evaluados: Multicuadrática y Multicuadrática Inversa.")
    print(" Evita x=0 o y=0 por discontinuidades de la función objetivo.")
    print("="*80)

    c_param = 1

    run_interpolation_scenario(multiquadric_rbf, "Multicuadrática", c_param)

    print("\n" + "="*80 + "\n")

    run_interpolation_scenario(inverse_multiquadric_rbf, "Multicuadrática Inversa", c_param)

    print("\n✅ Interpolación finalizada.")
