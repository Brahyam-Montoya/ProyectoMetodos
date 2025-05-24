import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calcular_error_local(valores_reales, valores_aproximados):

    return np.abs(valores_reales - valores_aproximados)

def graficar_errores_distribucion(errores, title="Distribución de Errores Absolutos"):
 
    # Histograma con la distribución de errores absolutos.

    plt.figure(figsize=(8, 6))
    plt.hist(errores, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Error Absoluto')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

def plot_surface(X_grid, Y_grid, Z_grid, title="Superficie"):
    """
    Superficie 3D de valores f(x, y).
    Args:
        X_grid (np.ndarray): Coordenadas X (meshgrid).
        Y_grid (np.ndarray): Coordenadas Y (meshgrid).
        Z_grid (np.ndarray): Valores Z (resultado de la función o interpolación).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Oculta valores indefinidos (NaN) en la superficie
    Z_grid_masked = np.ma.masked_invalid(Z_grid)

    surf = ax.plot_surface(X_grid, Y_grid, Z_grid_masked, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Valor de Z')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_error_surface(X_grid, Y_grid, Z_actual, Z_interpolated, title="Superficie de Error Absoluto"):
    
    #Superficie 3D del error absoluto entre valores reales e interpolados.
 
    errors_grid = np.abs(Z_actual - Z_interpolated)
    errors_grid_masked = np.ma.masked_invalid(errors_grid)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, errors_grid_masked, cmap='plasma', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Error Absoluto')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Error Absoluto')
    plt.show()
