import numpy as np

def multiquadric_rbf(r, c):
    """Función de Base Radial Multicuadrática: phi(r) = sqrt(r^2 + c^2)"""
    return np.sqrt(r**2 + c**2)

def inverse_multiquadric_rbf(r, c):
    """Función de Base Radial Multicuadrática Inversa: phi(r) = 1 / sqrt(r^2 + c^2)"""
    return 1.0 / np.sqrt(r**2 + c**2)

def construir_matriz(puntos, rbf_func, c_param):
    """
    Construye la matriz de interpolación A para RBF.
    A_ij = phi(||x_i - x_j||), donde ||x_i - x_j|| es la distancia euclidiana.

    Args:
        puntos (np.ndarray): Array (N, 2) de puntos de entrenamiento.
        rbf_func (function): La función RBF a utilizar (multiquadric_rbf o inverse_multiquadric_rbf).
        c_param (float): El parámetro de forma 'c' para la RBF.

    Returns:
        np.ndarray: La matriz RBF (N, N).
    """
    n = len(puntos)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(puntos[i] - puntos[j]) # Distancia euclidiana
            A[i, j] = rbf_func(r, c_param)
    return A
