# Modulo_Puntos.py
import numpy as np

def target_function(x, y):
    """
    Define la función objetivo f(x,y) = xy + 1/(2x) + 1/(2y).
    Devuelve NaN (Not a Number) en puntos donde x=0 o y=0 debido a la discontinuidad.
    """
    if np.isclose(x, 0) or np.isclose(y, 0):
        return np.nan
    return x * y + 1.0 / (2.0 * x) + 1.0 / (2.0 * y)

def generar_puntos_base():
    """
    Asegúrate de que ninguno de tus valores sea exactamente 0.0 si quieres
    que  todos los puntos sean válidos.
    Los puntos que caigan en una discontinuidad (x=0 o y=0) serán filtrados
    y la matriz de interpolación será de un tamaño menor al esperado.

    Returns:
        np.ndarray: Un array (9, 2) con los 9 puntos (x, y) de la cuadrícula de entrenamiento.
                    Si hay NaNs, los puntos con NaN serán filtrados más adelante.
    """
    # --- VALORES PARA X ---
    # Usaremos valores que NO son cero para garantizar 9 puntos válidos.
    #x_valores_definidos = [-3, 1, 3] # <<--- VALORES DE EJEMPLO NO-CERO
    x_valores_definidos = [-3, -1.5, 0.01, 1.5, 3] # <--- MODIFICADO AQUÍ

    # ---  VALORES PARA Y --
    #y_valores_definidos = [-1, 2.5, 6] 
    y_valores_definidos = [-1, 0.75, 2.5, 4.25, 6] # <--- MODIFICADO AQUÍ

    
    puntos = []
    # Se crea la cuadrícula combinando cada valor de X con cada valor de Y
    # (P1=[x1,y1], P2=[x1,y2], P3=[x1,y3], P4=[x2,y1], etc.)
    for x_val in x_valores_definidos: # Itera primero sobre los valores de X
        for y_val in y_valores_definidos: # Luego itera sobre los valores de Y
            puntos.append([x_val, y_val])
            
    return np.array(puntos)

def obtener_valores(puntos):
    """
    Evalúa la función objetivo en cada uno de los puntos dados.

    Args:
        puntos (np.ndarray): Array de puntos (x, y).

    Returns:
        np.ndarray: Array de valores f(x,y), incluyendo NaNs para las discontinuidades.
    """
    valores = []
    for punto in puntos:
        valores.append(target_function(punto[0], punto[1]))
    return np.array(valores)
