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
    # Define directamente los puntos como una lista de tuplas o listas [x, y]
    # Aquí puedes poner exactamente los 9 puntos que necesites
    # Por ejemplo, una selección de los que mostraste, o cualquier 9 puntos que quieras
    puntos = [
        [-3. , -1. ],
        [-3. ,  0.75],
        [-3. ,  2.5 ],
        [ 0.01, -1. ],
        [ 0.01,  0.75],
        [ 0.01,  2.5 ],
        [ 3.  , -1. ],
        [ 3.  ,  0.75],
        [ 3.  ,  2.5 ]
        # Añade o quita puntos hasta tener la cantidad deseada (por ejemplo, 9)
    ]
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
