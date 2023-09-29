import json
import numpy as np
import pygame
import random


class RedNeuronal:

    def __init__(self, tam_entrada, tam_oculto, tam_salida):
        self.tam_entrada, self.tam_oculto1, self.tam_oculto2, self.tam_oculto3,  self.tam_salida = tam_entrada, tam_oculto, tam_oculto, tam_oculto, tam_salida
        self.pesos_entrada_oculto1 = np.random.randn(self.tam_entrada, self.tam_oculto1)
        self.pesos_oculto1_oculto2 = np.random.randn(self.tam_oculto1, self.tam_oculto2)
        self.pesos_oculto2_oculto3 = np.random.randn(self.tam_oculto2, self.tam_oculto3)
        self.pesos_oculto3_salida = np.random.randn(self.tam_oculto3, self.tam_salida)

    def forward(self, x):
        oculto1 = np.dot(x, self.pesos_entrada_oculto1)
        oculto2 = np.dot(oculto1, self.pesos_oculto1_oculto2)
        oculto3 = np.dot(oculto2, self.pesos_oculto2_oculto3)
        salida = np.dot(oculto2, self.pesos_oculto3_salida)
        return salida

    def relu(self, x):
        # Función de activación ReLU [0, 1]
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Función de activación Sigmoide [0 a 1)
        return 1 / (1 + np.exp(-x))

    def guardar_pesos(self):
        pesos = {
            'pesos_e_o': self.pesos_entrada_oculto1.tolist(),
            'pesos_o1_o2': self.pesos_oculto1_oculto2.tolist(),
            'pesos_o2_o3': self.pesos_oculto2_oculto3.tolist(),
            'pesos_o_s': self.pesos_oculto3_salida.tolist()
        }

        with open('pesos1.json', 'w') as archivo:
            json.dump(pesos, archivo)

    def mutar(self, tasa_mutacion):
        for pesos in [self.pesos_entrada_oculto1, self.pesos_oculto1_oculto2, self.pesos_oculto2_oculto3,self.pesos_oculto3_salida]:
            pesos += np.random.randn(*pesos.shape) * tasa_mutacion

pygame.init()
ancho = 525
alto = 525
tam_celda = 35
tam_cuadricula_x = ancho // tam_celda
tam_cuadricula_y = alto // tam_celda
direcciones = [(0, -1), (0, 1), (-1, 0), (1, 0)]
umbral_puntuacion = 39


def generar_nueva_serpiente():
    # Genera una nueva serpiente con una posición y dirección aleatorias.
    serpiente = [(tam_cuadricula_x // 2, tam_cuadricula_y // 2)]
    direccion = random.choice(direcciones)
    return serpiente, direccion

def generar_comida(serpiente):
    # Genera comida en una posición aleatoria que no esté ocupada por la serpiente
    while True:
        comida = (random.randint(0, tam_cuadricula_x - 1), random.randint(0, tam_cuadricula_y - 1))
        if comida not in serpiente:
            return comida
def check_collision(segments):
    for segment in segments[1:]:
        if segments[0] == segment:
            return True
    return False

def jugar_juego(red):
    # Juega una partida con las órdenes de la red como entrada
    serpiente, direccion = generar_nueva_serpiente()
    comida = generar_comida(serpiente)
    puntuacion = 0
    pasos = 0
    movimientos_restantes = 66

    while movimientos_restantes > 0:
        # Genera las características de entrada para la red neuronal
        cabeza_x, cabeza_y = serpiente[0]
        comida_x, comida_y = comida
        entradas = np.array([
            cabeza_x < comida_x,  # ¿Está la comida a la izquierda?
            cabeza_x > comida_x,  # ¿Está la comida a la derecha?
            cabeza_y < comida_y,  # ¿Está la comida arriba?
            cabeza_y > comida_y,  # ¿Está la comida abajo?
            direccion[0] == -1,  # ¿Se mueve a la izquierda?
            direccion[0] == 1,  # ¿Se mueve a la derecha?
            direccion[1] == -1,  # ¿Se mueve hacia arriba?
            direccion[1] == 1,  # ¿Se mueve hacia abajo?
        ])

        salidas = red.forward(entradas)
        movimiento = np.argmax(salidas)
        direccion = direcciones[movimiento]
        serpiente.insert(0, (serpiente[0][0] + direccion[0], serpiente[0][1] + direccion[1]))
        if check_collision(serpiente):
            return puntuacion, pasos
        if (
            serpiente[0][0] < 0
            or serpiente[0][0] >= tam_cuadricula_x
            or serpiente[0][1] < 0
            or serpiente[0][1] >= tam_cuadricula_y
            or len(serpiente) != len(set(serpiente))
        ):
            return puntuacion, pasos

        if serpiente[0] == comida:
            puntuacion += 1
            pasos = 0
            comida = generar_comida(serpiente)
            movimientos_restantes += 85  # Aumenta los movimientos restantes
        else:
            serpiente.pop()

        # Actualiza el contador de pasos
        pasos += 1
        movimientos_restantes -= 1
    return puntuacion, pasos


# Define los parámetros del algoritmo genético
tam_poblacion = 1200
tasa_mutacion = 0.53
generaciones = 999999999999999999999999999

# Crea la población inicial de redes neuronales
poblacion = [RedNeuronal(8, 9, 4) for _ in range(tam_poblacion)]

mejor_red = None
mejor_puntuacion = 0

for generacion in range(generaciones):
    print("Generación:", generacion + 1)

    # Evalúa el rendimiento de cada red y selecciona las mejores
    puntuaciones = []
    for red in poblacion:
        puntuacion, _ = jugar_juego(red)
        puntuaciones.append(puntuacion)

    indices_top = np.argsort(puntuaciones)[::-1][:tam_poblacion // 70]#mayor numero, menores muestras para la siguiente generacion
    poblacion_top = [poblacion[i] for i in indices_top]

    # Guarda la mejor red de la generación actual
    mejor_puntuacion_actual = max(puntuaciones)
    if mejor_puntuacion_actual > mejor_puntuacion:
        mejor_red = poblacion_top[0]
        mejor_puntuacion = mejor_puntuacion_actual

    # Imprime la puntuación y el mensaje correspondiente
    if mejor_puntuacion_actual == 1:
        print("Score:", mejor_puntuacion_actual, "punto")
    else:
        print("Score:", mejor_puntuacion_actual, "puntos")

    # Comprueba si la puntuación es mayor que 21 para ofrecer guardar la configuración de la red
    if mejor_puntuacion_actual > umbral_puntuacion:
        opcion_guardar = input("¿Deseas guardar la configuración de la red neuronal? (s/n): ")
        if opcion_guardar.lower() == "s":
            print("Guardando la configuración de la red neuronal...")
            red.guardar_pesos()
        else:
            print("Pasando a la siguiente generación")
    else:
        print("Score menor que " + str(umbral_puntuacion) + ". Continuando...")

    # Muta y cruza las mejores redes para generar la siguiente generación
    poblacion_nueva = []
    for _ in range(tam_poblacion):
        padre1, padre2 = random.choices(poblacion_top, k=2)
        hijo = RedNeuronal(8, 9, 4)
        hijo.pesos_entrada_oculto1 = padre1.pesos_entrada_oculto1.copy()
        hijo.pesos_oculto1_oculto2 = padre2.pesos_oculto1_oculto2.copy()
        hijo.pesos_oculto2_oculto3 = padre2.pesos_oculto2_oculto3.copy()
        hijo.pesos_oculto3_salida = padre1.pesos_oculto3_salida.copy()
        hijo.mutar(tasa_mutacion)
        poblacion_nueva.append(hijo)

    # Reemplaza la población anterior con la nueva generación
    poblacion = poblacion_nueva

print("La mejor red neuronal obtuvo una puntuación de:", mejor_puntuacion)
jugar_juego(mejor_red)