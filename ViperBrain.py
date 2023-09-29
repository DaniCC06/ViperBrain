import numpy as np
import pygame
import random

class RedNeuronal:

    def __init__(self, tam_entrada, tam_oculto, tam_salida):
        self.tam_entrada = tam_entrada
        self.tam_oculto1 = tam_oculto
        self.tam_oculto2 = tam_oculto
        self.tam_salida = tam_salida
        self.pesos_entrada_oculto1 = np.random.randn(self.tam_entrada, self.tam_oculto1)
        self.pesos_oculto1_oculto2 = np.random.randn(self.tam_oculto1, self.tam_oculto2)
        self.pesos_oculto2_salida = np.random.randn(self.tam_oculto2, self.tam_salida)

    def forward(self, x):
        oculto1 = np.dot(x, self.pesos_entrada_oculto1)
        oculto1 = self.relu(oculto1)
        oculto2 = np.dot(oculto1, self.pesos_oculto1_oculto2)
        oculto2 = self.relu(oculto2)
        salida = np.dot(oculto2, self.pesos_oculto2_salida)
        salida = self.sigmoid(salida)
        return salida

    def relu(self, x):
        # Función de activación ReLU [0, 1]
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Función de activación Sigmoide [0 a 1)
        return 1 / (1 + np.exp(-x))

    def mutar(self, tasa_mutacion):
        # Aplica mutaciones aleatorias a los pesos de la red neuronal
        self.pesos_entrada_oculto1 += np.random.randn(*self.pesos_entrada_oculto1.shape) * tasa_mutacion
        self.pesos_oculto1_oculto2 += np.random.randn(*self.pesos_oculto1_oculto2.shape) * tasa_mutacion
        self.pesos_oculto2_salida += np.random.randn(*self.pesos_oculto2_salida.shape) * tasa_mutacion

#Pantalla:
ancho = 525
alto = 525
tam_celda = 35
tam_cuadricula_x = ancho // tam_celda
tam_cuadricula_y = alto // tam_celda
verde_claro = (0, 255, 0)
verde_oscuro = (0, 100, 0)
ojo_radio = 10
pupila_radio = 6
ojo_offset = 6
pantalla = pygame.display.set_mode((ancho, alto))
pygame.display.set_caption("ViperBrain")
ver_pantalla = True

#RedNeuonal
tam_poblacion = 800  # Número de individuos por generación
tasa_mutacion = 0.33
generaciones = 988888888888888888888888888
umbral_puntuacion = 28
poblacion = [RedNeuronal(8, 16, 4) for _ in range(tam_poblacion)]
mejor_red = None

#Configs del juego
mejor_puntuacion = 0
reloj = pygame.time.Clock()
direcciones = [(0, -1), (-1, 0), (0, 1), (1, 0)]
pygame.init()

def dibujar_cuadricula():
    # Dibuja una cuadrícula en la pantalla
    for x in range(0, ancho, tam_celda):
        pygame.draw.line(pantalla, (0, 0, 0), (x, 0), (x, alto))
    for y in range(0, alto, tam_celda):
        pygame.draw.line(pantalla, (0, 0, 0), (0, y), (ancho, y))

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

        # Realiza un pase hacia adelante a través de la red
        salidas = red.forward(entradas)

        # Elige la acción con el valor de salida más alto
        movimiento = np.argmax(salidas)

        direccion = direcciones[movimiento]

        # Actualiza la posición de la serpiente
        serpiente.insert(0, (serpiente[0][0] + direccion[0], serpiente[0][1] + direccion[1]))

        # Verifica si la serpiente ha chocado con la pared o consigo misma
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
            movimientos_restantes += 150  # Aumenta los movimientos restantes
        else:
            serpiente.pop()

        # Actualiza el contador de pasos
        pasos += 1
        movimientos_restantes -= 1
        if ver_pantalla == True:
            #Actualiza la pantalla
            pantalla.fill((255, 255, 255))  # Establece el fondo en blanco
            dibujar_cuadricula()  # Dibuja la cuadrícula
            for i, segmento in enumerate(serpiente):
                longitud = len(list(enumerate(serpiente)))
                color = tuple(
                    verde_oscuro[j] + (verde_claro[j] - verde_oscuro[j]) * i / len(serpiente) for j in range(3))
                segmento_index = len(serpiente) - i - 1
                segmento = serpiente[segmento_index]

                pygame.draw.rect(pantalla, color,
                                 (segmento[0] * tam_celda, segmento[1] * tam_celda, tam_celda, tam_celda))
                pygame.draw.rect(pantalla, (255, 0, 0),
                                 (comida[0] * tam_celda, comida[1] * tam_celda, tam_celda, tam_celda))
                if i == longitud - 1:  # Draw eyes on the head segment
                    ojo_izq = (segmento[0] * tam_celda + ojo_offset, segmento[1] * tam_celda + ojo_offset)
                    ojo_der = (segmento[0] * tam_celda + tam_celda - ojo_offset, segmento[1] * tam_celda + ojo_offset)
                    pygame.draw.circle(pantalla, (255, 255, 255), ojo_izq, ojo_radio)  # Esclerótica
                    pygame.draw.circle(pantalla, (0, 0, 0), ojo_izq, pupila_radio)  # Pupila
                    pygame.draw.circle(pantalla, (255, 255, 255), ojo_der, ojo_radio)  # Esclerótica
                    pygame.draw.circle(pantalla, (0, 0, 0), ojo_der, pupila_radio)  # Pupila
            pygame.draw.rect(pantalla, (255, 0, 0), (comida[0] * tam_celda, comida[1] * tam_celda, tam_celda, tam_celda))

            pygame.display.flip()

            #Limita la velocidad del juego
            reloj.tick(25)  # Velocidad maxima ->comentar reloj, Velocidad normal ->reloj.tick(25)

    # Si la serpiente alcanza el número máximo de movimientos, devuelve la puntuación y los pasos
    return puntuacion, pasos

for generacion in range(generaciones):
    puntuaciones = []
    for red in poblacion:
        puntuacion, _ = jugar_juego(red)
        puntuaciones.append(puntuacion)
        # Evalúa el rendimiento de cada red y selecciona las mejores
    indices_top = np.argsort(puntuaciones)[::-1][:tam_poblacion // 25]#mayor numero, menores muestras para la siguiente generacion
    poblacion_top = [poblacion[i] for i in indices_top] # Guarda la mejor red de la generación actual
    mejor_puntuacion_actual = max(puntuaciones)

    if mejor_puntuacion_actual > mejor_puntuacion:
        mejor_red = poblacion_top[0]
        mejor_puntuacion = mejor_puntuacion_actual  # Imprime la puntuación y el mensaje correspondiente
    print(mejor_puntuacion_actual, "puntos obtenidos en la generación nº", generacion + 1)
    # Muta y cruza las mejores redes para generar la siguiente generación
    poblacion_nueva = []
    for _ in range(tam_poblacion):
        padre1, padre2 = random.choices(poblacion_top, k=2)
        hijo = RedNeuronal(8, 16, 4)
        hijo.pesos_entrada_oculto1 = padre1.pesos_entrada_oculto1.copy()
        hijo.pesos_oculto1_oculto2 = padre2.pesos_oculto1_oculto2.copy()
        hijo.pesos_oculto2_salida = padre1.pesos_oculto2_salida.copy()
        hijo.mutar(tasa_mutacion)
        poblacion_nueva.append(hijo)
    poblacion = poblacion_nueva # Reemplaza la población anterior con la nueva generación
