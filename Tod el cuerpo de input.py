import json
import numpy as np
import pygame
import random

class RedNeuronal:
    def __init__(self, tam_entrada, tam_oculto1, tam_oculto2, tam_oculto3, tam_oculto4,tam_salida):
        self.tam_entrada = tam_entrada
        self.tam_oculto1 = tam_oculto1
        self.tam_oculto2 = tam_oculto2
        self.tam_oculto3 = tam_oculto3
        self.tam_oculto4 = tam_oculto4
        self.tam_salida = tam_salida
        self.pesos_entrada_oculto1 = np.random.randn(self.tam_entrada, self.tam_oculto1)
        self.pesos_oculto1_oculto2 = np.random.randn(self.tam_oculto1, self.tam_oculto2)
        self.pesos_oculto4_salida = np.random.randn(self.tam_oculto4, self.tam_salida)
        self.pesos_oculto2_salida = np.random.randn(self.tam_oculto2, self.tam_salida)
        self.pesos_oculto2_oculto3 = np.random.randn(self.tam_oculto2, self.tam_oculto3)
        self.pesos_oculto3_oculto4 = np.random.randn(self.tam_oculto3, self.tam_oculto4)



    def __str__(self):
        return "{\ntam_entrada: " + str(self.tam_entrada) + "\ntam_oculto: " + str(self.tam_oculto1) + "\ntam_salida: " + str(self.tam_salida) + "\npesos:[" + str(self.pesos_entrada_oculto1) + "]\npesos_oculto_salida: " + str(self.pesos_oculto2_salida) + "\n}"

    def forward(self, x):
        oculto1 = np.dot(x, self.pesos_entrada_oculto1)
        oculto1 = self.relu(oculto1)
        oculto2 = np.dot(oculto1, self.pesos_oculto1_oculto2)
        oculto2 = self.relu(oculto2)
        oculto3 = np.dot(oculto2, self.pesos_oculto2_oculto3)
        oculto3 = self.relu(oculto3)
        oculto4 = np.dot(oculto3, self.pesos_oculto3_oculto4)
        oculto4 = self.relu(oculto4)
        salida = np.dot(oculto4, self.pesos_oculto4_salida)
        salida = self.sigmoid(salida)
        return salida

    def relu(self, x):
        # Función de activación ReLU [0, 1]
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Función de activación Sigmoide [0 a 1)
        return 1 / (1 + np.exp(-x))

    #def guardar_mejor_red(self):
    #    # Guarda la configuración de la mejor red neuronal en un archivo
    #    nombre_archivo = "mejor_red.txt"
    #    with open(nombre_archivo, "a") as archivo:
    #        archivo.write("Tamaño de entrada: " + str(self.tam_entrada) + "\n")
    #        archivo.write("Tamaño oculto: " + str(self.tam_oculto1) + "\n")
    #        archivo.write("Tamaño de salida: " + str(self.tam_salida) + "\n")
    #        archivo.write("Pesos entrada-oculto:\n")
    #        archivo.write(str(self.pesos_entrada_oculto1) + "\n")
    #        archivo.write("Pesos oculto-salida:\n")
    #        archivo.write(str(self.pesos_oculto2_salida) + "\n")
    #    print("La configuración de la mejor red neuronal ha sido guardada en", nombre_archivo)

    def guardar_mejor_red2(self):
        # Guarda la configuración de la mejor red neuronal en un archivo
        archivo = "mejor_red_1.txt"
        datos = {
            "entrada": str(self.tam_entrada),
            "oculta1": str(self.tam_oculto1),
            "oculta2": str(self.tam_oculto2),
            "salida": str(self.tam_salida),
            "pesos_e_o": str(self.pesos_entrada_oculto1),
            "pesos_o_s": str(self.pesos_oculto2_salida)
        }
        with open(archivo, "w") as f:
            json.dump(datos, f)

    def mutar(self, tasa_mutacion):
        # Aplica mutaciones aleatorias a los pesos de la red neuronal
        self.pesos_entrada_oculto1 += np.random.randn(*self.pesos_entrada_oculto1.shape) * tasa_mutacion
        self.pesos_oculto1_oculto2 += np.random.randn(*self.pesos_oculto1_oculto2.shape) * tasa_mutacion
        self.pesos_oculto2_salida += np.random.randn(*self.pesos_oculto2_salida.shape) * tasa_mutacion

#pygame.init()
ancho = 525
alto = 525
tam_celda = 35
tam_cuadricula_x = ancho // tam_celda
tam_cuadricula_y = alto // tam_celda
direcciones = [(0, -1), (0, 1), (-1, 0), (1, 0)]
umbral_puntuacion = 39
#reloj = pygame.time.Clock()
#pantalla = pygame.display.set_mode((ancho, alto))
#pygame.display.set_caption("ViperBrain")


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
        tam_entrada = 524
        entradas = np.zeros(tam_entrada)

        # Entradas originales como 1 y 0
        entradas[0] = cabeza_x < comida_x
        entradas[1] = cabeza_x > comida_x
        entradas[2] = cabeza_y < comida_y
        entradas[3] = cabeza_y > comida_y
        entradas[4] = direccion[0] == -1
        entradas[5] = direccion[0] == 1
        entradas[6] = direccion[1] == -1
        entradas[7] = direccion[1] == 1

        # Coordenadas de segmentos
        for i, segmento in enumerate(serpiente[1:]):
           x, y = segmento
           entradas[8 + i*2] = x
           entradas[8 + i*2 + 1] = y

        for i, segmento in enumerate(serpiente[1:]):
            x, y = segmento
            entradas[8 + i * 2] = x
            entradas[8 + i * 2 + 1] = y

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
            movimientos_restantes += 85  # Aumenta los movimientos restantes
        else:
            serpiente.pop()

        # Actualiza el contador de pasos
        pasos += 1
        movimientos_restantes -= 1

        #Actualiza la pantalla
        #pantalla.fill((255, 255, 255))  # Establece el fondo en blanco
        #dibujar_cuadricula()  # Dibuja la cuadrícula
        #for segmento in serpiente:
        #    pygame.draw.rect(
        #        pantalla, (0, 0, 0), (segmento[0] * tam_celda, segmento[1] * tam_celda, tam_celda, tam_celda)
        #    )
        #pygame.draw.rect(pantalla, (255, 0, 0), (comida[0] * tam_celda, comida[1] * tam_celda, tam_celda, tam_celda))
#
        #pygame.display.flip()

        #Limita la velocidad del juego
        #reloj.tick(5)  # Velocidad maxima ->comentar reloj, Velocidad normal ->reloj.tick(25)

    # Si la serpiente alcanza el número máximo de movimientos, devuelve la puntuación y los pasos
    return puntuacion, pasos


# Define los parámetros del algoritmo genético
tam_poblacion = 120
tasa_mutacion = 0.33
generaciones = 999999999999999999999999999
entrada = 524
oculta1 = 1310
oculta2 = 786
oculta3 = 393
oculta4 = 131
salida = 4
# Crea la población inicial de redes neuronales
poblacion = [RedNeuronal(entrada, oculta1, oculta2, oculta3, oculta4 , salida) for _ in range(tam_poblacion)]

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
            red.guardar_mejor_red2()
        else:
            print("Pasando a la siguiente generación")
    else:
        print("Score menor que " + str(umbral_puntuacion) + ". Continuando...")

    # Muta y cruza las mejores redes para generar la siguiente generación
    poblacion_nueva = []
    for _ in range(tam_poblacion):
        padre1, padre2 = random.choices(poblacion_top, k=2)
        hijo = RedNeuronal(entrada, oculta1, oculta2, oculta3, oculta4 , salida)
        hijo.pesos_entrada_oculto1 = padre1.pesos_entrada_oculto1.copy()
        hijo.pesos_oculto1_oculto2 = padre2.pesos_oculto1_oculto2.copy()
        hijo.pesos_oculto2_oculto3 = padre2.pesos_oculto2_oculto3.copy()
        hijo.pesos_oculto3_oculto4 = padre2.pesos_oculto3_oculto4.copy()
        hijo.pesos_oculto4_salida = padre1.pesos_oculto4_salida.copy()
        hijo.mutar(tasa_mutacion)
        poblacion_nueva.append(hijo)

    # Reemplaza la población anterior con la nueva generación
    poblacion = poblacion_nueva

print("La mejor red neuronal obtuvo una puntuación de:", mejor_puntuacion)
jugar_juego(mejor_red)