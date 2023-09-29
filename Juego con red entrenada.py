import json, numpy as np, pygame, random
import sys

# Clase para la red
class RedNeuronal:
    def __init__(self, tam_entrada, tam_oculto, tam_salida):
        self.tam_entrada = tam_entrada
        self.tam_oculto = tam_oculto
        self.tam_salida = tam_salida

        with open('pesos.json') as archivo:
            pesos = json.load(archivo)
        self.pesos_entrada_oculto1 = np.array(pesos["pesos_e_o"], dtype=np.float64)
        self.pesos_oculto1_oculto2 = np.array(pesos["pesos_o_o"], dtype=np.float64)
        self.pesos_oculto2_salida = np.array(pesos["pesos_o_s"], dtype=np.float64)

    def forward(self, x):
        oculto1 = np.dot(x, self.pesos_entrada_oculto1)
        oculto2 = np.dot(oculto1, self.pesos_oculto1_oculto2)
        salida = np.dot(oculto2, self.pesos_oculto2_salida)
        return salida

    def relu(self, x):
        # Función de activación ReLU [0, 1]
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Función de activación Sigmoide [0 a 1)
        return 1 / (1 + np.exp(-x))


pygame.init()
ancho = 735
alto = 735
tam_celda = 49
tam_cuadricula_x = ancho // tam_celda
tam_cuadricula_y = alto // tam_celda
verde_claro = (0, 255, 0)
verde_oscuro = (0, 100, 0)
direcciones = [(0, -1), (-1, 0), (0, 1), (1, 0)]
reloj = pygame.time.Clock()
pantalla = pygame.display.set_mode((ancho, alto))
pygame.display.set_caption("ViperBrain")
puntuacion = 0
ojo_radio = 25
pupila_radio = 12


def dibujar_cuadricula():
    # Dibuja una cuadrícula en la pantalla
    for x in range(0, ancho, tam_celda):
        pygame.draw.line(pantalla, (255, 255, 255), (x, 0), (x, alto))
    for y in range(0, alto, tam_celda):
        pygame.draw.line(pantalla, (255, 255, 255), (0, y), (ancho, y))

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


#def check_colision(segments):
#   head = segments[0]

#   for segment in segments[1:]:
#       direccion = None
#       while abs(head[0] - segment[0]) + abs(head[1] - segment[1]) == 1:
#           # La cabeza está a un paso de chocar con el cuerpo
#           # Gira 90 grados
#           if head[0] < segment[0]:
#               direccion = (0, 1)  # Gira hacia abajo
#           elif head[0] > segment[0]:
#               direccion = (0, -1)  # Gira hacia arriba
#           elif head[1] < segment[1]:
#               direccion = (1, 0)  # Gira hacia la derecha
#           else:
#               direccion = (-1, 0)  # Gira hacia la izquierda
#       return direccion

def jugar_juego(red):
    global puntuacion
    serpiente, direccion = generar_nueva_serpiente()
    comida = generar_comida(serpiente)
    puntuacion = 0
    pasos = 0
    movimientos_restantes = 10**10
#    colision = check_colision(serpiente)
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
        #if colision is not None: # Añadido para probar
        #    direccion = colision

        if direccion == direcciones[movimiento]:
            # Se vuelve a elegir una dirección
            movimiento = np.argmax(salidas)
            direccion = direcciones[movimiento]

        serpiente.insert(0, (serpiente[0][0] + direccion[0], serpiente[0][1] + direccion[1]))
        if len(list(enumerate(serpiente))) > len(set(enumerate(serpiente))):

            serpiente, direccion = generar_nueva_serpiente()
            comida = generar_comida(serpiente)

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
            comida = generar_comida(serpiente)

        else:
            serpiente.pop()

        #Actualiza la pantalla
        pantalla.fill((0,0,0))  # Establece el fondo en blanco
        dibujar_cuadricula()  # Dibuja la cuadrícula
        for i, segmento in enumerate(serpiente):
            longitud = len(list(enumerate(serpiente)))
            color = (
                verde_claro[0] + (verde_oscuro[0] - verde_claro[0]) * i / len(serpiente),
                verde_claro[1] + (verde_oscuro[1] - verde_claro[1]) * i / len(serpiente),
                verde_claro[2] + (verde_oscuro[2] - verde_claro[2]) * i / len(serpiente)
            )
            if i == 0:
                # Cabeza - redondear esquinas
                border_radius = tam_celda // 2
            elif i == len(serpiente) - 1:
                # Cola - redondear esquinas
                border_radius = tam_celda // 2
            else:
                # Cuerpo - sin redondear
                border_radius = 0
            segmento_index = len(serpiente) - i - 1
            segmento = serpiente[segmento_index]

            pygame.draw.rect(pantalla, color,
                             (segmento[0] * tam_celda, segmento[1] * tam_celda, tam_celda, tam_celda))
            if i == longitud - 1:  # Draw eyes on the head segment
                # Draw eyes on head segment
                cabeza = serpiente[0]
                centro_x = cabeza[0] * tam_celda + tam_celda // 2
                centro_y = cabeza[1] * tam_celda + tam_celda // 2
                ojo = (centro_x, centro_y)
                pygame.draw.circle(pantalla, (255, 255, 255), ojo, ojo_radio)
                pygame.draw.circle(pantalla, (0, 0, 0), ojo, pupila_radio)

            pygame.draw.rect(pantalla, (255, 0, 0),
                             (comida[0] * tam_celda, comida[1] * tam_celda, tam_celda, tam_celda))
        pygame.display.flip()
        print(puntuacion)
        #Limita la velocidad del juego
        reloj.tick(15)  # Velocidad maxima ->comentar reloj, Velocidad normal ->reloj.tick(25)

    # Si la serpiente alcanza el número máximo de movimientos, devuelve la puntuación y los pasos
    return puntuacion, pasos

partidas = 999
mi_red_neuronal = RedNeuronal(8, 16, 4)
for partida in range(partidas + 1):
    if partida == 0 and puntuacion == 0:
        pass
    else:
        print(puntuacion, "puntos obtenidos en la partida nº", partida)
        jugar_juego(mi_red_neuronal)




