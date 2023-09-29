import json, numpy as np, pygame, random

class RedNeuronal:
    def __init__(self, tam_entrada, tam_oculto, tam_salida):
        self.tam_entrada = tam_entrada
        self.tam_oculto = tam_oculto
        self.tam_salida = tam_salida

        with open('pesos1.json') as archivo:
            pesos = json.load(archivo)
        self.pesos_entrada_oculto1 = np.array(pesos["pesos_e_o"], dtype=np.float64)
        self.pesos_oculto1_oculto2 = np.array(pesos["pesos_o1_o2"], dtype=np.float64)
        self.pesos_oculto2_oculto3 = np.array(pesos["pesos_o2_o3"], dtype=np.float64)
        self.pesos_oculto3_salida = np.array(pesos["pesos_o_s"], dtype=np.float64)

    def forward(self, x):
        oculto1 = np.dot(x, self.pesos_entrada_oculto1)
        oculto2 = np.dot(oculto1, self.pesos_oculto1_oculto2)
        oculto3 = np.dot(oculto2, self.pesos_oculto2_oculto3)
        salida = np.dot(oculto3, self.pesos_oculto3_salida)
        return salida

    def relu(self, x):
        # Función de activación ReLU [0, 1]
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Función de activación Sigmoide [0 a 1)
        return 1 / (1 + np.exp(-x))


pygame.init()
ancho = 525
alto = 525
tam_celda = 35
tam_cuadricula_x = ancho // tam_celda
tam_cuadricula_y = alto // tam_celda
direcciones = [(0, -1), (0, 1), (-1, 0), (1, 0)]
reloj = pygame.time.Clock()
pantalla = pygame.display.set_mode((ancho, alto))
pygame.display.set_caption("ViperBrain")
puntuacion = 0

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


def check_collision(segments):
    head = segments[0]

    for segment in segments[1:]:
        if abs(head[0] - segment[0]) + abs(head[1] - segment[1]) == 1:
            # La cabeza está a un paso de chocar con el cuerpo
            # Gira 90 grados
            if head[0] < segment[0]:
                direccion = (0, 1)  # Gira hacia abajo
            elif head[0] > segment[0]:
                direccion = (0, -1)  # Gira hacia arriba
            elif head[1] < segment[1]:
                direccion = (1, 0)  # Gira hacia la derecha
            else:
                direccion = (-1, 0)  # Gira hacia la izquierda
            return direccion

    return None

def jugar_juego(red):
    global puntuacion
    serpiente, direccion = generar_nueva_serpiente()
    comida = generar_comida(serpiente)
    puntuacion = 0
    pasos = 0
    movimientos_restantes = 66
    colision = check_collision(serpiente)
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
        if colision:
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
            pasos = 0
            comida = generar_comida(serpiente)
            movimientos_restantes += 150  # Aumenta los movimientos restantes

        else:
            serpiente.pop()

        # Actualiza el contador de pasos
        pasos += 1
        movimientos_restantes -= 1

        #Actualiza la pantalla
        pantalla.fill((255, 255, 255))  # Establece el fondo en blanco
        dibujar_cuadricula()  # Dibuja la cuadrícula
        for segmento in serpiente:
            pygame.draw.rect(
                pantalla, (0, 0, 0), (segmento[0] * tam_celda, segmento[1] * tam_celda, tam_celda, tam_celda)
            )
        pygame.draw.rect(pantalla, (255, 0, 0), (comida[0] * tam_celda, comida[1] * tam_celda, tam_celda, tam_celda))

        pygame.display.flip()

        #Limita la velocidad del juego
        reloj.tick(30)  # Velocidad maxima ->comentar reloj, Velocidad normal ->reloj.tick(25)

    # Si la serpiente alcanza el número máximo de movimientos, devuelve la puntuación y los pasos
    return puntuacion, pasos

partidas = 999
mi_red_neuronal = RedNeuronal(8, 9, 4)
for partida in range(partidas + 1):
    if partida == 0 and puntuacion == 0:
        pass
    else:
        print(puntuacion, "puntos obtenidos en la partida nº", partida)
        jugar_juego(mi_red_neuronal)




