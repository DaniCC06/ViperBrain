import json, numpy as np, pygame, random

class RedNeuronal:

    def __init__(self, tam_entrada, tam_oculto, tam_salida):
        self.tam_entrada, self.tam_oculto1, self.tam_oculto2, self.tam_salida = tam_entrada, tam_oculto, tam_oculto, tam_salida
        self.pesos_entrada_oculto1 = np.random.randn(self.tam_entrada, self.tam_oculto1)
        self.pesos_oculto1_oculto2 = np.random.randn(self.tam_oculto1, self.tam_oculto2)
        self.pesos_oculto2_salida = np.random.randn(self.tam_oculto2, self.tam_salida)

    def forward(self, x):
        oculto1 = np.dot(x, self.pesos_entrada_oculto1)
        oculto2 = np.dot(oculto1, self.pesos_oculto1_oculto2)
        salida = np.dot(oculto2, self.pesos_oculto2_salida)
        return salida

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def guardar_pesos(self):
        pesos = {
            'pesos_e_o': self.pesos_entrada_oculto1.tolist(),
            'pesos_o_o': self.pesos_oculto1_oculto2.tolist(),
            'pesos_o_s': self.pesos_oculto2_salida.tolist()
        }

        with open('pesos.json', 'w') as archivo:
            json.dump(pesos, archivo)

    def mutar(self, tasa_mutacion):
        for pesos in [self.pesos_entrada_oculto1, self.pesos_oculto1_oculto2, self.pesos_oculto2_salida]:
            pesos += np.random.randn(*pesos.shape) * tasa_mutacion

pygame.init()
ancho, alto, tam_celda = 525, 525, 35
direcciones = [(0, -1), (-1, 0), (0, 1), (1, 0)]
umbral_puntuacion = 41
tam_cuadricula_x, tam_cuadricula_y = ancho // tam_celda, alto // tam_celda
tam_poblacion, tasa_mutacion, generaciones = 1600, 0.29, 9**5
poblacion = [RedNeuronal(8, 16, 4) for _ in range(tam_poblacion)]
mejor_red = None
mejor_puntuacion = 0
def generar_nueva_serpiente():
    serpiente = [(tam_cuadricula_x // 2, tam_cuadricula_y // 2)]
    direccion = random.choice(direcciones)
    return serpiente, direccion

def generar_comida(serpiente):
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
    serpiente, direccion = generar_nueva_serpiente()
    comida = generar_comida(serpiente)
    puntuacion = 0
    pasos = 0
    movimientos_restantes = 66

    while movimientos_restantes > 0:
        cabeza_x, cabeza_y = serpiente[0]
        comida_x, comida_y = comida
        entradas = np.array([
            cabeza_x < comida_x,
            cabeza_x > comida_x,
            cabeza_y < comida_y,
            cabeza_y > comida_y,
            direccion[0] == -1,
            direccion[0] == 1,
            direccion[1] == -1,
            direccion[1] == 1,

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
            movimientos_restantes += 150
        else:
            serpiente.pop()

        pasos += 1
        movimientos_restantes -= 1

    return puntuacion, pasos

for generacion in range(generaciones):
    puntuaciones = []
    for red in poblacion:
        puntuacion, colisiones = jugar_juego(red)
        puntuaciones.append(puntuacion)

    indices_top = np.argsort(puntuaciones)[::-1][:tam_poblacion // 70]#mayor numero, menores muestras para la siguiente generacion
    poblacion_top = [poblacion[i] for i in indices_top]
    mejor_puntuacion_actual = max(puntuaciones)

    if mejor_puntuacion_actual > mejor_puntuacion:
        mejor_red = poblacion_top[0]
        mejor_puntuacion = mejor_puntuacion_actual
    print(mejor_puntuacion_actual, "puntos obtenidos en la generación nº", generacion)
    if mejor_puntuacion_actual > umbral_puntuacion:
        opcion_guardar = input("¿Deseas guardar la configuración de la red neuronal? (s/n): ")
        if opcion_guardar.lower() == "s":
            print("Guardando la configuración de la red neuronal...")
            mejor_red.guardar_pesos()
        else:
            print("Pasando a la siguiente generación")
    else:
        print("Score menor que " + str(umbral_puntuacion - 1) + ". Continuando...")

    poblacion_nueva = []
    for _ in range(tam_poblacion):
        padre1, padre2, padre3 = random.choices(poblacion_top, k=3)
        hijo = RedNeuronal(8, 16, 4)
        hijo.pesos_entrada_oculto1 = padre1.pesos_entrada_oculto1.copy()
        hijo.pesos_oculto1_oculto2 = padre2.pesos_oculto1_oculto2.copy()
        hijo.pesos_oculto2_salida = padre3.pesos_oculto2_salida.copy()
        hijo.mutar(tasa_mutacion)
        poblacion_nueva.append(hijo)

    poblacion = poblacion_nueva

