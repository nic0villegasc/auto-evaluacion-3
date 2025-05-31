# Proyecto Robot Paletizador - Evaluación 3

## Descripción General

Este proyecto implementa el sistema de control para una celda robótica de paletizado en miniatura. El sistema está diseñado para identificar piezas en una cinta transportadora, determinar su orientación y utilizar un brazo robótico para recogerlas y colocarlas en zonas de paletizado designadas según un patrón de mosaico. El control principal reside en una Raspberry Pi que orquesta los diferentes componentes de la celda.

Este proyecto fue desarrollado como parte de la "Evaluación 3 – MVP y Validación" del curso de Automatización Industrial.

## Componentes Principales del Código

El repositorio contiene los siguientes scripts principales de Python:

1.  **`CameraCalibration.py`**:
    * Responsable de la calibración de la cámara (PiCamera).
    * Incluye procedimientos para ajustar la exposición, el balance de blancos y la corrección de color utilizando una tarjeta de color SpyderCheckr24.
    * Maneja la calibración de la distorsión de la cámara mediante un patrón de tablero de ajedrez.
    * Guarda y carga los parámetros de calibración para su uso posterior.

2.  **`PalletizingRobot.py`**:
    * Contiene la lógica principal para el robot de paletizado.
    * Utiliza `CameraCalibrationHelper` de `CameraCalibration.py` para inicializar y usar la cámara para la detección de objetos (piezas de madera) en la cinta transportadora.
    * Determina la posición y orientación de las piezas.
    * Utiliza el SDK `ELITE` (definido en `SDK.py`) para comandar el brazo robótico para las operaciones de recoger y colocar (`pick and place`).
    * Clasifica las piezas según su orientación (ej. 0° o 90°) y las coloca en zonas de paletizado predefinidas.
    * Gestiona una cola de objetos, diferentes poses del robot, y la lógica para el llenado de los pallets.
    * Incluye manejo de casos como piezas no clasificadas o pallets llenos.

3.  **`SDK.py`**:
    * Proporciona la clase `ELITE`, que actúa como un Kit de Desarrollo de Software (SDK) para interactuar con un brazo robótico ELITE mediante una conexión Ethernet.
    * Ofrece métodos para conectarse al robot, enviar diversos comandos (movimiento, obtener estado, configurar E/S, gestionar sistemas de coordenadas y herramientas), realizar cálculos de cinemática inversa y controlar la pinza (gripper) del robot.

## Funcionalidades Clave

* **Detección de Piezas Basada en Cámara:** El sistema utiliza una cámara para detectar piezas, discerniendo su posición y rotación.
* **Paletizado Diferencial:** Las piezas se paletizan en diferentes zonas según su rotación, formando un patrón de mosaico.
* **Control del Brazo Robótico:** Gestión completa de las operaciones de recoger y colocar del brazo robótico.
* **Integración del Sistema:** Incluye el control de la cinta transportadora y lógica para manejar escenarios diversos, buscando una operación autónoma.
* **Calibración de Cámara:** Procedimientos detallados para asegurar una detección visual precisa.
* **Manejo de Casos Excepcionales:** Lógica para abordar situaciones como piezas no clasificables o zonas de paletizado completas.

## Requisitos

Las dependencias de Python necesarias para ejecutar este proyecto se encuentran listadas en el archivo `requirements.txt`. Se pueden instalar usando pip:

```bash
pip install -r requirements.txt
```

Algunas dependencias clave incluyen:
* `picamera2`
* `opencv-python`
* `numpy`
* `matplotlib` (para algunas funciones de calibración)

## Uso y Ejecución

1.  **Calibración de la Cámara (si es necesario):**
    * Si los parámetros de calibración (`CalibrationParams/`) no existen o las condiciones de iluminación han cambiado significativamente, puede ser necesario ejecutar `CameraCalibration.py` primero o seguir las instrucciones internas del script `PalletizingRobot.py` que invoca la calibración.
    * El script `CameraCalibration.py` guía al usuario a través del proceso de calibración usando una tarjeta SpyderCheckr24 y un patrón de tablero de ajedrez.

2.  **Ejecución del Sistema Principal:**
    * El script principal para operar el sistema de paletizado es `PalletizingRobot.py`.
    * Asegúrese de que la Raspberry Pi esté conectada al brazo robótico ELITE (configurando la IP correcta en el script) y a la cámara.
    * Ejecute el script desde la terminal:
        ```bash
        python PalletizingRobot.py
        ```
    * El script inicializará la cámara (realizando la calibración si es necesario y no existen parámetros guardados), se conectará al robot y comenzará el ciclo de detección y paletizado.

## Estructura del Proyecto

```
.
├── CalibrationParams/        # Directorio para guardar los parámetros de calibración de la cámara
├── CameraCalibration.py      # Script para la calibración de la cámara
├── PalletizingRobot.py       # Script principal de control del robot paletizador
├── SDK.py                    # SDK para la comunicación con el robot ELITE
└── requirements.txt          # Lista de dependencias de Python
```

## Consideraciones Adicionales

* **Configuración de Red:** Verificar que la dirección IP del robot ELITE esté correctamente configurada en `PalletizingRobot.py`.
* **Entorno Físico:** La precisión del sistema depende de una buena calibración de la cámara y de la configuración física estable de la celda robótica.
* **Modo Paso a Paso:** El script `PalletizingRobot.py` incluye una variable `step_mode` que puede ser útil para depuración, permitiendo confirmar cada paso del robot antes de su ejecución.
