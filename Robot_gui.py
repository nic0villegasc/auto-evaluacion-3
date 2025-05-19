import tkinter as tk
from tkinter import messagebox
from PalletizingRobot import PalletizingRobot
import threading
import time

class RobotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Robot Paletizador")
        self.master.geometry("300x250")
        self.robot = PalletizingRobot("192.168.1.100")  # Cambia al IP real si es necesario
        self.run_thread = None

        # Botón de estado del sensor
        self.sensor_status_btn = tk.Button(master, text="Estado del Sensor", bg="gray", command=self.actualizar_estado_sensor)
        self.sensor_status_btn.pack(pady=20)

        # Botón para iniciar el ciclo RUN
        tk.Button(master, text="Iniciar RUN", bg="lightgreen", command=self.iniciar_ciclo_run).pack(pady=10)

        # Botón para detener el ciclo RUN
        tk.Button(master, text="Detener RUN", bg="tomato", command=self.detener_ciclo_run).pack(pady=10)

        # Actualización automática del estado del sensor cada 1 segundo
        self.actualizar_estado_sensor_periodicamente()

    def actualizar_estado_sensor(self):
        if self.robot.object_detected:
            self.sensor_status_btn.configure(bg="green", text="Objeto Detectado")
        else:
            self.sensor_status_btn.configure(bg="red", text="Sin Objeto")

    def actualizar_estado_sensor_periodicamente(self):
        self.actualizar_estado_sensor()
        self.master.after(1000, self.actualizar_estado_sensor_periodicamente)

    def iniciar_ciclo_run(self):
        if not self.run_thread or not self.run_thread.is_alive():
            self.robot._stop_event.clear()
            self.run_thread = threading.Thread(target=self.robot.run, daemon=True)
            self.run_thread.start()
            messagebox.showinfo("RUN", "Ciclo RUN iniciado.")
        else:
            messagebox.showinfo("RUN", "Ya está en ejecución.")

    def detener_ciclo_run(self):
        if self.robot:
            self.robot._stop_event.set()
            messagebox.showinfo("RUN", "Ciclo RUN detenido.")

# Ejecutar interfaz
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()
