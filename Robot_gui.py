import tkinter as tk
from tkinter import messagebox
from PalletizingRobot import PalletizingRobot
import threading
import time

class RobotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Robot Paletizador")
        self.master.geometry("350x300")

        self.robot = None
        self.run_thread = None
        self.camera_thread = None

        # Entrada de IP
        tk.Label(master, text="IP del Robot:").pack()
        self.ip_entry = tk.Entry(master, width=30)
        self.ip_entry.insert(0, "192.168.1.100")  # Cambiar a IP por defecto si se conoce
        self.ip_entry.pack(pady=5)

        # Botón de estado del sensor
        self.sensor_status_btn = tk.Button(master, text="Estado del Sensor", bg="gray", command=self.actualizar_estado_sensor)
        self.sensor_status_btn.pack(pady=15)

        # Botón para iniciar RUN
        tk.Button(master, text="Iniciar RUN", bg="lightgreen", command=self.iniciar_ciclo_run).pack(pady=5)

        # Botón para detener RUN
        tk.Button(master, text="Detener RUN", bg="tomato", command=self.detener_ciclo_run).pack(pady=5)

        # Actualizar estado del sensor cada segundo
        self.actualizar_estado_sensor_periodicamente()

    def actualizar_estado_sensor(self):
        if self.robot and self.robot.object_detected:
            self.sensor_status_btn.configure(bg="green", text="Objeto Detectado")
        else:
            self.sensor_status_btn.configure(bg="red", text="Sin Objeto")

    def actualizar_estado_sensor_periodicamente(self):
        self.actualizar_estado_sensor()
        self.master.after(1000, self.actualizar_estado_sensor_periodicamente)

    def iniciar_ciclo_run(self):
        ip = self.ip_entry.get()
        try:
            self.robot = PalletizingRobot(ip)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo crear la instancia del robot:\n{e}")
            return

        # Inicializar cámara si es necesario
        if not self.robot.camera_available:
            try:
                self.robot.initialize_camera()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo iniciar la cámara:\n{e}")
                return

        # Iniciar hilo de cámara si no está activo
        if not self.camera_thread or not self.camera_thread.is_alive():
            self.camera_thread = threading.Thread(target=self.robot.camera_thread, daemon=True)
            self.camera_thread.start()

        # Intentar conexión al robot (verifica si realmente está en red)
        success = self.robot.robot.connect()
        if not success:
            messagebox.showerror("Error de conexión", "No se pudo conectar con el robot.\nVerifica la IP o la red.")
            return

        # Iniciar RUN si no está corriendo ya
        if not self.run_thread or not self.run_thread.is_alive():
            self.robot._stop_event.clear()
            self.run_thread = threading.Thread(target=self.robot.run, daemon=True)
            self.run_thread.start()
            messagebox.showinfo("RUN", "Ciclo RUN iniciado.")
        else:
            messagebox.showinfo("RUN", "El ciclo ya está en ejecución.")

    def detener_ciclo_run(self):
        if self.robot:
            self.robot._stop_event.set()
            messagebox.showinfo("RUN", "Ciclo RUN detenido.")

# Ejecutar GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()
