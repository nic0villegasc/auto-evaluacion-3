import tkinter as tk
from tkinter import messagebox
from PalletizingRobot import PalletizingRobot
import threading
import time

class RobotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Robot Paletizador")
        self.master.geometry("500x400")

        self.robot = None
        self.run_thread = None
        self.camera_thread = None

        # Etiqueta y campo de IP
        tk.Label(master, text="IP del Robot:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.ip_entry = tk.Entry(master, width=30)
        self.ip_entry.insert(0, "192.168.1.100")
        self.ip_entry.grid(row=0, column=1, padx=10, pady=5)

        # Botón de estado del sensor
        self.sensor_status_btn = tk.Button(master, text="Estado del Sensor", bg="gray", command=self.actualizar_estado_sensor)
        self.sensor_status_btn.grid(row=1, column=0, columnspan=2, pady=10)

        # Botón iniciar RUN
        tk.Button(master, text="Iniciar RUN", bg="lightgreen", command=self.iniciar_ciclo_run).grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # Botón detener RUN
        tk.Button(master, text="Detener RUN", bg="tomato", command=self.detener_ciclo_run).grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Botón Reset pallet 0
        tk.Button(master, text="Reset pallet 0", command=self.reset_pallet_0).grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # Botón Reset pallet 90
        tk.Button(master, text="Reset pallet 90", command=self.reset_pallet_90).grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        # Botón Stop (igual a detener RUN)
        tk.Button(master, text="STOP", bg="red", fg="white", command=self.detener_ciclo_run).grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Barra de estado (puedes usar para mostrar mensajes)
        self.status_label = tk.Label(master, text="Estado: Listo", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="we", padx=5, pady=10)

        # Actualizar sensor automáticamente
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

        if not self.robot.camera_available:
            try:
                self.robot.initialize_camera()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo iniciar la cámara:\n{e}")
                return

        if not self.camera_thread or not self.camera_thread.is_alive():
            self.camera_thread = threading.Thread(target=self.robot.camera_thread, daemon=True)
            self.camera_thread.start()

        success = self.robot.robot.connect()
        if not success:
            messagebox.showerror("Error de conexión", "No se pudo conectar con el robot.\nVerifica la IP o la red.")
            return

        if not self.run_thread or not self.run_thread.is_alive():
            self.robot._stop_event.clear()
            self.run_thread = threading.Thread(target=self.robot.run, daemon=True)
            self.run_thread.start()
            self.status_label.config(text="Estado: RUN en ejecución")
        else:
            messagebox.showinfo("RUN", "El ciclo ya está en ejecución.")

    def detener_ciclo_run(self):
        if self.robot:
            self.robot._stop_event.set()
            self.status_label.config(text="Estado: RUN detenido")

    def reset_pallet_0(self):
        if self.robot:
            self.robot.piece_count_zone_0_deg = 0
            messagebox.showinfo("Reset", "Contador de pallet 0 reiniciado.")
            self.status_label.config(text="Estado: pallet 0 = 0")

    def reset_pallet_90(self):
        if self.robot:
            self.robot.piece_count_zone_90_deg = 0
            messagebox.showinfo("Reset", "Contador de pallet 90 reiniciado.")
            self.status_label.config(text="Estado: pallet 90 = 0")

# Ejecutar GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()
