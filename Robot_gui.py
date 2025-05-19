import tkinter as tk
from tkinter import messagebox
from PalletizingRobot import PalletizingRobot
import threading
import time
import os

class RobotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Robot Paletizador")
        self.master.geometry("500x450")

        self.robot = None
        self.run_thread = None
        self.camera_thread = None

        tk.Label(master, text="IP del Robot:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.ip_entry = tk.Entry(master, width=30)
        self.ip_entry.insert(0, "192.168.1.100")
        self.ip_entry.grid(row=0, column=1, padx=10, pady=5)

        self.sensor_status_btn = tk.Button(master, text="Estado del Sensor", bg="gray", command=self.actualizar_estado_sensor)
        self.sensor_status_btn.grid(row=1, column=0, columnspan=2, pady=10)

        tk.Button(master, text="Iniciar RUN", bg="lightgreen", command=self.iniciar_ciclo_run).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        tk.Button(master, text="Detener RUN", bg="tomato", command=self.detener_ciclo_run).grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        tk.Button(master, text="Reset pallet 0", command=self.reset_pallet_0).grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        tk.Button(master, text="Reset pallet 90", command=self.reset_pallet_90).grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        tk.Button(master, text="STOP", bg="red", fg="white", command=self.detener_ciclo_run).grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        tk.Button(master, text="Ver log de errores", command=self.ver_log).grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.status_label = tk.Label(master, text="Estado: Listo", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status_label.grid(row=6, column=0, columnspan=2, sticky="we", padx=5, pady=10)

        self.actualizar_estado_sensor_periodicamente()
        # Ventana de estado
        estado_frame = tk.LabelFrame(master, text="Estado del robot", padx=10, pady=10)
        estado_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.label_sensor = tk.Label(estado_frame, text="Sensor: ---", anchor="w")
        self.label_sensor.pack(fill="x")

        self.label_gripper = tk.Label(estado_frame, text="Gripper: ---", anchor="w")
        self.label_gripper.pack(fill="x")

        self.label_queue = tk.Label(estado_frame, text="Objetos en cola: ---", anchor="w")
        self.label_queue.pack(fill="x")
        self.actualizar_estado_robot()


    def actualizar_estado_sensor(self):
        if self.robot and self.robot.object_detected:
            self.sensor_status_btn.configure(bg="green", text="Objeto Detectado")
        else:
            self.sensor_status_btn.configure(bg="red", text="Sin Objeto")

    def actualizar_estado_robot(self):
        if self.robot:
            sensor_estado = "Detectado" if self.robot.object_detected else "No detectado"
            gripper_estado = "Activado" if getattr(self.robot, "gripper_closed", False) else "Desactivado"
            queue_len = self.robot.object_queue.qsize() if self.robot.object_queue else 0

            self.label_sensor.config(text=f"Sensor: {sensor_estado}")
            self.label_gripper.config(text=f"Gripper: {gripper_estado}")
            self.label_queue.config(text=f"Objetos en cola: {queue_len}")
        
        self.master.after(1000, self.actualizar_estado_robot)


    def actualizar_estado_sensor_periodicamente(self):
        self.actualizar_estado_sensor()
        try:
            with open("robot_errors.log", "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    self.status_label.config(text=f"Último error: {last_line[-80:]}")
        except FileNotFoundError:
            pass
        self.master.after(3000, self.actualizar_estado_sensor_periodicamente)

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

    def ver_log(self):
        try:
            with open("robot_errors.log", "r") as f:
                log_content = f.read()
            log_window = tk.Toplevel(self.master)
            log_window.title("Log de errores")
            text_area = tk.Text(log_window, wrap="word")
            text_area.insert("1.0", log_content)
            text_area.pack(expand=True, fill="both")
        except FileNotFoundError:
            messagebox.showinfo("Log", "El archivo de log aún no existe.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()
