import numpy as np
import sys
import time
import cv2
from SDK import ELITE
from CameraCalibration import CameraCalibrationHelper
import threading

"""
Important parameters to twitch:

- gray_thresh: (0 to 255) the higher the value the less pixels it will detect
      twitch it till the camera detects perfectly the wood piece.

- area_thresh: (0 to 480x640) the perfectly detect wood piece should be around
        50000 pixels, the higher the value the more precise the center and
        angle of the piece will be, but with the risk of not detecting it.
        twitch it till the camera detects perfectly the center and angle
        of the wood piece.

- cam_min_lim/cam_max_lim: ((0, 0) to (640, 480)) the limits of them mask within
        the camera, it should be align with the conveyor belt so that in the mask
        there is only th conveyor belt, otherwise the camera will detect noise
        from the background. These limits will be displayed in the camera as a
        black rectangle.
        twitch it till the black rectangle is completely engulfed by the conveyor
        belt.
"""

class PalletizingRobot:
        
    def __init__(self, robot_ip, gray_thresh = 100, area_thresh = 45000, 
                 cam_min_lim = (0, 0), cam_max_lim = (640, 480)):
        self.robot = ELITE(robot_ip)
        self.frame = None
        self.camera = None
        self.piece_num = 0 # number of the piece that is being picked
        self.object_detected = False
        self.gray_thresh = gray_thresh
        self.area_thresh = area_thresh
        self.cam_min_lim = cam_min_lim
        self.cam_max_lim = cam_max_lim
        self.detected_angle=0.0
        
    def initialize_camera(self):
        self.helper = CameraCalibrationHelper()
        self.camera = self.helper.initialize_raspicam(headless = True, sensor_index = -1)
        self.helper.calibrate_raspberry()
        time.sleep(1)
        self.camera_available = True
    
    def camera_thread(self):
        if self.camera_available:
            while True:
                frame = self.camera.capture_array()[:, :, 0:3]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = self.helper.correct_image(frame)
                frame, mask, center, angle, success = self.detect_box(frame, self.gray_thresh,
                                                                      self.area_thresh, iter_ = 1)
                
                frame = cv2.rectangle(frame, self.cam_min_lim, self.cam_max_lim, (0, 0, 0), 10)
                cv2.imshow("Robot Camera", frame)
                cv2.imshow("Robot Camera mask", mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    def detect_box(self, frame, gray_thresh, area_thresh, iter_ = 1):
        """
        The angle of the wood piece is in the range of (-90, 90) in degreesm
        so that the conversion to the robot's Rz is easy.

        the iter_ parameter could be changed in case of very noisy environments,
        but it is not recommended to change it too much as it will distort the 
        calculation of the center of mass.        
        """
        aux = frame[self.cam_min_lim[1]:self.cam_max_lim[1],
                    self.cam_min_lim[0]:self.cam_max_lim[0]]
        
        # Grayscale detection
        gray_image = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)
        
        # mask thresh
        _, mask = cv2.threshold(gray_image, gray_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, None, iterations=iter_)
        mask = cv2.dilate(mask, None, iterations=iter_)
        
        # find contour with largest area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if there is contour:
        if not contours:
            return frame, mask, None, None, 0
        
        # check if area is over the min threshold
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < area_thresh:
            return frame, mask, None, None, 0
        
        # Detect square position and orientation 
        rect = cv2.minAreaRect(largest_contour)
        center, (width, height), angle = rect
        
        
        if width < height:
            angle += 90
        
        self.detected_width = width
        self.detected_height = height
        self.detected_angle = angle

        # Agregado: imprimir width, height y ángulo
        print(f"[INFO] Width: {width:.1f}, Height: {height:.1f}, Angle: {angle:.1f}°")
        
        # Ajuste de coordenadas al frame completo
        cx = int(center[0]) + self.cam_min_lim[0]
        cy = int(center[1]) + self.cam_min_lim[1]
        center = (cx, cy)
        # Print coordenadas desde la cámara
        print(f"[DETECT_BOX] Cámara → x={cx}, y={cy}, ángulo={angle:.1f}°")
        # Conversión afín a coordenadas del robot
        u, v = cx, cy
        a =  0.5208
        b = -132.57 # s·sin(θ)
        c = 0.098 # tx
        f = -52.005764  # ty
        Xr = a*u+b
        Yr = -34.796 
        corde=[Xr,Yr]    # -0.2561*u + 0.3541*v -86.17
        # Print coordenadas objetivo en el sistema del robot
        self.target_x=Xr
        self.target_y=Yr
        self.object_detected=True
        print(f"[MAP] Robot → X={Xr:.1f} mm, Y={Yr:.1f} mm")
        
        
        
        
        # Dibujar en pantalla
    
         #draw over frame
        box = cv2.boxPoints(rect).astype(int)
        box[:, 0] =  box[:, 0] + self.cam_min_lim[0]
        box[:, 1] =  box[:, 1] + self.cam_min_lim[1]
        frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2) 
        frame = cv2.circle(frame, center, 5, (255, 0, 0), 10)
        return frame, mask, center, angle, 1
        

    def map_camara2robot(self, center_x, angle):
        

        """
        [INCOMPLETE FUNCTION]: It should map the camera coordinates of the piece
        to the desired position of the robot, we will help you only with the 
        calculation of the robot x position, the rest is up to you!
        
        """
        # Calculation of the robot delta x position
        self.piece_angle = 90 - angle
        width = 90.0 # mean width of the wood piece
        height = 140.0 # mean height of the wood piece
        beta = np.arctan(height/width) 
        L = np.sqrt((width/2)**2 + (height/2)**2)
        if self.piece_angle < 0:
            aux = self.piece_angle
        else:
            aux = -self.piece_angle
        self.robot_x = L * np.sin(np.pi + (aux * (np.pi/180)) - beta)
        
        # Some hints
        self.robot_y_lims = None
        self.camera_x_center_lims = None
        self.robot_y = None
        self.robot_angle = None
        

    def mozaic_generator(self):
        """
        [INCOMPLETE FUNCTION]: It should generate the position of where the
        piece will be placed in the pallet.
        """
        # hint: do it with self.piece_num

        return None
    
    def pick_and_place(self):
        self.robot.open_gripper()
        self.robot.wait_until_motion_complete()
        
        origin_pose = [0, 0, -170, -0.584, -1.702, 91.0]  # Ajusta rx, ry, rz según tu orientación segura
        print("[INFO] Moviendo al origen seguro antes de iniciar pick and place...")
        self.robot.move_l_pose(np.array(origin_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()

        if not self.object_detected:
            return

        

        print("roberto come caca")
        print(f"[PICK_AND_PLACE] Ángulo detectado = {self.detected_angle:.1f}°")
        print(f"[INFO] Width: {self.detected_width:.1f}, Height: {self.detected_height:.1f}")

        # CASO 180 GRADOS (horizontal)
        if self.detected_width < self.detected_height:
            print("[INFO] El bloque está orientado en 0 grados (horizontal). Ejecutando flujo en X.")
            # Coordenadas fijas excepto X
            fixed_y = 0         # posición fija en Y
            fixed_z = -33       # altura segura para no tocar la cinta
            z_seguro = -170
            rx, ry, rz = 1.753, -0.411, -4.831  # orientación fija
            z_subida = -150
            y_caida = 470
            y_vuelta = 52.468
            x_vuelta = -35.709
            z_vuelta = -150.004
            x_pose = self.target_x - 16
            y_pose = self.target_y

            pose = [x_pose, self.target_y, z_seguro, rx, ry, rz]
            bajar = [x_pose, self.target_y, fixed_z, rx, ry, rz]
            levantar = [x_pose, y_pose, z_subida, rx, ry, rz]

            self.robot.open_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(pose), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            success, joints, _ = self.robot.get_current_joints()
            if not success:
                print("[ERROR] No se pudieron obtener los valores de los joints.")
                return
            joints[5] = 80.768
            self.robot.move_j_joint(joints, speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            print("[INFO] Joint 6 movido a 80.768 grados.")

            
            self.robot.open_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(pose), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(bajar), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.close_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(levantar), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(caida), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.open_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.close_gripper()
            self.robot.move_l_pose(np.array(vuelta), speed=20, acc=20)
            
            

        # CASO 90 GRADOS (vertical)
        elif self.detected_height < self.detected_width:
            print("[INFO] El bloque está orientado en 90 grados (vertical). Ejecutando flujo en Y.")
            # Coordenadas fijas excepto Y
            fixed_x = 0         # posición fija en X
            fixed_z = -33       # altura segura para no tocar la cinta
            z_seguro = -170
            rx, ry, rz = -0.584, -1.702, 91.0  # orientación fija
            z_subida = -150
            y_caida = 470
            y_vuelta = 52.468
            x_vuelta = -35.709
            z_vuelta = -150.004
            x_pose = self.target_x - 16
            # Mueve solo el Joint 6 a 80.768°
            

            x_pose = self.target_x - 16
            y_pose = self.target_y

            pose = [x_pose, y_pose, z_seguro, rx, ry, rz]
            bajar = [x_pose, y_pose, fixed_z, rx, ry, rz]
            levantar = [x_pose, y_pose, z_subida, rx, ry, rz]

            self.robot.open_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(pose), speed=10, acc=20)
            self.robot.wait_until_motion_complete()

            self.robot.open_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(pose), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(bajar), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.close_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(levantar), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.move_l_pose(np.array(caida), speed=10, acc=20)
            self.robot.wait_until_motion_complete()
            self.robot.open_gripper()
            self.robot.wait_until_motion_complete()
            self.robot.close_gripper()
            self.robot.move_l_pose(np.array(vuelta), speed=20, acc=20)
        else:
            print("[INFO] El bloque NO cumple condiciones para 180° ni 90°. No se ejecuta movimiento.")
            return

   
    def run(self):
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()

        if self.robot.connect():
            print("Successfully connected to robot")
            self.robot.set_servo_status(1)

            # --- Initialization (place this *before* your while loop) ---
            previous_sensor_state_is_detecting = None  # Stores the last known sensor state
            SENSOR_ADDRESS = 915
            
            while True:
                
                result = self.robot.send_cmd("getVirtualOutput", {"addr": SENSOR_ADDRESS})

                current_sensor_is_detecting = None # Reset for current iteration

                # Check if the read was successful and result format is valid
                if result and isinstance(result, tuple) and len(result) >= 2:
                    read_success = result[0]
                    sensor_value = result[1] # The middle element indicating sensor state

                    if read_success is True:
                        # We assume 1 means "high" (detecting), and 0 means "low" (not detecting)
                        if sensor_value == 1:
                            current_sensor_is_detecting = True
                        elif sensor_value == 0:
                            current_sensor_is_detecting = False
                        else:
                            # Optional: Handle unexpected sensor_value if it can be other than 0 or 1
                            print(f"Advertencia: Valor de sensor inesperado ({sensor_value}) en la dirección {SENSOR_ADDRESS}")
                            current_sensor_is_detecting = None # Treat as unknown to avoid incorrect state change message
                    else:
                        print(f"Advertencia: Fallo la lectura de la variable ({SENSOR_ADDRESS}). Resultado: {result}")
                        # If read failed, current state is unknown.
                        current_sensor_is_detecting = None 
                        
                else:
                    print(f"Error: Formato de resultado inesperado de get_virtual_var para la dirección {SENSOR_ADDRESS}: {result}")
                    current_sensor_is_detecting = None

                # Only proceed if we have a validly determined current sensor state
                if current_sensor_is_detecting is not None:
                    if current_sensor_is_detecting != previous_sensor_state_is_detecting:
                        if current_sensor_is_detecting is True:
                            print("Sensor no detectando")
                        else:
                            print("Sensor detectando")
                            if self.object_detected:
                              self.pick_and_place()
                        
                        previous_sensor_state_is_detecting = current_sensor_is_detecting

                time.sleep(0.5)
        else:
            print("No se pudo conectar al robot")

        self.robot.disconnect()



if __name__ == "__main__":
    # Example usage
    robot_ip = "169.168.0.200"
    robot = PalletizingRobot(robot_ip)
    robot.initialize_camera()
    robot.run()