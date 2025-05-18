import numpy as np
import sys
import time
import cv2
from SDK import ELITE
from CameraCalibration import CameraCalibrationHelper
import threading
import math

class PalletizingRobot:
        
    def __init__(self, robot_ip, gray_thresh = 100, area_thresh = 45000, 
                 cam_min_lim = (0, 0), cam_max_lim = (640, 480)):
        self.robot = ELITE(robot_ip)
        self.frame = None
        self.camera = None
        self.camera_available = False 
        self.helper = None 

        self.piece_num = 0 
        self.object_detected = False
        
        self.target_x = 0.0 
        self.target_y = 0.0 
        self.piece_angle = 0.0
        self.target_j6_rad = 0.0

        # Affine transformation coefficients for X-coordinate
        # X_robot = X_MAPPING_SLOPE * x_cam_pixel + X_MAPPING_INTERCEPT
        self.X_MAPPING_SLOPE = 0.5208
        self.X_MAPPING_INTERCEPT = -132.57

        # Affine transformation coefficients for Y-coordinate
        # Y_robot = Y_MAPPING_SLOPE * y_cam_pixel + Y_MAPPING_INTERCEPT
        # TODO: Need to calibrate these values!
        self.Y_MAPPING_SLOPE = 0.5  # Placeholder - CALIBRATE THIS: Y_MAPPING_SLOPE = (RobotY2 - RobotY1) / (v2 - v1)
        self.Y_MAPPING_INTERCEPT = -100.0 # Placeholder - CALIBRATE THIS: Y_MAPPING_INTERCEPT = RobotY1 - Y_MAPPING_SLOPE * v1

        self.gray_thresh = gray_thresh
        self.area_thresh = area_thresh
        self.cam_min_lim = cam_min_lim
        self.cam_max_lim = cam_max_lim
        
    def initialize_camera(self):
        """Initializes the camera and performs calibration."""
        self.helper = CameraCalibrationHelper()
        self.camera = self.helper.initialize_raspicam(headless = True, sensor_index = -1)
        self.helper.calibrate_raspberry()
        time.sleep(1)
        self.camera_available = True
        print("Camera initialized and calibrated.")
    
    def camera_thread(self):
        """
        Thread function for continuously capturing frames, detecting objects,
        and displaying the camera feed.
        """
        if not self.camera_available:
            print("Camera not available. Exiting camera thread.")
            return

        while True:
            frame = self.camera.capture_array()[:, :, 0:3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = self.helper.correct_image(frame)
            frame, mask, center, angle, success = self.detect_box(frame, self.gray_thresh,
                                                                  self.area_thresh, iter_ = 1)
            frame = cv2.rectangle(frame, self.cam_min_lim, self.cam_max_lim, (0, 0, 0), 10)
            cv2.imshow("Robot Camera", frame)
            cv2.imshow("Robot Camera mask", mask)
            
            if success:
                self.map_camara2robot(center, angle)
            else:
                self.object_detected = False 
            
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
              
          center = (int(center[0]) + self.cam_min_lim[0], int(center[1]) + self.cam_min_lim[1])
      
          # draw over frame
          box = cv2.boxPoints(rect).astype(int)
          box[:, 0] =  box[:, 0] + self.cam_min_lim[0]
          box[:, 1] =  box[:, 1] + self.cam_min_lim[1]
          frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2) 
          frame = cv2.circle(frame, center, 5, (255, 0, 0), 10)
          return frame, mask, center, angle, 1
        
    
    def map_camara2robot(self, center_coords, detected_angle):
        """
        Maps camera coordinates (cx, cy) to robot target (self.target_x, self.target_y).
        Uses calibrated linear mapping coefficients.
        """
        if center_coords is None:
            self.object_detected = False
            return

        cx, cy = center_coords
        u_cam_px = cx # camera x-pixel
        v_cam_px = cy # camera y-pixel

        # Calculate robot's target X 
        Xr = self.X_MAPPING_SLOPE * u_cam_px + self.X_MAPPING_INTERCEPT
        
        # Calculate robot's target Y
        Yr = self.Y_MAPPING_SLOPE * v_cam_px + self.Y_MAPPING_INTERCEPT

        self.target_x = Xr
        self.target_y = Yr
        self.piece_angle = detected_angle
        target_j6_deg = self.piece_angle + 180.0
        self.target_j6_rad = math.radians(target_j6_deg)
        
        self.object_detected = True
        
        print(f"[MAP] CamInput: u={u_cam_px}, v={v_cam_px}")
        print(f"[MAP] Robot Target → X={self.target_x:.1f} mm, Y={self.target_y:.1f} mm, Angle (cam): {self.piece_angle:.1f}°, Target Rz (rob): {target_j6_deg:.1f}° ({self.target_j6_rad:.3f} rad)")

    def mozaic_generator(self):
        """
        TODO:
        [INCOMPLETE FUNCTION]: It should generate the position of where the
        piece will be placed in the pallet.
        """
        # hint: do it with self.piece_num
        
        print("Mosaic generator called - currently a placeholder.")
        return None 
    
    def pick_and_place(self):
        if not self.object_detected:
            print("[PICK_AND_PLACE] No object detected to pick and place.")
            return
    
        print(f"[PICK_AND_PLACE] Target: X={self.target_x:.1f}, Y={self.target_y:.1f}, Angle: {self.piece_angle:.1f}")
    
        pick_z = -33       
        lift_z = -150      
        drop_y = 470       
        drop_x_offset = 0 
        
        # Gripper orientation:
        # TODO: Convert self.piece_angle (camera's view) to robot's Rz
        # This mapping is critical and depends on your coordinate system alignments.
        # For example, if robot's Rz=0 means gripper along robot's X-axis,
        # and camera's detected_angle=0 means piece's long side along camera's X-axis,
        # and camera's X-axis is aligned with robot's X-axis, then:
        # target_robot_rz = self.piece_angle
        # The previous fixed value was rz_orientation = 91.0
        rx_fixed, ry_fixed = -0.584, -1.702 
        target_robot_rz = 91.0 # Placeholder - MAKE THIS DYNAMIC based on self.piece_angle

        approach_pose = [self.target_x, self.target_y, lift_z, rx_fixed, ry_fixed, target_robot_rz]
        print(f"Moving to approach pose: {approach_pose}")
        self.robot.move_l_pose(np.array(approach_pose), speed=30, acc=30) 
        self.robot.wait_until_motion_complete()

        pick_pose = [self.target_x, self.target_y, pick_z, rx_fixed, ry_fixed, target_robot_rz]
        print(f"Moving to pick pose: {pick_pose}")
        self.robot.move_l_pose(np.array(pick_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()

        print("Closing gripper.")
        self.robot.close_gripper() 
        time.sleep(0.5)

        lift_pose = [self.target_x, self.target_y, lift_z, rx_fixed, ry_fixed, target_robot_rz]
        print(f"Moving to lift pose: {lift_pose}")
        self.robot.move_l_pose(np.array(lift_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()

        drop_approach_pose = [self.target_x + drop_x_offset, drop_y, lift_z, rx_fixed, ry_fixed, target_robot_rz] 
        print(f"Moving to drop approach pose: {drop_approach_pose}")
        self.robot.move_l_pose(np.array(drop_approach_pose), speed=30, acc=30)
        self.robot.wait_until_motion_complete()
        
        print("Opening gripper.")
        self.robot.open_gripper() 
        time.sleep(0.5)

        print(f"[PICK_AND_PLACE] Sequence complete for object at X={self.target_x:.1f}, Y={self.target_y:.1f}")
        self.object_detected = False 
        self.piece_num +=1
   
    def run(self):
        """Main execution loop for the robot."""
        
        if not self.camera_available:
            print("Error: Camera not initialized. Cannot start robot run sequence.")
            return

        cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
        cam_thread.start()
        print("Camera thread started.")

        if self.robot.connect():
            print("Successfully connected to robot.")
            self.robot.set_servo_status(1) 

            previous_sensor_state_is_detecting = None 
            SENSOR_ADDRESS = 915 
            
            try:
                while True:
                    if not cam_thread.is_alive():
                        print("Error: Camera thread has stopped. Exiting main loop.")
                        break

                    success_read, sensor_value, _ = self.robot.send_cmd("getVirtualOutput", {"addr": SENSOR_ADDRESS})
                    current_sensor_state_is_detecting = None 

                    if success_read:
                        sensor_value = int(sensor_value) if sensor_value is not None else -1
                        if sensor_value == 1: 
                            current_sensor_state_is_detecting = True
                        elif sensor_value == 0: 
                            current_sensor_state_is_detecting = False
                        else:
                            print(f"Warning: Unexpected sensor value ({sensor_value}) at address {SENSOR_ADDRESS}")
                    else:
                        print(f"Warning: Failed to read virtual output at address {SENSOR_ADDRESS}.")
                    
                    if current_sensor_state_is_detecting is not None:
                        if current_sensor_state_is_detecting != previous_sensor_state_is_detecting:
                            if current_sensor_state_is_detecting is True:
                                print("Sensor: Object detected at pick-up point.")
                                # Wait a brief moment for camera to process and set self.object_detected
                                time.sleep(0.1) # Small delay to allow camera thread to update flag
                                if self.object_detected:
                                    print("Camera has also detected an object. Initiating pick and place.")
                                    self.pick_and_place()
                                else:
                                    print("Sensor detected object, but camera has not confirmed/mapped yet.")
                            else: 
                                print("Sensor: No object at pick-up point / object removed.")
                            
                            previous_sensor_state_is_detecting = current_sensor_state_is_detecting
                    
                    time.sleep(0.2) # Polling interval for the sensor

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Shutting down.")
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
            finally:
                print("Disconnecting from robot.")
                if self.robot.sock: # Check if socket is still open
                    self.robot.set_servo_status(0) 
                    self.robot.disconnect()
        else:
          print("Failed to connect to the robot.")

        if cam_thread.is_alive():
            print("Waiting for camera thread to join...") 
            # To ensure clean exit, signal camera thread to stop if it hasn't already (e.g., via a flag)
            # For now, just join with a timeout.
            cam_thread.join(timeout=1.0) 

        print("Palletizing robot application finished.")


if __name__ == "__main__":
    robot_ip = "169.168.0.200" # Replace with your robot's IP
    
    # Define ROI for camera processing
    camera_roi_min = (50, 50)   
    camera_roi_max = (600, 400) 

    robot_controller = PalletizingRobot(robot_ip, 
                                        cam_min_lim=camera_roi_min, 
                                        cam_max_lim=camera_roi_max)
    
    print("Initializing camera...")
    robot_controller.initialize_camera()
    
    if robot_controller.camera_available:
        print("Camera initialized successfully.")
        robot_controller.run()
    else:
        print("Failed to initialize camera. Exiting.")

