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
        self.target_j6_deg  = 0.0

        # Affine transformation coefficients for X-coordinate
        # X_robot = X_MAPPING_SLOPE * x_cam_pixel + X_MAPPING_INTERCEPT
        self.X_MAPPING_SLOPE = 0.5208
        self.X_MAPPING_INTERCEPT = -132.57

        # Affine transformation coefficients for Y-coordinate
        # Y_robot = Y_MAPPING_SLOPE * y_cam_pixel + Y_MAPPING_INTERCEPT
        # TODO: Need to calibrate these values!
        self.Y_MAPPING_SLOPE = 0.5  # Placeholder - CALIBRATE THIS: Y_MAPPING_SLOPE = (RobotY2 - RobotY1) / (v2 - v1)
        self.Y_MAPPING_INTERCEPT = -100.0 # Placeholder - CALIBRATE THIS: Y_MAPPING_INTERCEPT = RobotY1 - Y_MAPPING_SLOPE * v1
        
        self.PICK_Z_CONVEYOR = -33.0      # Actual Z height for picking from conveyor
        self.LIFT_Z_COMMON = -150.0       # Common Z height for approach, lift, and retreat
        
        self.NOMINAL_RX_DEG = -0.584
        self.NOMINAL_RY_DEG = -1.702
        
        self.ANGLE_CLASSIFICATION_THRESHOLD_DEG = 20.0

        self.piece_count_zone_0_deg = 0
        self.piece_count_zone_90_deg = 0

        # Base ROBOT CARTESIAN coordinates for palletizing zones (mm and degrees)
        self.PALLET_ZONE_0_BASE_X = 300.0  # Robot X for corner/start of zone 0
        self.PALLET_ZONE_0_BASE_Y = -200.0 # Robot Y for corner/start of zone 0
        self.PALLET_ZONE_0_PLACE_RZ_DEG = 0.0 # Desired Robot Rz on pallet for 0-deg type pieces

        self.PALLET_ZONE_90_BASE_X = 300.0  # Robot X for corner/start of zone 90
        self.PALLET_ZONE_90_BASE_Y = 200.0  # Robot Y for corner/start of zone 90
        self.PALLET_ZONE_90_PLACE_RZ_DEG = 90.0 # Desired Robot Rz on pallet for 90-deg type pieces

        # Mosaic parameters (dimensions in mm)
        self.ITEMS_PER_ROW = 3 # Example: 3 items per row on pallet
        self.PHYSICAL_WIDTH_MM = 90.0  # Physical shorter side of the piece (mm)
        self.PHYSICAL_HEIGHT_MM = 140.0 # Physical longer side of the piece (mm)
        self.ITEM_GAP_MM = 5.0 # Gap between items on pallet

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
        self.target_j6_deg = self.piece_angle + 180.0
        
        self.object_detected = True
        
        print(f"[MAP] CamInput: u={u_cam_px}, v={v_cam_px}")
        print(f"[MAP] Robot Target → X={self.target_x:.1f} mm, Y={self.target_y:.1f} mm, Angle (cam): {self.piece_angle:.1f}°, Target Rz (rob): {self.target_j6_deg:.1f}°")

    def mozaic_generator(self, zone_type, piece_index_in_zone):
        """
        Calculates target ROBOT CARTESIAN pose (X, Y, Z, Rz) for placing a piece.
        Returns: (target_x, target_y, target_z, target_rz_on_pallet_deg) or None
        """
        print(f"Mozaic generator for zone: {zone_type}, piece index: {piece_index_in_zone}")
        
        target_x, target_y, target_rz_on_pallet_deg = None, None, None
        target_z = self.PLACE_Z_ON_PALLET # Z is fixed for placing on this layer

        col = piece_index_in_zone % self.ITEMS_PER_ROW
        row = piece_index_in_zone // self.ITEMS_PER_ROW

        if zone_type == "0_deg_type":
            base_x = self.PALLET_ZONE_0_BASE_X
            base_y = self.PALLET_ZONE_0_BASE_Y
            target_rz_on_pallet_deg = self.PALLET_ZONE_0_PLACE_RZ_DEG
            # Assuming for 0-deg type, PHYSICAL_WIDTH_MM aligns with X-spacing, PHYSICAL_HEIGHT_MM with Y-spacing
            spacing_x = self.PHYSICAL_WIDTH_MM + self.ITEM_GAP_MM
            spacing_y = self.PHYSICAL_HEIGHT_MM + self.ITEM_GAP_MM
        elif zone_type == "90_deg_type":
            base_x = self.PALLET_ZONE_90_BASE_X
            base_y = self.PALLET_ZONE_90_BASE_Y
            target_rz_on_pallet_deg = self.PALLET_ZONE_90_PLACE_RZ_DEG
            # Assuming for 90-deg type, PHYSICAL_HEIGHT_MM aligns with X-spacing (piece is rotated)
            spacing_x = self.PHYSICAL_HEIGHT_MM + self.ITEM_GAP_MM
            spacing_y = self.PHYSICAL_WIDTH_MM + self.ITEM_GAP_MM
        else:
            print(f"Error: Unknown zone_type '{zone_type}' in mozaic_generator.")
            return None, None, None, None

        target_x = base_x + col * spacing_x
        target_y = base_y + row * spacing_y
            
        # TODO: Implement logic to check if pallet zone is full based on rows/cols capacity
        # max_rows = 2 # Example
        # if row >= max_rows:
        #    print(f"Pallet zone {zone_type} is full.")
        #    return None, None, None, None
            
        print(f"  Mosaic for {zone_type}, Idx {piece_index_in_zone}: Col {col}, Row {row} -> X {target_x:.1f}, Y {target_y:.1f}, Z {target_z:.1f}, RzPlt {target_rz_on_pallet_deg:.1f}")
        return target_x, target_y, target_z, target_rz_on_pallet_deg

    def _pick_from_conveyor(self, pick_x, pick_y, target_j6_deg):
        """
        Commands the robot to pick an object from (pick_x, pick_y) on the conveyor.
        Uses self.target_j6_deg for final tool orientation via a joint move.
        Uses class attributes for Z heights.

        Args:
            pick_x (float): Target X coordinate for picking (center of object).
            pick_y (float): Target Y coordinate for picking (center of object).

        Returns:
            bool: True if pick sequence is successful (commands sent), False otherwise.
        """
        print(f"Executing pick at X:{pick_x:.1f}, Y:{pick_y:.1f} (J6 target: {target_j6_deg:.1f} deg)")

        # 1. Move to an approach position (X, Y, LIFT_Z_COMMON)
        #    Use a nominal Rz for this initial approach, as J6 will be set precisely next.
        #    If your robot tends to "wind up" J6, a more neutral initial Rz might be better.
        #    For simplicity, let's use a nominal Rz, e.g., 0 or an Rz close to the expected target.
        #    The previously calculated `self.piece_angle + 180.0` can be used here.
        initial_rz_deg = target_j6_deg # TODO: Check if value is correct

        approach_pose_cartesian = [pick_x, pick_y, self.LIFT_Z_COMMON,
                                   self.NOMINAL_RX_DEG, self.NOMINAL_RY_DEG, initial_rz_deg]
        print(f"  1. Moving to approach (Cartesian): {approach_pose_cartesian}")
        success, _, _ = self.robot.move_l_pose(np.array(approach_pose_cartesian), speed=30, acc=30)
        if not success:
            print("  Error: Failed to move to Cartesian approach pose.")
            return False
        self.robot.wait_until_motion_complete()

        # 2. Get current joint positions (robot is now at the X,Y,Z_lift, with initial orientation)
        success_joints, current_joints_deg, _ = self.robot.get_current_joints()
        if not success_joints:
            print("  Error: Failed to get current joint positions.")
            return False
        print(f"  2. Current joints (deg): {np.round(current_joints_deg, 2).tolist()}")

        # 3. Prepare target joint array for J6 orientation
        target_joints_deg = np.array(current_joints_deg) # Make a copy
        
        # Normalize target_j6_deg to be within typical robot joint limits if necessary
        # (e.g., -180 to 180 or -360 to 360, depending on robot's J6 range)
        # Example: target_j6_deg = (target_j6_deg + 180) % 360 - 180
        target_joints_deg[5] = target_j6_deg # Joint 6 is at index 5

        print(f"  3. Orienting J6 to {target_j6_deg:.1f} deg. Target joints: {np.round(target_joints_deg,2).tolist()}")
        success, _, _ = self.robot.move_j_joint(target_joints_deg, speed=20, acc=20) # Speed/acc for joint move
        if not success:
            print("  Error: Failed to orient Joint 6.")
            return False
        self.robot.wait_until_motion_complete()
        # After this move, the TCP X,Y,Z might have slightly changed if J1-J5 weren't perfectly held by the kinematic solution.
        # For precise downward motion, it's best to get the current TCP again.

        # 4. Get current TCP pose after J6 orientation for precise linear downward move
        success_tcp, tcp_after_j6_orientation, _ = self.robot.get_tool_pose_in_base_coords()
        if not success_tcp:
            print("  Error: Failed to get TCP pose after J6 orientation.")
            return False
        print(f"  4. TCP after J6 orient (deg): {np.round(tcp_after_j6_orientation,2).tolist()}")
        
        # 5. Move down to actual pick position (linearly)
        #    Use the X, Y, Rx, Ry, Rz from tcp_after_j6_orientation, only change Z.
        pick_pose_cartesian = [tcp_after_j6_orientation[0], tcp_after_j6_orientation[1], self.PICK_Z_CONVEYOR,
                               tcp_after_j6_orientation[3], tcp_after_j6_orientation[4], tcp_after_j6_orientation[5]]
        print(f"  5. Moving down to pick (Cartesian): {pick_pose_cartesian}")
        success, _, _ = self.robot.move_l_pose(np.array(pick_pose_cartesian), speed=10, acc=10) # Slower for precision
        if not success:
            print("  Error: Failed to move to actual pick pose.")
            return False
        self.robot.wait_until_motion_complete()

        # 6. Close gripper
        print("  6. Closing gripper.")
        self.robot.close_gripper()
        time.sleep(0.7)

        # 7. Lift the object (linearly)
        #    Use the X, Y, Rx, Ry, Rz from the pick pose, only change Z.
        lift_pose_cartesian = [tcp_after_j6_orientation[0], tcp_after_j6_orientation[1], self.LIFT_Z_COMMON,
                               tcp_after_j6_orientation[3], tcp_after_j6_orientation[4], tcp_after_j6_orientation[5]]
        print(f"  7. Lifting object (Cartesian): {lift_pose_cartesian}")
        success, _, _ = self.robot.move_l_pose(np.array(lift_pose_cartesian), speed=20, acc=20)
        if not success:
            print("  Error: Failed to lift object.")
            return False
        self.robot.wait_until_motion_complete()

        print("  Pick sequence successfully completed.")
        return True
    
    def _classify_piece_orientation(self):
        """
        Classifies the piece's orientation based on self.piece_angle.
        self.piece_angle is the raw detected angle from the camera in degrees.

        Returns:
            str: "0_deg_type", "90_deg_type", or "unclassified".
        """
        # Ensure self.piece_angle is within a known range if necessary,
        # e.g., normalize it to -180 to 180 if it can go beyond.
        # Assuming self.piece_angle from detect_box is reasonably constrained (e.g., -90 to +90, or 0 to 180 after adjustments)

        # Check for 0-degree type orientation
        if abs(self.piece_angle) < self.ANGLE_CLASSIFICATION_THRESHOLD_DEG:
            return "0_deg_type"
        # Check for 90-degree type orientation (handles +90 and -90)
        elif abs(abs(self.piece_angle) - 90.0) < self.ANGLE_CLASSIFICATION_THRESHOLD_DEG:
            return "90_deg_type"
        else:
            print(f"[CLASSIFY] Piece angle {self.piece_angle:.1f}° is unclassified (threshold: +/-{self.ANGLE_CLASSIFICATION_THRESHOLD_DEG}°).")
            return "unclassified"
    
    def pick_and_place(self):
        if not self.object_detected:
            print("[PICK_AND_PLACE] No object detected to pick and place.")
            return # Or return False

        # self.target_x, self.target_y, and self.target_j6_deg are set by map_camara2robot
        print(f"[PICK_AND_PLACE] Initiating pick for object at X={self.target_x:.1f}, Y={self.target_y:.1f} (J6 target: {self.target_j6_deg:.1f} deg)")

        # Call the revised pick helper function
        pick_successful = self._pick_from_conveyor(self.target_x, self.target_y, self.target_j6_deg)

        if not pick_successful:
            print("[PICK_AND_PLACE] Pick operation failed. Aborting.")
            self.object_detected = False # Reset flag
            return # Or return False

        self.object_detected = False 
        self.piece_num += 1 
        print("[PICK_AND_PLACE] Pick successful. Placement logic to follow.")
        return
   
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

