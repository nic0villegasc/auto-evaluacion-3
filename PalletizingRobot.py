import numpy as np
import sys
import time
import cv2
from SDK import ELITE # Assuming SDK.py is in the same directory or accessible
from CameraCalibration import CameraCalibrationHelper # Assuming CameraCalibration.py is accessible
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
        self.helper = None # Initialize helper
        self.camera_available = False # Initialize camera_available

        self.piece_num = 0 # number of the piece that is being picked
        self.object_detected = False
        self.target_x = 0.0 # Initialize target coordinates
        self.target_y = 0.0 # Initialize target coordinates
        self.piece_angle = 0.0 # Initialize piece_angle
        self.robot_x = 0.0 # Initialize robot_x from the original map_camara2robot

        self.gray_thresh = gray_thresh
        self.area_thresh = area_thresh
        self.cam_min_lim = cam_min_lim
        self.cam_max_lim = cam_max_lim
        
    def initialize_camera(self):
        """Initializes the camera and performs calibration."""
        self.helper = CameraCalibrationHelper()
        self.camera = self.helper.initialize_raspicam(headless = True, sensor_index = -1) # Assuming sensor_index -1 is valid
        self.helper.calibrate_raspberry()
        time.sleep(1) # Wait for camera to stabilize
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
            try:
                frame_data = self.camera.capture_array()
                if frame_data is None:
                    print("Failed to capture frame.")
                    time.sleep(0.1) # Wait a bit before retrying
                    continue

                # Ensure frame_data has at least 3 channels before slicing
                if frame_data.ndim < 3 or frame_data.shape[2] < 3:
                    print(f"Captured frame has unexpected shape: {frame_data.shape}")
                    # Convert to BGR if it's grayscale or has fewer than 3 channels
                    if frame_data.ndim == 2 : # Grayscale
                         frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                    elif frame_data.shape[2] == 1: # Single channel
                         frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                    else: # Default to black if cannot convert
                        frame_data = np.zeros((self.cam_max_lim[1], self.cam_max_lim[0], 3), dtype=np.uint8)


                frame = frame_data[:, :, 0:3] # Take RGB channels
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if self.helper:
                    frame = self.helper.correct_image(frame)
                
                processed_frame, mask, center_coords, detected_angle, success = self.detect_box(
                    frame.copy(), # Pass a copy to avoid modifying the original frame being displayed
                    self.gray_thresh,
                    self.area_thresh,
                    iter_ = 1
                )
                
                if success:
                    # If an object is detected, map its coordinates
                    self.map_camara2robot(center_coords, detected_angle)
                else:
                    # If no object is detected, ensure the flag is reset
                    self.object_detected = False 

                # Draw ROI rectangle on the frame to be displayed
                display_frame = cv2.rectangle(processed_frame, self.cam_min_lim, self.cam_max_lim, (0, 0, 0), 10)
                cv2.imshow("Robot Camera", display_frame)
                if mask is not None:
                    cv2.imshow("Robot Camera mask", mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting camera thread.")
                    break
            except Exception as e:
                print(f"Error in camera_thread: {e}")
                # Consider adding a small delay or specific error handling
                time.sleep(1) # Wait a bit if an error occurs
                # break # Optionally break the loop on critical errors


        cv2.destroyAllWindows() # Clean up OpenCV windows when loop exits
    
    def detect_box(self, frame, gray_thresh, area_thresh, iter_=1):
        """
        Detects a rectangular object within a defined ROI of the frame.
        Returns the processed frame, mask, center coordinates (cx, cy) in full frame,
        angle, and a success flag.
        The angle of the wood piece is in the range of (-90, 90) degrees.
        """
        # Ensure ROI coordinates are valid
        y_min, y_max = self.cam_min_lim[1], self.cam_max_lim[1]
        x_min, x_max = self.cam_min_lim[0], self.cam_max_lim[0]

        if not (0 <= y_min < y_max <= frame.shape[0] and 0 <= x_min < x_max <= frame.shape[1]):
            print(f"Invalid ROI coordinates: {self.cam_min_lim}, {self.cam_max_lim} for frame shape {frame.shape}")
            return frame, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8), None, None, 0


        aux = frame[y_min:y_max, x_min:x_max]
        
        # Grayscale detection
        gray_image = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)
        
        # Mask thresh
        _, mask = cv2.threshold(gray_image, gray_thresh, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5,5),np.uint8) # Define a kernel for morphology, can be adjusted
        mask = cv2.erode(mask, kernel, iterations=iter_)
        mask = cv2.dilate(mask, kernel, iterations=iter_)
        
        # Find contour with largest area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        full_frame_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        full_frame_mask[y_min:y_max, x_min:x_max] = mask


        if not contours:
            return frame, full_frame_mask, None, None, 0
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < area_thresh:
            return frame, full_frame_mask, None, None, 0
        
        # Detect square position and orientation 
        rect = cv2.minAreaRect(largest_contour)
        center_roi, (width, height), angle = rect # center_roi is relative to the ROI 'aux'
        
        # Adjust angle to be consistently for the longer side, if necessary
        if width < height: # Assuming width is along the x-axis of the rect, height along y-axis
            angle += 90
            # Swap width and height if you want 'width' to always be the longer dimension
            # width, height = height, width 
        
        # Adjust center coordinates from ROI to full frame
        cx = int(center_roi[0]) + x_min
        cy = int(center_roi[1]) + y_min
        center_full_frame = (cx, cy)
        
        print(f"[DETECT_BOX] Camera → x={cx}, y={cy}, angle={angle:.1f}°")
        
        # Draw detection on the frame (for visualization)
        box = cv2.boxPoints(rect) # Gets corners relative to ROI
        box = np.int0(box)
        box[:, 0] += x_min # Adjust x-coordinates of box points
        box[:, 1] += y_min # Adjust y-coordinates of box points
        
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2) 
        cv2.circle(frame, center_full_frame, 5, (255, 0, 0), -1) # Use -1 for filled circle
        
        return frame, full_frame_mask, center_full_frame, angle, 1
        
    
    def map_camara2robot(self, center_coords, detected_angle):
        """
        Maps the camera coordinates (cx, cy) and detected_angle of the piece
        to the desired target position (self.target_x, self.target_y) for the robot.
        It also sets self.object_detected to True.
        The original calculation for self.robot_x based on angle is kept for now.
        """
        if center_coords is None:
            self.object_detected = False
            return

        cx, cy = center_coords
        u, v = cx, cy

        # Affine transformation coefficients (these might need calibration)
        a = 0.5208
        b = -132.57 
        # c = 0.098  # tx - currently unused in direct Xr, Yr calculation
        # f = -52.005764 # ty - currently unused

        # Calculate robot's target X and Y using the affine transformation
        Xr = a * u + b
        Yr = -34.796  # Yr is currently hardcoded

        self.target_x = Xr
        self.target_y = Yr
        self.object_detected = True
        print(f"[MAP] Robot Target → X={self.target_x:.1f} mm, Y={self.target_y:.1f} mm")

        # --- Original calculations from the provided map_camara2robot ---
        # This part calculates self.piece_angle and self.robot_x.
        # It seems to be a separate calculation, possibly for an offset or a different target.
        # If self.robot_x was intended to be the primary X target, integrate accordingly.
        self.piece_angle = 90 - detected_angle # Using the angle from detect_box
        
        # Mean dimensions of the wood piece (example values)
        piece_width = 90.0  
        piece_height = 140.0 
        
        beta = np.arctan2(piece_height, piece_width) # Use arctan2 for angle in correct quadrant
        L = np.sqrt((piece_width / 2)**2 + (piece_height / 2)**2)
        
        # Adjust angle for calculation if necessary (original logic)
        angle_rad_for_calc = -self.piece_angle * (np.pi / 180) # Convert to radians, ensure consistent sign
        
        # This calculates an 'x' value based on the piece's angle and dimensions.
        # Its exact purpose in relation to self.target_x needs clarification from the original design.
        self.robot_x = L * np.sin(np.pi + angle_rad_for_calc - beta) 
        # print(f"[MAP] Calculated self.robot_x (offset/alternative?): {self.robot_x:.1f} mm")

        # The following were hints/placeholders in the original function
        # self.robot_y_lims = None
        # self.camera_x_center_lims = None
        # self.robot_y = None # Note: self.target_y is now set by the affine transformation
        # self.robot_angle = None # This might be where self.piece_angle or detected_angle is stored if needed globally
        

    def mozaic_generator(self):
        """
        [INCOMPLETE FUNCTION]: It should generate the position of where the
        piece will be placed in the pallet.
        """
        # hint: do it with self.piece_num
        print("Mosaic generator called - currently a placeholder.")
        return None # Placeholder
    
    def pick_and_place(self):
        """
        Commands the robot to pick up an object from self.target_x, self.target_y
        and place it at a predefined drop-off location.
        """
        if not self.object_detected:
            print("[PICK_AND_PLACE] No object detected to pick and place.")
            return
    
        print(f"[PICK_AND_PLACE] Target: X={self.target_x:.1f}, Y={self.target_y:.1f}")
    
        # Define pick-up and drop-off parameters
        # pick_z is the height for approaching and picking the object
        pick_z = -33       
        # lift_z is the height to lift the object to after picking
        lift_z = -150      
        # drop_y is the Y-coordinate for the drop-off location
        drop_y = 470       
        # drop_z is the Z-coordinate for placing the object at the drop-off
        # (could be same as lift_z or different)
        drop_x_offset = 0 # Example: if drop X depends on target_x or is fixed
        
        # Fixed orientation for the gripper
        rx, ry, rz_orientation = -0.584, -1.702, 91.0  

        # 1. Move to approach position above the object
        approach_pose = [self.target_x, self.target_y, lift_z, rx, ry, rz_orientation]
        print(f"Moving to approach pose: {approach_pose}")
        # self.robot.move_l_pose(np.array(approach_pose), speed=30, acc=30) # Higher speed for travel
        # self.robot.wait_until_motion_complete()

        # 2. Move down to pick position
        pick_pose = [self.target_x, self.target_y, pick_z, rx, ry, rz_orientation]
        print(f"Moving to pick pose: {pick_pose}")
        self.robot.move_l_pose(np.array(pick_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()
        
        # 3. Close gripper
        print("Closing gripper.")
        self.robot.close_gripper() # Assuming this function handles the delay
        
        # 4. Lift the object
        lift_pose = [self.target_x, self.target_y, lift_z, rx, ry, rz_orientation]
        print(f"Moving to lift pose: {lift_pose}")
        self.robot.move_l_pose(np.array(lift_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()
        
        # 5. Move to drop-off location (above)
        # Drop X can be same as target_x, or a fixed/calculated pallet position
        # For now, using target_x for drop, but ideally this comes from mozaic_generator
        drop_approach_pose = [self.target_x + drop_x_offset, drop_y, lift_z, rx, ry, rz_orientation] 
        print(f"Moving to drop approach pose: {drop_approach_pose}")
        self.robot.move_l_pose(np.array(drop_approach_pose), speed=30, acc=30)
        self.robot.wait_until_motion_complete()

        # 6. Move down to place object (if needed, or open gripper at lift_z)
        # drop_place_pose = [self.target_x + drop_x_offset, drop_y, pick_z, rx, ry, rz_orientation] # Example if placing lower
        # print(f"Moving to drop place pose: {drop_place_pose}")
        # self.robot.move_l_pose(np.array(drop_place_pose), speed=20, acc=20)
        # self.robot.wait_until_motion_complete()

        # 7. Open gripper
        print("Opening gripper.")
        self.robot.open_gripper() # Assuming this function handles the delay

        # 8. Optionally, retract to a safe position
        # retract_pose = [self.target_x + drop_x_offset, drop_y, lift_z, rx, ry, rz_orientation]
        # print(f"Moving to retract pose: {retract_pose}")
        # self.robot.move_l_pose(np.array(retract_pose), speed=30, acc=30)
        # self.robot.wait_until_motion_complete()

        print(f"[PICK_AND_PLACE] Sequence complete for object at X={self.target_x:.1f}, Y={self.target_y:.1f}")
        self.object_detected = False # Reset flag after pick and place
        self.piece_num +=1 # Increment piece count
   
    def run(self):
        """Main execution loop for the robot."""
        # Start the camera thread
        cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
        cam_thread.start()

        if self.robot.connect():
            print("Successfully connected to robot.")
            self.robot.set_servo_status(1) # Turn servos on

            previous_sensor_state_is_detecting = None 
            SENSOR_ADDRESS = 915 # Virtual output address for the sensor
            
            try:
                while True:
                    # Read sensor state (virtual output from robot)
                    # Assuming send_cmd returns (success_flag, result_value, id)
                    success_read, sensor_value, _ = self.robot.send_cmd("getVirtualOutput", {"addr": SENSOR_ADDRESS})
                    
                    current_sensor_state_is_detecting = None # Reset for current iteration

                    if success_read:
                        if sensor_value == 1: # Assuming 1 means object present / sensor high
                            current_sensor_state_is_detecting = True
                        elif sensor_value == 0: # Assuming 0 means no object / sensor low
                            current_sensor_state_is_detecting = False
                        else:
                            print(f"Warning: Unexpected sensor value ({sensor_value}) at address {SENSOR_ADDRESS}")
                    else:
                        print(f"Warning: Failed to read virtual output at address {SENSOR_ADDRESS}.")
                    
                    if current_sensor_state_is_detecting is not None:
                        if current_sensor_state_is_detecting != previous_sensor_state_is_detecting:
                            if current_sensor_state_is_detecting is True:
                                print("Sensor: Object detected at pick-up point.")
                                # Object is present, and if camera has also detected and mapped it, pick and place
                                if self.object_detected:
                                    print("Camera has also detected an object. Initiating pick and place.")
                                    self.pick_and_place()
                                else:
                                    print("Sensor detected object, but camera has not confirmed/mapped yet.")
                            else: # Sensor is low (false)
                                print("Sensor: No object at pick-up point / object removed.")
                            
                            previous_sensor_state_is_detecting = current_sensor_state_is_detecting
                    
                    time.sleep(0.5) # Polling interval for the sensor

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Shutting down.")
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
            finally:
                print("Disconnecting from robot.")
                self.robot.set_servo_status(0) # Turn servos off
                self.robot.disconnect()
        else:
            print("Failed to connect to the robot.")

        if cam_thread.is_alive():
             print("Waiting for camera thread to join...") # May need a signal to stop cam_thread gracefully
             # For a clean exit, you might need a global flag to signal the camera_thread to stop.
             # cam_thread.join() # This might block indefinitely if not handled carefully

        print("Palletizing robot application finished.")


if __name__ == "__main__":
    robot_ip_address = "169.168.0.200" # Make sure this is the correct IP
    
    # Initialize the robot system
    palletizer = PalletizingRobot(robot_ip_address)
    
    # Initialize the camera (this will also run calibration)
    palletizer.initialize_camera()
    
    # Run the main robot logic
    if palletizer.camera_available: # Only run if camera initialized successfully
        palletizer.run()
    else:
        print("Could not start robot run sequence because camera failed to initialize.")