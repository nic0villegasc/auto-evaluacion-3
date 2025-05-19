import numpy as np
import sys
import time
import cv2
from SDK import ELITE
from CameraCalibration import CameraCalibrationHelper
import threading
import math
import queue

class PalletizingRobot:

    def __init__(self, robot_ip, gray_thresh = 100, area_thresh = 45000, 
                 cam_min_lim = (0, 0), cam_max_lim = (640, 480), step_mode=True, max_queue_size=10):
        self.robot = ELITE(robot_ip)
        self.frame = None
        self.camera = None
        self.camera_available = False 
        self.helper = None 
        
        self.step_by_step_enabled = step_mode 

        self.piece_num = 0 
        self.object_detected = False
        self.object_queue = queue.Queue(maxsize=max_queue_size)
        self.last_queued_object_props = None
        self.active_object_props_in_view = None

        self.target_x = 0.0 
        self.target_y = 0.0 
        self.piece_angle = 0.0
        self.target_j6_deg  = 0.0

        # Affine transformation coefficients for X-coordinate
        # X_robot = X_MAPPING_SLOPE * x_cam_pixel + X_MAPPING_INTERCEPT
        self.X_MAPPING_SLOPE = 0.5208
        self.X_MAPPING_INTERCEPT = -148.57

        # Affine transformation coefficients for Y-coordinate
        # Y_robot = Y_MAPPING_SLOPE * y_cam_pixel + Y_MAPPING_INTERCEPT
        self.Y_MAPPING_SLOPE = -0.5455
        self.Y_MAPPING_INTERCEPT = 49.53
        
        self.ANG_MAPPING_SLOPE =	-0.5140
        self.ANG_MAPPING_INTERCEPT = 81.93
        
        self.PICK_Z_CONVEYOR = -33.0      # Actual Z height for picking from conveyor
        self.LIFT_Z_COMMON = -300.0       # Common Z height for approach, lift, and retreat
        self.PLACE_Z_PALLET_SURFACE = 70
        
        self.NOMINAL_RX_DEG = -0.584
        self.NOMINAL_RY_DEG = -1.702
        self.NOMINAL_RZ_DEG = 91.0
        
        self.FIXED_RX_DEG = self.NOMINAL_RX_DEG  # TODO: Place holder
        self.FIXED_RY_DEG = self.NOMINAL_RY_DEG  # TODO: Place holder
        
        self.ANGLE_CLASSIFICATION_THRESHOLD_DEG = 20.0

        self.piece_count_zone_0_deg = 0
        self.piece_count_zone_90_deg = 0

        # Base ROBOT CARTESIAN coordinates for palletizing zones (mm and degrees)
        self.PALLET_ZONE_0_BASE_X = 128  # Robot X for corner/start of zone 0
        self.PALLET_ZONE_0_BASE_Y = 581 # Robot Y for corner/start of zone 0
        self.PALLET_ZONE_0_PLACE_RZ_DEG = 0

        self.PALLET_ZONE_90_BASE_X = 665  # Robot X for corner/start of zone 90
        self.PALLET_ZONE_90_BASE_Y = 581  # Robot Y for corner/start of zone 90
        self.PALLET_ZONE_90_PLACE_RZ_DEG = 0

        # Mosaic parameters (dimensions in mm)
        self.ITEMS_PER_ROW = 2
        self.ITEMS_PER_Z_LAYER_PATTERN = 2
        self.MAX_Z_LAYERS_PER_ZONE = 3
        self.PHYSICAL_WIDTH_MM = 95.0  # Physical shorter side of the piece (mm)
        self.PHYSICAL_HEIGHT_MM = 150.0 # Physical longer side of the piece (mm)
        self.PHYSICAL_THICKNESS_MM = 90.0
        self.ITEM_GAP_MM = 30.0 # Gap between items on pallet
        
        self.BASE_Z_FOR_FIRST_LAYER = self.PLACE_Z_PALLET_SURFACE - self.PHYSICAL_THICKNESS_MM / 2.0
        
        self.PALLET_PRIMARY_RZ_DEG = 90.0
        self.PALLET_SECONDARY_RZ_DEG = 0.0

        self.gray_thresh = gray_thresh
        self.area_thresh = area_thresh
        self.cam_min_lim = cam_min_lim
        self.cam_max_lim = cam_max_lim
        
        self.NEW_OBJECT_X_DIFF_THRESHOLD = 30  # Min change in X (pixels) to be a new object
        self.NEW_OBJECT_Y_DIFF_THRESHOLD = 30  # Min change in Y (pixels)
        self.NEW_OBJECT_ANGLE_DIFF_THRESHOLD = 30  # Min change in angle (degrees)
        
        self.STANDARD_POSES = {
            "initial_neutral_conveyor": {
                "coords": [0.0, 0.0, self.LIFT_Z_COMMON, self.NOMINAL_RX_DEG, self.NOMINAL_RY_DEG, self.NOMINAL_RZ_DEG],
                "description": "initial neutral pose above conveyor (X=0, Y=0)",
                "speed": 60, 
                "acc": 30
            },
            "reject_bin":{
              "coords": [-415, 142, -144, self.NOMINAL_RX_DEG, self.NOMINAL_RY_DEG, self.NOMINAL_RZ_DEG],
                "description": "initial neutral pose above conveyor (X=0, Y=0)",
                "speed": 60, 
                "acc": 30
            }
        }
        
        self._stop_event = threading.Event()
        
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
            frame, mask, center, angle, width, height, success = self.detect_box(frame, self.gray_thresh,
                                                                  self.area_thresh, iter_ = 1)
            frame = cv2.rectangle(frame, self.cam_min_lim, self.cam_max_lim, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Queue: {self.object_queue.qsize()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Robot Camera", frame)
            cv2.imshow("Robot Camera mask", mask)
            
            if success:
                current_detected_props = {
                    'center_x': center[0],
                    'center_y': center[1],
                    'angle': angle,
                    'width': width,
                    'height': height
                }
                should_queue_object = False

                if self.active_object_props_in_view is None:
                    # CASE 1: No object was previously "active" in view. This is genuinely new.
                    should_queue_object = True
                    print(f"[CAMERA_THREAD] First detection of an object. Queuing. Props: {current_detected_props}")
                else:
                    # CASE 2: An object was already "active". Check if this is a *different* object.
                    last_active_props = self.active_object_props_in_view
                    
                    delta_x = abs(current_detected_props['center_x'] - last_active_props['center_x'])
                    delta_y = abs(current_detected_props['center_y'] - last_active_props['center_y'])
                    abs_angle_diff = abs(current_detected_props['angle'] - last_active_props['angle'])
                    sym_angle_diff = min(abs_angle_diff, 180.0 - abs_angle_diff)
                    
                    if (delta_x > self.NEW_OBJECT_X_DIFF_THRESHOLD or
                        delta_y > self.NEW_OBJECT_Y_DIFF_THRESHOLD or
                        sym_angle_diff > self.NEW_OBJECT_ANGLE_DIFF_THRESHOLD):
                        # The currently detected object is significantly different from the
                        # one previously considered active. This implies the old one left and this is new.
                        should_queue_object = True
                        print(f"[CAMERA_THREAD] Detected object is different from previous active. Queuing. New: {current_detected_props}, Prev Active: {last_active_props}")
                    # else: Properties are similar to the active object. It's likely the same one,
                    #       still in view (even if stationary). Do NOT re-queue.
                
                if should_queue_object:
                    self.map_camara2robot(center, angle) 
                    self.active_object_props_in_view = current_detected_props # Set/Update the active object

            else: 
                # No object detected in the current frame (success == False)
                if self.active_object_props_in_view is not None:
                    # An object *was* active, but is now no longer seen.
                    # This means it has left the view, so clear the active state.
                    print(f"[CAMERA_THREAD] Active object ({self.active_object_props_in_view}) no longer detected. Clearing active state.")
                    self.active_object_props_in_view = None
                # else: No object was active, and still no object detected. Nothing to do.

                    
            
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
          return frame, mask, None, None, None, None, 0
      
      # check if area is over the min threshold
      largest_contour = max(contours, key=cv2.contourArea)
      area = cv2.contourArea(largest_contour)
      if area < area_thresh:
          return frame, mask, None, None, None, None, 0
      
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
      
      center_y = center[1] # y-coordinate of the center
      bottom_vertex_y = np.max(box[:, 1])
      vertical_distance_to_bottom = bottom_vertex_y - center_y
      center = (center[0], vertical_distance_to_bottom)
      
      frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2) 
      frame = cv2.circle(frame, center, 5, (255, 0, 0), 10)
      return frame, mask, center, angle, width, height, 1
    
    def _wait_for_step_confirmation(self, step_message):
        """ If step-by-step mode is enabled, prints a message and waits for Enter key. """
        if self.step_by_step_enabled:
            input(f"--- PAUSED: {step_message} --- Press Enter to continue...")
    
    def map_camara2robot(self, center_coords, detected_angle):
        """
        Maps camera coordinates (cx, cy) to robot target (self.target_x, self.target_y).
        Uses calibrated linear mapping coefficients.
        """
        if center_coords is None:
            return

        cx, cy = center_coords
        u_cam_px = cx # camera x-pixel
        v_cam_px = cy # camera y-pixel

        # Calculate robot's target X 
        robot_target_x = self.X_MAPPING_SLOPE * u_cam_px + self.X_MAPPING_INTERCEPT
        
        # Calculate robot's target Y
        robot_target_y = self.Y_MAPPING_SLOPE * v_cam_px + self.Y_MAPPING_INTERCEPT

        # Calculate robot's target j6
        robot_target_j6_deg = self.ANG_MAPPING_SLOPE * detected_angle + self.ANG_MAPPING_INTERCEPT
        
        object_data_for_queue = {
            'target_x': robot_target_x,
            'target_y': robot_target_y,
            'piece_angle': detected_angle,  # Raw camera angle for classification
            'target_j6_deg': robot_target_j6_deg  # Robot's J6 target for the pick operation
        }
        
        try:
          self.object_queue.put(object_data_for_queue)
          print(f"[MAP & QUEUE] Object data added to queue. CamInput: u={u_cam_px}, v={v_cam_px}")
          print(f"    Queued Data -> X={object_data_for_queue['target_x']:.1f}mm, "
                f"Y={object_data_for_queue['target_y']:.1f}mm, "
                f"CamAngle={object_data_for_queue['piece_angle']:.1f}°, "
                f"RobJ6Target={object_data_for_queue['target_j6_deg']:.1f}°")
          print(f"    Current queue size: {self.object_queue.qsize()}")
        
        except queue.Full:
          print(f"[MAP & QUEUE] CRITICAL WARNING: Object queue is full. "
                  f"Failed to add new object. This may indicate an issue if using blocking put().")
        
        except Exception as e:
          print(f"[MAP & QUEUE] ERROR: An unexpected error occurred while adding object to queue: {e}")
        
        print(f"[MAP] CamInput: u={u_cam_px}, v={v_cam_px}")
        print(f"[MAP] Robot Target → X={self.target_x:.1f} mm, Y={self.target_y:.1f} mm, Angle (cam): {self.piece_angle:.1f}°, Target j6 (rob): {self.target_j6_deg:.1f}°")

    def _move_robot_to_standard_pose(self, pose_name, gripper_action_after_move=None):
        if pose_name not in self.STANDARD_POSES:
            print(f"[ROBOT_MOVE] ERROR: Standard pose '{pose_name}' not defined.")
            return False

        pose_data = self.STANDARD_POSES[pose_name]
        target_pose_coords = np.array(pose_data["coords"])
        description = pose_data["description"]
        speed = pose_data.get("speed", 80) # Default speed if not in dict
        acc = pose_data.get("acc", 50)   # Default acc if not in dict

        self._wait_for_step_confirmation(f"Moving to {description}: {np.round(target_pose_coords,1).tolist()}")
        print(f"[ROBOT_MOVE] Moving to {description}: {np.round(target_pose_coords,1).tolist()}")

        success_move, _, _ = self.robot.move_l_pose(target_pose_coords, speed=speed, acc=acc)

        if success_move:
            self.robot.wait_until_motion_complete()
            print(f"[ROBOT_MOVE] Successfully moved to {description}.")
            if gripper_action_after_move == "open":
                print("[ROBOT_MOVE] Opening gripper.")
                self.robot.open_gripper()
                time.sleep(0.7) # Allow time for gripper
            elif gripper_action_after_move == "close":
                print("[ROBOT_MOVE] Closing gripper.")
                self.robot.close_gripper()
                time.sleep(0.7) # Allow time for gripper
            return True
        else:
            print(f"[ROBOT_MOVE] CRITICAL: Failed to move to {description}.")
            return False

    def mozaic_generator(self, zone_type, piece_index_in_zone):
        """
        Calculates target ROBOT CARTESIAN pose (X, Y, Z, Rz) for placing a piece.
        Returns: (target_x, target_y, target_z, target_j6_on_pallet_deg) or None
        """
        print(f"Mozaic generator for zone: {zone_type}, piece index: {piece_index_in_zone}")
        
        base_center_x, base_center_y = 0.0, 0.0
        
        if zone_type == "0_deg_type":
            base_center_x = self.PALLET_ZONE_0_BASE_X
            base_center_y = self.PALLET_ZONE_0_BASE_Y
        elif zone_type == "90_deg_type":
            base_center_x = self.PALLET_ZONE_90_BASE_X
            base_center_y = self.PALLET_ZONE_90_BASE_Y
        else:
            print(f"Error: Unknown zone_type '{zone_type}' in mozaic_generator.")
            return None, None, None, None

        current_z_layer_index = piece_index_in_zone // self.ITEMS_PER_Z_LAYER_PATTERN
        index_within_pair = piece_index_in_zone % self.ITEMS_PER_Z_LAYER_PATTERN  # Will be 0 or 1

        target_z = self.BASE_Z_FOR_FIRST_LAYER - (current_z_layer_index * self.PHYSICAL_THICKNESS_MM)
        target_x, target_y, target_j6_on_pallet_deg = None, None, None
        

        target_j6_on_pallet_deg = self.PALLET_PRIMARY_RZ_DEG
        
        target_y = base_center_y
        
        offset_val_x = (self.PHYSICAL_WIDTH_MM / 2.0) + (self.ITEM_GAP_MM / 2.0)
        
        if index_within_pair == 0:
            target_x = base_center_x - offset_val_x
        else:
            target_x = base_center_x + offset_val_x
        print(f"  Zone {zone_type}, Z-Layer {current_z_layer_index} (X-offset): Piece in pair {index_within_pair}, j6={target_j6_on_pallet_deg} deg")

        
        if current_z_layer_index >= self.MAX_Z_LAYERS_PER_ZONE:
            print(f"Error: Pallet zone '{zone_type}' is full. Max Z-layers ({self.MAX_Z_LAYERS_PER_ZONE}) reached.")
            return None, None, None, None
        
        print(f"  Mosaic for Zone {zone_type}, Overall Idx {piece_index_in_zone}: ZLayer {current_z_layer_index}, Pair Idx {index_within_pair} -> X {target_x:.1f}, Y {target_y:.1f}, Z {target_z:.1f}, j6Plt {target_j6_on_pallet_deg:.1f}")
        return target_x, target_y, target_z, target_j6_on_pallet_deg

    def _pick_from_conveyor(self, pick_x, pick_y, target_j6_deg_for_pick): # Renamed last arg for clarity
        """ Commands the robot to pick an object, with step-by-step confirmation. """
        print(f"Executing pick at X:{pick_x:.1f}, Y:{pick_y:.1f} (J6 target: {target_j6_deg_for_pick:.1f} deg)")

        # 1. Move to an approach position (X, Y, LIFT_Z_COMMON)
        initial_rz_deg = target_j6_deg_for_pick # Using target J6 as nominal Rz for less travel
        approach_pose_cartesian = [pick_x, pick_y, self.LIFT_Z_COMMON,
                                   self.NOMINAL_RX_DEG, self.NOMINAL_RY_DEG, self.NOMINAL_RZ_DEG]
        
        self._wait_for_step_confirmation(f"Moving to approach pick pose: {np.round(approach_pose_cartesian,1).tolist()}")
        success, _, _ = self.robot.move_l_pose(np.array(approach_pose_cartesian), speed=40, acc=30)
        if not success:
            print("  Error: Failed to move to Cartesian approach pose.")
            return False
        self.robot.wait_until_motion_complete()

        # 2. Get current joint positions
        self._wait_for_step_confirmation("Getting current joint positions before J6 orient")
        success_joints, current_joints_deg, _ = self.robot.get_current_joints()
        if not success_joints:
            print("  Error: Failed to get current joint positions.")
            return False
        print(f"  Current joints (deg): {np.round(current_joints_deg, 2).tolist()}")

        # 3. Prepare and execute J6 orientation
        target_joints_deg_array = np.array(current_joints_deg)
        target_joints_deg_array[5] = target_j6_deg_for_pick # J6 is index 5
        
        self._wait_for_step_confirmation(f"Orienting J6 to {target_j6_deg_for_pick:.1f} deg. Target joints: {np.round(target_joints_deg_array,2).tolist()}")
        success, _, _ = self.robot.move_j_joint(target_joints_deg_array, speed=40, acc=30)
        if not success:
            print("  Error: Failed to orient Joint 6.")
            return False
        self.robot.wait_until_motion_complete()

        # 4. Get current TCP pose after J6 orientation
        self._wait_for_step_confirmation("Getting TCP pose after J6 orientation")
        success_tcp, tcp_after_j6_orient, _ = self.robot.get_tool_pose(user_coord=1, tool_num=2)
        if not success_tcp:
            print("  Error: Failed to get TCP pose after J6 orientation.")
            return False
        print(f"  TCP after J6 orient (deg): {np.round(tcp_after_j6_orient,2).tolist()}")
        
        # 5. Move down to actual pick position (linearly)
        pick_pose_cartesian_final = [tcp_after_j6_orient[0], tcp_after_j6_orient[1], self.PICK_Z_CONVEYOR,
                                     tcp_after_j6_orient[3], tcp_after_j6_orient[4], tcp_after_j6_orient[5]]
        self._wait_for_step_confirmation(f"Moving down to pick (Cartesian): {np.round(pick_pose_cartesian_final,1).tolist()}")
        success, _, _ = self.robot.move_l_pose(np.array(pick_pose_cartesian_final), speed=10, acc=60, dec=30)
        if not success: 
            print("  Error: Failed to move to actual pick pose.")
            return False
        self.robot.wait_until_motion_complete()

        # 6. Close gripper
        self._wait_for_step_confirmation("Closing gripper")
        self.robot.close_gripper()
        time.sleep(0.7)

        # 7. Lift the object (linearly)
        lift_pose_cartesian_final = [tcp_after_j6_orient[0], tcp_after_j6_orient[1], self.LIFT_Z_COMMON,
                                     tcp_after_j6_orient[3], tcp_after_j6_orient[4], tcp_after_j6_orient[5]]
        self._wait_for_step_confirmation(f"Lifting object (Cartesian): {np.round(lift_pose_cartesian_final,1).tolist()}")
        success, _, _ = self.robot.move_l_pose(np.array(lift_pose_cartesian_final), speed=20, acc=30, dec=60)
        if not success:
            print("  Error: Failed to lift object.")
            return False
        self.robot.wait_until_motion_complete()

        print("  Pick sequence successfully completed.")
        return True
    
    def _place_on_pallet(self, place_x, place_y, place_z_on_pallet, place_lift_z, place_j6_on_pallet_deg):
        """
        Commands the robot to perform the place sequence on the pallet.
        Includes step-by-step confirmation if enabled.

        Args:
            place_x (float): Target X coordinate for placing (robot mm).
            place_y (float): Target Y coordinate for placing (robot mm).
            place_z_on_pallet (float): Target Z coordinate for placing (robot mm).
            place_lift_z (float): Common Z height for approach/retreat (robot mm).
            place_rz_on_pallet_deg (float): Target Rz orientation on pallet (degrees).

        Returns:
            bool: True if place sequence is successful, False otherwise.
        """
        print(f"Executing place at X:{place_x:.1f}, Y:{place_y:.1f}, Z:{place_z_on_pallet:.1f}, j6:{place_j6_on_pallet_deg:.1f}")
        
        # 0. Get tool user pose.
        
        self._wait_for_step_confirmation("Getting TCP pose for place on pallet")
        success_tcp, current_pose_cartesian, _ = self.robot.get_tool_pose(user_coord=1, tool_num=2)
        if not success_tcp:
            print("  Error: Failed to get TCP pose for place on pallet.")
            return False
        print(f"  TCP pose before place on pallet: {np.round(current_pose_cartesian,2).tolist()}")

        # 1. Move to approach position (above the pallet spot)
        approach_place_pose = [place_x, place_y, place_lift_z,
                               current_pose_cartesian[3], current_pose_cartesian[4], place_j6_on_pallet_deg]
        
        self._wait_for_step_confirmation(f"Moving to approach place pose: {np.round(approach_place_pose,1).tolist()}")
        success, _, _ = self.robot.move_l_pose(np.array(approach_place_pose), speed=80, acc=50)
        if not success:
            print("  Error: Failed to move to approach place pose.")
            return False
        self.robot.wait_until_motion_complete()

        # 2.1 Get TCP
        self._wait_for_step_confirmation("Getting TCP pose before actual place")
        success_tcp, current_pose_cartesian, _ = self.robot.get_tool_pose(user_coord=1, tool_num=2)
        if not success_tcp:
            print("  Error: Failed to get TCP pose before actual place.")
            return False
        print(f"  TCP pose before actual place: {np.round(current_pose_cartesian,2).tolist()}")
        
        # 2.2 Move down to actual place position
        actual_place_pose = [place_x, place_y, place_z_on_pallet,
                             current_pose_cartesian[3], current_pose_cartesian[4], current_pose_cartesian[5]]
        
        self._wait_for_step_confirmation(f"Moving to actual place pose: {np.round(actual_place_pose,1).tolist()}")
        success, _, _ = self.robot.move_l_pose(np.array(actual_place_pose), speed=20, acc=20)
        if not success:
            print("  Error: Failed to move to actual place pose.")
            return False
        self.robot.wait_until_motion_complete()

        # 3. Open gripper
        self._wait_for_step_confirmation("Opening gripper")
        self.robot.open_gripper()
        time.sleep(0.7) # Allow time for gripper to open

        # 4. Retreat from pallet (lift up)
        # Using the same approach_place_pose for retreat for simplicity
        actual_place_pose = [place_x, place_y, place_lift_z,
                             current_pose_cartesian[3], current_pose_cartesian[4], current_pose_cartesian[5]]

        self._wait_for_step_confirmation(f"Retreating from place pose: {np.round(actual_place_pose,1).tolist()}")
        success, _, _ = self.robot.move_l_pose(np.array(actual_place_pose), speed=20, acc=20)
        if not success:
            print("  Error: Failed to retreat from place pose.")
            return False 
        self.robot.wait_until_motion_complete()

        print("  Place sequence successfully completed.")
        return True
    
    def _classify_piece_orientation(self, piece_angle_from_camera): # Parameter added
        """
        Classifies the piece's orientation based on the provided piece_angle_from_camera.
        This angle is the raw detected angle from the camera in degrees.

        Returns:
            str: "0_deg_type", "90_deg_type", or "unclassified".
        """
        # Assuming piece_angle_from_camera from detect_box (via queue) is reasonably constrained
        # (e.g., -90 to +90, or 0 to 180 after adjustments in detect_box).

        # Check for 0-degree type orientation
        if abs(piece_angle_from_camera) < self.ANGLE_CLASSIFICATION_THRESHOLD_DEG or abs(piece_angle_from_camera - 180.0) < self.ANGLE_CLASSIFICATION_THRESHOLD_DEG:
            return "0_deg_type"
        # Check for 90-degree type orientation (handles +90 and -90)
        # The logic abs(abs(angle) - 90) handles angles like -85 or +95 correctly as 90-type.
        elif abs(abs(piece_angle_from_camera) - 90.0) < self.ANGLE_CLASSIFICATION_THRESHOLD_DEG:
            return "90_deg_type"
        else:
            print(f"[CLASSIFY] Piece angle {piece_angle_from_camera:.1f}° is UNCLASSIFIED (threshold: +/-{self.ANGLE_CLASSIFICATION_THRESHOLD_DEG}°).")
            return "unclassified"
    
    def pick_and_place(self, object_data_from_queue): # <-- Parameter changed
        """
        Performs the complete pick and place cycle for a given object.
        object_data_from_queue is a dictionary retrieved from the queue, containing:
        {'target_x', 'target_y', 'piece_angle', 'target_j6_deg'}
        """
        self._wait_for_step_confirmation(f"Start of pick_and_place cycle for object: {object_data_from_queue}")

        if not object_data_from_queue: # Should ideally not happen if called correctly
            print("[PICK_AND_PLACE] CRITICAL: No object data provided. Aborting this cycle.")
            return False # Indicate failure

        # Extract data from the dictionary passed as argument
        pick_target_x = object_data_from_queue['target_x']
        pick_target_y = object_data_from_queue['target_y']
        # This is the raw camera angle, used for classification
        raw_piece_angle_from_cam = object_data_from_queue['piece_angle']
        # This is the robot's J6 target angle for the pick operation
        pick_j6_deg_for_robot = object_data_from_queue['target_j6_deg']

        print(f"[PICK_AND_PLACE] Processing object: X={pick_target_x:.1f}, Y={pick_target_y:.1f}, "
              f"CamAngle={raw_piece_angle_from_cam:.1f}°, RobotPickJ6={pick_j6_deg_for_robot:.1f}°")

        # 1. Execute Pick Operation
        # Ensure _pick_from_conveyor uses these specific coordinates and J6 angle
        pick_successful = self._pick_from_conveyor(pick_target_x, pick_target_y, pick_j6_deg_for_robot)

        if not pick_successful:
            print("[PICK_AND_PLACE] Pick operation failed. Aborting cycle for this object.")
            # Note: The robot might still be holding the gripper closed if pick started but failed mid-way.
            # Consider if an explicit gripper open is needed here for error recovery.
            return False # Indicate failure

        self._wait_for_step_confirmation("Pick successful. Proceeding to classify orientation.")

        # 2. Classify Piece Orientation
        # _classify_piece_orientation now needs to accept the angle.
        # We stored the raw camera angle in the queue for this purpose.
        zone_type = self._classify_piece_orientation(raw_piece_angle_from_cam)
        print(f"[PICK_AND_PLACE] Object classified as '{zone_type}' (based on cam angle: {raw_piece_angle_from_cam:.1f}°).")

        if zone_type == "unclassified":
            self._wait_for_step_confirmation(f"Piece is UNCLASSIFIED (angle: {raw_piece_angle_from_cam:.1f}°). Aborting placement.")
            # 1. Move to a reject bin/area.
            # 2. Open gripper.
            # 3. Return to a safe/home position.
            print("    Action: Aborting placement. Piece is currently held. Implement reject sequence!")
            
            success_move_to_bin = self._move_robot_to_standard_pose(
            pose_name="reject_bin",
            gripper_action_after_move="open",
            )

            if not success_move_to_bin:
                print("[PICK_AND_PLACE] WARNING: Move to reject bin failed.")

            else:
                print("[PICK_AND_PLACE] Successfully rejected piece.")
            
            success_return_to_neutral = self._move_robot_to_standard_pose(
              pose_name="initial_neutral_conveyor",
              gripper_action_after_move="open",
            )

            if not success_return_to_neutral:
                print("[PICK_AND_PLACE] WARNING: Failed to return to neutral pose after reject.")
                # Decide if this should make the whole pick_and_place operation fail.
                # For now, let's consider it a warning but the PnP itself was successful.
                # If it MUST return to neutral to be considered a success, then return False here.
                # return False 
            else:
                print("[PICK_AND_PLACE] Successfully returned to neutral pose.")
            
            return False

        self._wait_for_step_confirmation(f"Orientation classified as '{zone_type}'. Proceeding to mosaic generator.")

        # 3. Get Piece Index for the designated zone
        current_piece_index_in_zone = 0
        if zone_type == "0_deg_type":
            current_piece_index_in_zone = self.piece_count_zone_0_deg
        elif zone_type == "90_deg_type":
            current_piece_index_in_zone = self.piece_count_zone_90_deg

        # 4. Generate Placement Pose using the mosaic generator
        place_pose_details = self.mozaic_generator(zone_type, current_piece_index_in_zone)

        if place_pose_details is None or place_pose_details[0] is None: # Check if mozaic_generator returned valid data
            self._wait_for_step_confirmation(f"Mosaic generator FAILED or pallet zone full (Zone: {zone_type}, Index: {current_piece_index_in_zone}). Aborting placement.")

            print("    Action: Aborting placement. Piece is currently held. Implement alternative placement/reject!")
            
            success_move_to_bin = self._move_robot_to_standard_pose(
            pose_name="reject_bin",
            gripper_action_after_move="open",
            )

            if not success_move_to_bin:
                print("[PICK_AND_PLACE] WARNING: Move to reject bin failed.")

            else:
                print("[PICK_AND_PLACE] Successfully rejected piece.")
            
            success_return_to_neutral = self._move_robot_to_standard_pose(
              pose_name="initial_neutral_conveyor",
              gripper_action_after_move="open",
            )

            if not success_return_to_neutral:
                print("[PICK_AND_PLACE] WARNING: Failed to return to neutral pose after reject.")
                # Decide if this should make the whole pick_and_place operation fail.
                # For now, let's consider it a warning but the PnP itself was successful.
                # If it MUST return to neutral to be considered a success, then return False here.
                # return False 
            else:
                print("[PICK_AND_PLACE] Successfully returned to neutral pose.")
            
            return False

        place_x, place_y, place_z_on_pallet, place_j6_on_pallet_deg = place_pose_details
        self._wait_for_step_confirmation(f"Mosaic generated. Target place pose: X={place_x:.1f}, Y={place_y:.1f}, Z={place_z_on_pallet:.1f}, j6={place_j6_on_pallet_deg:.1f}. Proceeding to place.")

        # 5. Execute Place Sequence
        # _place_on_pallet needs: place_x, place_y, place_z_on_pallet, place_lift_z (self.LIFT_Z_COMMON), place_rz_on_pallet_deg
        place_successful = self._place_on_pallet(place_x, place_y, place_z_on_pallet,
                                                 self.LIFT_Z_COMMON, # Using common lift Z for approach/retreat during place
                                                 place_j6_on_pallet_deg)

        if not place_successful:
            print("[PICK_AND_PLACE] Place operation failed.")
            # TODO: Handle place failure (e.g., retry logic, move to error recovery state).
            # Piece is likely still held by the gripper at the failed placement attempt.
            return False # Indicate failure

        self._wait_for_step_confirmation("Place successful. Proceeding to update counts.")

        # 6. Update Counts
        if zone_type == "0_deg_type":
            self.piece_count_zone_0_deg += 1
        elif zone_type == "90_deg_type":
            self.piece_count_zone_90_deg += 1
        
        self.piece_num += 1 # Increment total pieces processed by this robot instance
        # self.object_detected = False # This class attribute is no longer used for this purpose
        print(f"[PICK_AND_PLACE] Cycle complete for this object. Counts: Zone0={self.piece_count_zone_0_deg}, Zone90={self.piece_count_zone_90_deg}, Total={self.piece_num}")
        self._wait_for_step_confirmation("Counts updated.")
        
        # 7. Return to initial neutral pose
        print("[PICK_AND_PLACE] Returning to initial neutral conveyor pose...")
        # The "initial_neutral_conveyor" pose in STANDARD_POSES should handle gripper opening.
        success_return_to_neutral = self._move_robot_to_standard_pose(
            pose_name="initial_neutral_conveyor",
            gripper_action_after_move="open",
        )

        if not success_return_to_neutral:
            print("[PICK_AND_PLACE] WARNING: Main pick and place was successful, but failed to return to neutral pose.")
            # Decide if this should make the whole pick_and_place operation fail.
            # For now, let's consider it a warning but the PnP itself was successful.
            # If it MUST return to neutral to be considered a success, then return False here.
            # return False 
        else:
            print("[PICK_AND_PLACE] Successfully returned to neutral pose.")

        self._wait_for_step_confirmation("End of pick_and_place cycle for this specific object.")
        return True # Indicate success for this object (assuming return to neutral is not a hard fail condition for the PnP itself)
   
    def run(self):
            """Main execution loop for the robot."""

            if not self.camera_available:
                print("Error: Camera not initialized. Cannot start robot run sequence.")
                self._stop_event.set()
                return

            self._stop_event.clear()

            cam_thread = threading.Thread(target=self.camera_thread)
            cam_thread.start()
            print("Camera thread started.")

            if self.robot.connect():
                print("Successfully connected to robot.")
                self.robot.set_servo_status(1)
                
                # --- Move to Initial Pick Position ---
                move_successful = self._move_robot_to_standard_pose(
                    pose_name="initial_neutral_conveyor", 
                    gripper_action_after_move="open" # As per your updated code
                )
                if not move_successful:
                    print("[ROBOT_RUN] CRITICAL: Failed to move to initial neutral pose. Aborting run.")
                    self._stop_event.set()
                    if self.robot.is_connected(): # Ensure this method exists or adapt
                        self.robot.set_servo_status(0)
                        self.robot.disconnect()
                    if cam_thread.is_alive(): # Make sure cam_thread is defined in this scope
                        cam_thread.join(timeout=5.0)
                    return # Exit the run method

                previous_sensor_state_is_detecting = False
                SENSOR_ADDRESS = 915
                QUEUE_GET_TIMEOUT = 0.05

                try:
                    while not self._stop_event.is_set():
                        if not cam_thread.is_alive() and not self._stop_event.is_set():
                            print("CRITICAL: Camera thread has stopped unexpectedly. Signaling shutdown.")
                            self._stop_event.set()
                            break

                        # Sensor reading logic
                        success_read, sensor_value_str, _ = self.robot.send_cmd("getVirtualOutput", {"addr": SENSOR_ADDRESS})
                        current_sensor_state_is_detecting = None

                        if success_read and sensor_value_str is not None:
                            try:
                                sensor_value = int(sensor_value_str)
                                if sensor_value == 1:
                                    current_sensor_state_is_detecting = False
                                elif sensor_value == 0:
                                    current_sensor_state_is_detecting = True
                                else:
                                    print(f"Warning: Unexpected sensor value ({sensor_value}) at address {SENSOR_ADDRESS}")
                            except ValueError:
                                print(f"Warning: Received non-integer sensor value '{sensor_value_str}' from address {SENSOR_ADDRESS}")
                        elif not success_read:
                            print(f"Warning: Failed to read virtual output at address {SENSOR_ADDRESS}.")

                        # --- Main Logic: Sensor Trigger and Queue Check ---
                        if current_sensor_state_is_detecting is True:
                            if not previous_sensor_state_is_detecting:
                                print(f"[ROBOT_RUN] Sensor at {SENSOR_ADDRESS} indicates object presence.")

                            try:
                                object_data_to_process = self.object_queue.get(block=True, timeout=QUEUE_GET_TIMEOUT)

                                print(f"[ROBOT_RUN] Sensor active & object retrieved from queue (size before get: {self.object_queue.qsize()+1}).")
                                print(f"    Data: X={object_data_to_process['target_x']:.1f}, Y={object_data_to_process['target_y']:.1f}")

                                success_pnp = self.pick_and_place(object_data_to_process)

                                if success_pnp:
                                    print(f"[ROBOT_RUN] Pick and place cycle successful for one object.")
                                else:
                                    print(f"[ROBOT_RUN] Pick and place cycle failed or was aborted for one object.")
                                
                                self.object_queue.task_done()

                            except queue.Empty:
                                if previous_sensor_state_is_detecting:
                                    print(f"[ROBOT_RUN] Sensor active, but object queue is empty (or timed out). Waiting...")
                            except Exception as e_pnp:
                                print(f"[ROBOT_RUN] CRITICAL ERROR during pick_and_place or queue get: {e_pnp}")
                                import traceback
                                traceback.print_exc()
                        
                        elif current_sensor_state_is_detecting is False:
                            if previous_sensor_state_is_detecting is True:
                                print(f"[ROBOT_RUN] Sensor at {SENSOR_ADDRESS} indicates object removed or no longer present.")
                        
                        if current_sensor_state_is_detecting is not None:
                            previous_sensor_state_is_detecting = current_sensor_state_is_detecting

                        self._stop_event.wait(0.1)

                except KeyboardInterrupt:
                    print("[ROBOT_RUN] Keyboard interrupt detected. Signaling shutdown.")
                    self._stop_event.set()
                except Exception as e:
                    print(f"[ROBOT_RUN] CRITICAL ERROR in main execution loop: {e}")
                    import traceback
                    traceback.print_exc()
                    self._stop_event.set()
                finally:
                    print("[ROBOT_RUN] Main loop ending. Initiating shutdown sequence...")
                    self._stop_event.set()
                    
            else:
                print("[ROBOT_RUN] Failed to connect to the robot. Application will exit.")
                self._stop_event.set()

            if cam_thread.is_alive():
                print("[ROBOT_RUN] Waiting for camera thread to join...")
                cam_thread.join(timeout=5.0)
                if cam_thread.is_alive():
                    print("[ROBOT_RUN] WARNING: Camera thread did not join in time. It might be stuck.")
            else:
                print("[ROBOT_RUN] Camera thread already finished.")

            print("Palletizing robot application finished.")


if __name__ == "__main__":
    robot_ip = "169.168.0.200" # Replace with your robot's IP

    robot_controller = PalletizingRobot(robot_ip,
                                        step_mode=False) 
    
    print("Initializing camera...")
    robot_controller.initialize_camera()
    
    if robot_controller.camera_available:
        print("Camera initialized successfully.")
        robot_controller.run()
    else:
        print("Failed to initialize camera. Exiting.")