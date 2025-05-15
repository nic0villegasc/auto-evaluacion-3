import socket
import json
import time
import numpy as np
import cv2
import threading

class ELITE:

    CHECK_TIME = 0.5
    
    def __init__(self, ip, port=8055):
        self.ip = ip
        self.port = port
        self.sock = None

        # robot values
        self.user_num = None
        self.tool_num = None
        self.curr_coord = None
        self.DOF = 6 # EA66 has 6 independent joints

        # thread safety to ensure that send.cmd is not called simultaneously
        self.lock = threading.Lock() 

    def connect(self):
        """
        Function to connect to the robot via ethernet, and to initialize basic variables
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5) # 5 seconds
        try:
            self.sock.connect((self.ip, self.port))
            self.get_current_coord()
            self.get_user_number()
            self.get_tool_number()
            return True
        except Exception as e:
            self.sock.close()
            self.sock = None
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """
        Always close the socket when you are done with the robot, to avoid leaving it open.
        """
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_cmd(self, method, params=None, id=1):
        """
        Function wrapper that is called by all other functions to send commands to the robot, this function 
        can be called by a single thread at a time to avoid sending erroneous commands.

        :returns a tuple of (success, result, id), where success is a boolean indicating if the command was successful,
            result is the result of the command, and id is the id of the command (in almost every case can be ignored).
        """
        with self.lock:
            if params is None:
                params = []
            else:
                params = json.dumps(params)
            send_str = f'{{"method": "{method}", "params": {params}, "jsonrpc": "2.0", "id": {id}}}\n'
            try:
                self.sock.sendall(send_str.encode('utf-8'))
                ret = self.sock.recv(1024)
                jdata = json.loads(ret.decode('utf-8'))
                if "result" in jdata:
                    return True, json.loads(jdata["result"]), jdata["id"]
                elif "error" in jdata:
                    return False, jdata["error"], jdata["id"]
                else:
                    return False, None, None
            except Exception as e:
                print(f"Command failed: {e}")
                print("method:", method)
                return False, None, None
    
    def send_multiple_cmds(self, cmds):
        """
        Function that sends multiple commands to the robot, this function is useful when you want to send multiple
        commands in a row, when you require to send a queue of commands.
        """

        with self.lock:

            for cmd in cmds:

                method = cmd["method"]
                params = cmd["params"]
                id = cmd["id"]

                if params is None:
                    params = []
                else:
                    params = json.dumps(params)
                send_str = f'{{"method": "{method}", "params": {params}, "jsonrpc": "2.0", "id": {id}}}\n'
                try:
                    self.sock.sendall(send_str.encode('utf-8'))
                    ret = self.sock.recv(1024)
                    jdata = json.loads(ret.decode('utf-8'))

                except Exception as e:
                    print(f"Command failed: {e}")
                    print("method:", method)
                    return False, None, None
            if "result" in jdata:
                return True, json.loads(jdata["result"]), jdata["id"]
            elif "error" in jdata:
                return False, jdata["error"], jdata["id"]
            else:
                return False, None, None

    def stop(self):
        """
        Inmediately halts the robot movement, and stops all the motors.
        """
        return self.send_cmd("stop")

    def get_robot_state(self):
        """
        :returns the overall state of the robot, where
            0: stop, 1: pause, 2: emergency stop, 3: running state, 4: alarm state,  5: collision state
        """
        return self.send_cmd("getRobotState")

    def set_user_number(self, user_num):
        """
        Sets the number of the user coordinate system

        :param user_num: 0: base, any other: customised user coordinate system
        """
        self.user_num = user_num
        return self.send_cmd("setUserNumber", {"user_num": user_num})

    def get_user_number(self):
        """
        :returns 0: base, any other: customised user coordinate system
        """
        succ, self.user_num, id = self.send_cmd("getUserNumber")
        return succ, self.user_num, id
    
    
    def get_user_coord_pose(self, user_num, radians = False):
        """
        :param user_num: 0: base, any other: customised user coordinate system
        :param radians: True: Radians, False: degrees

        :returns the pose values of the user_num coordinate system as
            [x, y, z, rx, ry, rz] in mm and degrees by default.
            this values can also be accessed in the pendant of the robot in
            preparation -> User Config, in the lower part of the screen.
        """
        success, user_pose, id_ = self.send_cmd("getUserFrame", {"user_num": user_num, "unit_type": int(radians)})
        if success:
            user_pose = np.array(user_pose)
        return success, user_pose, id_
    
    def get_current_joints(self):
        """
        :returns the current joint values in degrees as an array.

        [Note]: For some reason the joint values of the robot can also be
        read by the function "get_motor_pos".
        """
        success, joints, id_ = self.send_cmd("get_actual_joint")
        if success:
            joints = np.array(joints)
        return success, joints, id_
    
    def set_outputIO(self, io_num, value):
        """
        Function that sets a specific value to a 
        digital io_num output.

        :param io_num: Number of the digital output to set.
        :param value: Value to set to the digital output.
        """
        return self.send_cmd("setOutput", {"addr": io_num, "status": value})

    def get_inputIO(self, io_num):
        """
        Function that reads the value of a specific digital io_num input.

        :param io_num: Number of the digital input to read.

        :returns the value of the digital input.
        """
        return self.send_cmd("getInput", {"addr": io_num})
    
    def get_current_coord(self):
        """
        :returns the current system of coordinated used
            0: joint, 1: base, 2: tool, 3: user, 4: cylinder
        """
        return self.send_cmd("getCurrentCoord")

    def set_current_coord(self, coord_num):
        """
        :param coord_num: Sets the current system of coordinated used
            0: joint, 1: base, 2: tool, 3: user, 4: cylinder
        """
        self.curr_coord = coord_num
        return self.send_cmd("setCurrentCoord", {"coord_mode" : coord_num})
    
    def jog(self, index, exec_time, speed=10):
        """
        This function purpose is to move a joint in a specific time and speed
        mimicking the jog function in the pendant of the robot.

        :param index: Joint index to move [0 - 11]. first index is +movement and second index is -movement.
            for example index = 5 will jog the joint 3 in the negative direction.
        :param exec_time: Time to execute
        :param speed: Speed of the movement
        """
        iterations_ = max(int(exec_time/0.1), 1)
        for i in range(iterations_):
            self.send_cmd("jog", {"index": index, "speed": speed})
            time.sleep(0.1)
        return self.send_cmd("stop")
    
    def get_tool_pose_in_base_coords(self):
        """
        returns: the current pose of the tool in the base coordinate system
            as an array [x, y, z, rx, ry, rz] in mm and degrees
        """
        success, result, id_ = self.send_cmd("get_tcp_pose", {"coordinate_num": -1, "tool_num": -1, "unit_type": 0})
        if success:
            result = np.array(result)
        return success, result, id_
    
    def get_tool_pose(self, user_coord, tool_num):
        """
        :param user_coord: 0: base, any other: customised user coordinate system
        :param tool_num: tool number of course

        returns: the current pose of the tool in the selected user coordinate system
            as an array [x, y, z, rx, ry, rz] in mm and degrees.
        """
        success, result, id_ = self.send_cmd("get_tcp_pose", {"coordinate_num": user_coord,
                                                              "tool_num": tool_num, "unit_type": 0})
        if success:
            result = np.array(result)
        return success, result, id_
    
    def set_tool_number(self, tool_num):
        """
        Sets the current tool number, the tool specifications can be accessed in the
        pendant of the robot in preparation -> Tool Config.

        :param tool_num: tool number to set.
        """
        self.tool_num = tool_num
        
        return self.send_cmd("setToolNumber", {"tool_num": tool_num})

    def get_tool_number(self):
        """
        Returns the current tool number, and also sets the internal variable tool_num,
        the tool specifications can be accessed in the pendant of the robot
        in preparation -> Tool Config.

        :returns the current tool number.
        """
        succ, self.tool_num, id = self.send_cmd("getToolNumber")
        return succ, self.tool_num, id
    
    def set_servo_status(self, status):
        """
        Sets the status of the robot servo motors

        :param status: 0: Servo off, 1: Servo on, servos must be on to move the robot.
        """
        return self.send_cmd("set_servo_status", {"status": status})
    
    
    def move_j_joint(self, target_pos, speed=50, acc=50, dec=50):
        """
        The equivalent of MOVJ in the pendant of the robot, moves the robot to a specific joint position.

        :param target_pos: Array of desired joint positions.
        :param speed: Operation speed.
        :param acc: Acceleration.
        :param dec: Deceleration.

        [WARNING]: Remember that the MOVJ command moves each joint to the specific desired position, and it
        does not care about the trajectory of the end effector, meaning that it will most probably move 
        not in a straight line to the desired position.
        """
        params = {
            "targetPos": target_pos.tolist(),
            "speed": speed,
            "acc": acc,
            "dec": dec
        }
        return self.send_cmd("moveByJoint", params)
    
    def move_j_pose(self, target_pose, speed=50, acc=50, dec=50):
        """
        Similar to MOVJ but instead of requiring a desired joint position
        it needs a cartesian pose and moves in the CURRENT USER FRAME
        :param target_pos: Array of desired cartesian pose [x, y, z, rx, ry, rz]
        :param speed: Operation speed.
        :param acc: Acceleration.
        :param dec: Deceleration.

        [WARNING]: Remember that the MOVJ command moves each joint to the specific desired position, and it
        does not care about the trajectory of the end effector, meaning that it will most probably move 
        not in a straight line to the desired position.
        """
        
        succ, target_joints, _ = self.inverse_kinematics(target_pose,
                                                      user_coord=self.user_num,
                                                      unit_type=0)
                                                      
        if succ:
            return self.move_j_joint(target_joints, speed, acc, dec)
        else:
            return succ, target_joints, _
        
    def move_l_joint(self, target_pos, speed=50, acc=50, dec=50):
        """
        The equivalent of MOVL in the pendant of the robot, moves the robot to a specific position desired 
        poisition of the joints, while esuring that the trajectory of the end effector is a straight line.

        :param target_pos: Array of desired joint positions.
        :param speed: Operation speed.
        :param acc: Acceleration.
        
        [WARNING]: Remember that the MOVL command can be dangerous in robot configurations that are near
        to a singularity, because it could cause some joints to move unexpectedly fast, if there is enough space
        sometimes it is better to move to a intermediate position with MOVJ, and then MOVL to the desired
        final position.
        """
        params = {
            "targetPos": target_pos.tolist(),
            "speed_type": 0,
            "speed": speed,
            "acc": acc,
            "dec": dec
        }
        return self.send_cmd("moveByLine", params)
    
    def move_l_pose(self, target_pose, speed = 50, acc = 50, dec = 50):
        """
        Similar to MOVL but instead of requiring a desired joint position
        it needs a cartesian pose and moves in the CURRENT USER FRAME

        :param target_pos: Array of desired cartesian pose [x, y, z, rx, ry, rz]
        :param speed: Operation speed.
        :param acc: Acceleration.
        :param dec: Deceleration.
        
        [WARNING]: Remember that the MOVL command can be dangerous in robot configurations that are near
        to a singularity, because it could cause some joints to move unexpectedly fast, if there is enough space
        sometimes it is better to move to a intermediate position with MOVJ, and then MOVL to the desired
        final position.
        """
        MAX_L_SPEED = 550
        linear_speed = int(MAX_L_SPEED * (speed/100.0))
        _, user_num, _ = self.get_user_number()
        _, user_coord, _ = self.get_user_coord_pose(user_num, radians = False)
       
        params = {
            "targetUserPose": target_pose.tolist(),
            "speed_type": 0,
            "speed": linear_speed,
            "acc": acc,
            "dec": dec,
            "unit_type": 0,
            "user_coord": user_coord.tolist()}

        return self.send_cmd("moveByLineCoord", params)
    
    def inverse_kinematics(self, target_pos, user_coord = 0, reference_pos = None, unit_type=0, j1_sign = 0):
        """
        Customized function that calculates the inverse kinematics of the robot.

        :param target_pos: Array of desired end effector positions in the base coordinate system [x, y, z, rx, ry, rz].
        :j1_sign: Preference for j1 value sign of IK result, 1: Enforce positive j1,
                -1: Enforce negative j1, 0: no preference

        :returns the joint positions in degrees.
        """

        # Transform the target position to the base coords, if it is not already in the base coords.
        _, target_pos, _ = self.user_coord_to_base_coord(target_pos, user_coord, unit_type)

        if reference_pos is None:
            success, result, id_ = self.send_cmd("inverseKinematic", {"targetPose": target_pos.tolist(),
                                                                        "unit_type" : unit_type})
        else:
            success, result, id_ = self.send_cmd("inverseKinematic", {"targetPose": target_pos.tolist(),
                                                                    "unit_type" : unit_type,
                                                                    "referencePos": reference_pos.tolist()})
        if success:
            result = np.array(result)
            
            # Ensure result stays withing the joint range
            result = np.where(result >= 360, result % 360, result)
            result = np.where(result <= -360, result % -360, result)
            
            if j1_sign == 1: # Preference for j1 pos
                if result[0] < 0:
                    result[0] += 360
            
            elif j1_sign == -1:
                if result[0] > 0:
                    result[0] += -360
                
        return success, result, id_
    
    def get_transformation_matrix(self):

        # point_in_tool_coords: Point in the tool coordinate system [x, y, z, rx, ry, rz]
        _, tool_in_base, _ = self.get_tool_pose_in_base_coords()

        # Robot definition of euler angle rotations
        # Rot in Rx then Ry then Rz 

        # Rot in X
        alpha = tool_in_base[3] * np.pi / 180
        RotX = np.array([[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]])
        
        # Rot in Y
        beta = tool_in_base[4] * np.pi / 180
        RotY = np.array([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]])
        
        # Rot in Z
        gamma = tool_in_base[5] * np.pi / 180
        RotZ = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]])
        
        Rot = np.dot(RotZ, np.dot(RotY, RotX))

        T = np.array([[Rot[0, 0], Rot[0, 1], Rot[0, 2], tool_in_base[0]],
                        [Rot[1, 0], Rot[1, 1], Rot[1, 2], tool_in_base[1]],
                        [Rot[2, 0], Rot[2, 1], Rot[2, 2], tool_in_base[2]],
                        [0, 0, 0, 1]])
        return T

    def tool_transformation_matrix_to_pose(self, T, base_T):
        """
        Function that converts a transformation matrix to a pose in the base coordinate system.

        :param T: Transformation matrix [4x4] in tool coords.

        :returns pose: Pose in the base coordinate system [x, y, z, rx, ry, rz].
        """
        T = np.dot(base_T, T)

        pose = np.array([T[0, 3], T[1, 3], T[2, 3]])
        R = T[:3, :3]
        r = self.R2r(R)
        pose = np.concatenate((pose, r))
        return pose
    
    def R2r(self, R):
        """
        Function that converts a rotation matrix to euler angles.

        :param R: Rotation matrix [3x3].

        :returns r: Euler angles [rx, ry, rz].
        """
        r = np.zeros(3)
        r[0] = np.arctan2(R[2, 1], R[2, 2])
        r[1] = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        r[2] = np.arctan2(R[1, 0], R[0, 0])
        r = r * 180 / np.pi
        return r
    
    def base_coord_to_user_coord(self, target_pos, user_coord, unit_type=0):
        """
        Customized function that converts a position from the base coordinate system to a user coordinate system.

        :param target_pos: Array of desired end effector positions in the base coordinate system [x, y, z, rx, ry, rz].
        :param user_coord: 0: base, any other: customised user coordinate system.
        :param unit_type: Unit type (0: degrees, 1: radians), by default is 0 (degrees).

        :returns the position in the user coordinate system.
        """
        success, result, id_ = self.send_cmd("convertPoseFromCartToUser", {"targetPose": target_pos.tolist(),
                                                            "userNo": user_coord, "unit_type": unit_type})
        if success:
            result = np.array(result)
        return success, result, id_
    
    def user_coord_to_base_coord(self, target_pos, user_coord, unit_type=0):
        """
        Customized function that converts a position from a user coordinate system to the base coordinate system.

        :param target_pos: Array of desired end effector positions in the user coordinate system [x, y, z, rx, ry, rz].
        :param user_coord: 0: base, any other: customised user coordinate system.
        :param unit_type: Unit type (0: degrees, 1: radians), by default is 0 (degrees).

        :returns the position in the base coordinate system.
        """
        success, result, id_ = self.send_cmd("convertPoseFromUserToCart", {"targetPose": target_pos.tolist(),
                                                            "userNo": user_coord, "unit_type": unit_type})
        if success:
            result = np.array(result)
        return success, result, id_
        
        
    def wait_until_motion_complete(self):
        """
        Customized function that polls the state of the robot every self.CHECK_TIME until the movement
        is completed to syncronize movements.
        """
        while True:
            success, result, _ = self.get_robot_state()
            if success:
                state = result
                if state == 0:
                    return True
            time.sleep(self.CHECK_TIME)
        
    def set_payload(self, payload, center_of_gravity = [0, 0, 0]):
        success, result, id_ = self.send_cmd("cmd_set_payload",
            {"tool_num": self.tool_num, "m": payload, "cog": center_of_gravity})

    def get_virtual_var(self, address):
        return self.send_cmd("getVirtualInput", {"addr": address})
        
    def set_virtual_var(self, address, value):
        """
        Function that changes the value of "M" variables, equivalent
        of a coil in mosbusRTU nomenclature.
        [warning]: both set_virtual_register and set_virtual_var use the same
        M variables, so be aware of the addresses that you are using, to avoid
        unintended changes.
        
        Coil addresses: [0x0000, 0x020F] (0 - 527) READ ONLY, [0x0210 - 0x05BF] (528, 1471) read/write
        """
            
        if value not in [0, 1]:
            print("VALUE MUST BE 1 OR 0")
            raise(Exception)
        
        if address > 799 or address < 528:
            print("ADRESS OUT OF INDEX")
            raise(Exception)
        return self.send_cmd("setVirtualOutput", {"addr": address, "status": value})
            
    def get_payload(self):
        success, result, id_ = self.send_cmd("getPayload", {"tool_num": self.tool_num})
        
    def open_gripper(self):
        self.set_outputIO(49, 1)
        self.set_outputIO(48, 0)
        time.sleep(1.5)
    
    def close_gripper(self):
        self.set_outputIO(49, 0)
        self.set_outputIO(48, 1)
        time.sleep(1.5)