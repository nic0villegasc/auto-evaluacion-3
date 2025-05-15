from picamera2 import Picamera2, Preview
from libcamera import ColorSpace, controls
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")

from PIL import Image, ImageDraw, ImageFont
import time
import glob
import cv2
import sys
import os
import threading

"""
Script for camera calibration. In case that the camera is not able to correctly
detect the wood piece due to a change of the light conditions the following steps
should be followed:

1.- Erase the saved calibration parameters in the folder CalibrationParams
2.- Run the "PalletizingRobot.py" script
3.- Align the black rectangle in the camera with the big "50% gray" square in 
    the back of the spydercheckr24 card.
4.- When the terminal says "Press any key to take a picture of the gray card", 
    make sure that the spydercheck24 card is fully visible in the camera, then
	press any key.
5.- Click on squares from top to buttom (and lastly the big gray patch),
	make sure that the square is completely within the color patch,
	in case that the placed square is too big you can resize it by pressing
	"t" or "g", once all squares are correctly placed press "q" to continue.
6.- Turn the card around and press any key to take a picture of the color card,
	again make sure that the card is fully visible in the camera.
7.- Click on squares from top to buttom, and from left to right,
	make sure that the square is completely within the color patch,
	in case that the placed square is too big you can resize it by pressing
	"t" or "g", once all squares are correctly placed press "q" to continue.
8.- If all the process was correctly done then two windows should be displayed
	press any key to close them.
9.- Now all the parameters are saved in the folder CalibrationParams, re run the 
    "palletizingRobot.py" script and check if the camera is correctly detecting 
	the wood piece, if not repeat the process.

	
[IMPORTANT]: You are not supposed to change anything in this script, but you can
    read it in case that you are interested in color calibration or on how 
	cameras work!
"""

class CameraCalibrationHelper():
	
	def __init__(self):
		
		self.SpyderCheckr24rGBCodes = {
			"4D": [139, 136, 135],
			"1E": [249, 242, 238], "2E": [202, 198, 195],
			"3E": [161, 157, 154], "4E": [122, 118, 116],
			"5E": [80, 80, 78], "6E": [43, 41, 43],
			"1F": [0, 127, 159], "2F": [192, 75, 145],
			"3F": [245, 205, 0], "4F": [186, 26, 51],
			"5F": [57, 146, 64], "6F": [25, 55, 135],
			"1G": [222, 118, 32], "2G": [58, 88, 159],
			"3G": [195, 79, 95], "4G": [83, 58, 106],
			"5G": [157, 188, 54], "6G": [238, 158, 25],
			"1H": [98, 187, 166], "2H": [126, 125, 174],
			"3H": [82, 106, 60], "4H": [87, 120, 155],
			"5H": [197, 145, 125], "6H": [112, 76, 60]}
		   

		# Auxiliar variables for color calibration / white balance / exposure gain
		self.display_imgs = None
		self.square_ini_side = 40
		self.user_squares = []
		self.color_cal = False

		# Calibration params
		self.lut = None
		self.R = None
		self.CCM_matrix = None
		self.analogue_gain = None
		self.exposure_time = None

		# Camera matrix / Distortion Array
		self.camera_mtx = None
		self.new_camera_mtx = None
		self.camera_dist = None
				
		self.path = os.path.dirname(os.path.abspath(__file__)) + "/CalibrationParams/"
		self.frame = None
		self.show_square = True
		self.show = True
		
	def set_exposure_time(self, exposure_time, exp_gain):
		self.exposure_time = exposure_time
		self.picam2.set_controls({"AeEnable":False,
			"ExposureTime": self.exposure_time, "AnalogueGain": exp_gain,
			"NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
			"AwbEnable": False, "FrameDurationLimits": (100, self.exposure_time)})
		print("Setting calibrated exposure time!")
	
	def initialize_raspicam(self, headless = False, img_size = (640, 480),  sensor_index = 2):
		self.picam2 = Picamera2()

		buffer_count = 4
		colour_space = ColorSpace.Sycc()
		queue = True
		display = "main"
		size_array = [(640, 480), (1640, 1232), (1920, 1080), (3280, 2464),
					  (640, 480), (1640, 1232), (1920, 1080), (3280, 2464)]# (for camera V2.1)
					  
		bit_depth_array = [10, 10, 10, 10, 8, 8, 8, 8] # (for camera V2.1)
		self.size = img_size # (640, 480)
		format_ = "XBGR8888"
		sensor_size = size_array[sensor_index] #(2028, 1520) # force to always have the maximum FOV (nominal)
		sensor_bit_d = bit_depth_array[sensor_index] 

		config = self.picam2.create_preview_configuration(
			main = {"size":self.size, "format": format_, },
			sensor = {"output_size": sensor_size, "bit_depth": sensor_bit_d},
			colour_space = colour_space, buffer_count = buffer_count, 
			queue = queue, display = display)
		self.picam2.configure(config)

		self.picam2.start()
		time.sleep(2)
		return self.picam2
	
	def draw_cal_exposure(self, rect):
		overlay = Image.new("RGBA", self.size, (0, 0, 0, 0))
		draw = ImageDraw.Draw(overlay)
		draw.rectangle(rect, outline = (0, 0, 0, 255), width= 5)
		self.picam2.set_overlay(np.array(overlay))
	
	def camera_thread(self):

		while self.show:
			frame = self.picam2.capture_array()[:, :, 0:3]
			self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			
			if self.show_square:
				square_size = self.square_ini_side
				start_point = (int(self.size[0]/2) - int(square_size/2),
						   int(self.size[1]/2) - int(square_size/2))
				end_point =  (int(self.size[0]/2) + int(square_size/2),
							  int(self.size[1]/2) + int(square_size/2))
				color = (0, 0, 0)
				thickness = 4
				frame = cv2.rectangle(self.frame, start_point, end_point, color, thickness)
				
				self.display_frame = frame.copy()
			else:
				self.display_frame = self.frame.copy()
		print("Closing camera thread")
		
	def calibrate_raspberry(self):
		
		# Calibrate exposure using the big gray spot (50% gray) on the 
		# spydercheckr calibration palette.
		min_fps = 30
		max_frame_duration = int((1.0/min_fps)*1000000)
		
		# Check for saved parameters
		if self.load_calibration_params():
			self.picam2.set_controls({"AeEnable":False,
			"ExposureTime": self.exposure_time, "AnalogueGain": self.analogue_gain,
			"NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
			"AwbEnable": False, "FrameDurationLimits": (100, self.exposure_time)})
			return 
			
		thread = threading.Thread(target=self.camera_thread, daemon=True)
		thread.start()
			
		# Fix exposure time
		max_exp_time =  max_frame_duration 
		exp_time = 1 
		analogue_gain = 1
		delta = 1000
		
		err_thresh = 5
		self.picam2.set_controls({"AeEnable":False, "ExposureTime": exp_time,
			"AnalogueGain": 1,
			"NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
			"AwbEnable": False, "FrameDurationLimits": (100, max_frame_duration)})
		square_size = self.square_ini_side
		gray_rect = (int(self.size[0]/2) - int(square_size/2),
					 int(self.size[1]/2) - int(square_size/2),
					 int(self.size[0]/2) + int(square_size/2),
					 int(self.size[1]/2) + int(square_size/2))
		self.draw_cal_exposure(gray_rect)
		cont = 0
		try:
			while True:
				self.show_square = True
				frame = self.picam2.capture_array()[:, :, 0:3]
				gray = np.mean(frame[gray_rect[1]:gray_rect[3],
							   gray_rect[0]:gray_rect[2], :], axis = (0, 1, 2))
				gray = (gray/255)* 100
				err = gray - 50
				print("gray: ", gray)
				print("exp time: ", self.picam2.capture_metadata()["ExposureTime"])
				print("Analogue gain", analogue_gain)
				
				# Now with CV2 display instead of native one
				cv2.imshow("Camera", self.display_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

				if err < -err_thresh:
					exp_time += delta
					if exp_time + delta > max_exp_time:
						analogue_gain += 0.01
					self.picam2.set_controls({"AeEnable":False,
						 "AnalogueGain": analogue_gain,
						 "ExposureTime": exp_time})
					cont = 0
				elif err > err_thresh:
					exp_time -= delta
				
					self.picam2.set_controls({"AeEnable":False,
						 "ExposureTime": exp_time})
					cont = 0
				else:
					cont += 1
				
				if cont > 10:
					
					self.show_square = False
					exp_time = self.picam2.capture_metadata()["ExposureTime"]
					print("Camera exposure successfully calibrated!")
					print("Press any key to take a picture of the gray card")
					while True:
						cv2.imshow("Camera", self.display_frame)
						if cv2.waitKey(1) != -1:
							break
					
					img1 = self.frame.copy()
					cv2.imwrite(self.path + "cal_img1.jpg", img1)
					print("Turn Around the card")
					
					print("Press any key to take a picture of the color card")
					while True:
						cv2.imshow("Camera", self.display_frame)
						if cv2.waitKey(1) != -1:
							break

					img2 = self.frame.copy()
					cv2.imwrite(self.path + "cal_img2.jpg", img2)
					
					self.show = False
					cv2.destroyAllWindows()  # Force cleanup of previous windows
					time.sleep(0.5)  # Allow GUI backend to reset
					
					lut, R, CCM_matrix = self.complete_color_calibration(img1, img2)
					
					self.lut = lut
					self.R = R
					self.CCM_matrix = CCM_matrix
					self.analogue_gain = analogue_gain
					self.exposure_time = exp_time
					
					# Save parameters
					print("Saving parameters")
					np.save(self.path + "lut.npy", lut)
					np.save(self.path + "R.npy", R)
					np.save(self.path + "CCM_matrix.npy", CCM_matrix)
					print("exp time: ", exp_time)
					np.save(self.path + "exp_time.npy", exp_time)
					np.save(self.path + "analogue_gain.npy", analogue_gain)
					break
				
		except KeyboardInterrupt:
			print("STOPING PICAM!!!")
			self.picam2.stop()
	
	def load_calibration_params(self):
		
		try:
			self.lut = np.load(self.path + "lut.npy")
			self.R = np.load(self.path + "R.npy")
			self.CCM_matrix = np.load(self.path + "CCM_matrix.npy")
			self.exposure_time = np.load(self.path + "exp_time.npy")
			self.analogue_gain = np.load(self.path + "analogue_gain.npy")
			print("Saved parameters found!, exp time: ", self.exposure_time,
				  ", analogue gain: ", self.analogue_gain)
		except Exception as err:
			print(err)
			print("There are no saved calibration parameters...")
			print("Manual calibration required\n")
			return False
		return True
	
	def mouse_callback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.user_squares.append([x, y])
			self.draw_user_squares()
	
	
	def draw_user_squares(self):
        # Function that draws the user squares on the image
        # with the color of target spydercheckr24 patch
		aux = ["1E", "2E", "3E", "4E", "5E", "6E", "4D"]
		cont = 0
		display_img = self.display_imgs[0].copy() if not self.color_cal else self.display_imgs[1].copy()
		for (cx, cy) in self.user_squares:
			half = self.square_ini_side // 2
			x1, y1 = cx - half, cy - half
			x2, y2 = cx + half, cy + half
			if not self.color_cal:
				rgb = self.SpyderCheckr24rGBCodes[aux[cont]]
			else:
				rgb = self.SpyderCheckr24rGBCodes["{}{}".format(cont % 6 + 1 ,
															chr(int(cont/6)+69))]
			display_img[y1:y2, x1:x2, :] = (rgb[2], rgb[1], rgb[0])
			cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
			cv2.circle(display_img, (cx, cy), 3, (255, 0, 0), -1)
			cont += 1
		window_name = "Color Calibration" if self.color_cal else "Gain/white balance Calibration"
		cv2.imshow(window_name, display_img)

	def complete_color_calibration(self, image1, image2):
		# Functions that calibrates exposure gain, white balance and color correction
		# First step: Exposure Gain Calibration
		self.display_imgs = [image1.copy(), image2.copy()]
		cv2.namedWindow("Gain/white balance Calibration")
		cv2.setMouseCallback("Gain/white balance Calibration", self.mouse_callback)
		cv2.imshow("Gain/white balance Calibration", image1)
		self.color_cal = False
		self.run_loop()
		cv2.destroyWindow('Gain/white balance Calibration')
		
		(lut, sample_illuminator, target_illuminator,
			rgb_sample, rgb_target) = self.calibrate_gain_exposure(image1)
		
		# Second step: White Balance Calibration
		R = self.calibrate_gray(sample_illuminator, target_illuminator,
								  rgb_sample, rgb_target)
	
		# Third step: Color Calibration
		cv2.namedWindow("Color Calibration")
		cv2.setMouseCallback("Color Calibration", self.mouse_callback)
		cv2.imshow("Color Calibration", image2)
		self.color_cal = True
		self.run_loop()
		cv2.destroyWindow('Color Calibration')
		CCM_matrix, cal_image = self.calibrate_color(image2, R, lut)
		# show initial image and last image
		cv2.imshow("Initial Image", image2)
		cv2.imshow("Final Image", cal_image)
		cv2.waitKey(0)
		cv2.destroyWindow('Initial Image')
		cv2.destroyWindow('Final Image')
		return lut, R, CCM_matrix

	def calibrate_gain_exposure(self, image):
		# Calculate average color of user selected patches
		# [WARNING] ASSUMING IMAGE IS IN BGR FORMAT
		image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		
		print("ENTERIIING")
		patches_rgb = []
		for cx, cy in self.user_squares:
			(x, y, w, h) = (cx - self.square_ini_side//2, cy - self.square_ini_side//2,
							self.square_ini_side, self.square_ini_side)
			patch = image[y:y+h, x:x+w, :]
			avg_rgb = np.mean(np.mean(patch, axis=0), axis=0)
			patches_rgb.append(avg_rgb)

		# Create graph of illuminants
		rgb_sample = np.array(patches_rgb)
		aux = ["1E", "2E", "3E", "4E", "5E", "6E", "4D"]
		rgb_target = np.array([self.SpyderCheckr24rGBCodes[name] for name in aux])

		# convert to linear 
		rgb_sample = self.srgb_to_linear(rgb_sample)
		rgb_target = self.srgb_to_linear(rgb_target)

		# Add 0 to the values (because every illuminant should have 0 value)
		rgb_sample = np.concatenate(([np.zeros(3)], rgb_sample))
		rgb_target = np.concatenate(([np.zeros(3)], rgb_target))

		# brightness mapping (exposure gain)
		sample_gray = rgb_sample.mean(axis=1)
		target_gray = rgb_target.mean(axis=1)

		# Create look up table for brightness mapping
		lut = self.create_look_up_table(sample_gray, target_gray)
		rgb_sample_2 = (rgb_sample * 255).astype(np.uint8)
		rgb_sample_2 = lut[rgb_sample_2]

		# Calculate mean 
		mean_sample = rgb_sample.mean(axis=0)
		mean_sample_2 = rgb_sample_2.mean(axis=0)
		mean_target = rgb_target.mean(axis=0)
		
		# Calcuate first PCA and line equation
		_, _, vv_s = np.linalg.svd(rgb_sample - mean_sample)
		_, _, vv_s_2 = np.linalg.svd(rgb_sample_2 - mean_sample_2)
		_, _, vv_t = np.linalg.svd(rgb_target - mean_target)
		linepts_t = mean_target + vv_t[0] * np.mgrid[-500:500:100j][:, np.newaxis]
		linepts_s = mean_sample + vv_s[0] * np.mgrid[-500:500:100j][:, np.newaxis]
		linepts_s_2 = mean_sample_2 + vv_s_2[0] * np.mgrid[-500:500:100j][:, np.newaxis]

		return (lut, (mean_sample_2, vv_s_2[0]), (mean_target, vv_t[0]),
				rgb_sample_2, rgb_target)

	def calibrate_gray(self, sample_illuminator, target_illuminator,
					   rgb_sample, rgb_target):
		# Calculate rotation matrix from illuminators
		target_point, target_direction = target_illuminator
		_, sample_direction = sample_illuminator

		# Construct rotation matrix using rodrigues formula
		cross = np.cross(sample_direction, target_direction)
		dot = np.dot(sample_direction, target_direction)
		skew = np.array([[0, -cross[2], cross[1]],
						[cross[2], 0, -cross[0]],
						[-cross[1], cross[0], 0]])
		R = np.eye(3) + skew + np.dot(skew, skew) * (1 - dot) / (
			np.linalg.norm(cross) ** 2)
		
		# Rotate sample points
		new_points = np.dot(rgb_sample, R.T)
		linepts_t = target_point + target_direction * np.mgrid[-500:500:100j][
			:, np.newaxis]
		
		return R

	def calibrate_color(self, image, R, lut):
		# [WARNING] ASSUMING IMAGE IS IN BGR FORMAT
		image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

		# linearize image
		image = self.srgb_to_linear(image)

		# Apply gain / white balance correction
		image = lut[(image * 255).astype(np.uint8)]
		image =  np.dot(image, R.T)

		# Calculate average color of user selected patches
		patches_rgb = []
		for cx, cy in self.user_squares:
			(x, y, w, h) = (cx - self.square_ini_side//2, cy - self.square_ini_side//2,
							self.square_ini_side, self.square_ini_side)
			patch = image[y:y+h, x:x+w, :]
			avg_rgb = np.mean(np.mean(patch, axis=0), axis=0)
			patches_rgb.append(avg_rgb)
		
		# Calculate Color Correction matrix
		real_rgb_array = np.array(patches_rgb)
		
		# drop the first element of the dictionary (that is 50% only available
		# in the back of the spydercheckr24 card)
		aux = list(self.SpyderCheckr24rGBCodes.keys())[1:]
		target_rgb_array = np.array([
			self.SpyderCheckr24rGBCodes[name] for name in aux]).reshape(24, 3)
		target_rgb_array = self.srgb_to_linear(target_rgb_array)

		# Calculate Color Correction Matrix
		CCM_matrix = np.dot(np.linalg.pinv(real_rgb_array), target_rgb_array).T

		# Apply Color Correction matrix to the original image
		image =  np.clip(np.dot(image, CCM_matrix.T), 0, 1)
		image = self.linear_to_srgb(image)

		# [WARNING] ASSUMING IMAGE IS IN BGR FORMAT
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		return CCM_matrix, image

	def run_loop(self):
		print("Instructions:")
		print("  - Left-click: Add a new square centered where you click.")
		print("  - 't': Enlarge the last created square by 10 px.")
		print("  - 'g': Shrink the last created square by 10 px (min side = 10).")
		print("  - 'ESC': Remove the last square.")
		print("  - 'q': Quit.")
		self.user_squares = []
		while True:
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == 27:  # ESC key code
				# Remove last square if available self.display_imgs
				if self.user_squares:
					self.user_squares.pop()
					self.draw_user_squares()
			elif key == ord('t'):
				# Enlarge last square by 10
				self.square_ini_side += 10
				self.draw_user_squares()

			elif key == ord('g'):
				# Shrink last square by 10
				self.square_ini_side = max(10, self.square_ini_side - 10)
				self.draw_user_squares()
		print("Closing loop")

	def create_look_up_table(self, sample_gray, target_gray):
		# Add 1 so that the maximum value is included in the mapping
		sample_gray = np.concatenate((sample_gray, [1.0]))
		target_gray = np.concatenate((target_gray, [1.0]))
		sample_gray = np.sort(sample_gray)
		target_gray = np.sort(target_gray)
		lut = np.interp(np.linspace(0, 1, 256), sample_gray, target_gray)
		return lut

	def srgb_to_linear(self, srgb):
		# 'sRGB' in [0, 255] -> 'linear RGB' in [0.0, 1.0]
		ln_rgb = srgb.copy()
		ln_rgb = ln_rgb/255
		mask = ln_rgb > 0.04045
		ln_rgb[mask] = np.power((ln_rgb[mask] + 0.055) / 1.055, 2.4)
		ln_rgb[~mask] /= 12.92
		return ln_rgb

	def linear_to_srgb(self, linear):
		# 'linear RGB' in [0.0, 1.0] -> 'sRGB' in [0, 255]
		srgb = linear.copy()
		mask = srgb > 0.0031308
		srgb[mask] = 1.055 * np.power(srgb[mask], 1 / 2.4) - 0.055
		srgb[~mask] *= 12.92
		srgb = srgb * 255
		return np.clip(srgb, 0.0, 255.0).astype(np.uint8)

	def correct_image(self, image, lut = None, R = None,
					  CCM_matrix = None):
		
		if lut == None or R == None or CCM_matrix == None:
			# if calibration params are not specified the saved ones
			# are used
			lut = self.lut.copy()
			R = self.R.copy()
			CCM_matrix = self.CCM_matrix.copy()
			
		# [WARNING] ASSUMING IMAGE IS IN BGR FORMAT
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# linearize image
		image = self.srgb_to_linear(image)

		# Apply gain / white balance correction
		image = lut[(image * 255).astype(np.uint8)]
		image =  np.dot(image, R.T)

		# Apply Color Correction matrix to the original image
		image =  np.clip(np.dot(image, CCM_matrix.T), 0, 1)

		# Apply gamma to the image
		image = self.linear_to_srgb(image)

		# [WARNING] ASSUMING IMAGE IS IN BGR FORMAT
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		return image

	def distortion_calibration(self, pattern_size, pattern_square_size, file_path, alpha = 1):
		# Functions that calibrates the camera distortion using
		# taken images of a chessboard pattern
		

		try:
			mtx = np.load(file_path + "camera_mtx.npy")
			dist = np.load(file_path + "distortion_array.npy")
			new_mtx = np.load(file_path + "new_camera_mtx.npy")
			
			self.camera_mtx = mtx
			self.camera_dist = dist
			self.new_camera_mtx = new_mtx
		
			print("Saved parameters exist!, ignoring calibration")
			return mtx, dist
		except Exception as err:
			print(err)
			print("There are no saved parameters... searching for calibration folder")
		
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
		objpoints = []
		imgpoints = []
		objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
		objp[0,:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
		objp *= pattern_square_size
		prev_img_shape = None
		images = glob.glob(file_path)
		
		
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCornersSB(gray, pattern_size)
			print("Analizing: ", fname)
		
			if ret == True:
				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
				imgpoints.append(corners2)
				img = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
			
			#cv2.imshow("cal", img)
			#cv2.waitKey(0)
				
		
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,
														   gray.shape[::-1],
														   None, None)
		print("camera matrix : \n")
		print(mtx)
		print("Distortion Coefficients : \n")
		print(dist)

		(h, w, _) = img.shape
		new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
		
		np.save(file_path + "camera_mtx.npy", mtx)
		np.save(file_path + "distortion_array.npy", dist)
		np.save(file_path + "new_camera_mtx.npy", new_mtx)
		self.camera_mtx = mtx
		self.camera_dist = dist
		self.new_camera_mtx = new_mtx
		return mtx, dist, new_mtx


	def undistort_img(self, img, mtx = None, dist = None, new_mtx = None):
		if np.any(mtx):
			img = cv2.undistort(img, mtx, dist, None, new_mtx)
			return img
		else:
			if np.any(self.camera_mtx):
				img = cv2.undistort(img, self.camera_mtx, self.camera_dist, None, self.new_camera_mtx)
				return img
			else:
				print("there is no camera matrix as a parameter nor as a saved file")
				raise(Exception)

if __name__ == "__main__":
	helper = CameraCalibrationHelper()
	helper.initialize_raspicam(headless = True, sensor_index = 3)
	helper.calibrate_raspberry()
	cv2.destroyAllWindows()
	helper.picam2.stop_preview()
	helper.picam2.close()
	print("Camera stopped")