import sys
sys.path.append("/home/kevinhardin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games")
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
from time import sleep, time
from datetime import datetime, date
from PIL import Image
import os
os.environ["LIBCAMERA_LOG_LEVELS"] = "2"
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx708_noir.json")
camera = Picamera2(tuning=tuning)
os.system("v4l2-ctl --set-ctrl wide_dynamic_range=1 -d /dev/v4l-subdev0")
capture_config = camera.create_still_configuration(main = {"size": (1920,1080)})
camera.configure(capture_config)
camera.start()

# Give time to settle before changing settings
sleep(1)
camera.controls.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfMetering": controls.AfMeteringEnum.Windows, "AfWindows": [ (527,876,128,122) ], "FrameRate": 1.0})
# Wait for settings to take effect
sleep(1)

current_datetime = datetime.now()
current_hour = int(current_datetime.strftime("%H"))
current_filename = "Images/image" + current_datetime.strftime("_%y-%m-%d_%H_%M_%S") + ".png"
image = camera.capture_array()
im = Image.fromarray(image)
im.save(current_filename)
