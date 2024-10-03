import sys
sys.path.append("/home/kevinhardin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games")
import numpy as np
from picamera2 import Picamera2
from time import sleep, time
from datetime import datetime, date
from PIL import Image
import os
os.environ["LIBCAMERA_LOG_LEVELS"] = "2"
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json")
camera = Picamera2(tuning=tuning)
capture_config = camera.create_still_configuration(main = {"size": (3200,2404)})
camera.configure(capture_config)
camera.start()

# Give time to settle before changing settings
sleep(1)
camera.set_controls({"FrameRate": 3600.0})
# Wait for settings to take effect
sleep(1)

current_hour = 7

while True:
    current_datetime = datetime.now()
    current_hour = int(start_datetime.strftime("%H"))
    if 8 <= current_hour <= 18:
        current_filename = "~/GEORGE/Images/image" + current_datetime.strftime("_%y-%m-%d_%H_%M_%S") + ".png"
        image = camera.capture_array()
        im = Image.fromarray(image)
        im.save(current_filename)
    else:
        sleep(600)
