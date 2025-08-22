import sys
sys.path.append("/home/kevinhardin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games")
sys.path.append("CameraScript")
from CameraScript import takePic
from time import sleep, time
from datetime import datetime, date
import os
os.environ["LIBCAMERA_LOG_LEVELS"] = "2"
import warnings
warnings.filterwarnings('ignore')

while True:
    current_datetime = datetime.now()
    current_hour = int(current_datetime.strftime("%H"))
    if 8 <= current_hour <= 18:
        current_filename = "Images/image" + current_datetime.strftime("_%y-%m-%d_%H_%M_%S") + ".png"
        im, _, _ = takePic()
        im.save(current_filename)
        sleep(3600)
    else:
        sleep(600)
