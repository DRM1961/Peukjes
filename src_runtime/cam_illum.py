import time
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write
from picamera2 import Picamera2
from libcamera import controls
import cv2
from common import mylogger

# Configuration
NEOPIXEL_PIN = board.D13  # GPIO pin connected to NeoPixels
NUM_PIXELS = 12           # Number of LEDs in your ring
BRIGHTNESS = 0.5          # Set brightness here (0.0 to 1.0)

# Initialize LED buffer
led_buffer = bytearray(NUM_PIXELS * 3)

def clip_brightness(brightness):
    brightness = min(1.0, brightness)
    brightness = max(0.0, brightness)
    return brightness

def set_illum_white(brightness):
    clip_brightness(brightness)
    """Fill all LEDs with white at specified brightness"""
    value = int(255 * max(0.0, min(1.0, brightness)))  # Clamp and scale brightness
    for i in range(NUM_PIXELS):
        # Using GRB order (common for NeoPixels)
        led_buffer[i*3] = value    # Green
        led_buffer[i*3+1] = value  # Red
        led_buffer[i*3+2] = value  # Blue
    neopixel_write(NEOPIXEL_PIN, led_buffer)

def set_illum_red(brightness):
    clip_brightness(brightness)
    """Fill all LEDs with white at specified brightness"""
    value = int(255 * max(0.0, min(1.0, brightness)))  # Clamp and scale brightness
    for i in range(NUM_PIXELS):
        # Using GRB order (common for NeoPixels)
        led_buffer[i*3] = 0    # Green
        led_buffer[i*3+1] = value  # Red
        led_buffer[i*3+2] = 0  # Blue
    neopixel_write(NEOPIXEL_PIN, led_buffer)

def set_illum_green(brightness):
    clip_brightness(brightness)
    """Fill all LEDs with white at specified brightness"""
    value = int(255 * max(0.0, min(1.0, brightness)))  # Clamp and scale brightness
    for i in range(NUM_PIXELS):
        # Using GRB order (common for NeoPixels)
        led_buffer[i*3] = value    # Green
        led_buffer[i*3+1] = 0  # Red
        led_buffer[i*3+2] = 0  # Blue
    neopixel_write(NEOPIXEL_PIN, led_buffer)

def flash_illum(color, flashtime):
    starttime = time.time()
    endtime = time.time() + flashtime
    while(1):
        now = time.time()
        if now>endtime:
            break

        if int(now*10)%2:
            brightness = 0.0
        else:
            brightness = 0.5

        if color=='green':
            set_illum_green(brightness)
        elif color=='red':
            set_illum_red(brightness)
        if color=='white':
            set_illum_white(brightness)



def init_camera():
    img_format = 'png'
    resolution = [1920, 1080]
    sharpness = 2.0

    cam = Picamera2()
    
    # Configure for higher quality with anti-blur settings
    config = cam.create_still_configuration(
        main={"size": resolution, "format": "RGB888"},
        controls={          
            "AwbEnable": False,
            "AeEnable": True,
            "AnalogueGain": 1.0,
            "Sharpness": sharpness,
            "ExposureTime": 10000,  # microseconds (helps reduce motion blur)
        }
    )
    cam.configure(config)
    #mylogger.info(f'cam info keys {cam.camera_ctrl_info.keys()}')
    #mylogger.info(f'cam sensor modes {cam.sensor_modes}')
    #cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 10.0})
    #cam.set_controls({"ScalerCrop": scaler_crop})
        
    mylogger.info("Starting camera...")
    cam.start()

    return cam



if __name__ == '__main__':


    cam = init_camera()

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    print('check position of cup below the camera; press ESC to exit and start processing real images')
    focussed = False
    brightness = 0.0
    teststage = 0
    starttime = time.time()
    endtime = time.time() + 1
    while True:
        if not focussed:
            try:
                set_illum_white(0.1)
                cam.autofocus_cycle()
                print("Autotfocus done")
                set_illum_white(0.0)
            except:
                print("Autofocus not available - using fixed focus")
            focussed = True

        now = time.time()
        if now > endtime:
            if brightness >= 0.7:
                brightness = 0.0
                teststage += 1
                continue
            else:
                brightness += 0.3
                endtime = time.time() + 1

            if teststage==1:                
                set_illum_green(brightness)
            elif teststage==2:
                flash_illum('green', 0.5)
            elif teststage==3:
                set_illum_red(brightness)
            elif teststage==4:
                flash_illum('red', 0.5)
            elif teststage==5:
                set_illum_white(brightness)
            elif teststage==6:
                flash_illum('white', 0.5)
            elif teststage>6:
                set_illum_white(0.1)
                break
        
        
        
        frame = cam.capture_array()

        cv2.imshow('img', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if cv2.getWindowProperty('img', 1) < 0:
            break

    print(frame.shape)
    set_illum_white(0.1)
    time.sleep(2)
    while True:
        frame = cam.capture_array()

        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    set_illum_white(0.0)
