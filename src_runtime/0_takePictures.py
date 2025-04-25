#!/usr/bin/env python3

import time
import os
from picamera2 import Picamera2
from datetime import datetime
from servo_control import InitServo, SwitchAngle
from cam_illum import set_illum_white

from common import SetupLogger

servo_iopin = 5
servo_min_duty = 2.5
servo_max_duty = 10.5

def capture_image(output_dir, filename=None, 
                 delay=1, sharpness=2.0, format='png'):
    """
    Capture high-quality PNG image from Raspberry Pi Camera with anti-blur measures
    
    Args:
        output_dir (str): Directory to save images
        filename (str): Optional custom filename
        resolution (tuple): Image resolution (width, height)
        delay (int): Seconds to wait for camera to stabilize
        sharpness (float): Image sharpness (0.0 to 10.0)
        format (str): 'png' or 'jpg' output format
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.{format.lower()}"
        
        output_path = os.path.join(output_dir, filename)
        print(output_path)
       
        print(f"Waiting {delay} seconds for camera to stabilize...")
        time.sleep(delay)
        
        # Additional anti-blur: force autofocus if available
        try:
            picam2.autofocus_cycle()
            print("Autofocus completed")
        except:
            print("Autofocus not available - using fixed focus")
                
        print(f"Capturing {format.upper()} image to {output_path}...")
        
        # Capture in chosen format
        capture_config = {"format": format.upper()}
        picam2.capture_file(output_path, **capture_config)
        
        print("High-quality image captured successfully!")
        return output_path
        
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None
    finally:
        if 'picam2' in locals():
            picam2.stop()
            print("Camera resources released")

if __name__ == "__main__":
    import argparse

    cwd = os.getcwd()
    print(cwd)
    if 'src_runtime' not in cwd:
        os.chdir('src_runtime')

    mylogger = SetupLogger("../Debug/mylogger.txt", outtoconsole=True)

    parser = argparse.ArgumentParser(description="Capture high-quality PNG images from Raspberry Pi Camera")
    parser.add_argument("-d", "--directory", default="../Images/Originals/Peukjes", help="Output directory (default Peukjes)")
    parser.add_argument("-r", "--resolution", nargs=2, type=int, default=[1920, 1080],
                        help="Image resolution (width height)")
    parser.add_argument("-w", "--wait", type=int, default=1,
                        help="Seconds to wait before capturing (reduces blur)")
    parser.add_argument("-s", "--sharpness", type=float, default=2.0,
                        help="Sharpness setting (0.0-10.0, default 2.0)")
    parser.add_argument("--png", action="store_true",
                        help="Save as PNG (default)")
    parser.add_argument("--jpg", action="store_true",
                        help="Save as JPG instead of PNG")

    args = parser.parse_args()
    img_format = 'jpg' if args.jpg else 'png'
    resolution=tuple(args.resolution)
    sharpness=args.sharpness

    print(f"Initializing camera (resolution: {resolution}, sharpness: {sharpness})...")
        
    picam2 = Picamera2()
        
    # Configure for higher quality with anti-blur settings
    config = picam2.create_still_configuration(
        main={"size": resolution},
        controls={
            "AwbEnable": False,
            "AeEnable": False,
            "AnalogueGain": 1.0,
            "Sharpness": sharpness,
            "ExposureTime": 10000,  # microseconds (helps reduce motion blur)
        }
    )
    picam2.configure(config)
        
    print("Starting camera...")
    picam2.start()

    myservo = InitServo(servo_iopin, servo_min_duty, servo_max_duty)
    SwitchAngle(myservo, servo_iopin)
    time.sleep(2)
    SwitchAngle(myservo, servo_iopin)

    set_illum_white(brightness=0.1)
    print("\nStarting image capture loop.")
    print("Press [Enter] to capture an image, or type 'q' and press [Enter] to quit.\n")

    while True:
        user_input = input("Ready? Press [Enter] to take a photo, or 'q' to quit: ").strip().lower()
        if user_input in ["q", "quit", "exit"]:
            print("Exiting image capture loop.")
            break

        result = capture_image(
            output_dir=args.directory,
            filename=None,  # auto-generate
            delay=args.wait,
            format=img_format
        )

        if result:
            print(f"Image saved to: {result}\n")
        else:
            print("Failed to capture image.\n")

        SwitchAngle(myservo, pin=5)
        time.sleep(1)
        SwitchAngle(myservo, pin=5)
        time.sleep(1)

    set_illum_white(brightness=0.0)
