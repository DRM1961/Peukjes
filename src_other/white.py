import time
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write

# Configuration
NEOPIXEL_PIN = board.D13  # GPIO pin connected to NeoPixels
NUM_PIXELS = 12           # Number of LEDs in your ring
BRIGHTNESS = 0.5          # Set brightness here (0.0 to 1.0)

# Initialize LED buffer
led_buffer = bytearray(NUM_PIXELS * 3)

def set_white(brightness):
    """Fill all LEDs with white at specified brightness"""
    value = int(255 * max(0.0, min(1.0, brightness)))  # Clamp and scale brightness
    for i in range(NUM_PIXELS):
        # Using GRB order (common for NeoPixels)
        led_buffer[i*3] = value    # Green
        led_buffer[i*3+1] = value  # Red
        led_buffer[i*3+2] = value  # Blue
    neopixel_write(NEOPIXEL_PIN, led_buffer)

try:
    # Set initial brightness (change this value as needed)
    set_white(BRIGHTNESS)
    
    # Keep the program running
    while True:
        time.sleep(1)

finally:
    # Turn off LEDs when program exits
    set_white(0)