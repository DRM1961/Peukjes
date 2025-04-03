import asyncio
import cv2
import numpy as np
import aiomqtt
import pigpio
from multiprocessing import Process, Queue

# GPIO Setup
SERVO_PIN = 5
LED_GREEN = 23
LED_RED = 24
pi = pigpio.pi()

# MQTT Setup
MQTT_BROKER = "192.168.x.x"  # Replace with your MQTT broker IP
MQTT_TOPIC = "classification/result"

# Async Queues for triggers
motion_detected = asyncio.Queue()
image_ready = asyncio.Queue()
classification_done = asyncio.Queue()

def classify_image(image, queue):
    #Neural network classification process (runs in a separate process).
    result = "left" if np.mean(image) > 127 else "right"  # Placeholder classifier
    queue.put(result)

async def detect_motion():
    #Detects movement and triggers the image capture.
    cap = cv2.VideoCapture(0)
    _, prev_frame = cap.read()

    while True:
        await asyncio.sleep(0.1)  # Non-blocking delay
        _, frame = cap.read()
        diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))

        if np.sum(diff) > 50000:  # Example threshold for motion detection
            await motion_detected.put(frame)

        prev_frame = frame.copy()

async def capture_image():
    #Waits for motion trigger, increases illumination, and captures an image.
    while True:
        frame = await motion_detected.get()
        set_illumination(True)
        await asyncio.sleep(0.2)  # Simulate lighting adjustment
        await image_ready.put(frame)
        set_illumination(False)

async def evaluate_image():
    #Sends the captured image for classification in a separate process.
    queue = Queue()
    while True:
        frame = await image_ready.get()
        p = Process(target=classify_image, args=(frame, queue))
        p.start()
        p.join()
        result = queue.get()
        await classification_done.put(result)

async def sort_object():
    #Receives classification result and moves the servo accordingly.
    while True:
        result = await classification_done.get()
        if result == "left":
            move_servo(1000)  # Example PWM value
            set_led(LED_GREEN, True)
        else:
            move_servo(2000)
            set_led(LED_RED, True)

        await asyncio.sleep(1)
        set_led(LED_GREEN, False)
        set_led(LED_RED, False)

async def mqtt_publish():
    #Publishes classification results via MQTT.
    async with aiomqtt.Client(MQTT_BROKER) as client:
        while True:
            result = await classification_done.get()
            await client.publish(MQTT_TOPIC, result)

def set_illumination(state):
    #Controls illumination intensity => to be implemented
    print("Illumination ON" if state else "Illumination OFF")

def move_servo(position):
    #Moves the sorting servo => take from original
    pi.set_servo_pulsewidth(SERVO_PIN, position)

def set_led(pin, state):
    #Controls LED indicator.
    pi.write(pin, 1 if state else 0)

async def main():
    #Runs all tasks concurrently
    await asyncio.gather(
        detect_motion(),
        capture_image(),
        evaluate_image(),
        sort_object(),
        mqtt_publish()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pi.stop()
