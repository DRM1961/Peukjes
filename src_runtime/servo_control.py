#Set up libraries and overall settings
import RPi.GPIO as GPIO  # Imports the standard Raspberry Pi GPIO library 
# RPI 5 doesn't have GPIO library
# pip uninstall RPi.GPIO 
# and install lgpio instead
# pip install rpi.lgpio 
# check alternative 
# pip install gpiozero
# you get a warning to use pigpio, ignore it, this library uses the pigpio daemon which is not supported on RPI 5
# pigpio daemon is not supported on RPI 5: https://github.com/joan2937/pigpio/issues/589

from gpiozero import Servo, AngularServo
from time import sleep   # Imports sleep (aka wait or pause) into the program

from common import mylogger

pin = 5
last_angle = 0
freq = 50
minduty = 2.0
maxduty = 12.0
midduty = (maxduty+minduty)/2
#see observations in LIMIT_ANGLE_FINE to see how I got these values
if freq==50:
    minduty = 2.0
    maxduty = 12.0
    midduty = (maxduty+minduty)/2
elif freq==60:  
    minduty = 2.5
    maxduty = 14.5
    midduty = (maxduty+minduty)/2
else:    
    print("freq not supported")
    freq = 50
    minduty = 2.0
    maxduty = 12.0
    midduty = (maxduty+minduty)/2

def InitServo(pin, config_min_duty=2.0, config_max_duty=12.0):
    global minduty, maxduty, midduty
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, freq)
    pwm.start(0)

    minduty = config_min_duty
    maxduty = config_max_duty
    midduty = (maxduty+minduty)/2
    sleep(2)
    SetAngle(pwm, pin, 0)   
    sleep(1)
    return pwm

def SetAngle(pwm, pin, angle):
	duty = angle * (maxduty-minduty)/180 + minduty
	GPIO.output(pin, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1.0)
	GPIO.output(pin, False)
	pwm.ChangeDutyCycle(0)

def SetDutyCycle(pwm, pin, duty):
	GPIO.output(pin, True)
	pwm.ChangeDutyCycle(duty)
	sleep(0.5) #with 1 sec there is more jitter and the motor can even jump 45degrees
	GPIO.output(pin, False)
	pwm.ChangeDutyCycle(0)

def SwitchAngle(pwm, pin):
    global last_angle
    if last_angle == 0:
        last_angle = 180
    else:
        last_angle = 0
    SetAngle(pwm, pin, last_angle)
    mylogger.info(f'new angle = {last_angle}')


if __name__ == "__main__":
    tests = ["LIMIT_CHECK", "LIMIT_ANGLE_FINE", "DUTY_CHECK", "SWITCH_ANGLE", "SERVO", "ANGULAR_SERVO"]
    test = tests[0]

    if test=="LIMIT_CHECK":    
        pwm = InitServo(pin)
        duty = 0

        #check which limits are moving the motor FOR A GIVEN FREQ: I used 60Hz
        #check  the motor each time to see if it is moving or not and if the angle is stable
        #e.g. for 60Hz, the motor moves from duty=20/10 to duty=150/10
        #  after a move below 20/10 and above 150/10, the next move is unstable
        for duty in range(0,180,5):
            print(f'Move to approx center with duty: 8.5')
            SetDutyCycle(pwm, pin, 8.5)
            sleep(1)
            print(f'Duty: {duty/10}')
            SetDutyCycle(pwm, pin, duty/10)
            sleep(2)
        pwm.stop()

    if test=="LIMIT_ANGLE_FINE":
        pwm = InitServo(pin)
        SetDutyCycle(pwm, pin, 8.5)
        print(f'Move to approx center with duty: 8.5')
        sleep(2)
        print("go to debug console and enter SetDutyCycle(pwm, pin, <duty>) to move the motor, duty between 2.0 and 15.0")
        print("see at which duty values the motor is at 0, 90, 180 degrees")
        print("each time, move thet motor first to 8.5 to get it to the center")

        #observations using freq=50 and freq=60
        #values for pulse width on spec sheet of my motor DS-R005 are:
        # min pulse width = 0.5ms, max pulse width = 2.5ms
        #for 50Hz, one period = 20ms
        #so min pulse width = 0.5ms/20ms = 2.5%, max pulse width = 2.5ms/20ms = 12.5%
        #so minduty=2.5%, maxduty=12.5%
        #we observe that this is not 100% correct
        #at 50Hz, duty=2.0% corresponds to 0 degr, duty 12.0% = 180 degr, duty 7.0% = 90 degr 
        #so minduty=2.0%, maxduty=12.0%

        #for 60Hz, one period = 16.666ms
        #so min pulse width = 0.5ms/16.666ms = 3.0%, max pulse width = 2.5ms/16.666ms = 15.0%
        #so minduty=3.0%, maxduty=15.0%
        #we observe that this is not 100% correct
        #at 60Hz, duty=2.5% corresponds to 0 degr, duty 14.5% = 180 degr, duty 8.5% = 90 degr
        #so minduty=2.5%, maxduty=14.5%

        pwm.stop()

    if test=="DUTY_CHECK":
        pwm = InitServo(pin)
        duty = 0
        #check that switching between the limits works smoothly
        #check if the motor moves stable from 0 to 90 to 180 and back
        for i in range(5):
            duty = minduty
            print(f'Duty: {duty}')
            SetDutyCycle(pwm, pin, duty)
            sleep(2)
            duty = midduty
            print(f'Duty: {duty}')
            SetDutyCycle(pwm, pin, duty)
            sleep(2)
            duty = maxduty
            print(f'Duty: {duty}')
            SetDutyCycle(pwm, pin, duty)
            sleep(2)
            duty = midduty
            print(f'Duty: {duty}')
            SetDutyCycle(pwm, pin, duty)
            sleep(2)
        pwm.stop()

    if test=="SWITCH_ANGLE":
        #test the runtime need to switch between 0 and 180 degrees
        pwm = InitServo(pin)

        for i in range(10):
            SwitchAngle(pwm, pin)
            sleep(2)

        pwm.stop()


    if test=="SERVO":
        #test the servo library from gpiozero
        servo = Servo(pin)
        for i in range(5):
            servo.min()
            sleep(1)
            servo.mid()
            sleep(1)
            servo.max()
            sleep(1)

    if test=="ANGULAR_SERVO": 
        #test the amgularservo library from gpiozero
        #even though it says min_angle=-180, max_angle=180, it moves only -90 to +90 
        #if specified as min_angle=-90, max_angle=90, it moves -45 to +45
        #values for pulse width come from spec sheet of my motor DS-R005
        servo = AngularServo(5, min_angle=-180, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
        val = -10
        diff = 1

        servo.min()
        sleep(2)
        servo.mid()
        sleep(2)
        servo.max()
        sleep(2)

        while True:
            servo.value = val/10
            sleep(0.5)
            if val >= 10 :
                diff = -1
            elif val <= -10:
                diff = 1
            val += diff
