
import yaml
import os

CONFIG_FILE = "config.yaml"

class CConfig:
    def __init__(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {
                'mycounter': 1,
                'headless': False,
                'mqtt': {
                    'broker': '172.21.235.33',
                    'port': 1883,
                    'pw': 'client',
                    'user': 'rpi6'
                },
                'servo': {
                    'iopin': 5,
                    'min_duty': 2.2,
                    'max_duty': 9.5,
                },
                'illum': {
                    'brightness_autofocus': 0.1,
                    'brightness_detect': 0.1,
                    'brightness_measure': 0.1,
                    'flashtime': 1.0,
                },
                'detection': {
                    'method': 'profiles',
                    'min_mean_difference': 10,
                    'min_area_difference': 50,
                    'crop_x': 80,
                    'crop_y': 0,
                    'crop_w': 480,
                    'crop_h': 480,
                }
            }
            with open(CONFIG_FILE, "w") as file:
                yaml.safe_dump(self.config, file)

    def GetMyCounter(self):
        return(self.config['mycounter'])
    def GetHeadlessMode(self):
        return(self.config['headless'])
    def GetMQTT(self):
        return(self.config['mqtt'])
    def GetServo(self):
        return(self.config['servo'])
    def GetIllum(self):
        return(self.config['illum'])
    def GetDetectionMethod(self):
        return(self.config['detection']['method'])
    def GetDetectionThreshold(self):
        return({'profile_difference': self.config['detection']['min_mean_difference'],
                'area_difference': self.config['detection']['min_area_difference']
                })
    def GetDetectionCroppingArea(self):
        return({'crop_x': self.config['detection']['crop_x'],
                'crop_y': self.config['detection']['crop_y'],
                'crop_w': self.config['detection']['crop_w'],
                'crop_h': self.config['detection']['crop_h']
                })

if __name__ == "__main__":
    myconfig = CConfig()
    mycounter = myconfig.GetMyCounter()
    print(mycounter)
    headless = myconfig.GetHeadlessMode()
    print(headless)
    mqtt = myconfig.GetMQTT()
    print(mqtt)
    servo = myconfig.GetServo()
    print(servo)
    illum = myconfig.GetIllum()
    print(illum)
    brightness_autofocus = illum['brightness_autofocus']
    detection_method = myconfig.GetDetectionMethod()
    print(detection_method)
    detection_th = myconfig.GetDetectionThreshold()
    print(detection_th)
    print(detection_th['area_difference'])
    cropping = myconfig.GetDetectionCroppingArea()
    print(cropping)
    print(cropping['crop_x'])
