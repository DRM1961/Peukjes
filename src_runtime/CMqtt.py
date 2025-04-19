# https://pypi.org/project/paho-mqtt/
# sudo apt install python3-paho-mqtt
# mosquitto_sub -d -t testTopic -u rpi6 -P client
# mosquitto_pub -d -t  testTopic -m "counter-value"  -u dm -P client
# mosquitto_pub -d -t  topicFile -f ~/dev/todo.txt -u mqtt_admin -P admin


import paho.mqtt.client as mqtt
import time
import cv2
import os

broker_IP="172.21.235.38"
#broker_IP="192.168.129.82"

port = 1883
keepalive = 60

class CMqttClient:
    def __init__(self, userdata, user, pw):
        self.counter = [0,0]
        self.user = user
        self.pw = pw
        self.userdata = userdata
        self.data = []
        self.isFile = False
        self.filename = ""
        self.connected = False
        self.subscribed = False
        #TODO: set an LED to indicate connection status
        try:
            #self.client = mqtt.Client(userdata=userdata)  
            self.client = mqtt.Client()  
            self.client.username_pw_set(username=self.user,password=self.pw)
            self.client.on_connect = self.on_connect
            self.client.connect(broker_IP, port, keepalive) 
            self.client.loop_start() #loop_forever?
        except Exception as e:
            print(f"Error: {e}")
            self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        print(f"Connected client {client} userdata {userdata} flags {flags} return code {rc}")
        if rc == 0:
            self.connected = True

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print(f"UpdateMQTTSubscribed client {client} userdata {userdata} mid {mid} granted_qos {granted_qos}")
        self.subscribed = True

    def on_message(self, client, userdata, msg):
        if self.isFile:
            print(f"received client {client} userdata {userdata}")
            with open(self.filename, 'wb') as fd:
                fd.write(msg.payload)
        else:
            print(f"received client {client} userdata {userdata} msg {msg}")
            print(msg.topic+" "+str(msg.payload))
            if msg.topic=="set_counter/1":
                self.counter[0] = int(msg.payload)
            if msg.topic=="set_counter/2":
                self.counter[1] = int(msg.payload)

    def subscribe(self, topic, filename="", isFile=False):
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.filename = filename
        self.isFile = isFile
        self.client.subscribe(topic,0)

    def publish(self, topic, datastring, isImage=False, isFilename=False):
        if isImage:
            cv2.imwrite('debug/temp.jpg', datastring)
            isFilename = True
            datastring = 'debug/temp.jpg'

        if isFilename:
            file = open(datastring, "rb")
            filestring = file.read()
            byteArray = bytes(filestring)
            self.client.publish(topic=topic, payload=byteArray, qos=0)
        else:
            self.client.publish(topic=topic, payload=datastring, qos=0)

    def publish_via_cmd(self, topic, datastring, isImage=False, isFilename=False):
        if isImage:
            cv2.imwrite('debug/temp.jpg', datastring)
            isFilename = True
            datastring = 'debug/temp.jpg'
        if isFilename:
            file = open(datastring, "rb")
            filestring = file.read()
            byteArray = bytes(filestring)
            oscmd = f'mosquitto_pub -d -h {broker_IP} -t "{topic}" -f "{datastring}" -u {self.user} -P {self.pw}'
            #os.system(oscmd)
        else:
            oscmd = f'mosquitto_pub -d -h {broker_IP} -t "{topic}" -m "{datastring}" -u {self.user} -P {self.pw}'
            print(f'sending >{oscmd}<')
            os.system(oscmd)

    def increment_counter(self, i) -> int:
        self.counter[i-1] += 1
        return self.counter[i-1]


if __name__ == "__main__":
    mqtt = CMqttClient(userdata="dm", user="rpi6", pw="client")
    mycounter = 1
    count = mqtt.increment_counter(mycounter)
    count = 5

    if mqtt.connected:
        mqtt.publish(f"counter/{mycounter}", str(count))
        print(f"publishing counter {mycounter}: counter {count}")
    else:
        mqtt.publish_via_cmd(f"counter/{mycounter}", str(count)) 
        print(f"use command to publish counter {mycounter}: counter {count}")
        #TODO make connection LED blink a few times
