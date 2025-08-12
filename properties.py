import sys
import struct
import types
import time

sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
import DEAPI

deClient = DEAPI.Client()
deClient.Connect()

totalTime = 0
totalProps = 0

for camera in deClient.ListCameras():
    deClient.SetCurrentCamera(camera)
    camera_properties = deClient.ListProperties("")

    for name in camera_properties:
        t0 = deClient.GetTime()   
        value = deClient.GetProperty(name)
        durSec = deClient.GetTime() - t0
        totalTime += durSec
        totalProps += 1

        print("%-56s: %-30s %-6.2f ms" % (name, value, durSec*1000))

deClient.Disconnect()

print ("\n%d properties, %f.2 ms, avg: %.2f ms" % (totalProps, totalTime*1000, totalTime*1000 / totalProps))
