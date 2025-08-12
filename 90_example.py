import re
import os
import sys
import numpy
import struct
import types
import time
import math
from PIL import Image
from time import sleep
from datetime import datetime
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
import DEAPI

def WaitAcqFinished():
    while(True):
        if deClient.GetProperty("Acquisition Status") != "Acquiring":
            break




def Display(image, attributes, histogram):
    # implement your own function to display the image and/or the image statistics
    return 


def CoolDown():
    deClient.SetProperty("Temperature - Control", "Cool Down")
    while True:
        tempratureStatus   = deClient.GetProperty("Temperature - Detector Status")
        detectorTemprature = deClient.GetProperty("Temperature - Detector (Celsius)")
        print ("Detector Temprature: %s: %.1fC" % (tempratureStatus, detectorTemprature))
        if tempratureStatus == "Cooled":
            break;
        else:
            sleep(1) # wait for 1 second


def WarmUp():
    deClient.SetProperty("Temperature - Control", "Warm Up")
    while True:
        tempratureStatus   = deClient.GetProperty("Temperature - Detector Status")
        detectorTemprature = deClient.GetProperty("Temperature - Detector (Celsius)")
        print ("Detector Temprature: %s: %.1fC" % (tempratureStatus, detectorTemprature))
        if tempratureStatus == "Warmed":
            break;
        else:
            sleep(1) # wait for 1 second


def TakeDarkReference(frameRate=20):
    sys.stdout.write("Taking dark references: ")
    sys.stdout.flush()

    prevExposureMode = deClient.GetProperty("Exposure Mode")
    prevExposureTime = deClient.GetProperty("Exposure Time (seconds)") 

    acquisitions = 10
    deClient.SetProperty("Exposure Mode", "Dark")
    deClient.SetProperty("Frames Per Second", frameRate)
    deClient.SetProperty("Exposure Time (seconds)", 1)
    deClient.StartAcquisition(acquisitions)

    while(True):
        attributes = DEAPI.Attributes()
        histogram  = DEAPI.Histogram()
        image = deClient.GetResult(DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.FLOAT32, attributes, histogram)

        sys.stdout.write(str(attributes.acqIndex) + " ")
        sys.stdout.flush()
        Display(image, attributes, histogram)

        if (attributes.acqIndex >= acquisitions - 1):
            print("done.")
            break
            
    WaitAcqFinished()

    deClient.SetProperty("Exposure Mode", prevExposureMode)
    deClient.SetProperty("Exposure Time (seconds)", prevExposureTime) 
        

def TakeIntegratingTrial(frameRate=20):
    deClient.SetProperty("Image Processing - Mode", "Integrating")
    deClient.SetProperty("Exposure Mode", "Trial")
    deClient.SetProperty("Frames Per Second", frameRate)
    deClient.SetProperty("Exposure Time (seconds)", 0)

    deClient.StartAcquisition(1000)

    minIntensity = 0.5
    maxIntensity = 4
    okCount = 0;
    print("Trial for integrating beam condition ...")

    while(True):
        attributes = DEAPI.Attributes()
        histogram  = DEAPI.Histogram()
        image = deClient.GetResult(DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.UINT16, attributes, histogram)
        intensity = attributes.eppixpf

        if intensity < minIntensity:
            print("Trial for integrating: beam too dim \n")
            okCount = 0
        elif intensity > maxIntensity:
            print("Trial for integrating: beam too bright \n");
            okCount = 0
        else:
            okCount += 1
        
        if okCount > frameRate * 5:
            deClient.StopAcquisition()
            print("done.")
            break

    WaitAcqFinished()


def TakeCountingTrial(frameRate=280, sparsenessLowerLimit=0.001, sparsenessUpperLimit=0.02):
    deClient.SetProperty("Image Processing - Mode", "Counting")
    deClient.SetProperty("Exposure Mode", "Trial")
    deClient.SetProperty("Frames Per Second", frameRate)
    deClient.SetProperty("Exposure Time (seconds)", 0)

    deClient.StartAcquisition(1000)

    okCount = 0;
    print("Trial for counting beam condition ...")

    while(True):
        attributes = DEAPI.Attributes()
        histogram  = DEAPI.Histogram()
        image = deClient.GetResult(DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.UINT16, attributes, histogram)
        Display(image, attributes, histogram)

        sparseness = attributes.eppixpf             
        if sparseness >= sparsenessLowerLimit and sparseness <= sparsenessUpperLimit :
            print("Trial for counting beam condition, sparseness range: [%d, %d], sparseness: %d%%, OK" % (sparsenessLowerLimit, sparsenessUpperLimit, sparseness))
            okCount += 1
        elif sparseness < sparsenessLowerLimit:
            print("Trial for counting beam condition, sparseness range: [%d, %d], sparseness: %d%%, too sparse" % (sparsenessLowerLimit, sparsenessUpperLimit, sparseness))
            okCount = 0
        else:
            print("Trial for counting beam condition, sparseness range: [%d, %d], sparseness: %d%%, too dense" % (sparsenessLowerLimit, sparsenessUpperLimit, sparseness))
            okCount = 0

        if okCount > frameRate * 5:
            deClient.StopAcquisition()
            print("done.")
            break

    WaitAcqFinished()



def TakeIntegratingGain(frameRate=20):
    deClient.SetProperty("Image Processing - Mode", "Integrating")
    deClient.SetProperty("Exposure Mode", "Gain")
    deClient.SetProperty("Frames Per Second", frameRate)

    gainExposureTime = deClient.GetProperty("Reference - Integrating Gain Exposure Time (seconds)")
    gainAcquisitions = deClient.GetProperty("Reference - Integrating Gain Acquisitions")

    deClient.SetProperty("Exposure Time (seconds)", gainExposureTime)
    deClient.StartAcquisition(gainAcquisitions)

    print("Taking integrating gain references")

    while(True):
        attributes = DEAPI.Attributes()
        histogram  = DEAPI.Histogram()
        image = deClient.GetResult(DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.FLOAT32, attributes, histogram)
        Display(image, attributes, histogram)
        print("integrating gain ref %d of %d" % (attributes.acqIndex + 1, gainAcquisitions))

        if attributes.acqIndex >= gainAcquisitions - 1:
            deClient.StopAcquisition()
            print("done.")
            break
    
    WaitAcqFinished()

def TakeCountingGain(frameRate=280):
    deClient.SetProperty("Image Processing - Mode", "Counting")
    deClient.SetProperty("Exposure Mode", "Gain")
    deClient.SetProperty("Frames Per Second", frameRate)

    gainExposureTime = deClient.GetProperty("Reference - Counting Gain Exposure Time (seconds)")
    gainAcquisitions = deClient.GetProperty("Reference - Counting Gain Acquisitions")

    deClient.SetProperty("Exposure Time (seconds)", gainExposureTime)
    deClient.StartAcquisition(gainAcquisitions)

    print("Taking counting gain references")

    while(True):
        attributes = DEAPI.Attributes()
        histogram  = DEAPI.Histogram()
        image = deClient.GetResult(DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.FLOAT32, attributes, histogram)
        Display(image, attributes, histogram)
        print("counting gain ref %d of %d" % (attributes.acqIndex + 1, gainAcquisitions))

        if attributes.acqIndex >= gainAcquisitions - 1:
            deClient.StopAcquisition()
            print("done.")
            break\

    WaitAcqFinished() 


def TakeDataImages(frameRate=20, exposureTimeSecs=1, countingMode=False, sumCount=1, numAcquisions=1):
    deClient.SetProperty("Exposure Mode", "Normal")
    deClient.SetProperty("Frames Per Second", frameRate)
    deClient.SetProperty("Exposure Time (seconds)", exposureTimeSecs)

    frameType = DEAPI.FrameType.SUMTOTAL

    deClient.SetProperty("Autosave Movie - Sum Count", sumCount)
    deClient.SetProperty("Autosave Movie",       "On") 
    deClient.SetProperty("Autosave Final Image", "On") 

    deClient.StartAcquisition(numAcquisions)

    while(True):
        remainingAcqs = deClient.GetProperty("Remaining Number of Acquisitions")
        sys.stdout.write(str(remainingAcqs) + " ")
        sys.stdout.flush()

        pixelFormat = DEAPI.PixelFormat.AUTO
        attributes = DEAPI.Attributes()
        histogram  = DEAPI.Histogram()
        image = deClient.GetResult(frameType, pixelFormat, attributes, histogram)
        print("DataImage: frameType=%s, pixelFormat=%s, datasetName=%s, acqCount=%d" % (frameType, pixelFormat, attributes.datasetName, attributes.acqIndex))

        Display(image, attributes, histogram)

        if (remainingAcqs == 0):
            print("done.")
            break

    WaitAcqFinished()

#  1. Manually Power on camera.

#  2. Manually Start DE-de.

#  3. Connect client software to DE-Server
deClient = DEAPI.Client()
deClient.Connect("localhost", 13240)
cameras = deClient.ListCameras()
deClient.SetCurrentCamera(cameras[0])
deClient.PrintServerInfo(cameras[0])

#  4. Use client software to cool down the camera and wait until the camera temperature has stabilized.
#CoolDown()

#  5. Use client software to acquire a dark reference for integrating mode (20 fps). 
#     Each individual dark reference acquisition should be displayed in the client software 
#     while the dark reference is being acquired.TakeDarkReference(20)
TakeDarkReference(20)

#  6. Use client software to acquire a dark reference for counting mode (280 fps). 
#     Each individual dark reference acquisition should be displayed in the client software 
#     while the dark reference is being acquired.TakeDarkReference(20)
##TakeDarkReference(280)

#  7. Use client software to acquire trial images for integrating mode (20 fps) 
#     and set the microscope beam intensity to the optimal value for this mode. 
#     The trial images should be displayed in the client software.
TakeIntegratingTrial(20)

#  8. Use client software to acquire a gain reference for integrating mode (20 fps). 
#     Each individual gain reference acquisition should be displayed in the client software 
#     while the gain reference is being acquired.
TakeIntegratingGain(20)

#  9. Use client software to acquire trial images for counting mode (280 fps) 
#     and set the microscope beam intensity to the optimal value for this mode. 
#     The trial images should be displayed in the client software.
##TakeCountingTrial(280)

# 10. Use client software to acquire a gain reference for counting mode (280 fps). 
#     Each individual gain reference acquisition should be displayed in the client software
#     while the gain reference is being acquired.
##TakeCountingGain(280)

# 11. Use client software to acquire an exposure in integrating mode, 
#     saving the final image and a movie of all processed frames, for a 2 second exposure. 
#     The final image should be displayed in the client software.
TakeDataImages(20, 2, False, 1, 1)

# 12. Use client software to repeat the acquisition three times, 
TakeDataImages(20, 2, False, 1, 3)

# 13. Use client software to acquire an exposure in counting mode, 
#     saving the final image and a movie of the sum of every 20 frames, for a 60 second exposure. 
#     The final image should be displayed in the client software.
##TakeDataImages(280, 60, True, 20, 1)

# 14. Use client software to repeat the acquisition three times, 
##TakeDataImages(280, 60, True, 20, 3)

# 15. Use client software to acquire an exposure in integrating mode, 
#     saving the final image and a movie of all processed frames, for a 2 second exposure. 
#     The final image should be displayed in the client software.
TakeDataImages(20, 2, False, 1, 1)

# 16. Use client software to acquire a new dark reference for counting mode (280 fps).
##TakeDarkReference(280)

# 17. Use client software to acquire an exposure in counting mode, 
#   saving the final image and a movie of the sum of every 20 frames, for a 60 second exposure. 
#   The final image should be displayed in the client software.
##TakeDataImages(280, 60, True, 20, 1)

# 18. Use client software to warm up the camera.
WarmUp()

# 19. Disconnect client software from DE-Server
deClient.Disconnect()

# 20. Manually Power off the camera
