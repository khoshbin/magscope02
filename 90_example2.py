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



deClient = DEAPI.Client()
deClient.Connect("localhost", 13240)
cameras = deClient.ListCameras()
deClient.SetCurrentCamera(cameras[0])


## test for changedProperties  
def PrintChangedProps(changedProps, setPropName):
    for prop,propV in changedProps.items():
        print(f"setProperty: setPropName Changed prop name:{prop}, value:{propV}")

def TestChangedProperties(deClient):
    changedProps = {}
    ## SetProperty and pass an empty dictionary 
    deClient.SetPropertyAndGetChangedProperties("Hardware Binning X", 2, changedProps);
    ## Show all changed properties name and value 
    PrintChangedProps(changedProps, "Hardware Binning X");
    ## Clear the dictionary
    changedProps.clear();

    deClient.SetPropertyAndGetChangedProperties("Hardware Binning Y", 2, changedProps);
    PrintChangedProps(changedProps, "Hardware Binning Y");
    changedProps.clear();

    deClient.SetPropertyAndGetChangedProperties("Hardware ROI Offset X", 0, changedProps);
    PrintChangedProps(changedProps, "Hardware ROI Offset X");
    changedProps.clear();

    deClient.SetPropertyAndGetChangedProperties("Hardware ROI Offset Y", 0, changedProps);
    PrintChangedProps(changedProps, "Hardware ROI Offset Y");
    changedProps.clear();

    deClient.SetPropertyAndGetChangedProperties("Hardware ROI Size X", 512, changedProps);
    PrintChangedProps(changedProps, "Hardware ROI Size X");
    changedProps.clear();

    deClient.SetPropertyAndGetChangedProperties("Hardware ROI Size Y", 512, changedProps);
    PrintChangedProps(changedProps, "Hardware ROI Size Y");
    changedProps.clear();

    deClient.SetSWROIAndGetChangedProperties(100, 100, 400, 400, changedProps);
    PrintChangedProps(changedProps, "SetSWRoi");
    changedProps.clear();

    deClient.SetHWROIAndGetChangedProperties(100, 100, 300, 300, changedProps);
    PrintChangedProps(changedProps, "SetHWRoi");
    changedProps.clear();


def TestMovieBuffer(m_de):
    ## Set properties
    m_de.SetHWROI(0, 0, 512, 512);
    m_de.SetProperty("Hardware Binning X", 1);
    m_de.SetProperty("Hardware Binning Y", 1);
    m_de.SetProperty("Frame Time (nanoseconds)", 0.01 * 1000 * 1000 * 1000);
    m_de.SetProperty("Frame Count", 50);
    m_de.SetProperty("Autosave Movie", "Save");
    m_de.SetProperty("Autosave Integrated Movie Pixel Format", "uint8"); ## "Auto""uint8""uint16""float32"

    ## Set test pattern
    m_de.SetProperty("Test Pattern", "SW Frame Number");

    ## initialize 
    movieBufferInfo = m_de.GetMovieBufferInfo()
    if movieBufferInfo.imageDataType == DEAPI.DataType.DE8u:
        imageType = numpy.uint8
    elif movieBufferInfo.imageDataType == DEAPI.DataType.DE16u:
        imageType = numpy.uint16
    elif movieBufferInfo.imageDataType == DEAPI.DataType.DE32f:
        imageType = numpy.float32 

    ## Allocate movie buffers
    totalBytes = movieBufferInfo.headerBytes + movieBufferInfo.imageBufferBytes
    movieBuffer = bytearray(totalBytes)

    ## Start Acquisition
    m_de.StartAcquisition(1, True)

    ## Get movie buffer
    numberFrames = 0
    index = 0 
    status = DEAPI.MovieBufferStatus.OK
    tb_info = "" 

    success = True
    while status == DEAPI.MovieBufferStatus.OK and success:
        status,totalBytes,numberFrames,movieBuffer = m_de.GetMovieBuffer(movieBuffer, totalBytes, numberFrames)

        ## CovertToImage(movieBuffer, headerBytes, dataType, imageW, imageH, numberFrames);
        frameIndexArray = numpy.frombuffer(movieBuffer, numpy.longlong, offset=0, count = numberFrames)
        movieBuffer = numpy.frombuffer(movieBuffer, dtype=imageType,offset=movieBufferInfo.headerBytes,count=movieBufferInfo.imageH*movieBufferInfo.imageW*numberFrames)
        
        ## Verify the value

        for i in range(numberFrames):
            # Calculate the starting index for each 64-bit integer (8 bytes per integer)
            start_index = i * 8

            # Extract the 64-bit integer frameIndex using struct.unpack
            frame_index = frameIndexArray[i]

            # Extract the first pixel value
            first_pixel_value = movieBuffer[i * movieBufferInfo.imageW * movieBufferInfo.imageH]

            if frame_index != index:
                tb_info += f"\nError: The frame index should be {index}, but the actual frame index is {frame_index}"

            if first_pixel_value != index:
                tb_info += f"\nError: The first pixel value of frame index {i} should be {index}, but the actual value is {first_pixel_value}"

            success = success and (frame_index == index) and (first_pixel_value == index)

            index +=1 
            if not success:
                break

    if success:
        tb_info += "\nOnTestMovieBuffer : Check value passed!"
    else:
        tb_info += f"the failed index is {index} "

    return tb_info 


# main 
## TestChangedProperties(deClient)

res = TestMovieBuffer(deClient)
print(res)