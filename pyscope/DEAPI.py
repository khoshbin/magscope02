import socket
import sys
import math
import struct
import types
import numpy
import time
import os
import mmap
import ctypes
import logging
from PIL import Image
from enum import Enum
from enum import IntEnum
import mmap
from datetime import datetime
from time import sleep
from version import version
import re


versionInfo = list(map(int,version.split(".")))

## the commandInfo contains [VERSION_MAJOR.VERSION_MINOR.VERSION_PATCH.VERSION_REVISION]
commandVersion = (versionInfo[0] - 4) * 10 + versionInfo[1]
pyVersion = sys.version.split("(")[0][0:4].split(".")
logLevel  = logging.DEBUG
logLevel  = logging.INFO
logLevel  = logging.WARNING
logging.basicConfig(format="%(asctime)s DE %(levelname)-8s %(message)s",level=logLevel)
log = logging.getLogger("DECameraClientLib")
log.info("Python    : " + sys.version.split("(")[0])
log.info("DEClient  : " + version)
log.info("CommandVer: " + str(commandVersion))
log.info("logLevel  : " + str(logging.getLevelName(logLevel)))

class FrameType(Enum):
    NONE                                   = 0
    AUTO                                   = 1
    CRUDEFRAME                             = 2
    SINGLEFRAME_RAWLEVEL0                  = 3
    SINGLEFRAME_RAWLEVEL1                  = 4
    SINGLEFRAME_RAWLEVEL2                  = 5
    SINGLEFRAME_RAWOFFCHIPCDS              = 6
    SINGLEFRAME_INTEGRATED                 = 7
    SINGLEFRAME_COUNTED                    = 8
    SUMINTERMEDIATE                        = 9
    SUMTOTAL                               = 10
    CUMULATIVE                             = 11
    VIRTUAL_MASK0                          = 12
    VIRTUAL_MASK1                          = 13
    VIRTUAL_MASK2                          = 14
    VIRTUAL_MASK3                          = 15
    VIRTUAL_MASK4                          = 16
    VIRTUAL_IMAGE0                         = 17
    VIRTUAL_IMAGE1                         = 18
    VIRTUAL_IMAGE2                         = 19
    VIRTUAL_IMAGE3                         = 20
    VIRTUAL_IMAGE4                         = 21
    EXTERNAL_IMAGE1                        = 22
    EXTERNAL_IMAGE2                        = 23
    EXTERNAL_IMAGE3                        = 24
    EXTERNAL_IMAGE4                        = 25
    DEBUG_INPUT1                           = 26
    DEBUG_INPUT2                           = 27
    DEBUG_CENTROIDINGEVENTMOMENTSX         = 28
    DEBUG_CENTROIDINGEVENTMOMENTSY         = 29
    DEBUG_CENTROIDINGEVENTMOMENTSINTENSITY = 30
    DEBUG_CENTROIDINGEVENTLABELS           = 31
    DEBUG_CENTROIDINGEVENTOTHERDATA        = 32
    DEBUG_DEEPMINX                         = 33
    DEBUG_DEEPMINY                         = 34
    DEBUG_DEEPMAXX                         = 35
    DEBUG_DEEPMAXY                         = 36
    REFERENCE_DARKLEVEL0                   = 37
    REFERENCE_DARKLEVEL1                   = 38
    REFERENCE_DARKLEVEL2                   = 39
    REFERENCE_DARKOFFCHIPCDS               = 40
    REFERENCE_GAININTEGRATINGLEVEL0        = 41
    REFERENCE_GAININTEGRATINGLEVEL1        = 42
    REFERENCE_GAININTEGRATINGLEVEL2        = 43
    REFERENCE_GAINCOUNTING                 = 44
    REFERENCE_GAINFINALINTEGRATINGLEVEL0   = 45
    REFERENCE_GAINFINALCOUNTING            = 46
    REFERENCE_BADPIXELS                    = 47
    REFERENCE_BADPIXELMAP                  = 48
    SUMTOTAL_MOTIONCORRECTED               = 49
    SCAN_SUBSAMPLINGMASK                   = 50
    NUMBER_OF_OPTIONS                      = 51
    # deprecated frame type 
    CRUDE_FRAME                            =  2    # for engineering testing        #Same as CRUDEFRAME  
    SINGLE_FRAME_RAW_LEVEL0                =  3    # for engineering testing        #Same as SINGLEFRAME_RAWLEVEL0
    SINGLE_FRAME_RAW_LEVEL1                =  4    # for engineering testing        #Same as SINGLEFRAME_RAWLEVEL1
    SINGLE_FRAME_RAW_LEVEL2                =  5    # for engineering testing        #Same as SINGLEFRAME_RAWLEVEL2
    SINGLE_FRAME_RAW_CDS                   =  6    # for engineering testing        #Same as SINGLEFRAME_RAWOFFCHIPCDS
    SINGLE_FRAME_INTEGRATED                =  7    # single integrated frame        #Same as SINGLEFRAME_INTEGRATED
    SINGLE_FRAME_COUNTED                   =  8    # single counted frame           #same as SINGLEFRAME_COUNTED
    MOVIE_INTEGRATED                       =  9    # fractionated integrated frame  #same as SUMINTERMEDIATE
    MOVIE_COUNTED                          =  9    # fractionated counted frame     #same as SUMINTERMEDIATE
    TOTAL_SUM_INTEGRATED                   =  10   # sum of all integrated frame    #same as SUMTOTAL
    TOTAL_SUM_COUNTED                      =  10   # sum of all counted frame       #same as SUMTOTAL
    CUMULATIVE_INTEGRATED                  =  11   # cumulative integrted frame     #same as CUMULATIVE
    CUMULATIVE_COUNTED                     =  11   # cumulative counted frame       #same as CUMULATIVE

class PixelFormat(Enum):
    UINT8                   = 1     # 8-bit integer
    UINT16                  = 5     # 16-bit integer
    FLOAT32                 = 13    # 32-bit float
    AUTO                    = -1    # Automatically determined by the server


class DataType(Enum):
    DEUndef = -1
    DE8u    = 1
    DE16u   = 5
    DE16s   = 7
    DE32f   = 13

class MovieBufferStatus(Enum):
    UNKNOWN  = 0
    FAILED   = 1   
    TIMEOUT  = 3
    FINISHED = 4
    OK       = 5

class ContrastStretchType(IntEnum):
    NONE = 0
    MANUAL = 1
    LINEAR = 2
    DIFFRACTION = 3
    THONRINGS = 4
    NATURAL = 5
    HIGHCONTRAST = 6
    WIDERANGE = 7

class Attributes:
    centerX                 = 0     # In:  x coordinate of the center of the original image to pan to                  
    centerY                 = 0     # In:  y coordinate of the center of the original image to pan to         
    zoom                    = 1.0   # In:  zoom level on the returned image           
    windowWidth             = 0     # In:  Width of the returned image in pixels         
    windowHeight            = 0     # In:  Height of the returned image in pixels         
    fft                     = False # In:  request to return the FFT of the image    
    linearStretch           = False
    stretchType             = ContrastStretchType.LINEAR     # In: Contrast Stretch type. It is Linear by default TODO: enum
    manualStretchMin        = 0.0   # In: Manual Stretch Minimum is adjusted by the user when the contrast stretch type is Manual.
    manualStretchMax        = 0.0   # In: Manual Stretch Maximum is adjusted by the user when the contrast stretch type is Manual.
    manualStretchGamma      = 1.0   # In: Manual Stretch Gamma is adjusted by the user when the contrast stretch type is Manual.
    outlierPercentage       = 2.0   # In:  Percentage of outlier pixels at each end of the image histogram to ignore during contrast stretching if linearStretch is true.           
    buffered                = False # In:  Deprecated attribute, please use GetMovieBuffer to get all movie frames
    timeoutMsec             = -1    # In:  Timeout in milliseconds for GetResult call. 
    frameWidth              = 0     # Out: Width of the processed image in pixels before panning and zooming. This matches the size of the saved images.         
    frameHeight             = 0     # Out: Height of the processed image in pixels before panning and zooming. This matches the size of the saved images.         
    datasetName             = None  # Out: Data set name that the retrieved frame belongs to. Each repeat of acquisition has its own unique dataset name. e.g. 20210902_00081            
    acqIndex                = 0     # Out: Zero-based acquisition index. Incremented after each repeat of acquisition, up to one less of the numberOfAcquisitions parameter of the StartAcquisition call         
    acqFinished             = False # Out: Indicates whether the current acquisition is finished.             
    imageIndex              = 0     # Out: Zero-based index within the acquisition that the retrieved frame belongs to         
    frameCount              = 0     # Out: Number of frames summed up to make the returned image         
    imageMin                = 0     # Out: Minimum pixel value of the retrieved image.         
    imageMax                = 0     # Out: Maximum pixel value of the retrieved image.         
    imageMean               = 0     # Out: Mean pixel values of the retrieved image.         
    imageStd                = 0     # Out: Standard deviation of pixel values of the retrieved image.       
    eppix                   = 0     # Out: The mean total electrons per pixel calculated for the acquisition.         
    eps                     = 0     # Out: The mean total electrons per second over the entire readout area for the acquisition.         
    eppixps                 = 0     # Out: The mean electrons per pixel per second calculated for the acquisition.         
    epa2                    = 0     # Out: The mean total electrons per square Angstrom calculated for the acquisition.         
    eppixpf                 = 0     # Out: The mean electrons per pixel per frame calculated for the acquisition.
    eppix_incident          = 0     # Out: The incident mean total electrons per pixel calculated for the acquisition.         
    eps_incident            = 0     # Out: The incident mean total electrons per second over the entire readout area for the acquisition.         
    eppixps_incident        = 0     # Out: The incident mean electrons per pixel per second calculated for the acquisition.         
    epa2_incident           = 0     # Out: The incident mean total electrons per square Angstrom calculated for the acquisition.         
    eppixpf_incident        = 0     # Out: The incident mean electrons per pixel per frame calculated for the acquisition.                  
    underExposureRate       = 0     # Out: Percentage of pixels under the Statistics - Underexposure Mark (percentage) property value.         
    overExposureRate        = 0     # Out: Percentage of pixels over the Statistics - Overexposure Mark (percentage) property value.
    timestamp               = 0     # Out: Timestamp in seconds since the Unix epoch time when the frame arrived at the computer.        
    autoStretchMin          = 0.0   # Out: Auto Stretch Minimum is returned by getAsyncOutput in server.
    autoStretchMax          = 0.0   # Out: Auto Stretch Maximum is returned by getAsyncOutput in server.
    autoStretchGamma        = 1.0   # Out: Auto Stretch Gamma is returned by getAsyncOutput in server.
    saturation              = 0.0   


class Histogram:
    min                     = 0.0   # In/Out: minimum histogram value, if min and max is set to the same value, or in trial of gain acquisition mode, the server will determine this value.
    max                     = 0.0   # In/Out: maximum histogram value, if min and max is set to the same value, or in trial of gain acquisition mode, the server will determine this value.
    upperMostLocalMaxima    = 0     # In/Out: upper local max histogram value
    bins                    = 256   # In:     number of bins requested, 0 means histogram data is not requested
    data                    = None  # Out:    buffer containing histogram data


class MovieBufferInfo:
    headerBytes = 0
    imageBufferBytes = 0
    frameIndexStartPos = 0
    imageStartPos = 0
    imageW = 0
    imageH = 0
    framesInBuffer = 0
    imageDataType = DataType.DEUndef    

class PropertySpec:
        dataType            = None # "String"   | "Integer"  | "Float"
        valueType           = None # "ReadOnly" | "Set"      | "Range"       | "AllowAll" 
        category            = None # "Basic"    | "Advanced" | "Engineering" | "Deprecated" | Obselete"
        options             = None # List of options
        defaultValue        = None # default value
        currentValue        = None # current value


class Client:
    # Connect to DE-Server
    def Connect(self, host="127.0.0.1", port=13240):
        global commandVersion
        if host == "localhost" or host == "127.0.0.1":
            tcpNoDelay = 0 # on loopback interface, nodelay causes delay

            if self.usingMmf:
                self.mmf = mmap.mmap(0, MMF_DATA_BUFFER_SIZE, "ImageFileMappingObject")
                self.mmf[0] = True
        else :
            self.usingMmf   = False # Disabled MMF if connected remotely
            tcpNoDelay = 1

        if logLevel == logging.DEBUG:
            log.debug("Connecting to server: %s", host)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Create a socket (SOCK_STREAM means a TCP socket)
        self.socket.connect((host, port)) # Connect to server reading port for sending data
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, tcpNoDelay)
        self.socket.setblocking(0)
        self.socket.settimeout(2) 

        self.cameras = self.__getStrings(self.LIST_CAMERAS)
        if logLevel == logging.DEBUG:  
            log.debug("Available cameras: %s", self.cameras)

        self.currCamera = self.cameras[0]
        if logLevel == logging.DEBUG:  
            log.debug("Current camera: %s", self.currCamera)

        self.connected = True
        self.host = host
        self.port = port
        log.info("Connected to server: %s, port: %d", host, port)


        serverVersion = self.GetProperty("Server Software Version");
        serverVersion = re.findall(r'\d+', serverVersion)

        version = [int(part) for part in serverVersion[:4]]
        temp = version[2] + version[1] * 1000 + version[0] * 1000000
        if temp >= 2007004:
            ## version after 2.7.4
            commandVersion = 12;
        elif temp >= 2007003:
            ## version after 2.7.3
            commandVersion = 11;
        elif temp >= 2007002:
            ## version after 2.7.2
            commandVersion = 10
        elif temp >= 2005025:
            ##version after 2.5.25
            commandVersion = 4;
        elif temp >= 2001017:
            ## version after 2.1.17
            commandVersion = 3;
            
    # Disonnect from DE-Server
    def Disconnect(self):
        if self.mmf != 0:
            self.mmf.close()

        if self.connected:
            self.socket.close()
            self.socket.close()
            self.connected = False
            log.info("Disconnected.")

    # Get the current server version 
    def GetServerVersion(self):
        serverVersion = self.GetProperty("Server Software Version");
        serverVersion = re.findall(r'\d+', serverVersion)

        version = [int(part) for part in serverVersion[:4]]
        res = version[2] + version[1] * 1000 + version[0] * 1000000
        return res 




    # List available cameras from DE-Server
    def ListCameras(self):
        return self.cameras


    # Get the current camera
    def GetCurrentCamera(self):
        if self.currCamera is None:
            return "No current camera"
        else:
            return self.currCamera


    # Set current camera on DE-Server
    def SetCurrentCamera(self, camera_name = None):
        if camera_name is None:            
            return False

        self.currCamera = camera_name

        if logLevel == logging.DEBUG:  
            log.debug("current camera: %s", camera_name)

        self.refreshProperties = True
        return True


    # Get a list of property names from the current camera on DE-Server
    def ListProperties(self, options=None):
        available_properties = self.__getStrings(self.LIST_PROPERTIES, options)
        if available_properties != False:
            self.available_properties = available_properties   

        if logLevel == logging.DEBUG:  
            log.debug("Available camera properties: %s", available_properties)

        return self.available_properties


    # Get a list of allowed values for a property of the current camera on DE-Server
    def GetPropertySpec(self, propertyName):
        t0 = self.GetTime()   
        values   = False
        command  = self.__addSingleCommand(self.LIST_ALLOWED_VALUES, propertyName)
        response = self.__sendCommand(command) 
        if response == False:
            return None

        values = self.__getParameters(response.acknowledge[0])

        propSpec = PropertySpec()
        propSpec.dataType        = values[0]
        propSpec.valueType       = values[1]
        propSpec.category        = values[len(values)-3]
        propSpec.options         = list(values[2:len(values)-3])
        propSpec.defaultValue    = str(values[len(values)-2])
        propSpec.currentValue    = str(values[len(values)-1])

        optionsLength   = len(propSpec.options)

        if propSpec.valueType == "Range":
            if optionsLength == 2:
                rangeString = ""
                for i in range(optionsLength):
                    if propSpec.dataType == "Integer":
                        rangeString += str(int(propSpec.options[i])) 
                    else:
                        rangeString += str(propSpec.options[i])
                    if i == 0:
                        rangeString += str(" - ")

                propSpec.options.append(rangeString)

        if propSpec.valueType == "Set":
            for i in range(optionsLength):
                if propSpec.defaultValue == propSpec.options[i]:
                    if propSpec.defaultValue != "":
                        propSpec.options[i] = propSpec.defaultValue + str("*")
                    else:
                        emptyStringIndex = i
            if propSpec.defaultValue == "":
                propSpec.options.pop(emptyStringIndex)

        if "allow_all" in propSpec.valueType:
            propSpec.options = ""
        elif propSpec.dataType == "String":
            propSpec.options = str(list(map(lambda a: str(a), propSpec.options)))[1:-1]
        else:
            propSpec.options = str(propSpec.options)[1:-1]

        return propSpec


    # Get a list of allowed values for a property of the current camera on DE-Server
    def PropertyValidValues(self, propertyName):
        t0 = self.GetTime()   
        values   = False
        command  = self.__addSingleCommand(self.LIST_ALLOWED_VALUES, propertyName)
        response = self.__sendCommand(command) 
        if response != False:
            values = self.__getParameters(response.acknowledge[0])

            if logLevel == logging.DEBUG:
                log.debug("get allowed property values: %s = %s, completed in %.1f ms", propertyName, values, (self.GetTime() - t0) * 1000)
        
        return values


    # Get the value of a property of the current camera on DE-Server
    def GetProperty(self, propertyName):
        t0 = self.GetTime()   
        ret = False

        if propertyName is not None:            
            command = self.__addSingleCommand(self.GET_PROPERTY, propertyName)
            response = self.__sendCommand(command)        
            if response != False:
                values = self.__getParameters(response.acknowledge[0])
                if type(values) is list:
                    if (len(values) > 0):
                        ret = values[0] #always return the first value
                    else:    
                        ret = values

                if logLevel == logging.DEBUG:  
                    log.debug("GetProperty: %s = %s, completed in %.1f ms", propertyName, values, (self.GetTime() - t0) * 1000)

        return ret


    # Set the value of a property of the current camera on DE-Server
    def SetProperty(self, name, value):
        t0 = self.GetTime()   
        ret = False

        if name is not None and value is not None :            
            command = self.__addSingleCommand(self.SET_PROPERTY, name, [value])
            response = self.__sendCommand(command)                
            if response != False:
                ret = response.acknowledge[0].error != True
                self.refreshProperties = True

        if logLevel == logging.DEBUG:  
            log.debug("SetProperty: %s = %s, completed in %.1f ms", name, value, (self.GetTime() - t0) * 1000)
        
        return ret
 
    def SetPropertyAndGetChangedProperties(self, name, value, changedProperties):
        t0 = self.GetTime()   
        ret = False

        if name is not None and value is not None :            
            command = self.__addSingleCommand(self.SET_PROPERTY_AND_GET_CHANGED_PROPERTIES, name, [value])
            response = self.__sendCommand(command)                
            if response != False:
                ret = response.acknowledge[0].error != True
                self.refreshProperties = True

            if ret:
                ret = self.ParseChangedProperties(changedProperties, response)
        if logLevel == logging.DEBUG:  
            log.debug("SetProperty: %s = %s, completed in %.1f ms", name, value, (self.GetTime() - t0) * 1000)
        
        return ret

    
    def setEngMode(self, enable, password):
        
        ret = False
        
        command = self.__addSingleCommand(self.SET_ENG_MODE, None, [enable,password])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True
        return ret 


    def setEngModeAndGetChangedProperties(self, enable, password, changedProperties):
        
        ret = False
        
        command = self.__addSingleCommand(self.SET_ENG_MODE_GET_CHANGED_PROPERTIES, None, [enable,password])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if ret:
            ret = self.ParseChangedProperties(changedProperties, response)
            
        return ret


    def SetHWROI(self,offsetX,offsetY, sizeX, sizeY):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_HW_ROI, None, [offsetX, offsetY, sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True


        if logLevel == logging.DEBUG:  
            log.debug("SetHwRoi: (%i,%i,%i,%i) , completed in %.1f ms", offsetX, offsetY, sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret 


    def SetHWROIAndGetChangedProperties(self, offsetX, offsetY, sizeX, sizeY, changedProperties, timeoutMsec = 5000):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_HW_ROI_AND_GET_CHANGED_PROPERTIES, None, [offsetX, offsetY, sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if ret:
            ret = self.ParseChangedProperties(changedProperties, response)

        if logLevel == logging.DEBUG:  
            log.debug("SetHwRoi: (%i,%i,%i,%i) , completed in %.1f ms", offsetX, offsetY, sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret        


    def SetSWROI(self,offsetX,offsetY, sizeX, sizeY):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_SW_ROI, None, [offsetX, offsetY, sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if logLevel == logging.DEBUG:  
            log.debug("SetSwRoi: (%i,%i,%i,%i) , completed in %.1f ms", offsetX, offsetY, sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret      


    def SetSWROIAndGetChangedProperties(self, offsetX, offsetY, sizeX, sizeY, changedProperties, timeoutMsec = 5000):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_SW_ROI_AND_GET_CHANGED_PROPERTIES, None, [offsetX, offsetY, sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if ret:
            ret = self.ParseChangedProperties(changedProperties, response)

        if logLevel == logging.DEBUG:  
            log.debug("SetSWRoi: (%i,%i,%i,%i) , completed in %.1f ms", offsetX, offsetY, sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret  

    def SetScanSize(self,sizeX, sizeY):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_SCAN_SIZE, None, [sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if logLevel == logging.DEBUG:  
            log.debug("SetScanSize: (%i,%i) , completed in %.1f ms", sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret  

    def SetScanSizeAndGetChangedProperties(self,sizeX, sizeY,changedProperties):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_SCAN_SIZE_AND_GET_CHANGED_PROPERTIES, None, [sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if ret:
            ret = self.ParseChangedProperties(changedProperties, response)

        if logLevel == logging.DEBUG:  
            log.debug("SetScanSize: (%i,%i) , completed in %.1f ms", sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret  


    def SetScanROI(self,enable, offsetX, offsetY, sizeX, sizeY):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_SCAN_SIZE, None, [enable, offsetX, offsetY, sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if logLevel == logging.DEBUG:  
            log.debug("SetScanROI: (%i,%i,%i,%i) , completed in %.1f ms", offsetX, offsetY, sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret  

    def SetScanROI(self,enable, offsetX, offsetY, sizeX, sizeY, changedProperties):
        t0 = self.GetTime()   
        ret = False
            
        command = self.__addSingleCommand(self.SET_SCAN_ROI__AND_GET_CHANGED_PROPERTIES, None, [enable, offsetX, offsetY, sizeX, sizeY])
        response = self.__sendCommand(command)                
        if response != False:
            ret = response.acknowledge[0].error != True
            self.refreshProperties = True

        if ret:
            ret = self.ParseChangedProperties(changedProperties, response)

        if logLevel == logging.DEBUG:  
            log.debug("SetScanROI: (%i,%i,%i,%i) , completed in %.1f ms", offsetX, offsetY, sizeX, sizeY, (self.GetTime() - t0) * 1000)

        return ret 

    # Start acquiring images with the all the propeties set to desired values
    # numberOfAcquisitions: specify number of acquistions to repeat
    def StartAcquisition(self, numberOfAcquisitions=1, requestMovieBuffer = False):
        start_time = self.GetTime()
        step_time  = self.GetTime()

        if self.refreshProperties:
            self.roi_x          = self.GetProperty("Crop Size X")
            self.roi_y          = self.GetProperty("Crop Size Y")
            self.binning_x      = self.GetProperty("Binning X")
            self.binning_y      = self.GetProperty("Binning Y")
            self.width          = self.GetProperty("Image Size X (pixels)")     
            self.height         = self.GetProperty("Image Size Y (pixels)")    
            self.exposureTime   = self.GetProperty("Exposure Time (seconds)")
            self.refreshProperties = False

        if logLevel == logging.DEBUG:
            lapsed = (self.GetTime() - step_time) * 1000
            log.debug(" Prepare Time: %.1f ms", lapsed)
            step_time = self.GetTime()

        if (self.width * self.height == 0):            
            log.error("  Image size is 0! " )
        else:            
            bytesize = 0
            command = self.__addSingleCommand(self.START_ACQUISITION, None, [numberOfAcquisitions,requestMovieBuffer])

            if logLevel == logging.DEBUG:
                lapsed = (self.GetTime() - step_time) * 1000
                log.debug("   Build Time: %.1f ms", lapsed)
                step_time = self.GetTime()

            response = self.__sendCommand(command)        
            if logLevel == logging.DEBUG:
                lapsed = (self.GetTime() - step_time) * 1000
                log.debug(" Command Time: %.1f ms", lapsed)
                step_time = self.GetTime()

            if response != False:
                ret = response.acknowledge[0].error != True
                self.refreshProperties = True

        if logLevel == logging.DEBUG:
            lapsed = (self.GetTime() - step_time) * 1000
            log.debug("  Typing Time: %.1f ms", lapsed)
            step_time = self.GetTime()

        if logLevel <= logging.DEBUG:
            lapsed = (self.GetTime() - start_time) * 1000
            log.debug("  Start Time: %.1f ms, ROI:[%d, %d], Binning:[%d, %d], Image size:[%d, %d]", lapsed, self.roi_x, self.roi_y, self.binning_x, self.binning_y, self.width, self.height)


    # Can be called in the same thread or another thread to stop the current acquisitions
    # This will cause GetReult calls to return immediately.
    def StopAcquisition(self):
        start_time = self.GetTime()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        sock.sendto(b"PyClientStopAcq", (self.host, self.port))
        respond = sock.recv(32);
        if logLevel == logging.INFO:
            log.info(f"{self.host} {self.port} {respond}")
        if logLevel <= logging.DEBUG:
            lapsed = (self.GetTime() - start_time) * 1000
            log.debug("    Stop Time: %.1f ms", lapsed)

        return b"Stopped" in respond


    # Get the specified type of frames in the desired pixel format and associated information
    # Parameters: 
    #   frameType:   Input:
    #   pixelFormat: Input/output
    #   attributes:  Defines the image to be returned, some members can be updated.
    #                Some members of this parameter are input only, some are input/output.
    #   histogram:   Returns the hisogram if desired.
    #                Some members of this parameter are input only, some are input/output.
    # return:
    #   image object
    # remark: 
    #   During acquisition, live frames will be returned; after acquisition, the last image will be returned
    def GetResult(self, frameType, pixelFormat, attributes=None, histogram=None):
        log.debug("GetResult frameType:%s, pixelFormat:%s", frameType, pixelFormat)
        start_time = self.GetTime()
        step_time  = self.GetTime()

        if attributes == None:
            attributes = Attributes()

        if histogram == None:
            histogram = Histogram()

        if attributes.windowWidth > 0:
            self.width = attributes.windowWidth

        if attributes.windowHeight > 0:
            self.height = attributes.windowHeight

        image = None
        imageDataType = numpy.uint16

        if logLevel == logging.DEBUG:
            lapsed = (self.GetTime() - step_time) * 1000
            log.debug(" Prepare Time: %.1f ms", lapsed)
            step_time = self.GetTime()

        if histogram != None:
            histoMin  = histogram.min
            histoMax  = histogram.max
            histoBins = histogram.bins
        else:
            histoMin = 0
            histoMax = 0
            histoBins = 0

        bytesize = 0
        command = self.__addSingleCommand(self.GET_RESULT, 
                                          None, 
                                          [frameType.value, 
                                           pixelFormat.value, 
                                           attributes.centerX, 
                                           attributes.centerY, 
                                           attributes.zoom,
                                           attributes.windowWidth,
                                           attributes.windowHeight,
                                           attributes.fft, 
                                           attributes.stretchType,
                                           attributes.manualStretchMin,
                                           attributes.manualStretchMax,
                                           attributes.manualStretchGamma,
                                           attributes.outlierPercentage,
                                           attributes.timeoutMsec, 
                                           histoMin, 
                                           histoMax, 
                                           histoBins])

        if logLevel == logging.DEBUG:
            lapsed = (self.GetTime() - step_time) * 1000
            log.debug("   Build Time: %.1f ms", lapsed)
            step_time = self.GetTime()

        response = self.__sendCommand(command)        
        if logLevel == logging.DEBUG:
            lapsed = (self.GetTime() - step_time) * 1000
            log.debug(" Command Time: %.1f ms", lapsed)
            step_time = self.GetTime()

        if response != False :
            values = self.__getParameters(response.acknowledge[0])
            if type(values) is list and len(values) >= 20:
                i = 0 
                pixelFormat                     = PixelFormat(values[i])
                i += 1
                attributes.frameWidth           = values[i]
                i += 1 
                attributes.frameHeight          = values[i]
                i += 1 
                attributes.datasetName          = values[i]
                i += 1 
                attributes.acqIndex             = values[i]
                i += 1
                attributes.acqFinished          = values[i]
                i += 1
                attributes.imageIndex           = values[i]
                i += 1
                attributes.frameCount           = values[i]
                i += 1
                attributes.imageMin             = values[i]
                i += 1
                attributes.imageMax             = values[i]
                i += 1
                attributes.imageMean            = values[i]
                i += 1
                attributes.imageStd             = values[i]
                i += 1
                attributes.eppix                = values[i]
                i += 1
                attributes.eps                  = values[i]
                i += 1
                attributes.eppixps              = values[i]
                i += 1
                attributes.epa2                 = values[i]
                i += 1
                attributes.eppixpf              = values[i]
                i += 1
                if commandVersion >= 12:
                    attributes.eppix_incident         = values[i]
                    i += 1
                    attributes.eps_incident           = values[i]
                    i += 1
                    attributes.eppixps_incident       = values[i]
                    i += 1
                    attributes.epa2_incident          = values[i]
                    i += 1
                    attributes.eppixpf_incident       = values[i]
                    i += 1
                if commandVersion >= 11:
                    attributes.saturation           = values[i]
                    i += 1
                if commandVersion < 10:
                    attributes.underExposureRate = values[i]
                    i += 1
                    attributes.overExposureRate = values[i]
                    i +=1 
                attributes.timestamp            = float(values[i])
                i += 1
                if commandVersion >= 10:
                    attributes.autoStretchMin       = values[i]
                    i += 1
                    attributes.autoStretchMax       = values[i]
                    i += 1
                    attributes.autoStretchGamma     = values[i]
                    i += 1
                

                if (histogram != None and histoBins > 0 and len(values) >= i + histogram.bins):
                    histogram.data = [0] * histogram.bins
                    histogram.min = values[i]
                    i += 1
                    histogram.max = values[i]
                    i += 1
                    if commandVersion >= 11:
                        histogram.upperMostLocalMaxima = values[i]
                        i += 1 
                    for j in range(histogram.bins):
                        log.debug("%d: %d" % (j, values[i + j]))
                        histogram.data[j] = values[i + j]

                if (pixelFormat == PixelFormat.FLOAT32):
                    imageDataType   = numpy.float32
                elif (pixelFormat == PixelFormat.UINT16):
                    imageDataType   = numpy.uint16
                else:
                    imageDataType   = numpy.uint8

                self.width = attributes.frameWidth;
                self.height = attributes.frameHeight;

            recvbyteSizeString = self.__recvFromSocket(self.socket, 4) # get the first 4 bytes        
            if (len(recvbyteSizeString) == 4):
                recvbyteSize = struct.unpack("I",recvbyteSizeString) # interpret as size
                received_string = self.__recvFromSocket(self.socket, recvbyteSize[0]) # get the rest
                data_header = pb.DEPacket()
                data_header.ParseFromString(received_string)
                bytesize = data_header.data_header.bytesize        

            if self.usingMmf:
                image = numpy.frombuffer(self.mmf, offset=MMF_DATA_HEADER_SIZE, dtype=imageDataType, count=self.width*self.height)
                image.shape = [self.height, self.width] 
                bytesize = self.width * self.height * 2
            elif (bytesize > 0):
                packet = self.__recvFromSocket(self.socket, bytesize)
                if (len(packet) == bytesize):
                    image = numpy.frombuffer(packet, imageDataType)
                    bytesize = self.height * self.width * 2
                    image.shape = [self.height, self.width] 

            if logLevel == logging.DEBUG:
                elapsed = self.GetTime() - step_time
                log.debug("Transfer time: %.1f ms, %d bytes, %d mbps",  elapsed*1000, bytesize, bytesize*8/elapsed/1024/1024)
                step_time = self.GetTime()

            if bytesize <= 0:
                log.error("  GetResut failed! An empty image will be returned.")
                image = None

            if logLevel == logging.DEBUG:
                lapsed = (self.GetTime() - step_time) * 1000
                log.debug("  Saving Time: %.1f ms", lapsed)
                step_time = self.GetTime()

        if image is None:
            log.error("  GetResut failed!")
        else:
            image  = image.astype(imageDataType)            

            if logLevel == logging.DEBUG:
                lapsed = (self.GetTime() - step_time) * 1000
                log.debug("  Typing Time: %.1f ms", lapsed)
                step_time = self.GetTime()

        if logLevel <= logging.DEBUG:
            lapsed = (self.GetTime() - start_time) * 1000
            log.debug("GetResult frameType:%s, pixelFormat:%s ROI:[%d, %d] Binning:[%d, %d], Return size:[%d, %d], datasetName:%s acqCount:%d, frameCount:%d min:%.1f max:%.1f mean:%.1f std:%.1f %.1f ms", 
                       frameType,   pixelFormat, self.roi_x, self.roi_y, self.binning_x, self.binning_y, self.width, self.height, 
                       attributes.datasetName, attributes.acqIndex, attributes.frameCount,
                       attributes.imageMin,    attributes.imageMax, attributes.imageMean, attributes.imageStd, 
                       lapsed)

        return image, pixelFormat, attributes, histogram
    
    def SetVirtualMask(self,id, w, h, mask):
        ret = True

        if id < 1 or id > 4:
            log.error(" SetVirtualMask The virtual mask id must be selected between 1-4")
            ret = False
        elif w < 0 or h < 0:
            log.error(" SetVirtualMask The virtual mask width and height must greater than 0")
            ret = False 
        else: 

            command = self.__addSingleCommand(self.SET_VIRTUAL_MASK, None, [id-1,w,h])
            try:
                packet = struct.pack("I", command.ByteSize()) + command.SerializeToString()
                self.socket.send(packet)
            except socket.error as e:
                ret = False 

            if ret: 
                self.__sendToSocket(self.socket,mask,w*h)
                ret = self.__ReceiveResponseForCommand(command) != False

        return ret 


    def GetMovieBufferInfo(self, movieBufferInfo=None, timeoutMsec = 5000):

        if movieBufferInfo == None:
            movieBufferInfo = MovieBufferInfo()
        command = self.__addSingleCommand(self.GET_MOVIE_BUFFER_INFO, None, None)

        response = self.__sendCommand(command) 
        
        if response != False :
            values = self.__getParameters(response.acknowledge[0])
            if type(values) is list:
                movieBufferInfo.headerBytes = values[0]
                movieBufferInfo.imageBufferBytes = values[1]
                movieBufferInfo.frameIndexStartPos = values[2]
                movieBufferInfo.imageStartPos = values[3]
                movieBufferInfo.imageW = values[4]
                movieBufferInfo.imageH = values[5]
                movieBufferInfo.framesInBuffer = values[6]
                dataType = values[7]
                movieBufferInfo.imageDataType = DataType(dataType)  

        return movieBufferInfo

    def GetMovieBuffer(self, movieBuffer, movieBufferSize, numFrames, timeoutMsec=5000):
        
        movieBufferStatus = MovieBufferStatus.UNKNOWN
        retval = True 

        command = self.__addSingleCommand(self.GET_MOVIE_BUFFER, None, [timeoutMsec])

        response = self.__sendCommand(command)

        if response != False:
            totalBytes = 0
            status = 0 
            values = self.__getParameters(response.acknowledge[0])
            if type(values) is list:
                status     = values[0]
                totalBytes = values[1]
                numFrames = values[2]
                movieBufferStatus = MovieBufferStatus(status)

                if movieBufferStatus == MovieBufferStatus.OK:
                    if totalBytes == 0 or movieBufferSize < totalBytes:
                        retval = False 
                        log.error("Image received did not have the expected size.")
                    else:
                        movieBuffer = self.__recvFromSocket(self.socket, totalBytes)
        else:
            retval = False 

        if not retval:  
            movieBufferStatus = MovieBufferStatus.FAILED
               
        return movieBufferStatus, totalBytes, numFrames, movieBuffer 

    def SaveImage(self, image, fileName, textSize=0):
        t0 = self.GetTime()
        filePath = self.debugImagesFolder + fileName + ".tif"
        try:  
            if not os.path.exists(self.debugImagesFolder):
                os.makedirs(self.debugImagesFolder)

            tiff = Image.fromarray(image)
            tiff.save(filePath) 
            log.info("Saved %s" % filePath)

            # if textSize > 0:
            #     self.__saveText(image, fileName, textSize)

        except OSError:  
            log.error ("Failed to save file")


        if logLevel == logging.DEBUG or True:
            log.debug("Save time: %.1f ms", (self.GetTime() - t0) * 1000)

        return filePath

    # Print out the server information
    def PrintServerInfo(self, camera):
        print("Time        : " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))    
        print("Computer    : " + socket.gethostname())
        print("DE-Server   : " + self.GetProperty("Server Software Version"))
        print("CPU         : " + self.GetProperty("Computer CPU Info"))
        print("GPU         : " + self.GetProperty("Computer GPU Info"))
        print("Memory      : " + self.GetProperty("Computer Memory Info"))
        print("Camera Name : " + camera)
        print("Camera S/N  : " + str(self.GetProperty("Camera SN")))
        print("Sensor S/N  : " + str(self.GetProperty("Sensor Module SN")))
        print("Firmware    : " + str(self.GetProperty("Firmware Version")))
        print("Python      : " + sys.version.split("(")[0])
        print("Client      : " + version)
        print("Interrupt   : " + str(self.GetProperty("Interrupt Status")))


    # Print out the acquisition information  
    def PrintAcqInfo(self):
        hwW    = self.GetProperty("Hardware ROI - Size X")
        hwH    = self.GetProperty("Hardware ROI - Size Y")
        hwX    = self.GetProperty("Hardware ROI Offset X")
        hwY    = self.GetProperty("Hardware ROI Offset Y")
        hwBinX = self.GetProperty("Hardware Binning X")
        hwBinY = self.GetProperty("Hardware Binning Y")        

        swW    = self.GetProperty("ROI Size X")
        swH    = self.GetProperty("ROI Size Y")
        swX    = self.GetProperty("ROI Offset X")
        swY    = self.GetProperty("ROI Offset Y")
        swBinX = self.GetProperty("Binning X")
        swBinY = self.GetProperty("Binning Y")        

        print("Test Pattern: " + self.GetProperty("Test Pattern"))
        print("Log Level   : " + self.GetProperty("Log Level"))
        print("Sensor size : " + str(self.GetProperty("Sensor Size X (pixels)")) + " x " + str(self.GetProperty("Sensor Size Y (pixels)")))
        print("HW ROI      : " + str("%d x %d at (%d, %d)" % (hwW, hwH, hwX, hwY)) )
        print("HW Binning  : " + "%d x %d" % (hwBinX, hwBinY))
        print("SW ROI      : " + str("%d x %d at (%d, %d)" % (swW, swH, swX, swY)) )
        print("SW Binning  : " + "%d x %d" % (swBinX, swBinY))
        print("Image size  : " + str(self.GetProperty("Image Size X (pixels)")) + " x " + str(self.GetProperty("Image Size Y (pixels)")) )
        print("Max FPS     : " + str(self.GetProperty("Frames Per Second (Max)")))
        print("FPS         : " + str(self.GetProperty("Frames Per Second")))
        print("Grab Buffers: " + str(self.GetProperty("Grab Buffer Size")))


    def PrintSavingInfo(self):
        sys.stdout.write("Saving      : ")

        if self.GetProperty("Autosave Crude Frames") == "On":
            sys.stdout.write("crude ")
        if self.GetProperty("Autosave Raw Frames") == "On":
            sys.stdout.write("raw ")
        if self.GetProperty("Autosave Single Frames") == "On":
            sys.stdout.write("frames ")
        if self.GetProperty("Autosave Integrating Frames") == "On":
            sys.stdout.write("integrating frames ")
        if self.GetProperty("Autosave Movie") == "On":
            sys.stdout.write("movie(")
            sys.stdout.write(str(self.GetProperty("Autosave Movie - Sum Count")))
            sys.stdout.write(") ")
        if self.GetProperty("Autosave Final Image") == "On":
            sys.stdout.write("final ")

        print("")


    # Grab specified number of frames and print out stats. Mostly used for testing purposes
    # frames: number of frames requested
    # dataSetName: Data set name to be used
    # fileName: save the returned image as a file if provided
    def Grab(self, frames=1, dataSetName="", fileName=None):
        imageW = self.GetProperty("Image Size X (pixels)")
        imageH = self.GetProperty("Image Size Y (pixels)")
        fps    = self.GetProperty("Frames Per Second")

        self.SetProperty("Exposure Time (seconds)", frames / fps) 
        expoSec    = self.GetProperty("Exposure Time (seconds)") 
        maxExpoSec = self.GetProperty("Exposure Time Max (seconds)")
        prevSuffix  = self.GetProperty("Autosave Filename Suffix")

        frames = round(expoSec * fps)
        # frames = expoSec * fps

        if dataSetName != "" and dataSetName != None:
            self.SetProperty("Autosave Filename Suffix", dataSetName)
            dataSetName = dataSetName + " "

        if dataSetName != None:
            sys.stdout.write("%s%dx%d fps:%.3f " % (dataSetName, imageW, imageH, fps))
            sys.stdout.flush()

        t0 = self.GetTime()   
        self.StartAcquisition(1)

        frameType = FrameType.SUMTOTAL

        pixelFormat = PixelFormat.AUTO
        attributes = Attributes()
        histogram  = Histogram()
        image = self.GetResult(frameType, pixelFormat, attributes, histogram)[0]

        duration = (self.GetTime() - t0)
        measuredFps  = self.GetProperty("Measured Frame Rate")
        missedFrames = self.GetProperty("Missed Frames")

        frameCount = self.GetProperty("Number of Frames Processed")
        sys.stdout.write("mfps:%.3f et:%.2fs dur:%.2fs frames:%d/%d %s %s, min:%4.1f max:%4.0f mean:%8.3f std:%8.3f timestamp:%10.6f" % 
                        (measuredFps, expoSec, duration, frameCount, frames, 
                        frameType, pixelFormat, 
                        attributes.imageMin, attributes.imageMax, attributes.imageMean, attributes.imageStd, attributes.timestamp))
        sys.stdout.flush()

        if max == 0:
            sys.stdout.write(", empty")
            
        if missedFrames > 0:
            sys.stdout.write(", missed:%d " % (missedFrames))

        sys.stdout.flush()

        self.WaitForSavingFiles(dataSetName == None)
        self.SetProperty("Autosave Filename Suffix", prevSuffix)

        if fileName and len(fileName) > 0:
            self.SaveImage(image, fileName)

        return image

    
    def WaitForSavingFiles(self, quiet=True):
        t0 = self.GetTime()  
        saveCrude                 = True if self.GetProperty("Autosave Crude Frames")   == "On" else False
        saveRaw                   = True if self.GetProperty("Autosave Raw Frames")     == "On" else False
        saveFrame                 = True if self.GetProperty("Autosave Single Frames")  == "On" else False
        saveMovie                 = True if self.GetProperty("Autosave Movie")          == "On" else False
        counting                  = True if self.GetProperty("Image Processing - Mode") != "Integrating" else False

        saveIntegratingFrames     = saveFrame and (not counting or self.GetProperty("Autosave Integrating Frames") == "On")
        saveCountingFrames        = saveFrame and counting
        saveIntegratingMovie      = saveMovie and not counting 
        saveCountingMovie         = saveMovie and counting

        saveFinal                 = True if self.GetProperty("Autosave Final Image") == "On" else False
        expMode                   = self.GetProperty("Exposure Mode")

        if expMode == "Dark" or expMode == "Gain":
            repeats = self.GetProperty("Remaining Number of Acquisitions")        
            remaining = self.GetProperty("Remaining Number of Acquisitions") 
            if remaining > 0:
                for i in range(repeats * 10):
                    remaining = self.GetProperty("Remaining Number of Acquisitions") 
                    if remaining > 0:
                        sleep(1)
                    else:
                        break
        elif saveCrude or saveRaw  or saveIntegratingFrames or saveCountingFrames or saveIntegratingMovie or saveCountingMovie or saveFinal: 
            for i in range(1000):
                if self.GetProperty("Autosave Status") in ["Starting", "In Progress"]:
                    if not quiet:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    sleep(1)
                else:
                    if not quiet:
                        sys.stdout.write(" \tgrab:%4.0fMB/s" % self.GetProperty("Speed - Grabbing (MB/s)"))
                        sys.stdout.write(" proc:%4.0fMB/s" % self.GetProperty("Speed - Processing (MB/s)"))
                        sys.stdout.write(" save:%4.0fMB/s" % self.GetProperty("Speed - Writing (MB/s)"))

                        if saveCrude:
                            sys.stdout.write(" crude:%d" %  self.GetProperty("Autosave Crude Frames Written") )
                        if saveRaw:
                            sys.stdout.write(" raw:%d" %  self.GetProperty("Autosave Raw Frames Written") )
                        if saveIntegratingFrames:
                            # sys.stdout.write(" frames:%d" %  self.GetProperty("Autosave Single Frames - Frames Written") )
                            sys.stdout.write(" integrating frames:%d" %  self.GetProperty("Autosave Integrated Single Frames Written") )
                        if saveCountingFrames:
                            sys.stdout.write(" counting frames:%d" %  self.GetProperty("Autosave Counted Single Frames Written") )
                        if saveIntegratingMovie:
                            sys.stdout.write(" integrating movie:%d" %  self.GetProperty("Autosave Integrated Movie Frames Written") )
                        if saveCountingMovie:
                            sys.stdout.write(" counting movie:%d" %  self.GetProperty("Autosave Counted Movie Frames Written" ))
                        if saveFinal:
                            sys.stdout.write(" final sum count:%d" %  self.GetProperty("Autosave Final Image - Sum Count"))

                    break

        duration = (self.GetTime() - t0)
        if not quiet:
            print (" %.1fs" % duration)
            sys.stdout.flush()


    # Start acquisition and get a single image
    # fileName: the image will be saved to disk if a file name is provided
    # textSize: a text file with pixel values will be saved if textSize is given, for debugging/test
    def GetImage(self, pixelFormat = PixelFormat.AUTO, fileName=None, textSize=0):
        self.StartAcquisition(1)
        frameType = FrameType.SUMTOTAL

        if pixelFormat == "float32":
            pixelFormat = PixelFormat.FLOAT32

        elif pixelFormat == "uint16":
            pixelFormat = PixelFormat.UINT16 

        attributes = Attributes()
        histogram  = Histogram()
        image = self.GetResult(frameType, pixelFormat, attributes, histogram)[0]

        if fileName and len(fileName) > 0:
            self.SaveImage(image, fileName, textSize)

        return image


    # Acquire dark reference
    def TakeDarkReference(self, frameRate=20):
        sys.stdout.write("Taking dark references: ")
        sys.stdout.flush()

        prevExposureMode = self.GetProperty("Exposure Mode")
        prevExposureTime = self.GetProperty("Exposure Time (seconds)") 

        acquisitions = 10
        self.SetProperty("Exposure Mode", "Dark")
        self.SetProperty("Frames Per Second", frameRate)
        self.SetProperty("Exposure Time (seconds)", 1)
        self.StartAcquisition(acquisitions)

        while(True):
            attributes = Attributes()
            histogram  = Histogram()
            image = self.GetResult(FrameType.SUMTOTAL, PixelFormat.FLOAT32, attributes, histogram)

            sys.stdout.write(str(attributes.acqIndex) + " ")
            sys.stdout.flush()
            remaining = self.GetProperty("Remaining Number of Acquisitions")
     
            if (remaining == 0):
                print("done.")
                break
            
        self.SetProperty("Exposure Mode", prevExposureMode)
        self.SetProperty("Exposure Time (seconds)", prevExposureTime) 


    def GetTime(self):
        if sys.version_info[0] < 3:
            return time.clock()
        else:
            return time.perf_counter()


    #private methods


    def __del__(self):
        if self.connected:
            self.Disconnect()


    #get multiple parameters from a single acknowledge packet
    def __getParameters(self, single_acknowledge = None):
        output = []
        if single_acknowledge is None:            
            return output
        if single_acknowledge.error == True:
            return output
        for one_parameter in single_acknowledge.parameter:
            if one_parameter.type == pb.AnyParameter.P_BOOL:
                output.append(one_parameter.p_bool)
            elif one_parameter.type == pb.AnyParameter.P_STRING:
                output.append(one_parameter.p_string)
            elif one_parameter.type == pb.AnyParameter.P_INT:
                output.append(one_parameter.p_int)
            elif one_parameter.type == pb.AnyParameter.P_FLOAT:
                output.append(one_parameter.p_float)
        return output
                

    #get strings from a single command response
    def __getStrings(self, command_id = None, param = None):
        if command_id is None:            
            return False
        command = self.__addSingleCommand(command_id, param)
        response = self.__sendCommand(command)
        if response != False:            
            return self.__getParameters(response.acknowledge[0])            
        else:
            return False            
            

    #add a new command (with optional label and parameter)
    def __addSingleCommand(self, command_id = None, label = None, params = None):
        if command_id is None:
            return False        
        command = pb.DEPacket()       # create the command packet
        command.type = pb.DEPacket.P_COMMAND 
        singlecommand1 = command.command.add() # add the first single command
        singlecommand1.command_id = command_id  + commandVersion * 100
        
        if not label is None:
            str_param = command.command[0].parameter.add()
            str_param.type = pb.AnyParameter.P_STRING
            str_param.p_string = label
            str_param.name = "label"

        if not params is None:       
            for param in params:
                if isinstance(param, bool):
                    bool_param = command.command[0].parameter.add()
                    bool_param.type = pb.AnyParameter.P_BOOL
                    bool_param.p_bool = bool(param)
                    bool_param.name = "val"
                elif isinstance(param, int) or isinstance(param, long):
                    int_param = command.command[0].parameter.add()
                    int_param.type = pb.AnyParameter.P_INT
                    int_param.p_int = int(param)
                    int_param.name = "val"
                elif isinstance(param, float):
                    float_param = command.command[0].parameter.add()
                    float_param.type = pb.AnyParameter.P_FLOAT
                    float_param.p_float = param
                    float_param.name = "val"
                else:
                    str_param = command.command[0].parameter.add()
                    str_param.type = pb.AnyParameter.P_STRING
                    str_param.p_string = str(param)
                    str_param.name = "val"
        return command

    #send single command and get a response, if error occurred, return False
    def __sendCommand(self, command = None):
        step_time = self.GetTime()

        if command is None:
            return False

        if (len(command.camera_name)==0):
            command.camera_name = self.currCamera # append the current camera name if necessary


        try:
            packet = struct.pack("I", command.ByteSize()) + command.SerializeToString()
            res = self.socket.send(packet)
            #packet.PrintDebugString()
            #log.debug("sent result = %d\n", res)    
        except:
            log.error("Error sending %s\n", command)    

        if logLevel == logging.DEBUG:
            lapsed = (self.GetTime() - step_time) * 1000
            log.debug(" Send Time: %.1f ms", lapsed)
            step_time = self.GetTime()

        return self.__ReceiveResponseForCommand(command)

    def __ReceiveResponseForCommand(self, command):
        step_time = self.GetTime()

        recvbyteSizeString = self.__recvFromSocket(self.socket, 4) # get the first 4 byte

        if (len(recvbyteSizeString) == 4):
            recvbyteSize = struct.unpack("I", recvbyteSizeString) # interpret as size
            log.debug("-- recvbyteSize: " + str(recvbyteSize))
            received_string = self.__recvFromSocket(self.socket, recvbyteSize[0]) # get the rest
            if logLevel == logging.DEBUG:
                lapsed = (self.GetTime() - step_time) * 1000
                log.debug(" Recv Time: %.1f ms, %d bytes", lapsed, recvbyteSize[0])
                step_time = self.GetTime()

            Acknowledge_return = pb.DEPacket()
            Acknowledge_return.ParseFromString(received_string)
            if logLevel == logging.DEBUG:
                lapsed = (self.GetTime() - step_time) * 1000
                log.debug("Parse Time: %.1f ms", lapsed)
                step_time = self.GetTime()

            if (Acknowledge_return.type == pb.DEPacket.P_ACKNOWLEDGE): #has to be an acknowledge packet
                if (len(command.command) <= len(Acknowledge_return.acknowledge)): 
                    error = False
                    for one_ack in Acknowledge_return.acknowledge:
                        error = error or one_ack.error
                    if (error):
                        message = Acknowledge_return.acknowledge[0].error_message
                        if logLevel == logging.DEBUG:
                            log.error("Server returned error for request :\n" + str(command) +"\n" + "Response :\n" + str(message))
                        elif not message.startswith("Unknown property"):
                            log.error(message)
                    else:                       
                        if logLevel == logging.DEBUG:
                            lapsed = (self.GetTime() - step_time) * 1000
                            log.debug("  Ack Time: %.1f ms", lapsed)
                            step_time = self.GetTime()
                        return Acknowledge_return
                else:
                    log.error("len(command.command):%d != len(Acknowledge_return.acknowledge):%d", len(command.command), len(Acknowledge_return.acknowledge))
            else:
                log.error("Response from server is not ACK")
        else:
            log.error("Server response is %d bytes, shorter than mimumum of 4 bytes", len(recvbyteSizeString))

        return False 




    def __recvFromSocket(self, sock, bytes):
        timeout = self.exposureTime * 10 + 30
        startTime = self.GetTime()
        self.socket.settimeout(timeout) 

        buffer = ""
        try:
            buffer = sock.recv(bytes)
        except:
            pass #continue if more needed

        log.debug(" __recvFromSocket : %d of %d in %.1f ms", len(buffer), bytes, (self.GetTime() - startTime)*1000)
        total_len = len(buffer)
        
        if total_len < bytes:
            while total_len < bytes:
                loopTime = self.GetTime()
                try:
                    buffer += sock.recv(bytes)
                    log.debug(" __recvFromSocket : %d bytes of %d in %.1f ms", len(buffer), bytes, (self.GetTime() - loopTime)*1000)

                except socket.timeout:
                    log.debug(" __recvFromSocket : timeout in trying to receive %d bytes in %.1f ms", bytes, (self.GetTime() - loopTime)*1000)
                    if self.GetTime() - startTime > timeout:
                        log.error(" __recvFromSocket: max timeout %d seconds", timeout)
                        break
                    else:
                        pass #continue further
                except: 
                    log.error("Unknown exception occurred. Current Length: %d in %.1f ms", len(buffer), (self.GetTime() - loopTime)*1000)
                    break
                total_len = len(buffer)

        totalTimeMs = (self.GetTime() - startTime) * 1000
        Gbps = total_len * 8 / (totalTimeMs / 1000) / 1024 / 1024 / 1024  
        log.debug(" __recvFromSocket :received %d of %d bytes in total in %.1f ms, %.1f Gbps",  total_len, bytes, totalTimeMs, Gbps)

        return buffer


    def __sendToSocket(self, sock, buffer,bytes):
        timeout = self.exposureTime * 10 + 30
        startTime = self.GetTime()
        self.socket.settimeout(timeout) 

        retval = True 
        chunkSize = 4096
        for i in range(0, len(buffer), chunkSize):
            try:
                sock.send(buffer[i:min(len(buffer),i+chunkSize)])
            except socket.timeout:
                
                log.debug(" __sendToSocket : timeout in trying to send %d bytes in %.1f ms", bytes, (self.GetTime() - loopTime)*1000)
                if self.GetTime() - startTime > timeout:
                    log.error(" __recvFromSocket: max timeout %d seconds", timeout)
                    retval = False 
                    break
                else:
                    pass #continue further
            except socket.error as e:
                log.error(f"Error during send: {e}")
                # Handle the error as needed, e.g., close the connection
                retval = False 
                break


        return buffer


    def __saveText(self, image, fileName, textSize):
        text = open(self.debugImagesFolder + fileName + ".txt", "w+") 
        line = "%s: [%d x %d]\n" % (fileName, image.shape[1], image.shape[0])
        text.write(line)
        for i in range(min(image.shape[0], textSize)):
            for j in range(min(image.shape[1], textSize)):
                text.write("%8.3f\t" % image[i][j])

            if image.shape[1] > textSize:
                text.write(" ...\n")
            else:
                text.write("\n")

        if image.shape[0] > textSize:
            text.write("...\n")
        else:
            text.write("\n")
    

    def ParseChangedProperties(self,changedProperties, response):
        value = self.__getParameters(response.acknowledge[0])[0]

        props = value.split("|")

        try:
            for prop in props: 
                p = prop.split(":")

                if len(p) == 2: 
                    changedProperties[p[0]] = p[1]
        except Exception as e:
           log.error("Parse changed properties failed." + e.Message)

        return changedProperties

        


    # method setProperty was renamed to SetProperty. please use SetProperty
    setProperty = SetProperty
    getProperty = GetProperty

    #private members
    width               = 0
    height              = 0
    mmf                 = 0
    usingMmf            = True
    debugImagesFolder   = "D:\\DebugImages\\"
    connected           = False
    cameras             = None
    currCamera          = ""
    refreshProperties   = True
    exposureTime        = 1
    host                = 0
    port                = 0

    #command lists
    LIST_CAMERAS        =  0
    LIST_PROPERTIES     =  1
    LIST_ALLOWED_VALUES =  2
    GET_PROPERTY        =  3
    SET_PROPERTY        =  4
    GET_IMAGE_16U       =  5
    GET_IMAGE_32F       = 10
    STOP_ACQUISITION    = 11 
    GET_RESULT          = 14
    START_ACQUISITION   = 15
    SET_HW_ROI          = 16
    SET_SW_ROI          = 17 
    GET_MOVIE_BUFFER_INFO = 18 
    GET_MOVIE_BUFFER    = 19
    SET_PROPERTY_AND_GET_CHANGED_PROPERTIES = 20
    SET_HW_ROI_AND_GET_CHANGED_PROPERTIES = 21
    SET_SW_ROI_AND_GET_CHANGED_PROPERTIES = 22
    SET_VIRTUAL_MASK = 23
    SAVE_FINAL_AFTER_ACQ = 24
    SET_ENG_MODE = 25
    SET_ENG_MODE_GET_CHANGED_PROPERTIES = 26
    SET_SCAN_SIZE = 27 
    SET_SCAN_ROI = 28
    SET_SCAN_SIZE_AND_GET_CHANGED_PROPERTIES = 29 
    SET_SCAN_ROI__AND_GET_CHANGED_PROPERTIES = 30

MMF_DATA_HEADER_SIZE    = 24
MMF_IMAGE_BUFFER_SIZE   = 8192 * 16384 * 4
MMF_DATA_BUFFER_SIZE    = MMF_IMAGE_BUFFER_SIZE + MMF_DATA_HEADER_SIZE

if int(pyVersion[0]) < 3:
    from cStringIO import StringIO # use string io to speed up, refer to http://www.skymind.com/~ocrow/python_string/
    import pb_2_3_0 as pb
else:
    from io import StringIO #use string io to speed up, refer to http://www.skymind.com/~ocrow/python_string/
    long = int  # python 3 no longer has int
    if  int(pyVersion[1]) >= 10:
        #import pb_3_19_3 as pb
        import pb_3_23_3 as pb
    elif int(pyVersion[1]) >= 8:
        import pb_3_11_4 as pb
    else:
        import pb_3_6_1 as pb
