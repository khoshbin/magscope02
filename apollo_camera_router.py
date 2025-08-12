from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import sys
import numpy as np
from io import BytesIO
import base64
import logging
import asyncio

# Add DEAPI to path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
from pyscope import DEAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/camera", tags=["camera"])

# Global client instance
client: Optional[DEAPI.Client] = None


# Pydantic models
class CameraStatus(BaseModel):
    connected: bool
    current_camera: Optional[str] = None
    server_version: Optional[str] = None
    available_cameras: List[str] = []


class ConnectionConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 13240


class PropertyUpdate(BaseModel):
    name: str
    value: Union[str, int, float, bool]


class ROIConfig(BaseModel):
    offset_x: int = Field(ge=0)
    offset_y: int = Field(ge=0)
    size_x: int = Field(gt=0)
    size_y: int = Field(gt=0)


class AcquisitionConfig(BaseModel):
    number_of_acquisitions: int = Field(default=1, ge=1)
    exposure_time_seconds: Optional[float] = Field(default=None, gt=0)
    frames_per_second: Optional[float] = Field(default=None, gt=0)
    frame_type: str = Field(default="SUMTOTAL")
    pixel_format: str = Field(default="AUTO")


class ImageAttributes(BaseModel):
    frame_width: int
    frame_height: int
    dataset_name: str
    acq_index: int
    acq_finished: bool
    image_index: int
    frame_count: int
    image_min: float
    image_max: float
    image_mean: float
    image_std: float
    eppix: float
    eps: float
    timestamp: float


class FrameTypeEnum(str, Enum):
    SUMTOTAL = "SUMTOTAL"
    SINGLEFRAME_INTEGRATED = "SINGLEFRAME_INTEGRATED"
    SINGLEFRAME_COUNTED = "SINGLEFRAME_COUNTED"
    CUMULATIVE = "CUMULATIVE"


class PixelFormatEnum(str, Enum):
    AUTO = "AUTO"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    FLOAT32 = "FLOAT32"


# Helper functions
def ensure_connected():
    """Ensure client is connected, raise HTTPException if not"""
    global client
    if client is None:
        raise HTTPException(status_code=503, detail="Camera client not initialized")
    # Note: We can't easily check connection status from DEAPI, so we assume it's connected


def get_frame_type_enum(frame_type: str) -> DEAPI.FrameType:
    """Convert string to DEAPI.FrameType enum"""
    try:
        return getattr(DEAPI.FrameType, frame_type.upper())
    except AttributeError:
        raise HTTPException(status_code=400, detail=f"Invalid frame type: {frame_type}")


def get_pixel_format_enum(pixel_format: str) -> DEAPI.PixelFormat:
    """Convert string to DEAPI.PixelFormat enum"""
    try:
        return getattr(DEAPI.PixelFormat, pixel_format.upper())
    except AttributeError:
        raise HTTPException(status_code=400, detail=f"Invalid pixel format: {pixel_format}")


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded string"""
    if image_array is None:
        return ""

    # Convert to bytes
    buffer = BytesIO()
    np.save(buffer, image_array)
    buffer.seek(0)

    # Encode to base64
    return base64.b64encode(buffer.read()).decode('utf-8')


# Connection endpoints
@router.post("/connect", response_model=CameraStatus)
async def connect_camera(config: ConnectionConfig):
    """Connect to DE-Server"""
    global client

    try:
        client = DEAPI.Client()
        client.Connect(host=config.host, port=config.port)

        cameras = client.ListCameras()
        current_camera = cameras[0] if cameras else None

        if current_camera:
            client.SetCurrentCamera(current_camera)

        server_version = client.GetProperty("Server Software Version") if current_camera else None

        logger.info(f"Connected to camera server at {config.host}:{config.port}")

        return CameraStatus(
            connected=True,
            current_camera=current_camera,
            server_version=server_version,
            available_cameras=cameras
        )
    except Exception as e:
        logger.error(f"Failed to connect: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


@router.post("/disconnect")
async def disconnect_camera():
    """Disconnect from DE-Server"""
    global client

    if client:
        try:
            client.Disconnect()
            client = None
            logger.info("Disconnected from camera server")
            return {"message": "Disconnected successfully"}
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")

    return {"message": "No active connection"}


@router.get("/status", response_model=CameraStatus)
async def get_camera_status():
    """Get current camera connection status"""
    global client

    if client is None:
        return CameraStatus(connected=False)

    try:
        cameras = client.ListCameras()
        current_camera = client.GetCurrentCamera()
        server_version = client.GetProperty("Server Software Version")

        return CameraStatus(
            connected=True,
            current_camera=current_camera,
            server_version=server_version,
            available_cameras=cameras
        )
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return CameraStatus(connected=False)


# Camera management endpoints
@router.get("/cameras", response_model=List[str])
async def list_cameras():
    """List available cameras"""
    ensure_connected()

    try:
        cameras = client.ListCameras()
        return cameras
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {str(e)}")


@router.post("/cameras/{camera_name}/select")
async def select_camera(camera_name: str = Path(..., description="Name of camera to select")):
    """Select current camera"""
    ensure_connected()

    try:
        success = client.SetCurrentCamera(camera_name)
        if success:
            logger.info(f"Selected camera: {camera_name}")
            return {"message": f"Camera {camera_name} selected successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to select camera: {camera_name}")
    except Exception as e:
        logger.error(f"Error selecting camera: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to select camera: {str(e)}")


# Property management endpoints
@router.get("/properties", response_model=List[str])
async def list_properties():
    """List all available camera properties"""
    ensure_connected()

    try:
        properties = client.ListProperties()
        return properties
    except Exception as e:
        logger.error(f"Error listing properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list properties: {str(e)}")


@router.get("/properties/all", response_model=Dict[str, Any])
async def get_all_properties():
    """Get all camera properties with their current values"""
    ensure_connected()

    try:
        # Get list of all properties
        property_names = client.ListProperties()
        logger.info(f"Getting values for {len(property_names)} properties")

        # Get current value for each property
        all_properties = {}
        failed_properties = []

        for prop_name in property_names:
            try:
                value = client.GetProperty(prop_name)
                all_properties[prop_name] = value
            except Exception as e:
                logger.warning(f"Failed to get property '{prop_name}': {str(e)}")
                failed_properties.append(prop_name)

        logger.info(f"Successfully retrieved {len(all_properties)} properties")
        if failed_properties:
            logger.warning(f"Failed to retrieve {len(failed_properties)} properties")

        return {
            "properties": all_properties,
            "total_count": len(all_properties),
            "failed_count": len(failed_properties),
            "failed_properties": failed_properties
        }

    except Exception as e:
        logger.error(f"Error getting all properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get all properties: {str(e)}")


@router.get("/properties/detailed")
async def get_all_properties_with_specs():
    """Get all camera properties with their current values and specifications"""
    ensure_connected()

    try:
        # Get list of all properties
        property_names = client.ListProperties()
        logger.info(f"Getting detailed info for {len(property_names)} properties")

        # Get current value and spec for each property
        all_properties = {}
        failed_properties = []

        for prop_name in property_names:
            try:
                # Get current value
                value = client.GetProperty(prop_name)

                # Get property specification
                spec = client.GetPropertySpec(prop_name)

                property_info = {
                    "current_value": value,
                    "data_type": spec.dataType if spec else None,
                    "value_type": spec.valueType if spec else None,
                    "category": spec.category if spec else None,
                    "options": spec.options if spec else None,
                    "default_value": spec.defaultValue if spec else None
                }

                all_properties[prop_name] = property_info

            except Exception as e:
                logger.warning(f"Failed to get detailed info for property '{prop_name}': {str(e)}")
                failed_properties.append(prop_name)

        logger.info(f"Successfully retrieved detailed info for {len(all_properties)} properties")
        if failed_properties:
            logger.warning(f"Failed to retrieve {len(failed_properties)} properties")

        return {
            "properties": all_properties,
            "total_count": len(all_properties),
            "failed_count": len(failed_properties),
            "failed_properties": failed_properties
        }

    except Exception as e:
        logger.error(f"Error getting detailed properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get detailed properties: {str(e)}")


@router.get("/properties/categories")
async def get_properties_by_category():
    """Get all properties organized by category"""
    ensure_connected()

    try:
        # Get list of all properties
        property_names = client.ListProperties()
        logger.info(f"Categorizing {len(property_names)} properties")

        # Group properties by category
        categories = {}
        uncategorized = []
        failed_properties = []

        for prop_name in property_names:
            try:
                # Get property specification to determine category
                spec = client.GetPropertySpec(prop_name)
                category = spec.category if spec and spec.category else "Uncategorized"

                if category not in categories:
                    categories[category] = []

                # Get current value
                value = client.GetProperty(prop_name)

                categories[category].append({
                    "name": prop_name,
                    "value": value,
                    "data_type": spec.dataType if spec else None,
                    "value_type": spec.valueType if spec else None
                })

            except Exception as e:
                logger.warning(f"Failed to categorize property '{prop_name}': {str(e)}")
                failed_properties.append(prop_name)

        # Sort categories and properties within categories
        sorted_categories = {}
        for category in sorted(categories.keys()):
            sorted_categories[category] = sorted(categories[category], key=lambda x: x["name"])

        logger.info(f"Properties organized into {len(sorted_categories)} categories")

        return {
            "categories": sorted_categories,
            "category_counts": {cat: len(props) for cat, props in sorted_categories.items()},
            "total_properties": sum(len(props) for props in sorted_categories.values()),
            "failed_count": len(failed_properties),
            "failed_properties": failed_properties
        }

    except Exception as e:
        logger.error(f"Error categorizing properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to categorize properties: {str(e)}")


# Enhanced property endpoints with search and filtering
@router.get("/properties/search")
async def search_properties(
        query: str = Query(..., description="Search term to find in property names"),
        case_sensitive: bool = Query(default=False, description="Case sensitive search")
):
    """Search for properties by name"""
    ensure_connected()

    try:
        property_names = client.ListProperties()

        if case_sensitive:
            matching_props = [name for name in property_names if query in name]
        else:
            query_lower = query.lower()
            matching_props = [name for name in property_names if query_lower in name.lower()]

        # Get values for matching properties
        properties_with_values = {}
        for prop_name in matching_props:
            try:
                value = client.GetProperty(prop_name)
                properties_with_values[prop_name] = value
            except Exception as e:
                logger.warning(f"Failed to get property '{prop_name}': {str(e)}")

        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "matches": properties_with_values,
            "match_count": len(properties_with_values),
            "total_properties": len(property_names)
        }

    except Exception as e:
        logger.error(f"Error searching properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search properties: {str(e)}")


@router.get("/properties/filter/temperature")
async def get_temperature_properties():
    """Get all temperature-related properties"""
    ensure_connected()

    try:
        property_names = client.ListProperties()
        temp_keywords = ["temperature", "temp", "thermal", "cooling", "chilled"]

        temp_properties = {}
        for prop_name in property_names:
            if any(keyword in prop_name.lower() for keyword in temp_keywords):
                try:
                    value = client.GetProperty(prop_name)
                    temp_properties[prop_name] = value
                except Exception as e:
                    logger.warning(f"Failed to get temperature property '{prop_name}': {str(e)}")

        return {
            "temperature_properties": temp_properties,
            "count": len(temp_properties)
        }

    except Exception as e:
        logger.error(f"Error getting temperature properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get temperature properties: {str(e)}")


@router.get("/properties/filter/acquisition")
async def get_acquisition_properties():
    """Get all acquisition-related properties"""
    ensure_connected()

    try:
        property_names = client.ListProperties()
        acq_keywords = ["acquisition", "exposure", "frame", "fps", "remaining", "total", "autosave"]

        acq_properties = {}
        for prop_name in property_names:
            if any(keyword in prop_name.lower() for keyword in acq_keywords):
                try:
                    value = client.GetProperty(prop_name)
                    acq_properties[prop_name] = value
                except Exception as e:
                    logger.warning(f"Failed to get acquisition property '{prop_name}': {str(e)}")

        return {
            "acquisition_properties": acq_properties,
            "count": len(acq_properties)
        }

    except Exception as e:
        logger.error(f"Error getting acquisition properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get acquisition properties: {str(e)}")


@router.get("/properties/filter/image")
async def get_image_properties():
    """Get all image-related properties"""
    ensure_connected()

    try:
        property_names = client.ListProperties()
        image_keywords = ["image", "size", "pixel", "crop", "binning", "resolution", "processing"]

        image_properties = {}
        for prop_name in property_names:
            if any(keyword in prop_name.lower() for keyword in image_keywords):
                try:
                    value = client.GetProperty(prop_name)
                    image_properties[prop_name] = value
                except Exception as e:
                    logger.warning(f"Failed to get image property '{prop_name}': {str(e)}")

        return {
            "image_properties": image_properties,
            "count": len(image_properties)
        }

    except Exception as e:
        logger.error(f"Error getting image properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get image properties: {str(e)}")


@router.get("/properties/essential")
async def get_essential_properties():
    """Get essential camera properties for quick status check"""
    ensure_connected()

    essential_props = [
        "Camera Model",
        "Camera SN",
        "Sensor Module SN",
        "Server Software Version",
        "Firmware Version",
        "Temperature - Detector (Celsius)",
        "Temperature - Detector Status",
        "Exposure Time (seconds)",
        "Frames Per Second",
        "Image Size X (pixels)",
        "Image Size Y (pixels)",
        "Exposure Mode",
        "Image Processing - Mode",
        "Acquisition Counter",
        "System Status",
        "Protection Cover Status",
        "Camera Position Status",
        "Remaining Number of Acquisitions",
        "Autosave Status"
    ]

    try:
        essential_data = {}
        missing_props = []

        for prop_name in essential_props:
            try:
                value = client.GetProperty(prop_name)
                essential_data[prop_name] = value
            except Exception as e:
                logger.warning(f"Failed to get essential property '{prop_name}': {str(e)}")
                missing_props.append(prop_name)

        return {
            "essential_properties": essential_data,
            "count": len(essential_data),
            "missing_properties": missing_props
        }

    except Exception as e:
        logger.error(f"Error getting essential properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get essential properties: {str(e)}")


@router.post("/properties/batch")
async def set_multiple_properties(properties: Dict[str, Union[str, int, float, bool]]):
    """Set multiple properties at once"""
    ensure_connected()

    try:
        results = {}
        failed_properties = {}

        for prop_name, value in properties.items():
            try:
                success = client.SetProperty(prop_name, value)
                results[prop_name] = {
                    "success": success,
                    "value": value
                }
                if success:
                    logger.info(f"Set property {prop_name} = {value}")
            except Exception as e:
                error_msg = str(e)
                failed_properties[prop_name] = error_msg
                logger.error(f"Failed to set property {prop_name}: {error_msg}")

        return {
            "message": f"Processed {len(properties)} properties",
            "successful": {k: v for k, v in results.items() if v["success"]},
            "failed": failed_properties,
            "success_count": len([r for r in results.values() if r["success"]]),
            "failure_count": len(failed_properties)
        }

    except Exception as e:
        logger.error(f"Error setting multiple properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set properties: {str(e)}")


async def get_property(property_name: str = Path(..., description="Name of property to get")):
    """Get value of a specific property"""
    ensure_connected()

    try:
        value = client.GetProperty(property_name)
        return {"property": property_name, "value": value}
    except Exception as e:
        logger.error(f"Error getting property {property_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get property: {str(e)}")


@router.post("/properties")
async def set_property(prop_update: PropertyUpdate):
    """Set value of a specific property"""
    ensure_connected()

    try:
        success = client.SetProperty(prop_update.name, prop_update.value)
        if success:
            logger.info(f"Set property {prop_update.name} = {prop_update.value}")
            return {"message": f"Property {prop_update.name} set successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to set property {prop_update.name}")
    except Exception as e:
        logger.error(f"Error setting property {prop_update.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set property: {str(e)}")


@router.get("/properties/{property_name}/spec")
async def get_property_spec(property_name: str = Path(..., description="Name of property to get spec for")):
    """Get property specification (allowed values, type, etc.)"""
    ensure_connected()

    try:
        spec = client.GetPropertySpec(property_name)
        if spec:
            return {
                "property": property_name,
                "data_type": spec.dataType,
                "value_type": spec.valueType,
                "category": spec.category,
                "options": spec.options,
                "default_value": spec.defaultValue,
                "current_value": spec.currentValue
            }
        else:
            raise HTTPException(status_code=404, detail=f"Property spec not found: {property_name}")
    except Exception as e:
        logger.error(f"Error getting property spec {property_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get property spec: {str(e)}")


# ROI management endpoints
@router.post("/roi/hardware")
async def set_hardware_roi(roi: ROIConfig):
    """Set hardware ROI (Region of Interest)"""
    ensure_connected()

    try:
        success = client.SetHWROI(roi.offset_x, roi.offset_y, roi.size_x, roi.size_y)
        if success:
            logger.info(f"Set HW ROI: offset=({roi.offset_x},{roi.offset_y}), size=({roi.size_x},{roi.size_y})")
            return {"message": "Hardware ROI set successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to set hardware ROI")
    except Exception as e:
        logger.error(f"Error setting HW ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set hardware ROI: {str(e)}")


@router.post("/roi/software")
async def set_software_roi(roi: ROIConfig):
    """Set software ROI (Region of Interest)"""
    ensure_connected()

    try:
        success = client.SetSWROI(roi.offset_x, roi.offset_y, roi.size_x, roi.size_y)
        if success:
            logger.info(f"Set SW ROI: offset=({roi.offset_x},{roi.offset_y}), size=({roi.size_x},{roi.size_y})")
            return {"message": "Software ROI set successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to set software ROI")
    except Exception as e:
        logger.error(f"Error setting SW ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set software ROI: {str(e)}")


# Temperature control endpoints
@router.post("/temperature/cooldown")
async def start_cooldown():
    """Start camera cooldown process"""
    ensure_connected()

    try:
        client.SetProperty("Temperature - Control", "Cool Down")
        logger.info("Started camera cooldown")
        return {"message": "Cooldown started successfully"}
    except Exception as e:
        logger.error(f"Error starting cooldown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start cooldown: {str(e)}")


@router.post("/temperature/warmup")
async def start_warmup():
    """Start camera warmup process"""
    ensure_connected()

    try:
        client.SetProperty("Temperature - Control", "Warm Up")
        logger.info("Started camera warmup")
        return {"message": "Warmup started successfully"}
    except Exception as e:
        logger.error(f"Error starting warmup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start warmup: {str(e)}")


@router.get("/temperature/status")
async def get_temperature_status():
    """Get current temperature status"""
    ensure_connected()

    try:
        status = client.GetProperty("Temperature - Detector Status")
        temperature = client.GetProperty("Temperature - Detector (Celsius)")

        return {
            "status": status,
            "temperature_celsius": temperature
        }
    except Exception as e:
        logger.error(f"Error getting temperature status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get temperature status: {str(e)}")


# Acquisition endpoints
@router.post("/acquisition/start")
async def start_acquisition(config: AcquisitionConfig):
    """Start image acquisition"""
    ensure_connected()

    try:
        # Set acquisition parameters if provided
        if config.exposure_time_seconds is not None:
            client.SetProperty("Exposure Time (seconds)", config.exposure_time_seconds)

        if config.frames_per_second is not None:
            client.SetProperty("Frames Per Second", config.frames_per_second)

        # Start acquisition
        client.StartAcquisition(config.number_of_acquisitions)

        logger.info(f"Started acquisition: {config.number_of_acquisitions} acquisitions")
        return {"message": f"Acquisition started: {config.number_of_acquisitions} acquisitions"}
    except Exception as e:
        logger.error(f"Error starting acquisition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start acquisition: {str(e)}")


@router.post("/acquisition/stop")
async def stop_acquisition():
    """Stop current acquisition"""
    ensure_connected()

    try:
        success = client.StopAcquisition()
        if success:
            logger.info("Acquisition stopped")
            return {"message": "Acquisition stopped successfully"}
        else:
            return {"message": "Stop command sent, but status unclear"}
    except Exception as e:
        logger.error(f"Error stopping acquisition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop acquisition: {str(e)}")


@router.get("/acquisition/status")
async def get_acquisition_status():
    """Get current acquisition status"""
    ensure_connected()

    try:
        status = client.GetProperty("Acquisition Status")
        remaining = client.GetProperty("Remaining Number of Acquisitions")

        return {
            "status": status,
            "remaining_acquisitions": remaining
        }
    except Exception as e:
        logger.error(f"Error getting acquisition status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get acquisition status: {str(e)}")


# Image retrieval endpoints
@router.get("/image/latest")
async def get_latest_image(
        frame_type: FrameTypeEnum = Query(default=FrameTypeEnum.SUMTOTAL),
        pixel_format: PixelFormatEnum = Query(default=PixelFormatEnum.AUTO),
        include_image_data: bool = Query(default=False, description="Include base64 encoded image data")
):
    """Get the latest image from the camera"""
    ensure_connected()

    try:
        frame_type_enum = get_frame_type_enum(frame_type.value)
        pixel_format_enum = get_pixel_format_enum(pixel_format.value)

        attributes = DEAPI.Attributes()
        histogram = DEAPI.Histogram()

        image, returned_pixel_format, attributes, histogram = client.GetResult(
            frame_type_enum, pixel_format_enum, attributes, histogram
        )

        # Build response
        response = {
            "attributes": ImageAttributes(
                frame_width=attributes.frameWidth,
                frame_height=attributes.frameHeight,
                dataset_name=attributes.datasetName,
                acq_index=attributes.acqIndex,
                acq_finished=attributes.acqFinished,
                image_index=attributes.imageIndex,
                frame_count=attributes.frameCount,
                image_min=attributes.imageMin,
                image_max=attributes.imageMax,
                image_mean=attributes.imageMean,
                image_std=attributes.imageStd,
                eppix=attributes.eppix,
                eps=attributes.eps,
                timestamp=attributes.timestamp
            ),
            "pixel_format": returned_pixel_format.name,
            "histogram": {
                "min": histogram.min,
                "max": histogram.max,
                "bins": histogram.bins,
                "data": histogram.data if histogram.data else []
            }
        }

        if include_image_data:
            response["image_data"] = numpy_to_base64(image)
            response["image_shape"] = image.shape if image is not None else None
            response["image_dtype"] = str(image.dtype) if image is not None else None

        logger.info(f"Retrieved image: {attributes.datasetName}, frame {attributes.acqIndex}")
        return response

    except Exception as e:
        logger.error(f"Error getting latest image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest image: {str(e)}")


# System info endpoints
@router.get("/info/server")
async def get_server_info():
    """Get detailed server and camera information"""
    ensure_connected()

    try:
        info = {
            "server_version": client.GetProperty("Server Software Version"),
            "camera_name": client.GetCurrentCamera(),
            "camera_sn": client.GetProperty("Camera SN"),
            "sensor_sn": client.GetProperty("Sensor Module SN"),
            "firmware_version": client.GetProperty("Firmware Version"),
            "computer_cpu": client.GetProperty("Computer CPU Info"),
            "computer_gpu": client.GetProperty("Computer GPU Info"),
            "computer_memory": client.GetProperty("Computer Memory Info"),
            "sensor_size_x": client.GetProperty("Sensor Size X (pixels)"),
            "sensor_size_y": client.GetProperty("Sensor Size Y (pixels)"),
            "image_size_x": client.GetProperty("Image Size X (pixels)"),
            "image_size_y": client.GetProperty("Image Size Y (pixels)"),
            "max_fps": client.GetProperty("Frames Per Second (Max)"),
            "current_fps": client.GetProperty("Frames Per Second")
        }

        return info
    except Exception as e:
        logger.error(f"Error getting server info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get server info: {str(e)}")


# Reference acquisition endpoints
@router.post("/references/dark")
async def take_dark_reference(frame_rate: float = Query(default=20.0, gt=0)):
    """Take dark reference images"""
    ensure_connected()

    try:
        # Store current settings
        prev_exposure_mode = client.GetProperty("Exposure Mode")
        prev_exposure_time = client.GetProperty("Exposure Time (seconds)")

        # Set dark reference parameters
        client.SetProperty("Exposure Mode", "Dark")
        client.SetProperty("Frames Per Second", frame_rate)
        client.SetProperty("Exposure Time (seconds)", 1)

        # Start acquisition
        acquisitions = 10
        client.StartAcquisition(acquisitions)

        logger.info(f"Started dark reference acquisition at {frame_rate} fps")

        # Wait for completion (simplified - in practice you might want to poll status)
        import time
        time.sleep(2)  # Give it some time to start

        # Restore previous settings
        client.SetProperty("Exposure Mode", prev_exposure_mode)
        client.SetProperty("Exposure Time (seconds)", prev_exposure_time)

        return {"message": f"Dark reference acquisition started at {frame_rate} fps"}

    except Exception as e:
        logger.error(f"Error taking dark reference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to take dark reference: {str(e)}")


# Quick imaging endpoint
@router.post("/quick-image")
async def take_quick_image(
        exposure_time: float = Query(default=1.0, gt=0),
        frame_rate: float = Query(default=20.0, gt=0),
        include_image_data: bool = Query(default=True),
        timeout_seconds: float = Query(default=300.0, gt=0, description="Timeout in seconds for image acquisition")
):
    """Take a quick single image with specified parameters and timeout handling"""
    ensure_connected()

    async def image_acquisition_task():
        """Async wrapper for the image acquisition process"""
        try:
            # Set parameters
            client.SetProperty("Exposure Time (seconds)", exposure_time)
            client.SetProperty("Frames Per Second", frame_rate)
            client.SetProperty("Exposure Mode", "Normal")

            # Start single acquisition
            client.StartAcquisition(1)

            # Wait for completion with polling (non-blocking)
            max_wait_cycles = int(timeout_seconds * 10)  # Check every 100ms
            for _ in range(max_wait_cycles):
                try:
                    status = client.GetProperty("Acquisition Status")
                    remaining = client.GetProperty("Remaining Number of Acquisitions")

                    if remaining == 0 or status == "Complete":
                        break

                    await asyncio.sleep(0.1)  # Non-blocking sleep
                except:
                    await asyncio.sleep(0.1)  # Continue polling if property read fails

            # Get the result
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()

            image, pixel_format, attributes, histogram = client.GetResult(
                DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.AUTO, attributes, histogram
            )

            response = {
                "message": "Quick image captured successfully",
                "exposure_time": exposure_time,
                "frame_rate": frame_rate,
                "timeout_used": False,
                "attributes": ImageAttributes(
                    frame_width=attributes.frameWidth,
                    frame_height=attributes.frameHeight,
                    dataset_name=attributes.datasetName,
                    acq_index=attributes.acqIndex,
                    acq_finished=attributes.acqFinished,
                    image_index=attributes.imageIndex,
                    frame_count=attributes.frameCount,
                    image_min=attributes.imageMin,
                    image_max=attributes.imageMax,
                    image_mean=attributes.imageMean,
                    image_std=attributes.imageStd,
                    eppix=attributes.eppix,
                    eps=attributes.eps,
                    timestamp=attributes.timestamp
                )
            }

            if include_image_data:
                response["image_data"] = numpy_to_base64(image)
                response["image_shape"] = image.shape if image is not None else None
                response["image_dtype"] = str(image.dtype) if image is not None else None

            return response

        except Exception as e:
            logger.error(f"Error in image acquisition task: {str(e)}")
            raise

    try:
        # Use asyncio.wait_for to implement timeout
        result = await asyncio.wait_for(image_acquisition_task(), timeout=timeout_seconds)

        logger.info(f"Quick image captured: {exposure_time}s exposure at {frame_rate} fps")
        return result

    except asyncio.TimeoutError:
        # Handle timeout gracefully
        logger.warning(f"Image acquisition timed out after {timeout_seconds} seconds")

        # Try to stop the acquisition
        try:
            client.StopAcquisition()
        except:
            pass

        # Return timeout response
        return {
            "message": f"Image acquisition timed out after {timeout_seconds} seconds",
            "exposure_time": exposure_time,
            "frame_rate": frame_rate,
            "timeout_used": True,
            "timeout_seconds": timeout_seconds,
            "error": "timeout"
        }

    except Exception as e:
        logger.error(f"Error taking quick image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to take quick image: {str(e)}")


# Alternative: Using run_in_threadpool for blocking operations
from fastapi.concurrency import run_in_threadpool


@router.post("/quick-image-threaded")
async def take_quick_image_threaded(
        exposure_time: float = Query(default=1.0, gt=0),
        frame_rate: float = Query(default=20.0, gt=0),
        include_image_data: bool = Query(default=True),
        timeout_seconds: float = Query(default=300.0, gt=0)
):
    """Alternative implementation using threadpool for blocking operations"""
    ensure_connected()

    def blocking_image_acquisition():
        """Blocking image acquisition function to run in threadpool"""
        # Set parameters
        client.SetProperty("Exposure Time (seconds)", exposure_time)
        client.SetProperty("Frames Per Second", frame_rate)
        client.SetProperty("Exposure Mode", "Normal")

        # Start single acquisition
        client.StartAcquisition(1)

        # Wait for completion (blocking)
        import time
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                status = client.GetProperty("Acquisition Status")
                remaining = client.GetProperty("Remaining Number of Acquisitions")

                if remaining == 0 or status == "Complete":
                    break

                time.sleep(0.1)  # Blocking sleep in thread
            except:
                time.sleep(0.1)
        else:
            # Timeout occurred
            raise TimeoutError(f"Acquisition timed out after {timeout_seconds} seconds")

        # Get the result
        attributes = DEAPI.Attributes()
        histogram = DEAPI.Histogram()

        image, pixel_format, attributes, histogram = client.GetResult(
            DEAPI.FrameType.SUMTOTAL, DEAPI.PixelFormat.AUTO, attributes, histogram
        )

        return image, pixel_format, attributes, histogram

    try:
        # Run the blocking operation in a separate thread with timeout
        image, pixel_format, attributes, histogram = await asyncio.wait_for(
            run_in_threadpool(blocking_image_acquisition),
            timeout=timeout_seconds
        )

        response = {
            "message": "Quick image captured successfully (threaded)",
            "exposure_time": exposure_time,
            "frame_rate": frame_rate,
            "timeout_used": False,
            "attributes": ImageAttributes(
                frame_width=attributes.frameWidth,
                frame_height=attributes.frameHeight,
                dataset_name=attributes.datasetName,
                acq_index=attributes.acqIndex,
                acq_finished=attributes.acqFinished,
                image_index=attributes.imageIndex,
                frame_count=attributes.frameCount,
                image_min=attributes.imageMin,
                image_max=attributes.imageMax,
                image_mean=attributes.imageMean,
                image_std=attributes.imageStd,
                eppix=attributes.eppix,
                eps=attributes.eps,
                timestamp=attributes.timestamp
            )
        }

        if include_image_data:
            response["image_data"] = numpy_to_base64(image)
            response["image_shape"] = image.shape if image is not None else None
            response["image_dtype"] = str(image.dtype) if image is not None else None

        logger.info(f"Quick image captured (threaded): {exposure_time}s exposure at {frame_rate} fps")
        return response

    except asyncio.TimeoutError:
        logger.warning(f"Image acquisition timed out after {timeout_seconds} seconds")

        try:
            client.StopAcquisition()
        except:
            pass

        return {
            "message": f"Image acquisition timed out after {timeout_seconds} seconds",
            "exposure_time": exposure_time,
            "frame_rate": frame_rate,
            "timeout_used": True,
            "timeout_seconds": timeout_seconds,
            "error": "timeout"
        }

    except Exception as e:
        logger.error(f"Error taking quick image (threaded): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to take quick image: {str(e)}")