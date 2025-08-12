#!/usr/bin/env python3
"""
Complete PyScope FastAPI Camera Router
Full-featured FastAPI implementation using PyScope for DE Apollo camera control
Mirrors the functionality of the original DEAPI router with PyScope backend
"""

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import sys
import numpy as np
from io import BytesIO
import base64
import logging
import time
import asyncio
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress DE camera property warnings that are harmless
de_logger = logging.getLogger('DEClient')
de_logger.setLevel(logging.ERROR)

# Create router
router = APIRouter(prefix="/pyscope-camera", tags=["pyscope-camera"])

# Global camera instance
camera_instance: Optional[Any] = None
camera_connected: bool = False
camera_name: str = ""


# Pydantic Models
class CameraStatus(BaseModel):
    connected: bool
    camera_name: Optional[str] = None
    camera_model: Optional[str] = None
    camera_size: Optional[Dict[str, int]] = None
    available_cameras: List[str] = []
    access_method: Optional[str] = None


class ConnectionConfig(BaseModel):
    camera_name: Optional[str] = "DEApollo"
    auto_connect: bool = True


class PropertyUpdate(BaseModel):
    name: str
    value: Union[str, int, float, bool]


class PropertyBatch(BaseModel):
    properties: Dict[str, Union[str, int, float, bool]]


class CameraGeometry(BaseModel):
    binning_x: int = Field(ge=1, le=8)
    binning_y: int = Field(ge=1, le=8)
    dimension_x: int = Field(ge=1, le=8192)
    dimension_y: int = Field(ge=1, le=8192)
    offset_x: int = Field(ge=0)
    offset_y: int = Field(ge=0)


class AcquisitionConfig(BaseModel):
    exposure_time_seconds: Optional[float] = Field(default=None, gt=0)
    frames_per_second: Optional[float] = Field(default=None, gt=0)
    binning_x: Optional[int] = Field(default=None, ge=1, le=8)
    binning_y: Optional[int] = Field(default=None, ge=1, le=8)
    crop_size_x: Optional[int] = Field(default=None, ge=1, le=8192)
    crop_size_y: Optional[int] = Field(default=None, ge=1, le=8192)
    exposure_mode: Optional[str] = Field(default=None)


class ImageResponse(BaseModel):
    success: bool
    message: str
    image_info: Optional[Dict[str, Any]] = None
    image_data: Optional[str] = None  # Base64 encoded
    acquisition_time: Optional[float] = None


class TemperatureStatus(BaseModel):
    detector_temperature: Optional[float] = None
    detector_status: Optional[str] = None
    chilled_water_temperature: Optional[float] = None
    chilled_water_status: Optional[str] = None
    control_status: Optional[str] = None


# Utility Functions
def ensure_connected():
    """Ensure camera is connected, raise HTTPException if not"""
    global camera_instance, camera_connected
    if not camera_connected or camera_instance is None:
        raise HTTPException(status_code=503, detail="Camera not connected")


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded string"""
    if image_array is None:
        return ""

    try:
        # Normalize to uint8 for web transfer
        if image_array.dtype == np.uint16:
            # Scale 16-bit to 8-bit
            image_8bit = (image_array / 256).astype(np.uint8)
        elif image_array.dtype in [np.float32, np.float64]:
            # Normalize float to 0-255
            img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            image_8bit = (img_norm * 255).astype(np.uint8)
        else:
            image_8bit = image_array.astype(np.uint8)

        # Convert to bytes
        buffer = BytesIO()
        np.save(buffer, image_8bit)
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""


class PyScopeCameraManager:
    """Enhanced camera manager for PyScope DE cameras"""

    def __init__(self):
        self.camera = None
        self.camera_name = None
        self.connected = False
        self.properties_cache = {}
        self.last_property_refresh = 0
        self.cache_timeout = 5.0  # Cache properties for 5 seconds

    def connect(self, camera_name: str = "DEApollo"):
        """Connect to DE camera via PyScope"""
        try:
            import pyscope.registry

            # Try different camera names
            camera_attempts = [camera_name, "DEApollo", "DE Apollo", "DE12", "Apollo"]

            for attempt_name in camera_attempts:
                try:
                    logger.info(f"Attempting PyScope connection to: {attempt_name}")

                    # Create camera instance using PyScope registry
                    self.camera = pyscope.registry.getClass(attempt_name)()
                    self.camera_name = attempt_name

                    # Test basic connectivity
                    camera_size = self.camera.getCameraSize()
                    logger.info(f"✓ PyScope connected to {attempt_name}")
                    logger.info(f"  Camera size: {camera_size}")

                    # Initialize properties list
                    try:
                        if hasattr(self.camera, 'getPropertiesList'):
                            self.camera.getPropertiesList()
                        logger.info("✓ Properties initialized")
                    except Exception as e:
                        logger.debug(f"Properties initialization warning: {e}")

                    self.connected = True
                    return True

                except Exception as e:
                    logger.debug(f"  PyScope connection failed for {attempt_name}: {e}")
                    continue

            logger.error("All PyScope camera connection attempts failed")
            return False

        except Exception as e:
            logger.error(f"Error in camera connection process: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera"""
        if self.camera:
            try:
                # PyScope cameras disconnect automatically
                self.connected = False
                self.camera = None
                self.camera_name = None
                logger.info("✓ Disconnected from camera")
                return True
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
                return False
        return True

    def get_model_name(self) -> str:
        """Get camera model name for DE API calls"""
        try:
            if hasattr(self.camera, 'model_name'):
                return self.camera.model_name
            elif hasattr(self.camera, 'de_name'):
                return self.camera.de_name
            elif 'Apollo' in self.camera_name:
                return 'Apollo'
            else:
                return 'Apollo'
        except:
            return 'Apollo'

    def get_available_properties(self) -> List[str]:
        """Get list of all available properties"""
        if not self.connected:
            return []

        try:
            # Try to access camera's properties list
            if hasattr(self.camera, 'properties_list') and self.camera.properties_list:
                return self.camera.properties_list

            # Try to get properties via internal methods
            if hasattr(self.camera, 'getPropertiesList'):
                self.camera.getPropertiesList()
                if hasattr(self.camera, 'properties_list'):
                    return self.camera.properties_list

            # Fallback to basic properties
            return [
                'Exposure Time (seconds)', 'Frames Per Second', 'Camera Model', 'Camera Name',
                'Sensor Size X (pixels)', 'Sensor Size Y (pixels)', 'Image Size X (pixels)', 'Image Size Y (pixels)',
                'Binning X', 'Binning Y', 'Temperature - Detector (Celsius)', 'System Status'
            ]

        except Exception as e:
            logger.error(f"Error getting available properties: {e}")
            return []

    def get_property(self, name: str) -> Any:
        """Get property value with multiple access methods"""
        if not self.connected:
            return None

        try:
            # Method 1: Direct camera getProperty
            if hasattr(self.camera, 'getProperty'):
                try:
                    return self.camera.getProperty(name)
                except Exception as e:
                    logger.debug(f"Direct getProperty failed for '{name}': {e}")

            # Method 2: Property mappings for common properties
            property_mappings = {
                'Exposure Time (seconds)': lambda: self.camera.getExposureTime() / 1000.0,
                'Frames Per Second': lambda: getattr(self.camera, 'frames_per_second', 60.0),
                'Camera Model': lambda: self.get_model_name(),
                'Camera Name': lambda: self.get_model_name(),
                'Sensor Size X (pixels)': lambda: self.camera.getCameraSize()['x'],
                'Sensor Size Y (pixels)': lambda: self.camera.getCameraSize()['y'],
                'Image Size X (pixels)': lambda: self.camera.getDimension().get('x', self.camera.getCameraSize()['x']),
                'Image Size Y (pixels)': lambda: self.camera.getDimension().get('y', self.camera.getCameraSize()['y']),
                'Binning X': lambda: self.camera.getBinning()['x'],
                'Binning Y': lambda: self.camera.getBinning()['y']
            }

            if name in property_mappings:
                return property_mappings[name]()

            return None

        except Exception as e:
            logger.debug(f"Error getting property '{name}': {e}")
            return None

    def set_property(self, name: str, value: Any) -> bool:
        """Set property value"""
        if not self.connected:
            return False

        try:
            # Method 1: Direct camera setProperty
            if hasattr(self.camera, 'setProperty'):
                try:
                    self.camera.setProperty(name, value)
                    return True
                except Exception as e:
                    logger.debug(f"Direct setProperty failed for '{name}': {e}")

            # Method 2: Specific setter methods
            property_setters = {
                'Exposure Time (seconds)': lambda v: self.camera.setExposureTime(v * 1000),
                'Binning X': lambda v: self.camera.setBinning({'x': v, 'y': self.camera.getBinning()['y']}),
                'Binning Y': lambda v: self.camera.setBinning({'x': self.camera.getBinning()['x'], 'y': v})
            }

            if name in property_setters:
                property_setters[name](value)
                return True

            return False

        except Exception as e:
            logger.error(f"Error setting property '{name}' to '{value}': {e}")
            return False

    def get_all_properties(self) -> Dict[str, Any]:
        """Get all properties with caching"""
        current_time = time.time()

        # Return cached properties if recent
        if (current_time - self.last_property_refresh) < self.cache_timeout and self.properties_cache:
            return self.properties_cache

        if not self.connected:
            return {}

        properties = {}
        property_names = self.get_available_properties()

        for prop_name in property_names:
            try:
                value = self.get_property(prop_name)
                if value is not None:
                    properties[prop_name] = value
            except Exception as e:
                logger.debug(f"Failed to get property '{prop_name}': {e}")

        self.properties_cache = properties
        self.last_property_refresh = current_time

        return properties

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture image from camera"""
        if not self.connected:
            return None

        try:
            return self.camera.getImage()
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None


# Global camera manager
camera_manager = PyScopeCameraManager()


# Connection Endpoints
@router.post("/connect", response_model=CameraStatus)
async def connect_camera(config: ConnectionConfig):
    """Connect to DE camera via PyScope"""
    global camera_instance, camera_connected, camera_name

    try:
        success = camera_manager.connect(config.camera_name)

        if success:
            camera_instance = camera_manager.camera
            camera_connected = True
            camera_name = camera_manager.camera_name

            # Get camera information
            camera_size = camera_manager.camera.getCameraSize()
            camera_model = camera_manager.get_property("Camera Model") or "Apollo"

            # Determine access method
            properties = camera_manager.get_available_properties()
            access_method = "Full-DEAPI" if len(properties) > 50 else "PyScope-Limited"

            logger.info(f"Connected to PyScope camera: {camera_name}")

            return CameraStatus(
                connected=True,
                camera_name=camera_name,
                camera_model=camera_model,
                camera_size=camera_size,
                available_cameras=[camera_name],
                access_method=access_method
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to connect to any camera")

    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


@router.post("/disconnect")
async def disconnect_camera():
    """Disconnect from camera"""
    global camera_instance, camera_connected, camera_name

    try:
        success = camera_manager.disconnect()

        camera_instance = None
        camera_connected = False
        camera_name = ""

        return {"message": "Disconnected successfully", "success": success}
    except Exception as e:
        logger.error(f"Error during disconnect: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")


@router.get("/status", response_model=CameraStatus)
async def get_camera_status():
    """Get current camera connection status"""
    if not camera_connected:
        return CameraStatus(connected=False)

    try:
        camera_size = camera_manager.camera.getCameraSize()
        camera_model = camera_manager.get_property("Camera Model") or "Apollo"
        properties = camera_manager.get_available_properties()
        access_method = "Full-DEAPI" if len(properties) > 50 else "PyScope-Limited"

        return CameraStatus(
            connected=True,
            camera_name=camera_name,
            camera_model=camera_model,
            camera_size=camera_size,
            available_cameras=[camera_name],
            access_method=access_method
        )
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return CameraStatus(connected=False)


# Property Management Endpoints
@router.get("/properties", response_model=List[str])
async def list_properties():
    """List all available camera properties"""
    ensure_connected()

    try:
        properties = camera_manager.get_available_properties()
        return properties
    except Exception as e:
        logger.error(f"Error listing properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list properties: {str(e)}")


@router.get("/properties/all")
async def get_all_properties():
    """Get all camera properties with their current values"""
    ensure_connected()

    try:
        properties = camera_manager.get_all_properties()

        return {
            "properties": properties,
            "total_count": len(properties),
            "failed_count": 0,
            "failed_properties": [],
            "camera_name": camera_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting all properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get all properties: {str(e)}")


@router.get("/properties/essential")
async def get_essential_properties():
    """Get essential camera properties for quick status check"""
    ensure_connected()

    essential_props = [
        "Camera Model", "Camera SN", "Server Software Version", "Firmware Version",
        "Temperature - Detector (Celsius)", "Temperature - Detector Status",
        "Exposure Time (seconds)", "Frames Per Second",
        "Image Size X (pixels)", "Image Size Y (pixels)",
        "Exposure Mode", "System Status", "Camera Position Status"
    ]

    try:
        essential_data = {}
        missing_props = []

        for prop_name in essential_props:
            try:
                value = camera_manager.get_property(prop_name)
                if value is not None:
                    essential_data[prop_name] = value
                else:
                    missing_props.append(prop_name)
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


@router.get("/properties/search")
async def search_properties(
        query: str = Query(..., description="Search term to find in property names"),
        case_sensitive: bool = Query(default=False, description="Case sensitive search")
):
    """Search for properties by name"""
    ensure_connected()

    try:
        all_properties = camera_manager.get_all_properties()

        if case_sensitive:
            matching_props = {name: value for name, value in all_properties.items() if query in name}
        else:
            query_lower = query.lower()
            matching_props = {name: value for name, value in all_properties.items() if query_lower in name.lower()}

        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "matches": matching_props,
            "match_count": len(matching_props),
            "total_properties": len(all_properties)
        }
    except Exception as e:
        logger.error(f"Error searching properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search properties: {str(e)}")


@router.get("/properties/categories")
async def get_properties_by_category():
    """Get properties organized by category"""
    ensure_connected()

    try:
        all_properties = camera_manager.get_all_properties()

        # Define categories for DE camera properties
        categories = {
            "Camera Info": ["Camera Model", "Camera Name", "Camera SN", "Sensor Module SN",
                            "Server Software Version", "Firmware Version"],
            "Temperature": ["Temperature - Detector (Celsius)", "Temperature - Detector Status",
                            "Temperature - Control", "Temperature - Chilled Water (Celsius)",
                            "Temperature - Chilled Water Status"],
            "Acquisition": ["Exposure Time (seconds)", "Frames Per Second", "Exposure Mode",
                            "Acquisition Counter", "Remaining Number of Acquisitions",
                            "Total Number of Acquisitions", "Frame Count"],
            "Image Settings": ["Image Size X (pixels)", "Image Size Y (pixels)",
                               "Sensor Size X (pixels)", "Sensor Size Y (pixels)",
                               "Binning X", "Binning Y", "Binning Method"],
            "ROI/Crop": ["Crop Offset X", "Crop Offset Y", "Crop Size X", "Crop Size Y"],
            "Processing": ["Image Processing - Mode", "Image Processing - Flatfield Correction",
                           "Image Processing - Apply Gain on Final", "Image Processing - Apply Gain on Movie"],
            "Status": ["System Status", "Protection Cover Status", "Camera Position Status",
                       "Autosave Status", "Vacuum State"],
            "Autosave": ["Autosave Directory", "Autosave File Format", "Autosave Final Image",
                         "Autosave Movie", "Autosave Movie Sum Count"]
        }

        # Organize properties into categories
        categorized = {}
        uncategorized = []

        for category, prop_list in categories.items():
            categorized[category] = []
            for prop_name in prop_list:
                if prop_name in all_properties:
                    categorized[category].append({
                        "name": prop_name,
                        "value": all_properties[prop_name]
                    })

        # Find uncategorized properties
        all_categorized_props = set()
        for prop_list in categories.values():
            all_categorized_props.update(prop_list)

        for prop_name, value in all_properties.items():
            if prop_name not in all_categorized_props:
                uncategorized.append({
                    "name": prop_name,
                    "value": value
                })

        if uncategorized:
            categorized["Uncategorized"] = uncategorized

        return {
            "categories": categorized,
            "category_counts": {cat: len(props) for cat, props in categorized.items()},
            "total_properties": len(all_properties)
        }
    except Exception as e:
        logger.error(f"Error categorizing properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to categorize properties: {str(e)}")


@router.get("/properties/{property_name}")
async def get_property(property_name: str = Path(..., description="Name of property to get")):
    """Get value of a specific property"""
    ensure_connected()

    try:
        value = camera_manager.get_property(property_name)
        if value is not None:
            return {"property": property_name, "value": value}
        else:
            raise HTTPException(status_code=404, detail=f"Property not found or inaccessible: {property_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting property {property_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get property: {str(e)}")


@router.post("/properties/{property_name}")
async def set_property(
        property_name: str = Path(..., description="Name of property to set"),
        prop_update: PropertyUpdate = None
):
    """Set value of a specific property"""
    ensure_connected()

    # Handle both path parameter and body parameter approaches
    if prop_update:
        name = prop_update.name
        value = prop_update.value
    else:
        name = property_name
        # For simple path-based setting, we'd need the value in query params
        raise HTTPException(status_code=400, detail="Property value must be provided in request body")

    try:
        success = camera_manager.set_property(name, value)
        if success:
            logger.info(f"Set property {name} = {value}")
            return {"message": f"Property {name} set successfully", "property": name, "value": value}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to set property {name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting property {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set property: {str(e)}")


@router.post("/properties/batch")
async def set_multiple_properties(batch: PropertyBatch):
    """Set multiple properties at once"""
    ensure_connected()

    try:
        results = {}
        failed_properties = {}

        for prop_name, value in batch.properties.items():
            try:
                success = camera_manager.set_property(prop_name, value)
                results[prop_name] = {
                    "success": success,
                    "value": value
                }
                if success:
                    logger.info(f"Set property {prop_name} = {value}")
                else:
                    failed_properties[prop_name] = "Set operation returned False"
            except Exception as e:
                error_msg = str(e)
                failed_properties[prop_name] = error_msg
                logger.error(f"Failed to set property {prop_name}: {error_msg}")

        return {
            "message": f"Processed {len(batch.properties)} properties",
            "successful": {k: v for k, v in results.items() if v["success"]},
            "failed": failed_properties,
            "success_count": len([r for r in results.values() if r["success"]]),
            "failure_count": len(failed_properties)
        }
    except Exception as e:
        logger.error(f"Error setting multiple properties: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set properties: {str(e)}")


# Camera Geometry Endpoints
@router.get("/geometry")
async def get_camera_geometry():
    """Get current camera geometry settings"""
    ensure_connected()

    try:
        geometry = {
            "binning_x": camera_manager.get_property("Binning X") or 1,
            "binning_y": camera_manager.get_property("Binning Y") or 1,
            "dimension_x": camera_manager.get_property("Image Size X (pixels)") or 8192,
            "dimension_y": camera_manager.get_property("Image Size Y (pixels)") or 8192,
            "offset_x": camera_manager.get_property("Crop Offset X") or 0,
            "offset_y": camera_manager.get_property("Crop Offset Y") or 0,
            "crop_size_x": camera_manager.get_property("Crop Size X") or 8192,
            "crop_size_y": camera_manager.get_property("Crop Size Y") or 8192,
            "sensor_size_x": camera_manager.get_property("Sensor Size X (pixels)") or 4096,
            "sensor_size_y": camera_manager.get_property("Sensor Size Y (pixels)") or 4096
        }

        return geometry
    except Exception as e:
        logger.error(f"Error getting camera geometry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get camera geometry: {str(e)}")


@router.post("/geometry")
async def set_camera_geometry(geometry: CameraGeometry):
    """Set camera geometry settings"""
    ensure_connected()

    try:
        results = {}

        # Set binning
        if hasattr(camera_manager.camera, 'setBinning'):
            camera_manager.camera.setBinning({'x': geometry.binning_x, 'y': geometry.binning_y})
            results['binning'] = f"Set to {geometry.binning_x}x{geometry.binning_y}"
        else:
            # Try individual property setting
            camera_manager.set_property("Binning X", geometry.binning_x)
            camera_manager.set_property("Binning Y", geometry.binning_y)
            results['binning'] = f"Set to {geometry.binning_x}x{geometry.binning_y}"

        # Set dimensions via crop size
        camera_manager.set_property("Crop Size X", geometry.dimension_x)
        camera_manager.set_property("Crop Size Y", geometry.dimension_y)
        results['dimension'] = f"Set to {geometry.dimension_x}x{geometry.dimension_y}"

        # Set offset
        camera_manager.set_property("Crop Offset X", geometry.offset_x)
        camera_manager.set_property("Crop Offset Y", geometry.offset_y)
        results['offset'] = f"Set to {geometry.offset_x},{geometry.offset_y}"

        return {
            "message": "Camera geometry updated successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error setting camera geometry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set camera geometry: {str(e)}")


# Temperature Control Endpoints
@router.get("/temperature", response_model=TemperatureStatus)
async def get_temperature_status():
    """Get current temperature status"""
    ensure_connected()

    try:
        return TemperatureStatus(
            detector_temperature=camera_manager.get_property("Temperature - Detector (Celsius)"),
            detector_status=camera_manager.get_property("Temperature - Detector Status"),
            chilled_water_temperature=camera_manager.get_property("Temperature - Chilled Water (Celsius)"),
            chilled_water_status=camera_manager.get_property("Temperature - Chilled Water Status"),
            control_status=camera_manager.get_property("Temperature - Control")
        )
    except Exception as e:
        logger.error(f"Error getting temperature status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get temperature status: {str(e)}")


@router.post("/temperature/cooldown")
async def start_cooldown():
    """Start camera cooldown process"""
    ensure_connected()

    try:
        success = camera_manager.set_property("Temperature - Control", "Cool Down")
        if success:
            logger.info("Started camera cooldown")
            return {"message": "Cooldown started successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start cooldown")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting cooldown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start cooldown: {str(e)}")


@router.post("/temperature/warmup")
async def start_warmup():
    """Start camera warmup process"""
    ensure_connected()

    try:
        success = camera_manager.set_property("Temperature - Control", "Warm Up")
        if success:
            logger.info("Started camera warmup")
            return {"message": "Warmup started successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start warmup")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting warmup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start warmup: {str(e)}")


# Image Acquisition Endpoints
@router.post("/acquisition/configure")
async def configure_acquisition(config: AcquisitionConfig):
    """Configure camera for acquisition"""
    ensure_connected()

    try:
        results = {}

        # Set exposure time
        if config.exposure_time_seconds is not None:
            if hasattr(camera_manager.camera, 'setExposureTime'):
                camera_manager.camera.setExposureTime(config.exposure_time_seconds * 1000)
                results['exposure_time'] = f"Set to {config.exposure_time_seconds}s"
            else:
                success = camera_manager.set_property("Exposure Time (seconds)", config.exposure_time_seconds)
                if success:
                    results['exposure_time'] = f"Set to {config.exposure_time_seconds}s"

        # Set frames per second
        if config.frames_per_second is not None:
            success = camera_manager.set_property("Frames Per Second", config.frames_per_second)
            if success:
                results['frames_per_second'] = f"Set to {config.frames_per_second} fps"

        # Set binning
        if config.binning_x is not None and config.binning_y is not None:
            if hasattr(camera_manager.camera, 'setBinning'):
                camera_manager.camera.setBinning({'x': config.binning_x, 'y': config.binning_y})
                results['binning'] = f"Set to {config.binning_x}x{config.binning_y}"

        # Set crop size
        if config.crop_size_x is not None:
            success = camera_manager.set_property("Crop Size X", config.crop_size_x)
            if success:
                results['crop_size_x'] = f"Set to {config.crop_size_x}"

        if config.crop_size_y is not None:
            success = camera_manager.set_property("Crop Size Y", config.crop_size_y)
            if success:
                results['crop_size_y'] = f"Set to {config.crop_size_y}"

        # Set exposure mode
        if config.exposure_mode is not None:
            success = camera_manager.set_property("Exposure Mode", config.exposure_mode)
            if success:
                results['exposure_mode'] = f"Set to {config.exposure_mode}"

        return {
            "message": "Acquisition configured successfully",
            "configuration": results
        }
    except Exception as e:
        logger.error(f"Error configuring acquisition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure acquisition: {str(e)}")


@router.post("/acquisition/capture", response_model=ImageResponse)
async def capture_image(
        include_image_data: bool = Query(default=False, description="Include base64 encoded image data"),
        timeout_seconds: float = Query(default=30.0, gt=0, description="Timeout for image acquisition")
):
    """Capture a single image"""
    ensure_connected()

    try:
        start_time = time.time()

        # Capture image
        image = camera_manager.capture_image()

        if image is None:
            return ImageResponse(
                success=False,
                message="Failed to capture image - camera returned None"
            )

        acquisition_time = time.time() - start_time

        # Get image info
        image_info = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min_value": float(np.min(image)),
            "max_value": float(np.max(image)),
            "mean_value": float(np.mean(image)),
            "std_value": float(np.std(image)),
            "size_bytes": image.nbytes
        }

        # Encode image if requested
        image_data = None
        if include_image_data:
            image_data = numpy_to_base64(image)

        logger.info(f"Image captured successfully in {acquisition_time:.3f}s")

        return ImageResponse(
            success=True,
            message="Image captured successfully",
            image_info=image_info,
            image_data=image_data,
            acquisition_time=acquisition_time
        )

    except Exception as e:
        logger.error(f"Error capturing image: {str(e)}")
        return ImageResponse(
            success=False,
            message=f"Image capture failed: {str(e)}"
        )


@router.post("/acquisition/quick-capture", response_model=ImageResponse)
async def quick_capture_image(
        exposure_time: float = Query(default=1.0, gt=0, description="Exposure time in seconds"),
        include_image_data: bool = Query(default=True, description="Include base64 encoded image data")
):
    """Quick image capture with specified exposure time"""
    ensure_connected()

    try:
        # Configure for quick capture
        if hasattr(camera_manager.camera, 'setExposureTime'):
            camera_manager.camera.setExposureTime(exposure_time * 1000)
        else:
            camera_manager.set_property("Exposure Time (seconds)", exposure_time)

        # Set to normal mode
        camera_manager.set_property("Exposure Mode", "Normal")

        # Small delay to ensure settings take effect
        await asyncio.sleep(0.1)

        # Capture image
        start_time = time.time()
        image = camera_manager.capture_image()
        acquisition_time = time.time() - start_time

        if image is None:
            return ImageResponse(
                success=False,
                message="Quick capture failed - camera returned None"
            )

        # Get image info
        image_info = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min_value": float(np.min(image)),
            "max_value": float(np.max(image)),
            "mean_value": float(np.mean(image)),
            "std_value": float(np.std(image)),
            "exposure_time_used": exposure_time,
            "size_bytes": image.nbytes
        }

        # Encode image if requested
        image_data = None
        if include_image_data:
            image_data = numpy_to_base64(image)

        logger.info(f"Quick capture completed in {acquisition_time:.3f}s with {exposure_time}s exposure")

        return ImageResponse(
            success=True,
            message=f"Quick capture successful ({exposure_time}s exposure)",
            image_info=image_info,
            image_data=image_data,
            acquisition_time=acquisition_time
        )

    except Exception as e:
        logger.error(f"Error in quick capture: {str(e)}")
        return ImageResponse(
            success=False,
            message=f"Quick capture failed: {str(e)}"
        )


# System Information Endpoints
@router.get("/info/system")
async def get_system_info():
    """Get detailed system and camera information"""
    ensure_connected()

    try:
        # Get camera info
        camera_size = camera_manager.camera.getCameraSize()

        info = {
            "camera_name": camera_name,
            "camera_model": camera_manager.get_property("Camera Model"),
            "camera_sn": camera_manager.get_property("Camera SN"),
            "sensor_sn": camera_manager.get_property("Sensor Module SN"),
            "firmware_version": camera_manager.get_property("Firmware Version"),
            "server_version": camera_manager.get_property("Server Software Version"),
            "sensor_size_x": camera_manager.get_property("Sensor Size X (pixels)"),
            "sensor_size_y": camera_manager.get_property("Sensor Size Y (pixels)"),
            "image_size_x": camera_manager.get_property("Image Size X (pixels)"),
            "image_size_y": camera_manager.get_property("Image Size Y (pixels)"),
            "camera_size": camera_size,
            "system_status": camera_manager.get_property("System Status"),
            "camera_position": camera_manager.get_property("Camera Position Status"),
            "protection_cover": camera_manager.get_property("Protection Cover Status"),
            "vacuum_state": camera_manager.get_property("Vacuum State"),
            "connection_method": "PyScope",
            "properties_available": len(camera_manager.get_available_properties())
        }

        return info
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@router.get("/info/performance")
async def get_performance_info():
    """Get camera performance information"""
    ensure_connected()

    try:
        # Test property access performance
        start_time = time.time()
        properties = camera_manager.get_all_properties()
        property_time = time.time() - start_time

        # Test image acquisition performance (small crop)
        original_crop_x = camera_manager.get_property("Crop Size X")
        original_crop_y = camera_manager.get_property("Crop Size Y")

        # Set small crop for performance test
        camera_manager.set_property("Crop Size X", 512)
        camera_manager.set_property("Crop Size Y", 512)

        start_time = time.time()
        test_image = camera_manager.capture_image()
        acquisition_time = time.time() - start_time if test_image is not None else None

        # Restore original crop
        if original_crop_x:
            camera_manager.set_property("Crop Size X", original_crop_x)
        if original_crop_y:
            camera_manager.set_property("Crop Size Y", original_crop_y)

        performance_info = {
            "property_access": {
                "total_properties": len(properties),
                "total_time_seconds": property_time,
                "average_time_ms": (property_time / len(properties) * 1000) if properties else 0
            },
            "image_acquisition": {
                "test_size": "512x512",
                "acquisition_time_seconds": acquisition_time,
                "success": acquisition_time is not None
            },
            "current_fps": camera_manager.get_property("Frames Per Second"),
            "max_fps": camera_manager.get_property("Frames Per Second (Max)"),
            "exposure_time": camera_manager.get_property("Exposure Time (seconds)"),
            "timestamp": datetime.now().isoformat()
        }

        return performance_info
    except Exception as e:
        logger.error(f"Error getting performance info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance info: {str(e)}")


# Health Check Endpoint
@router.get("/health")
async def health_check():
    """Camera health check endpoint"""
    if not camera_connected:
        return {
            "status": "disconnected",
            "connected": False,
            "message": "Camera not connected"
        }

    try:
        # Quick health checks
        system_status = camera_manager.get_property("System Status")
        temperature = camera_manager.get_property("Temperature - Detector (Celsius)")

        is_healthy = (
                system_status not in ["Error", "Warning"] and
                temperature is not None and
                temperature < 0  # Detector should be cooled
        )

        return {
            "status": "healthy" if is_healthy else "warning",
            "connected": True,
            "camera_name": camera_name,
            "system_status": system_status,
            "detector_temperature": temperature,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "connected": camera_connected,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }