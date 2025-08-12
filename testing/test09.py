#!/usr/bin/env python3
"""
Enhanced PyScope DE Camera Properties Reader with PyScope Image Capture
Combines property reading capabilities with generalized PyScope image acquisition
Uses PyScope's unified camera interface that works with all camera types
"""

import sys
import os
import json
import time
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging to suppress DE camera property warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DE camera property warnings that are harmless
de_logger = logging.getLogger('DEClient')
de_logger.setLevel(logging.ERROR)  # Only show real errors


class UniversalPyScope_Camera:
    """
    Universal PyScope camera interface that works with all camera types
    Combines property reading with generalized image capture using PyScope's unified API
    """

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.properties_cache = {}
        self.de_module = None
        self.capture_count = 0
        self.camera_type = None  # 'DE', 'FEI', 'Tietz', 'TIA', etc.

    def connect(self, camera_name=None):
        """Connect to any PyScope-supported camera using unified interface"""
        try:
            import pyscope.registry

            # Determine camera to connect to
            target_camera = camera_name or self.camera_name

            # Try specific camera first, then attempt auto-detection
            camera_attempts = []
            if target_camera:
                camera_attempts.append(target_camera)

            # Add common camera names for auto-detection
            camera_attempts.extend([
                "DEApollo", "DE Apollo", "DE12", "DE20", "DE16",
                "Falcon3", "Falcon3EC", "Falcon4EC", "Ceta",
                "TietzF416", "TietzF816", "TietzF216",
                "TIA_Falcon3", "TIA_Ceta", "SimCCDCamera"
            ])

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

                    # Determine camera type
                    self.camera_type = self._determine_camera_type(attempt_name)
                    logger.info(f"  Camera type: {self.camera_type}")

                    self.connected = True
                    break

                except Exception as e:
                    logger.debug(f"  PyScope connection failed for {attempt_name}: {e}")
                    continue

            if not self.connected:
                logger.error("All PyScope camera connection attempts failed")
                return False

            # Initialize camera-specific features
            self._initialize_camera_features()

            return True

        except Exception as e:
            logger.error(f"Error in camera connection process: {e}")
            return False

    def _determine_camera_type(self, camera_name: str) -> str:
        """Determine the camera type from the name"""
        name_lower = camera_name.lower()
        if any(de_name in name_lower for de_name in ['de', 'apollo', 'direct electron']):
            return 'DE'
        elif any(fei_name in name_lower for fei_name in ['falcon', 'ceta']):
            return 'FEI'
        elif 'tietz' in name_lower:
            return 'Tietz'
        elif 'tia' in name_lower:
            return 'TIA'
        elif 'sim' in name_lower:
            return 'Simulation'
        else:
            return 'Unknown'

    def _initialize_camera_features(self):
        """Initialize camera-specific features and access methods"""
        try:
            if self.camera_type == 'DE':
                # For DE cameras, try to access internal DEAPI features
                self._initialize_de_features()
            elif self.camera_type == 'FEI':
                # For FEI cameras, initialize TIA features if available
                self._initialize_fei_features()
            elif self.camera_type == 'Tietz':
                # For Tietz cameras, initialize CAMC features
                self._initialize_tietz_features()

            logger.info(f"✓ Initialized {self.camera_type} camera features")

        except Exception as e:
            logger.debug(f"Could not initialize camera-specific features: {e}")

    def _initialize_de_features(self):
        """Initialize DE camera specific features"""
        try:
            # Try to access PyScope's DE module
            import pyscope
            pyscope_path = os.path.dirname(pyscope.__file__)

            possible_paths = [
                os.path.join(pyscope_path, 'de.py'),
                os.path.join(pyscope_path, 'instruments', 'de.py'),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("de", path)
                    self.de_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.de_module)
                    logger.debug("✓ Loaded PyScope DE module for enhanced property access")
                    break

        except Exception as e:
            logger.debug(f"Could not load DE module: {e}")

    def _initialize_fei_features(self):
        """Initialize FEI camera specific features"""
        # FEI cameras typically work well with standard PyScope interface
        logger.debug("FEI camera initialized with standard PyScope interface")

    def _initialize_tietz_features(self):
        """Initialize Tietz camera specific features"""
        # Tietz cameras typically work well with standard PyScope interface
        logger.debug("Tietz camera initialized with standard PyScope interface")

    def get_camera_model_name(self) -> str:
        """Get the camera model name"""
        try:
            if hasattr(self.camera, 'model_name'):
                return self.camera.model_name
            elif hasattr(self.camera, 'name'):
                return self.camera.name
            else:
                return self.camera_name
        except:
            return self.camera_name or "Unknown"

    def setup_camera_for_acquisition(self, exposure_time_ms: float = 1000,
                                     binning: Dict[str, int] = None,
                                     dimension: Dict[str, int] = None,
                                     offset: Dict[str, int] = None):
        """Configure camera for image acquisition using PyScope's unified interface"""
        try:
            logger.info("Configuring camera for image acquisition...")

            settings_applied = 0
            total_settings = 0

            # Set exposure time
            total_settings += 1
            try:
                self.camera.setExposureTime(exposure_time_ms)
                current_exposure = self.camera.getExposureTime()
                logger.debug(f"✓ Set exposure time to {exposure_time_ms} ms (current: {current_exposure} ms)")
                settings_applied += 1
            except Exception as e:
                logger.warning(f"Could not set exposure time: {e}")

            # Set binning if specified
            if binning:
                total_settings += 1
                try:
                    self.camera.setBinning(binning)
                    current_binning = self.camera.getBinning()
                    logger.debug(f"✓ Set binning to {binning} (current: {current_binning})")
                    settings_applied += 1
                except Exception as e:
                    logger.warning(f"Could not set binning: {e}")

            # Set dimension if specified
            if dimension:
                total_settings += 1
                try:
                    self.camera.setDimension(dimension)
                    current_dimension = self.camera.getDimension()
                    logger.debug(f"✓ Set dimension to {dimension} (current: {current_dimension})")
                    settings_applied += 1
                except Exception as e:
                    logger.warning(f"Could not set dimension: {e}")

            # Set offset if specified
            if offset:
                total_settings += 1
                try:
                    self.camera.setOffset(offset)
                    current_offset = self.camera.getOffset()
                    logger.debug(f"✓ Set offset to {offset} (current: {current_offset})")
                    settings_applied += 1
                except Exception as e:
                    logger.warning(f"Could not set offset: {e}")

            # Camera-specific settings
            if self.camera_type == 'DE':
                self._setup_de_specific_settings()
            elif self.camera_type == 'FEI':
                self._setup_fei_specific_settings()

            logger.info(f"Camera configuration: {settings_applied}/{total_settings} settings applied successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
            return False

    def _setup_de_specific_settings(self):
        """Apply DE camera specific settings"""
        try:
            # Try to disable autosave for quick capture
            if hasattr(self.camera, 'setProperty'):
                try:
                    self.camera.setProperty("Autosave Final Image", "Off")
                    self.camera.setProperty("Autosave Movie", "Off")
                    logger.debug("✓ Disabled DE autosave for quick capture")
                except:
                    pass
        except Exception as e:
            logger.debug(f"Could not apply DE specific settings: {e}")

    def _setup_fei_specific_settings(self):
        """Apply FEI camera specific settings"""
        try:
            # FEI cameras may have specific frame settings
            if hasattr(self.camera, 'setSaveRawFrames'):
                try:
                    self.camera.setSaveRawFrames(False)  # Don't save raw frames for simple capture
                    logger.debug("✓ Disabled FEI raw frame saving for quick capture")
                except:
                    pass
        except Exception as e:
            logger.debug(f"Could not apply FEI specific settings: {e}")

    def capture_image_pyscope(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Capture a single image using PyScope's unified getImage() method"""
        if not self.camera:
            logger.error("No PyScope camera available for image capture")
            return None, None

        try:
            logger.info("Starting PyScope image acquisition...")

            # Capture image using PyScope's universal interface
            start_time = time.time()
            image = self.camera.getImage()
            capture_time = time.time() - start_time

            if image is not None:
                # Create metadata dictionary
                attr_dict = {
                    'camera_name': self.camera_name,
                    'camera_type': self.camera_type,
                    'dataset': f"pyscope_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'shape': image.shape,
                    'pixel_format': str(image.dtype),
                    'min_value': float(image.min()),
                    'max_value': float(image.max()),
                    'mean_value': float(image.mean()),
                    'std_value': float(image.std()),
                    'capture_method': 'PyScope_Universal',
                    'capture_time_seconds': capture_time,
                    'exposure_time_ms': self._get_current_exposure_time(),
                    'binning': self._get_current_binning(),
                    'dimension': self._get_current_dimension(),
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"✓ PyScope image captured successfully!")
                logger.info(f"  Camera: {self.camera_name} ({self.camera_type})")
                logger.info(f"  Image size: {image.shape}")
                logger.info(f"  Data type: {image.dtype}")
                logger.info(f"  Value range: [{image.min():.1f}, {image.max():.1f}]")
                logger.info(f"  Mean ± Std: {image.mean():.1f} ± {image.std():.1f}")
                logger.info(f"  Capture time: {capture_time:.3f} seconds")

                return image, attr_dict
            else:
                logger.error("PyScope returned None for image")
                return None, None

        except Exception as e:
            logger.error(f"Error during PyScope image capture: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _get_current_exposure_time(self) -> float:
        """Get current exposure time safely"""
        try:
            return self.camera.getExposureTime()
        except:
            return -1.0

    def _get_current_binning(self) -> Dict[str, int]:
        """Get current binning safely"""
        try:
            return self.camera.getBinning()
        except:
            return {'x': 1, 'y': 1}

    def _get_current_dimension(self) -> Dict[str, int]:
        """Get current dimension safely"""
        try:
            return self.camera.getDimension()
        except:
            try:
                return self.camera.getCameraSize()
            except:
                return {'x': -1, 'y': -1}

    def capture_image(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Capture an image using the best available method"""
        if not self.connected:
            logger.error("Camera not connected")
            return None, None

        # Use PyScope's universal interface for all camera types
        image, metadata = self.capture_image_pyscope()
        if image is not None:
            self.capture_count += 1
            metadata['capture_number'] = self.capture_count

        return image, metadata

    def get_available_properties(self) -> Optional[List[str]]:
        """Get list of all available properties using camera-specific methods"""
        if not self.connected:
            logger.error("Camera not connected")
            return None

        try:
            # Method 1: Try camera-specific property listing
            if self.camera_type == 'DE' and hasattr(self.camera, 'getPropertiesList'):
                try:
                    self.camera.getPropertiesList()
                    if hasattr(self.camera, 'properties_list'):
                        properties = self.camera.properties_list
                        logger.info(f"Found {len(properties)} properties via DE camera properties_list")
                        return properties
                except Exception as e:
                    logger.debug(f"Camera getPropertiesList failed: {e}")

            # Method 2: Try standard PyScope property methods
            if hasattr(self.camera, 'listProperties'):
                try:
                    properties = self.camera.listProperties()
                    logger.info(f"Found {len(properties)} properties via listProperties")
                    return properties
                except Exception as e:
                    logger.debug(f"listProperties failed: {e}")

            # Method 3: Use common properties based on camera type
            common_properties = [
                'Exposure Time (seconds)', 'Camera Model', 'Camera Name',
                'Sensor Size X (pixels)', 'Sensor Size Y (pixels)',
                'Image Size X (pixels)', 'Image Size Y (pixels)',
                'Binning X', 'Binning Y'
            ]

            # Add camera-specific properties
            if self.camera_type == 'DE':
                common_properties.extend([
                    'Frames Per Second', 'Temperature - Detector (Celsius)',
                    'Camera SN', 'Server Software Version', 'System Status',
                    'Autosave Final Image', 'Autosave Movie', 'Image Processing - Mode'
                ])
            elif self.camera_type == 'FEI':
                common_properties.extend([
                    'Frame Time', 'Number of Frames', 'Save Raw Frames'
                ])
            elif self.camera_type == 'Tietz':
                common_properties.extend([
                    'Camera Temperature', 'Gain', 'Speed'
                ])

            logger.info(f"Using {len(common_properties)} common properties for {self.camera_type} camera")
            return common_properties

        except Exception as e:
            logger.error(f"Error getting available properties: {e}")
            return None

    def get_property_value(self, property_name: str) -> Any:
        """Get the value of a specific property using multiple access methods"""
        if not self.connected:
            return None

        try:
            # Method 1: Try camera's getProperty method (works for DE and some others)
            if hasattr(self.camera, 'getProperty'):
                try:
                    return self.camera.getProperty(property_name)
                except Exception as e:
                    logger.debug(f"Camera getProperty failed for '{property_name}': {e}")

            # Method 2: Try PyScope standard methods for common properties
            property_methods = {
                'Exposure Time (seconds)': lambda: self.camera.getExposureTime() / 1000.0,
                'Camera Model': lambda: self.get_camera_model_name(),
                'Camera Name': lambda: self.camera_name,
                'Sensor Size X (pixels)': lambda: self.camera.getCameraSize()['x'],
                'Sensor Size Y (pixels)': lambda: self.camera.getCameraSize()['y'],
                'Image Size X (pixels)': lambda: self.camera.getDimension().get('x', self.camera.getCameraSize()['x']),
                'Image Size Y (pixels)': lambda: self.camera.getDimension().get('y', self.camera.getCameraSize()['y']),
                'Binning X': lambda: self.camera.getBinning()['x'],
                'Binning Y': lambda: self.camera.getBinning()['y']
            }

            if property_name in property_methods:
                try:
                    return property_methods[property_name]()
                except Exception as e:
                    logger.debug(f"Standard method failed for '{property_name}': {e}")

            # Method 3: Try camera-specific property access
            if self.camera_type == 'FEI' and property_name in ['Frame Time', 'Number of Frames']:
                try:
                    if property_name == 'Frame Time' and hasattr(self.camera, 'getFrameTime'):
                        return self.camera.getFrameTime()
                    elif property_name == 'Number of Frames' and hasattr(self.camera, 'getNumberOfFrames'):
                        return self.camera.getNumberOfFrames()
                except Exception as e:
                    logger.debug(f"FEI specific method failed for '{property_name}': {e}")

            return None

        except Exception as e:
            logger.debug(f"Error getting property '{property_name}': {e}")
            return None

    def get_all_properties(self) -> Dict[str, Any]:
        """Get all camera properties with comprehensive error handling"""
        if not self.connected:
            logger.error("Camera not connected")
            return {}

        logger.info("Starting comprehensive property read...")
        start_time = time.time()

        # Get list of all properties
        property_names = self.get_available_properties()
        if not property_names:
            logger.error("Could not get property list")
            return {}

        logger.info(f"Reading values for {len(property_names)} properties...")

        # Read all properties
        all_properties = {}
        failed_properties = []
        successful_count = 0

        for i, prop_name in enumerate(property_names):
            try:
                value = self.get_property_value(prop_name)
                if value is not None:
                    all_properties[prop_name] = value
                    successful_count += 1
                else:
                    failed_properties.append(prop_name)

                # Progress indicator for large property lists
                if len(property_names) > 50 and (i + 1) % 25 == 0:
                    logger.info(f"Progress: {i + 1}/{len(property_names)} properties processed")

            except Exception as e:
                logger.warning(f"Failed to get property '{prop_name}': {str(e)}")
                failed_properties.append(prop_name)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"✓ Successfully retrieved {successful_count} properties")
        if failed_properties:
            logger.warning(f"✗ Failed to retrieve {len(failed_properties)} properties")
        logger.info(f"Total time: {total_time:.2f} seconds")

        # Determine access method based on results and camera type
        if successful_count >= 100:
            access_method = f"Full-{self.camera_type}"
        elif successful_count >= 20:
            access_method = f"Standard-{self.camera_type}"
        else:
            access_method = f"Limited-{self.camera_type}"

        # Store results
        self.properties_cache = {
            "properties": all_properties,
            "total_count": len(all_properties),
            "failed_count": len(failed_properties),
            "failed_properties": failed_properties,
            "camera_name": self.camera_name,
            "camera_type": self.camera_type,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "average_time_ms": total_time / len(property_names) * 1000 if property_names else 0,
            "access_method": access_method
        }

        return self.properties_cache

    def convert_and_save_image(self, image: np.ndarray, metadata: Dict[str, Any],
                               output_format: str = "PNG", output_dir: str = "captured_images") -> Tuple[
        Optional[str], Optional[Image.Image]]:
        """Convert numpy array to PIL Image and save as JPG/PNG"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Normalize the image data to 0-255 range for display
            if image.dtype == np.float32 or image.dtype == np.float64:
                # For float images, normalize to 0-1 range first
                image_normalized = (image - image.min()) / (image.max() - image.min())
                image_8bit = (image_normalized * 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                # For 16-bit images, scale down to 8-bit
                image_8bit = (image / 256).astype(np.uint8)
            elif image.dtype == np.uint8:
                image_8bit = image
            else:
                # Generic conversion
                image_normalized = (image - image.min()) / (image.max() - image.min())
                image_8bit = (image_normalized * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image_8bit, mode='L')  # 'L' for grayscale

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = metadata.get('dataset', 'unknown')
            camera_type = metadata.get('camera_type', 'unknown')
            filename = f"universal_{self.camera_name}_{camera_type}_{dataset_name}_{timestamp}.{output_format.lower()}"
            filepath = os.path.join(output_dir, filename)

            # Save the image
            if output_format.upper() == "JPG" or output_format.upper() == "JPEG":
                pil_image.save(filepath, "JPEG", quality=95)
            else:  # PNG
                pil_image.save(filepath, "PNG")

            logger.info(f"✓ Image saved as: {filepath}")
            return filepath, pil_image

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None, None

    def display_image(self, pil_image: Image.Image, metadata: Dict[str, Any] = None):
        """Display the image using matplotlib"""
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(pil_image, cmap='gray')
            plt.colorbar(label='Intensity')

            if metadata:
                title = f"Universal PyScope Image\n"
                title += f"Camera: {metadata.get('camera_name', 'N/A')} ({metadata.get('camera_type', 'N/A')})\n"
                title += f"Method: {metadata.get('capture_method', 'N/A')}\n"
                title += f"Size: {metadata.get('shape', 'N/A')}\n"
                title += f"Exposure: {metadata.get('exposure_time_ms', 'N/A')} ms\n"
                title += f"Min/Max: {metadata.get('min_value', 'N/A'):.1f}/{metadata.get('max_value', 'N/A'):.1f}"
                plt.title(title)
            else:
                plt.title(f"Universal PyScope Image - {self.camera_name}")

            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def save_properties_to_file(self, filename: str = None, format: str = "json"):
        """Save all properties to a file"""
        if not self.properties_cache:
            logger.warning("No properties cached. Call get_all_properties() first.")
            return False

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            access_method = self.properties_cache.get("access_method", "unknown")
            filename = f"universal_pyscope_properties_{self.camera_name}_{access_method}_{timestamp}.{format}"

        try:
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(self.properties_cache, f, indent=2, default=str)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            logger.info(f"✓ Properties saved to: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving properties to file: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera"""
        try:
            if self.camera and hasattr(self.camera, 'disconnect'):
                self.camera.disconnect()
                logger.info(f"Disconnected from {self.camera_name}")

            self.connected = False

        except Exception as e:
            logger.debug(f"Error during disconnect: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def main():
    """Main function to test the universal PyScope camera interface"""
    print("=== Universal PyScope Camera Properties Reader with Image Capture ===")

    # Test PyScope imports
    try:
        import pyscope.registry
        import pyscope.config
        logger.info("✓ PyScope imports successful")
    except ImportError as e:
        logger.error(f"PyScope import failed: {e}")
        return

    # List available cameras
    try:
        available_cameras = []
        common_camera_names = [
            "DEApollo", "DE12", "DE20", "DE16",
            "Falcon3", "Falcon3EC", "Falcon4EC", "Ceta",
            "TietzF416", "TietzF816", "TietzF216",
            "TIA_Falcon3", "TIA_Ceta", "SimCCDCamera"
        ]

        for camera_name in common_camera_names:
            try:
                camera_class = pyscope.registry.getClass(camera_name)
                available_cameras.append(camera_name)
            except:
                pass

        if available_cameras:
            logger.info(f"Available cameras: {', '.join(available_cameras)}")
        else:
            logger.warning("No cameras found in registry")

    except Exception as e:
        logger.warning(f"Could not list available cameras: {e}")

    # Create and connect to camera using context manager
    with UniversalPyScope_Camera() as camera:
        if not camera.connect():
            logger.error("Failed to connect to any camera")
            return

        try:
            print(f"\n🔍 Reading all properties from {camera.camera_name} ({camera.camera_type})...")

            # Get all properties
            all_properties = camera.get_all_properties()

            if not all_properties:
                logger.error("Failed to read properties")
                return

            # Display summary
            print(f"\n📊 Properties Summary:")
            print(f"  Camera: {camera.camera_name} ({camera.camera_type})")
            print(f"  Total properties: {all_properties['total_count']}")
            print(f"  Failed properties: {all_properties['failed_count']}")
            print(f"  Total time: {all_properties['total_time_seconds']:.2f} seconds")
            print(f"  Access method: {all_properties['access_method']}")

            # Show some sample properties
            print(f"\n🔍 Sample Properties:")
            sample_props = [
                "Camera Model", "Camera Name", "Sensor Size X (pixels)", "Sensor Size Y (pixels)",
                "Exposure Time (seconds)", "Binning X", "Binning Y"
            ]

            properties = all_properties["properties"]
            for prop in sample_props:
                if prop in properties:
                    print(f"  {prop}: {properties[prop]}")

            # Save properties to file
            print(f"\n💾 Saving properties to file...")
            if camera.save_properties_to_file():
                print("✓ Properties saved successfully")

            # Now test image capture functionality
            print(f"\n📸 Testing Universal Image Capture...")

            # Setup camera for acquisition
            exposure_time_ms = 1000  # 1 second exposure
            binning = {'x': 1, 'y': 1}  # No binning

            if camera.setup_camera_for_acquisition(
                    exposure_time_ms=exposure_time_ms,
                    binning=binning
            ):
                print(f"✓ Camera configured for {exposure_time_ms} ms exposure")

                # Capture image using PyScope's universal interface
                image, metadata = camera.capture_image()

                if image is not None:
                    print(f"\n🎉 Universal image capture successful!")
                    print(f"  Camera: {metadata['camera_name']} ({metadata['camera_type']})")
                    print(f"  Method: {metadata['capture_method']}")
                    print(f"  Image shape: {metadata['shape']}")
                    print(f"  Data type: {metadata['pixel_format']}")
                    print(f"  Exposure time: {metadata['exposure_time_ms']} ms")
                    print(f"  Binning: {metadata['binning']}")
                    print(f"  Value range: [{metadata['min_value']:.1f}, {metadata['max_value']:.1f}]")
                    print(f"  Mean ± Std: {metadata['mean_value']:.1f} ± {metadata['std_value']:.1f}")
                    print(f"  Capture time: {metadata['capture_time_seconds']:.3f} seconds")
                    print(f"  Capture number: {metadata['capture_number']}")

                    # Save in both formats
                    print(f"\n💾 Saving captured image...")
                    for fmt in ["PNG", "JPG"]:
                        filepath, pil_image = camera.convert_and_save_image(image, metadata, fmt)
                        if filepath:
                            print(f"✓ Saved {fmt} format: {os.path.basename(filepath)}")

                    # Display the image
                    if pil_image:
                        print(f"\n📊 Displaying image...")
                        camera.display_image(pil_image, metadata)

                    print(f"\n🎉 Universal PyScope image capture completed successfully!")
                    print(f"This demonstrates PyScope's unified interface working with {camera.camera_type} cameras!")

                else:
                    print(f"\n❌ Image capture failed")
            else:
                print(f"\n❌ Camera configuration failed")

            # Show failed properties if any (first 10)
            if all_properties['failed_properties']:
                print(f"\n⚠️  Failed to read {len(all_properties['failed_properties'])} properties:")
                for prop in all_properties['failed_properties'][:10]:
                    print(f"    - {prop}")
                if len(all_properties['failed_properties']) > 10:
                    print(f"    ... and {len(all_properties['failed_properties']) - 10} more")

            print(f"\n🎉 Universal PyScope camera interface testing completed!")
            print(f"Camera type: {camera.camera_type}")
            print(f"Properties accessed: {all_properties['total_count']}")
            print(f"Images captured: {camera.capture_count}")

        except Exception as e:
            logger.error(f"Error in main process: {e}")
            import traceback
            traceback.print_exc()


def demo_multiple_cameras():
    """Demonstrate the universal interface with multiple camera types"""
    print("\n" + "=" * 60)
    print("=== Multi-Camera Demo (if available) ===")

    try:
        import pyscope.registry

        # Try different camera types
        test_cameras = ["DEApollo", "Falcon3", "TietzF416", "SimCCDCamera"]

        for camera_name in test_cameras:
            try:
                print(f"\n--- Testing {camera_name} ---")

                with UniversalPyScope_Camera() as camera:
                    if camera.connect(camera_name):
                        print(f"✓ Connected to {camera.camera_name} ({camera.camera_type})")

                        # Quick property test
                        properties = camera.get_all_properties()
                        print(f"  Properties: {properties.get('total_count', 0)}")

                        # Quick image test
                        if camera.setup_camera_for_acquisition(exposure_time_ms=500):
                            image, metadata = camera.capture_image()
                            if image is not None:
                                print(f"  ✓ Image captured: {image.shape} {image.dtype}")
                            else:
                                print(f"  ❌ Image capture failed")
                        else:
                            print(f"  ❌ Camera setup failed")
                    else:
                        print(f"  ❌ Connection failed")

            except Exception as e:
                print(f"  ❌ Error with {camera_name}: {e}")

    except Exception as e:
        print(f"Multi-camera demo failed: {e}")


if __name__ == "__main__":
    main()

    # Uncomment to test multiple cameras
    # demo_multiple_cameras()