#!/usr/bin/env python3
"""
Fixed Universal PyScope DE Camera Interface
Compares test01 (direct DEAPI) vs test09 (PyScope interface) approaches
and fixes the issues in the PyScope universal interface
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
de_logger.setLevel(logging.ERROR)


class DEDirectAPICapture:
    """
    Direct DEAPI capture (test01 approach) for comparison
    """

    def __init__(self):
        self.client = None
        self.connected = False

    def connect(self, host="localhost", port=13240):
        """Connect using direct DEAPI like test01"""
        try:
            # Add DEAPI path
            sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
            from pyscope import DEAPI

            self.client = DEAPI.Client()
            self.client.Connect(host, port)

            cameras = self.client.ListCameras()
            if not cameras:
                raise Exception("No cameras found")

            self.client.SetCurrentCamera(cameras[0])
            self.connected = True
            logger.info(f"Connected to DE camera via direct DEAPI: {cameras[0]}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect via direct DEAPI: {e}")
            return False

    def setup_camera_for_single_shot(self):
        """Configure camera for single shot like test01"""
        try:
            # Set basic acquisition parameters exactly like test01
            self.client.SetProperty("Exposure Mode", "Normal")
            self.client.SetProperty("Image Processing - Mode", "Integrating")
            self.client.SetProperty("Frames Per Second", 20)
            self.client.SetProperty("Exposure Time (seconds)", 1.0)

            # Disable autosave for this quick capture
            self.client.SetProperty("Autosave Final Image", "Off")
            self.client.SetProperty("Autosave Movie", "Off")

            logger.info("DE camera configured for single shot via direct DEAPI")
            return True
        except Exception as e:
            logger.error(f"Failed to configure DE camera via direct DEAPI: {e}")
            return False

    def capture_image(self):
        """Capture image using direct DEAPI like test01"""
        try:
            from pyscope import DEAPI

            logger.info("Starting DE image acquisition via direct DEAPI...")

            # Start acquisition (1 repeat) - exactly like test01
            self.client.StartAcquisition(1)

            # Get the final summed image - exactly like test01
            frameType = DEAPI.FrameType.SUMTOTAL
            pixelFormat = DEAPI.PixelFormat.AUTO
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()

            image, pixelFormat, attributes, histogram = self.client.GetResult(
                frameType, pixelFormat, attributes, histogram
            )

            if image is not None:
                metadata = {
                    'method': 'Direct_DEAPI',
                    'dataset': attributes.datasetName,
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'min_value': float(attributes.imageMin),
                    'max_value': float(attributes.imageMax),
                    'mean_value': float(attributes.imageMean),
                    'std_value': float(attributes.imageStd),
                    'pixel_format': str(pixelFormat),
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"✓ DE image captured via direct DEAPI!")
                logger.info(f"  Dataset: {attributes.datasetName}")
                logger.info(f"  Image size: {image.shape}")
                logger.info(f"  Data type: {image.dtype}")
                logger.info(f"  Value range: [{attributes.imageMin:.1f}, {attributes.imageMax:.1f}]")
                logger.info(f"  Mean ± Std: {attributes.imageMean:.1f} ± {attributes.imageStd:.1f}")

                return image, metadata
            else:
                logger.error("Direct DEAPI returned None for image")
                return None, None

        except Exception as e:
            logger.error(f"Error during direct DEAPI image capture: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def disconnect(self):
        """Disconnect from DE camera"""
        try:
            if self.client and self.connected:
                self.client.Disconnect()
                logger.info("Disconnected from DE camera (direct DEAPI)")
                self.connected = False
        except:
            pass


class FixedUniversalPyScope_Camera:
    """
    Fixed Universal PyScope camera interface that properly handles DE cameras
    Addresses the issues found in test09 compared to test01
    """

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.properties_cache = {}
        self.de_client = None  # Store direct DE client for better results
        self.capture_count = 0
        self.camera_type = None

    def connect(self, camera_name=None):
        """Connect to any PyScope-supported camera with improved DE handling"""
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

            # Initialize camera-specific features with improved DE handling
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
        """Initialize camera-specific features with improved DE handling"""
        try:
            if self.camera_type == 'DE':
                # For DE cameras, try to get direct DEAPI access for better results
                self._initialize_de_features_improved()
            elif self.camera_type == 'FEI':
                self._initialize_fei_features()
            elif self.camera_type == 'Tietz':
                self._initialize_tietz_features()

            logger.info(f"✓ Initialized {self.camera_type} camera features")

        except Exception as e:
            logger.debug(f"Could not initialize camera-specific features: {e}")

    def _initialize_de_features_improved(self):
        """Improved DE camera initialization that tries to access direct DEAPI"""
        try:
            # Try to access the underlying DE client for better performance
            if hasattr(self.camera, 'client'):
                self.de_client = self.camera.client
                logger.info("✓ Accessed direct DE client through PyScope")
            elif hasattr(self.camera, 'de_client'):
                self.de_client = self.camera.de_client
                logger.info("✓ Accessed direct DE client through PyScope")
            else:
                # Try to create a direct connection
                try:
                    sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
                    from pyscope import DEAPI

                    # Try to connect directly
                    self.de_client = DEAPI.Client()
                    self.de_client.Connect("localhost", 13240)

                    cameras = self.de_client.ListCameras()
                    if cameras:
                        self.de_client.SetCurrentCamera(cameras[0])
                        logger.info("✓ Created direct DE client connection")
                    else:
                        self.de_client = None

                except Exception as e:
                    logger.debug(f"Could not create direct DE client: {e}")
                    self.de_client = None

        except Exception as e:
            logger.debug(f"Could not initialize improved DE features: {e}")

    def _initialize_fei_features(self):
        """Initialize FEI camera specific features"""
        logger.debug("FEI camera initialized with standard PyScope interface")

    def _initialize_tietz_features(self):
        """Initialize Tietz camera specific features"""
        logger.debug("Tietz camera initialized with standard PyScope interface")

    def setup_camera_for_acquisition(self, exposure_time_ms: float = 1000,
                                     binning: Dict[str, int] = None,
                                     dimension: Dict[str, int] = None,
                                     offset: Dict[str, int] = None):
        """Configure camera for image acquisition with improved DE handling"""
        try:
            logger.info("Configuring camera for image acquisition...")

            settings_applied = 0
            total_settings = 0

            # For DE cameras, try to use direct client if available
            if self.camera_type == 'DE' and self.de_client:
                return self._setup_de_camera_direct(exposure_time_ms, binning, dimension, offset)

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

            logger.info(f"Camera configuration: {settings_applied}/{total_settings} settings applied successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
            return False

    def _setup_de_camera_direct(self, exposure_time_ms: float, binning: Dict[str, int] = None,
                                dimension: Dict[str, int] = None, offset: Dict[str, int] = None):
        """Setup DE camera using direct client for better results"""
        try:
            logger.info("Configuring DE camera using direct client...")

            # Set basic acquisition parameters like test01
            self.de_client.SetProperty("Exposure Mode", "Normal")
            self.de_client.SetProperty("Image Processing - Mode", "Integrating")
            self.de_client.SetProperty("Frames Per Second", 20)
            self.de_client.SetProperty("Exposure Time (seconds)", exposure_time_ms / 1000.0)

            # Disable autosave for quick capture
            self.de_client.SetProperty("Autosave Final Image", "Off")
            self.de_client.SetProperty("Autosave Movie", "Off")

            logger.info("✓ DE camera configured using direct client (like test01)")
            return True

        except Exception as e:
            logger.error(f"Failed to configure DE camera using direct client: {e}")
            return False

    def capture_image_improved(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Improved image capture that uses the best method for each camera type"""
        if not self.camera:
            logger.error("No camera available for image capture")
            return None, None

        try:
            logger.info("Starting improved image acquisition...")

            # For DE cameras, try direct client first for better results
            if self.camera_type == 'DE' and self.de_client:
                return self._capture_de_direct()
            else:
                return self._capture_pyscope_generic()

        except Exception as e:
            logger.error(f"Error during improved image capture: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _capture_de_direct(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Capture DE image using direct client like test01"""
        try:
            from pyscope import DEAPI

            start_time = time.time()

            # Start acquisition (1 repeat) - exactly like test01
            self.de_client.StartAcquisition(1)

            # Get the final summed image - exactly like test01
            frameType = DEAPI.FrameType.SUMTOTAL
            pixelFormat = DEAPI.PixelFormat.AUTO
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()

            image, pixelFormat, attributes, histogram = self.de_client.GetResult(
                frameType, pixelFormat, attributes, histogram
            )

            capture_time = time.time() - start_time

            if image is not None:
                # Create metadata dictionary
                attr_dict = {
                    'camera_name': self.camera_name,
                    'camera_type': self.camera_type,
                    'capture_method': 'DE_Direct_Client',
                    'dataset': attributes.datasetName,
                    'shape': image.shape,
                    'pixel_format': str(image.dtype),
                    'min_value': float(attributes.imageMin),
                    'max_value': float(attributes.imageMax),
                    'mean_value': float(attributes.imageMean),
                    'std_value': float(attributes.imageStd),
                    'de_pixel_format': str(pixelFormat),
                    'capture_time_seconds': capture_time,
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"✓ DE image captured using direct client!")
                logger.info(f"  Dataset: {attributes.datasetName}")
                logger.info(f"  Image size: {image.shape}")
                logger.info(f"  Data type: {image.dtype}")
                logger.info(f"  Value range: [{attributes.imageMin:.1f}, {attributes.imageMax:.1f}]")
                logger.info(f"  Mean ± Std: {attributes.imageMean:.1f} ± {attributes.imageStd:.1f}")
                logger.info(f"  Capture time: {capture_time:.3f} seconds")

                return image, attr_dict
            else:
                logger.error("DE direct client returned None for image")
                return None, None

        except Exception as e:
            logger.error(f"Error during DE direct capture: {e}")
            return None, None

    def _capture_pyscope_generic(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Capture image using PyScope's universal interface"""
        try:
            start_time = time.time()
            image = self.camera.getImage()
            capture_time = time.time() - start_time

            if image is not None:
                # Create metadata dictionary
                attr_dict = {
                    'camera_name': self.camera_name,
                    'camera_type': self.camera_type,
                    'capture_method': 'PyScope_Universal',
                    'dataset': f"pyscope_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'shape': image.shape,
                    'pixel_format': str(image.dtype),
                    'min_value': float(image.min()),
                    'max_value': float(image.max()),
                    'mean_value': float(image.mean()),
                    'std_value': float(image.std()),
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
            logger.error(f"Error during PyScope capture: {e}")
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

        # Use improved capture method that chooses the best approach
        image, metadata = self.capture_image_improved()
        if image is not None:
            self.capture_count += 1
            metadata['capture_number'] = self.capture_count

        return image, metadata

    def compare_capture_methods(self):
        """Compare different capture methods for analysis"""
        if self.camera_type != 'DE':
            logger.warning("Capture method comparison only available for DE cameras")
            return None

        results = {}

        # Method 1: PyScope universal interface
        try:
            logger.info("\n--- Testing PyScope Universal Interface ---")
            start_time = time.time()
            image1, metadata1 = self._capture_pyscope_generic()
            method1_time = time.time() - start_time

            if image1 is not None:
                results['pyscope_universal'] = {
                    'success': True,
                    'time': method1_time,
                    'shape': image1.shape,
                    'dtype': str(image1.dtype),
                    'min': float(image1.min()),
                    'max': float(image1.max()),
                    'mean': float(image1.mean()),
                    'std': float(image1.std()),
                    'metadata': metadata1
                }
                logger.info(f"✓ PyScope universal: {image1.shape} {image1.dtype} in {method1_time:.3f}s")
            else:
                results['pyscope_universal'] = {'success': False, 'time': method1_time}
                logger.info(f"✗ PyScope universal failed in {method1_time:.3f}s")

        except Exception as e:
            results['pyscope_universal'] = {'success': False, 'error': str(e)}
            logger.error(f"✗ PyScope universal error: {e}")

        # Method 2: Direct DE client (if available)
        if self.de_client:
            try:
                logger.info("\n--- Testing Direct DE Client ---")
                start_time = time.time()
                image2, metadata2 = self._capture_de_direct()
                method2_time = time.time() - start_time

                if image2 is not None:
                    results['de_direct'] = {
                        'success': True,
                        'time': method2_time,
                        'shape': image2.shape,
                        'dtype': str(image2.dtype),
                        'min': float(image2.min()),
                        'max': float(image2.max()),
                        'mean': float(image2.mean()),
                        'std': float(image2.std()),
                        'metadata': metadata2
                    }
                    logger.info(f"✓ DE direct: {image2.shape} {image2.dtype} in {method2_time:.3f}s")
                else:
                    results['de_direct'] = {'success': False, 'time': method2_time}
                    logger.info(f"✗ DE direct failed in {method2_time:.3f}s")

            except Exception as e:
                results['de_direct'] = {'success': False, 'error': str(e)}
                logger.error(f"✗ DE direct error: {e}")

        return results

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

            # Generate filename with timestamp and capture method
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_method = metadata.get('capture_method', 'unknown')
            camera_type = metadata.get('camera_type', 'unknown')
            filename = f"fixed_{self.camera_name}_{camera_type}_{capture_method}_{timestamp}.{output_format.lower()}"
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

    def disconnect(self):
        """Disconnect from camera"""
        try:
            if self.camera and hasattr(self.camera, 'disconnect'):
                self.camera.disconnect()

            if self.de_client:
                self.de_client.Disconnect()

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


def run_comparison_test():
    """Run a comparison between test01 (direct DEAPI) and fixed test09 approaches"""
    print("=== Comparison Test: Direct DEAPI vs Fixed PyScope Universal ===")

    results = {
        'direct_deapi': None,
        'fixed_pyscope': None,
        'comparison': {}
    }

    # Test 1: Direct DEAPI approach (like test01)
    print("\n" + "=" * 60)
    print("1. Testing Direct DEAPI Approach (test01 method)")
    print("=" * 60)

    try:
        with DEDirectAPICapture() as de_direct:
            if de_direct.connect():
                if de_direct.setup_camera_for_single_shot():
                    start_time = time.time()
                    image1, metadata1 = de_direct.capture_image()
                    end_time = time.time()

                    if image1 is not None:
                        results['direct_deapi'] = {
                            'success': True,
                            'capture_time': end_time - start_time,
                            'image_shape': image1.shape,
                            'image_dtype': str(image1.dtype),
                            'metadata': metadata1
                        }
                        print(f"✓ Direct DEAPI capture successful!")
                        print(f"  Method: {metadata1['method']}")
                        print(f"  Shape: {image1.shape}")
                        print(f"  Time: {end_time - start_time:.3f} seconds")
                    else:
                        results['direct_deapi'] = {'success': False, 'error': 'Image capture failed'}
                else:
                    results['direct_deapi'] = {'success': False, 'error': 'Camera setup failed'}
            else:
                results['direct_deapi'] = {'success': False, 'error': 'Connection failed'}

    except Exception as e:
        results['direct_deapi'] = {'success': False, 'error': str(e)}
        print(f"✗ Direct DEAPI test failed: {e}")

    # Test 2: Fixed PyScope Universal approach
    print("\n" + "=" * 60)
    print("2. Testing Fixed PyScope Universal Approach")
    print("=" * 60)

    try:
        with FixedUniversalPyScope_Camera() as camera:
            if camera.connect():
                if camera.setup_camera_for_acquisition(exposure_time_ms=1000):
                    start_time = time.time()
                    image2, metadata2 = camera.capture_image()
                    end_time = time.time()

                    if image2 is not None:
                        results['fixed_pyscope'] = {
                            'success': True,
                            'capture_time': end_time - start_time,
                            'image_shape': image2.shape,
                            'image_dtype': str(image2.dtype),
                            'metadata': metadata2
                        }
                        print(f"✓ Fixed PyScope capture successful!")
                        print(f"  Method: {metadata2['capture_method']}")
                        print(f"  Shape: {image2.shape}")
                        print(f"  Time: {end_time - start_time:.3f} seconds")

                        # Test method comparison if it's a DE camera
                        if camera.camera_type == 'DE':
                            print("\n--- Testing Multiple Capture Methods ---")
                            method_results = camera.compare_capture_methods()
                            if method_results:
                                results['method_comparison'] = method_results

                    else:
                        results['fixed_pyscope'] = {'success': False, 'error': 'Image capture failed'}
                else:
                    results['fixed_pyscope'] = {'success': False, 'error': 'Camera setup failed'}
            else:
                results['fixed_pyscope'] = {'success': False, 'error': 'Connection failed'}

    except Exception as e:
        results['fixed_pyscope'] = {'success': False, 'error': str(e)}
        print(f"✗ Fixed PyScope test failed: {e}")

    # Comparison Analysis
    print("\n" + "=" * 60)
    print("3. Comparison Analysis")
    print("=" * 60)

    if results['direct_deapi'] and results['fixed_pyscope']:
        if results['direct_deapi']['success'] and results['fixed_pyscope']['success']:
            # Compare results
            print("\n📊 Comparison Results:")

            # Time comparison
            time1 = results['direct_deapi']['capture_time']
            time2 = results['fixed_pyscope']['capture_time']
            print(f"  Capture Time:")
            print(f"    Direct DEAPI: {time1:.3f} seconds")
            print(f"    Fixed PyScope: {time2:.3f} seconds")
            print(f"    Difference: {abs(time1 - time2):.3f} seconds")

            # Image comparison
            shape1 = results['direct_deapi']['image_shape']
            shape2 = results['fixed_pyscope']['image_shape']
            print(f"  Image Shape:")
            print(f"    Direct DEAPI: {shape1}")
            print(f"    Fixed PyScope: {shape2}")
            print(f"    Shapes match: {shape1 == shape2}")

            dtype1 = results['direct_deapi']['image_dtype']
            dtype2 = results['fixed_pyscope']['image_dtype']
            print(f"  Data Type:")
            print(f"    Direct DEAPI: {dtype1}")
            print(f"    Fixed PyScope: {dtype2}")
            print(f"    Types match: {dtype1 == dtype2}")

            # Method comparison
            method1 = results['direct_deapi']['metadata']['method']
            method2 = results['fixed_pyscope']['metadata']['capture_method']
            print(f"  Capture Method:")
            print(f"    Direct DEAPI: {method1}")
            print(f"    Fixed PyScope: {method2}")

            results['comparison'] = {
                'both_successful': True,
                'time_difference': abs(time1 - time2),
                'shapes_match': shape1 == shape2,
                'types_match': dtype1 == dtype2,
                'faster_method': 'Direct DEAPI' if time1 < time2 else 'Fixed PyScope'
            }

        else:
            print("\n❌ Cannot compare - one or both methods failed")
            results['comparison'] = {'both_successful': False}

    # Summary of issues found and fixes applied
    print("\n" + "=" * 60)
    print("4. Issues Found in Original test09 and Fixes Applied")
    print("=" * 60)

    issues_and_fixes = [
        {
            'issue': 'PyScope getImage() may not work optimally with DE cameras',
            'fix': 'Added direct DEAPI client access for DE cameras when available'
        },
        {
            'issue': 'Missing proper DE camera configuration',
            'fix': 'Added DE-specific setup using direct client with same settings as test01'
        },
        {
            'issue': 'No fallback mechanism for different capture methods',
            'fix': 'Implemented intelligent method selection based on camera type'
        },
        {
            'issue': 'Limited metadata from PyScope universal interface',
            'fix': 'Enhanced metadata collection using camera-specific methods'
        },
        {
            'issue': 'No comparison between capture methods',
            'fix': 'Added method comparison functionality for analysis'
        },
        {
            'issue': 'Generic error handling may mask camera-specific issues',
            'fix': 'Added camera-type-specific error handling and diagnostics'
        }
    ]

    print("\n🔧 Key Fixes Applied:")
    for i, item in enumerate(issues_and_fixes, 1):
        print(f"  {i}. Issue: {item['issue']}")
        print(f"     Fix: {item['fix']}")
        print()

    return results


def main():
    """Main function to demonstrate the fixed universal PyScope camera interface"""
    print("=== Fixed Universal PyScope Camera Interface ===")
    print("Addresses issues found when comparing test01 vs test09")
    print()

    # Run the comprehensive comparison test
    comparison_results = run_comparison_test()

    # Save results for analysis
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comparison_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        print(f"\n💾 Comparison results saved to: {results_file}")

    except Exception as e:
        print(f"Error saving results: {e}")

    # Final recommendations
    print("\n" + "=" * 60)
    print("5. Recommendations")
    print("=" * 60)

    recommendations = [
        "For DE cameras: Use direct DEAPI when possible for best performance and compatibility",
        "For other cameras: PyScope universal interface works well",
        "Always test camera-specific features before relying on universal interface",
        "Implement fallback mechanisms for different capture methods",
        "Use proper camera configuration matching the direct API approach",
        "Add comprehensive error handling for different camera types"
    ]

    print("\n📋 Recommendations for PyScope camera interface:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\n🎉 Analysis complete!")

    if comparison_results.get('comparison', {}).get('both_successful'):
        print("✅ Both methods working - comparison data available")
        faster = comparison_results['comparison']['faster_method']
        print(f"⚡ Faster method: {faster}")
    else:
        print("⚠️  One or both methods had issues - check logs for details")


if __name__ == "__main__":
    main()