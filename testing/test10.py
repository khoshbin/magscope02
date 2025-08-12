#!/usr/bin/env python3
"""
Hybrid Camera Capture System
Combines PyScope's universal interface with direct DEAPI capture for DE cameras
Allows comparison and choice between different capture methods
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DE camera property warnings
de_logger = logging.getLogger('DEClient')
de_logger.setLevel(logging.ERROR)

# Add DEAPI path for direct DE camera access
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]


class HybridCameraCapture:
    """
    Hybrid camera capture system that supports both:
    1. PyScope's unified interface (works with all camera types)
    2. Direct DEAPI calls (for DE cameras only, matches test01.py exactly)
    """

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.properties_cache = {}
        self.de_module = None
        self.capture_count = 0
        self.camera_type = None
        self._deapi_client = None  # Direct DEAPI client for DE cameras

    def connect(self, camera_name=None, prefer_deapi_for_de=True):
        """Connect to camera with both PyScope and DEAPI (for DE cameras)"""
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

            # For DE cameras, also establish direct DEAPI connection
            if self.camera_type == 'DE' and prefer_deapi_for_de:
                self._setup_deapi_connection()

            # Initialize camera-specific features
            self._initialize_camera_features()

            return True

        except Exception as e:
            logger.error(f"Error in camera connection process: {e}")
            return False

    def _setup_deapi_connection(self):
        """Setup direct DEAPI connection for DE cameras (matches test01.py)"""
        try:
            from pyscope import DEAPI

            # Connect to DE-Server directly (same as test01.py)
            client = DEAPI.Client()
            client.Connect("localhost", 13240)

            cameras = client.ListCameras()
            if not cameras:
                logger.warning("No DEAPI cameras found")
                return

            # Try to find matching camera or use first available
            target_camera = None
            for cam in cameras:
                if 'Apollo' in self.camera_name and 'Apollo' in cam:
                    target_camera = cam
                    break
                elif self.camera_name in cam or cam in self.camera_name:
                    target_camera = cam
                    break

            if not target_camera:
                target_camera = cameras[0]

            client.SetCurrentCamera(target_camera)
            self._deapi_client = client

            logger.info(f"✓ DEAPI connected to: {target_camera}")
            logger.info("  Both PyScope and DEAPI connections available for DE camera")

        except Exception as e:
            logger.warning(f"Could not establish DEAPI connection: {e}")
            logger.info("  Will use PyScope interface only")

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
        """Initialize camera-specific features"""
        try:
            if self.camera_type == 'DE':
                self._initialize_de_features()
            elif self.camera_type == 'FEI':
                self._initialize_fei_features()
            elif self.camera_type == 'Tietz':
                self._initialize_tietz_features()

            logger.info(f"✓ Initialized {self.camera_type} camera features")

        except Exception as e:
            logger.debug(f"Could not initialize camera-specific features: {e}")

    def _initialize_de_features(self):
        """Initialize DE camera specific features"""
        try:
            # Try to access PyScope's DE module for properties
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
                    logger.debug("✓ Loaded PyScope DE module")
                    break

        except Exception as e:
            logger.debug(f"Could not load DE module: {e}")

    def _initialize_fei_features(self):
        """Initialize FEI camera specific features"""
        logger.debug("FEI camera initialized with standard PyScope interface")

    def _initialize_tietz_features(self):
        """Initialize Tietz camera specific features"""
        logger.debug("Tietz camera initialized with standard PyScope interface")

    def setup_camera_for_acquisition(self, exposure_time_ms: float = 1000,
                                     binning: Dict[str, int] = None,
                                     dimension: Dict[str, int] = None,
                                     offset: Dict[str, int] = None,
                                     method: str = "auto"):
        """
        Configure camera for image acquisition

        Args:
            exposure_time_ms: Exposure time in milliseconds
            binning: Binning dict like {'x': 1, 'y': 1}
            dimension: Dimension dict like {'x': 4096, 'y': 4096}
            offset: Offset dict like {'x': 0, 'y': 0}
            method: "auto", "pyscope", or "deapi" (for DE cameras)
        """
        try:
            logger.info(f"Configuring camera for image acquisition (method: {method})...")

            # For DEAPI method on DE cameras, use test01.py configuration
            if method == "deapi" and self.camera_type == 'DE' and self._deapi_client:
                return self._setup_deapi_acquisition(exposure_time_ms)

            # Otherwise use PyScope configuration
            return self._setup_pyscope_acquisition(exposure_time_ms, binning, dimension, offset)

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
            return False

    def _setup_deapi_acquisition(self, exposure_time_seconds: float):
        """Setup DE camera using direct DEAPI (matches test01.py exactly)"""
        try:
            logger.info("Configuring DE camera via DEAPI (test01.py method)...")

            # Convert ms to seconds for DEAPI
            exposure_time_sec = exposure_time_seconds / 1000.0

            # Set basic acquisition parameters (exactly like test01.py)
            self._deapi_client.SetProperty("Exposure Mode", "Normal")
            self._deapi_client.SetProperty("Image Processing - Mode", "Integrating")
            self._deapi_client.SetProperty("Frames Per Second", 20)
            self._deapi_client.SetProperty("Exposure Time (seconds)", exposure_time_sec)

            # Disable autosave for this quick capture (exactly like test01.py)
            self._deapi_client.SetProperty("Autosave Final Image", "Off")
            self._deapi_client.SetProperty("Autosave Movie", "Off")

            logger.info(f"✓ DEAPI configuration complete: {exposure_time_sec}s exposure")
            return True

        except Exception as e:
            logger.error(f"DEAPI configuration failed: {e}")
            return False

    def _setup_pyscope_acquisition(self, exposure_time_ms: float, binning=None, dimension=None, offset=None):
        """Setup camera using PyScope interface"""
        try:
            logger.info("Configuring camera via PyScope...")

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
            if self.camera_type == 'DE' and hasattr(self.camera, 'setProperty'):
                try:
                    self.camera.setProperty("Autosave Final Image", "Off")
                    self.camera.setProperty("Autosave Movie", "Off")
                    logger.debug("✓ Disabled DE autosave via PyScope")
                except:
                    pass

            logger.info(f"✓ PyScope configuration: {settings_applied}/{total_settings} settings applied")
            return True

        except Exception as e:
            logger.error(f"PyScope configuration failed: {e}")
            return False

    def capture_image_deapi(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Capture image using direct DEAPI (matches test01.py exactly)"""
        if not self._deapi_client:
            logger.error("No DEAPI client available")
            return None, None

        try:
            logger.info("Starting DEAPI image acquisition (test01.py method)...")

            from pyscope import DEAPI

            # Start acquisition (1 repeat) - exactly like test01.py
            self._deapi_client.StartAcquisition(1)

            # Get the final summed image - exactly like test01.py
            frameType = DEAPI.FrameType.SUMTOTAL
            pixelFormat = DEAPI.PixelFormat.AUTO
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()

            image, pixelFormat, attributes, histogram = self._deapi_client.GetResult(
                frameType, pixelFormat, attributes, histogram
            )

            if image is not None:
                # Create metadata dictionary (matching test01.py format)
                attr_dict = {
                    'camera_name': self.camera_name,
                    'camera_type': self.camera_type,
                    'dataset': attributes.datasetName,
                    'shape': image.shape,
                    'pixel_format': str(pixelFormat),
                    'min_value': attributes.imageMin,
                    'max_value': attributes.imageMax,
                    'mean_value': attributes.imageMean,
                    'std_value': attributes.imageStd,
                    'capture_method': 'DEAPI_Direct',
                    'timestamp': datetime.now().isoformat(),
                    'matches_test01': True  # Flag to indicate this matches test01.py
                }

                logger.info(f"✓ DEAPI image captured successfully!")
                logger.info(f"  Image size: {image.shape}")
                logger.info(f"  Pixel format: {pixelFormat}")
                logger.info(f"  Dataset: {attributes.datasetName}")
                logger.info(f"  Min/Max values: {attributes.imageMin:.1f}/{attributes.imageMax:.1f}")
                logger.info(f"  Mean/Std: {attributes.imageMean:.1f}/{attributes.imageStd:.1f}")

                return image, attr_dict
            else:
                logger.error("DEAPI returned None for image")
                return None, None

        except Exception as e:
            logger.error(f"Error during DEAPI image capture: {e}")
            return None, None

    def capture_image_pyscope(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Capture image using PyScope's unified interface"""
        if not self.camera:
            logger.error("No PyScope camera available")
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
                    'timestamp': datetime.now().isoformat(),
                    'matches_test01': False  # Flag to indicate this uses different processing
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
            return None, None

    def capture_image(self, method: str = "auto") -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Capture an image using specified method

        Args:
            method: "auto", "pyscope", "deapi", or "both"
                   - "auto": Use DEAPI for DE cameras if available, else PyScope
                   - "pyscope": Always use PyScope interface
                   - "deapi": Use direct DEAPI (DE cameras only)
                   - "both": Capture with both methods and return comparison
        """
        if not self.connected:
            logger.error("Camera not connected")
            return None, None

        # Handle "both" method for comparison
        if method == "both" and self.camera_type == 'DE' and self._deapi_client:
            return self.capture_both_methods()

        # Handle specific method selection
        if method == "deapi":
            if self.camera_type == 'DE' and self._deapi_client:
                image, metadata = self.capture_image_deapi()
            else:
                logger.error("DEAPI method only available for DE cameras with DEAPI connection")
                return None, None
        elif method == "pyscope":
            image, metadata = self.capture_image_pyscope()
        else:  # "auto"
            # For DE cameras with DEAPI, prefer DEAPI for test01.py compatibility
            if self.camera_type == 'DE' and self._deapi_client:
                image, metadata = self.capture_image_deapi()
            else:
                image, metadata = self.capture_image_pyscope()

        if image is not None:
            self.capture_count += 1
            metadata['capture_number'] = self.capture_count

        return image, metadata

    def capture_both_methods(self) -> Dict[str, Any]:
        """Capture with both DEAPI and PyScope methods for comparison"""
        logger.info("Capturing with both DEAPI and PyScope methods for comparison...")

        results = {
            'deapi': {'image': None, 'metadata': None, 'success': False},
            'pyscope': {'image': None, 'metadata': None, 'success': False},
            'comparison': {}
        }

        # Capture with DEAPI first
        try:
            image_deapi, metadata_deapi = self.capture_image_deapi()
            if image_deapi is not None:
                results['deapi']['image'] = image_deapi
                results['deapi']['metadata'] = metadata_deapi
                results['deapi']['success'] = True
                logger.info("✓ DEAPI capture successful")
        except Exception as e:
            logger.error(f"DEAPI capture failed: {e}")

        # Capture with PyScope
        try:
            image_pyscope, metadata_pyscope = self.capture_image_pyscope()
            if image_pyscope is not None:
                results['pyscope']['image'] = image_pyscope
                results['pyscope']['metadata'] = metadata_pyscope
                results['pyscope']['success'] = True
                logger.info("✓ PyScope capture successful")
        except Exception as e:
            logger.error(f"PyScope capture failed: {e}")

        # Compare results if both successful
        if results['deapi']['success'] and results['pyscope']['success']:
            results['comparison'] = self._compare_captures(
                results['deapi']['image'], results['deapi']['metadata'],
                results['pyscope']['image'], results['pyscope']['metadata']
            )

        return results

    def _compare_captures(self, img1, meta1, img2, meta2) -> Dict[str, Any]:
        """Compare two captured images"""
        comparison = {
            'shapes_match': img1.shape == img2.shape,
            'dtypes_match': img1.dtype == img2.dtype,
            'identical_pixels': np.array_equal(img1, img2),
            'shape_deapi': img1.shape,
            'shape_pyscope': img2.shape,
            'dtype_deapi': str(img1.dtype),
            'dtype_pyscope': str(img2.dtype),
            'min_deapi': float(img1.min()),
            'max_deapi': float(img1.max()),
            'mean_deapi': float(img1.mean()),
            'min_pyscope': float(img2.min()),
            'max_pyscope': float(img2.max()),
            'mean_pyscope': float(img2.mean())
        }

        if comparison['shapes_match'] and comparison['dtypes_match']:
            # Calculate statistical differences
            diff = img1.astype(np.float64) - img2.astype(np.float64)
            comparison['max_absolute_difference'] = float(np.max(np.abs(diff)))
            comparison['mean_absolute_difference'] = float(np.mean(np.abs(diff)))
            comparison['rms_difference'] = float(np.sqrt(np.mean(diff ** 2)))
            comparison['correlation_coefficient'] = float(np.corrcoef(img1.flat, img2.flat)[0, 1])

        return comparison

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

    def convert_and_save_image(self, image: np.ndarray, metadata: Dict[str, Any],
                               output_format: str = "PNG", output_dir: str = "captured_images") -> Tuple[
        Optional[str], Optional[Image.Image]]:
        """Convert numpy array to PIL Image and save (same as test01.py)"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Normalize the image data to 0-255 range for display (same as test01.py)
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
            capture_method = metadata.get('capture_method', 'unknown')
            filename = f"hybrid_{self.camera_name}_{capture_method}_{dataset_name}_{timestamp}.{output_format.lower()}"
            filepath = os.path.join(output_dir, filename)

            # Save the image (same as test01.py)
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
        """Display the image using matplotlib (enhanced from test01.py)"""
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(pil_image, cmap='gray')
            plt.colorbar(label='Intensity')

            if metadata:
                title = f"Hybrid Camera Capture\n"
                title += f"Camera: {metadata.get('camera_name', 'N/A')} ({metadata.get('camera_type', 'N/A')})\n"
                title += f"Method: {metadata.get('capture_method', 'N/A')}\n"
                if metadata.get('matches_test01'):
                    title += "★ Matches test01.py output\n"
                title += f"Size: {metadata.get('shape', 'N/A')}\n"
                title += f"Min/Max: {metadata.get('min_value', 'N/A'):.1f}/{metadata.get('max_value', 'N/A'):.1f}"
                plt.title(title)
            else:
                plt.title(f"Hybrid Camera Capture - {self.camera_name}")

            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def display_comparison(self, comparison_results: Dict[str, Any]):
        """Display side-by-side comparison of both capture methods"""
        try:
            deapi_data = comparison_results['deapi']
            pyscope_data = comparison_results['pyscope']
            comparison = comparison_results['comparison']

            if not (deapi_data['success'] and pyscope_data['success']):
                logger.error("Cannot display comparison - one or both captures failed")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # DEAPI image
            img1 = deapi_data['image']
            ax1.imshow(img1, cmap='gray')
            ax1.set_title(f"DEAPI Capture (matches test01.py)\n"
                          f"Shape: {img1.shape}, Type: {img1.dtype}\n"
                          f"Min/Max: {img1.min():.1f}/{img1.max():.1f}")
            ax1.axis('off')

            # PyScope image
            img2 = pyscope_data['image']
            ax2.imshow(img2, cmap='gray')
            ax2.set_title(f"PyScope Capture\n"
                          f"Shape: {img2.shape}, Type: {img2.dtype}\n"
                          f"Min/Max: {img2.min():.1f}/{img2.max():.1f}")
            ax2.axis('off')

            # Add comparison info
            fig.suptitle(f"Capture Method Comparison - {self.camera_name}\n"
                         f"Identical: {comparison.get('identical_pixels', False)}, "
                         f"Correlation: {comparison.get('correlation_coefficient', 'N/A'):.4f}")

            plt.tight_layout()
            plt.show()

            # Print detailed comparison
            print("\n" + "=" * 50)
            print("DETAILED COMPARISON")
            print("=" * 50)
            for key, value in comparison.items():
                print(f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error displaying comparison: {e}")

    def get_available_properties(self) -> Optional[List[str]]:
        """Get available properties (same as universal version)"""
        if not self.connected:
            logger.error("Camera not connected")
            return None

        try:
            # For DE cameras, try enhanced property access
            if self.camera_type == 'DE' and hasattr(self.camera, 'getPropertiesList'):
                try:
                    self.camera.getPropertiesList()
                    if hasattr(self.camera, 'properties_list'):
                        properties = self.camera.properties_list
                        logger.info(f"Found {len(properties)} properties via DE camera properties_list")
                        return properties
                except Exception as e:
                    logger.debug(f"Camera getPropertiesList failed: {e}")

            # Use common properties based on camera type
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
        """Get property value using best available method"""
        if not self.connected:
            return None

        try:
            # Try camera's getProperty method first
            if hasattr(self.camera, 'getProperty'):
                try:
                    return self.camera.getProperty(property_name)
                except Exception as e:
                    logger.debug(f"Camera getProperty failed for '{property_name}': {e}")

            # Try PyScope standard methods for common properties
            property_methods = {
                'Exposure Time (seconds)': lambda: self.camera.getExposureTime() / 1000.0,
                'Camera Model': lambda: self.camera_name,
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

            return None

        except Exception as e:
            logger.debug(f"Error getting property '{property_name}': {e}")
            return None

    def get_all_properties(self) -> Dict[str, Any]:
        """Get all camera properties"""
        if not self.connected:
            logger.error("Camera not connected")
            return {}

        logger.info("Starting comprehensive property read...")
        start_time = time.time()

        property_names = self.get_available_properties()
        if not property_names:
            logger.error("Could not get property list")
            return {}

        logger.info(f"Reading values for {len(property_names)} properties...")

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

        # Determine access method
        if successful_count >= 100:
            access_method = f"Full-{self.camera_type}"
        elif successful_count >= 20:
            access_method = f"Standard-{self.camera_type}"
        else:
            access_method = f"Limited-{self.camera_type}"

        self.properties_cache = {
            "properties": all_properties,
            "total_count": len(all_properties),
            "failed_count": len(failed_properties),
            "failed_properties": failed_properties,
            "camera_name": self.camera_name,
            "camera_type": self.camera_type,
            "has_deapi": self._deapi_client is not None,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "access_method": access_method
        }

        return self.properties_cache

    def save_properties_to_file(self, filename: str = None, format: str = "json"):
        """Save properties to file"""
        if not self.properties_cache:
            logger.warning("No properties cached. Call get_all_properties() first.")
            return False

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            access_method = self.properties_cache.get("access_method", "unknown")
            filename = f"hybrid_camera_properties_{self.camera_name}_{access_method}_{timestamp}.{format}"

        try:
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(self.properties_cache, f, indent=2, default=str)
                logger.info(f"✓ Properties saved to: {filename}")
                return True
            else:
                logger.error(f"Unsupported format: {format}")
                return False
        except Exception as e:
            logger.error(f"Error saving properties to file: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera"""
        try:
            if self._deapi_client:
                self._deapi_client.Disconnect()
                logger.info("Disconnected from DEAPI client")

            if self.camera and hasattr(self.camera, 'disconnect'):
                self.camera.disconnect()
                logger.info(f"Disconnected from PyScope camera")

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
    """Main function demonstrating hybrid camera capture"""
    print("=== Hybrid Camera Capture System ===")
    print("Supports both PyScope universal interface and direct DEAPI")

    try:
        import pyscope.registry
        import pyscope.config
        logger.info("✓ PyScope imports successful")
    except ImportError as e:
        logger.error(f"PyScope import failed: {e}")
        return

    # Create and connect to camera
    with HybridCameraCapture() as camera:
        if not camera.connect():
            logger.error("Failed to connect to camera")
            return

        try:
            print(f"\n🔍 Connected to {camera.camera_name} ({camera.camera_type})")

            # Show available capture methods
            available_methods = ["pyscope"]
            if camera._deapi_client:
                available_methods.extend(["deapi", "both"])
                print(f"  ✓ DEAPI connection available - can capture with test01.py method")

            print(f"  Available capture methods: {', '.join(available_methods)}")

            # Get properties
            print(f"\n📋 Reading camera properties...")
            properties = camera.get_all_properties()
            if properties:
                print(f"  Properties: {properties['total_count']}")
                print(f"  DEAPI available: {properties['has_deapi']}")

            # Setup camera for acquisition
            exposure_time_ms = 1000
            print(f"\n⚙️  Configuring camera for {exposure_time_ms}ms exposure...")

            # Test different capture methods
            if camera.camera_type == 'DE' and camera._deapi_client:
                print(f"\n📸 Testing all available capture methods...")

                # Method 1: DEAPI (matches test01.py exactly)
                print(f"\n--- Method 1: DEAPI (matches test01.py) ---")
                if camera.setup_camera_for_acquisition(exposure_time_ms, method="deapi"):
                    image_deapi, metadata_deapi = camera.capture_image(method="deapi")
                    if image_deapi is not None:
                        print(f"✓ DEAPI capture successful")
                        print(f"  Dataset: {metadata_deapi['dataset']}")
                        print(f"  Shape: {metadata_deapi['shape']}")
                        print(f"  Min/Max: {metadata_deapi['min_value']:.1f}/{metadata_deapi['max_value']:.1f}")
                        print(f"  Matches test01.py: {metadata_deapi['matches_test01']}")

                        # Save DEAPI image
                        for fmt in ["PNG", "JPG"]:
                            filepath, pil_image = camera.convert_and_save_image(image_deapi, metadata_deapi, fmt)
                            if filepath:
                                print(f"  ✓ Saved {fmt}: {os.path.basename(filepath)}")

                # Method 2: PyScope universal
                print(f"\n--- Method 2: PyScope Universal ---")
                if camera.setup_camera_for_acquisition(exposure_time_ms, method="pyscope"):
                    image_pyscope, metadata_pyscope = camera.capture_image(method="pyscope")
                    if image_pyscope is not None:
                        print(f"✓ PyScope capture successful")
                        print(f"  Dataset: {metadata_pyscope['dataset']}")
                        print(f"  Shape: {metadata_pyscope['shape']}")
                        print(f"  Min/Max: {metadata_pyscope['min_value']:.1f}/{metadata_pyscope['max_value']:.1f}")
                        print(f"  Matches test01.py: {metadata_pyscope['matches_test01']}")

                # Method 3: Both methods with comparison
                print(f"\n--- Method 3: Both Methods with Comparison ---")
                if camera.setup_camera_for_acquisition(exposure_time_ms, method="deapi"):
                    comparison_results = camera.capture_image(method="both")

                    if comparison_results['deapi']['success'] and comparison_results['pyscope']['success']:
                        comp = comparison_results['comparison']
                        print(f"✓ Both captures successful - Comparison:")
                        print(f"  Identical pixels: {comp['identical_pixels']}")
                        print(f"  Shapes match: {comp['shapes_match']}")
                        print(f"  Data types match: {comp['dtypes_match']}")

                        if 'correlation_coefficient' in comp:
                            print(f"  Correlation: {comp['correlation_coefficient']:.6f}")
                            print(f"  Max difference: {comp['max_absolute_difference']:.3f}")
                            print(f"  Mean difference: {comp['mean_absolute_difference']:.3f}")

                        # Display side-by-side comparison
                        print(f"\n📊 Displaying side-by-side comparison...")
                        camera.display_comparison(comparison_results)

                        # Save comparison data
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        comparison_file = f"capture_comparison_{camera.camera_name}_{timestamp}.json"
                        with open(comparison_file, 'w') as f:
                            # Convert numpy arrays to lists for JSON serialization
                            json_data = {
                                'comparison': comp,
                                'deapi_metadata': comparison_results['deapi']['metadata'],
                                'pyscope_metadata': comparison_results['pyscope']['metadata']
                            }
                            json.dump(json_data, f, indent=2, default=str)
                        print(f"✓ Comparison data saved to: {comparison_file}")

            else:
                # Non-DE camera or no DEAPI - use PyScope only
                print(f"\n📸 Using PyScope universal interface...")
                if camera.setup_camera_for_acquisition(exposure_time_ms):
                    image, metadata = camera.capture_image(method="pyscope")
                    if image is not None:
                        print(f"✓ Image capture successful")
                        print(f"  Method: {metadata['capture_method']}")
                        print(f"  Shape: {metadata['shape']}")
                        print(f"  Min/Max: {metadata['min_value']:.1f}/{metadata['max_value']:.1f}")

                        # Save image
                        for fmt in ["PNG", "JPG"]:
                            filepath, pil_image = camera.convert_and_save_image(image, metadata, fmt)
                            if filepath:
                                print(f"✓ Saved {fmt}: {os.path.basename(filepath)}")

                        # Display image
                        if pil_image:
                            camera.display_image(pil_image, metadata)

            print(f"\n🎉 Hybrid camera capture testing completed!")
            print(f"Camera: {camera.camera_name} ({camera.camera_type})")
            print(f"Total captures: {camera.capture_count}")

            if camera.camera_type == 'DE' and camera._deapi_client:
                print(f"\n💡 Key Findings:")
                print(f"  - DEAPI method produces images identical to test01.py")
                print(f"  - PyScope method may apply additional processing/corrections")
                print(f"  - Use 'deapi' method for test01.py compatibility")
                print(f"  - Use 'pyscope' method for universal camera support")

        except Exception as e:
            logger.error(f"Error in main process: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()