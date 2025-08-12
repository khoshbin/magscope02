#!/usr/bin/env python3
"""
Modified test09.py - PyScope DE Camera Script to Match test01.py Results Exactly
Key changes:
1. Use direct DEAPI access through PyScope
2. Bypass finalizeGeometry() processing
3. Use exact test01.py pixel format and settings
4. Apply identical normalization
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging to match test01.py style
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add DEAPI path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]

try:
    from pyscope import DEAPI

    DEAPI_AVAILABLE = True
except ImportError:
    DEAPI_AVAILABLE = False

try:
    import pyscope.registry

    PYSCOPE_AVAILABLE = True
except ImportError:
    PYSCOPE_AVAILABLE = False


class UniversalPyScope_Camera_Test01_Match:
    """
    Modified UniversalPyScope_Camera to produce test01.py identical results
    """

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.capture_count = 0
        self.camera_type = None

        # Add direct DEAPI client for test01.py matching
        self.deapi_client = None

    def connect(self, camera_name=None):
        """Connect to camera with both PyScope and direct DEAPI access"""
        try:
            import pyscope.registry

            # Determine camera to connect to
            target_camera = camera_name or self.camera_name or "DEApollo"

            # Connect via PyScope
            logger.info(f"Connecting PyScope to: {target_camera}")
            self.camera = pyscope.registry.getClass(target_camera)()
            self.camera_name = target_camera

            # Test basic connectivity
            camera_size = self.camera.getCameraSize()
            logger.info(f"✓ PyScope connected to {target_camera}")
            logger.info(f"  Camera size: {camera_size}")

            self.camera_type = 'DE'  # We know it's a DE camera
            self.connected = True

            # CRITICAL: Also establish direct DEAPI connection for test01.py matching
            if DEAPI_AVAILABLE:
                try:
                    logger.info("Establishing direct DEAPI connection for test01.py matching...")
                    self.deapi_client = DEAPI.Client()
                    self.deapi_client.Connect()
                    cameras = self.deapi_client.ListCameras()
                    if cameras:
                        self.deapi_client.SetCurrentCamera(cameras[0])
                        logger.info(f"✓ Direct DEAPI connected to: {cameras[0]}")
                    else:
                        logger.warning("No DEAPI cameras found")
                        self.deapi_client = None
                except Exception as e:
                    logger.warning(f"Direct DEAPI connection failed: {e}")
                    self.deapi_client = None
            else:
                logger.warning("DEAPI not available - cannot match test01.py exactly")

            return True

        except Exception as e:
            logger.error(f"Error in camera connection: {e}")
            return False

    def setup_camera_for_test01_match(self, exposure_time_s=1.0):
        """Configure camera with EXACT test01.py settings"""
        try:
            logger.info("Configuring camera to match test01.py exactly...")

            # Exact test01.py settings
            test01_settings = {
                "Exposure Mode": "Normal",
                "Image Processing - Mode": "Integrating",
                "Frames Per Second": 20,
                "Exposure Time (seconds)": exposure_time_s,
                "Autosave Final Image": "Off",
                "Autosave Movie": "Off"
            }

            settings_applied = 0

            # Apply settings via direct DEAPI (preferred for test01.py match)
            if self.deapi_client:
                logger.info("Applying settings via direct DEAPI...")
                for prop_name, prop_value in test01_settings.items():
                    try:
                        self.deapi_client.SetProperty(prop_name, prop_value)
                        logger.debug(f"✓ DEAPI set {prop_name}: {prop_value}")
                        settings_applied += 1
                    except Exception as e:
                        logger.warning(f"DEAPI failed to set {prop_name}: {e}")

            # Also apply via PyScope as backup
            if hasattr(self.camera, 'setProperty'):
                logger.info("Applying settings via PyScope...")
                for prop_name, prop_value in test01_settings.items():
                    try:
                        self.camera.setProperty(prop_name, prop_value)
                        logger.debug(f"✓ PyScope set {prop_name}: {prop_value}")
                    except Exception as e:
                        logger.warning(f"PyScope failed to set {prop_name}: {e}")

            # Set exposure time via PyScope method as well
            try:
                exposure_ms = int(exposure_time_s * 1000)
                self.camera.setExposureTime(exposure_ms)
                logger.debug(f"✓ PyScope set exposure time: {exposure_ms} ms")
            except Exception as e:
                logger.warning(f"PyScope setExposureTime failed: {e}")

            logger.info(f"✓ Camera configured for test01.py match")
            return True

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
            return False

    def capture_image_test01_exact_match(self):
        """Capture image using EXACT test01.py method"""
        if not self.connected:
            logger.error("Camera not connected")
            return None, None

        # Method 1: Use direct DEAPI client (EXACT test01.py approach)
        if self.deapi_client:
            try:
                logger.info("Capturing via direct DEAPI (exact test01.py method)...")

                # This is EXACTLY what test01.py does:
                self.deapi_client.StartAcquisition(1)

                frameType = DEAPI.FrameType.SUMTOTAL
                pixelFormat = DEAPI.PixelFormat.AUTO  # EXACT same as test01.py
                attributes = DEAPI.Attributes()
                histogram = DEAPI.Histogram()

                image, pixelFormat, attributes, histogram = self.deapi_client.GetResult(
                    frameType, pixelFormat, attributes, histogram
                )

                if image is not None:
                    # Create metadata identical to test01.py
                    metadata = {
                        'method': 'DEAPI_GetResult_Exact_Match',
                        'frame_type': 'SUMTOTAL',
                        'pixel_format': str(pixelFormat),
                        'dataset': attributes.datasetName,
                        'image_min': attributes.imageMin,
                        'image_max': attributes.imageMax,
                        'image_mean': attributes.imageMean,
                        'image_std': attributes.imageStd,
                        'frame_count': attributes.frameCount,
                        'shape': image.shape,
                        'dtype': str(image.dtype),
                        'capture_method': 'Direct_DEAPI_via_PyScope',
                        'timestamp': datetime.now().isoformat()
                    }

                    logger.info(f"✓ Direct DEAPI captured (test01.py exact match)!")
                    logger.info(f"  Image size: {image.shape}")
                    logger.info(f"  Data type: {image.dtype}")
                    logger.info(f"  Value range: [{image.min():.1f}, {image.max():.1f}]")
                    logger.info(f"  Mean ± Std: {image.mean():.1f} ± {image.std():.1f}")
                    logger.info(f"  Dataset: {attributes.datasetName}")

                    self.capture_count += 1
                    return image, metadata

                else:
                    logger.error("Direct DEAPI returned None")

            except Exception as e:
                logger.error(f"Direct DEAPI capture failed: {e}")

        # Method 2: Try PyScope raw access (bypass finalizeGeometry)
        logger.info("Attempting PyScope raw access (bypass finalizeGeometry)...")
        return self._capture_pyscope_raw_for_test01_match()

    def _capture_pyscope_raw_for_test01_match(self):
        """Try to capture raw PyScope image without finalizeGeometry processing"""
        try:
            # Try to access raw image before finalizeGeometry
            if hasattr(self.camera, '_getImage'):
                logger.info("Using PyScope _getImage (raw method)...")
                raw_image = self.camera._getImage()

                if raw_image is not None:
                    metadata = {
                        'method': 'PyScope_Raw_getImage_for_test01_match',
                        'camera_name': self.camera_name,
                        'shape': raw_image.shape,
                        'dtype': str(raw_image.dtype),
                        'capture_method': 'PyScope_Raw_Bypass_finalizeGeometry',
                        'note': 'Attempted_test01_match_via_raw_access',
                        'timestamp': datetime.now().isoformat()
                    }

                    logger.info(f"✓ PyScope raw captured!")
                    logger.info(f"  Image size: {raw_image.shape}")
                    logger.info(f"  Data type: {raw_image.dtype}")
                    logger.info(f"  Value range: [{raw_image.min():.1f}, {raw_image.max():.1f}]")
                    logger.info(f"  Mean ± Std: {raw_image.mean():.1f} ± {raw_image.std():.1f}")

                    self.capture_count += 1
                    return raw_image, metadata

                else:
                    logger.error("PyScope _getImage returned None")
            else:
                logger.warning("PyScope _getImage method not available")

        except Exception as e:
            logger.error(f"PyScope raw capture failed: {e}")

        # Method 3: Fallback with warning
        logger.warning("Falling back to standard PyScope - may not match test01.py exactly")
        return self._capture_pyscope_standard_with_warning()

    def _capture_pyscope_standard_with_warning(self):
        """Fallback to standard PyScope with processing warning"""
        try:
            logger.warning("Using standard PyScope getImage - includes finalizeGeometry processing")

            image = self.camera.getImage()

            if image is not None:
                metadata = {
                    'method': 'PyScope_Standard_getImage_with_processing',
                    'camera_name': self.camera_name,
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'capture_method': 'PyScope_Standard_with_finalizeGeometry',
                    'warning': 'May_not_match_test01_exactly_due_to_processing',
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"✓ PyScope standard captured (with processing)!")
                logger.info(f"  Image size: {image.shape}")
                logger.info(f"  Data type: {image.dtype}")
                logger.info(f"  Value range: [{image.min():.1f}, {image.max():.1f}]")
                logger.info(f"  Mean ± Std: {image.mean():.1f} ± {image.std():.1f}")
                logger.warning("  ⚠️ This image includes PyScope finalizeGeometry processing")

                self.capture_count += 1
                return image, metadata
            else:
                logger.error("PyScope standard getImage returned None")
                return None, None

        except Exception as e:
            logger.error(f"PyScope standard capture failed: {e}")
            return None, None

    def convert_and_save_image_test01_exact(self, image, metadata, output_format="PNG", output_dir="captured_images"):
        """Convert and save image using EXACT test01.py normalization"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Use EXACT test01.py normalization logic (copied verbatim)
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

            # Generate filename with test01.py style
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = metadata.get('dataset', metadata.get('method', 'pyscope_test01_match'))
            filename = f"pyscope_test01_match_{dataset_name}_{timestamp}.{output_format.lower()}"
            filepath = os.path.join(output_dir, filename)

            # Save the image (exact test01.py method)
            if output_format.upper() == "JPG" or output_format.upper() == "JPEG":
                pil_image.save(filepath, "JPEG", quality=95)
            else:  # PNG
                pil_image.save(filepath, "PNG")

            logger.info(f"✓ Image saved as: {filepath}")
            return filepath, pil_image

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None, None

    def display_image_test01_style(self, pil_image, metadata=None):
        """Display image using test01.py style"""
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(pil_image, cmap='gray')
            plt.colorbar(label='Intensity')

            if metadata:
                title = f"PyScope→test01.py Match\n"
                title += f"Method: {metadata.get('capture_method', metadata.get('method', 'Unknown'))}\n"

                if metadata.get('dataset'):
                    title += f"Dataset: {metadata['dataset']}\n"
                    title += f"Size: {metadata.get('shape', 'N/A')}\n"
                    title += f"Min/Max: {metadata.get('image_min', 'N/A'):.1f}/{metadata.get('image_max', 'N/A'):.1f}"
                else:
                    title += f"Size: {metadata.get('shape', 'N/A')}\n"
                    title += f"Type: {metadata.get('dtype', 'N/A')}"

                plt.title(title)
            else:
                plt.title("PyScope→test01.py Match")

            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def disconnect(self):
        """Disconnect from both PyScope and DEAPI"""
        try:
            # Disconnect direct DEAPI client
            if self.deapi_client:
                self.deapi_client.Disconnect()
                logger.info("Disconnected from direct DEAPI")
                self.deapi_client = None

            self.connected = False
            logger.info(f"Disconnected from PyScope camera")

        except Exception as e:
            logger.debug(f"Error during disconnect: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def main():
    """Main function to capture images exactly matching test01.py"""
    print("=" * 80)
    print("Modified test09.py - PyScope DE Camera to Match test01.py EXACTLY")
    print("This version uses direct DEAPI access to produce identical results")
    print("=" * 80)

    # Test imports
    if not PYSCOPE_AVAILABLE:
        logger.error("PyScope not available")
        return

    if not DEAPI_AVAILABLE:
        logger.warning("DEAPI not available - may not be able to match test01.py exactly")

    # Create and connect to camera
    with UniversalPyScope_Camera_Test01_Match() as camera:
        if not camera.connect():
            logger.error("Failed to connect to camera")
            return

        try:
            print(f"\n🔧 Configuring camera to match test01.py settings...")

            # Setup camera for test01.py match
            if not camera.setup_camera_for_test01_match(exposure_time_s=1.0):
                logger.error("Failed to configure camera")
                return

            print(f"\n📸 Capturing image using test01.py exact method...")

            # Capture image using test01.py approach
            image, metadata = camera.capture_image_test01_exact_match()

            if image is not None:
                print(f"\n🎉 Image captured successfully!")
                print(f"  Method: {metadata['capture_method']}")
                print(f"  Image shape: {metadata['shape']}")
                print(f"  Data type: {metadata['dtype']}")

                # Display detailed results
                if 'image_min' in metadata:
                    print(f"  Min/Max values: {metadata['image_min']:.1f}/{metadata['image_max']:.1f}")
                    print(f"  Mean/Std: {metadata['image_mean']:.1f}/{metadata['image_std']:.1f}")
                    print(f"  Dataset: {metadata['dataset']}")
                    print(f"  Frame count: {metadata['frame_count']}")
                    print(f"  Pixel format: {metadata['pixel_format']}")

                # Check if this matches expected test01.py characteristics
                print(f"\n📊 test01.py Match Analysis:")
                expected_dtype = 'float32'
                expected_range_min = 0.0
                expected_range_max_approx = 120.0

                if metadata['dtype'] == expected_dtype:
                    print(f"  ✅ Data type matches test01.py: {expected_dtype}")
                else:
                    print(f"  ⚠️ Data type differs: expected {expected_dtype}, got {metadata['dtype']}")

                if metadata.get('capture_method') == 'Direct_DEAPI_via_PyScope':
                    print(f"  ✅ Using direct DEAPI - should be IDENTICAL to test01.py!")
                elif 'Raw' in metadata.get('capture_method', ''):
                    print(f"  ✅ Using raw PyScope - should be very close to test01.py")
                else:
                    print(f"  ⚠️ Using processed PyScope - may differ from test01.py")

                # Save in both formats using test01.py exact normalization
                print(f"\n💾 Saving images using test01.py exact normalization...")
                for fmt in ["PNG", "JPG"]:
                    filepath, pil_image = camera.convert_and_save_image_test01_exact(
                        image, metadata, fmt
                    )
                    if filepath:
                        print(f"✓ Saved {fmt} format: {os.path.basename(filepath)}")

                # Display the image
                if pil_image:
                    print(f"\n📊 Displaying image...")
                    camera.display_image_test01_style(pil_image, metadata)

                print(f"\n🎉 PyScope→test01.py match completed!")
                print(f"Images captured: {camera.capture_count}")

                if metadata.get('capture_method') == 'Direct_DEAPI_via_PyScope':
                    print(f"✅ SUCCESS: Used direct DEAPI - results should be IDENTICAL to test01.py!")
                else:
                    print(f"⚠️ Used fallback method - results may not be identical to test01.py")

            else:
                print(f"\n❌ Image capture failed")

        except Exception as e:
            logger.error(f"Error in main process: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()