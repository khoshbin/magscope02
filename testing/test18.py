#!/usr/bin/env python3
"""
SIMPLE FIX: Just fix the normalization issue in test17.py
No dual connections needed - just the proper uint16 → uint8 conversion
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

# Add DEAPI path for direct access
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]

# Configure logging to suppress DE camera property warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DE camera property warnings that are harmless
de_logger = logging.getLogger('DEClient')
de_logger.setLevel(logging.ERROR)  # Only show real errors


class SimplePyScope_Camera:
    """
    SIMPLE FIX: Just fix the normalization issue, use PyScope only
    No dual connections - just proper image conversion
    """

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.capture_count = 0
        self.camera_type = None

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

    def setup_camera_for_acquisition(self, exposure_time_ms: float = 1000,
                                     binning: Dict[str, int] = None,
                                     dimension: Dict[str, int] = None,
                                     offset: Dict[str, int] = None):
        """Configure camera for image acquisition using PyScope interface"""
        try:
            logger.info("Configuring camera for image acquisition...")

            settings_applied = 0
            total_settings = 0

            # Set exposure time
            total_settings += 1
            try:
                self.camera.setExposureTime(exposure_time_ms)
                current_exposure = self.camera.getExposureTime()
                logger.info(f"✓ Set exposure time to {exposure_time_ms} ms (current: {current_exposure} ms)")
                settings_applied += 1
            except Exception as e:
                logger.warning(f"Could not set exposure time: {e}")

            # Set binning if specified
            if binning:
                total_settings += 1
                try:
                    self.camera.setBinning(binning)
                    current_binning = self.camera.getBinning()
                    logger.info(f"✓ Set binning to {binning} (current: {current_binning})")
                    settings_applied += 1
                except Exception as e:
                    logger.warning(f"Could not set binning: {e}")

            # Set dimension if specified
            if dimension:
                total_settings += 1
                try:
                    self.camera.setDimension(dimension)
                    current_dimension = self.camera.getDimension()
                    logger.info(f"✓ Set dimension to {dimension} (current: {current_dimension})")
                    settings_applied += 1
                except Exception as e:
                    logger.warning(f"Could not set dimension: {e}")

            # Set offset if specified
            if offset:
                total_settings += 1
                try:
                    self.camera.setOffset(offset)
                    current_offset = self.camera.getOffset()
                    logger.info(f"✓ Set offset to {offset} (current: {current_offset})")
                    settings_applied += 1
                except Exception as e:
                    logger.warning(f"Could not set offset: {e}")

            logger.info(f"Camera configuration: {settings_applied}/{total_settings} settings applied successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
            return False

    def capture_image_pyscope_simple(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
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
                    'capture_method': 'PyScope_Simple_Fixed',
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
        """Capture an image using PyScope (simple fix)"""
        if not self.connected:
            logger.error("Camera not connected")
            return None, None

        # Use PyScope universal interface (no dual connections)
        image, metadata = self.capture_image_pyscope_simple()

        if image is not None:
            self.capture_count += 1
            metadata['capture_number'] = self.capture_count

        return image, metadata

    def convert_and_save_image_FIXED(self, image: np.ndarray, metadata: Dict[str, Any],
                                     output_format: str = "PNG", output_dir: str = "captured_images") -> Tuple[
        Optional[str], Optional[Image.Image]]:
        """
        FIXED: Convert numpy array to PIL Image and save as JPG/PNG
        This version uses the CORRECT normalization to fix the brightness issue
        """
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger.info(
                f"BEFORE CONVERSION: dtype={image.dtype}, min={image.min()}, max={image.max()}, mean={image.mean():.2f}")

            # ✅ FIXED: Use PROPER normalization for all data types
            if image.dtype == np.float32 or image.dtype == np.float64:
                # For float images, normalize to 0-1 range first (like test01.py)
                if image.max() > image.min():
                    image_normalized = (image - image.min()) / (image.max() - image.min())
                    image_8bit = (image_normalized * 255).astype(np.uint8)
                else:
                    image_8bit = np.zeros_like(image, dtype=np.uint8)
                logger.info("✅ FIXED: Used float32/float64 normalization (like test01.py)")

            elif image.dtype == np.uint16:
                # ✅ FIXED: For uint16, use actual image range, NOT fixed division by 256
                if image.max() > image.min():
                    # Use actual range normalization (THE FIX!)
                    image_normalized = (image.astype(np.float32) - image.min()) / (image.max() - image.min())
                    image_8bit = (image_normalized * 255).astype(np.uint8)
                    logger.info(f"✅ FIXED: Used uint16 range normalization ({image.min()}-{image.max()} → 0-255)")
                    logger.info(f"✅ FIXED: NOT using broken /256 method!")
                else:
                    image_8bit = np.zeros_like(image, dtype=np.uint8)

            elif image.dtype == np.uint8:
                image_8bit = image
                logger.info("✅ FIXED: uint8 image used as-is")
            else:
                # Generic conversion
                if image.max() > image.min():
                    image_normalized = (image.astype(np.float64) - image.min()) / (image.max() - image.min())
                    image_8bit = (image_normalized * 255).astype(np.uint8)
                else:
                    image_8bit = np.zeros_like(image, dtype=np.uint8)
                logger.info(f"✅ FIXED: Used generic normalization for {image.dtype}")

            logger.info(
                f"AFTER CONVERSION: dtype={image_8bit.dtype}, min={image_8bit.min()}, max={image_8bit.max()}, mean={image_8bit.mean():.2f}")

            # Verify the fix worked
            if image_8bit.max() > 50:  # Should have good brightness now
                logger.info("🎉 SUCCESS: Image now has proper brightness!")
            else:
                logger.warning("⚠️  Image might still be dark")

            # Convert to PIL Image
            pil_image = Image.fromarray(image_8bit, mode='L')  # 'L' for grayscale

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = metadata.get('dataset', 'unknown')
            camera_type = metadata.get('camera_type', 'unknown')
            capture_method = metadata.get('capture_method', 'unknown')
            filename = f"SIMPLE_FIXED_{self.camera_name}_{camera_type}_{capture_method}_{dataset_name}_{timestamp}.{output_format.lower()}"
            filepath = os.path.join(output_dir, filename)

            # Save the image
            if output_format.upper() == "JPG" or output_format.upper() == "JPEG":
                pil_image.save(filepath, "JPEG", quality=95)
            else:  # PNG
                pil_image.save(filepath, "PNG")

            logger.info(f"✓ FIXED Image saved as: {filepath}")
            return filepath, pil_image

        except Exception as e:
            logger.error(f"Error saving FIXED image: {e}")
            return None, None

    def display_image_fixed(self, pil_image: Image.Image, metadata: Dict[str, Any] = None):
        """Display the image using matplotlib with fix info"""
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(pil_image, cmap='gray')
            plt.colorbar(label='Intensity')

            if metadata:
                title = f"SIMPLE FIXED PyScope Image\n"
                title += f"Camera: {metadata.get('camera_name', 'N/A')} ({metadata.get('camera_type', 'N/A')})\n"
                title += f"Method: {metadata.get('capture_method', 'N/A')}\n"
                title += f"Original range: {metadata.get('min_value', 'N/A'):.1f} to {metadata.get('max_value', 'N/A'):.1f}\n"
                title += f"Fixed normalization applied!"
                plt.title(title)
            else:
                plt.title(f"SIMPLE FIXED Image - {self.camera_name}")

            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def disconnect(self):
        """Disconnect from camera"""
        try:
            # Disconnect from PyScope camera
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


def demonstrate_the_exact_fix():
    """Show exactly what the fix does with your real data"""
    print("=" * 80)
    print("=== DEMONSTRATING THE EXACT FIX FOR YOUR PROBLEM ===")

    print("\n📊 YOUR REAL DATA (from diagnostic):")
    print("test01.py: float32, range 0.0 to 407.59 → works fine")
    print("test17.py: uint16, range 0 to 460 → was broken, now FIXED")

    # Simulate your exact uint16 problem
    print(f"\n🔍 SIMULATING YOUR EXACT uint16 PROBLEM:")
    your_uint16_data = np.array([0, 50, 100, 200, 300, 460], dtype=np.uint16)
    print(f"Your uint16 image: {your_uint16_data}")
    print(f"Range: {your_uint16_data.min()} to {your_uint16_data.max()}")

    # OLD broken method (what your test17.py was doing)
    print(f"\n❌ OLD METHOD (what test17.py was doing):")
    old_broken = (your_uint16_data / 256).astype(np.uint8)
    print(f"After /256: {old_broken}")
    print(f"Result range: {old_broken.min()} to {old_broken.max()}")
    print(f"Problem: 460/256 = 1.8 → 1, so max brightness was only 1/255 = VERY DARK!")

    # FIXED method
    print(f"\n✅ FIXED METHOD (what we do now):")
    if your_uint16_data.max() > your_uint16_data.min():
        normalized = (your_uint16_data.astype(np.float32) - your_uint16_data.min()) / (
                    your_uint16_data.max() - your_uint16_data.min())
        fixed_result = (normalized * 255).astype(np.uint8)
    print(f"After proper normalization: {fixed_result}")
    print(f"Result range: {fixed_result.min()} to {fixed_result.max()}")
    print(f"Solution: Full 0-255 range! Now 460 → 255 = BRIGHT!")

    print(f"\n🎉 This simple fix makes your images bright instead of dark!")


def main():
    """Main function with simple fix - no dual connections needed"""
    print("=== SIMPLE FIXED PyScope Camera Interface ===")
    print("Just fixing the normalization issue - no complex connections needed!")

    # First show the fix
    demonstrate_the_exact_fix()

    # Test PyScope imports
    try:
        import pyscope.registry
        import pyscope.config
        logger.info("✓ PyScope imports successful")
    except ImportError as e:
        logger.error(f"PyScope import failed: {e}")
        return

    # Create and connect to camera using context manager
    with SimplePyScope_Camera() as camera:
        if not camera.connect():
            logger.error("Failed to connect to any camera")
            return

        try:
            print(f"\n📸 Testing SIMPLE FIXED Image Capture...")

            # Setup camera for acquisition
            exposure_time_ms = 1000  # 1 second exposure

            print(f"✓ Using simple PyScope configuration")
            if camera.setup_camera_for_acquisition(exposure_time_ms):
                print(f"✓ Camera configured for {exposure_time_ms}ms exposure")
            else:
                print(f"❌ Camera configuration failed")
                return

            # Capture image using the simple method
            image, metadata = camera.capture_image()

            if image is not None:
                print(f"\n🎉 SIMPLE FIXED image capture successful!")
                print(f"  Camera: {metadata['camera_name']} ({metadata['camera_type']})")
                print(f"  Method: {metadata['capture_method']}")
                print(f"  Image shape: {metadata['shape']}")
                print(f"  Data type: {metadata['pixel_format']}")
                print(f"  Original range: [{metadata['min_value']:.1f}, {metadata['max_value']:.1f}]")
                print(f"  Mean ± Std: {metadata['mean_value']:.1f} ± {metadata['std_value']:.1f}")
                print(f"  Capture time: {metadata['capture_time_seconds']:.3f} seconds")

                # Save with FIXED normalization
                print(f"\n💾 Saving image with FIXED normalization...")
                for fmt in ["PNG", "JPG"]:
                    filepath, pil_image = camera.convert_and_save_image_FIXED(image, metadata, fmt)
                    if filepath:
                        print(f"✓ Saved SIMPLE FIXED {fmt}: {os.path.basename(filepath)}")

                # Display the image
                if pil_image:
                    print(f"\n📊 Displaying SIMPLE FIXED image...")
                    camera.display_image_fixed(pil_image, metadata)

                print(f"\n🎉 SUCCESS: SIMPLE FIX applied!")
                print(f"Your image should now be BRIGHT instead of dark!")
                print(f"The fix: Use actual data range instead of /256 division")

            else:
                print(f"\n❌ Image capture failed")

            print(f"\n🎉 SIMPLE FIXED version completed!")
            print(f"Images captured: {camera.capture_count}")

        except Exception as e:
            logger.error(f"Error in SIMPLE FIXED process: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()