#!/usr/bin/env python3
"""
Diagnostic PyScope DE Camera Image Capture with test01.py comparison
Identifies differences in properties, data types, and normalization
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

# Add DEAPI path for direct comparison
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]


class DiagnosticPyScope_Camera:
    """
    Diagnostic version that compares PyScope getImage() with direct DEAPI
    """

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.camera_type = None
        self.direct_client = None

    def connect(self, camera_name=None):
        """Connect to PyScope camera and optionally direct DEAPI"""
        try:
            import pyscope.registry

            # Connect to PyScope camera
            target_camera = camera_name or self.camera_name or "DEApollo"

            logger.info(f"Connecting to PyScope camera: {target_camera}")
            self.camera = pyscope.registry.getClass(target_camera)()
            self.camera_name = target_camera
            self.camera_type = 'DE'

            # Test basic connectivity
            camera_size = self.camera.getCameraSize()
            logger.info(f"✓ PyScope connected to {target_camera}")
            logger.info(f"  Camera size: {camera_size}")

            self.connected = True

            # Also try to connect direct DEAPI for comparison
            self._connect_direct_deapi()

            return True

        except Exception as e:
            logger.error(f"PyScope connection failed: {e}")
            return False

    def _connect_direct_deapi(self):
        """Connect to direct DEAPI like test01.py"""
        try:
            from pyscope import DEAPI

            self.direct_client = DEAPI.Client()
            self.direct_client.Connect()

            cameras = self.direct_client.ListCameras()
            if cameras:
                self.direct_client.SetCurrentCamera(cameras[0])
                logger.info(f"✓ Direct DEAPI connected to: {cameras[0]}")
            else:
                logger.warning("No cameras found via direct DEAPI")
                self.direct_client = None

        except Exception as e:
            logger.warning(f"Direct DEAPI connection failed: {e}")
            self.direct_client = None

    def compare_properties(self):
        """Compare properties between PyScope and direct DEAPI"""
        logger.info("\n" + "=" * 60)
        logger.info("PROPERTY COMPARISON")
        logger.info("=" * 60)

        # Get PyScope properties
        pyscope_props = {}
        if self.camera:
            try:
                # Try to get key imaging properties from PyScope
                key_props = [
                    'Exposure Time (seconds)', 'Frames Per Second', 'Image Processing - Mode',
                    'Exposure Mode', 'Autosave Final Image', 'Autosave Movie',
                    'Sensor Size X (pixels)', 'Sensor Size Y (pixels)'
                ]

                for prop in key_props:
                    try:
                        if hasattr(self.camera, 'getProperty'):
                            value = self.camera.getProperty(prop)
                            pyscope_props[prop] = value
                    except:
                        pass

                # Also get standard PyScope settings
                try:
                    pyscope_props['PyScope_ExposureTime_ms'] = self.camera.getExposureTime()
                except:
                    pass
                try:
                    pyscope_props['PyScope_Binning'] = self.camera.getBinning()
                except:
                    pass
                try:
                    pyscope_props['PyScope_Dimension'] = self.camera.getDimension()
                except:
                    pass

            except Exception as e:
                logger.error(f"Error getting PyScope properties: {e}")

        # Get Direct DEAPI properties
        direct_props = {}
        if self.direct_client:
            try:
                key_props = [
                    'Exposure Time (seconds)', 'Frames Per Second', 'Image Processing - Mode',
                    'Exposure Mode', 'Autosave Final Image', 'Autosave Movie',
                    'Sensor Size X (pixels)', 'Sensor Size Y (pixels)'
                ]

                for prop in key_props:
                    try:
                        value = self.direct_client.GetProperty(prop)
                        direct_props[prop] = value
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error getting direct DEAPI properties: {e}")

        # Compare properties
        logger.info("PyScope Properties:")
        for key, value in pyscope_props.items():
            logger.info(f"  {key}: {value}")

        logger.info("\nDirect DEAPI Properties:")
        for key, value in direct_props.items():
            logger.info(f"  {key}: {value}")

        # Find differences
        differences = []
        for key in set(pyscope_props.keys()) | set(direct_props.keys()):
            pyscope_val = pyscope_props.get(key, "NOT_SET")
            direct_val = direct_props.get(key, "NOT_SET")
            if pyscope_val != direct_val:
                differences.append((key, pyscope_val, direct_val))

        if differences:
            logger.warning(f"\n⚠️  Found {len(differences)} property differences:")
            for key, pyscope_val, direct_val in differences:
                logger.warning(f"  {key}: PyScope={pyscope_val} vs Direct={direct_val}")
        else:
            logger.info("\n✓ No property differences found")

        return pyscope_props, direct_props, differences

    def setup_camera_like_test01(self):
        """Setup camera with same properties as test01.py"""
        logger.info("\n" + "=" * 60)
        logger.info("SETTING UP CAMERA LIKE TEST01.PY")
        logger.info("=" * 60)

        success_count = 0
        total_attempts = 0

        # Properties from test01.py
        test01_properties = {
            "Exposure Mode": "Normal",
            "Image Processing - Mode": "Integrating",
            "Frames Per Second": 20,
            "Exposure Time (seconds)": 1.0,
            "Autosave Final Image": "Off",
            "Autosave Movie": "Off"
        }

        # Apply to PyScope camera
        if self.camera and hasattr(self.camera, 'setProperty'):
            logger.info("Setting PyScope camera properties:")
            for prop, value in test01_properties.items():
                total_attempts += 1
                try:
                    self.camera.setProperty(prop, value)
                    current_value = self.camera.getProperty(prop)
                    logger.info(f"  ✓ {prop}: {value} (current: {current_value})")
                    success_count += 1
                except Exception as e:
                    logger.warning(f"  ❌ {prop}: {value} - Failed: {e}")

        # Also try PyScope standard methods
        try:
            self.camera.setExposureTime(1000)  # 1000ms = 1 second
            logger.info(f"  ✓ PyScope setExposureTime: 1000ms")
            success_count += 1
        except Exception as e:
            logger.warning(f"  ❌ PyScope setExposureTime failed: {e}")
        total_attempts += 1

        # Apply to direct DEAPI camera for comparison
        if self.direct_client:
            logger.info("\nSetting Direct DEAPI camera properties:")
            for prop, value in test01_properties.items():
                try:
                    self.direct_client.SetProperty(prop, value)
                    current_value = self.direct_client.GetProperty(prop)
                    logger.info(f"  ✓ {prop}: {value} (current: {current_value})")
                except Exception as e:
                    logger.warning(f"  ❌ {prop}: {value} - Failed: {e}")

        logger.info(f"\nPyScope setup: {success_count}/{total_attempts} properties set successfully")
        return success_count > 0

    def capture_image_pyscope_diagnostic(self):
        """Capture image using PyScope with detailed diagnostics"""
        logger.info("\n" + "=" * 60)
        logger.info("PYSCOPE IMAGE CAPTURE (DIAGNOSTIC)")
        logger.info("=" * 60)

        if not self.camera:
            logger.error("No PyScope camera available")
            return None, None

        try:
            logger.info("Starting PyScope image acquisition...")

            # Get pre-capture state
            try:
                pre_exposure = self.camera.getExposureTime()
                pre_binning = self.camera.getBinning()
                pre_dimension = self.camera.getDimension()
                logger.info(f"Pre-capture state:")
                logger.info(f"  Exposure: {pre_exposure} ms")
                logger.info(f"  Binning: {pre_binning}")
                logger.info(f"  Dimension: {pre_dimension}")
            except Exception as e:
                logger.warning(f"Could not get pre-capture state: {e}")

            # Capture image
            start_time = time.time()
            image = self.camera.getImage()
            capture_time = time.time() - start_time

            if image is not None:
                # Detailed image analysis
                logger.info(f"✓ PyScope image captured successfully!")
                logger.info(f"  Shape: {image.shape}")
                logger.info(f"  Data type: {image.dtype}")
                logger.info(f"  Size in bytes: {image.nbytes}")
                logger.info(f"  Min value: {image.min()}")
                logger.info(f"  Max value: {image.max()}")
                logger.info(f"  Mean value: {image.mean():.3f}")
                logger.info(f"  Std deviation: {image.std():.3f}")
                logger.info(f"  Capture time: {capture_time:.3f} seconds")

                # Check for common issues
                if image.max() == image.min():
                    logger.warning("⚠️  Image has constant values (likely all zeros or constant)")
                if image.max() < 100:
                    logger.warning("⚠️  Image has very low values (possible underexposure)")
                if image.dtype != np.uint16:
                    logger.warning(f"⚠️  Unexpected data type: {image.dtype} (expected uint16)")

                # Create metadata
                metadata = {
                    'method': 'PyScope_getImage',
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'min_value': float(image.min()),
                    'max_value': float(image.max()),
                    'mean_value': float(image.mean()),
                    'std_value': float(image.std()),
                    'capture_time_seconds': capture_time,
                    'timestamp': datetime.now().isoformat()
                }

                return image, metadata
            else:
                logger.error("PyScope returned None for image")
                return None, None

        except Exception as e:
            logger.error(f"Error during PyScope image capture: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def capture_image_direct_like_test01(self):
        """Capture image using direct DEAPI like test01.py"""
        logger.info("\n" + "=" * 60)
        logger.info("DIRECT DEAPI IMAGE CAPTURE (LIKE TEST01.PY)")
        logger.info("=" * 60)

        if not self.direct_client:
            logger.warning("No direct DEAPI client available")
            return None, None

        try:
            from pyscope import DEAPI

            logger.info("Starting direct DEAPI acquisition...")

            # Start acquisition exactly like test01.py
            self.direct_client.StartAcquisition(1)

            # Get the result exactly like test01.py
            frameType = DEAPI.FrameType.SUMTOTAL
            pixelFormat = DEAPI.PixelFormat.AUTO
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()

            start_time = time.time()
            image, pixelFormat, attributes, histogram = self.direct_client.GetResult(
                frameType, pixelFormat, attributes, histogram
            )
            capture_time = time.time() - start_time

            if image is not None:
                logger.info(f"✓ Direct DEAPI image captured successfully!")
                logger.info(f"  Shape: {image.shape}")
                logger.info(f"  Data type: {image.dtype}")
                logger.info(f"  Size in bytes: {image.nbytes}")
                logger.info(f"  Pixel format: {pixelFormat}")
                logger.info(f"  Dataset: {attributes.datasetName}")
                logger.info(f"  Min value: {attributes.imageMin}")
                logger.info(f"  Max value: {attributes.imageMax}")
                logger.info(f"  Mean value: {attributes.imageMean:.3f}")
                logger.info(f"  Std deviation: {attributes.imageStd:.3f}")
                logger.info(f"  Capture time: {capture_time:.3f} seconds")

                # Check for differences with numpy stats
                numpy_min = float(image.min())
                numpy_max = float(image.max())
                numpy_mean = float(image.mean())
                numpy_std = float(image.std())

                if abs(numpy_min - attributes.imageMin) > 0.1:
                    logger.warning(f"⚠️  Min value mismatch: numpy={numpy_min} vs attributes={attributes.imageMin}")
                if abs(numpy_max - attributes.imageMax) > 0.1:
                    logger.warning(f"⚠️  Max value mismatch: numpy={numpy_max} vs attributes={attributes.imageMax}")
                if abs(numpy_mean - attributes.imageMean) > 0.1:
                    logger.warning(f"⚠️  Mean value mismatch: numpy={numpy_mean} vs attributes={attributes.imageMean}")

                # Create metadata
                metadata = {
                    'method': 'Direct_DEAPI',
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'min_value': float(numpy_min),
                    'max_value': float(numpy_max),
                    'mean_value': float(numpy_mean),
                    'std_value': float(numpy_std),
                    'attributes_min': attributes.imageMin,
                    'attributes_max': attributes.imageMax,
                    'attributes_mean': attributes.imageMean,
                    'attributes_std': attributes.imageStd,
                    'pixel_format': str(pixelFormat),
                    'dataset_name': attributes.datasetName,
                    'capture_time_seconds': capture_time,
                    'timestamp': datetime.now().isoformat()
                }

                return image, metadata
            else:
                logger.error("Direct DEAPI returned None for image")
                return None, None

        except Exception as e:
            logger.error(f"Error during direct DEAPI image capture: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def compare_images(self, pyscope_image, pyscope_metadata, direct_image, direct_metadata):
        """Compare the two captured images"""
        logger.info("\n" + "=" * 60)
        logger.info("IMAGE COMPARISON")
        logger.info("=" * 60)

        if pyscope_image is None or direct_image is None:
            logger.error("Cannot compare - one or both images are None")
            return

        # Shape comparison
        logger.info(f"Shape comparison:")
        logger.info(f"  PyScope: {pyscope_image.shape}")
        logger.info(f"  Direct:  {direct_image.shape}")

        # Data type comparison
        logger.info(f"Data type comparison:")
        logger.info(f"  PyScope: {pyscope_image.dtype}")
        logger.info(f"  Direct:  {direct_image.dtype}")

        # Value range comparison
        logger.info(f"Value range comparison:")
        logger.info(f"  PyScope: [{pyscope_image.min():.3f}, {pyscope_image.max():.3f}]")
        logger.info(f"  Direct:  [{direct_image.min():.3f}, {direct_image.max():.3f}]")

        # Statistics comparison
        logger.info(f"Statistics comparison:")
        logger.info(f"  PyScope: mean={pyscope_image.mean():.3f}, std={pyscope_image.std():.3f}")
        logger.info(f"  Direct:  mean={direct_image.mean():.3f}, std={direct_image.std():.3f}")

        # Pixel-by-pixel comparison if same shape
        if pyscope_image.shape == direct_image.shape:
            try:
                # Convert to same type for comparison
                if pyscope_image.dtype != direct_image.dtype:
                    logger.warning(
                        f"Converting data types for comparison: {pyscope_image.dtype} vs {direct_image.dtype}")
                    # Convert both to float for comparison
                    img1_float = pyscope_image.astype(np.float64)
                    img2_float = direct_image.astype(np.float64)
                else:
                    img1_float = pyscope_image.astype(np.float64)
                    img2_float = direct_image.astype(np.float64)

                # Calculate differences
                diff = img1_float - img2_float
                abs_diff = np.abs(diff)

                logger.info(f"Pixel difference analysis:")
                logger.info(f"  Mean absolute difference: {abs_diff.mean():.3f}")
                logger.info(f"  Max absolute difference: {abs_diff.max():.3f}")
                logger.info(
                    f"  Pixels with zero difference: {np.sum(abs_diff == 0)}/{abs_diff.size} ({100 * np.sum(abs_diff == 0) / abs_diff.size:.1f}%)")

                if abs_diff.max() == 0:
                    logger.info("  ✓ Images are IDENTICAL pixel-by-pixel")
                elif abs_diff.mean() < 1.0:
                    logger.info("  ✓ Images are very similar (mean diff < 1.0)")
                elif abs_diff.mean() < 10.0:
                    logger.warning("  ⚠️  Images have moderate differences (mean diff < 10.0)")
                else:
                    logger.error("  ❌ Images are significantly different (mean diff >= 10.0)")

                # Check for scaling differences
                if pyscope_image.max() > 0 and direct_image.max() > 0:
                    scale_factor = direct_image.max() / pyscope_image.max()
                    logger.info(f"  Max value ratio (Direct/PyScope): {scale_factor:.6f}")

                    if abs(scale_factor - 1.0) < 0.01:
                        logger.info("    ✓ No significant scaling difference")
                    else:
                        logger.warning(f"    ⚠️  Significant scaling difference detected")

                        # Test if images are scaled versions
                        scaled_pyscope = img1_float * scale_factor
                        scaled_diff = np.abs(scaled_pyscope - img2_float)
                        if scaled_diff.mean() < abs_diff.mean() / 2:
                            logger.info(f"    ✓ Images appear to be scaled versions (factor: {scale_factor:.6f})")

            except Exception as e:
                logger.error(f"Error in pixel comparison: {e}")
        else:
            logger.warning("Cannot do pixel comparison - different shapes")

    def save_comparison_images(self, pyscope_image, pyscope_metadata, direct_image, direct_metadata,
                               output_dir="comparison_images"):
        """Save both images with identical normalization for visual comparison"""
        logger.info("\n" + "=" * 60)
        logger.info("SAVING COMPARISON IMAGES")
        logger.info("=" * 60)

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Function to normalize image like test01.py
            def normalize_like_test01(image):
                """Normalize exactly like test01.py"""
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
                return image_8bit

            # Save PyScope image
            if pyscope_image is not None:
                pyscope_8bit = normalize_like_test01(pyscope_image)
                pyscope_pil = Image.fromarray(pyscope_8bit, mode='L')
                pyscope_filename = f"comparison_pyscope_{timestamp}.png"
                pyscope_filepath = os.path.join(output_dir, pyscope_filename)
                pyscope_pil.save(pyscope_filepath, "PNG")
                logger.info(f"✓ PyScope image saved: {pyscope_filename}")

            # Save Direct DEAPI image
            if direct_image is not None:
                direct_8bit = normalize_like_test01(direct_image)
                direct_pil = Image.fromarray(direct_8bit, mode='L')
                direct_filename = f"comparison_direct_{timestamp}.png"
                direct_filepath = os.path.join(output_dir, direct_filename)
                direct_pil.save(direct_filepath, "PNG")
                logger.info(f"✓ Direct DEAPI image saved: {direct_filename}")

            # Create side-by-side comparison
            if pyscope_image is not None and direct_image is not None:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                    # PyScope image
                    axes[0].imshow(normalize_like_test01(pyscope_image), cmap='gray')
                    axes[0].set_title(
                        f"PyScope getImage()\nShape: {pyscope_image.shape}\nType: {pyscope_image.dtype}\nRange: [{pyscope_image.min():.1f}, {pyscope_image.max():.1f}]")
                    axes[0].axis('off')

                    # Direct DEAPI image
                    axes[1].imshow(normalize_like_test01(direct_image), cmap='gray')
                    axes[1].set_title(
                        f"Direct DEAPI (test01.py style)\nShape: {direct_image.shape}\nType: {direct_image.dtype}\nRange: [{direct_image.min():.1f}, {direct_image.max():.1f}]")
                    axes[1].axis('off')

                    plt.tight_layout()
                    comparison_filename = f"comparison_sidebyside_{timestamp}.png"
                    comparison_filepath = os.path.join(output_dir, comparison_filename)
                    plt.savefig(comparison_filepath, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"✓ Side-by-side comparison saved: {comparison_filename}")

                except Exception as e:
                    logger.error(f"Error creating side-by-side comparison: {e}")

        except Exception as e:
            logger.error(f"Error saving comparison images: {e}")

    def disconnect(self):
        """Disconnect from cameras"""
        try:
            if self.direct_client:
                self.direct_client.Disconnect()
                logger.info("Disconnected from direct DEAPI")

            if self.camera and hasattr(self.camera, 'disconnect'):
                self.camera.disconnect()
                logger.info("Disconnected from PyScope camera")

            self.connected = False

        except Exception as e:
            logger.debug(f"Error during disconnect: {e}")


def main():
    """Main diagnostic function"""
    print("=== DIAGNOSTIC: PyScope vs test01.py Image Capture Comparison ===")

    # Test PyScope imports
    try:
        import pyscope.registry
        logger.info("✓ PyScope imports successful")
    except ImportError as e:
        logger.error(f"PyScope import failed: {e}")
        return

    # Test DEAPI imports
    try:
        from pyscope import DEAPI
        logger.info("✓ DEAPI imports successful")
    except ImportError as e:
        logger.error(f"DEAPI import failed: {e}")
        return

    # Create diagnostic camera interface
    camera = DiagnosticPyScope_Camera()

    try:
        # Connect to camera
        if not camera.connect("DEApollo"):
            logger.error("Failed to connect to camera")
            return

        # Step 1: Compare initial properties
        pyscope_props, direct_props, differences = camera.compare_properties()

        # Step 2: Setup camera like test01.py
        camera.setup_camera_like_test01()

        # Step 3: Compare properties after setup
        logger.info("\nProperties after setup:")
        pyscope_props_after, direct_props_after, differences_after = camera.compare_properties()

        # Step 4: Capture images using both methods
        pyscope_image, pyscope_metadata = camera.capture_image_pyscope_diagnostic()
        direct_image, direct_metadata = camera.capture_image_direct_like_test01()

        # Step 5: Compare the captured images
        if pyscope_image is not None and direct_image is not None:
            camera.compare_images(pyscope_image, pyscope_metadata, direct_image, direct_metadata)

            # Step 6: Save comparison images
            camera.save_comparison_images(pyscope_image, pyscope_metadata, direct_image, direct_metadata)

        elif pyscope_image is not None:
            logger.warning("Only PyScope image captured successfully")
        elif direct_image is not None:
            logger.warning("Only Direct DEAPI image captured successfully")
        else:
            logger.error("Both image captures failed")

        # Step 7: Summary report
        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 60)

        if differences:
            logger.warning(f"Found {len(differences)} property differences that may affect image capture:")
            for key, pyscope_val, direct_val in differences[:5]:  # Show first 5
                logger.warning(f"  {key}: PyScope={pyscope_val} vs Direct={direct_val}")
        else:
            logger.info("No significant property differences found")

        if pyscope_image is not None and direct_image is not None:
            # Quick comparison summary
            same_shape = pyscope_image.shape == direct_image.shape
            same_dtype = pyscope_image.dtype == direct_image.dtype
            similar_range = abs(pyscope_image.max() - direct_image.max()) < (direct_image.max() * 0.1)

            logger.info(f"Image comparison summary:")
            logger.info(f"  Same shape: {same_shape}")
            logger.info(f"  Same data type: {same_dtype}")
            logger.info(f"  Similar value range: {similar_range}")

            if same_shape and same_dtype and similar_range:
                logger.info("✓ Images appear to be very similar - differences likely minor")
            else:
                logger.warning("⚠️  Images have significant differences - check property settings and normalization")

        logger.info("\nRecommendations:")
        if differences:
            logger.info("1. Check property differences above - ensure PyScope uses same settings as test01.py")
        if pyscope_image is not None and direct_image is not None:
            if pyscope_image.dtype != direct_image.dtype:
                logger.info("2. Data type mismatch detected - check PyScope vs DEAPI image format handling")
            if abs(pyscope_image.max() - direct_image.max()) > (direct_image.max() * 0.1):
                logger.info("3. Value range difference detected - check normalization and scaling")
        logger.info("4. Review saved comparison images to visually inspect differences")

    except Exception as e:
        logger.error(f"Error in diagnostic process: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()