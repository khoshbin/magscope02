#!/usr/bin/env python3
"""
Single Connection Bypass - Use PyScope's internal DE methods directly
Since only one connection to DE server is allowed, we'll use PyScope's existing connection
but bypass its getImage() method and use internal DE functions directly
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


class SingleConnectionBypassCamera:
    """
    Use PyScope's existing connection but bypass getImage() to use internal DE methods
    """

    def __init__(self):
        self.pyscope_camera = None
        self.connected = False
        self.capture_count = 0

    def connect(self):
        """Connect only via PyScope"""
        logger.info("=== CONNECTING VIA PYSCOPE ONLY ===")

        try:
            import pyscope.registry
            self.pyscope_camera = pyscope.registry.getClass("DEApollo")()
            camera_size = self.pyscope_camera.getCameraSize()
            logger.info(f"✓ PyScope connected - Camera size: {camera_size}")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"PyScope connection failed: {e}")
            return False

    def setup_exactly_like_test01(self):
        """Setup PyScope exactly like test01.py"""
        logger.info("\n=== SETTING UP LIKE TEST01.PY ===")

        if not self.pyscope_camera:
            return False

        # Properties from test01.py
        test01_properties = {
            "Exposure Mode": "Normal",
            "Image Processing - Mode": "Integrating",
            "Frames Per Second": 20,
            "Exposure Time (seconds)": 1.0,
            "Autosave Final Image": "Off",
            "Autosave Movie": "Off"
        }

        success_count = 0
        total_attempts = len(test01_properties)

        logger.info("Setting PyScope properties:")
        for prop, value in test01_properties.items():
            try:
                self.pyscope_camera.setProperty(prop, value)
                current = self.pyscope_camera.getProperty(prop)
                logger.info(f"  ✓ {prop}: {value} (current: {current})")
                success_count += 1
            except Exception as e:
                logger.warning(f"  ❌ {prop}: {e}")

        # Additional PyScope settings
        try:
            self.pyscope_camera.setExposureTime(1000)  # 1000ms
            logger.info(f"  ✓ setExposureTime: 1000ms")
            success_count += 1
        except Exception as e:
            logger.warning(f"  ❌ setExposureTime: {e}")
        total_attempts += 1

        logger.info(f"Setup complete: {success_count}/{total_attempts} settings applied")
        return success_count > (total_attempts // 2)

    def investigate_internal_methods(self):
        """Investigate PyScope's internal methods and the DE module"""
        logger.info("\n=== INVESTIGATING INTERNAL METHODS ===")

        # Check PyScope camera internals
        logger.info("PyScope camera internal attributes:")
        interesting_attrs = []
        for attr in dir(self.pyscope_camera):
            if not attr.startswith('_') and any(keyword in attr.lower() for keyword in
                                                ['image', 'get', 'acquire', 'capture', 'de_', 'deapi', 'internal']):
                interesting_attrs.append(attr)

        for attr in sorted(interesting_attrs):
            try:
                value = getattr(self.pyscope_camera, attr)
                if callable(value):
                    logger.info(f"  METHOD: {attr}")
                else:
                    logger.info(f"  ATTR: {attr} = {value}")
            except:
                logger.info(f"  ATTR: {attr} = <cannot access>")

        # Check PyScope DE module access
        logger.info("\nChecking PyScope DE module access:")
        try:
            import pyscope.de as de_module
            logger.info("✓ pyscope.de module imported successfully")

            # Check DE module functions
            de_functions = []
            for attr in dir(de_module):
                if not attr.startswith('_') and callable(getattr(de_module, attr)):
                    de_functions.append(attr)

            logger.info(f"Available DE functions: {de_functions}")

            # Check if we can access DE server state
            try:
                if hasattr(de_module, '__deserver'):
                    deserver = getattr(de_module, '__deserver')
                    logger.info(f"DE server object: {deserver}")
                else:
                    logger.info("No __deserver attribute found")
            except:
                logger.info("Cannot access DE server object")

        except Exception as e:
            logger.error(f"Cannot access pyscope.de module: {e}")

        # Check camera model name for DE calls
        if hasattr(self.pyscope_camera, 'model_name'):
            model_name = self.pyscope_camera.model_name
            logger.info(f"Camera model name: {model_name}")
            return model_name
        else:
            logger.warning("No model_name found")
            return None

    def try_all_internal_capture_methods(self):
        """Try every possible internal method to capture images"""
        logger.info("\n=== TRYING ALL INTERNAL CAPTURE METHODS ===")

        results = {}

        # Method 1: Standard PyScope getImage() (for comparison)
        try:
            logger.info("\n--- Method 1: Standard PyScope getImage() ---")
            start_time = time.time()
            image1 = self.pyscope_camera.getImage()
            capture_time1 = time.time() - start_time

            if image1 is not None:
                results['pyscope_getImage'] = {
                    'shape': image1.shape,
                    'dtype': str(image1.dtype),
                    'min': float(image1.min()),
                    'max': float(image1.max()),
                    'mean': float(image1.mean()),
                    'capture_time': capture_time1,
                    'image': image1
                }
                logger.info(f"PyScope getImage(): {image1.shape} {image1.dtype} range=[{image1.min()}, {image1.max()}]")
        except Exception as e:
            logger.error(f"PyScope getImage() failed: {e}")

        # Method 2: PyScope _getImage() if available
        if hasattr(self.pyscope_camera, '_getImage'):
            try:
                logger.info("\n--- Method 2: PyScope _getImage() ---")
                start_time = time.time()
                image2 = self.pyscope_camera._getImage()
                capture_time2 = time.time() - start_time

                if image2 is not None:
                    results['pyscope_private_getImage'] = {
                        'shape': image2.shape,
                        'dtype': str(image2.dtype),
                        'min': float(image2.min()),
                        'max': float(image2.max()),
                        'mean': float(image2.mean()),
                        'capture_time': capture_time2,
                        'image': image2
                    }
                    logger.info(
                        f"PyScope _getImage(): {image2.shape} {image2.dtype} range=[{image2.min()}, {image2.max()}]")
            except Exception as e:
                logger.error(f"PyScope _getImage() failed: {e}")

        # Method 3: Internal DE module de_getImage()
        try:
            logger.info("\n--- Method 3: Internal DE de_getImage() ---")
            import pyscope.de as de_module

            if hasattr(self.pyscope_camera, 'model_name'):
                model_name = self.pyscope_camera.model_name

                start_time = time.time()
                image3 = de_module.de_getImage(model_name)
                capture_time3 = time.time() - start_time

                if image3 is not None:
                    results['internal_de_getImage'] = {
                        'shape': image3.shape,
                        'dtype': str(image3.dtype),
                        'min': float(image3.min()),
                        'max': float(image3.max()),
                        'mean': float(image3.mean()),
                        'capture_time': capture_time3,
                        'image': image3
                    }
                    logger.info(
                        f"Internal DE getImage(): {image3.shape} {image3.dtype} range=[{image3.min()}, {image3.max()}]")
        except Exception as e:
            logger.error(f"Internal DE de_getImage() failed: {e}")

        # Method 4: Try to access DE server directly through PyScope's connection
        try:
            logger.info("\n--- Method 4: Direct DE server access through PyScope ---")
            import pyscope.de as de_module

            # Try to get the DE server object from PyScope's connection
            if hasattr(de_module, '_BypassPyScopeCamera__deserver'):
                deserver = getattr(de_module, '_BypassPyScopeCamera__deserver')
            elif hasattr(de_module, '__deserver'):
                deserver = de_module.__deserver
            else:
                # Try different module-level server variables
                deserver = None
                for attr in dir(de_module):
                    if 'server' in attr.lower():
                        try:
                            deserver = getattr(de_module, attr)
                            if deserver is not None:
                                break
                        except:
                            continue

            if deserver is not None:
                logger.info(f"Found DE server object: {deserver}")

                # Try to get image directly from server
                start_time = time.time()
                image4 = deserver.GetImage('uint16')
                capture_time4 = time.time() - start_time

                if image4 is not None:
                    results['direct_deserver'] = {
                        'shape': image4.shape,
                        'dtype': str(image4.dtype),
                        'min': float(image4.min()),
                        'max': float(image4.max()),
                        'mean': float(image4.mean()),
                        'capture_time': capture_time4,
                        'image': image4
                    }
                    logger.info(
                        f"Direct DE server: {image4.shape} {image4.dtype} range=[{image4.min()}, {image4.max()}]")
            else:
                logger.warning("Could not access DE server object")

        except Exception as e:
            logger.error(f"Direct DE server access failed: {e}")

        # Method 5: Try using StartAcquisition/GetResult through PyScope's DE connection
        try:
            logger.info("\n--- Method 5: StartAcquisition through PyScope DE connection ---")
            import pyscope.de as de_module
            from pyscope import DEAPI

            # Try to access the server and do acquisition like test01.py
            if hasattr(de_module, '__deserver') and de_module.__deserver is not None:
                deserver = de_module.__deserver

                # Set active camera
                if hasattr(self.pyscope_camera, 'model_name'):
                    model_name = self.pyscope_camera.model_name
                    de_module.de_setActiveCamera(model_name)

                # Start acquisition like test01.py
                deserver.StartAcquisition(1)

                frameType = DEAPI.FrameType.SUMTOTAL
                pixelFormat = DEAPI.PixelFormat.AUTO
                attributes = DEAPI.Attributes()
                histogram = DEAPI.Histogram()

                start_time = time.time()
                image5, pixelFormat, attributes, histogram = deserver.GetResult(
                    frameType, pixelFormat, attributes, histogram
                )
                capture_time5 = time.time() - start_time

                if image5 is not None:
                    results['pyscope_de_acquisition'] = {
                        'shape': image5.shape,
                        'dtype': str(image5.dtype),
                        'min': float(image5.min()),
                        'max': float(image5.max()),
                        'mean': float(image5.mean()),
                        'capture_time': capture_time5,
                        'image': image5,
                        'pixel_format': str(pixelFormat),
                        'dataset': attributes.datasetName
                    }
                    logger.info(
                        f"PyScope DE acquisition: {image5.shape} {image5.dtype} range=[{image5.min()}, {image5.max()}]")
                    logger.info(f"  Dataset: {attributes.datasetName}, Pixel format: {pixelFormat}")

        except Exception as e:
            logger.error(f"PyScope DE acquisition failed: {e}")

        return results

    def analyze_results(self, results):
        """Analyze all capture results to find the best method"""
        logger.info("\n=== ANALYZING RESULTS ===")

        if not results:
            logger.error("No successful capture methods found")
            return None, None

        # Print summary
        logger.info("All capture results:")
        for method, data in results.items():
            logger.info(
                f"  {method:25s}: {data['shape']} {data['dtype']} range=[{data['min']:.1f}, {data['max']:.1f}] time={data['capture_time']:.3f}s")

        # Find methods that produce 4096x4096
        correct_methods = []
        for method, data in results.items():
            if data['shape'] == (4096, 4096):
                correct_methods.append(method)

        if correct_methods:
            logger.info(f"\n✓ METHODS PRODUCING 4096x4096: {correct_methods}")

            # Priority order for choosing best method
            priority_order = [
                'pyscope_de_acquisition',  # Most like test01.py
                'internal_de_getImage',  # Direct DE method
                'direct_deserver',  # Direct server access
                'pyscope_private_getImage',  # Internal PyScope method
                'pyscope_getImage'  # Standard method (last resort)
            ]

            # Choose the best available method
            best_method = None
            for preferred in priority_order:
                if preferred in correct_methods:
                    best_method = preferred
                    break

            if best_method:
                logger.info(f"\n🎉 BEST METHOD FOUND: {best_method}")
                return results[best_method]['image'], {
                    'method': best_method,
                    'shape': results[best_method]['shape'],
                    'dtype': results[best_method]['dtype'],
                    'min_value': results[best_method]['min'],
                    'max_value': results[best_method]['max'],
                    'mean_value': results[best_method]['mean'],
                    'capture_time_seconds': results[best_method]['capture_time'],
                    'timestamp': datetime.now().isoformat()
                }
        else:
            logger.warning(f"\n⚠️  NO METHOD PRODUCED 4096x4096")

            # Check if all methods produce the same wrong size
            shapes = set(data['shape'] for data in results.values())
            if len(shapes) == 1:
                shape = list(shapes)[0]
                logger.warning(
                    f"All methods produce {shape} - this suggests camera hardware is configured for this size")

                # Return the best available method anyway
                if 'pyscope_de_acquisition' in results:
                    best_method = 'pyscope_de_acquisition'
                elif 'internal_de_getImage' in results:
                    best_method = 'internal_de_getImage'
                else:
                    best_method = list(results.keys())[0]

                logger.info(f"Returning best available method: {best_method}")
                return results[best_method]['image'], {
                    'method': best_method,
                    'shape': results[best_method]['shape'],
                    'dtype': results[best_method]['dtype'],
                    'min_value': results[best_method]['min'],
                    'max_value': results[best_method]['max'],
                    'mean_value': results[best_method]['mean'],
                    'capture_time_seconds': results[best_method]['capture_time'],
                    'timestamp': datetime.now().isoformat()
                }

        return None, None

    def save_working_image(self, image: np.ndarray, metadata: Dict[str, Any],
                           output_format: str = "PNG", output_dir: str = "working_images") -> Tuple[
        Optional[str], Optional[Image.Image]]:
        """Save image using test01.py normalization"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Normalize exactly like test01.py
            if image.dtype == np.float32 or image.dtype == np.float64:
                image_normalized = (image - image.min()) / (image.max() - image.min())
                image_8bit = (image_normalized * 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                image_8bit = (image / 256).astype(np.uint8)
            elif image.dtype == np.uint8:
                image_8bit = image
            else:
                image_normalized = (image - image.min()) / (image.max() - image.min())
                image_8bit = (image_normalized * 255).astype(np.uint8)

            pil_image = Image.fromarray(image_8bit, mode='L')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method = metadata.get('method', 'unknown').replace('_', '-')
            filename = f"WORKING_{method}_{timestamp}.{output_format.lower()}"
            filepath = os.path.join(output_dir, filename)

            if output_format.upper() == "JPG":
                pil_image.save(filepath, "JPEG", quality=95)
            else:
                pil_image.save(filepath, "PNG")

            logger.info(f"✓ Working image saved: {filepath}")
            return filepath, pil_image

        except Exception as e:
            logger.error(f"Error saving working image: {e}")
            return None, None

    def disconnect(self):
        """Disconnect from camera"""
        try:
            if self.pyscope_camera and hasattr(self.pyscope_camera, 'disconnect'):
                self.pyscope_camera.disconnect()
                logger.info("Disconnected from PyScope")
        except Exception as e:
            logger.debug(f"Error disconnecting: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    """Main function to find working capture method"""
    print("=== SINGLE CONNECTION BYPASS - FIND WORKING METHOD ===")
    print("Using PyScope's connection but bypassing getImage() to find working capture method")

    with SingleConnectionBypassCamera() as camera:
        if not camera.connect():
            logger.error("Failed to connect")
            return

        try:
            # Setup like test01.py
            if camera.setup_exactly_like_test01():
                logger.info("✓ Camera setup completed")
            else:
                logger.warning("Camera setup had issues but continuing...")

            # Investigate internal methods
            model_name = camera.investigate_internal_methods()

            # Try all internal capture methods
            results = camera.try_all_internal_capture_methods()

            # Analyze results to find best method
            best_image, best_metadata = camera.analyze_results(results)

            if best_image is not None:
                print(f"\n🎉 WORKING METHOD FOUND!")
                print(f"  Method: {best_metadata['method']}")
                print(f"  Shape: {best_metadata['shape']}")
                print(f"  Data type: {best_metadata['dtype']}")
                print(f"  Value range: [{best_metadata['min_value']:.1f}, {best_metadata['max_value']:.1f}]")
                print(f"  Mean: {best_metadata['mean_value']:.1f}")
                print(f"  Capture time: {best_metadata['capture_time_seconds']:.3f} seconds")

                # Save the working image
                print(f"\n💾 Saving working image...")
                for fmt in ["PNG", "JPG"]:
                    filepath, pil_image = camera.save_working_image(best_image, best_metadata, fmt)
                    if filepath:
                        print(f"✓ Saved {fmt}: {os.path.basename(filepath)}")

                # Save metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata_file = f"working_images/WORKING_metadata_{timestamp}.json"
                os.makedirs("working_images", exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(best_metadata, f, indent=2, default=str)
                print(f"✓ Saved metadata: {os.path.basename(metadata_file)}")

                print(f"\n✅ SOLUTION:")
                print(f"Replace camera.getImage() with the working method: {best_metadata['method']}")

                if best_metadata['shape'] == (4096, 4096):
                    print(f"🎯 SUCCESS: This method produces 4096x4096 images like test01.py!")
                else:
                    print(f"⚠️  Note: This method produces {best_metadata['shape']} images")
                    print(f"The camera may be hardware-configured for this size")

            else:
                print(f"\n❌ No working capture method found")
                print(f"All available methods failed or produced unexpected results")

        except Exception as e:
            logger.error(f"Error in bypass process: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()