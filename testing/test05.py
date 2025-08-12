#!/usr/bin/env python3
"""
Enhanced PyScope DE Apollo Camera Script
Improved version with better error handling and configuration management
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time

# Configure logging to filter out DE camera property warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DE camera property warnings that are harmless
de_logger = logging.getLogger('DEClient')
de_logger.setLevel(logging.WARNING)


class EnhancedDECamera:
    """Enhanced wrapper for DE cameras via PyScope with better error handling"""

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False

    def connect(self):
        """Connect to DE camera with improved error handling"""
        try:
            import pyscope.registry

            # Try different camera names in order of preference
            camera_attempts = []

            if self.camera_name:
                camera_attempts.append(self.camera_name)

            # Add known DE camera names
            camera_attempts.extend([
                "DEApollo",
                "DE Apollo",
                "DE12",
                "Apollo"
            ])

            for attempt_name in camera_attempts:
                try:
                    logger.info(f"Attempting to connect to camera: {attempt_name}")

                    # Create camera instance using PyScope registry
                    self.camera = pyscope.registry.getClass(attempt_name)()
                    self.camera_name = attempt_name

                    # Test basic connectivity
                    camera_size = self.camera.getCameraSize()
                    logger.info(f"✓ Connected to {attempt_name}")
                    logger.info(f"  Camera size: {camera_size}")

                    self.connected = True
                    return True

                except Exception as e:
                    logger.debug(f"  Failed to connect to {attempt_name}: {e}")
                    continue

            logger.error("All DE camera connection attempts failed")
            return False

        except Exception as e:
            logger.error(f"Error in camera connection process: {e}")
            return False

    def configure_for_single_shot(self, exposure_ms=1000, dimension=None):
        """Configure camera with better property handling"""
        if not self.connected or not self.camera:
            logger.error("Camera not connected")
            return False

        try:
            logger.info("Configuring camera for single shot...")

            # Set exposure time
            self.camera.setExposureTime(exposure_ms)
            logger.info(f"  ✓ Exposure time set to {exposure_ms}ms")

            # Set exposure type
            self.camera.setExposureType('normal')
            logger.info("  ✓ Exposure type set to normal")

            # Set binning (this usually works)
            self.camera.setBinning({'x': 1, 'y': 1})
            logger.info("  ✓ Binning set to 1x1")

            # Handle dimension setting more carefully
            if dimension is None:
                try:
                    camera_size = self.camera.getCameraSize()
                    # Use a smaller test region for faster capture
                    dimension = {
                        'x': min(1024, camera_size.get('x', 1024)),
                        'y': min(1024, camera_size.get('y', 1024))
                    }
                except:
                    dimension = {'x': 1024, 'y': 1024}

            # Try to set dimension, but don't fail if it doesn't work
            try:
                self.camera.setDimension(dimension)
                logger.info(f"  ✓ Dimension set to {dimension}")
            except Exception as e:
                logger.warning(f"  Could not set dimension: {e}")
                logger.info(f"  Will use camera default dimension")

            # Try to set offset, but don't fail if it doesn't work
            try:
                self.camera.setOffset({'x': 0, 'y': 0})
                logger.info("  ✓ Offset set to 0,0")
            except Exception as e:
                logger.warning(f"  Could not set offset: {e}")

            # Validate geometry if available
            try:
                if hasattr(self.camera, 'validateGeometry'):
                    is_valid = self.camera.validateGeometry()
                    logger.info(f"  ✓ Geometry validation: {is_valid}")
            except Exception as e:
                logger.debug(f"  Geometry validation not available: {e}")

            logger.info("Camera configuration completed!")
            return True

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
            return False

    def capture_image(self):
        """Capture image with enhanced error handling and timing"""
        if not self.connected or not self.camera:
            logger.error("Camera not connected")
            return None, None

        try:
            logger.info("Starting image acquisition...")

            # Log current settings for debugging
            try:
                current_settings = {
                    'exposure': self.camera.getExposureTime(),
                    'binning': self.camera.getBinning(),
                    'dimension': self.camera.getDimension(),
                    'offset': self.camera.getOffset()
                }
                logger.info(f"  Current settings: {current_settings}")
            except Exception as e:
                logger.debug(f"  Could not read all current settings: {e}")

            # Capture image with timing
            start_time = time.time()
            image = self.camera.getImage()
            end_time = time.time()

            if image is None:
                logger.error("Camera returned None image")
                return None, None

            acquisition_time = end_time - start_time
            logger.info(f"✓ Image captured successfully in {acquisition_time:.3f}s")
            logger.info(f"  Image shape: {image.shape}")
            logger.info(f"  Image dtype: {image.dtype}")
            logger.info(f"  Min/Max values: {np.min(image):.1f}/{np.max(image):.1f}")
            logger.info(f"  Mean/Std: {np.mean(image):.1f}/{np.std(image):.1f}")

            # Create attributes for compatibility
            attributes = {
                'datasetName': f'pyscope_{self.camera_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'imageMin': float(np.min(image)),
                'imageMax': float(np.max(image)),
                'imageMean': float(np.mean(image)),
                'imageStd': float(np.std(image)),
                'acquisitionTime': acquisition_time
            }

            return image, attributes

        except Exception as e:
            logger.error(f"Error during image capture: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def get_camera_info(self):
        """Get comprehensive camera information"""
        if not self.connected or not self.camera:
            return {}

        info = {'name': self.camera_name}

        test_methods = [
            ('getExposureTime', 'exposure_time'),
            ('getExposureType', 'exposure_type'),
            ('getBinning', 'binning'),
            ('getDimension', 'dimension'),
            ('getOffset', 'offset'),
            ('getCameraSize', 'camera_size'),
            ('getPixelSize', 'pixel_size'),
            ('getExposureTypes', 'exposure_types'),
            ('getRetractable', 'retractable')
        ]

        for method_name, info_key in test_methods:
            try:
                if hasattr(self.camera, method_name):
                    method = getattr(self.camera, method_name)
                    result = method()
                    info[info_key] = result
            except Exception as e:
                logger.debug(f"Could not get {method_name}: {e}")

        return info


def save_image(image, attributes, output_format="PNG", output_dir="captured_images_enhanced"):
    """Save image with improved normalization"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Improved normalization handling
        if image.dtype in [np.float32, np.float64]:
            # For float images
            if np.max(image) > np.min(image):
                image_normalized = (image - image.min()) / (image.max() - image.min())
            else:
                image_normalized = np.zeros_like(image)
            image_8bit = (image_normalized * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            # For 16-bit images, use percentile scaling for better contrast
            p1, p99 = np.percentile(image, [1, 99])
            image_clipped = np.clip(image, p1, p99)
            if p99 > p1:
                image_normalized = (image_clipped - p1) / (p99 - p1)
            else:
                image_normalized = np.zeros_like(image_clipped, dtype=np.float32)
            image_8bit = (image_normalized * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            image_8bit = image
        else:
            # Generic conversion
            if np.max(image) > np.min(image):
                image_normalized = (image - image.min()) / (image.max() - image.min())
            else:
                image_normalized = np.zeros_like(image)
            image_8bit = (image_normalized * 255).astype(np.uint8)

        # Convert to PIL Image (without deprecated mode parameter)
        pil_image = Image.fromarray(image_8bit, 'L')

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = attributes.get('datasetName', 'unknown')
        filename = f"enhanced_{dataset_name}.{output_format.lower()}"
        filepath = os.path.join(output_dir, filename)

        # Save image
        if output_format.upper() in ["JPG", "JPEG"]:
            pil_image.save(filepath, "JPEG", quality=95)
        else:
            pil_image.save(filepath, "PNG")

        logger.info(f"✓ Image saved as: {filepath}")
        return filepath, pil_image

    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None, None


def display_image_with_info(pil_image, image_info=None, camera_info=None):
    """Enhanced image display with comprehensive information"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Display image
        ax1.imshow(pil_image, cmap='gray')
        ax1.set_title("Enhanced PyScope DE Camera Image")
        ax1.axis('off')

        # Display information
        info_text = "=== Camera Information ===\n"
        if camera_info:
            for key, value in camera_info.items():
                info_text += f"{key}: {value}\n"

        info_text += "\n=== Image Information ===\n"
        if image_info:
            for key, value in image_info.items():
                if isinstance(value, float):
                    info_text += f"{key}: {value:.2f}\n"
                else:
                    info_text += f"{key}: {value}\n"

        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error displaying image: {e}")


def main():
    """Enhanced main function with better error handling"""
    print("=== Enhanced PyScope DE Apollo Camera Script ===")

    # Test PyScope imports
    try:
        import pyscope.registry
        import pyscope.config
        logger.info("✓ PyScope imports successful")
    except ImportError as e:
        logger.error(f"PyScope import failed: {e}")
        return

    # Create and connect to camera
    de_camera = EnhancedDECamera()

    if not de_camera.connect():
        logger.error("Failed to connect to DE camera")
        return

    try:
        # Get camera information
        camera_info = de_camera.get_camera_info()
        logger.info(f"Camera info: {camera_info}")

        # Configure camera
        if not de_camera.configure_for_single_shot(exposure_ms=1000):
            logger.error("Failed to configure camera")
            return

        # Capture image
        image, attributes = de_camera.capture_image()
        if image is None:
            logger.error("Failed to capture image")
            return

        # Save images in both formats
        for fmt in ["PNG", "JPG"]:
            filepath, pil_image = save_image(image, attributes, fmt)
            if filepath:
                logger.info(f"✓ Saved {fmt} format")

        # Display image with comprehensive info
        if pil_image:
            image_info = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min': attributes['imageMin'],
                'max': attributes['imageMax'],
                'mean': attributes['imageMean'],
                'std': attributes['imageStd'],
                'acquisition_time': f"{attributes['acquisitionTime']:.3f}s"
            }

            logger.info("Displaying image with information...")
            display_image_with_info(pil_image, image_info, camera_info)

        logger.info("🎉 Enhanced PyScope image capture completed successfully!")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()