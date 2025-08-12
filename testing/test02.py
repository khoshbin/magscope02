#!/usr/bin/env python3
"""
CCD Camera Image Capture Script (DE Camera compatible)
Captures an image from any camera supported by the ccdcamera interface and saves/displays it as JPG/PNG
This version uses the ccdcamera interface for better camera abstraction and future extensibility
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import importlib

# Add pyscope path for ccdcamera modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyscope'))


def get_available_camera_classes():
    """Get available camera classes from the codebase with DE camera detection"""
    camera_modules = {
        # DE Camera modules (prioritized)
        'de_modern': 'de',  # Modern DE API
        'de_legacy': 'de1',  # Legacy DE API
        'de12_individual': 'de12',  # Individual DE12 module

        # Other camera types
        'tia': 'tia',  # TIA cameras
        'feicam': 'feicam',  # FEI cameras
        'tietz': 'tietz',  # Tietz cameras
        'tietz2': 'tietz2',  # Tietz F416
        'emmenu': 'emmenu',  # EmMenu cameras if available
    }

    available_cameras = {}
    for name, module_name in camera_modules.items():
        try:
            if name.startswith('de'):
                module = importlib.import_module(f'pyscope.{module_name}')
            else:
                module = importlib.import_module(f'pyscope.{module_name}')

            # Get camera classes from module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                        hasattr(attr, '__bases__') and
                        any('CCDCamera' in str(base) for base in attr.__bases__)):

                    # Add model_name info if available
                    model_info = ""
                    if hasattr(attr, 'model_name'):
                        model_info = f" (model: {attr.model_name})"
                    elif hasattr(attr, 'camera_name'):
                        model_info = f" (camera: {attr.camera_name})"

                    camera_key = f"{name}_{attr_name}{model_info}"
                    available_cameras[camera_key] = attr

        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")

    return available_cameras


def connect_to_camera(camera_type="de_auto"):
    """Connect to a CCD camera using the ccdcamera interface with DE camera priority"""
    try:
        if camera_type == "de_auto":
            # Try DE cameras in priority order (modern API first, then legacy)
            de_camera_priority = [
                # Modern DE API cameras (de.py)
                ('pyscope.de', 'DE20', 'DE20 - Latest model'),
                ('pyscope.de', 'DEApollo', 'Apollo - High-end model'),
                ('pyscope.de', 'DE16Counting', 'DE16 Counting mode'),
                ('pyscope.de', 'DE16Integrating', 'DE16 Integrating mode'),
                ('pyscope.de', 'DE12', 'DE12 - Entry model'),
                ('pyscope.de', 'DEDirectView', 'DirectView'),
                ('pyscope.de', 'DECenturiCounting', 'Centuri Counting mode'),
                ('pyscope.de', 'DECenturiIntegrating', 'Centuri Integrating mode'),

                # Legacy DE API cameras (de1.py)
                ('pyscope.de1', 'DE20', 'DE20 Legacy'),
                ('pyscope.de1', 'DE20c', 'DE20 Counting Legacy'),
                ('pyscope.de1', 'DE12', 'DE12 Legacy'),
                ('pyscope.de1', 'DE64', 'DE64 Legacy'),

                # Individual DE camera modules
                ('pyscope.de12', 'DE12', 'DE12 Individual module'),
            ]

            print("Attempting to connect to DE cameras in priority order...")
            for module_name, class_name, description in de_camera_priority:
                try:
                    print(f"Trying {description} ({module_name}.{class_name})...")
                    module = importlib.import_module(module_name)
                    camera_class = getattr(module, class_name)
                    camera = camera_class()
                    print(f"✓ Successfully connected to {description}")
                    print(f"  Camera model: {getattr(camera, 'model_name', 'Unknown')}")
                    print(f"  Camera name: {getattr(camera, 'name', 'Unknown')}")
                    return camera
                except ImportError as e:
                    print(f"  ✗ Module {module_name} not available: {e}")
                except AttributeError as e:
                    print(f"  ✗ Class {class_name} not found in {module_name}: {e}")
                except Exception as e:
                    print(f"  ✗ Failed to initialize {class_name}: {e}")
                    continue

            print("No DE cameras could be connected.")

        # If DE auto-connection fails, try other camera types
        print("\nTrying other camera types...")
        available_cameras = get_available_camera_classes()

        if not available_cameras:
            raise Exception("No camera classes found in the codebase")

        print("Available non-DE cameras:")
        for name, camera_class in available_cameras.items():
            if 'de_' not in name.lower():  # Skip DE cameras we already tried
                print(f"  - {name}: {camera_class.__name__}")

        # Try to connect to the first available non-DE camera
        for name, camera_class in available_cameras.items():
            if 'de_' not in name.lower():  # Skip DE cameras
                try:
                    camera = camera_class()
                    print(f"✓ Connected to camera: {camera.__class__.__name__}")
                    return camera
                except Exception as e:
                    print(f"  ✗ Failed to connect to {name}: {e}")
                    continue

        raise Exception("Could not connect to any available camera")

    except Exception as e:
        print(f"Failed to connect to camera: {e}")
        return None


def setup_camera_for_single_shot(camera):
    """Configure camera for a single image acquisition using ccdcamera interface"""
    try:
        # Set basic acquisition parameters using ccdcamera interface
        # These are standard methods available in most ccdcamera implementations

        # Set exposure time (1000ms = 1 second)
        camera.setExposureTime(1000)

        # Set exposure type to normal if supported
        if hasattr(camera, 'setExposureType'):
            camera.setExposureType('normal')

        # Set binning to 1x1 for full resolution
        if hasattr(camera, 'setBinning'):
            camera.setBinning({'x': 1, 'y': 1})

        # Reset offset to capture full frame
        if hasattr(camera, 'setOffset'):
            camera.setOffset({'x': 0, 'y': 0})

        # Set dimension to full camera size if supported
        if hasattr(camera, 'setDimension') and hasattr(camera, 'getDimension'):
            full_dimension = camera.getDimension()
            camera.setDimension(full_dimension)

        print("Camera configured for single shot")
        print(f"Exposure time: {camera.getExposureTime()}ms")

        if hasattr(camera, 'getBinning'):
            print(f"Binning: {camera.getBinning()}")
        if hasattr(camera, 'getDimension'):
            print(f"Dimension: {camera.getDimension()}")

        return True

    except Exception as e:
        print(f"Failed to configure camera: {e}")
        return False


def capture_image(camera):
    """Capture a single image from the camera using ccdcamera interface"""
    try:
        print("Starting image acquisition...")

        # Capture image using the standard ccdcamera interface
        # The _getImage() method is the standard way to capture in ccdcamera
        image = camera._getImage()

        if image is not None:
            print(f"Image captured successfully!")
            print(f"Image size: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Min/Max values: {image.min():.1f}/{image.max():.1f}")
            print(f"Mean/Std: {image.mean():.1f}/{image.std():.1f}")

            # Create attributes-like object for compatibility
            class ImageAttributes:
                def __init__(self, image):
                    self.imageMin = float(image.min())
                    self.imageMax = float(image.max())
                    self.imageMean = float(image.mean())
                    self.imageStd = float(image.std())
                    self.datasetName = getattr(camera, 'name', 'unknown_camera')

            attributes = ImageAttributes(image)
            return image, attributes
        else:
            print("Failed to capture image")
            return None, None

    except Exception as e:
        print(f"Error during image capture: {e}")
        return None, None


def convert_and_save_image(image, attributes, output_format="PNG", output_dir="captured_images"):
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
        dataset_name = attributes.datasetName if hasattr(attributes, 'datasetName') else "unknown"
        filename = f"ccd_image_{dataset_name}_{timestamp}.{output_format.lower()}"
        filepath = os.path.join(output_dir, filename)

        # Save the image
        if output_format.upper() == "JPG" or output_format.upper() == "JPEG":
            pil_image.save(filepath, "JPEG", quality=95)
        else:  # PNG
            pil_image.save(filepath, "PNG")

        print(f"Image saved as: {filepath}")
        return filepath, pil_image

    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None


def display_image(pil_image, image_info=None):
    """Display the image using matplotlib"""
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(pil_image, cmap='gray')
        plt.colorbar(label='Intensity')

        if image_info:
            title = f"CCD Camera Image\nCamera: {image_info.get('camera', 'N/A')}"
            title += f"\nSize: {image_info.get('shape', 'N/A')}"
            title += f"\nMin/Max: {image_info.get('min', 'N/A'):.1f}/{image_info.get('max', 'N/A'):.1f}"
            plt.title(title)
        else:
            plt.title("CCD Camera Image")

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error displaying image: {e}")


def list_camera_capabilities(camera):
    """Print camera capabilities and current settings"""
    print("\n=== Camera Capabilities ===")
    print(f"Camera class: {camera.__class__.__name__}")
    print(f"Camera name: {getattr(camera, 'name', 'Unknown')}")

    # Check common capabilities
    capabilities = [
        'getExposureTime', 'setExposureTime',
        'getBinning', 'setBinning',
        'getDimension', 'setDimension',
        'getOffset', 'setOffset',
        'getExposureTypes',
        '_getCameraSize'
    ]

    print("\nAvailable methods:")
    for cap in capabilities:
        if hasattr(camera, cap):
            try:
                if cap.startswith('get'):
                    value = getattr(camera, cap)()
                    print(f"  ✓ {cap}: {value}")
                else:
                    print(f"  ✓ {cap}: Available")
            except Exception as e:
                print(f"  ✓ {cap}: Available (error getting value: {e})")
        else:
            print(f"  ✗ {cap}: Not available")


def main():
    """Main function to capture and process image"""
    print("=== CCD Camera Image Capture Script ===")

    # Connect to camera
    camera = connect_to_camera("de_auto")  # Try DE camera first, fallback to others
    if not camera:
        return

    try:
        # Show camera capabilities
        list_camera_capabilities(camera)

        # Setup camera
        if not setup_camera_for_single_shot(camera):
            return

        # Capture image
        image, attributes = capture_image(camera)
        if image is None:
            return

        # Save as both PNG and JPG
        for fmt in ["PNG", "JPG"]:
            filepath, pil_image = convert_and_save_image(image, attributes, fmt)
            if filepath:
                print(f"✓ Saved {fmt} format")

        # Display the image
        if pil_image:
            image_info = {
                'camera': getattr(camera, 'name', camera.__class__.__name__),
                'shape': image.shape,
                'min': attributes.imageMin,
                'max': attributes.imageMax
            }
            print("Displaying image...")
            display_image(pil_image, image_info)

        print("Image capture completed successfully!")

    except Exception as e:
        print(f"Error in main process: {e}")

    finally:
        # Cleanup camera connection if needed
        try:
            if hasattr(camera, 'disconnect'):
                camera.disconnect()
                print("Disconnected from camera")
            elif hasattr(camera, '__del__'):
                del camera
                print("Camera object cleaned up")
        except:
            pass


if __name__ == "__main__":
    main()