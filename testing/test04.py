#!/usr/bin/env python3
"""
Apollo Camera Usage Example
Shows how your code would look when using the Apollo camera specifically
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Add pyscope path for ccdcamera modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyscope'))


def connect_to_apollo_camera():
    """Connect specifically to Apollo camera"""
    try:
        from pyscope.de import DEApollo
        print("Initializing Apollo camera...")

        # This is how Apollo camera gets initialized in your codebase
        camera = DEApollo()

        print(f"✓ Connected to Apollo camera")
        print(f"  Camera name: {camera.name}")  # 'DEApollo'
        print(f"  Model name: {camera.model_name}")  # 'Apollo'
        print(f"  Hardware binning: {camera.hardware_binning}")  # {'x': 1, 'y': 1}

        return camera

    except ImportError as e:
        print(f"Could not import Apollo camera: {e}")
        return None
    except Exception as e:
        print(f"Failed to initialize Apollo camera: {e}")
        return None


def setup_apollo_camera_for_acquisition(camera):
    """Configure Apollo camera for image acquisition using Apollo-specific methods"""
    try:
        print("\n=== Apollo Camera Setup ===")

        # Apollo has a FIXED frame rate - you cannot change it
        # It automatically sets to maximum frames per second
        print("Apollo frame rate is hardware-determined (fixed):")
        current_fps = camera.getProperty('Frames Per Second')
        print(f"  Current FPS: {current_fps}")

        # Apollo uses dose fractionation timing, not regular exposure time
        print("\nSetting exposure using Apollo's dose fractionation method:")
        frame_time_ms = 1000  # 1 second worth of dose fractionation
        camera.setFrameTime(frame_time_ms)  # Uses setFrameTimeForCounting internally

        actual_frame_time = camera.getFrameTime()  # Uses getFrameTimeForCounting internally
        print(f"  Requested frame time: {frame_time_ms}ms")
        print(f"  Actual frame time: {actual_frame_time}ms")

        # Standard ccdcamera interface methods still work
        camera.setBinning({'x': 1, 'y': 1})
        camera.setOffset({'x': 0, 'y': 0})

        # Get Apollo-specific properties
        print(f"\nApollo camera properties:")
        print(f"  Binning: {camera.getBinning()}")
        print(f"  Offset: {camera.getOffset()}")
        print(f"  Dimension: {camera.getDimension()}")

        # Apollo-specific electron counting setup (done automatically in __init__)
        print(f"  Movie sum count: {camera.getProperty('Movie Sum Count')}")

        return True

    except Exception as e:
        print(f"Failed to configure Apollo camera: {e}")
        return False


def capture_apollo_image(camera):
    """Capture image from Apollo camera"""
    try:
        print("\n=== Apollo Image Acquisition ===")
        print("Starting Apollo image acquisition...")

        # Apollo uses the standard ccdcamera interface for image capture
        # The _getImage() method internally:
        # 1. Calls self.preAcquisitionSetup() if available
        # 2. Uses de_getImage(self.model_name) where model_name = 'Apollo'
        # 3. Records timing and validates the returned array

        image = camera._getImage()

        if image is not None:
            print(f"✓ Apollo image captured successfully!")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Value range: {image.min():.1f} to {image.max():.1f}")
            print(f"  Mean ± Std: {image.mean():.1f} ± {image.std():.1f}")

            # Apollo timestamp from exposure
            if hasattr(camera, 'exposure_timestamp'):
                print(f"  Exposure timestamp: {camera.exposure_timestamp}")

            return image
        else:
            print("✗ Failed to capture Apollo image")
            return None

    except Exception as e:
        print(f"Error during Apollo image capture: {e}")
        return None


def apollo_specific_properties(camera):
    """Demonstrate Apollo-specific property access"""
    try:
        print("\n=== Apollo-Specific Properties ===")

        # Apollo uses the DE-Server with model_name = 'Apollo'
        # All property access goes through de_getProperty('Apollo', property_name)

        apollo_properties = [
            'Frames Per Second',
            'Movie Sum Count',
            'Electron Counting',
            'Sensor Size X',
            'Sensor Size Y',
            'Exposure Time (seconds)',
            'Image Processing - Mode',
            'Correction Mode'
        ]

        for prop in apollo_properties:
            try:
                value = camera.getProperty(prop)
                print(f"  {prop}: {value}")
            except Exception as e:
                print(f"  {prop}: <Error: {e}>")

        # Apollo size information
        sensor_size = camera._getCameraSize()
        print(f"\nApollo sensor size: {sensor_size}")

    except Exception as e:
        print(f"Error accessing Apollo properties: {e}")


def save_apollo_image(image, output_dir="apollo_images"):
    """Save Apollo image with Apollo-specific naming"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Normalize for saving
        if image.dtype in [np.float32, np.float64]:
            image_normalized = (image - image.min()) / (image.max() - image.min())
            image_8bit = (image_normalized * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            image_8bit = (image / 256).astype(np.uint8)
        else:
            image_8bit = image.astype(np.uint8)

        # Convert to PIL and save
        pil_image = Image.fromarray(image_8bit, mode='L')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"apollo_image_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        pil_image.save(filepath, "PNG")
        print(f"✓ Apollo image saved: {filepath}")

        return filepath, pil_image

    except Exception as e:
        print(f"Error saving Apollo image: {e}")
        return None, None


def main():
    """Main function demonstrating Apollo camera usage"""
    print("=== Apollo Camera Demonstration ===")

    # Connect to Apollo camera
    camera = connect_to_apollo_camera()
    if not camera:
        print("Could not connect to Apollo camera")
        return

    try:
        # Show Apollo-specific properties
        apollo_specific_properties(camera)

        # Setup for acquisition (Apollo-specific considerations)
        if not setup_apollo_camera_for_acquisition(camera):
            print("Apollo camera setup failed")
            return

        # Capture image
        image = capture_apollo_image(camera)
        if image is None:
            print("Apollo image capture failed")
            return

        # Save the image
        filepath, pil_image = save_apollo_image(image)

        # Display with matplotlib
        if pil_image:
            plt.figure(figsize=(10, 8))
            plt.imshow(pil_image, cmap='gray')
            plt.colorbar(label='Intensity')
            plt.title(f"Apollo Camera Image\nShape: {image.shape}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        print("\n✓ Apollo camera demonstration completed successfully!")

    except Exception as e:
        print(f"Error in main Apollo process: {e}")

    finally:
        # Apollo cleanup is handled automatically by DECameraBase destructor
        # which calls disconnectDEAPI()
        print("Apollo camera session ended")


if __name__ == "__main__":
    main()