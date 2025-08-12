#!/usr/bin/env python3
"""
Direct Electron Camera Image Capture Script
Captures an image from DE camera and saves/displays it as JPG/PNG
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Add DEAPI path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
import DEAPI


def connect_to_camera(host="localhost", port=13240):
    """Connect to DE-Server and select the first available camera"""
    try:
        client = DEAPI.Client()
        client.Connect(host, port)

        cameras = client.ListCameras()
        if not cameras:
            raise Exception("No cameras found")

        client.SetCurrentCamera(cameras[0])
        print(f"Connected to camera: {cameras[0]}")
        return client
    except Exception as e:
        print(f"Failed to connect to camera: {e}")
        return None


def setup_camera_for_single_shot(client):
    """Configure camera for a single image acquisition"""
    try:
        # Set basic acquisition parameters
        client.SetProperty("Exposure Mode", "Normal")
        client.SetProperty("Image Processing - Mode", "Integrating")
        client.SetProperty("Frames Per Second", 20)
        client.SetProperty("Exposure Time (seconds)", 1.0)

        # Disable autosave for this quick capture
        client.SetProperty("Autosave Final Image", "Off")
        client.SetProperty("Autosave Movie", "Off")

        print("Camera configured for single shot")
        return True
    except Exception as e:
        print(f"Failed to configure camera: {e}")
        return False


def capture_image(client):
    """Capture a single image from the camera"""
    try:
        print("Starting image acquisition...")

        # Start acquisition (1 repeat)
        client.StartAcquisition(1)

        # Get the final summed image
        frameType = DEAPI.FrameType.SUMTOTAL
        pixelFormat = DEAPI.PixelFormat.AUTO
        attributes = DEAPI.Attributes()
        histogram = DEAPI.Histogram()

        image, pixelFormat, attributes, histogram = client.GetResult(
            frameType, pixelFormat, attributes, histogram
        )

        if image is not None:
            print(f"Image captured successfully!")
            print(f"Image size: {image.shape}")
            print(f"Pixel format: {pixelFormat}")
            print(f"Dataset: {attributes.datasetName}")
            print(f"Min/Max values: {attributes.imageMin:.1f}/{attributes.imageMax:.1f}")
            print(f"Mean/Std: {attributes.imageMean:.1f}/{attributes.imageStd:.1f}")
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
        dataset_name = attributes.datasetName if attributes.datasetName else "unknown"
        filename = f"de_image_{dataset_name}_{timestamp}.{output_format.lower()}"
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
            title = f"DE Camera Image\nDataset: {image_info.get('dataset', 'N/A')}"
            title += f"\nSize: {image_info.get('shape', 'N/A')}"
            title += f"\nMin/Max: {image_info.get('min', 'N/A'):.1f}/{image_info.get('max', 'N/A'):.1f}"
            plt.title(title)
        else:
            plt.title("DE Camera Image")

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error displaying image: {e}")


def main():
    """Main function to capture and process image"""
    print("=== DE Camera Image Capture Script ===")

    # Connect to camera
    client = connect_to_camera()
    if not client:
        return

    try:
        # Setup camera
        if not setup_camera_for_single_shot(client):
            return

        # Capture image
        image, attributes = capture_image(client)
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
                'dataset': attributes.datasetName,
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
        # Disconnect from camera
        try:
            client.Disconnect()
            print("Disconnected from camera")
        except:
            pass


if __name__ == "__main__":
    main()