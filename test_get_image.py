import sys

sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
import DEAPI

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def acquire_and_display_image():
    client = DEAPI.Client()

    try:
        # Connect and acquire image
        client.Connect("localhost")

        # Check system status
        status = client.GetProperty("System Status")
        if status != "OK":
            print(f"System not ready: {status}")
            return

        # Configure acquisition
        client.SetProperty("Exposure Mode", "Normal")
        client.SetProperty("Frame Count", 50)

        # Start acquisition
        client.StartAcquisition(1)

        # Get the image with auto contrast stretching
        image_attributes = DEAPI.ImageAttributes()
        image_attributes.stretchType = DEAPI.ContrastStretchType.LINEAR
        image_attributes.outlierPercentage = 0.1  # Ignore 0.1% outliers on each side

        image_data, pixel_format, image_attributes = client.GetResult(
            frame_type=de.FrameType.SUMTOTAL,
            pixel_format=de.PixelFormat.UINT8,  # Get as 8-bit for easy display
            image_attributes=image_attributes
        )

        # Display with matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(image_data, cmap='gray')
        plt.title(f'Acquired Image - Dataset: {image_attributes.datasetName}')
        plt.colorbar()
        plt.axis('off')
        plt.show()

        return image_data, image_attributes

    except Exception as e:
        print(f"Error: {e}")
        return None, None

    finally:
        client.Disconnect()


# Run the function
image, attributes = acquire_and_display_image()