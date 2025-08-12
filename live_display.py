import sys
from time import sleep
from datetime import datetime
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
import DEAPI
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time


def live_view_with_jpeg_save():
    client = DEAPI.Client()

    try:
        client.Connect("localhost")

        # Set up for live view
        client.SetProperty("Exposure Mode", "Normal")
        client.SetProperty("Frame Count", 10)  # Short exposures for live view
        client.SetProperty("Autosave Movie", "Off")  # Don't save movie files

        # Start multiple acquisitions for live view
        client.StartAcquisition(100)  # 100 acquisitions

        # Set up matplotlib for live display
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        im = None

        acquisition_count = 0
        last_acq_index = -1

        while acquisition_count < 100:
            try:
                # Get image with contrast stretching
                image_attributes = de.ImageAttributes()
                image_attributes.stretchType = de.ContrastStretchType.LINEAR
                image_attributes.outlierPercentage = 0.1

                image_data, pixel_format, image_attributes = client.GetResult(
                    frame_type=de.FrameType.SUMTOTAL,
                    pixel_format=de.PixelFormat.UINT8,
                    image_attributes=image_attributes
                )

                # Only update display when we get a new acquisition
                if image_attributes.acqIndex != last_acq_index:
                    # Update display
                    if im is None:
                        im = ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)
                        ax.set_title('Live View - Press Ctrl+C to stop and save')
                        ax.axis('off')
                        plt.colorbar(im)
                    else:
                        im.set_array(image_data)

                    plt.draw()
                    plt.pause(0.01)

                    acquisition_count += 1
                    last_acq_index = image_attributes.acqIndex

                    print(f"Acquisition {acquisition_count}, "
                          f"Mean e-/pixel: {image_attributes.eppix:.1f}")

            except KeyboardInterrupt:
                print("\nStopping acquisition...")
                client.StopAcquisition()

                # Save the last image as JPEG
                if 'image_data' in locals():
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"de_image_{timestamp}.jpg"

                    pil_image = Image.fromarray(image_data, mode='L')
                    pil_image.save(filename, 'JPEG', quality=95)
                    print(f"Last image saved as {filename}")

                break

            except Exception as e:
                print(f"Error getting image: {e}")
                break

        plt.ioff()  # Turn off interactive mode

    except Exception as e:
        print(f"Error: {e}")

    finally:
        client.Disconnect()


# Run live view
live_view_with_jpeg_save()