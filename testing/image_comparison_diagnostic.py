#!/usr/bin/env python3
"""
Image Comparison Diagnostic Script
Compares images from test01.py and test17.py to identify normalization differences
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add DEAPI path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]

def analyze_image_data(image, label, attributes=None):
    """Analyze raw image data in detail"""
    print(f"\n=== {label} Analysis ===")
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print(f"Min: {image.min()}")
    print(f"Max: {image.max()}")
    print(f"Mean: {image.mean():.6f}")
    print(f"Std: {image.std():.6f}")
    print(f"Unique values (first 10): {np.unique(image.flatten())[:10]}")
    
    if attributes:
        print(f"DEAPI imageMin: {attributes.imageMin}")
        print(f"DEAPI imageMax: {attributes.imageMax}")
        print(f"DEAPI imageMean: {attributes.imageMean:.6f}")
        print(f"DEAPI imageStd: {attributes.imageStd:.6f}")
    
    return {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(image.min()),
        'max': float(image.max()),
        'mean': float(image.mean()),
        'std': float(image.std()),
        'deapi_min': float(attributes.imageMin) if attributes else None,
        'deapi_max': float(attributes.imageMax) if attributes else None,
        'deapi_mean': float(attributes.imageMean) if attributes else None,
        'deapi_std': float(attributes.imageStd) if attributes else None
    }

def capture_test01_style():
    """Capture image using test01.py method"""
    from pyscope import DEAPI
    
    try:
        client = DEAPI.Client()
        client.Connect()

        cameras = client.ListCameras()
        if not cameras:
            raise Exception("No cameras found")

        client.SetCurrentCamera(cameras[0])
        print(f"Connected to camera: {cameras[0]}")

        # Set basic acquisition parameters (exactly like test01.py)
        client.SetProperty("Exposure Mode", "Normal")
        client.SetProperty("Image Processing - Mode", "Integrating")
        client.SetProperty("Frames Per Second", 20)
        client.SetProperty("Exposure Time (seconds)", 1.0)

        # Disable autosave for quick capture
        client.SetProperty("Autosave Final Image", "Off")
        client.SetProperty("Autosave Movie", "Off")

        print("Starting test01.py style acquisition...")

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

        print(f"test01.py capture completed")
        print(f"Pixel format returned: {pixelFormat}")

        client.Disconnect()
        return image, attributes, pixelFormat

    except Exception as e:
        print(f"Error in test01.py style capture: {e}")
        return None, None, None

def capture_test17_style():
    """Capture image using test17.py (PyScope) method"""
    try:
        import pyscope.registry

        # Connect via PyScope
        camera = pyscope.registry.getClass("DEApollo")()
        camera_size = camera.getCameraSize()
        print(f"PyScope connected to DEApollo, size: {camera_size}")

        # Configure like test17.py
        camera.setExposureTime(1000)  # 1000ms = 1s

        print("Starting test17.py style acquisition...")

        # Capture using PyScope
        image = camera.getImage()

        print(f"test17.py capture completed")
        return image, None, None

    except Exception as e:
        print(f"Error in test17.py style capture: {e}")
        return None, None, None

def compare_conversion_methods(image, attributes=None):
    """Compare different image conversion methods"""
    print(f"\n=== Image Conversion Comparison ===")
    
    conversions = {}
    
    # Method 1: test01.py conversion
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_normalized = (image - image.min()) / (image.max() - image.min())
        image_8bit_test01 = (image_normalized * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        image_8bit_test01 = (image / 256).astype(np.uint8)
    else:
        image_8bit_test01 = image.astype(np.uint8)
    
    conversions['test01_method'] = {
        'min': image_8bit_test01.min(),
        'max': image_8bit_test01.max(),
        'mean': image_8bit_test01.mean(),
        'std': image_8bit_test01.std()
    }
    
    # Method 2: Direct scaling using DEAPI attributes
    if attributes:
        # Use DEAPI min/max for scaling
        deapi_normalized = (image - attributes.imageMin) / (attributes.imageMax - attributes.imageMin)
        image_8bit_deapi = np.clip(deapi_normalized * 255, 0, 255).astype(np.uint8)
        conversions['deapi_method'] = {
            'min': image_8bit_deapi.min(),
            'max': image_8bit_deapi.max(),
            'mean': image_8bit_deapi.mean(),
            'std': image_8bit_deapi.std()
        }
    
    # Method 3: Percentile-based normalization
    p_low, p_high = np.percentile(image, [2, 98])
    image_clipped = np.clip(image, p_low, p_high)
    image_normalized_perc = (image_clipped - p_low) / (p_high - p_low)
    image_8bit_perc = (image_normalized_perc * 255).astype(np.uint8)
    conversions['percentile_method'] = {
        'min': image_8bit_perc.min(),
        'max': image_8bit_perc.max(),
        'mean': image_8bit_perc.mean(),
        'std': image_8bit_perc.std()
    }
    
    # Method 4: No normalization - direct conversion
    if image.dtype == np.uint16:
        image_8bit_direct = (image >> 8).astype(np.uint8)  # Bit shift instead of division
        conversions['direct_shift_method'] = {
            'min': image_8bit_direct.min(),
            'max': image_8bit_direct.max(),
            'mean': image_8bit_direct.mean(),
            'std': image_8bit_direct.std()
        }
    
    for method, stats in conversions.items():
        print(f"{method}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    return conversions

def save_raw_image_data(image, filename_prefix, attributes=None):
    """Save raw image data for later analysis"""
    try:
        # Save as numpy array
        np.save(f"{filename_prefix}_raw.npy", image)
        
        # Save metadata
        metadata = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'timestamp': datetime.now().isoformat()
        }
        
        if attributes:
            metadata.update({
                'deapi_imageMin': float(attributes.imageMin),
                'deapi_imageMax': float(attributes.imageMax),
                'deapi_imageMean': float(attributes.imageMean),
                'deapi_imageStd': float(attributes.imageStd),
                'deapi_datasetName': attributes.datasetName,
                'deapi_frameCount': attributes.frameCount
            })
        
        with open(f"{filename_prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved raw data: {filename_prefix}_raw.npy and {filename_prefix}_metadata.json")
        return True
    except Exception as e:
        print(f"Error saving raw data: {e}")
        return False

def create_comparison_plot(image1, image2, labels, attributes1=None, attributes2=None):
    """Create side-by-side comparison plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Raw images
    im1 = ax1.imshow(image1, cmap='gray')
    ax1.set_title(f"{labels[0]} - Raw Image\nMin: {image1.min():.1f}, Max: {image1.max():.1f}")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(image2, cmap='gray')
    ax2.set_title(f"{labels[1]} - Raw Image\nMin: {image2.min():.1f}, Max: {image2.max():.1f}")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    # Difference image
    if image1.shape == image2.shape and image1.dtype == image2.dtype:
        diff = image2.astype(np.float64) - image1.astype(np.float64)
        im3 = ax3.imshow(diff, cmap='RdBu_r')
        ax3.set_title(f"Difference: {labels[1]} - {labels[0]}\nMin: {diff.min():.1f}, Max: {diff.max():.1f}")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3)
        
        # Histogram comparison
        ax4.hist(image1.flatten(), bins=100, alpha=0.5, label=labels[0], density=True)
        ax4.hist(image2.flatten(), bins=100, alpha=0.5, label=labels[1], density=True)
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Histogram Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, f"Cannot compare:\n{labels[0]}: {image1.shape}, {image1.dtype}\n{labels[1]}: {image2.shape}, {image2.dtype}", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
        
        ax4.text(0.5, 0.5, "No histogram comparison available", 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"image_comparison_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main diagnostic function"""
    print("=== Image Comparison Diagnostic ===")
    print("Comparing test01.py (direct DEAPI) vs test17.py (PyScope) image capture")
    
    # Capture with both methods
    print("\n1. Capturing with test01.py method (direct DEAPI)...")
    image1, attributes1, pixelFormat1 = capture_test01_style()
    
    print("\n2. Capturing with test17.py method (PyScope)...")
    image2, attributes2, pixelFormat2 = capture_test17_style()
    
    if image1 is None or image2 is None:
        print("❌ Failed to capture images with one or both methods")
        return
    
    # Analyze both images
    analysis1 = analyze_image_data(image1, "test01.py (DEAPI)", attributes1)
    analysis2 = analyze_image_data(image2, "test17.py (PyScope)", attributes2)
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_raw_image_data(image1, f"test01_image_{timestamp}", attributes1)
    save_raw_image_data(image2, f"test17_image_{timestamp}", attributes2)
    
    # Compare conversion methods
    print("\n3. Analyzing conversion methods for test01.py image...")
    conversions1 = compare_conversion_methods(image1, attributes1)
    
    print("\n4. Analyzing conversion methods for test17.py image...")
    conversions2 = compare_conversion_methods(image2, attributes2)
    
    # Check if images are identical
    if image1.shape == image2.shape and image1.dtype == image2.dtype:
        if np.array_equal(image1, image2):
            print("\n✅ Raw images are IDENTICAL!")
        else:
            diff_stats = {
                'max_diff': np.abs(image2.astype(np.float64) - image1.astype(np.float64)).max(),
                'mean_diff': np.mean(image2.astype(np.float64) - image1.astype(np.float64)),
                'std_diff': np.std(image2.astype(np.float64) - image1.astype(np.float64))
            }
            print(f"\n⚠️  Raw images are DIFFERENT!")
            print(f"Max difference: {diff_stats['max_diff']}")
            print(f"Mean difference: {diff_stats['mean_diff']:.6f}")
            print(f"Std of differences: {diff_stats['std_diff']:.6f}")
    else:
        print(f"\n⚠️  Images have different shapes or dtypes:")
        print(f"test01.py: {image1.shape}, {image1.dtype}")
        print(f"test17.py: {image2.shape}, {image2.dtype}")
    
    # Create comparison plot
    print("\n5. Creating comparison plot...")
    create_comparison_plot(image1, image2, ["test01.py", "test17.py"], attributes1, attributes2)
    
    # Summary and recommendations
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    # Check pixel format
    if pixelFormat1:
        print(f"test01.py pixel format: {pixelFormat1}")
    print(f"test17.py uses PyScope default (typically uint16)")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS TO FIX DIFFERENCES:")
    
    if image1.dtype != image2.dtype:
        print(f"1. Data type mismatch: {image1.dtype} vs {image2.dtype}")
        print("   → Ensure both methods use the same pixel format")
    
    if attributes1 and (image1.min() != attributes1.imageMin or image1.max() != attributes1.imageMax):
        print("2. DEAPI attributes differ from actual array values")
        print("   → Use array min/max instead of DEAPI attributes for normalization")
    
    value_range1 = image1.max() - image1.min()
    value_range2 = image2.max() - image2.min()
    if abs(value_range1 - value_range2) > 0.1 * max(value_range1, value_range2):
        print("3. Different dynamic ranges detected")
        print("   → Check exposure settings and image processing modes")
    
    print("\n💡 TO GET IDENTICAL RESULTS:")
    print("1. Use the same pixel format (PixelFormat.AUTO vs uint16)")
    print("2. Use raw array min/max for normalization, not DEAPI attributes")
    print("3. Ensure identical camera settings")
    print("4. For display: Use identical normalization method")
    
    # Save comparison results
    comparison_results = {
        'test01_analysis': analysis1,
        'test17_analysis': analysis2,
        'test01_conversions': conversions1,
        'test17_conversions': conversions2,
        'timestamp': datetime.now().isoformat(),
        'pixel_format_test01': str(pixelFormat1) if pixelFormat1 else None,
        'recommendations': [
            "Use the same pixel format",
            "Use array min/max for normalization",
            "Ensure identical camera settings",
            "Use identical normalization method"
        ]
    }
    
    with open(f"image_comparison_results_{timestamp}.json", 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: image_comparison_results_{timestamp}.json")

if __name__ == "__main__":
    main()