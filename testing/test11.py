#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Image Differences Between test01.py and test09.py
Identifies why images generated from DE cameras look different between DEAPI and PyScope approaches
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, Any, Tuple, Optional

# Add DEAPI path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]

try:
    from pyscope import DEAPI

    DEAPI_AVAILABLE = True
except ImportError:
    DEAPI_AVAILABLE = False
    print("DEAPI not available")

try:
    import pyscope.registry

    PYSCOPE_AVAILABLE = True
except ImportError:
    PYSCOPE_AVAILABLE = False
    print("PyScope not available")


class ImageDiagnostics:
    """Diagnostic tools for comparing images from different acquisition methods"""

    def __init__(self):
        self.results = {}

    def analyze_raw_image_data(self, image: np.ndarray, source: str) -> Dict[str, Any]:
        """Comprehensive analysis of raw image data"""
        stats = {
            'source': source,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(image.min()),
            'max_value': float(image.max()),
            'mean_value': float(image.mean()),
            'std_value': float(image.std()),
            'median_value': float(np.median(image)),
            'percentiles': {
                '1': float(np.percentile(image, 1)),
                '5': float(np.percentile(image, 5)),
                '95': float(np.percentile(image, 95)),
                '99': float(np.percentile(image, 99))
            },
            'dynamic_range': float(image.max() - image.min()),
            'bit_depth_used': np.ceil(np.log2(image.max() + 1)) if image.max() > 0 else 0,
            'histogram': np.histogram(image.flatten(), bins=50)[0].tolist(),
            'histogram_bins': np.histogram(image.flatten(), bins=50)[1].tolist()
        }

        # Check for specific patterns
        stats['has_negative_values'] = bool(image.min() < 0)
        stats['is_integer_data'] = image.dtype.kind in 'ui'
        stats['is_float_data'] = image.dtype.kind == 'f'
        stats['saturated_pixels'] = int(np.sum(image == image.max()))
        stats['zero_pixels'] = int(np.sum(image == 0))

        return stats

    def compare_images(self, image1: np.ndarray, image2: np.ndarray,
                       label1: str, label2: str) -> Dict[str, Any]:
        """Compare two images and identify differences"""

        # Analyze both images
        stats1 = self.analyze_raw_image_data(image1, label1)
        stats2 = self.analyze_raw_image_data(image2, label2)

        comparison = {
            'image1_stats': stats1,
            'image2_stats': stats2,
            'differences': {}
        }

        # Shape comparison
        comparison['differences']['shape_same'] = image1.shape == image2.shape

        if image1.shape == image2.shape:
            # Pixel-wise comparison (if same shape)
            diff = image1.astype(np.float64) - image2.astype(np.float64)
            comparison['differences']['pixel_differences'] = {
                'max_absolute_diff': float(np.max(np.abs(diff))),
                'mean_absolute_diff': float(np.mean(np.abs(diff))),
                'rms_diff': float(np.sqrt(np.mean(diff ** 2))),
                'correlation': float(np.corrcoef(image1.flatten(), image2.flatten())[0, 1]),
                'identical_pixels': int(np.sum(image1 == image2)),
                'total_pixels': int(image1.size)
            }

            # Normalization comparison
            comparison['differences']['normalization_analysis'] = self._compare_normalizations(image1, image2)

        # Data type comparison
        comparison['differences']['dtype_same'] = stats1['dtype'] == stats2['dtype']
        comparison['differences']['range_comparison'] = {
            'range1': stats1['dynamic_range'],
            'range2': stats2['dynamic_range'],
            'range_ratio': stats2['dynamic_range'] / stats1['dynamic_range'] if stats1['dynamic_range'] > 0 else float(
                'inf')
        }

        return comparison

    def _compare_normalizations(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Compare different normalization approaches"""

        def normalize_to_uint8(img):
            """Standard normalization to 8-bit"""
            if img.dtype in [np.float32, np.float64]:
                normalized = (img - img.min()) / (img.max() - img.min())
                return (normalized * 255).astype(np.uint8)
            elif img.dtype == np.uint16:
                return (img / 256).astype(np.uint8)
            elif img.dtype == np.uint8:
                return img
            else:
                normalized = (img - img.min()) / (img.max() - img.min())
                return (normalized * 255).astype(np.uint8)

        # Test different normalization approaches
        norm1_std = normalize_to_uint8(image1)
        norm2_std = normalize_to_uint8(image2)

        # Min-max normalization
        norm1_minmax = ((image1 - image1.min()) / (image1.max() - image1.min()) * 255).astype(np.uint8)
        norm2_minmax = ((image2 - image2.min()) / (image2.max() - image2.min()) * 255).astype(np.uint8)

        # Z-score normalization
        def zscore_normalize(img):
            mean_val = img.mean()
            std_val = img.std()
            if std_val > 0:
                normalized = (img - mean_val) / std_val
                # Clip to reasonable range and scale
                normalized = np.clip(normalized, -3, 3)
                return ((normalized + 3) / 6 * 255).astype(np.uint8)
            else:
                return np.zeros_like(img, dtype=np.uint8)

        norm1_zscore = zscore_normalize(image1)
        norm2_zscore = zscore_normalize(image2)

        return {
            'standard_normalization_correlation': float(np.corrcoef(norm1_std.flatten(), norm2_std.flatten())[0, 1]),
            'minmax_normalization_correlation': float(
                np.corrcoef(norm1_minmax.flatten(), norm2_minmax.flatten())[0, 1]),
            'zscore_normalization_correlation': float(np.corrcoef(norm1_zscore.flatten(), norm2_zscore.flatten())[0, 1])
        }

    def visualize_comparison(self, image1: np.ndarray, image2: np.ndarray,
                             label1: str, label2: str, save_path: str = None):
        """Create comprehensive visualization comparing two images"""

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Image Comparison: {label1} vs {label2}', fontsize=16)

        # Normalize for display
        def normalize_for_display(img):
            return (img - img.min()) / (img.max() - img.min())

        norm1 = normalize_for_display(image1)
        norm2 = normalize_for_display(image2)

        # Row 1: Original images
        axes[0, 0].imshow(norm1, cmap='gray')
        axes[0, 0].set_title(f'{label1}\nShape: {image1.shape}, Type: {image1.dtype}')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(norm2, cmap='gray')
        axes[0, 1].set_title(f'{label2}\nShape: {image2.shape}, Type: {image2.dtype}')
        axes[0, 1].axis('off')

        # Difference image (if same shape)
        if image1.shape == image2.shape:
            diff = norm1 - norm2
            im_diff = axes[0, 2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
            axes[0, 2].set_title('Normalized Difference\n(Image1 - Image2)')
            axes[0, 2].axis('off')
            plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)
        else:
            axes[0, 2].text(0.5, 0.5, 'Different\nShapes', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')

        # Row 2: Histograms
        axes[1, 0].hist(image1.flatten(), bins=50, alpha=0.7, density=True, label=label1)
        axes[1, 0].set_title('Histogram Comparison (Raw Values)')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()

        axes[1, 1].hist(image2.flatten(), bins=50, alpha=0.7, density=True, label=label2, color='orange')
        axes[1, 1].set_title('Histogram Comparison (Raw Values)')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()

        # Overlay histograms
        axes[1, 2].hist(norm1.flatten(), bins=50, alpha=0.5, density=True, label=f'{label1} (norm)', color='blue')
        axes[1, 2].hist(norm2.flatten(), bins=50, alpha=0.5, density=True, label=f'{label2} (norm)', color='orange')
        axes[1, 2].set_title('Normalized Histograms Overlay')
        axes[1, 2].set_xlabel('Normalized Pixel Value')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()

        # Row 3: Statistical comparisons
        stats1 = self.analyze_raw_image_data(image1, label1)
        stats2 = self.analyze_raw_image_data(image2, label2)

        # Create comparison table
        comparison_data = [
            ['Metric', label1, label2, 'Ratio'],
            ['Min', f"{stats1['min_value']:.2f}", f"{stats2['min_value']:.2f}",
             f"{stats2['min_value'] / stats1['min_value']:.3f}" if stats1['min_value'] != 0 else "∞"],
            ['Max', f"{stats1['max_value']:.2f}", f"{stats2['max_value']:.2f}",
             f"{stats2['max_value'] / stats1['max_value']:.3f}" if stats1['max_value'] != 0 else "∞"],
            ['Mean', f"{stats1['mean_value']:.2f}", f"{stats2['mean_value']:.2f}",
             f"{stats2['mean_value'] / stats1['mean_value']:.3f}" if stats1['mean_value'] != 0 else "∞"],
            ['Std', f"{stats1['std_value']:.2f}", f"{stats2['std_value']:.2f}",
             f"{stats2['std_value'] / stats1['std_value']:.3f}" if stats1['std_value'] != 0 else "∞"],
            ['Range', f"{stats1['dynamic_range']:.2f}", f"{stats2['dynamic_range']:.2f}",
             f"{stats2['dynamic_range'] / stats1['dynamic_range']:.3f}" if stats1['dynamic_range'] != 0 else "∞"]
        ]

        axes[2, 0].axis('tight')
        axes[2, 0].axis('off')
        table = axes[2, 0].table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                                 cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[2, 0].set_title('Statistical Comparison')

        # Profile comparison (if same shape)
        if image1.shape == image2.shape:
            # Take center line profiles
            center_y = image1.shape[0] // 2
            center_x = image1.shape[1] // 2

            profile1_h = image1[center_y, :]
            profile2_h = image2[center_y, :]
            profile1_v = image1[:, center_x]
            profile2_v = image2[:, center_x]

            axes[2, 1].plot(profile1_h, label=f'{label1} (horizontal)', alpha=0.7)
            axes[2, 1].plot(profile2_h, label=f'{label2} (horizontal)', alpha=0.7)
            axes[2, 1].set_title('Center Line Profiles')
            axes[2, 1].set_xlabel('Pixel Position')
            axes[2, 1].set_ylabel('Pixel Value')
            axes[2, 1].legend()

            axes[2, 2].plot(profile1_v, label=f'{label1} (vertical)', alpha=0.7)
            axes[2, 2].plot(profile2_v, label=f'{label2} (vertical)', alpha=0.7)
            axes[2, 2].set_title('Center Line Profiles')
            axes[2, 2].set_xlabel('Pixel Position')
            axes[2, 2].set_ylabel('Pixel Value')
            axes[2, 2].legend()
        else:
            axes[2, 1].text(0.5, 0.5, 'Different\nShapes\nNo Profile\nComparison',
                            ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].axis('off')
            axes[2, 2].text(0.5, 0.5, 'Different\nShapes\nNo Profile\nComparison',
                            ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison visualization saved to: {save_path}")

        plt.show()


class DEAPICapture:
    """Test01.py style DEAPI direct capture"""

    def __init__(self):
        self.client = None

    def connect(self, host="localhost", port=13240):
        """Connect using DEAPI direct approach"""
        if not DEAPI_AVAILABLE:
            raise ImportError("DEAPI not available")

        try:
            self.client = DEAPI.Client()
            self.client.Connect(host, port)

            cameras = self.client.ListCameras()
            if not cameras:
                raise Exception("No cameras found")

            self.client.SetCurrentCamera(cameras[0])
            print(f"DEAPI connected to camera: {cameras[0]}")
            return True
        except Exception as e:
            print(f"DEAPI connection failed: {e}")
            return False

    def setup_camera(self):
        """Configure camera for single shot (test01.py style)"""
        try:
            self.client.SetProperty("Exposure Mode", "Normal")
            self.client.SetProperty("Image Processing - Mode", "Integrating")
            self.client.SetProperty("Frames Per Second", 20)
            self.client.SetProperty("Exposure Time (seconds)", 1.0)
            self.client.SetProperty("Autosave Final Image", "Off")
            self.client.SetProperty("Autosave Movie", "Off")
            return True
        except Exception as e:
            print(f"DEAPI setup failed: {e}")
            return False

    def capture_image(self):
        """Capture image using DEAPI GetResult (test01.py style)"""
        try:
            print("DEAPI: Starting image acquisition...")
            self.client.StartAcquisition(1)

            frameType = DEAPI.FrameType.SUMTOTAL
            pixelFormat = DEAPI.PixelFormat.AUTO
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()

            image, pixelFormat, attributes, histogram = self.client.GetResult(
                frameType, pixelFormat, attributes, histogram
            )

            if image is not None:
                metadata = {
                    'method': 'DEAPI_GetResult',
                    'frame_type': 'SUMTOTAL',
                    'pixel_format': str(pixelFormat),
                    'dataset': attributes.datasetName,
                    'image_min': attributes.imageMin,
                    'image_max': attributes.imageMax,
                    'image_mean': attributes.imageMean,
                    'image_std': attributes.imageStd,
                    'frame_count': attributes.frameCount
                }
                print(f"DEAPI captured: {image.shape} {image.dtype}, range: {image.min()}-{image.max()}")
                return image, metadata
            else:
                print("DEAPI: Failed to capture image")
                return None, None

        except Exception as e:
            print(f"DEAPI capture error: {e}")
            return None, None

    def disconnect(self):
        """Disconnect from DEAPI"""
        if self.client:
            try:
                self.client.Disconnect()
            except:
                pass


class PyScopeCapture:
    """Test09.py style PyScope capture"""

    def __init__(self):
        self.camera = None
        self.camera_name = None

    def connect(self, camera_name=None):
        """Connect using PyScope approach"""
        if not PYSCOPE_AVAILABLE:
            raise ImportError("PyScope not available")

        try:
            import pyscope.registry

            # Try different DE camera names
            camera_attempts = [
                "DEApollo", "DE Apollo", "DE12", "DE20", "DE16"
            ]

            if camera_name:
                camera_attempts.insert(0, camera_name)

            for attempt_name in camera_attempts:
                try:
                    print(f"PyScope: Attempting connection to: {attempt_name}")
                    self.camera = pyscope.registry.getClass(attempt_name)()
                    self.camera_name = attempt_name

                    # Test connectivity
                    camera_size = self.camera.getCameraSize()
                    print(f"PyScope connected to {attempt_name}, size: {camera_size}")
                    return True

                except Exception as e:
                    print(f"PyScope connection failed for {attempt_name}: {e}")
                    continue

            print("PyScope: All connection attempts failed")
            return False

        except Exception as e:
            print(f"PyScope connection error: {e}")
            return False

    def setup_camera(self, exposure_time_ms=1000):
        """Configure camera using PyScope approach"""
        try:
            self.camera.setExposureTime(exposure_time_ms)
            print(f"PyScope: Set exposure time to {exposure_time_ms} ms")
            return True
        except Exception as e:
            print(f"PyScope setup failed: {e}")
            return False

    def capture_image(self):
        """Capture image using PyScope getImage() (test09.py style)"""
        try:
            print("PyScope: Starting image acquisition...")
            image = self.camera.getImage()

            if image is not None:
                metadata = {
                    'method': 'PyScope_getImage',
                    'camera_name': self.camera_name,
                    'exposure_time_ms': self._get_current_exposure_time(),
                    'binning': self._get_current_binning(),
                    'dimension': self._get_current_dimension()
                }
                print(f"PyScope captured: {image.shape} {image.dtype}, range: {image.min()}-{image.max()}")
                return image, metadata
            else:
                print("PyScope: Failed to capture image")
                return None, None

        except Exception as e:
            print(f"PyScope capture error: {e}")
            return None, None

    def _get_current_exposure_time(self):
        try:
            return self.camera.getExposureTime()
        except:
            return -1.0

    def _get_current_binning(self):
        try:
            return self.camera.getBinning()
        except:
            return {'x': 1, 'y': 1}

    def _get_current_dimension(self):
        try:
            return self.camera.getDimension()
        except:
            return {'x': -1, 'y': -1}


def main():
    """Main diagnostic function to compare the two approaches"""
    print("=" * 80)
    print("DE Camera Image Comparison Diagnostic")
    print("Comparing test01.py (DEAPI) vs test09.py (PyScope) approaches")
    print("=" * 80)

    diagnostics = ImageDiagnostics()

    # Test both approaches
    deapi_image = None
    pyscope_image = None
    deapi_metadata = None
    pyscope_metadata = None

    # Test DEAPI approach
    print("\n1. Testing DEAPI approach (test01.py style)...")
    if DEAPI_AVAILABLE:
        deapi_capture = DEAPICapture()
        if deapi_capture.connect():
            if deapi_capture.setup_camera():
                deapi_image, deapi_metadata = deapi_capture.capture_image()
            deapi_capture.disconnect()
    else:
        print("   DEAPI not available - skipping")

    # Test PyScope approach
    print("\n2. Testing PyScope approach (test09.py style)...")
    if PYSCOPE_AVAILABLE:
        pyscope_capture = PyScopeCapture()
        if pyscope_capture.connect():
            if pyscope_capture.setup_camera():
                pyscope_image, pyscope_metadata = pyscope_capture.capture_image()
    else:
        print("   PyScope not available - skipping")

    # Compare results
    print("\n3. Analyzing differences...")

    if deapi_image is not None and pyscope_image is not None:
        print("\n✓ Both methods captured images successfully!")

        # Comprehensive comparison
        comparison = diagnostics.compare_images(
            deapi_image, pyscope_image,
            "DEAPI_GetResult", "PyScope_getImage"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("IMAGE COMPARISON SUMMARY")
        print("=" * 60)

        deapi_stats = comparison['image1_stats']
        pyscope_stats = comparison['image2_stats']

        print(f"\nDEAPI Image ({deapi_stats['source']}):")
        print(f"  Shape: {deapi_stats['shape']}")
        print(f"  Data type: {deapi_stats['dtype']}")
        print(f"  Range: {deapi_stats['min_value']:.2f} to {deapi_stats['max_value']:.2f}")
        print(f"  Mean ± Std: {deapi_stats['mean_value']:.2f} ± {deapi_stats['std_value']:.2f}")
        print(f"  Bit depth used: {deapi_stats['bit_depth_used']:.1f} bits")

        print(f"\nPyScope Image ({pyscope_stats['source']}):")
        print(f"  Shape: {pyscope_stats['shape']}")
        print(f"  Data type: {pyscope_stats['dtype']}")
        print(f"  Range: {pyscope_stats['min_value']:.2f} to {pyscope_stats['max_value']:.2f}")
        print(f"  Mean ± Std: {pyscope_stats['mean_value']:.2f} ± {pyscope_stats['std_value']:.2f}")
        print(f"  Bit depth used: {pyscope_stats['bit_depth_used']:.1f} bits")

        # Key differences
        differences = comparison['differences']
        print(f"\nKEY DIFFERENCES:")
        print(f"  Same shape: {differences['shape_same']}")
        print(f"  Same data type: {differences['dtype_same']}")

        if 'range_comparison' in differences:
            range_ratio = differences['range_comparison']['range_ratio']
            print(f"  Dynamic range ratio (PyScope/DEAPI): {range_ratio:.3f}")

            if range_ratio > 1.1:
                print(f"    → PyScope image has {range_ratio:.2f}x larger dynamic range")
            elif range_ratio < 0.9:
                print(f"    → DEAPI image has {1 / range_ratio:.2f}x larger dynamic range")
            else:
                print(f"    → Similar dynamic ranges")

        if differences['shape_same'] and 'pixel_differences' in differences:
            pixel_diff = differences['pixel_differences']
            correlation = pixel_diff['correlation']
            print(f"  Pixel correlation: {correlation:.6f}")

            if correlation > 0.99:
                print(f"    → Images are highly correlated (likely same data, different scaling)")
            elif correlation > 0.95:
                print(f"    → Images are well correlated (similar patterns)")
            else:
                print(f"    → Images have different patterns")

            print(f"  Max absolute pixel difference: {pixel_diff['max_absolute_diff']:.2f}")
            print(f"  Mean absolute pixel difference: {pixel_diff['mean_absolute_diff']:.2f}")

        # Normalization analysis
        if differences['shape_same'] and 'normalization_analysis' in differences:
            norm_analysis = differences['normalization_analysis']
            print(f"\nNORMALIZATION ANALYSIS:")
            print(f"  Standard normalization correlation: {norm_analysis['standard_normalization_correlation']:.6f}")
            print(f"  Min-max normalization correlation: {norm_analysis['minmax_normalization_correlation']:.6f}")
            print(f"  Z-score normalization correlation: {norm_analysis['zscore_normalization_correlation']:.6f}")

        # Root cause analysis
        print(f"\n" + "=" * 60)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 60)

        # Check metadata differences
        print(f"\nMetadata Comparison:")
        if deapi_metadata:
            print(f"DEAPI metadata: {deapi_metadata}")
        if pyscope_metadata:
            print(f"PyScope metadata: {pyscope_metadata}")

        # Identify likely causes
        print(f"\nLikely causes of differences:")

        if not differences['dtype_same']:
            print(f"• Different data types: {deapi_stats['dtype']} vs {pyscope_stats['dtype']}")
            print(f"  This affects the numerical range and precision of pixel values")

        if differences['shape_same'] and 'pixel_differences' in differences:
            correlation = differences['pixel_differences']['correlation']
            if correlation > 0.99 and differences['range_comparison']['range_ratio'] != 1.0:
                print(f"• Images appear to be the same data with different scaling/normalization")
                print(f"  PyScope may apply additional processing or use different data types")

                # Check if it's a simple scaling
                if abs(differences['range_comparison']['range_ratio'] - 256) < 1:
                    print(f"  → Likely 8-bit vs 16-bit conversion (256x difference)")
                elif abs(differences['range_comparison']['range_ratio'] - 65536) < 100:
                    print(f"  → Likely 16-bit vs 32-bit conversion (65536x difference)")

        # Check for geometry processing differences
        if hasattr(pyscope_capture, 'camera') and pyscope_capture.camera:
            print(f"• PyScope applies finalizeGeometry() processing which may include:")
            print(f"  - ROI cropping based on offset and dimensions")
            print(f"  - Binning operations")
            print(f"  - Image flipping (numpy.fliplr)")
            print(f"  - Data type conversions")

        if deapi_metadata and 'pixel_format' in deapi_metadata:
            print(f"• DEAPI uses pixel format: {deapi_metadata['pixel_format']}")
            print(f"  This directly controls the output data type")

        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed comparison to JSON
        json_filename = f"image_comparison_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n📁 Detailed comparison saved to: {json_filename}")

        # Create and save visualization
        viz_filename = f"image_comparison_visualization_{timestamp}.png"
        diagnostics.visualize_comparison(
            deapi_image, pyscope_image,
            "DEAPI_GetResult", "PyScope_getImage",
            viz_filename
        )

        # Test different normalization approaches
        print(f"\n" + "=" * 60)
        print("NORMALIZATION TESTING")
        print("=" * 60)

        print(f"\nTesting if images become identical after normalization...")

        # Test 1: Min-max normalization to 0-1
        def normalize_minmax(img):
            return (img - img.min()) / (img.max() - img.min())

        norm_deapi = normalize_minmax(deapi_image.astype(np.float64))
        norm_pyscope = normalize_minmax(pyscope_image.astype(np.float64))

        if norm_deapi.shape == norm_pyscope.shape:
            max_diff = np.max(np.abs(norm_deapi - norm_pyscope))
            correlation = np.corrcoef(norm_deapi.flatten(), norm_pyscope.flatten())[0, 1]
            print(f"  Min-max normalized (0-1): max_diff={max_diff:.6f}, correlation={correlation:.6f}")

            if max_diff < 1e-6:
                print(f"    ✓ Images are IDENTICAL after min-max normalization!")
                print(f"    → The difference is purely in scaling/offset")
            elif correlation > 0.999:
                print(f"    ✓ Images are nearly identical after normalization")
                print(f"    → Small differences may be due to data type precision")

        # Test 2: Check if one is a simple linear transformation of the other
        if deapi_image.shape == pyscope_image.shape:
            # Flatten for easier analysis
            flat_deapi = deapi_image.flatten().astype(np.float64)
            flat_pyscope = pyscope_image.flatten().astype(np.float64)

            # Try to find linear relationship: pyscope = a * deapi + b
            try:
                # Use least squares to find best fit line
                A = np.vstack([flat_deapi, np.ones(len(flat_deapi))]).T
                a, b = np.linalg.lstsq(A, flat_pyscope, rcond=None)[0]

                # Calculate how well this linear relationship fits
                predicted = a * flat_deapi + b
                residuals = flat_pyscope - predicted
                r_squared = 1 - (np.sum(residuals ** 2) / np.sum((flat_pyscope - np.mean(flat_pyscope)) ** 2))

                print(f"\n  Linear relationship test: PyScope = {a:.6f} * DEAPI + {b:.6f}")
                print(f"  R-squared: {r_squared:.6f}")

                if r_squared > 0.999:
                    print(f"    ✓ Strong linear relationship found!")
                    if abs(b) < 1e-6:
                        print(f"    → PyScope image is DEAPI image scaled by {a:.6f}")
                    else:
                        print(f"    → PyScope image is DEAPI image scaled by {a:.6f} with offset {b:.6f}")
                elif r_squared > 0.99:
                    print(f"    ✓ Good linear relationship found with some noise")
                else:
                    print(f"    ❌ No strong linear relationship found")

            except Exception as e:
                print(f"  Could not compute linear relationship: {e}")

    elif deapi_image is not None:
        print("\n✓ DEAPI captured image successfully")
        print("❌ PyScope capture failed")

        # Analyze DEAPI image only
        deapi_stats = diagnostics.analyze_raw_image_data(deapi_image, "DEAPI_only")
        print(f"\nDEAPI Image Analysis:")
        print(f"  Shape: {deapi_stats['shape']}")
        print(f"  Data type: {deapi_stats['dtype']}")
        print(f"  Range: {deapi_stats['min_value']:.2f} to {deapi_stats['max_value']:.2f}")
        print(f"  Mean ± Std: {deapi_stats['mean_value']:.2f} ± {deapi_stats['std_value']:.2f}")

    elif pyscope_image is not None:
        print("❌ DEAPI capture failed")
        print("✓ PyScope captured image successfully")

        # Analyze PyScope image only
        pyscope_stats = diagnostics.analyze_raw_image_data(pyscope_image, "PyScope_only")
        print(f"\nPyScope Image Analysis:")
        print(f"  Shape: {pyscope_stats['shape']}")
        print(f"  Data type: {pyscope_stats['dtype']}")
        print(f"  Range: {pyscope_stats['min_value']:.2f} to {pyscope_stats['max_value']:.2f}")
        print(f"  Mean ± Std: {pyscope_stats['mean_value']:.2f} ± {pyscope_stats['std_value']:.2f}")

    else:
        print("❌ Both DEAPI and PyScope captures failed")
        print("\nPossible reasons:")
        print("• No DE camera connected")
        print("• Camera driver issues")
        print("• PyScope or DEAPI installation problems")
        print("• Camera is in use by another application")

    # Recommendations
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if deapi_image is not None and pyscope_image is not None:
        differences = diagnostics.compare_images(
            deapi_image, pyscope_image,
            "DEAPI", "PyScope"
        )['differences']

        if differences['shape_same'] and 'pixel_differences' in differences:
            correlation = differences['pixel_differences']['correlation']

            if correlation > 0.99:
                print("\n✓ Images are highly correlated - difference is mainly in data processing")
                print("\nTo make images more similar:")
                print("1. Use the same pixel format in both approaches:")
                print("   - DEAPI: Set pixelFormat = DEAPI.PixelFormat.UINT16")
                print("   - Ensure PyScope returns same data type")
                print("\n2. Apply consistent normalization:")
                print("   - Use identical min-max scaling")
                print("   - Apply same bit depth conversion")
                print("\n3. Check PyScope finalizeGeometry() processing:")
                print("   - This may apply ROI cropping, binning, or flipping")
                print("   - Consider bypassing or matching this processing")
            else:
                print("\n⚠️ Images have significant differences beyond scaling")
                print("\nInvestigate:")
                print("1. Different camera settings between approaches")
                print("2. Different image processing pipelines")
                print("3. Different ROI or binning settings")
                print("4. Timing differences in acquisition")

    print("\n4. For consistent image output across both methods:")
    print("   - Standardize the pixel format (uint16 recommended)")
    print("   - Use identical normalization functions")
    print("   - Match camera settings exactly")
    print("   - Consider bypassing PyScope's finalizeGeometry if needed")

    print(f"\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


def test_normalization_functions():
    """Test different normalization approaches used in test01.py vs test09.py"""
    print("\n" + "=" * 60)
    print("NORMALIZATION FUNCTION TESTING")
    print("=" * 60)

    # Create test images with different characteristics
    test_images = {
        'uint16_image': np.random.randint(0, 65536, (100, 100), dtype=np.uint16),
        'float32_image': np.random.random((100, 100)).astype(np.float32) * 1000,
        'uint8_image': np.random.randint(0, 256, (100, 100), dtype=np.uint8),
        'high_dynamic_range': np.random.exponential(100, (100, 100)).astype(np.float32)
    }

    def normalize_test01_style(image):
        """test01.py normalization approach"""
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
        return image_8bit

    def normalize_test09_style(image):
        """test09.py normalization approach"""
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
        return image_8bit

    print("\nTesting normalization functions on different image types:")

    for img_name, img in test_images.items():
        print(f"\n{img_name}:")
        print(f"  Original: {img.dtype}, range {img.min():.2f}-{img.max():.2f}")

        norm1 = normalize_test01_style(img)
        norm2 = normalize_test09_style(img)

        print(f"  test01 style: {norm1.dtype}, range {norm1.min()}-{norm1.max()}")
        print(f"  test09 style: {norm2.dtype}, range {norm2.min()}-{norm2.max()}")

        # Check if identical
        identical = np.array_equal(norm1, norm2)
        print(f"  Identical results: {identical}")

        if not identical:
            max_diff = np.max(np.abs(norm1.astype(np.float32) - norm2.astype(np.float32)))
            print(f"  Max difference: {max_diff:.2f}")


if __name__ == "__main__":
    main()

    # Optional: Test normalization functions
    test_normalization_functions()