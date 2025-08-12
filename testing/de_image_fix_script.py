#!/usr/bin/env python3
"""
DE Camera Image Consistency Fix
Based on diagnostic findings, this script provides corrected versions of both approaches
to generate consistent images between DEAPI and PyScope methods
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Add DEAPI path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]

try:
    from pyscope import DEAPI
    DEAPI_AVAILABLE = True
except ImportError:
    DEAPI_AVAILABLE = False

try:
    import pyscope.registry
    PYSCOPE_AVAILABLE = True
except ImportError:
    PYSCOPE_AVAILABLE = False


class ConsistentDECapture:
    """Consistent DE camera capture for both DEAPI and PyScope approaches"""
    
    def __init__(self):
        self.target_dtype = np.uint16  # Use consistent data type
        self.target_pixel_format = DEAPI.PixelFormat.UINT16 if DEAPI_AVAILABLE else None
    
    def setup_consistent_camera_properties(self, camera_obj, method="deapi"):
        """Apply identical camera settings for both methods"""
        
        common_properties = {
            "Exposure Mode": "Normal",
            "Image Processing - Mode": "Integrating", 
            "Frames Per Second": 20,
            "Exposure Time (seconds)": 1.0,
            "Autosave Final Image": "Off",
            "Autosave Movie": "Off",
            "Correction Mode": "Uncorrected Raw"  # Ensure raw data
        }
        
        if method == "deapi":
            # DEAPI direct property setting
            for prop_name, prop_value in common_properties.items():
                try:
                    camera_obj.SetProperty(prop_name, prop_value)
                    print(f"  DEAPI set {prop_name}: {prop_value}")
                except Exception as e:
                    print(f"  DEAPI failed to set {prop_name}: {e}")
        
        elif method == "pyscope":
            # PyScope property setting
            for prop_name, prop_value in common_properties.items():
                try:
                    camera_obj.setProperty(prop_name, prop_value)
                    print(f"  PyScope set {prop_name}: {prop_value}")
                except Exception as e:
                    print(f"  PyScope failed to set {prop_name}: {e}")
            
            # Fix PyScope geometry issues
            try:
                camera_size = camera_obj.getCameraSize()
                print(f"  PyScope camera size: {camera_size}")
                
                # Set proper dimensions instead of {x:0, y:0}
                camera_obj.setDimension(camera_size)
                camera_obj.setOffset({'x': 0, 'y': 0})
                camera_obj.setBinning({'x': 1, 'y': 1})
                
                print(f"  PyScope set dimension: {camera_obj.getDimension()}")
                print(f"  PyScope set offset: {camera_obj.getOffset()}")
                print(f"  PyScope set binning: {camera_obj.getBinning()}")
                
            except Exception as e:
                print(f"  PyScope geometry setup failed: {e}")
    
    def capture_deapi_consistent(self, host="localhost", port=13240):
        """Capture using DEAPI with consistent settings"""
        if not DEAPI_AVAILABLE:
            print("DEAPI not available")
            return None, None
        
        try:
            # Connect
            client = DEAPI.Client()
            client.Connect(host, port)
            cameras = client.ListCameras()
            if not cameras:
                raise Exception("No cameras found")
            client.SetCurrentCamera(cameras[0])
            print(f"DEAPI connected to: {cameras[0]}")
            
            # Apply consistent settings
            self.setup_consistent_camera_properties(client, "deapi")
            
            # Capture with UINT16 format (not AUTO)
            print(f"DEAPI capturing with UINT16 format...")
            client.StartAcquisition(1)
            
            frameType = DEAPI.FrameType.SUMTOTAL
            pixelFormat = DEAPI.PixelFormat.UINT16  # FIXED: Use UINT16 instead of AUTO
            attributes = DEAPI.Attributes()
            histogram = DEAPI.Histogram()
            
            image, pixelFormat, attributes, histogram = client.GetResult(
                frameType, pixelFormat, attributes, histogram
            )
            
            if image is not None:
                metadata = {
                    'method': 'DEAPI_Consistent_UINT16',
                    'frame_type': 'SUMTOTAL',
                    'pixel_format': str(pixelFormat),
                    'dataset': attributes.datasetName,
                    'image_min': attributes.imageMin,
                    'image_max': attributes.imageMax,
                    'image_mean': attributes.imageMean,
                    'image_std': attributes.imageStd,
                    'frame_count': attributes.frameCount,
                    'shape': image.shape,
                    'dtype': str(image.dtype)
                }
                print(f"DEAPI consistent captured: {image.shape} {image.dtype}, range: {image.min()}-{image.max()}")
                
                client.Disconnect()
                return image, metadata
            else:
                print("DEAPI consistent capture failed")
                client.Disconnect()
                return None, None
                
        except Exception as e:
            print(f"DEAPI consistent capture error: {e}")
            return None, None
    
    def capture_pyscope_consistent(self, camera_name=None):
        """Capture using PyScope with consistent settings"""
        if not PYSCOPE_AVAILABLE:
            print("PyScope not available")
            return None, None
        
        try:
            import pyscope.registry
            
            # Connect to camera
            camera_attempts = ["DEApollo", "DE Apollo", "DE12", "DE20", "DE16"]
            if camera_name:
                camera_attempts.insert(0, camera_name)
            
            camera = None
            camera_name_used = None
            
            for attempt_name in camera_attempts:
                try:
                    camera = pyscope.registry.getClass(attempt_name)()
                    camera_name_used = attempt_name
                    camera_size = camera.getCameraSize()
                    print(f"PyScope consistent connected to: {attempt_name}")
                    break
                except Exception as e:
                    continue
            
            if not camera:
                print("PyScope consistent connection failed")
                return None, None
            
            # Apply consistent settings
            self.setup_consistent_camera_properties(camera, "pyscope")
            
            # Set exposure time
            camera.setExposureTime(1000)
            
            # Capture using RAW method if available
            print(f"PyScope capturing (attempting raw method)...")
            
            # Method 1: Try raw capture first
            image = None
            capture_method = None
            
            if hasattr(camera, '_getImage'):
                try:
                    print("  Trying _getImage (raw)...")
                    image = camera._getImage()
                    capture_method = "PyScope_Raw_getImage"
                    print(f"  ✓ Raw capture successful: {image.shape} {image.dtype}")
                except Exception as e:
                    print(f"  Raw capture failed: {e}")
            
            # Method 2: Standard capture with geometry bypass attempt
            if image is None:
                try:
                    print("  Trying standard getImage...")
                    image = camera.getImage()
                    capture_method = "PyScope_Standard_getImage"
                    print(f"  ✓ Standard capture successful: {image.shape} {image.dtype}")
                except Exception as e:
                    print(f"  Standard capture failed: {e}")
                    return None, None
            
            # Convert to consistent data type if needed
            if image.dtype != self.target_dtype:
                print(f"  Converting {image.dtype} to {self.target_dtype}...")
                if image.dtype == np.float32:
                    # Scale float32 to uint16 range
                    image_normalized = (image - image.min()) / (image.max() - image.min())
                    image = (image_normalized * 65535).astype(self.target_dtype)
                else:
                    image = image.astype(self.target_dtype)
                print(f"  ✓ Converted to {image.dtype}")
            
            if image is not None:
                metadata = {
                    'method': capture_method,
                    'camera_name': camera_name_used,
                    'exposure_time_ms': camera.getExposureTime() if hasattr(camera, 'getExposureTime') else 1000,
                    'binning': camera.getBinning() if hasattr(camera, 'getBinning') else {'x': 1, 'y': 1},
                    'dimension': camera.getDimension() if hasattr(camera, 'getDimension') else {'x': -1, 'y': -1},
                    'offset': camera.getOffset() if hasattr(camera, 'getOffset') else {'x': 0, 'y': 0},
                    'shape': image.shape,
                    'dtype': str(image.dtype)
                }
                print(f"PyScope consistent captured: {image.shape} {image.dtype}, range: {image.min()}-{image.max()}")
                return image, metadata
            else:
                print("PyScope consistent capture failed")
                return None, None
                
        except Exception as e:
            print(f"PyScope consistent capture error: {e}")
            return None, None
    
    def capture_pyscope_with_manual_geometry(self, camera_name=None):
        """Alternative PyScope capture manually handling geometry"""
        if not PYSCOPE_AVAILABLE:
            return None, None
        
        try:
            import pyscope.registry
            
            # Connect
            camera = pyscope.registry.getClass("DEApollo")()
            print("PyScope manual geometry connected")
            
            # Apply settings
            self.setup_consistent_camera_properties(camera, "pyscope")
            camera.setExposureTime(1000)
            
            # Get raw image and manually apply minimal processing
            print("Capturing with manual geometry handling...")
            
            if hasattr(camera, '_getImage'):
                raw_image = camera._getImage()
                print(f"  Raw image: {raw_image.shape} {raw_image.dtype}")
                
                # Apply only essential geometry corrections
                # Skip the problematic finalizeGeometry() processing
                processed_image = raw_image
                
                # Only apply data type conversion if needed
                if processed_image.dtype != self.target_dtype:
                    if processed_image.dtype == np.float32:
                        processed_image = (processed_image / processed_image.max() * 65535).astype(self.target_dtype)
                    else:
                        processed_image = processed_image.astype(self.target_dtype)
                
                metadata = {
                    'method': 'PyScope_Manual_Geometry',
                    'camera_name': 'DEApollo',
                    'processing': 'Minimal_geometry_correction',
                    'shape': processed_image.shape,
                    'dtype': str(processed_image.dtype)
                }
                
                print(f"PyScope manual captured: {processed_image.shape} {processed_image.dtype}, range: {processed_image.min()}-{processed_image.max()}")
                return processed_image, metadata
            else:
                print("Raw image method not available")
                return None, None
                
        except Exception as e:
            print(f"PyScope manual geometry error: {e}")
            return None, None
    
    def compare_images_detailed(self, img1, img2, label1, label2):
        """Detailed comparison of two images"""
        print(f"\n🔍 Detailed Comparison: {label1} vs {label2}")
        
        if img1 is None or img2 is None:
            print("  Cannot compare - one or both images are None")
            return
        
        print(f"  {label1}: {img1.shape} {img1.dtype}, range {img1.min()}-{img1.max()}")
        print(f"  {label2}: {img2.shape} {img2.dtype}, range {img2.min()}-{img2.max()}")
        
        if img1.shape != img2.shape:
            print("  ❌ Different shapes - cannot compare pixels")
            return
        
        # Check if identical
        if np.array_equal(img1, img2):
            print("  ✅ Images are IDENTICAL!")
            return
        
        # Calculate correlation
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        print(f"  Correlation: {correlation:.6f}")
        
        if correlation > 0.99:
            print("  ✅ Very high correlation - likely same data with minor differences")
        elif correlation > 0.95:
            print("  ✅ High correlation - similar images")
        elif correlation > 0.5:
            print("  ⚠️ Moderate correlation - some similarities")
        else:
            print("  ❌ Low correlation - different images")
        
        # Check for simple transformations
        if np.array_equal(img1, np.fliplr(img2)):
            print("  ✅ Images are identical after horizontal flip")
        elif np.array_equal(img1, np.flipud(img2)):
            print("  ✅ Images are identical after vertical flip")
        elif np.array_equal(img1, np.rot90(img2)):
            print("  ✅ Images are identical after 90° rotation")
        
        # Statistical comparison
        diff = img1.astype(np.float64) - img2.astype(np.float64)
        print(f"  Max absolute difference: {np.max(np.abs(diff)):.2f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(diff)):.2f}")
        print(f"  RMS difference: {np.sqrt(np.mean(diff**2)):.2f}")
    
    def save_comparison_images(self, images_dict, output_dir="comparison_output"):
        """Save all captured images for visual comparison"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for label, (image, metadata) in images_dict.items():
            if image is not None:
                # Normalize for display
                if image.dtype in [np.float32, np.float64]:
                    normalized = (image - image.min()) / (image.max() - image.min())
                    display_image = (normalized * 255).astype(np.uint8)
                elif image.dtype == np.uint16:
                    display_image = (image / 256).astype(np.uint8)
                else:
                    display_image = image
                
                # Save image
                pil_image = Image.fromarray(display_image, mode='L')
                filename = f"{label}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                pil_image.save(filepath)
                print(f"  Saved {label}: {filepath}")
                
                # Save metadata
                metadata_filename = f"{label}_{timestamp}_metadata.json"
                metadata_filepath = os.path.join(output_dir, metadata_filename)
                import json
                with open(metadata_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
    
    def create_comparison_visualization(self, images_dict):
        """Create side-by-side comparison visualization"""
        valid_images = {k: v for k, v in images_dict.items() if v[0] is not None}
        
        if len(valid_images) < 2:
            print("Need at least 2 images for comparison")
            return
        
        n_images = len(valid_images)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for i, (label, (image, metadata)) in enumerate(valid_images.items()):
            # Normalize for display
            if image.dtype in [np.float32, np.float64]:
                normalized = (image - image.min()) / (image.max() - image.min())
            elif image.dtype == np.uint16:
                normalized = image / 65535.0
            else:
                normalized = image / 255.0
            
            # Top row: Full image
            axes[0, i].imshow(normalized, cmap='gray')
            axes[0, i].set_title(f'{label}\n{image.shape} {image.dtype}\n{image.min()}-{image.max()}')
            axes[0, i].axis('off')
            
            # Bottom row: Center crop for detail
            center_y, center_x = image.shape[0]//2, image.shape[1]//2
            crop_size = 200
            y1, y2 = center_y - crop_size, center_y + crop_size
            x1, x2 = center_x - crop_size, center_x + crop_size
            
            if y1 >= 0 and y2 < image.shape[0] and x1 >= 0 and x2 < image.shape[1]:
                crop = normalized[y1:y2, x1:x2]
                axes[1, i].imshow(crop, cmap='gray')
                axes[1, i].set_title(f'Center Detail\n{crop_size*2}x{crop_size*2} pixels')
            else:
                axes[1, i].text(0.5, 0.5, 'Crop\nN/A', ha='center', va='center', 
                               transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to test consistent capture methods"""
    print("=" * 80)
    print("DE CAMERA IMAGE CONSISTENCY FIX")
    print("Testing corrected capture methods for identical results")
    print("=" * 80)
    
    capture = ConsistentDECapture()
    results = {}
    
    # Test 1: DEAPI with UINT16 format
    print("\n1. Testing DEAPI with consistent UINT16 format...")
    deapi_image, deapi_metadata = capture.capture_deapi_consistent()
    if deapi_image is not None:
        results['DEAPI_UINT16'] = (deapi_image, deapi_metadata)
        print("  ✅ DEAPI consistent capture successful")
    else:
        print("  ❌ DEAPI consistent capture failed")
    
    # Test 2: PyScope with consistent settings
    print("\n2. Testing PyScope with consistent settings...")
    pyscope_image, pyscope_metadata = capture.capture_pyscope_consistent()
    if pyscope_image is not None:
        results['PyScope_Consistent'] = (pyscope_image, pyscope_metadata)
        print("  ✅ PyScope consistent capture successful")
    else:
        print("  ❌ PyScope consistent capture failed")
    
    # Test 3: PyScope with manual geometry
    print("\n3. Testing PyScope with manual geometry...")
    pyscope_manual_image, pyscope_manual_metadata = capture.capture_pyscope_with_manual_geometry()
    if pyscope_manual_image is not None:
        results['PyScope_Manual'] = (pyscope_manual_image, pyscope_manual_metadata)
        print("  ✅ PyScope manual capture successful")
    else:
        print("  ❌ PyScope manual capture failed")
    
    # Compare all results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if len(results) >= 2:
        print(f"\n📊 Captured {len(results)} images for comparison:")
        for label in results.keys():
            image, metadata = results[label]
            print(f"  {label}: {image.shape} {image.dtype}, range {image.min()}-{image.max()}")
        
        # Compare each pair
        result_items = list(results.items())
        for i in range(len(result_items)):
            for j in range(i+1, len(result_items)):
                label1, (img1, meta1) = result_items[i]
                label2, (img2, meta2) = result_items[j]
                capture.compare_images_detailed(img1, img2, label1, label2)
        
        # Save all images
        print(f"\n💾 Saving comparison images...")
        capture.save_comparison_images(results)
        
        # Create visualization
        print(f"\n📊 Creating comparison visualization...")
        capture.create_comparison_visualization(results)
        
    else:
        print("\n❌ Need at least 2 successful captures for comparison")
        for label, (image, metadata) in results.items():
            if image is not None:
                print(f"  {label}: Available")
            else:
                print(f"  {label}: Failed")
    
    # Final recommendations based on results
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)
    
    if 'DEAPI_UINT16' in results and 'PyScope_Consistent' in results:
        img1, _ = results['DEAPI_UINT16']
        img2, _ = results['PyScope_Consistent']
        
        if img1.shape == img2.shape:
            correlation = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
            
            if correlation > 0.99:
                print("\n✅ SUCCESS: Images are now highly consistent!")
                print("   Recommended approach:")
                print("   - Use DEAPI with PixelFormat.UINT16")
                print("   - Use PyScope with proper dimension settings")
                print("   - Apply identical camera properties")
            elif correlation > 0.95:
                print("\n✅ GOOD: Images are much more consistent")
                print("   Minor differences may be due to:")
                print("   - Slight timing differences")
                print("   - Small property variations")
            else:
                print("\n⚠️ PARTIAL: Images still have significant differences")
                print("   Additional investigation needed:")
                print("   - Check for different correction modes")
                print("   - Verify exact property matching")
                print("   - Consider hardware timing issues")
    
    if 'PyScope_Manual' in results:
        print("\n💡 PyScope Manual Geometry approach:")
        print("   This bypasses finalizeGeometry() processing")
        print("   Use if standard PyScope gives inconsistent results")
    
    print("\n🎯 Key Fixes Applied:")
    print("   1. DEAPI: Use PixelFormat.UINT16 instead of AUTO")
    print("   2. PyScope: Set proper dimensions instead of {x:0, y:0}")
    print("   3. Both: Apply identical camera properties")
    print("   4. Both: Use consistent data types (uint16)")
    print("   5. PyScope: Option to bypass finalizeGeometry()")


if __name__ == "__main__":
    main()