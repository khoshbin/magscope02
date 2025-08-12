#!/usr/bin/env python3
"""
Targeted Diagnostic: Identify Exact Cause of DE Camera Image Differences
Focus on PyScope's finalizeGeometry and camera settings differences
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

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


class DetailedDEDiagnostic:
    """Detailed diagnostic focusing on PyScope vs DEAPI differences"""
    
    def __init__(self):
        self.results = {}
        
    def capture_deapi_with_different_formats(self, client):
        """Capture DEAPI images with different pixel formats"""
        results = {}
        
        formats_to_test = [
            ("AUTO", DEAPI.PixelFormat.AUTO),
            ("UINT16", DEAPI.PixelFormat.UINT16),
            ("FLOAT32", DEAPI.PixelFormat.FLOAT32),
            ("UINT8", DEAPI.PixelFormat.UINT8)
        ]
        
        for name, pixel_format in formats_to_test:
            try:
                print(f"  Testing DEAPI with {name} format...")
                client.StartAcquisition(1)
                
                frameType = DEAPI.FrameType.SUMTOTAL
                attributes = DEAPI.Attributes()
                histogram = DEAPI.Histogram()
                
                image, returned_format, attributes, histogram = client.GetResult(
                    frameType, pixel_format, attributes, histogram
                )
                
                if image is not None:
                    results[name] = {
                        'image': image,
                        'format_requested': name,
                        'format_returned': str(returned_format),
                        'shape': image.shape,
                        'dtype': str(image.dtype),
                        'min': float(image.min()),
                        'max': float(image.max()),
                        'mean': float(image.mean()),
                        'std': float(image.std()),
                        'frame_count': attributes.frameCount,
                        'dataset': attributes.datasetName
                    }
                    print(f"    ✓ {name}: {image.shape} {image.dtype}, range: {image.min():.2f}-{image.max():.2f}")
                else:
                    print(f"    ❌ {name}: Failed to capture")
                    
                time.sleep(0.5)  # Small delay between captures
                
            except Exception as e:
                print(f"    ❌ {name}: Error - {e}")
                
        return results
    
    def analyze_pyscope_processing_steps(self, camera):
        """Analyze PyScope's internal processing steps"""
        print("\n🔍 Analyzing PyScope processing steps...")
        
        try:
            # Get camera properties that affect image processing
            properties = {}
            
            # Critical properties for image processing
            critical_props = [
                'Correction Mode',
                'Image Processing - Mode', 
                'Binning Mode',
                'Image Size X',
                'Image Size Y',
                'ROI Dimension X',
                'ROI Dimension Y', 
                'ROI Offset X',
                'ROI Offset Y',
                'Sensor Size X (pixels)',
                'Sensor Size Y (pixels)'
            ]
            
            for prop in critical_props:
                try:
                    value = camera.getProperty(prop)
                    properties[prop] = value
                    print(f"  {prop}: {value}")
                except Exception as e:
                    print(f"  {prop}: ERROR - {e}")
            
            # Get PyScope geometry settings
            try:
                camera_size = camera.getCameraSize()
                dimension = camera.getDimension() 
                offset = camera.getOffset()
                binning = camera.getBinning()
                
                geometry = {
                    'camera_size': camera_size,
                    'dimension': dimension,
                    'offset': offset,
                    'binning': binning
                }
                
                print(f"\n📐 PyScope Geometry Settings:")
                for key, value in geometry.items():
                    print(f"  {key}: {value}")
                    
            except Exception as e:
                print(f"  Error getting geometry: {e}")
                geometry = {}
            
            return properties, geometry
            
        except Exception as e:
            print(f"Error analyzing PyScope properties: {e}")
            return {}, {}
    
    def capture_pyscope_raw_vs_processed(self, camera):
        """Compare PyScope raw vs processed images"""
        print("\n🔬 Comparing PyScope raw vs processed images...")
        
        results = {}
        
        # Method 1: Try to get raw image (bypass finalizeGeometry)
        try:
            print("  Attempting to capture raw image...")
            if hasattr(camera, '_getImage'):
                raw_image = camera._getImage()
                results['raw'] = {
                    'image': raw_image,
                    'method': '_getImage (raw)',
                    'shape': raw_image.shape,
                    'dtype': str(raw_image.dtype),
                    'min': float(raw_image.min()),
                    'max': float(raw_image.max()),
                    'mean': float(raw_image.mean()),
                    'std': float(raw_image.std())
                }
                print(f"    ✓ Raw: {raw_image.shape} {raw_image.dtype}, range: {raw_image.min():.2f}-{raw_image.max():.2f}")
            else:
                print("    ❌ _getImage method not available")
        except Exception as e:
            print(f"    ❌ Raw capture failed: {e}")
        
        # Method 2: Standard processed image
        try:
            print("  Capturing processed image...")
            processed_image = camera.getImage()
            results['processed'] = {
                'image': processed_image,
                'method': 'getImage (processed)',
                'shape': processed_image.shape,
                'dtype': str(processed_image.dtype),
                'min': float(processed_image.min()),
                'max': float(processed_image.max()),
                'mean': float(processed_image.mean()),
                'std': float(processed_image.std())
            }
            print(f"    ✓ Processed: {processed_image.shape} {processed_image.dtype}, range: {processed_image.min():.2f}-{processed_image.max():.2f}")
        except Exception as e:
            print(f"    ❌ Processed capture failed: {e}")
        
        # Compare if both available
        if 'raw' in results and 'processed' in results:
            raw_img = results['raw']['image']
            proc_img = results['processed']['image']
            
            print(f"\n📊 Raw vs Processed Comparison:")
            
            # Check if same shape
            if raw_img.shape == proc_img.shape:
                # Check if identical
                if np.array_equal(raw_img, proc_img):
                    print(f"  ✓ Images are IDENTICAL - no processing applied")
                else:
                    # Check correlation
                    correlation = np.corrcoef(raw_img.flatten(), proc_img.flatten())[0,1]
                    print(f"  Correlation: {correlation:.6f}")
                    
                    # Check for simple transformations
                    # 1. Check flip
                    if np.array_equal(raw_img, np.fliplr(proc_img)):
                        print(f"  ✓ Processed image is horizontally flipped raw image")
                    elif np.array_equal(raw_img, np.flipud(proc_img)):
                        print(f"  ✓ Processed image is vertically flipped raw image")
                    elif np.array_equal(raw_img, np.rot90(proc_img)):
                        print(f"  ✓ Processed image is 90° rotated raw image")
                    else:
                        print(f"  ❓ Images are different - complex processing applied")
                        
                    # Check value transformations
                    if raw_img.dtype != proc_img.dtype:
                        print(f"  Data type changed: {raw_img.dtype} → {proc_img.dtype}")
                        
                    # Range comparison
                    raw_range = raw_img.max() - raw_img.min()
                    proc_range = proc_img.max() - proc_img.min()
                    if raw_range > 0:
                        range_ratio = proc_range / raw_range
                        print(f"  Range ratio (processed/raw): {range_ratio:.3f}")
            else:
                print(f"  ❓ Different shapes: raw {raw_img.shape} vs processed {proc_img.shape}")
        
        return results
    
    def test_pyscope_geometry_effects(self, camera):
        """Test effects of different PyScope geometry settings"""
        print("\n🧪 Testing PyScope geometry effects...")
        
        try:
            # Save original settings
            original_settings = {
                'dimension': camera.getDimension(),
                'offset': camera.getOffset(),
                'binning': camera.getBinning()
            }
            print(f"  Original settings: {original_settings}")
            
            results = {}
            
            # Test 1: Default settings
            print("  Test 1: Default settings")
            image1 = camera.getImage()
            results['default'] = {
                'image': image1,
                'settings': original_settings.copy(),
                'shape': image1.shape,
                'dtype': str(image1.dtype),
                'range': (float(image1.min()), float(image1.max()))
            }
            
            # Test 2: Try to set specific dimension (full sensor)
            try:
                camera_size = camera.getCameraSize()
                print(f"  Test 2: Setting dimension to full sensor {camera_size}")
                camera.setDimension(camera_size)
                camera.setOffset({'x': 0, 'y': 0})
                
                image2 = camera.getImage()
                results['full_sensor'] = {
                    'image': image2,
                    'settings': {
                        'dimension': camera.getDimension(),
                        'offset': camera.getOffset(),
                        'binning': camera.getBinning()
                    },
                    'shape': image2.shape,
                    'dtype': str(image2.dtype),
                    'range': (float(image2.min()), float(image2.max()))
                }
                print(f"    Result: {image2.shape} {image2.dtype}")
            except Exception as e:
                print(f"    Failed to set full sensor: {e}")
            
            # Restore original settings
            try:
                camera.setDimension(original_settings['dimension'])
                camera.setOffset(original_settings['offset'])
                camera.setBinning(original_settings['binning'])
            except:
                pass
            
            return results
            
        except Exception as e:
            print(f"  Error testing geometry effects: {e}")
            return {}
    
    def run_comprehensive_diagnostic(self):
        """Run complete diagnostic comparing DEAPI and PyScope"""
        print("=" * 80)
        print("COMPREHENSIVE DE CAMERA DIAGNOSTIC")
        print("=" * 80)
        
        # Test DEAPI with multiple formats
        print("\n1. DEAPI Multi-Format Testing")
        if DEAPI_AVAILABLE:
            try:
                client = DEAPI.Client()
                client.Connect()
                cameras = client.ListCameras()
                if cameras:
                    client.SetCurrentCamera(cameras[0])
                    print(f"DEAPI connected to: {cameras[0]}")
                    
                    # Setup camera
                    client.SetProperty("Exposure Mode", "Normal")
                    client.SetProperty("Image Processing - Mode", "Integrating")
                    client.SetProperty("Frames Per Second", 20)
                    client.SetProperty("Exposure Time (seconds)", 1.0)
                    client.SetProperty("Autosave Final Image", "Off")
                    client.SetProperty("Autosave Movie", "Off")
                    
                    deapi_results = self.capture_deapi_with_different_formats(client)
                    client.Disconnect()
                else:
                    print("No DEAPI cameras found")
                    deapi_results = {}
            except Exception as e:
                print(f"DEAPI testing failed: {e}")
                deapi_results = {}
        else:
            print("DEAPI not available")
            deapi_results = {}
        
        # Test PyScope detailed analysis
        print("\n2. PyScope Detailed Analysis")
        if PYSCOPE_AVAILABLE:
            try:
                import pyscope.registry
                camera = pyscope.registry.getClass('DEApollo')()
                camera.setExposureTime(1000)
                
                # Analyze properties and geometry
                properties, geometry = self.analyze_pyscope_processing_steps(camera)
                
                # Compare raw vs processed
                pyscope_raw_processed = self.capture_pyscope_raw_vs_processed(camera)
                
                # Test geometry effects
                pyscope_geometry_tests = self.test_pyscope_geometry_effects(camera)
                
            except Exception as e:
                print(f"PyScope detailed analysis failed: {e}")
                properties, geometry = {}, {}
                pyscope_raw_processed = {}
                pyscope_geometry_tests = {}
        else:
            print("PyScope not available")
            properties, geometry = {}, {}
            pyscope_raw_processed = {}
            pyscope_geometry_tests = {}
        
        # Analysis and recommendations
        print("\n" + "=" * 60)
        print("DIAGNOSTIC CONCLUSIONS")
        print("=" * 60)
        
        # Check DEAPI format consistency
        if deapi_results:
            print(f"\n📊 DEAPI Format Analysis:")
            for format_name, data in deapi_results.items():
                print(f"  {format_name}: {data['dtype']}, range {data['min']:.2f}-{data['max']:.2f}")
            
            # Check if AUTO and UINT16 give different results
            if 'AUTO' in deapi_results and 'UINT16' in deapi_results:
                auto_img = deapi_results['AUTO']['image']
                uint16_img = deapi_results['UINT16']['image']
                
                if auto_img.shape == uint16_img.shape and not np.array_equal(auto_img, uint16_img):
                    correlation = np.corrcoef(auto_img.flatten(), uint16_img.flatten())[0,1]
                    print(f"  🔍 AUTO vs UINT16 correlation: {correlation:.6f}")
                    if correlation > 0.99:
                        print(f"    → Same data, different scaling")
                    else:
                        print(f"    → Different data or processing")
        
        # Check PyScope processing pipeline
        if pyscope_raw_processed:
            print(f"\n🔬 PyScope Processing Pipeline:")
            if 'raw' in pyscope_raw_processed and 'processed' in pyscope_raw_processed:
                raw_data = pyscope_raw_processed['raw']
                proc_data = pyscope_raw_processed['processed']
                print(f"  Raw: {raw_data['dtype']}, range {raw_data['min']:.2f}-{raw_data['max']:.2f}")
                print(f"  Processed: {proc_data['dtype']}, range {proc_data['min']:.2f}-{proc_data['max']:.2f}")
                
                # This is the key insight!
                if raw_data['dtype'] != proc_data['dtype']:
                    print(f"  🎯 KEY FINDING: PyScope changes data type during processing!")
                    print(f"    Raw ({raw_data['dtype']}) → Processed ({proc_data['dtype']})")
        
        # Geometry impact
        if geometry:
            print(f"\n📐 PyScope Geometry Impact:")
            print(f"  Camera size: {geometry.get('camera_size', 'Unknown')}")
            print(f"  Dimension setting: {geometry.get('dimension', 'Unknown')}")
            print(f"  Offset setting: {geometry.get('offset', 'Unknown')}")
            
            # Key insight about dimension = 0
            if geometry.get('dimension', {}).get('x') == 0:
                print(f"  🎯 KEY FINDING: Dimension is set to 0!")
                print(f"    This may cause PyScope to use different image processing logic")
        
        # Final recommendations
        print(f"\n" + "=" * 60)
        print("RECOMMENDED FIXES")
        print("=" * 60)
        
        print(f"\n1. 🎯 Force DEAPI to use UINT16 format:")
        print(f"   Replace: pixelFormat = DEAPI.PixelFormat.AUTO")
        print(f"   With:    pixelFormat = DEAPI.PixelFormat.UINT16")
        
        print(f"\n2. 🎯 Fix PyScope dimension settings:")
        print(f"   The dimension {'x': 0, 'y': 0} is problematic")
        print(f"   Set proper dimensions before capture:")
        print(f"   camera.setDimension(camera.getCameraSize())")
        print(f"   camera.setOffset({{'x': 0, 'y': 0}})")
        
        print(f"\n3. 🎯 Use PyScope raw image method if available:")
        print(f"   Try using camera._getImage() instead of camera.getImage()")
        print(f"   This bypasses finalizeGeometry() processing")
        
        print(f"\n4. 🎯 Match camera properties exactly:")
        print(f"   Ensure both methods use identical:")
        print(f"   - Correction Mode")
        print(f"   - Image Processing Mode") 
        print(f"   - ROI settings")
        print(f"   - Binning settings")
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"comprehensive_diagnostic_{timestamp}.json"
        
        complete_results = {
            'deapi_formats': deapi_results,
            'pyscope_properties': properties,
            'pyscope_geometry': geometry,
            'pyscope_raw_vs_processed': pyscope_raw_processed,
            'pyscope_geometry_tests': pyscope_geometry_tests,
            'timestamp': timestamp
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        try:
            with open(results_filename, 'w') as f:
                json.dump(convert_for_json(complete_results), f, indent=2, default=str)
            print(f"\n📁 Complete diagnostic results saved to: {results_filename}")
        except Exception as e:
            print(f"Could not save results: {e}")


def main():
    """Run the comprehensive diagnostic"""
    diagnostic = DetailedDEDiagnostic()
    diagnostic.run_comprehensive_diagnostic()


if __name__ == "__main__":
    main()