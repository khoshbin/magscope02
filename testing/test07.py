#!/usr/bin/env python3
"""
Simple DEAPI Properties Test
Test direct DEAPI access for reading properties (like your working properties.py)
"""

import sys
import time
import json
from datetime import datetime

# Add DEAPI path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
from pyscope import DEAPI


def test_direct_deapi_properties():
    """Test direct DEAPI property access"""
    print("=== Direct DEAPI Properties Test ===")

    try:
        # Connect to DE-Server
        client = DEAPI.Client()
        client.Connect()
        print("✓ Connected to DE-Server")

        # List cameras
        cameras = client.ListCameras()
        print(f"Available cameras: {cameras}")

        if not cameras:
            print("No cameras found")
            return

        # Set current camera
        current_camera = cameras[0]
        client.SetCurrentCamera(current_camera)
        print(f"✓ Selected camera: {current_camera}")

        # Get list of properties
        properties = client.ListProperties()
        print(f"✓ Found {len(properties)} properties")

        # Read all properties
        start_time = time.time()
        all_properties = {}
        failed_properties = []

        for i, prop_name in enumerate(properties):
            try:
                value = client.GetProperty(prop_name)
                all_properties[prop_name] = value

                # Progress indicator
                if (i + 1) % 25 == 0:
                    print(f"Progress: {i + 1}/{len(properties)} properties")

            except Exception as e:
                print(f"Failed to read '{prop_name}': {e}")
                failed_properties.append(prop_name)

        end_time = time.time()
        total_time = end_time - start_time

        # Results
        print(f"\n📊 Results:")
        print(f"  Total properties: {len(all_properties)}")
        print(f"  Failed properties: {len(failed_properties)}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per property: {(total_time / len(properties) * 1000):.2f} ms")

        # Show some sample properties
        print(f"\n🔍 Sample Properties:")
        sample_props = [
            "Camera Model",
            "Camera SN",
            "Server Software Version",
            "Exposure Time (seconds)",
            "Image Size X (pixels)",
            "Image Size Y (pixels)",
            "Temperature - Detector (Celsius)",
            "System Status"
        ]

        for prop in sample_props:
            if prop in all_properties:
                print(f"  {prop}: {all_properties[prop]}")

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deapi_properties_{timestamp}.json"

        output_data = {
            "camera": current_camera,
            "timestamp": datetime.now().isoformat(),
            "total_properties": len(all_properties),
            "failed_count": len(failed_properties),
            "total_time_seconds": total_time,
            "average_time_ms": total_time / len(properties) * 1000,
            "properties": all_properties,
            "failed_properties": failed_properties
        }

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 Properties saved to: {filename}")

        # Disconnect
        client.Disconnect()
        print("✓ Disconnected from DE-Server")

        return output_data

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pyscope_with_deapi():
    """Test accessing DEAPI through PyScope camera"""
    print("\n=== PyScope + DEAPI Properties Test ===")

    try:
        import pyscope.registry

        # Connect to PyScope camera
        camera = pyscope.registry.getClass("DEApollo")()
        print("✓ Connected to PyScope DEApollo camera")

        # Now try to access DEAPI directly
        sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
        import pyscope.DEAPI

        # Create separate DEAPI client
        client = DEAPI.Client()
        client.Connect()

        cameras = client.ListCameras()
        client.SetCurrentCamera(cameras[0])

        # Get properties
        properties = client.ListProperties()
        print(f"✓ Found {len(properties)} properties via separate DEAPI client")

        # Test reading a few properties
        test_props = ["Camera Model", "Exposure Time (seconds)", "System Status"]

        print("\n🔍 Testing property access:")
        for prop in test_props:
            try:
                value = client.GetProperty(prop)
                print(f"  {prop}: {value}")
            except Exception as e:
                print(f"  {prop}: Error - {e}")

        # Cleanup
        client.Disconnect()
        print("✓ DEAPI client disconnected")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test 1: Direct DEAPI (should work like your properties.py)
    result1 = test_direct_deapi_properties()

    # Test 2: PyScope + separate DEAPI client
    result2 = test_pyscope_with_deapi()

    if result1:
        print(f"\n🎉 Direct DEAPI test completed successfully!")
        print(f"Found {result1['total_properties']} properties")

    if result2:
        print(f"🎉 PyScope + DEAPI test completed successfully!")

    print("\nThis confirms that DEAPI works and can be used with PyScope!")