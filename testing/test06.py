#!/usr/bin/env python3
"""
PyScope Camera Properties Reader
Read all properties from DE Apollo camera using PyScope interface instead of direct DEAPI
Based on the FastAPI router implementation but as a standalone script
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging to suppress DE camera property warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DE camera property warnings that are harmless
de_logger = logging.getLogger('DEClient')
de_logger.setLevel(logging.ERROR)  # Only show real errors


class PyScope_DECamera_PropertiesReader:
    """Enhanced PyScope wrapper for reading DE camera properties"""

    def __init__(self, camera_name=None):
        self.camera = None
        self.camera_name = camera_name
        self.connected = False
        self.properties_cache = {}
        self._deapi_client = None

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, '_deapi_client') and self._deapi_client:
            try:
                self._deapi_client.Disconnect()
            except:
                pass

    def connect(self):
        """Connect to DE camera via PyScope"""
        try:
            import pyscope.registry

            # Try different camera names
            camera_attempts = []

            if self.camera_name:
                camera_attempts.append(self.camera_name)

            # Add known DE camera names
            camera_attempts.extend([
                "DEApollo",
                "DE Apollo",
                "DE12",
                "Apollo"
            ])

            for attempt_name in camera_attempts:
                try:
                    logger.info(f"Attempting to connect to camera: {attempt_name}")

                    # Create camera instance using PyScope registry
                    self.camera = pyscope.registry.getClass(attempt_name)()
                    self.camera_name = attempt_name

                    # Test basic connectivity
                    camera_size = self.camera.getCameraSize()
                    logger.info(f"✓ Connected to {attempt_name}")
                    logger.info(f"  Camera size: {camera_size}")

                    self.connected = True
                    return True

                except Exception as e:
                    logger.debug(f"  Failed to connect to {attempt_name}: {e}")
                    continue

            logger.error("All DE camera connection attempts failed")
            return False

        except Exception as e:
            logger.error(f"Error in camera connection process: {e}")
            return False

    def get_available_properties(self) -> Optional[List[str]]:
        """Get list of all available properties from the camera"""
        if not self.connected or not self.camera:
            logger.error("Camera not connected")
            return None

        try:
            # Add DEAPI to path for direct access
            import sys
            sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]
            import DEAPI

            # PyScope DE camera should have access to the underlying DEAPI client
            # Try different approaches to access the DEAPI client

            # Method 1: Check if camera has direct DEAPI client access
            if hasattr(self.camera, '_DECameraBase__deserver') and self.camera._DECameraBase__deserver:
                deapi_client = self.camera._DECameraBase__deserver
                properties = deapi_client.ListProperties()
                logger.info(f"Found {len(properties)} properties via direct DEAPI client access")
                return properties

            # Method 2: Try to create a new DEAPI client connection
            try:
                deapi_client = DEAPI.Client()
                deapi_client.Connect()

                # Set the current camera to match what PyScope is using
                cameras = deapi_client.ListCameras()
                if self.camera_name in cameras or 'Apollo' in cameras:
                    camera_to_use = self.camera_name if self.camera_name in cameras else 'Apollo'
                    deapi_client.SetCurrentCamera(camera_to_use)
                    properties = deapi_client.ListProperties()
                    logger.info(f"Found {len(properties)} properties via new DEAPI client")

                    # Store the client for future use
                    self._deapi_client = deapi_client
                    return properties

            except Exception as e:
                logger.debug(f"Could not create new DEAPI client: {e}")

            # Method 3: Try the 'de' module approach
            try:
                # Try to access the global de functions
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))

                # Import the DEAPI module directly and create client
                deapi_client = DEAPI.Client()
                deapi_client.Connect()
                properties = deapi_client.ListProperties()
                logger.info(f"Found {len(properties)} properties via direct DEAPI")
                self._deapi_client = deapi_client
                return properties

            except Exception as e:
                logger.debug(f"Direct DEAPI approach failed: {e}")

            # Fallback: Return a basic set of properties that should be available
            logger.warning("Could not access full DE property list, using limited PyScope properties")

            basic_properties = [
                'Exposure Time (seconds)',
                'Frames Per Second',
                'Camera Model',
                'Camera Name',
                'Sensor Size X (pixels)',
                'Sensor Size Y (pixels)',
                'Image Size X (pixels)',
                'Image Size Y (pixels)',
                'Binning X',
                'Binning Y'
            ]

            return basic_properties

        except Exception as e:
            logger.error(f"Error getting available properties: {e}")
            return None

    def get_property_value(self, property_name: str) -> Any:
        """Get the value of a specific property"""
        if not self.connected or not self.camera:
            return None

        try:
            # Try to use the stored DEAPI client first
            if hasattr(self, '_deapi_client') and self._deapi_client:
                try:
                    return self._deapi_client.GetProperty(property_name)
                except Exception as e:
                    logger.debug(f"DEAPI client error for '{property_name}': {e}")

            # Try to access through PyScope camera's DEAPI connection
            if hasattr(self.camera, '_DECameraBase__deserver') and self.camera._DECameraBase__deserver:
                try:
                    return self.camera._DECameraBase__deserver.GetProperty(property_name)
                except Exception as e:
                    logger.debug(f"PyScope DEAPI error for '{property_name}': {e}")

            # Try direct property access methods if available
            if hasattr(self.camera, 'getProperty'):
                try:
                    return self.camera.getProperty(property_name)
                except Exception as e:
                    logger.debug(f"Direct getProperty error for '{property_name}': {e}")

            # For generic PyScope properties, try different mappings
            property_mappings = {
                'Exposure Time (seconds)': lambda: self.camera.getExposureTime() / 1000.0,
                'Camera Model': lambda: self.camera.__class__.__name__,
                'Camera Name': lambda: self.camera_name,
                'Sensor Size X (pixels)': lambda: self.camera.getCameraSize()['x'],
                'Sensor Size Y (pixels)': lambda: self.camera.getCameraSize()['y'],
                'Image Size X (pixels)': lambda: self.camera.getDimension()['x'] or self.camera.getCameraSize()['x'],
                'Image Size Y (pixels)': lambda: self.camera.getDimension()['y'] or self.camera.getCameraSize()['y'],
                'Binning X': lambda: self.camera.getBinning()['x'],
                'Binning Y': lambda: self.camera.getBinning()['y']
            }

            if property_name in property_mappings:
                return property_mappings[property_name]()

            # If no mapping found, return None
            logger.debug(f"No mapping found for property: {property_name}")
            return None

        except Exception as e:
            logger.debug(f"Error getting property '{property_name}': {e}")
            return None

    def get_all_properties(self) -> Dict[str, Any]:
        """Get all camera properties with their current values"""
        if not self.connected or not self.camera:
            logger.error("Camera not connected")
            return {}

        logger.info("Starting to read all camera properties...")
        start_time = time.time()

        # Get list of all properties
        property_names = self.get_available_properties()
        if not property_names:
            logger.error("Could not get property list")
            return {}

        logger.info(f"Reading values for {len(property_names)} properties...")

        # Get current value for each property
        all_properties = {}
        failed_properties = []
        successful_count = 0

        for i, prop_name in enumerate(property_names):
            try:
                value = self.get_property_value(prop_name)
                if value is not None:
                    all_properties[prop_name] = value
                    successful_count += 1
                else:
                    failed_properties.append(prop_name)

                # Progress indicator
                if (i + 1) % 25 == 0:
                    logger.info(f"Progress: {i + 1}/{len(property_names)} properties processed")

            except Exception as e:
                logger.warning(f"Failed to get property '{prop_name}': {str(e)}")
                failed_properties.append(prop_name)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"✓ Successfully retrieved {successful_count} properties")
        if failed_properties:
            logger.warning(f"✗ Failed to retrieve {len(failed_properties)} properties")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per property: {(total_time / len(property_names) * 1000):.2f} ms")

        # Store results
        self.properties_cache = {
            "properties": all_properties,
            "total_count": len(all_properties),
            "failed_count": len(failed_properties),
            "failed_properties": failed_properties,
            "camera_name": self.camera_name,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "average_time_ms": total_time / len(property_names) * 1000
        }

        return self.properties_cache

    def get_properties_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get properties organized by category"""
        all_props = self.get_all_properties()
        if not all_props or "properties" not in all_props:
            return {}

        # Define categories for DE camera properties
        categories = {
            "Camera Info": ["Camera Model", "Camera Name", "Camera SN", "Sensor Module SN",
                            "Server Software Version", "Firmware Version"],
            "Temperature": ["Temperature - Detector (Celsius)", "Temperature - Detector Status",
                            "Temperature - Control", "Temperature - Chilled Water (Celsius)",
                            "Temperature - Chilled Water Status"],
            "Acquisition": ["Exposure Time (seconds)", "Frames Per Second", "Exposure Mode",
                            "Acquisition Counter", "Remaining Number of Acquisitions",
                            "Total Number of Acquisitions", "Frame Count"],
            "Image Settings": ["Image Size X (pixels)", "Image Size Y (pixels)",
                               "Sensor Size X (pixels)", "Sensor Size Y (pixels)",
                               "Binning X", "Binning Y", "Binning Method"],
            "ROI/Crop": ["Crop Offset X", "Crop Offset Y", "Crop Size X", "Crop Size Y"],
            "Processing": ["Image Processing - Mode", "Image Processing - Flatfield Correction",
                           "Image Processing - Apply Gain on Final", "Image Processing - Apply Gain on Movie"],
            "Status": ["System Status", "Protection Cover Status", "Camera Position Status",
                       "Autosave Status", "Vacuum State"],
            "Hardware": ["Sensor Pixel Depth", "Sensor Pixel Pitch (micrometers)",
                         "Hardware Frame Size X", "Hardware Frame Size Y"],
            "References": ["Reference - Dark", "Reference - Integrating Gain"],
            "Autosave": ["Autosave Directory", "Autosave File Format", "Autosave Final Image",
                         "Autosave Movie", "Autosave Movie Sum Count"]
        }

        # Organize properties into categories
        categorized = {}
        uncategorized = []

        properties = all_props["properties"]

        for category, prop_list in categories.items():
            categorized[category] = []
            for prop_name in prop_list:
                if prop_name in properties:
                    categorized[category].append({
                        "name": prop_name,
                        "value": properties[prop_name]
                    })

        # Find uncategorized properties
        all_categorized_props = set()
        for prop_list in categories.values():
            all_categorized_props.update(prop_list)

        for prop_name, value in properties.items():
            if prop_name not in all_categorized_props:
                uncategorized.append({
                    "name": prop_name,
                    "value": value
                })

        if uncategorized:
            categorized["Uncategorized"] = uncategorized

        return categorized

    def search_properties(self, query: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """Search for properties by name"""
        all_props = self.get_all_properties()
        if not all_props or "properties" not in all_props:
            return {}

        properties = all_props["properties"]

        if case_sensitive:
            matching_props = {name: value for name, value in properties.items() if query in name}
        else:
            query_lower = query.lower()
            matching_props = {name: value for name, value in properties.items() if query_lower in name.lower()}

        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "matches": matching_props,
            "match_count": len(matching_props),
            "total_properties": len(properties)
        }

    def get_essential_properties(self) -> Dict[str, Any]:
        """Get essential camera properties for quick status check"""
        essential_props = [
            "Camera Model",
            "Camera SN",
            "Sensor Module SN",
            "Server Software Version",
            "Firmware Version",
            "Temperature - Detector (Celsius)",
            "Temperature - Detector Status",
            "Exposure Time (seconds)",
            "Frames Per Second",
            "Image Size X (pixels)",
            "Image Size Y (pixels)",
            "Exposure Mode",
            "Image Processing - Mode",
            "Acquisition Counter",
            "System Status",
            "Protection Cover Status",
            "Camera Position Status",
            "Remaining Number of Acquisitions",
            "Autosave Status"
        ]

        all_props = self.get_all_properties()
        if not all_props or "properties" not in all_props:
            return {}

        properties = all_props["properties"]

        essential_data = {}
        missing_props = []

        for prop_name in essential_props:
            if prop_name in properties:
                essential_data[prop_name] = properties[prop_name]
            else:
                missing_props.append(prop_name)

        return {
            "essential_properties": essential_data,
            "count": len(essential_data),
            "missing_properties": missing_props
        }

    def save_properties_to_file(self, filename: str = None, format: str = "json"):
        """Save all properties to a file"""
        if not self.properties_cache:
            logger.warning("No properties cached. Call get_all_properties() first.")
            return False

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"de_camera_properties_{self.camera_name}_{timestamp}.{format}"

        try:
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(self.properties_cache, f, indent=2, default=str)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            logger.info(f"✓ Properties saved to: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving properties to file: {e}")
            return False


def main():
    """Main function to demonstrate properties reading"""
    print("=== PyScope DE Camera Properties Reader ===")

    # Test PyScope imports
    try:
        import pyscope.registry
        import pyscope.config
        logger.info("✓ PyScope imports successful")
    except ImportError as e:
        logger.error(f"PyScope import failed: {e}")
        return

    # Create and connect to camera
    reader = PyScope_DECamera_PropertiesReader()

    if not reader.connect():
        logger.error("Failed to connect to DE camera")
        return

    try:
        print(f"\n🔍 Reading all properties from {reader.camera_name}...")

        # Get all properties
        all_properties = reader.get_all_properties()

        if not all_properties:
            logger.error("Failed to read properties")
            return

        # Display summary
        print(f"\n📊 Properties Summary:")
        print(f"  Total properties: {all_properties['total_count']}")
        print(f"  Failed properties: {all_properties['failed_count']}")
        print(f"  Total time: {all_properties['total_time_seconds']:.2f} seconds")
        print(f"  Average time per property: {all_properties['average_time_ms']:.2f} ms")

        # Show essential properties
        print(f"\n🎯 Essential Properties:")
        essential = reader.get_essential_properties()
        for name, value in essential['essential_properties'].items():
            print(f"  {name}: {value}")

        # Show properties by category
        print(f"\n📂 Properties by Category:")
        categorized = reader.get_properties_by_category()
        for category, props in categorized.items():
            if props:  # Only show categories with properties
                print(f"  {category}: {len(props)} properties")
                # Show first 3 properties in each category
                for prop in props[:3]:
                    print(f"    - {prop['name']}: {prop['value']}")
                if len(props) > 3:
                    print(f"    ... and {len(props) - 3} more")

        # Search example
        print(f"\n🔍 Search Example (temperature):")
        search_results = reader.search_properties("temperature")
        print(f"  Found {search_results['match_count']} temperature-related properties:")
        for name, value in list(search_results['matches'].items())[:5]:
            print(f"    {name}: {value}")

        # Save to file
        print(f"\n💾 Saving properties to file...")
        if reader.save_properties_to_file():
            print("✓ Properties saved successfully")

        # Show failed properties if any
        if all_properties['failed_properties']:
            print(f"\n⚠️  Failed to read {len(all_properties['failed_properties'])} properties:")
            for prop in all_properties['failed_properties'][:10]:  # Show first 10
                print(f"    - {prop}")
            if len(all_properties['failed_properties']) > 10:
                print(f"    ... and {len(all_properties['failed_properties']) - 10} more")

        print(f"\n🎉 Properties reading completed successfully!")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()