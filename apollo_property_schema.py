#!/usr/bin/env python3
"""
Apollo Camera Property Schema System
Provides structured property definitions with validation, defaults, and type safety
"""

from enum import Enum
from typing import Union, List, Optional, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
import json

# Property value types
PropertyValue = Union[str, int, float, bool]


class PropertyCategory(str, Enum):
    BASIC = "Basic"
    ADVANCED = "Advanced"
    ENGINEERING = "Engineering"
    DEPRECATED = "Deprecated"
    OBSOLETE = "Obsolete"


class PropertyDataType(str, Enum):
    STRING = "String"
    INTEGER = "Integer"
    FLOAT = "Float"
    BOOLEAN = "Boolean"


class PropertyValueType(str, Enum):
    READ_ONLY = "ReadOnly"
    SET = "Set"
    RANGE = "Range"
    ALLOW_ALL = "AllowAll"


class PropertySchema(BaseModel):
    """Schema definition for a camera property"""
    name: str
    data_type: PropertyDataType
    value_type: PropertyValueType
    category: PropertyCategory
    default_value: PropertyValue
    description: str
    unit: Optional[str] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[PropertyValue]] = None
    read_only: bool = False
    requires_restart: bool = False
    affects_acquisition: bool = False
    group: Optional[str] = None

    @validator('allowed_values')
    def validate_allowed_values(cls, v, values):
        if values.get('value_type') == PropertyValueType.SET and not v:
            raise ValueError("SET properties must have allowed_values")
        return v

    @validator('min_value', 'max_value')
    def validate_range(cls, v, values):
        if values.get('value_type') == PropertyValueType.RANGE:
            if v is None:
                raise ValueError("RANGE properties must have min_value and max_value")
        return v


class ApolloPropertyRegistry:
    """Registry of all Apollo camera properties with their schemas"""

    def __init__(self):
        self._properties: Dict[str, PropertySchema] = {}
        self._initialize_properties()

    def _initialize_properties(self):
        """Initialize all known Apollo camera properties"""

        # Camera Information Properties
        self.register_property(PropertySchema(
            name="Camera Model",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="Apollo",
            description="Camera model name",
            read_only=True,
            group="camera_info"
        ))

        self.register_property(PropertySchema(
            name="Camera SN",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="",
            description="Camera serial number",
            read_only=True,
            group="camera_info"
        ))

        self.register_property(PropertySchema(
            name="Sensor Module SN",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="",
            description="Sensor module serial number",
            read_only=True,
            group="camera_info"
        ))

        self.register_property(PropertySchema(
            name="Firmware Version",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="",
            description="Camera firmware version",
            read_only=True,
            group="camera_info"
        ))

        # Acquisition Properties
        self.register_property(PropertySchema(
            name="Exposure Time (seconds)",
            data_type=PropertyDataType.FLOAT,
            value_type=PropertyValueType.RANGE,
            category=PropertyCategory.BASIC,
            default_value=1.0,
            description="Exposure time in seconds",
            unit="seconds",
            min_value=0.001,
            max_value=3600.0,
            affects_acquisition=True,
            group="acquisition"
        ))

        self.register_property(PropertySchema(
            name="Frames Per Second",
            data_type=PropertyDataType.FLOAT,
            value_type=PropertyValueType.RANGE,
            category=PropertyCategory.BASIC,
            default_value=20.0,
            description="Frame rate in frames per second",
            unit="fps",
            min_value=0.1,
            max_value=60.0,
            affects_acquisition=True,
            group="acquisition"
        ))

        self.register_property(PropertySchema(
            name="Exposure Mode",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value="Normal",
            description="Camera exposure mode",
            allowed_values=["Normal", "Dark", "Gain", "Trial"],
            affects_acquisition=True,
            group="acquisition"
        ))

        # Image Properties
        self.register_property(PropertySchema(
            name="Image Size X (pixels)",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value=8192,
            description="Image width in pixels",
            unit="pixels",
            read_only=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Image Size Y (pixels)",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value=8192,
            description="Image height in pixels",
            unit="pixels",
            read_only=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Binning X",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value=1,
            description="Horizontal binning factor",
            allowed_values=[1, 2, 4, 8],
            affects_acquisition=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Binning Y",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value=1,
            description="Vertical binning factor",
            allowed_values=[1, 2, 4, 8],
            affects_acquisition=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Crop Size X",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.RANGE,
            category=PropertyCategory.BASIC,
            default_value=8192,
            description="Crop width in pixels",
            unit="pixels",
            min_value=1,
            max_value=8192,
            affects_acquisition=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Crop Size Y",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.RANGE,
            category=PropertyCategory.BASIC,
            default_value=8192,
            description="Crop height in pixels",
            unit="pixels",
            min_value=1,
            max_value=8192,
            affects_acquisition=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Crop Offset X",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.RANGE,
            category=PropertyCategory.BASIC,
            default_value=0,
            description="Crop X offset in pixels",
            unit="pixels",
            min_value=0,
            max_value=8191,
            affects_acquisition=True,
            group="image"
        ))

        self.register_property(PropertySchema(
            name="Crop Offset Y",
            data_type=PropertyDataType.INTEGER,
            value_type=PropertyValueType.RANGE,
            category=PropertyCategory.BASIC,
            default_value=0,
            description="Crop Y offset in pixels",
            unit="pixels",
            min_value=0,
            max_value=8191,
            affects_acquisition=True,
            group="image"
        ))

        # Temperature Properties
        self.register_property(PropertySchema(
            name="Temperature - Detector (Celsius)",
            data_type=PropertyDataType.FLOAT,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value=-25.0,
            description="Detector temperature in Celsius",
            unit="°C",
            read_only=True,
            group="temperature"
        ))

        self.register_property(PropertySchema(
            name="Temperature - Detector Status",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="Cooled",
            description="Detector temperature status",
            allowed_values=["Cooled", "Cooling", "Warmed", "Warming"],
            read_only=True,
            group="temperature"
        ))

        self.register_property(PropertySchema(
            name="Temperature - Control",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value="Cool Down",
            description="Temperature control command",
            allowed_values=["Cool Down", "Warm Up"],
            requires_restart=False,
            group="temperature"
        ))

        # Image Processing Properties
        self.register_property(PropertySchema(
            name="Image Processing - Mode",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value="Integrating",
            description="Image processing mode",
            allowed_values=["Integrating", "Counting"],
            affects_acquisition=True,
            group="processing"
        ))

        self.register_property(PropertySchema(
            name="Image Processing - Rotation",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.ADVANCED,
            default_value="None",
            description="Image rotation setting",
            allowed_values=["None", "90", "180", "270"],
            group="processing"
        ))

        # Autosave Properties
        self.register_property(PropertySchema(
            name="Autosave Final Image",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value="Off",
            description="Enable/disable final image autosave",
            allowed_values=["On", "Off"],
            group="autosave"
        ))

        self.register_property(PropertySchema(
            name="Autosave Movie",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value="Off",
            description="Enable/disable movie autosave",
            allowed_values=["On", "Off"],
            group="autosave"
        ))

        self.register_property(PropertySchema(
            name="Autosave File Format",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.SET,
            category=PropertyCategory.BASIC,
            default_value="MRC",
            description="File format for autosaved files",
            allowed_values=["MRC", "TIFF", "DM4"],
            group="autosave"
        ))

        self.register_property(PropertySchema(
            name="Autosave Directory",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.ALLOW_ALL,
            category=PropertyCategory.BASIC,
            default_value="D:\\DEOutput\\Apollo",
            description="Directory for autosaved files",
            group="autosave"
        ))

        # System Properties
        self.register_property(PropertySchema(
            name="System Status",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="OK",
            description="Overall system status",
            allowed_values=["OK", "Warning", "Error"],
            read_only=True,
            group="system"
        ))

        self.register_property(PropertySchema(
            name="Camera Position Status",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="Extended",
            description="Camera position status",
            allowed_values=["Extended", "Retracted", "Moving"],
            read_only=True,
            group="system"
        ))

        self.register_property(PropertySchema(
            name="Protection Cover Status",
            data_type=PropertyDataType.STRING,
            value_type=PropertyValueType.READ_ONLY,
            category=PropertyCategory.BASIC,
            default_value="Closed",
            description="Protection cover status",
            allowed_values=["Open", "Closed", "Moving"],
            read_only=True,
            group="system"
        ))

        # Add more properties as needed...

    def register_property(self, schema: PropertySchema):
        """Register a property schema"""
        self._properties[schema.name] = schema

    def get_property_schema(self, name: str) -> Optional[PropertySchema]:
        """Get schema for a specific property"""
        return self._properties.get(name)

    def get_all_schemas(self) -> Dict[str, PropertySchema]:
        """Get all property schemas"""
        return self._properties.copy()

    def get_schemas_by_category(self, category: PropertyCategory) -> Dict[str, PropertySchema]:
        """Get schemas filtered by category"""
        return {name: schema for name, schema in self._properties.items()
                if schema.category == category}

    def get_schemas_by_group(self, group: str) -> Dict[str, PropertySchema]:
        """Get schemas filtered by group"""
        return {name: schema for name, schema in self._properties.items()
                if schema.group == group}

    def validate_property_value(self, name: str, value: PropertyValue) -> tuple[bool, str]:
        """Validate a property value against its schema"""
        schema = self.get_property_schema(name)
        if not schema:
            return False, f"Unknown property: {name}"

        if schema.read_only:
            return False, f"Property '{name}' is read-only"

        # Type validation
        expected_type = {
            PropertyDataType.STRING: str,
            PropertyDataType.INTEGER: int,
            PropertyDataType.FLOAT: (int, float),
            PropertyDataType.BOOLEAN: bool
        }[schema.data_type]

        if not isinstance(value, expected_type):
            return False, f"Expected {schema.data_type.value}, got {type(value).__name__}"

        # Value type validation
        if schema.value_type == PropertyValueType.SET:
            if value not in schema.allowed_values:
                return False, f"Value must be one of: {schema.allowed_values}"

        elif schema.value_type == PropertyValueType.RANGE:
            if schema.min_value is not None and value < schema.min_value:
                return False, f"Value must be >= {schema.min_value}"
            if schema.max_value is not None and value > schema.max_value:
                return False, f"Value must be <= {schema.max_value}"

        return True, "Valid"


class PropertyConfiguration(BaseModel):
    """Configuration model for multiple properties"""
    properties: Dict[str, PropertyValue]

    def validate_with_registry(self, registry: ApolloPropertyRegistry) -> Dict[str, str]:
        """Validate all properties against the registry"""
        errors = {}
        for name, value in self.properties.items():
            is_valid, message = registry.validate_property_value(name, value)
            if not is_valid:
                errors[name] = message
        return errors


# Predefined configurations
class PresetConfigurations:
    """Predefined camera configurations"""

    @staticmethod
    def high_resolution_cryo_em() -> PropertyConfiguration:
        """High resolution configuration for Cryo-EM"""
        return PropertyConfiguration(properties={
            "Binning X": 1,
            "Binning Y": 1,
            "Crop Size X": 8192,
            "Crop Size Y": 8192,
            "Crop Offset X": 0,
            "Crop Offset Y": 0,
            "Image Processing - Mode": "Integrating",
            "Exposure Time (seconds)": 2.0,
            "Frames Per Second": 20.0,
            "Autosave Final Image": "On",
            "Autosave File Format": "MRC"
        })

    @staticmethod
    def fast_acquisition() -> PropertyConfiguration:
        """Fast acquisition configuration"""
        return PropertyConfiguration(properties={
            "Binning X": 2,
            "Binning Y": 2,
            "Crop Size X": 4096,
            "Crop Size Y": 4096,
            "Crop Offset X": 2048,
            "Crop Offset Y": 2048,
            "Image Processing - Mode": "Integrating",
            "Exposure Time (seconds)": 0.5,
            "Frames Per Second": 40.0,
            "Autosave Final Image": "Off",
            "Autosave Movie": "Off"
        })

    @staticmethod
    def counting_mode() -> PropertyConfiguration:
        """Electron counting configuration"""
        return PropertyConfiguration(properties={
            "Binning X": 1,
            "Binning Y": 1,
            "Image Processing - Mode": "Counting",
            "Exposure Time (seconds)": 5.0,
            "Frames Per Second": 280.0,
            "Autosave Final Image": "On",
            "Autosave Movie": "On",
            "Autosave File Format": "MRC"
        })

    @staticmethod
    def low_dose_imaging() -> PropertyConfiguration:
        """Low dose imaging configuration"""
        return PropertyConfiguration(properties={
            "Binning X": 1,
            "Binning Y": 1,
            "Crop Size X": 8192,
            "Crop Size Y": 8192,
            "Image Processing - Mode": "Integrating",
            "Exposure Time (seconds)": 10.0,
            "Frames Per Second": 10.0,
            "Autosave Final Image": "On",
            "Autosave Movie": "On"
        })


# Create global registry instance
apollo_registry = ApolloPropertyRegistry()


def export_schema_to_json(filename: str = "apollo_properties_schema.json"):
    """Export the complete property schema to JSON"""
    schemas = apollo_registry.get_all_schemas()
    schema_dict = {}

    for name, schema in schemas.items():
        schema_dict[name] = {
            "data_type": schema.data_type.value,
            "value_type": schema.value_type.value,
            "category": schema.category.value,
            "default_value": schema.default_value,
            "description": schema.description,
            "unit": schema.unit,
            "min_value": schema.min_value,
            "max_value": schema.max_value,
            "allowed_values": schema.allowed_values,
            "read_only": schema.read_only,
            "requires_restart": schema.requires_restart,
            "affects_acquisition": schema.affects_acquisition,
            "group": schema.group
        }

    with open(filename, 'w') as f:
        json.dump(schema_dict, f, indent=2)

    return filename


if __name__ == "__main__":
    # Example usage
    registry = ApolloPropertyRegistry()

    # Test validation
    is_valid, message = registry.validate_property_value("Exposure Time (seconds)", 2.5)
    print(f"Exposure Time validation: {is_valid}, {message}")

    is_valid, message = registry.validate_property_value("Binning X", 3)  # Invalid
    print(f"Binning X validation: {is_valid}, {message}")

    # Test configuration
    config = PresetConfigurations.high_resolution_cryo_em()
    errors = config.validate_with_registry(registry)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration is valid!")

    # Export schema
    filename = export_schema_to_json()
    print(f"Schema exported to: {filename}")