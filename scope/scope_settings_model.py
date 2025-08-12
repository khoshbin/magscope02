from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Union
from enum import Enum
import numpy as np


class ProjectionMode(str, Enum):
    """Projection mode options"""
    IMAGING = "imaging"
    DIFFRACTION = "diffraction"
    EFTEM = "EFTEM"


class ProbeMode(str, Enum):
    """Probe mode options (JEOL specific)"""
    MICRO = "micro"
    NANO = "nano"
    SAMAG = "samag"
    MAG1 = "mag1"
    MAG2 = "mag2"
    DIFF = "diff"


class ScopeOptics(BaseModel):
    """Extended optics settings with all Leginon parameters"""

    # Core optics
    magnification: Optional[int] = Field(default=None, gt=0, description="Magnification")
    defocus: Optional[float] = Field(default=None, description="Defocus (meters)")

    # Beam settings
    spot_size: Optional[int] = Field(default=None, ge=1, le=11, description="Spot size (1-11)")
    intensity: Optional[float] = Field(default=None, ge=0, le=1, description="Beam intensity (0-1)")

    # Beam alignment
    image_shift: Optional[Dict[str, float]] = Field(
        default=None,
        description="Image shift in meters {'x': float, 'y': float}"
    )
    beam_shift: Optional[Dict[str, float]] = Field(
        default=None,
        description="Beam shift in meters {'x': float, 'y': float}"
    )
    diffraction_shift: Optional[Dict[str, float]] = Field(
        default=None,
        description="Diffraction shift for selected area {'x': float, 'y': float}"
    )

    # Advanced optics
    projection_mode: Optional[ProjectionMode] = Field(
        default=None,
        description="Projection mode (imaging/diffraction/EFTEM)"
    )
    probe_mode: Optional[ProbeMode] = Field(
        default=None,
        description="Probe mode (JEOL specific)"
    )

    @validator('image_shift', 'beam_shift', 'diffraction_shift', pre=True)
    def validate_shift_dict(cls, v):
        if v is not None and isinstance(v, dict):
            if 'x' not in v or 'y' not in v:
                raise ValueError("Shift dictionary must contain 'x' and 'y' keys")
            # Convert to float
            v['x'] = float(v['x'])
            v['y'] = float(v['y'])
        return v


class DefocusSeriesSettings(BaseModel):
    """Defocus series configuration"""

    defocus: Optional[float] = Field(default=None, description="Primary defocus (meters)")
    defocus_range_min: Optional[float] = Field(
        default=None,
        description="Minimum defocus for series (meters)"
    )
    defocus_range_max: Optional[float] = Field(
        default=None,
        description="Maximum defocus for series (meters)"
    )
    eucentric_focus: Optional[float] = Field(
        default=None,
        description="Eucentric focus position (meters)"
    )

    @validator('defocus_range_max')
    def validate_defocus_range(cls, v, values):
        if v is not None and 'defocus_range_min' in values and values['defocus_range_min'] is not None:
            if v <= values['defocus_range_min']:
                raise ValueError("defocus_range_max must be greater than defocus_range_min")
        return v

    def generate_defocus_series(self, steps: int = 5) -> List[float]:
        """Generate defocus values for series acquisition"""
        if self.defocus_range_min is None or self.defocus_range_max is None:
            return [self.defocus] if self.defocus is not None else []

        return list(np.linspace(self.defocus_range_min, self.defocus_range_max, steps))


class TiltSettings(BaseModel):
    """Tilt series configuration"""

    usetilt: bool = Field(default=False, description="Enable tilt for acquisition")
    tilt: Optional[float] = Field(
        default=None,
        ge=-1.57, le=1.57,  # ±90 degrees in radians
        description="Tilt angle in radians"
    )

    # Tilt series parameters (not in original table but useful)
    tilt_min: Optional[float] = Field(default=None, description="Minimum tilt angle")
    tilt_max: Optional[float] = Field(default=None, description="Maximum tilt angle")
    tilt_step: Optional[float] = Field(default=None, description="Tilt step size")

    @validator('tilt', 'tilt_min', 'tilt_max', 'tilt_step', pre=True)
    def convert_degrees_to_radians(cls, v):
        """Convert degrees to radians if value seems to be in degrees"""
        if v is not None:
            # If absolute value > π, assume it's in degrees
            if abs(v) > 3.15:
                return np.radians(v)
        return v

    def generate_tilt_series(self) -> List[float]:
        """Generate tilt angles for series acquisition"""
        if not self.usetilt or None in [self.tilt_min, self.tilt_max, self.tilt_step]:
            return [self.tilt] if self.tilt is not None else []

        angles = []
        current = self.tilt_min
        while current <= self.tilt_max:
            angles.append(current)
            current += self.tilt_step

        return angles


class EnergyFilterSettings(BaseModel):
    """Energy filter configuration for both camera and TEM"""

    # Camera energy filter
    energy_filter: bool = Field(default=False, description="Enable camera energy filter")
    energy_filter_width: Optional[float] = Field(
        default=None,
        gt=0,
        description="Camera energy filter slit width (eV)"
    )

    # TEM energy filter (in-column filter like GIF)
    tem_energy_filter: bool = Field(default=False, description="Enable TEM energy filter")
    tem_energy_filter_width: Optional[float] = Field(
        default=None,
        gt=0,
        description="TEM energy filter slit width (eV)"
    )

    @validator('energy_filter_width', 'tem_energy_filter_width')
    def validate_filter_width(cls, v, values, field):
        # Check if corresponding filter is enabled
        filter_enabled_field = field.name.replace('_width', '')
        if v is not None and not values.get(filter_enabled_field, False):
            raise ValueError(f"{filter_enabled_field} must be enabled to set width")
        return v


class DoseCalculationSettings(BaseModel):
    """Electron dose calculation and monitoring"""

    dose: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target electron dose (e-/Å²)"
    )

    # Parameters for dose calculation
    beam_current: Optional[float] = Field(default=None, description="Beam current (pA)")
    specimen_area: Optional[float] = Field(default=None, description="Illuminated area (m²)")

    def calculate_dose_rate(self, exposure_time: float) -> Optional[float]:
        """Calculate dose rate from exposure time"""
        if self.dose is not None and exposure_time > 0:
            return self.dose / (exposure_time / 1000.0)  # Convert ms to s
        return None

    def calculate_exposure_time(self, target_dose: float) -> Optional[float]:
        """Calculate required exposure time for target dose"""
        dose_rate = self.calculate_dose_rate(1000.0)  # Get dose rate per second
        if dose_rate is not None and dose_rate > 0:
            return (target_dose / dose_rate) * 1000.0  # Convert to ms
        return None


class PresetReferenceSettings(BaseModel):
    """Reference settings for preset management"""

    name: Optional[str] = Field(default=None, description="Preset name")
    number: Optional[int] = Field(default=None, description="Preset number/order")
    session_id: Optional[int] = Field(default=None, description="Associated session ID")

    # Instrument references
    tem_id: Optional[int] = Field(default=None, description="TEM instrument ID")
    ccdcamera_id: Optional[int] = Field(default=None, description="Camera instrument ID")

    # State flags
    removed: bool = Field(default=False, description="Soft delete flag")
    hasref: bool = Field(default=False, description="Has reference images")
    skip: bool = Field(default=False, description="Skip this preset in acquisition")
    film: bool = Field(default=False, description="Film mode acquisition")


class ScopeConfiguration(BaseModel):
    """Complete scope configuration matching Leginon PresetData table"""

    # Core optics and beam settings
    optics: ScopeOptics = Field(default_factory=ScopeOptics)

    # Defocus management
    defocus_settings: DefocusSeriesSettings = Field(default_factory=DefocusSeriesSettings)

    # Tilt configuration
    tilt_settings: TiltSettings = Field(default_factory=TiltSettings)

    # Energy filtering
    energy_filter: EnergyFilterSettings = Field(default_factory=EnergyFilterSettings)

    # Dose management
    dose_settings: DoseCalculationSettings = Field(default_factory=DoseCalculationSettings)

    # Preset metadata
    preset_info: PresetReferenceSettings = Field(default_factory=PresetReferenceSettings)