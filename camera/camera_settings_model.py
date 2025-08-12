from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict


class CameraGeometry(BaseModel):
    """Extended camera geometry with all Leginon parameters"""
    # Basic geometry
    binning: Dict[str, int] = Field(default={"x": 1, "y": 1})
    offset: Dict[str, int] = Field(default={"x": 0, "y": 0})
    dimension: Dict[str, int] = Field(default={"x": 1024, "y": 1024})

    # Image shifts (for beam positioning)
    image_shift: Optional[Dict[str, float]] = Field(default=None)
    beam_shift: Optional[Dict[str, float]] = Field(default=None)
    diffraction_shift: Optional[Dict[str, float]] = Field(default=None)


class FrameSettings(BaseModel):
    """Frame-based acquisition settings"""
    save_frames: bool = Field(default=False, description="Save individual frames")
    frame_time: Optional[float] = Field(default=None, description="Frame exposure time (ms)")
    use_frames: Optional[List[int]] = Field(default=None, description="Frame indices to use")
    align_frames: bool = Field(default=False, description="Enable frame alignment")
    align_filter: Optional[str] = Field(default=None, description="Alignment filter type")
    request_nframes: Optional[int] = Field(default=None, description="Number of frames")
    readout_delay: Optional[int] = Field(default=None, description="Readout delay (ms)")


class EnergyFilterSettings(BaseModel):
    """Energy filter configuration"""
    energy_filter: bool = Field(default=False, description="Enable energy filter")
    energy_filter_width: Optional[float] = Field(default=None, description="Filter width (eV)")
    tem_energy_filter: bool = Field(default=False, description="TEM energy filter")
    tem_energy_filter_width: Optional[float] = Field(default=None, description="TEM filter width")


class ExposureSettings(BaseModel):
    """Extended exposure settings"""
    exposure_time: float = Field(gt=0, description="Main exposure time (ms)")
    exposure_type: str = Field(default="normal")
    pre_exposure: Optional[float] = Field(default=None, description="Pre-exposure time")
    dose: Optional[float] = Field(default=None, description="Electron dose")


class CameraHardwareSettings(BaseModel):
    """Hardware-specific camera settings"""
    alt_channel: bool = Field(default=False, description="Use alternative channel")
    use_cds: bool = Field(default=False, description="Correlated double sampling")
    fast_save: bool = Field(default=False, description="Fast save mode")


class PresetData(BaseModel):
    """Complete compatible preset data"""
    name: str = Field(description="Preset name")

    # Camera geometry
    geometry: CameraGeometry

    # Exposure settings
    exposure: ExposureSettings

    # Frame settings
    frames: Optional[FrameSettings] = None

    # Energy filter
    energy_filter: Optional[EnergyFilterSettings] = None

    # Hardware settings
    hardware: Optional[CameraHardwareSettings] = None

    # TEM settings (scope-related)
    magnification: Optional[int] = None
    spot_size: Optional[int] = None
    intensity: Optional[float] = None
    defocus: Optional[float] = None
    probe_mode: Optional[str] = None
    projection_mode: Optional[str] = None