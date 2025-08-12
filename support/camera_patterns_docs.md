# DE Apollo Camera Patterns & Architecture Documentation

## 🏗️ **Camera Access Architecture**

### **Three Access Methods Comparison**

| Method | Connection | Properties | Performance | Use Case |
|--------|------------|------------|-------------|----------|
| **Direct DEAPI** | Direct to DE-Server | All 152 props | 0.23ms/prop | Production, Scripts |
| **PyScope + DEAPI** | PyScope → DE-Server | All 152 props | 0.26ms/prop | **Recommended** |
| **PyScope Only** | PyScope Generic | 10 basic props | Fast | Testing, Simulation |

### **Successful Pattern: PyScope + Internal DEAPI**

```python
# 1. Connect via PyScope
camera = pyscope.registry.getClass("DEApollo")()

# 2. Access internal property management
camera.getPropertiesList()  # Populates properties_list
properties = camera.properties_list  # Gets all 152 properties

# 3. Read properties via PyScope DE interface
for prop_name in properties:
    value = camera.getProperty(prop_name)  # Full DEAPI access
```

## 🔧 **Camera Object Structure**

### **PyScope DE Camera Hierarchy**

```
CCDCamera (Base)
    ↓
DECameraBase (DE Common)
    ↓
DEApollo (Apollo Specific)
    ↓
Properties:
- model_name: "Apollo"
- properties_list: [152 properties]
- Methods: getProperty(), setProperty(), getImage()
- Internal: __deserver (DEAPI Client)
```

### **Key Camera Attributes**

```python
class DEApollo:
    # Connection
    model_name: str = "Apollo"
    connected: bool
    
    # Internal DEAPI Access
    __deserver: DEAPI.Client  # Internal DE-Server connection
    properties_list: List[str]  # All 152 properties
    
    # Standard PyScope Interface
    def getExposureTime() -> float
    def setExposureTime(ms: float)
    def getBinning() -> Dict[str, int]
    def getDimension() -> Dict[str, int]
    def getImage() -> np.ndarray
    
    # DE-Specific Interface
    def getProperty(name: str) -> Any
    def setProperty(name: str, value: Any)
    def getPropertiesList() -> None  # Populates properties_list
```

## 📊 **Property System Architecture**

### **Property Categories (152 Total)**

| Category | Count | Examples |
|----------|-------|----------|
| **Camera Info** | 8 | Model, SN, Firmware |
| **Temperature** | 5 | Detector Temp, Status, Control |
| **Acquisition** | 12 | Exposure Time, FPS, Mode |
| **Image Settings** | 18 | Size, Binning, Crop, ROI |
| **Processing** | 15 | Mode, Rotation, Gain |
| **Autosave** | 20 | Directory, Format, Status |
| **System** | 10 | Status, Position, Protection |
| **Hardware** | 25 | Frame Size, Pixel Depth |
| **Features** | 12 | AKRA, Counting, HDR |
| **References** | 8 | Dark, Gain References |
| **Advanced** | 19 | Engineering, Diagnostic |

### **Property Value Types**

```python
class PropertyType:
    READ_ONLY = "ReadOnly"      # Cannot be changed
    SET = "Set"                 # Fixed list of values
    RANGE = "Range"             # Min/Max numeric range
    ALLOW_ALL = "AllowAll"      # Any valid value
```

### **Property Access Patterns**

```python
# READ-ONLY Properties (Status/Info)
camera.getProperty("Camera Model")  # → "Apollo"
camera.getProperty("Temperature - Detector (Celsius)")  # → -25

# SET Properties (Predefined Options)
camera.setProperty("Exposure Mode", "Normal")  # ["Normal", "Dark", "Gain"]
camera.setProperty("Binning X", 2)  # [1, 2, 4, 8]

# RANGE Properties (Numeric Ranges)
camera.setProperty("Exposure Time (seconds)", 1.5)  # 0.001 - 3600.0
camera.setProperty("Crop Size X", 4096)  # 1 - 8192

# ALLOW_ALL Properties (Free Text)
camera.setProperty("Autosave Directory", "D:\\Custom\\Path")
```

## 🎛️ **Camera Control Patterns**

### **Standard Acquisition Workflow**

```python
# 1. Configure Camera
camera.setProperty("Exposure Time (seconds)", 2.0)
camera.setProperty("Frames Per Second", 20.0)
camera.setProperty("Exposure Mode", "Normal")

# 2. Set Image Parameters
camera.setProperty("Binning X", 1)
camera.setProperty("Binning Y", 1)
camera.setProperty("Crop Size X", 8192)
camera.setProperty("Crop Size Y", 8192)

# 3. Configure Processing
camera.setProperty("Image Processing - Mode", "Integrating")
camera.setProperty("Image Processing - Apply Gain on Final", "On")

# 4. Setup Autosave (Optional)
camera.setProperty("Autosave Final Image", "On")
camera.setProperty("Autosave File Format", "MRC")

# 5. Acquire Image
image = camera.getImage()  # Returns numpy array
```

### **Temperature Control Pattern**

```python
# Check Status
temp = camera.getProperty("Temperature - Detector (Celsius)")
status = camera.getProperty("Temperature - Detector Status")

# Control Temperature
camera.setProperty("Temperature - Control", "Cool Down")  # or "Warm Up"

# Monitor Until Ready
while camera.getProperty("Temperature - Detector Status") != "Cooled":
    time.sleep(1.0)
```

### **Image Processing Modes**

```python
# Integrating Mode (Standard)
camera.setProperty("Image Processing - Mode", "Integrating")
camera.setProperty("Exposure Time (seconds)", 2.0)

# Counting Mode (High Sensitivity)
camera.setProperty("Image Processing - Mode", "Counting")
camera.setProperty("Frames Per Second", 280.0)
```

## 🚀 **Performance Characteristics**

### **Property Access Performance**

- **Total Properties**: 152
- **Access Time**: ~0.26ms per property
- **Full Read**: ~40ms for all properties
- **Connection**: Shared DE-Server connection

### **Image Acquisition Performance**

- **Full Frame (8192×8192)**: ~1-5 seconds
- **Binned (4096×4096)**: ~0.5-2 seconds
- **Cropped Region**: Proportionally faster
- **Data Type**: uint16 (16-bit) or float32

### **Memory Usage**

- **Full Frame**: 134MB (8192² × 2 bytes)
- **Half Frame**: 33.5MB (4096² × 2 bytes)
- **Quarted**: 8.4MB (2048² × 2 bytes)

## 🔌 **Integration Patterns**

### **PyScope Registry Pattern**

```python
import pyscope.registry

# Get camera class
camera_class = pyscope.registry.getClass("DEApollo")
camera = camera_class()

# Camera automatically connects to DE-Server
# Properties become available immediately
```

### **Property Validation Pattern**

```python
def validate_property(camera, name, value):
    """Validate property before setting"""
    try:
        # Get current value to test if property exists
        current = camera.getProperty(name)
        
        # Attempt to set new value
        camera.setProperty(name, value)
        
        # Verify it was set correctly
        actual = camera.getProperty(name)
        return actual == value
        
    except Exception as e:
        return False, str(e)
```

### **Error Handling Pattern**

```python
def safe_property_access(camera, name, default=None):
    """Safe property access with fallback"""
    try:
        return camera.getProperty(name)
    except Exception as e:
        logger.warning(f"Property '{name}' access failed: {e}")
        return default
```

## 📁 **File System Integration**

### **Autosave Configuration**

```python
# Setup autosave directory
camera.setProperty("Autosave Directory", "D:\\Data\\Session_2025")

# Configure file formats
camera.setProperty("Autosave File Format", "MRC")  # or "TIFF", "DM4"
camera.setProperty("Autosave Final Image", "On")
camera.setProperty("Autosave Movie", "Off")

# Custom filename suffix
camera.setProperty("Autosave Filename Suffix", "_sample01")
```

### **Manual Save Pattern**

```python
# Acquire image
image = camera.getImage()

# Get image metadata
dataset_name = camera.getProperty("Dataset Name")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save manually
filename = f"{dataset_name}_{timestamp}.tiff"
Image.fromarray(image).save(filename)
```

## 🔄 **State Management**

### **Camera State Properties**

```python
# System Status
system_status = camera.getProperty("System Status")  # "OK", "Warning", "Error"

# Camera Position
position = camera.getProperty("Camera Position Status")  # "Extended", "Retracted"

# Protection Cover
cover = camera.getProperty("Protection Cover Status")  # "Open", "Closed"

# Acquisition Status
remaining = camera.getProperty("Remaining Number of Acquisitions")
total = camera.getProperty("Total Number of Acquisitions")
```

### **Configuration Backup/Restore**

```python
def backup_camera_config(camera):
    """Backup current camera configuration"""
    config = {}
    
    # Key properties to backup
    backup_props = [
        "Exposure Time (seconds)", "Frames Per Second", "Exposure Mode",
        "Binning X", "Binning Y", "Crop Size X", "Crop Size Y",
        "Image Processing - Mode", "Autosave File Format"
    ]
    
    for prop in backup_props:
        config[prop] = camera.getProperty(prop)
    
    return config

def restore_camera_config(camera, config):
    """Restore camera configuration"""
    for prop, value in config.items():
        camera.setProperty(prop, value)
```

## 🎯 **Best Practices**

### **Connection Management**

1. **Single Connection**: Use one PyScope camera instance per application
2. **Graceful Cleanup**: Always disconnect when done
3. **Error Recovery**: Handle connection drops gracefully
4. **Shared Access**: DE-Server supports multiple clients

### **Property Management**

1. **Batch Operations**: Group related property changes
2. **Validation**: Always validate critical properties
3. **Monitoring**: Poll status properties during acquisition
4. **Caching**: Cache read-only properties for performance

### **Image Acquisition**

1. **Pre-flight Check**: Verify camera state before acquisition
2. **Timeout Handling**: Set reasonable timeouts for acquisition
3. **Memory Management**: Process large images in chunks if needed
4. **Error Recovery**: Handle acquisition failures gracefully

### **Performance Optimization**

1. **Property Caching**: Cache frequently accessed read-only properties
2. **Batch Reads**: Read multiple related properties together
3. **Async Operations**: Use async patterns for long operations
4. **Resource Cleanup**: Release resources promptly

## 🔧 **Troubleshooting Guide**

### **Common Issues**

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection Failed | DE-Server not running | Start DE-Server application |
| Property Not Found | Invalid property name | Check property list |
| Invalid Value | Wrong type/range | Validate against schema |
| Acquisition Timeout | Camera busy | Check status, retry |
| Image Size Mismatch | Crop/binning changed | Verify geometry settings |

### **Diagnostic Properties**

```python
# Connection Status
server_version = camera.getProperty("Server Software Version")
camera_model = camera.getProperty("Camera Model")

# System Health
system_status = camera.getProperty("System Status")
temperature = camera.getProperty("Temperature - Detector (Celsius)")
vacuum_state = camera.getProperty("Vacuum State")

# Performance Metrics
fps_current = camera.getProperty("Frames Per Second")
fps_max = camera.getProperty("Frames Per Second (Max)")
grab_buffer_size = camera.getProperty("Grab Buffer Size")
```

This documentation provides the foundation for building robust applications with the DE Apollo camera using PyScope integration patterns.