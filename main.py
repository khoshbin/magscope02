import asyncio
import os
import sys
from fastapi import FastAPI,Request,Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Force local directory priority and remove problematic paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Remove ALL paths containing myami-beta
sys.path = [p for p in sys.path if 'myami-beta' not in p.lower()]

# Now your imports will work



# Add DEAPI to path
sys.path += ["DEAPI", "..\\DEAPI", "../DEAPI"]

# Import the camera router
from apollo_camera_router import router as camera_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global variable to store microscope connection
microscope = None

class Position(BaseModel):
    x: float
    y: float
    z: float





class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, timeout: int = 5):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            # Run the next middleware/path operation with a timeout
            response = await asyncio.wait_for(call_next(request), timeout=self.timeout)
            return response
        except TimeoutError:
            return JSONResponse(
                status_code=504,  # Gateway Timeout
                content={"detail": "Request timed out"}
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Apollo Camera Control API...")
    yield
    logger.info("Shutting down Apollo Camera Control API...")

# Create FastAPI app
app = FastAPI(
    title="Apollo Camera Control API",
    description="""
    Comprehensive FastAPI interface for Direct Electron Apollo camera control.

    This API provides full control over Apollo cameras including:
    - Connection management
    - Property configuration
    - Image acquisition with timeout handling
    - Temperature control
    - ROI management
    - Reference image capture

    ## Getting Started

    1. Connect to camera: `POST /camera/connect`
    2. List available cameras: `GET /camera/cameras`
    3. Select a camera: `POST /camera/cameras/{camera_name}/select`
    4. Take a quick image: `POST /camera/quick-image?timeout_seconds=300`

    ## Timeout Handling

    All long-running operations (like image acquisition) now support configurable timeouts
    to prevent hanging requests and improve responsiveness.

    ## Authentication

    This API currently runs without authentication. In production environments,
    consider adding proper authentication and authorization.
    """,
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)
from pyami import moduleconfig



def connect_to_fei_microscope():
    """
    Connect to FEI microscope using existing pyscope framework
    Returns microscope instance
    """
    try:
        # Initialize FEI TEM connection
        tem = fei.Krios()

        print("Successfully connected to FEI microscope")
        return tem
    except Exception as e:
        print(f"Failed to connect to FEI microscope: {e}")
        return None


def get_stage_position(tem):
    """
    Read stage position (x, y, z) from FEI microscope

    Args:
        tem: FEI microscope instance

    Returns:
        dict: Stage position with keys 'x', 'y', 'z', 'a', 'b' in meters/radians
    """
    try:
        position = tem.getStagePosition()
        return position
    except Exception as e:
        print(f"Error reading stage position: {e}")
        return None


def print_stage_position(position):
    """
    Print stage position in a readable format

    Args:
        position: dict with stage coordinates
    """
    if position:
        print("\n=== Stage Position ===")
        print(f"X: {position['x'] * 1e6:.2f} μm")
        print(f"Y: {position['y'] * 1e6:.2f} μm")
        print(f"Z: {position['z'] * 1e6:.2f} μm")
        if position['a'] is not None:
            print(f"Alpha (tilt): {position['a'] * 180 / 3.14159:.2f}°")
        if position['b'] is not None:
            print(f"Beta: {position['b'] * 180 / 3.14159:.2f}°")
        print("=====================")
    else:
        print("No position data available")



app.add_middleware(TimeoutMiddleware, timeout=300)
# Add CORS middleware for web client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the camera router
app.include_router(camera_router)


# Root endpoints
@app.get("/", tags=["root"])
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Apollo Camera Control API",
        "version": "2.1.0",
        "documentation": "/docs",
        "camera_endpoints": "/camera",
        "status": "online",
        "features": ["timeout_handling", "async_operations", "long_running_tasks"]
    }

@app.get("/scope-stage", tags=["root"])
async def scope_stage():

    tem = connect_to_fei_microscope()
    if not tem:
        sys.exit(1)

    try:
        # Single position reading
        print("\n1. Single position reading:")
        position = get_stage_position(tem)
        print_stage_position(position)


        # Example of checking if stage is moving
        print("\n3. Stage status check:")
        try:
            stage_status = tem.tecnai.Stage.Status
            if stage_status in (2, 3, 4):
                print("Stage is currently moving")
            else:
                print("Stage is ready/stopped")
        except:
            print("Could not check stage status")

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error during operation: {e}")
    finally:
        print("Disconnecting from microscope...")



@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "apollo-camera-api",
        "version": "2.1.0",
        "timeout_support": True
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": "An unexpected error occurred"
    }


# class SmartTimeoutMiddleware(BaseHTTPMiddleware):
#     """
#     Smart timeout middleware that applies different timeouts based on endpoint type
#     """
#
#     def __init__(self, app, default_timeout: int = 60):
#         super().__init__(app)
#         self.default_timeout = default_timeout
#
#         # Define endpoint-specific timeouts
#         self.endpoint_timeouts = {
#             "/camera/quick-image": 300,  # 5 minutes for image acquisition
#             "/camera/acquisition/start": 300,  # 5 minutes for starting acquisition
#             "/camera/image/latest": 120,  # 2 minutes for getting latest image
#             "/camera/references/dark": 180,  # 3 minutes for dark reference
#             "/camera/temperature/cooldown": 600,  # 10 minutes for cooldown
#             "/camera/temperature/warmup": 300,  # 5 minutes for warmup
#         }
#
#     async def dispatch(self, request: Request, call_next):
#         # Determine timeout based on endpoint
#         path = request.url.path
#         timeout = self.endpoint_timeouts.get(path, self.default_timeout)
#
#         # Check if timeout is specified in query parameters
#         if "timeout_seconds" in request.query_params:
#             try:
#                 timeout = float(request.query_params["timeout_seconds"])
#             except ValueError:
#                 pass
#
#         start_time = time.time()
#
#         try:
#             # Apply timeout to the request
#             response = await asyncio.wait_for(call_next(request), timeout=timeout)
#
#             # Add timing headers
#             process_time = time.time() - start_time
#             response.headers["X-Process-Time"] = str(process_time)
#             response.headers["X-Timeout-Used"] = str(timeout)
#
#             return response
#
#         except asyncio.TimeoutError:
#             process_time = time.time() - start_time
#             logger.warning(f"Request to {path} timed out after {timeout} seconds (actual: {process_time:.2f}s)")
#
#             return JSONResponse(
#                 status_code=504,  # Gateway Timeout
#                 content={
#                     "error": "Request timeout",
#                     "message": f"Request timed out after {timeout} seconds",
#                     "path": path,
#                     "timeout_seconds": timeout,
#                     "actual_time": process_time
#                 },
#                 headers={
#                     "X-Timeout-Used": str(timeout),
#                     "X-Process-Time": str(process_time)
#                 }
#             )
#
#         except Exception as e:
#             process_time = time.time() - start_time
#             logger.error(f"Error in timeout middleware for {path}: {str(e)}")
#
#             return JSONResponse(
#                 status_code=500,
#                 content={
#                     "error": "Internal server error",
#                     "message": str(e),
#                     "path": path,
#                     "process_time": process_time
#                 },
#                 headers={
#                     "X-Process-Time": str(process_time)
#                 }
#             )
#
#
# class RequestLoggingMiddleware(BaseHTTPMiddleware):
#     """
#     Middleware to log request details for debugging timeout issues
#     """
#
#     async def dispatch(self, request: Request, call_next):
#         start_time = time.time()
#
#         # Log request start
#         logger.info(f"Request started: {request.method} {request.url.path}")
#
#         try:
#             response = await call_next(request)
#             process_time = time.time() - start_time
#
#             # Log successful completion
#             logger.info(
#                 f"Request completed: {request.method} {request.url.path} "
#                 f"- Status: {response.status_code} - Time: {process_time:.2f}s"
#             )
#
#             return response
#
#         except Exception as e:
#             process_time = time.time() - start_time
#
#             # Log error
#             logger.error(
#                 f"Request failed: {request.method} {request.url.path} "
#                 f"- Error: {str(e)} - Time: {process_time:.2f}s"
#             )
#
#             raise
#
#
# # Updated main.py with middleware
# def create_app_with_timeouts():
#     """Create FastAPI app with timeout middleware configured"""
#
#     app = FastAPI(
#         title="Apollo Camera Control API",
#         description="FastAPI interface for Apollo camera with smart timeout handling",
#         version="2.1.0"
#     )
#
#     # Add timeout middleware (order matters - add before CORS)
#     app.add_middleware(SmartTimeoutMiddleware, default_timeout=60)
#     app.add_middleware(RequestLoggingMiddleware)
#
#     # Add CORS middleware
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )
#
#     return app


# Alternative: Use Starlette's built-in TimeoutMiddleware for simple cases








if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Apollo Camera Control API server with extended timeouts...")

    # Enhanced uvicorn configuration for long-running operations
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=300,  # Keep connections alive for 5 minutes
        timeout_graceful_shutdown=60,  # Wait 60 seconds for graceful shutdown
        limit_concurrency=100,  # Limit concurrent connections
        limit_max_requests=1000,  # Restart worker after 1000 requests to prevent memory leaks
        # Additional timeout configurations
        access_log=True,
        use_colors=True
    )