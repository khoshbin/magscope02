#!/usr/bin/env python3
"""
Complete FastAPI Application with PyScope Camera Integration
Main application file that brings together all camera functionality
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
from datetime import datetime
import os

# Import our custom routers
from camera.pyscope_camera_router import router as pyscope_camera_router

# from apollo_camera_router import router as deapi_camera_router  # Your original DEAPI router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="DE Apollo Camera Control API",
    description="Complete FastAPI application for controlling DE Apollo cameras via PyScope and DEAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pyscope_camera_router)
# app.include_router(deapi_camera_router)  # Your original DEAPI router

# Global application state
app_state = {
    "startup_time": datetime.now(),
    "camera_connections": {
        "pyscope": {"connected": False, "camera_name": None},
        "deapi": {"connected": False, "camera_name": None}
    }
}


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DE Apollo Camera Control API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .section { margin: 20px 0; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; }
            .method { font-weight: bold; color: #e74c3c; }
            .path { font-family: monospace; color: #2c3e50; }
            .description { color: #7f8c8d; margin-top: 5px; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.ok { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .status.warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 DE Apollo Camera Control API</h1>
            <p>Complete FastAPI application for controlling DE Apollo cameras</p>
        </div>

        <div class="status ok">
            <h3>📡 API Status: Online</h3>
            <p>The camera control API is running and ready to accept connections.</p>
        </div>

        <div class="section">
            <h2>🔗 Quick Links</h2>
            <ul>
                <li><a href="/docs">📚 Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">📖 Alternative API Documentation (ReDoc)</a></li>
                <li><a href="/status">📊 Application Status</a></li>
                <li><a href="/pyscope-camera/health">💊 Camera Health Check</a></li>
            </ul>
        </div>

        <div class="section">
            <h2>🚀 PyScope Camera Endpoints</h2>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/pyscope-camera/connect</span>
                <div class="description">Connect to DE Apollo camera via PyScope</div>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/pyscope-camera/status</span>
                <div class="description">Get current camera connection status</div>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/pyscope-camera/properties/all</span>
                <div class="description">Get all camera properties (152 properties)</div>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/pyscope-camera/acquisition/capture</span>
                <div class="description">Capture a single image from the camera</div>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/pyscope-camera/temperature</span>
                <div class="description">Get camera temperature status</div>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/pyscope-camera/properties/batch</span>
                <div class="description">Set multiple camera properties at once</div>
            </div>
        </div>

        <div class="section">
            <h2>🎛️ Camera Features</h2>
            <ul>
                <li><strong>Full Property Access:</strong> All 152 DE Apollo camera properties</li>
                <li><strong>Image Acquisition:</strong> Full resolution (8192×8192) and cropped images</li>
                <li><strong>Temperature Control:</strong> Monitor and control detector temperature</li>
                <li><strong>Real-time Configuration:</strong> Adjust exposure, binning, ROI settings</li>
                <li><strong>Property Search:</strong> Find properties by name or category</li>
                <li><strong>Performance Monitoring:</strong> Track acquisition and property access speed</li>
                <li><strong>Health Monitoring:</strong> Continuous camera status monitoring</li>
            </ul>
        </div>

        <div class="section">
            <h2>⚡ Getting Started</h2>
            <ol>
                <li><strong>Connect Camera:</strong> POST to <code>/pyscope-camera/connect</code></li>
                <li><strong>Check Status:</strong> GET <code>/pyscope-camera/status</code></li>
                <li><strong>Configure Settings:</strong> Use property endpoints to set exposure, binning, etc.</li>
                <li><strong>Capture Images:</strong> POST to <code>/pyscope-camera/acquisition/capture</code></li>
                <li><strong>Monitor Health:</strong> GET <code>/pyscope-camera/health</code></li>
            </ol>
        </div>

        <div class="section">
            <h2>📊 Example API Calls</h2>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
# Connect to camera
curl -X POST "http://localhost:8000/pyscope-camera/connect" \\
     -H "Content-Type: application/json" \\
     -d '{"camera_name": "DEApollo"}'

# Get camera status  
curl "http://localhost:8000/pyscope-camera/status"

# Set exposure time
curl -X POST "http://localhost:8000/pyscope-camera/properties/batch" \\
     -H "Content-Type: application/json" \\
     -d '{"properties": {"Exposure Time (seconds)": 2.0}}'

# Capture image
curl -X POST "http://localhost:8000/pyscope-camera/acquisition/capture?include_image_data=false"
            </pre>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/status")
async def get_application_status():
    """Get overall application status"""
    uptime = datetime.now() - app_state["startup_time"]

    return {
        "application": {
            "name": "DE Apollo Camera Control API",
            "version": "1.0.0",
            "status": "running",
            "uptime_seconds": int(uptime.total_seconds()),
            "startup_time": app_state["startup_time"].isoformat()
        },
        "cameras": app_state["camera_connections"],
        "endpoints": {
            "pyscope_camera": "/pyscope-camera",
            "documentation": "/docs",
            "health": "/pyscope-camera/health"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api-info")
async def get_api_info():
    """Get detailed API information"""
    return {
        "title": "DE Apollo Camera Control API",
        "description": "Complete FastAPI application for controlling DE Apollo cameras",
        "version": "1.0.0",
        "features": [
            "PyScope camera integration",
            "Full property access (152 properties)",
            "Image acquisition and processing",
            "Temperature monitoring and control",
            "Real-time configuration",
            "Performance monitoring",
            "Health checking"
        ],
        "camera_support": {
            "models": ["DE Apollo", "DE12"],
            "connection_methods": ["PyScope", "Direct DEAPI"],
            "property_count": 152,
            "max_resolution": "8192x8192",
            "data_types": ["uint16", "float32"]
        },
        "endpoints": {
            "connection": [
                "POST /pyscope-camera/connect",
                "POST /pyscope-camera/disconnect",
                "GET /pyscope-camera/status"
            ],
            "properties": [
                "GET /pyscope-camera/properties",
                "GET /pyscope-camera/properties/all",
                "GET /pyscope-camera/properties/essential",
                "POST /pyscope-camera/properties/batch"
            ],
            "acquisition": [
                "POST /pyscope-camera/acquisition/configure",
                "POST /pyscope-camera/acquisition/capture",
                "POST /pyscope-camera/acquisition/quick-capture"
            ],
            "temperature": [
                "GET /pyscope-camera/temperature",
                "POST /pyscope-camera/temperature/cooldown",
                "POST /pyscope-camera/temperature/warmup"
            ],
            "system": [
                "GET /pyscope-camera/info/system",
                "GET /pyscope-camera/info/performance",
                "GET /pyscope-camera/health"
            ]
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "available_endpoints": [
                "/docs - API Documentation",
                "/pyscope-camera/* - Camera Control Endpoints",
                "/status - Application Status",
                "/api-info - API Information"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 DE Apollo Camera Control API starting up...")

    # Test PyScope availability
    try:
        import pyscope.registry
        logger.info("✓ PyScope is available")
    except ImportError:
        logger.warning("⚠️ PyScope not available - camera functionality will be limited")

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    logger.info("✅ Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("🛑 DE Apollo Camera Control API shutting down...")

    # Disconnect cameras if connected
    try:
        from camera.pyscope_camera_router import camera_manager
        if camera_manager.connected:
            camera_manager.disconnect()
            logger.info("✓ PyScope camera disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting camera: {e}")

    logger.info("✅ Application shutdown complete")


# Development configuration
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DE Apollo Camera Control API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()

    # Configuration for different environments
    config = {
        "app": "main:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "reload": args.reload
    }

    print(f"""
🎯 DE Apollo Camera Control API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 Server: http://{args.host}:{args.port}
📚 Docs: http://{args.host}:{args.port}/docs
🔧 API Info: http://{args.host}:{args.port}/api-info
💊 Health: http://{args.host}:{args.port}/pyscope-camera/health
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    # Run the application
    uvicorn.run(**config)