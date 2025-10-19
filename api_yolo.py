from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics_Yolo.triton_client import (
    InferenceModule, 
    YOLOv8Postprocessor, 
    FPSCounter, 
    plot, 
    draw_performance_info, 
    draw_error_message
)
from ultralytics_Yolo.triton_client.config import CONFIDENCE_THRESHOLD

import cv2
import numpy as np
import io
from PIL import Image
import base64
import asyncio
import time

app = FastAPI(title="YOLOv8 Triton Video Stream API", version="1.0.0")

# Setup templates
templates = Jinja2Templates(directory="template")

# Global variables for FPS tracking
stream_fps = 0.0
total_frames_processed = 0
stream_start_time = time.time()

# Initialize inference modules
try:
    infer = InferenceModule("localhost:8001")
    post = YOLOv8Postprocessor(conf_threshold=CONFIDENCE_THRESHOLD)
    print("✅ Triton inference modules initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize inference modules: {e}")
    infer = None
    post = None

def generate_video_stream():
    """
    Generate video stream with YOLOv8 detection drawn on server
    """
    global stream_fps, total_frames_processed
    
    print("🎥 Attempting to open webcam...")
    cap = cv2.VideoCapture(0)  # Use webcam
    
    if not cap.isOpened():
        print("❌ Failed to open webcam index 0, trying index 1...")
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("❌ No webcam found! Check camera connection.")
        raise RuntimeError("Could not open webcam")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✅ Webcam opened successfully!")
    print(f"📐 Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"🎬 FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to read frame {frame_count}")
                break
            
            frame_count += 1
            total_frames_processed += 1
            
            # Update FPS counter
            fps_updated = fps_counter.update_frame()
            if fps_updated:
                stream_fps = fps_counter.current_fps  # Update global FPS
                print(f"📊 Stream FPS: {fps_counter.current_fps:.1f} | Frame: {frame_count}")
            
            try:
                # Run YOLOv8 inference on frame
                inference_start = time.time()
                result = infer.infer_array_image(frame, "yolov8")
                output0 = result["output0"]
                
                # Postprocess - get detections
                detections = post.process(output0)
                inference_time = (time.time() - inference_start) * 1000  # Convert to ms
                
                # Draw detections on frame (server-side plotting)
                frame_with_detections = plot(frame, detections, 
                                           result["ratio"], result["padding"])
                
                # Add performance info to frame using new function
                frame_with_info = draw_performance_info(
                    frame_with_detections,
                    fps_counter.get_fps_text(),
                    inference_time,
                    len(detections)
                )
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_info, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"❌ Error processing frame {frame_count}: {e}")
                # Send original frame with error message
                frame_with_error = draw_error_message(frame, "ERROR: Detection Failed", fps_counter.get_fps_text())
                _, buffer = cv2.imencode('.jpg', frame_with_error)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
    except Exception as e:
        print(f"❌ Stream error: {e}")
    finally:
        cap.release()
        print("📹 Webcam released")

@app.get("/video")
async def video_stream():
    """
    Video stream endpoint - returns MJPEG stream
    """
    if not infer or not post:
        raise HTTPException(status_code=500, detail="Inference modules not initialized")
    
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/detections")
async def get_detections():
    """
    Get current frame detections as JSON
    """
    if not infer or not post:
        raise HTTPException(status_code=500, detail="Inference modules not initialized")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")
    
    try:
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read frame")
        
        # Run inference
        result = infer.infer_array_image(frame, "yolov8")
        output0 = result["output0"]
        
        # Postprocess
        detections = post.process(output0)
        
        return JSONResponse(content={
            "detections": detections,
            "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
            "ratio": result["ratio"],
            "padding": result["padding"]
        })
        
    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def video_page(request: Request):
    """
    HTML page with video stream and client-side detection plotting
    """
    return templates.TemplateResponse("yolo.html", {
        "request": request,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict objects in uploaded image
    """
    if not infer or not post:
        raise HTTPException(status_code=500, detail="Inference modules not initialized")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        result = infer.infer_array_image(opencv_image, "yolov8")
        output0 = result["output0"]
        
        # Postprocess
        detections = post.process(output0)
        
        return JSONResponse(content={
            "detections": detections,
            "image_size": {"width": opencv_image.shape[1], "height": opencv_image.shape[0]},
            "model_info": {
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "total_detections": len(detections)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    """Detailed health check with performance metrics"""
    global stream_fps, total_frames_processed, stream_start_time
    
    uptime = time.time() - stream_start_time
    avg_fps = total_frames_processed / uptime if uptime > 0 else 0
    
    return {
        "triton_connected": infer is not None,
        "postprocessor_loaded": post is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "performance": {
            "current_fps": round(stream_fps, 2),
            "average_fps": round(avg_fps, 2),
            "total_frames": total_frames_processed,
            "uptime_seconds": round(uptime, 1)
        },
        "endpoints": {
            "video_stream": "/video",
            "html_page": "/",
            "image_prediction": "/predict",
            "detections": "/detections",
            "health": "/health",
            "test_camera": "/test-camera",
            "simple_video": "/video-simple"
        }
    }

@app.get("/fps")
async def get_fps_metrics():
    """Get real-time FPS metrics"""
    global stream_fps, total_frames_processed, stream_start_time
    
    uptime = time.time() - stream_start_time
    avg_fps = total_frames_processed / uptime if uptime > 0 else 0
    
    return JSONResponse(content={
        "current_fps": round(stream_fps, 2),
        "average_fps": round(avg_fps, 2),
        "total_frames": total_frames_processed,
        "uptime_seconds": round(uptime, 1),
        "frames_per_minute": round(total_frames_processed / (uptime / 60), 1) if uptime > 0 else 0
    })

@app.get("/test-camera")
async def test_camera():
    """Test camera without inference"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            return JSONResponse(content={
                "camera_available": False,
                "error": "No camera found"
            }, status_code=404)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return JSONResponse(content={
                "camera_available": False,
                "error": "Failed to read from camera"
            }, status_code=500)
        
        return JSONResponse(content={
            "camera_available": True,
            "frame_shape": frame.shape,
            "message": "Camera working successfully"
        })
        
    except Exception as e:
        return JSONResponse(content={
            "camera_available": False,
            "error": str(e)
        }, status_code=500)

@app.get("/video-simple")
async def video_stream_simple():
    """Simple video stream without inference for testing"""
    def generate_simple_video():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            
        if not cap.isOpened():
            return
        
        # Initialize FPS counter for simple stream
        fps_counter = FPSCounter()
        frame_count = 0
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                fps_updated = fps_counter.update_frame()
                
                if fps_updated:
                    print(f"📊 Simple Stream FPS: {fps_counter.current_fps:.1f}")
                
                # Add info to frame using new functions
                fps_text = fps_counter.get_fps_text()
                
                # Draw info manually for simple stream
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Simple Stream (No AI)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_simple_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )