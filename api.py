# Standard library imports
import base64
import os
from pathlib import Path

# Third-party imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

# Local imports
from RizNet.infer_riznet import InferenceModule


# Initialize FastAPI app
app = FastAPI(
    title="ResNet Image Classification API", 
    description="API for image classification using ResNet models via Triton",
    version="1.0.0"
)

# Initialize templates
templates = Jinja2Templates(directory="template")

# Configure CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_imagenet_classes():
    """Load ImageNet class names from file with proper error handling"""
    try:
        # Use absolute path to avoid issues
        script_dir = Path(__file__).parent
        classes_path = script_dir / "RizNet" / "imagenet_classes.txt"
        
        with open(classes_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"⚠️ ImageNet classes file not found at: {classes_path}")
        # Return dummy classes as fallback
        return [f"class_{i}" for i in range(1000)]
    except Exception as e:
        print(f"⚠️ Error loading ImageNet classes: {e}")
        return [f"class_{i}" for i in range(1000)]


# Load ImageNet class names
class_names = load_imagenet_classes()

# Initialize inference module
inference_module = InferenceModule()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main index page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(..., description="Image file for classification"),
    model_name: str = Form(..., description="Model name for inference")
):
    """
    Perform image classification using ResNet models via Triton.

    Args:
        file (UploadFile): The uploaded image file
        model_name (str): Name of the model to use for inference

    Returns:
        dict: Contains class name and confidence logit

    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )

        # Convert uploaded file to base64
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file provided"
            )
            
        img_base64 = base64.b64encode(contents).decode("utf-8")

        # Perform inference using specified model
        result = await inference_module.infer_image(img_base64, model_name=model_name)

        # Extract class ID and logit
        class_id = result["class_id"]
        logit = round(result["logit"], 3)

        # Validate class_id bounds
        if not (0 <= class_id < len(class_names)):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid class ID: {class_id}"
            )

        # Get class name from ImageNet list
        class_name = class_names[class_id]
        
        # Log result for debugging
        print(f"✅ Prediction: {class_name} (logit: {logit})")
        
        return {
            "class_name": class_name,
            "class_id": class_id,
            "confidence_logit": logit,
            "model_used": model_name
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        print(f"❌ Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ResNet Image Classification API"}


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404 errors"""
    return Response(status_code=204)


# To start the server, run:
# python -m uvicorn api:app --reload --port 5000