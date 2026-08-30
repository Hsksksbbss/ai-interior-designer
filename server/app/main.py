"""
AI Interior Designer - Backend Server
FastAPI application for handling design requests
"""

import sys
import os

# Add parent directory to path so we can import ai_generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, status, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import bcrypt
import uuid
from datetime import datetime
from dotenv import load_dotenv
from app.database import init_db, get_db
from app import models  # Import models to register them with Base
from app.models import User
from app.schemas import UserSignup, UserLogin, AuthResponse, UserResponse
from app.ai_generator import analyze_room_image, analyze_furniture_design, analyze_plants_decor, analyze_smart_house

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Interior Designer API",
    description="Backend API for AI-powered interior design generation",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on application startup"""
    init_db()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_http_error_detail(message: str) -> str:
    """Normalize backend errors so the frontend shows a clean message."""
    lowered = message.lower()
    if (
        "credit_balance_exhausted" in lowered
        or "insufficient_quota" in lowered
        or "no credits remaining" in lowered
        or "quota exhausted" in lowered
    ):
        return "OpenAI quota exhausted. Please add credits to your OpenAI account and try again."
    return message


# Routes
@app.get("/")
async def read_root():
    """Test route to verify backend is running"""
    return {"message": "Backend Running Successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: UserSignup, db: Session = Depends(get_db)):
    """
    User signup endpoint - Creates a new user account
    
    - **name**: User's full name
    - **email**: User's email address (must be valid and unique)
    - **password**: User's password (minimum 8 characters)
    
    Returns a success message with user details
    
    Raises:
        HTTPException 400: If email already exists
        HTTPException 500: If database error occurs
    """
    try:
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered. Please use a different email or login."
            )
        
        # Hash the password using bcrypt
        password_hash = bcrypt.hashpw(
            request.password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        # Create new user
        new_user = User(
            name=request.name,
            email=request.email,
            password=password_hash,
            is_active=True
        )
        
        # Save to database
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Return success response
        return AuthResponse(
            message="Account created successfully!",
            user=UserResponse.model_validate(new_user)
        )
    
    except IntegrityError:
        # Handle database integrity errors (e.g., duplicate email)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered. Please use a different email."
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Handle unexpected errors
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during signup. Please try again later."
        )


@app.post("/login", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def login(request: UserLogin, db: Session = Depends(get_db)):
    """
    User login endpoint - Authenticates a user and returns their details
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns a success message with user details and access token
    
    Raises:
        HTTPException 401: If email not found or password is incorrect
        HTTPException 500: If database error occurs
    """
    try:
        # Query user by email
        user = db.query(User).filter(User.email == request.email).first()
        
        # Check if user exists
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password. Please check your credentials."
            )
        
        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account has been deactivated. Please contact support."
            )
        
        # Verify password
        password_match = bcrypt.checkpw(
            request.password.encode('utf-8'),
            user.password.encode('utf-8')
        )
        
        if not password_match:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password. Please check your credentials."
            )
        
        # Return success response
        return AuthResponse(
            message="Login successful!",
            user=UserResponse.model_validate(user)
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again later."
        )


@app.post("/upload-image", status_code=status.HTTP_200_OK)
async def upload_image(
    file: UploadFile = File(...),
    section: str = Form(...),
    language: str = Form(default='english')
):
    """
    Image upload endpoint - Accepts image file from frontend
    
    - **file**: Image file (JPG, JPEG, PNG)
    - **section**: Design section (wall-painting, furniture, plants, etc.)
    
    Returns upload confirmation with file details
    
    Raises:
        HTTPException 400: If file type is invalid
        HTTPException 500: If file processing fails
    """
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: JPG, JPEG, PNG"
            )
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds 10MB limit"
            )
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{section}_{timestamp}_{unique_id}{file_extension}"
        file_path = os.path.join(uploads_dir, new_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return {
            "message": "Image uploaded successfully!",
            "filename": new_filename,
            "original_filename": file.filename,
            "section": section,
            "file_size": len(file_content),
            "upload_time": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image: {str(e)}"
        )


@app.post("/generate-design", status_code=status.HTTP_200_OK)
async def generate_design(
    image_path: str = Form(...),
    section: str = Form(...),
    language: str = Form(default='english')
):
    """
    AI room analysis endpoint - Analyzes room image and provides design suggestions using OpenAI Vision API
    
    - **image_path**: Path to the uploaded image (e.g., "uploads/wall-painting_20260528_145230_a3c2f1e9.jpg")
    - **section**: Design section (wall-painting, furniture, plants, general)
    
    Returns structured design suggestions including wall colors, furniture, plants, and tips
    
    Raises:
        HTTPException 400: If image path is invalid or not found
        HTTPException 500: If analysis fails
    """
    try:
        # Validate image path exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image not found: {image_path}"
            )
        
        print(f"Starting room analysis for: {image_path}")
        
        # Analyze room using OpenAI Vision API
        analysis = analyze_room_image(
            image_path=image_path,
            design_section=section,
            language=language
        )
        
        return {
            "message": "Room analyzed successfully!",
            "analysis": analysis,
            "original_image_path": image_path,
            "section": section,
            "analysis_time": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing error: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=format_http_error_detail(f"Failed to analyze room: {str(e)}")
        )


@app.post("/generate-furniture-design", status_code=status.HTTP_200_OK)
async def generate_furniture_design(
    image_path: str = Form(...),
    language: str = Form(default='english')
):
    """
    AI furniture design endpoint - Analyzes room image and provides furniture recommendations using OpenAI Vision API
    
    - **image_path**: Path to the uploaded image (e.g., "uploads/furniture_20260528_145230_a3c2f1e9.jpg")
    
    Returns furniture recommendations with placement, style, color, and purchase options for Indian e-commerce
    
    Raises:
        HTTPException 400: If image path is invalid or not found
        HTTPException 500: If analysis fails
    """
    try:
        # Validate image path exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image not found: {image_path}"
            )
        
        print(f"Starting furniture design analysis for: {image_path}")
        
        # Analyze furniture using OpenAI Vision API
        analysis = analyze_furniture_design(image_path=image_path, language=language)
        
        return {
            "message": "Furniture design analyzed successfully!",
            "analysis": analysis,
            "original_image_path": image_path,
            "analysis_section": "furniture",
            "analysis_time": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing error: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze furniture design: {str(e)}"
        )


@app.post("/generate-plants-decor", status_code=status.HTTP_200_OK)
async def generate_plants_decor(
    image_path: str = Form(...),
    language: str = Form(default='english')
):
    """
    AI plants & decor endpoint - Analyzes room image and provides plant and decor recommendations using OpenAI Vision API
    
    - **image_path**: Path to the uploaded image (e.g., "uploads/plants_20260528_145230_a3c2f1e9.jpg")
    
    Returns plant and decor recommendations with placement, style, and purchase options for Indian e-commerce
    
    Raises:
        HTTPException 400: If image path is invalid or not found
        HTTPException 500: If analysis fails
    """
    try:
        # Validate image path exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image not found: {image_path}"
            )
        
        print(f"Starting plants & decor analysis for: {image_path}")
        
        # Analyze plants & decor using OpenAI Vision API
        analysis = analyze_plants_decor(image_path=image_path, language=language)
        
        return {
            "message": "Plants & decor analyzed successfully!",
            "analysis": analysis,
            "original_image_path": image_path,
            "analysis_section": "plants-decor",
            "analysis_time": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing error: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze plants and decor: {str(e)}"
        )


@app.post("/analyze-smart-house", status_code=status.HTTP_200_OK)
async def analyze_smart_house_endpoint(
    family_members: str = Form(...),
    adults: str = Form(...),
    children: str = Form(...),
    senior_citizens: str = Form(...),
    plot_size: str = Form(...),
    floors: str = Form(...),
    budget: str = Form(...),
    language: str = Form(default='english')
):
    """
    Smart House Analysis endpoint - Analyzes complete home requirements and provides comprehensive design recommendations.
    """
    try:
        print(f"\n{'='*60}")
        print(f"SMART HOUSE ANALYSIS REQUEST RECEIVED")
        print(f"{'='*60}")
        print(f"Raw values received:")
        print(f"  family_members: {family_members} (type: {type(family_members).__name__})")
        print(f"  adults: {adults}")
        print(f"  children: {children}")
        print(f"  senior_citizens: {senior_citizens}")
        print(f"  plot_size: {plot_size}")
        print(f"  floors: {floors}")
        print(f"  budget: {budget}")
        
        # Convert all values to integers/proper types
        try:
            family_members_int = int(family_members)
            adults_int = int(adults)
            children_int = int(children)
            senior_citizens_int = int(senior_citizens)
            floors_int = int(floors)
            
            # Extract numeric values from plot_size and budget (in case they have formatting)
            plot_size_numeric = int(''.join(filter(str.isdigit, str(plot_size))))
            budget_numeric = int(''.join(filter(str.isdigit, str(budget))))
        except (ValueError, TypeError) as e:
            print(f"ERROR: Failed to convert parameters to correct types: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameter types: {str(e)}"
            )
        
        print(f"\nParsed values:")
        print(f"  Family: {family_members_int} members ({adults_int} adults, {children_int} children, {senior_citizens_int} seniors)")
        print(f"  Property: {plot_size_numeric} sq ft, {floors_int} floors, Budget: ₹{budget_numeric}")
        
        # Validate parameters
        if family_members_int < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Family members must be at least 1"
            )
        
        if (adults_int + children_int + senior_citizens_int) > family_members_int:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sum of adults, children, and seniors cannot exceed total family members"
            )
        
        if plot_size_numeric < 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plot size must be at least 500 sq ft"
            )
        
        if floors_int < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of floors must be at least 1"
            )
        
        print(f"\nValidation passed. Calling analyze_smart_house()...")
        
        # Analyze smart house
        analysis = analyze_smart_house(
            family_members=family_members_int,
            adults=adults_int,
            children=children_int,
            senior_citizens=senior_citizens_int,
            plot_size=str(plot_size_numeric),
            floors=floors_int,
            budget=str(budget_numeric),
            language=language
        )
        
        print(f"Analysis completed successfully")
        print(f"{'='*60}\n")
        
        return {
            "message": "Smart house analyzed successfully!",
            "analysis": analysis,
            "analysis_section": "smart-house-analysis",
            "analysis_time": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    
    except ValueError as e:
        print(f"VALUE ERROR: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {str(e)}"
        )
    
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {str(e)}")
        print(f"{'='*60}\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze smart house: {str(e)}"
        )

    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
