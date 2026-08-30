"""
AI Room Analysis Service using OpenAI Vision API
Analyzes uploaded room images and provides interior design suggestions

"""




import os
import base64
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Return the Gemini API key from the active env settings."""
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to your .env file")
    return api_key


def get_model_name() -> str:
    """Return the configured Gemini model name, defaulting to a model that is available."""
    configured = os.getenv("GEMINI_MODEL")
    if configured and configured.strip():
        return configured.strip()
    return "gemini-2.5-flash"


def call_gemini_api(prompt: str, image_data: Optional[str] = None) -> str:
    """Call the Gemini REST API and return the generated text response."""
    api_key = get_api_key()
    model_name = get_model_name()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    parts = [{"text": prompt}]
    if image_data:
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": image_data,
            }
        })

    payload = {"contents": [{"parts": parts}]}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise RuntimeError(
                f"Gemini model '{model_name}' is unavailable. Set GEMINI_MODEL=gemini-2.5-flash in your .env file."
            ) from exc
        raise RuntimeError(f"Gemini API request failed: {detail}") from exc

    candidates = result.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {result}")

    text_parts = []
    for part in candidates[0].get("content", {}).get("parts", []):
        if "text" in part:
            text_parts.append(part["text"])

    if not text_parts:
        raise RuntimeError(f"Gemini API response did not include text: {result}")

    return "".join(text_parts).strip()


def get_language_instruction(language: str = "english") -> str:
    """Generate language instruction for AI prompts"""
    if language == "bengali":
        return "IMPORTANT: Return ALL output in Bengali script. Use only Bengali language for all text, including JSON keys and values."
    return "IMPORTANT: Return ALL output in English."


def format_openai_error(error: Exception) -> str:
    """Convert AI provider exceptions into user-friendly messages."""
    error_text = str(error).lower()

    if (
        "credit_balance_exhausted" in error_text
        or "insufficient_quota" in error_text
        or "no credits remaining" in error_text
        or ("quota" in error_text and "exhausted" in error_text)
        or ("quota" in error_text and "exceeded" in error_text)
    ):
        return "Gemini API quota exhausted. Please add credits to your Gemini account and try again."

    if (
        "incorrect api key" in error_text
        or "invalid api key" in error_text
        or ("api key" in error_text and "invalid" in error_text)
        or "forbidden" in error_text
    ):
        return "Gemini API key is invalid or missing. Please check your GEMINI_API_KEY configuration."

    if "429" in error_text or "rate limit" in error_text:
        return "Gemini request limit reached. Please wait a moment and try again."

    if "model" in error_text and "not found" in error_text:
        return "The configured Gemini model is unavailable. Please check the backend model setting."

    return f"Gemini API request failed: {error}"


def compress_image_for_api(image_path: str, max_dimension: int = 1024, quality: int = 85) -> str:
    """
    Compress image for OpenAI API to reduce tokens and costs.
    
    Args:
        image_path (str): Path to the image file
        max_dimension (int): Maximum width or height (default 1024)
        quality (int): JPEG quality 1-100 (default 85)
    
    Returns:
        str: Base64 encoded compressed image
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if necessary
        if image.width > max_dimension or image.height > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # Compress to JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        
        # Return base64 encoded
        compressed_data = base64.standard_b64encode(buffer.read()).decode("utf-8")
        print(f"Image compression successful. Base64 length: {len(compressed_data)} characters")
        return compressed_data
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"IMAGE COMPRESSION ERROR")
        print(f"{'='*60}")
        print(f"Image Path: {image_path}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Full Exception: {repr(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise ValueError(f"Failed to compress image: {str(e)}") from e


def analyze_room_image(
    image_path: str,
    design_section: str = "wall-painting",
    language: str = "english",
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    Analyze a room image using OpenAI Vision API and return design suggestions.
    Response format varies based on design_section parameter.
    
    Args:
        image_path (str): Path to the uploaded room image
        design_section (str): Design section type (default: wall-painting)
            - "wall-painting": Detailed wall color recommendations with hex codes
            - "furniture": Furniture suggestions and placement
            - "plants": Plant and decor suggestions
            - "general": Comprehensive multi-category analysis
        model (str): OpenAI model to use (default: gpt-4o-mini)
    
    Returns:
        Dict: Structured analysis. Format depends on design_section:
        
        For wall-painting:
        {
            "wall_recommendations": [
                {
                    "wall_position": "left wall",
                    "recommended_color": "Soft sage green",
                    "hex_code": "#9CAF88",
                    "reason": "Complements natural light and makes space feel larger",
                    "paint_products": [
                        {
                            "brand": "Asian Paints",
                            "shade_name": "Green Mint",
                            "color_family": "Green",
                            "official_link": "https://www.asianpaints.com/paint-colours.html"
                        }
                    ]
                }
            ],
            "overall_style": "Modern minimalist",
            "lighting_analysis": "Room has good natural light from windows",
            "analysis_section": "wall-painting",
            "status": "success"
        }
        
        For furniture:
        {
            "furniture_recommendations": [
                {
                    "furniture_item": "sofa",
                    "placement": "against left wall",
                    "recommended_style": "modern minimalist",
                    "recommended_color": "gray",
                    "reason": "Complements room size and light",
                    "purchase_options": [
                        {"store": "Amazon", "search_link": "..."},
                        {"store": "Flipkart", "search_link": "..."}
                    ]
                }
            ],
            "room_analysis": "Small compact living room",
            "space_constraints": "Limited floor space",
            "overall_style": "contemporary minimalist",
            "analysis_section": "furniture",
            "status": "success"
        }
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is invalid
        RuntimeError: If API call fails
    """
    try:
        # Validate image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Analyzing room image: {image_path}")
        
        # Normalize design_section to match backend values
        # Frontend sends: wall-painting, furniture, plants-decor, general
        # Backend uses: wall-painting, furniture, plants, general
        if design_section == "plants-decor":
            design_section = "plants"
        
        # Compress image for API efficiency
        image_data = compress_image_for_api(image_path)
        
        # Create section-specific prompt
        section_prompts = {
            "wall-painting": "Focus EXCLUSIVELY on wall colors and paint finishes. DO NOT suggest furniture or plants.",
            "furniture": "Focus on furniture arrangement, style compatibility, and placement optimization.",
            "plants": "Focus on plant types, placement, and decorative foliage suitable for the space.",
            "general": "Provide comprehensive design suggestions for all aspects."
        }
        
        section_prompt = section_prompts.get(design_section, section_prompts["general"])
        
        # Create section-specific response format and prompt
        if design_section == "wall-painting":
            # Wall painting specific prompt with real paint brand shade matching
            lang_instruction = get_language_instruction(language)
            analysis_prompt = f"""{lang_instruction}

Analyze this room image. Suggest wall colors for 2-3 different walls.

For each wall, recommend a real color that matches shades from Asian Paints, Berger Paints, or Dulux.

IMPORTANT RULES:
- Only suggest shade names that ACTUALLY exist in official brand catalogues
- Do NOT invent shade codes or product codes
- Do NOT hallucinate
- If unsure about exact shade name, suggest the color family (Beige, Blue, Green, etc)
- Provide official colour catalogue links only

Return ONLY this JSON format (no markdown or code blocks):
{{
    "wall_recommendations": [
        {{
            "wall_position": "front wall",
            "recommended_color": "color name (e.g. Warm Beige, Soft Green)",
            "hex_code": "#XXXXXX",
            "reason": "why this color works for this wall",
            "paint_products": [
                {{
                    "brand": "Asian Paints",
                    "shade_name": "official shade name from Asian Paints catalogue",
                    "color_family": "color family (Beige/Blue/Green/Gray/etc)",
                    "official_link": "https://www.asianpaints.com/"
                }},
                {{
                    "brand": "Berger Paints",
                    "shade_name": "official shade name from Berger catalogue",
                    "color_family": "color family",
                    "official_link": "https://www.bergerpaints.com/"
                }},
                {{
                    "brand": "Dulux",
                    "shade_name": "official shade name from Dulux catalogue",
                    "color_family": "color family",
                    "official_link": "https://www.dulux.in/"
                }}
            ]
        }}
    ],
    "overall_style": "style detected",
    "lighting_analysis": "lighting notes"
}}"""
        else:
            # Default multi-category prompt
            lang_instruction = get_language_instruction(language)
            analysis_prompt = f"""{lang_instruction}

Analyze this room image and provide interior design suggestions.
{section_prompt}

Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:
{{
    "wall_colors": ["color1", "color2", "color3"],
    "furniture": ["item1", "item2", "item3"],
    "plants": ["plant1", "plant2", "plant3"],
    "tips": ["tip1", "tip2", "tip3"],
    "overall_style": "style name",
    "color_palette": "description of recommended colors"
}}

Be specific and actionable. Provide 2-3 suggestions per category."""
        
        # Call Gemini Vision API
        print("Calling Gemini API for analysis...")
        try:
            response_text = call_gemini_api(analysis_prompt, image_data)
            print("Gemini API call successful!")
        except Exception as api_error:
            print(f"\n{'='*60}")
            print(f"GEMINI API ERROR DETAILS")
            print(f"{'='*60}")
            print(f"Error Type: {type(api_error).__name__}")
            print(f"Error Message: {str(api_error)}")
            print(f"Full Exception: {repr(api_error)}")
            print(f"{'='*60}\n")
            raise RuntimeError(format_openai_error(api_error)) from api_error

        try:
            print(f"Raw response received ({len(response_text)} characters)")
            print(f"First 200 chars: {response_text[:200]}")
        except Exception as e:
            print(f"ERROR: Failed to extract response text")
            print(f"Error: {str(e)}")
            raise RuntimeError(f"Failed to extract response text: {str(e)}") from e
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Clean up response - remove common issues
        response_text = response_text.strip()
        if response_text.startswith("{"):
            # Find the last closing brace
            last_brace = response_text.rfind("}")
            if last_brace != -1:
                response_text = response_text[:last_brace + 1]
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
            print(f"JSON parsing successful. Keys: {list(analysis.keys())}")
        except json.JSONDecodeError as json_error:
            print(f"\n{'='*60}")
            print(f"JSON PARSE ERROR")
            print(f"{'='*60}")
            print(f"Error: {str(json_error)}")
            print(f"Line {json_error.lineno}, Column {json_error.colno}")
            print(f"Problem: {json_error.msg}")
            print(f"\nResponse text (first 500 chars):")
            print(response_text[:500])
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to parse API response as JSON: {str(json_error)}") from json_error
        
        # Ensure all required fields exist based on section type
        if design_section == "wall-painting":
            # For wall painting, ensure wall_recommendations structure with real paint shade matching
            if "wall_recommendations" not in analysis:
                analysis["wall_recommendations"] = []
            
            # Ensure each recommendation has required fields including paint products
            for rec in analysis.get("wall_recommendations", []):
                if "wall_position" not in rec:
                    rec["wall_position"] = "Unknown"
                if "recommended_color" not in rec:
                    rec["recommended_color"] = "Color TBD"
                if "hex_code" not in rec:
                    rec["hex_code"] = "#CCCCCC"
                if "reason" not in rec:
                    rec["reason"] = "Based on room analysis"
                
                # Ensure paint_products exists and has proper structure
                if "paint_products" not in rec or not isinstance(rec["paint_products"], list):
                    rec["paint_products"] = []
                
                # Validate each paint product has required fields
                for product in rec.get("paint_products", []):
                    if "brand" not in product:
                        product["brand"] = "Paint Brand"
                    if "shade_name" not in product:
                        product["shade_name"] = rec.get("recommended_color", "Color")
                    if "color_family" not in product:
                        product["color_family"] = "Neutral"
                    if "official_link" not in product:
                        # Generate official brand catalogue link based on brand
                        brand = product.get("brand", "").lower().replace(" ", "")
                        if "asian" in brand:
                            product["official_link"] = "https://www.asianpaints.com/paint-colours.html"
                        elif "berger" in brand:
                            product["official_link"] = "https://www.bergerpaintsind.com/colours"
                        elif "dulux" in brand:
                            product["official_link"] = "https://www.dulux.co.in/en/colours"
                        else:
                            product["official_link"] = "https://www.asianpaints.com/paint-colours.html"
                
                # If no paint products provided, create default ones with official brand links
                if not rec.get("paint_products"):
                    default_products = [
                        {
                            "brand": "Asian Paints",
                            "shade_name": rec.get("recommended_color", "Color"),
                            "color_family": "Neutral",
                            "official_link": "https://www.asianpaints.com/paint-colours.html"
                        },
                        {
                            "brand": "Berger Paints",
                            "shade_name": rec.get("recommended_color", "Color"),
                            "color_family": "Neutral",
                            "official_link": "https://www.bergerpaintsind.com/colours"
                        },
                        {
                            "brand": "Dulux",
                            "shade_name": rec.get("recommended_color", "Color"),
                            "color_family": "Neutral",
                            "official_link": "https://www.dulux.co.in/en/colours"
                        }
                    ]
                    rec["paint_products"] = default_products
            
            print(f"Wall painting analysis with {len(analysis.get('wall_recommendations', []))} recommendations with real paint shade matching")
        else:
            # For other sections, ensure multi-category fields
            required_fields = ["wall_colors", "furniture", "plants", "tips"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = []
            print(f"Multi-category analysis with keys: {list(analysis.keys())}")
        
        # Add metadata
        analysis["analysis_section"] = design_section
        analysis["status"] = "success"
        
        print(f"Analysis completed successfully")
        return analysis
    
    except FileNotFoundError as e:
        print(f"ERROR: Image file not found - {str(e)}")
        raise FileNotFoundError(f"Image processing error: {str(e)}")
    
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode failed - {str(e)}")
        raise RuntimeError(f"Failed to parse API response as JSON: {str(e)}")
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"UNEXPECTED ERROR IN ANALYZE_ROOM_IMAGE")
        print(f"{'='*60}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Full Exception: {repr(e)}")
        import traceback
        print(f"\nFull Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise RuntimeError(f"Failed to analyze room: {str(e)}") from e


def analyze_furniture_design(image_path: str, language: str = "english") -> Dict:
    """
    Analyze room image and recommend furniture items with placement and purchase options.
    
    Uses OpenAI Vision API to analyze room layout, space, and suggest appropriate furniture
    with styles, colors, and buying links for Indian e-commerce platforms.
    
    Args:
        image_path (str): Path to the uploaded room image
    
    Returns:
        Dict: Furniture analysis with recommendations and purchase links
        
        {
            "furniture_recommendations": [
                {
                    "furniture_item": "sofa",
                    "placement": "against left wall",
                    "recommended_style": "modern minimalist",
                    "recommended_color": "gray",
                    "reason": "Complements room size and light",
                    "purchase_options": [
                        {
                            "store": "Amazon",
                            "search_link": "https://www.amazon.in/s?k=modern+gray+sofa"
                        },
                        {
                            "store": "Flipkart",
                            "search_link": "https://www.flipkart.com/search?q=modern+gray+sofa"
                        }
                    ]
                }
            ],
            "room_analysis": "Small compact living room with good natural light",
            "space_constraints": "Limited floor space, need vertical storage",
            "overall_style": "contemporary minimalist",
            "analysis_section": "furniture",
            "status": "success"
        }
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is invalid
        RuntimeError: If API call fails
    """
    try:
        # Validate image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Analyzing furniture for room image: {image_path}")
        
        # Compress image for API efficiency
        image_data = compress_image_for_api(image_path)
        
        # Furniture-specific prompt focusing only on furniture recommendations
        lang_instruction = get_language_instruction(language)
        furniture_prompt = f"""{lang_instruction}

Analyze this room image and recommend appropriate furniture items.

FOCUS ONLY ON:
- Furniture recommendations (sofa, table, bed, chair, storage, etc.)
- Placement and positioning
- Style compatibility with room
- Color matching
- Realistic options for Indian buyers

DO NOT INCLUDE:
- Wall color suggestions
- Plant recommendations
- Wall decor ideas

For each furniture item, consider:
- Room size and available space
- Current layout and flow
- Lighting conditions
- Style consistency

Generate realistic, practical furniture suggestions that fit the room.

Return ONLY this JSON format (no markdown or code blocks):
{{
    "furniture_recommendations": [
        {{
            "furniture_item": "sofa/table/bed/chair/etc",
            "placement": "where to place in room",
            "recommended_style": "modern/minimal/wooden/industrial/etc",
            "recommended_color": "best matching furniture color",
            "reason": "why this furniture suits this room",
            "purchase_options": [
                {{
                    "store": "Amazon",
                    "search_link": "https://www.amazon.in/s?k=search+terms"
                }},
                {{
                    "store": "Flipkart",
                    "search_link": "https://www.flipkart.com/search?q=search+terms"
                }}
            ]
        }}
    ],
    "room_analysis": "description of room characteristics",
    "space_constraints": "space and layout notes",
    "overall_style": "detected style of the room"
}}

Provide 3-4 furniture items. Create useful search links for Indian e-commerce."""
        
        # Call Gemini Vision API
        print("Calling Gemini API for furniture analysis...")
        try:
            response_text = call_gemini_api(furniture_prompt, image_data)
            print("Gemini API call successful!")
        except Exception as api_error:
            print(f"\n{'='*60}")
            print(f"FURNITURE ANALYSIS API ERROR")
            print(f"{'='*60}")
            print(f"Error Type: {type(api_error).__name__}")
            print(f"Error Message: {str(api_error)}")
            print(f"Full Exception: {repr(api_error)}")
            print(f"{'='*60}\n")
            raise RuntimeError(format_openai_error(api_error)) from api_error

        try:
            print(f"Raw response received ({len(response_text)} characters)")
            print(f"First 200 chars: {response_text[:200]}")
        except Exception as e:
            print(f"ERROR: Failed to extract response text")
            print(f"Error: {str(e)}")
            raise RuntimeError(f"Failed to extract response text: {str(e)}") from e
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Clean up response - find last closing brace
        response_text = response_text.strip()
        if response_text.startswith("{"):
            last_brace = response_text.rfind("}")
            if last_brace != -1:
                response_text = response_text[:last_brace + 1]
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
            print(f"JSON parsing successful. Keys: {list(analysis.keys())}")
        except json.JSONDecodeError as json_error:
            print(f"\n{'='*60}")
            print(f"FURNITURE ANALYSIS JSON PARSE ERROR")
            print(f"{'='*60}")
            print(f"Error: {str(json_error)}")
            print(f"Line {json_error.lineno}, Column {json_error.colno}")
            print(f"Problem: {json_error.msg}")
            print(f"\nResponse text (first 500 chars):")
            print(response_text[:500])
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to parse API response as JSON: {str(json_error)}") from json_error
        
        # Ensure furniture_recommendations structure exists
        if "furniture_recommendations" not in analysis:
            analysis["furniture_recommendations"] = []
        
        # Validate and fix each furniture recommendation
        for rec in analysis.get("furniture_recommendations", []):
            if "furniture_item" not in rec:
                rec["furniture_item"] = "Furniture"
            if "placement" not in rec:
                rec["placement"] = "Room placement"
            if "recommended_style" not in rec:
                rec["recommended_style"] = "Contemporary"
            if "recommended_color" not in rec:
                rec["recommended_color"] = "Neutral"
            if "reason" not in rec:
                rec["reason"] = "Suitable for room layout"
            
            # Ensure purchase_options exists with proper structure
            if "purchase_options" not in rec or not isinstance(rec["purchase_options"], list):
                rec["purchase_options"] = []
            
            # Validate each purchase option
            for option in rec.get("purchase_options", []):
                if "store" not in option:
                    option["store"] = "E-commerce"
                if "search_link" not in option:
                    # Generate default search link based on store
                    store = option.get("store", "").lower()
                    item = rec.get("furniture_item", "furniture").lower().replace(" ", "+")
                    style = rec.get("recommended_style", "").lower().replace(" ", "+")
                    search_query = f"{item}+{style}"
                    
                    if "amazon" in store:
                        option["search_link"] = f"https://www.amazon.in/s?k={search_query}"
                    elif "flipkart" in store:
                        option["search_link"] = f"https://www.flipkart.com/search?q={search_query}"
                    else:
                        option["search_link"] = f"https://www.amazon.in/s?k={search_query}"
            
            # If no purchase options, create default ones
            if not rec.get("purchase_options"):
                item_search = rec.get("furniture_item", "furniture").lower().replace(" ", "+")
                style_search = rec.get("recommended_style", "").lower().replace(" ", "+")
                search_query = f"{item_search}+{style_search}"
                
                default_options = [
                    {
                        "store": "Amazon",
                        "search_link": f"https://www.amazon.in/s?k={search_query}"
                    },
                    {
                        "store": "Flipkart",
                        "search_link": f"https://www.flipkart.com/search?q={search_query}"
                    }
                ]
                rec["purchase_options"] = default_options
        
        # Ensure metadata fields
        if "room_analysis" not in analysis:
            analysis["room_analysis"] = "Room analyzed"
        if "space_constraints" not in analysis:
            analysis["space_constraints"] = "Space optimized"
        if "overall_style" not in analysis:
            analysis["overall_style"] = "Contemporary"
        
        # Add metadata
        analysis["analysis_section"] = "furniture"
        analysis["status"] = "success"
        
        print(f"Furniture analysis completed successfully with {len(analysis.get('furniture_recommendations', []))} recommendations")
        return analysis
    
    except FileNotFoundError as e:
        print(f"ERROR: Image file not found - {str(e)}")
        raise FileNotFoundError(f"Image processing error: {str(e)}")
    
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode failed - {str(e)}")
        raise RuntimeError(f"Failed to parse API response as JSON: {str(e)}")
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"UNEXPECTED ERROR IN ANALYZE_FURNITURE_DESIGN")
        print(f"{'='*60}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Full Exception: {repr(e)}")
        import traceback
        print(f"\nFull Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise RuntimeError(f"Failed to analyze furniture: {str(e)}") from e


def analyze_plants_decor(image_path: str, language: str = "english") -> Dict:
    """
    Analyze room image and recommend indoor plants and decor items.
    
    Uses OpenAI Vision API to analyze room conditions and suggest appropriate
    plants and decorative items with styles, placements, and buying links for
    Indian e-commerce platforms.
    
    Args:
        image_path (str): Path to the uploaded room image
    
    Returns:
        Dict: Plants and decor analysis with recommendations and purchase links
        
        {
            "plant_decor_recommendations": [
                {
                    "item_name": "snake plant / peace lily / wall art / lamp",
                    "category": "plant or decor",
                    "placement": "where to place",
                    "style": "modern/minimalist/boho etc",
                    "reason": "why it suits this room",
                    "purchase_options": [
                        {
                            "store": "Amazon",
                            "search_link": "https://www.amazon.in/s?k=..."
                        },
                        {
                            "store": "Flipkart",
                            "search_link": "https://www.flipkart.com/search?q=..."
                        }
                    ]
                }
            ],
            "room_analysis": "lighting and space analysis",
            "decor_theme": "suggested decor theme",
            "overall_style": "detected style of the room",
            "analysis_section": "plants-decor",
            "status": "success"
        }
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is invalid
        RuntimeError: If API call fails
    """
    try:
        # Validate image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Analyzing plants & decor for room image: {image_path}")
        
        # Compress image for API efficiency
        image_data = compress_image_for_api(image_path)
        
        # Plants & Decor specific prompt focusing only on plants and decor items
        lang_instruction = get_language_instruction(language)
        plants_decor_prompt = f"""{lang_instruction}

Analyze this room image and recommend appropriate indoor plants and decorative items.

FOCUS ONLY ON:
- Indoor plants (consider light availability, room size)
- Decorative items (wall art, mirrors, lamps, sculptures, throw pillows, rugs, etc.)
- Plant and decor placement and styling
- Style compatibility with room

DO NOT INCLUDE:
- Wall color suggestions
- Furniture recommendations
- Wall painting suggestions

For each plant/decor item, consider:
- Available natural light in the room
- Room size and available space
- Current room style and theme
- Maintenance requirements for plants
- How items complement existing design

Recommend realistic, commonly available items in India. Suggest 4-5 plants and decorative items total.

Return ONLY this JSON format (no markdown or code blocks):
{{
    "plant_decor_recommendations": [
        {{
            "item_name": "snake plant / peace lily / monstera / wall art / lamp / etc",
            "category": "plant or decor",
            "placement": "where to place in room",
            "style": "modern/minimalist/boho/traditional/industrial/etc",
            "reason": "why this item suits this room",
            "purchase_options": [
                {{
                    "store": "Amazon",
                    "search_link": "https://www.amazon.in/s?k=search+terms"
                }},
                {{
                    "store": "Flipkart",
                    "search_link": "https://www.flipkart.com/search?q=search+terms"
                }}
            ]
        }}
    ],
    "room_analysis": "description of lighting and space conditions",
    "decor_theme": "suggested theme for decoration",
    "overall_style": "detected style of the room"
}}

Create realistic search links for Indian e-commerce platforms."""
        
        # Call Gemini Vision API
        print("Calling Gemini API for plants & decor analysis...")
        try:
            response_text = call_gemini_api(plants_decor_prompt, image_data)
            print("Gemini API call successful!")
        except Exception as api_error:
            print(f"\n{'='*60}")
            print(f"PLANTS & DECOR ANALYSIS API ERROR")
            print(f"{'='*60}")
            print(f"Error Type: {type(api_error).__name__}")
            print(f"Error Message: {str(api_error)}")
            print(f"Full Exception: {repr(api_error)}")
            print(f"{'='*60}\n")
            raise RuntimeError(format_openai_error(api_error)) from api_error

        try:
            print(f"Raw response received ({len(response_text)} characters)")
            print(f"First 200 chars: {response_text[:200]}")
        except Exception as e:
            print(f"ERROR: Failed to extract response text")
            print(f"Error: {str(e)}")
            raise RuntimeError(f"Failed to extract response text: {str(e)}") from e
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Clean up response - find last closing brace
        response_text = response_text.strip()
        if response_text.startswith("{"):
            last_brace = response_text.rfind("}")
            if last_brace != -1:
                response_text = response_text[:last_brace + 1]
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
            print(f"JSON parsing successful. Keys: {list(analysis.keys())}")
        except json.JSONDecodeError as json_error:
            print(f"\n{'='*60}")
            print(f"PLANTS & DECOR ANALYSIS JSON PARSE ERROR")
            print(f"{'='*60}")
            print(f"Error: {str(json_error)}")
            print(f"Line {json_error.lineno}, Column {json_error.colno}")
            print(f"Problem: {json_error.msg}")
            print(f"\nResponse text (first 500 chars):")
            print(response_text[:500])
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to parse API response as JSON: {str(json_error)}") from json_error
        
        # Ensure plant_decor_recommendations structure exists
        if "plant_decor_recommendations" not in analysis:
            analysis["plant_decor_recommendations"] = []
        
        # Validate and fix each recommendation
        for rec in analysis.get("plant_decor_recommendations", []):
            if "item_name" not in rec:
                rec["item_name"] = "Plant/Decor Item"
            if "category" not in rec:
                rec["category"] = "plant"
            if "placement" not in rec:
                rec["placement"] = "Room placement"
            if "style" not in rec:
                rec["style"] = "Contemporary"
            if "reason" not in rec:
                rec["reason"] = "Suitable for room"
            
            # Ensure purchase_options exists with proper structure
            if "purchase_options" not in rec or not isinstance(rec["purchase_options"], list):
                rec["purchase_options"] = []
            
            # Validate each purchase option
            for option in rec.get("purchase_options", []):
                if "store" not in option:
                    option["store"] = "E-commerce"
                if "search_link" not in option:
                    # Generate default search link based on store
                    store = option.get("store", "").lower()
                    item = rec.get("item_name", "plant decor").lower().replace(" ", "+")
                    
                    if "amazon" in store:
                        option["search_link"] = f"https://www.amazon.in/s?k={item}"
                    elif "flipkart" in store:
                        option["search_link"] = f"https://www.flipkart.com/search?q={item}"
                    else:
                        option["search_link"] = f"https://www.amazon.in/s?k={item}"
            
            # If no purchase options, create default ones
            if not rec.get("purchase_options"):
                item_search = rec.get("item_name", "plant decor").lower().replace(" ", "+")
                
                default_options = [
                    {
                        "store": "Amazon",
                        "search_link": f"https://www.amazon.in/s?k={item_search}"
                    },
                    {
                        "store": "Flipkart",
                        "search_link": f"https://www.flipkart.com/search?q={item_search}"
                    }
                ]
                rec["purchase_options"] = default_options
        
        # Ensure metadata fields
        if "room_analysis" not in analysis:
            analysis["room_analysis"] = "Room analyzed for plants and decor"
        if "decor_theme" not in analysis:
            analysis["decor_theme"] = "Contemporary"
        if "overall_style" not in analysis:
            analysis["overall_style"] = "Contemporary"
        
        # Add metadata
        analysis["analysis_section"] = "plants-decor"
        analysis["status"] = "success"
        
        print(f"Plants & decor analysis completed successfully with {len(analysis.get('plant_decor_recommendations', []))} recommendations")
        return analysis
    
    except FileNotFoundError as e:
        print(f"ERROR: Image file not found - {str(e)}")
        raise FileNotFoundError(f"Image processing error: {str(e)}")
    
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode failed - {str(e)}")
        raise RuntimeError(f"Failed to parse API response as JSON: {str(e)}")
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"UNEXPECTED ERROR IN ANALYZE_PLANTS_DECOR")
        print(f"{'='*60}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Full Exception: {repr(e)}")
        import traceback
        print(f"\nFull Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise RuntimeError(f"Failed to analyze plants and decor: {str(e)}") from e


def analyze_smart_house(
    family_members: int,
    adults: int,
    children: int,
    senior_citizens: int,
    plot_size: str,
    floors: int,
    budget: str,
    language: str = "english"
) -> Dict:
    """
    Analyze complete home requirements and provide comprehensive design recommendations.
    
    Uses OpenAI to provide Vastu-aligned recommendations considering family structure,
    plot size, number of floors, and budget.
    
    Args:
        family_members (int): Total family members
        adults (int): Number of adults
        children (int): Number of children
        senior_citizens (int): Number of senior citizens
        plot_size (str): Plot size in square feet (e.g., "2000")
        floors (int): Number of floors
        budget (str): Budget in Indian Rupees (e.g., "500000")
    
    Returns:
        Dict: Comprehensive home analysis with recommendations
        
        {
            "vastu_recommendations": [...],
            "room_recommendations": [...],
            "bedroom_count": 3,
            "bathroom_count": 2,
            "balcony_count": 2,
            "kitchen_placement": "North-East",
            "puja_room_placement": "North-East corner",
            "house_style_recommendation": "Modern Vastu",
            "detailed_explanation": "...",
            "analysis_section": "smart-house-analysis",
            "status": "success"
        }
    """
    try:
        print(f"Analyzing smart house with parameters:")
        print(f"  Family Members: {family_members}")
        print(f"  Adults: {adults}, Children: {children}, Seniors: {senior_citizens}")
        print(f"  Plot Size: {plot_size} sq ft")
        print(f"  Floors: {floors}")
        print(f"  Budget: ₹{budget}")
        
        # Create comprehensive analysis prompt
        lang_instruction = get_language_instruction(language)
        smart_house_prompt = f"""{lang_instruction}

Analyze this home configuration and provide comprehensive design recommendations based on Indian Vastu principles and modern home design.

FAMILY COMPOSITION:
- Total Members: {family_members}
- Adults: {adults}
- Children: {children}
- Senior Citizens: {senior_citizens}

PROPERTY DETAILS:
- Plot Size: {plot_size} square feet
- Number of Floors: {floors}
- Budget: ₹{budget}

Based on this information, provide comprehensive recommendations for:
1. Optimal room distribution and types
2. Vastu-compliant placements
3. Space utilization
4. Lifestyle optimization

IMPORTANT: Consider all Vastu principles while being practical:
- Kitchen should be in South-East or East
- Puja room in North-East
- Master bedroom in South-West
- Children's rooms in North-West or North
- Guest room in North-West
- Dining in South-West or West
- Living room in North or East
- Bathrooms in South-East

Return ONLY this exact JSON format (no markdown or code blocks):
{{
    "vastu_recommendations": [
        "Vastu recommendation 1",
        "Vastu recommendation 2",
        "Vastu recommendation 3",
        "Vastu recommendation 4",
        "Vastu recommendation 5"
    ],
    "room_recommendations": [
        {{
            "room_type": "Master Bedroom",
            "placement": "South-West",
            "size": "14x14 ft",
            "reason": "Vastu compliant, spacious for adults"
        }},
        {{
            "room_type": "Children's Room",
            "placement": "North-West",
            "size": "12x12 ft",
            "reason": "Perfect for children, good natural light"
        }}
    ],
    "bedroom_count": 3,
    "bathroom_count": 2,
    "balcony_count": 2,
    "kitchen_placement": "South-East",
    "puja_room_placement": "North-East corner",
    "house_style_recommendation": "Modern Vastu with Contemporary Design",
    "detailed_explanation": "Based on your family structure and plot size, this configuration maximizes space efficiency while adhering to Vastu principles. The layout supports aging parents with accessible spaces and provides safe, separate areas for children. The budget allows for quality finishes with modern amenities..."
}}

CONSTRAINTS:
- Base recommendations on actual plot size and family needs
- Ensure rooms scale proportionally to plot size
- Account for all family members (children, seniors)
- Keep budget in mind for recommendations
- Make recommendations practical and Vastu-aligned"""
        
        # Call Gemini API
        print("Calling Gemini API for smart house analysis...")
        try:
            response_text = call_gemini_api(smart_house_prompt)
            print("Gemini API call successful!")
        except Exception as api_error:
            print(f"\n{'='*60}")
            print(f"SMART HOUSE ANALYSIS API ERROR")
            print(f"{'='*60}")
            print(f"Error Type: {type(api_error).__name__}")
            print(f"Error Message: {str(api_error)}")
            print(f"Full Exception: {repr(api_error)}")
            print(f"{'='*60}\n")
            raise RuntimeError(format_openai_error(api_error)) from api_error

        try:
            print(f"Raw response received ({len(response_text)} characters)")
            print(f"First 200 chars: {response_text[:200]}")
        except Exception as e:
            print(f"ERROR: Failed to extract response text")
            print(f"Error: {str(e)}")
            raise RuntimeError(f"Failed to extract response text: {str(e)}") from e
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Clean up response - find last closing brace
        response_text = response_text.strip()
        if response_text.startswith("{"):
            last_brace = response_text.rfind("}")
            if last_brace != -1:
                response_text = response_text[:last_brace + 1]
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
            print(f"JSON parsing successful. Keys: {list(analysis.keys())}")
        except json.JSONDecodeError as json_error:
            print(f"\n{'='*60}")
            print(f"SMART HOUSE ANALYSIS JSON PARSE ERROR")
            print(f"{'='*60}")
            print(f"Error: {str(json_error)}")
            print(f"Line {json_error.lineno}, Column {json_error.colno}")
            print(f"Problem: {json_error.msg}")
            print(f"\nResponse text (first 500 chars):")
            print(response_text[:500])
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to parse API response as JSON: {str(json_error)}") from json_error
        
        # Validate and ensure all required fields exist
        required_fields = [
            "vastu_recommendations",
            "room_recommendations",
            "bedroom_count",
            "bathroom_count",
            "balcony_count",
            "kitchen_placement",
            "puja_room_placement",
            "house_style_recommendation",
            "detailed_explanation"
        ]
        
        for field in required_fields:
            if field not in analysis:
                print(f"WARNING: Missing field '{field}' in response, adding default")
                if field in ["vastu_recommendations", "room_recommendations"]:
                    analysis[field] = [] if field == "vastu_recommendations" else []
                elif field in ["bedroom_count", "bathroom_count", "balcony_count"]:
                    analysis[field] = 1
                else:
                    analysis[field] = "Not specified"
        
        # Ensure arrays exist and are properly formatted
        if "vastu_recommendations" not in analysis or not isinstance(analysis["vastu_recommendations"], list):
            analysis["vastu_recommendations"] = []
        
        if "room_recommendations" not in analysis or not isinstance(analysis["room_recommendations"], list):
            analysis["room_recommendations"] = []
        
        # Validate room recommendations structure
        for room in analysis.get("room_recommendations", []):
            if "room_type" not in room:
                room["room_type"] = "Room"
            if "placement" not in room:
                room["placement"] = "To be determined"
            if "size" not in room:
                room["size"] = "Standard"
            if "reason" not in room:
                room["reason"] = "Optimized for comfort"
        
        # Add metadata
        analysis["analysis_section"] = "smart-house-analysis"
        analysis["status"] = "success"
        
        print(f"Smart house analysis completed successfully")
        return analysis
    
    except ValueError as e:
        print(f"ERROR: Invalid input parameters - {str(e)}")
        raise ValueError(f"Invalid parameters: {str(e)}")
    
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode failed - {str(e)}")
        raise RuntimeError(f"Failed to parse API response as JSON: {str(e)}")
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"UNEXPECTED ERROR IN ANALYZE_SMART_HOUSE")
        print(f"{'='*60}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Full Exception: {repr(e)}")
        import traceback
        print(f"\nFull Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise RuntimeError(f"Failed to analyze smart house: {str(e)}") from e


def get_api_usage_info() -> Dict:
    """
    Return information about API usage optimization.
    
    Returns:
        Dict: Information about current API usage settings
    """
    return {
        "model": "gpt-4o-mini",
        "max_tokens_per_request": "500-1500 (section-dependent)",
        "image_compression": "Enabled (1024px max, 85% quality)",
        "optimization": "Low-cost analysis with high-quality suggestions",
        "estimated_cost_per_request": "$0.002-0.004 USD (gpt-4o-mini)"
    }
