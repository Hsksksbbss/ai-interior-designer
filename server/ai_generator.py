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


def call_gemini_api(prompt: str, image_data: Optional[str] = None) -> str:
    """Call the Gemini REST API and return the generated text response."""
    api_key = get_api_key()
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
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
                        },
                        {
                            "brand": "Berger Paints",
                            "shade_name": "Berger Soft Green",
                            "shade_code": "SG-025",
                            "product": "interior wall paint",
                            "official_link": "https://www.bergerpaintsind.com/colours/soft-green"
                        },
                        {
                            "brand": "Dulux",
                            "shade_name": "Dulux Soft Green",
                            "shade_code": "70GY 45/100",
                            "product": "interior wall paint",
                            "official_link": "https://www.dulux.co.in/en/colours/soft-green"
                        }
                    ]
                }
            ],
            "overall_style": "Modern minimalist",
            "lighting_analysis": "Room has good natural light from windows",
            "analysis_section": "wall-painting",
            "status": "success"
        }
        
        For other sections (general format):
        {
            "wall_colors": ["Soft sage green", "Warm beige"],
            "furniture": ["Minimalist sofa", "Glass table"],
            "plants": ["Snake plant", "Pothos"],
            "tips": ["Maximize natural light"],
            "overall_style": "Modern minimalist",
            "analysis_section": "general",
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
            analysis_prompt = f"""Analyze this room image. Suggest wall colors for 2-3 different walls.

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
            analysis_prompt = f"""Analyze this room image and provide interior design suggestions.
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
        
        # Call OpenAI Vision API using Chat Completions
        print("Calling OpenAI Vision API for analysis...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1000,  # Increased for wall-painting with paint products
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            )
            print("OpenAI API call successful!")
        except Exception as api_error:
            # Print detailed error information for debugging
            print(f"\n{'='*60}")
            print(f"OPENAI API ERROR DETAILS")
            print(f"{'='*60}")
            print(f"Error Type: {type(api_error).__name__}")
            print(f"Error Message: {str(api_error)}")
            print(f"Full Exception: {repr(api_error)}")
            
            # Print additional error attributes if available
            if hasattr(api_error, 'response'):
                print(f"\nResponse Status: {getattr(api_error.response, 'status_code', 'N/A')}")
                print(f"Response Body: {getattr(api_error.response, 'text', 'N/A')}")
            
            if hasattr(api_error, 'message'):
                print(f"Error Message Field: {api_error.message}")
            
            print(f"{'='*60}\n")
            
            # Re-raise with a user-friendly message for quota and auth issues
            raise RuntimeError(format_openai_error(api_error)) from api_error
        
        # Extract response text from Chat Completions format
        try:
            response_text = response.choices[0].message.content.strip()
            print(f"Raw response received ({len(response_text)} characters)")
            print(f"First 200 chars: {response_text[:200]}")
        except Exception as e:
            print(f"ERROR: Failed to extract response text")
            print(f"Error: {str(e)}")
            print(f"Response object: {repr(response)}")
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


def analyze_furniture_design(image_path: str) -> Dict:
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
        furniture_prompt = """Analyze this room image and recommend appropriate furniture items.

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
{
    "furniture_recommendations": [
        {
            "furniture_item": "sofa/table/bed/chair/etc",
            "placement": "where to place in room",
            "recommended_style": "modern/minimal/wooden/industrial/etc",
            "recommended_color": "best matching furniture color",
            "reason": "why this furniture suits this room",
            "purchase_options": [
                {
                    "store": "Amazon",
                    "search_link": "https://www.amazon.in/s?k=search+terms"
                },
                {
                    "store": "Flipkart",
                    "search_link": "https://www.flipkart.com/search?q=search+terms"
                }
            ]
        }
    ],
    "room_analysis": "description of room characteristics",
    "space_constraints": "space and layout notes",
    "overall_style": "detected style of the room"
}

Provide 3-4 furniture items. Create useful search links for Indian e-commerce."""
        
        # Call OpenAI Vision API
        print("Calling OpenAI Vision API for furniture analysis...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1500,  # Higher limit for furniture with multiple items and links
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": furniture_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            )
            print("OpenAI API call successful!")
        except Exception as api_error:
            print(f"\n{'='*60}")
            print(f"FURNITURE ANALYSIS API ERROR")
            print(f"{'='*60}")
            print(f"Error Type: {type(api_error).__name__}")
            print(f"Error Message: {str(api_error)}")
            print(f"Full Exception: {repr(api_error)}")
            
            if hasattr(api_error, 'response'):
                print(f"Response Status: {getattr(api_error.response, 'status_code', 'N/A')}")
                print(f"Response Body: {getattr(api_error.response, 'text', 'N/A')}")
            
            print(f"{'='*60}\n")
            raise RuntimeError(f"OpenAI API call failed: {str(api_error)}") from api_error
        
        # Extract response text
        try:
            response_text = response.choices[0].message.content.strip()
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

