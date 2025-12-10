import streamlit as st
from ultralytics import YOLO
import cv2
import os
import random
import time
import requests
import json
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="IoT Smart Fridge Simulation",
    page_icon="🧊",
    layout="wide"
)

# Initialize session state
if 'fruits' not in st.session_state:
    st.session_state.fruits = []
if 'closed_products' not in st.session_state:
    st.session_state.closed_products = []
if 'simulation_day' not in st.session_state:
    st.session_state.simulation_day = 0
if 'humidity' not in st.session_state:
    st.session_state.humidity = 60.0
if 'temperature' not in st.session_state:
    st.session_state.temperature = 5.0
if 'ethylene' not in st.session_state:
    st.session_state.ethylene = 0.5
if 'show_expiry_form' not in st.session_state:
    st.session_state.show_expiry_form = False
if 'selected_product_type' not in st.session_state:
    st.session_state.selected_product_type = None
if 'yolo_results' not in st.session_state:
    st.session_state.yolo_results = None

# 1) Load YOLO model
@st.cache_resource
def load_model():
    model_path = "model/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2) Auto-detect fruit type from image
def detect_fruit_type(image_path):
    """Automatically detect fruit type (apple, banana, orange) from image"""
    if model is None:
        return None, {}
    
    try:
        results = model(image_path)[0]
        fruit_counts = {"apple": 0, "banana": 0, "orange": 0}
        all_detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            
            if "_" in label:
                parts = label.split("_", 1)
                detected_fruit = parts[0].lower()
                status = parts[1].lower() if len(parts) > 1 else "unknown"
            else:
                detected_fruit = label.lower()
                status = "unknown"
            
            if detected_fruit in fruit_counts:
                fruit_counts[detected_fruit] += 1
                all_detections.append({
                    "fruit": detected_fruit,
                    "status": status,
                    "confidence": round(conf, 2)
                })
        
        # Find most detected fruit
        detected_fruit = max(fruit_counts, key=fruit_counts.get) if max(fruit_counts.values()) > 0 else None
        return detected_fruit, all_detections
    
    except Exception as e:
        st.error(f"Error detecting fruit: {e}")
        return None, []

# 3) Run YOLO detection on uploaded image - returns detailed info
def analyze_fruit_image(image_path, fruit_type=None):
    """Analyze fruit image - if fruit_type is None, auto-detect it"""
    if model is None:
        return None, 1.0, "unknown", {"fresh_count": 0, "rotten_count": 0, "total": 0, "detections": []}, None
    
    try:
        results = model(image_path)[0]
        detections = []
        fresh_count = 0
        rotten_count = 0
        total_confidence = 0.0
        detected_fruit_type = None
        fruit_counts = {"apple": 0, "banana": 0, "orange": 0}
        all_labels = []  # Debug: store all labels
        
        # Fruit name mapping - handle variations
        fruit_mapping = {
            "apple": ["apple", "apples"],
            "banana": ["banana", "bananas"],
            "orange": ["orange", "oranges"]
        }
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            all_labels.append(label)  # Debug
            
            # Parse label more flexibly - handle cases like "rottenapples", "apple_fresh", etc.
            label_lower = label.lower()
            detected_fruit = None
            status = "unknown"
            
            # First, check if label contains any fruit name (handles "rottenapples", "apples", etc.)
            for std_fruit, variations in fruit_mapping.items():
                # Check if fruit name appears anywhere in label (handles "rottenapples")
                if std_fruit in label_lower:
                    detected_fruit = std_fruit
                    # Check for status indicators
                    if "rot" in label_lower or "rotten" in label_lower:
                        status = "rotten"
                    elif "fresh" in label_lower:
                        status = "fresh"
                    break
                # Also check variations (apples, bananas, oranges)
                for variation in variations:
                    if variation in label_lower:
                        detected_fruit = std_fruit
                        if "rot" in label_lower or "rotten" in label_lower:
                            status = "rotten"
                        elif "fresh" in label_lower:
                            status = "fresh"
                        break
                if detected_fruit:
                    break
            
            # If still not found, try underscore split
            if not detected_fruit and "_" in label:
                parts = label.split("_", 1)
                fruit_part = parts[0].lower()
                status_part = parts[1].lower() if len(parts) > 1 else ""
                
                # Map to standard fruit names - check if fruit name is in the part
                for std_fruit, variations in fruit_mapping.items():
                    if std_fruit in fruit_part or fruit_part in variations or any(v in fruit_part for v in variations):
                        detected_fruit = std_fruit
                        break
                
                if not detected_fruit:
                    # Try reverse - check if fruit_part contains any fruit name
                    for std_fruit in fruit_mapping.keys():
                        if std_fruit in fruit_part:
                            detected_fruit = std_fruit
                            break
                    if not detected_fruit:
                        detected_fruit = fruit_part
                
                if status == "unknown":
                    status = status_part
            elif not detected_fruit:
                # Last resort: check if any fruit name is in the label
                for std_fruit in fruit_mapping.keys():
                    if std_fruit in label_lower:
                        detected_fruit = std_fruit
                        if "rot" in label_lower or "rotten" in label_lower:
                            status = "rotten"
                        break
                if not detected_fruit:
                    detected_fruit = label_lower
            
            # Count all detected fruits (even if not in our list)
            if detected_fruit in fruit_counts:
                fruit_counts[detected_fruit] += 1
            
            # Count fresh/rotten for detected fruits
            if detected_fruit in fruit_counts:
                if "rot" in status or "rotten" in status:
                    rotten_count += 1
                else:
                    fresh_count += 1
                total_confidence += conf
                detections.append({
                    "fruit": detected_fruit,
                    "status": status,
                    "confidence": round(conf, 2),
                    "original_label": label
                })
        
        # Auto-detect fruit type if not specified
        if fruit_type is None:
            # Find the most detected fruit
            if max(fruit_counts.values()) > 0:
                detected_fruit_type = max(fruit_counts, key=fruit_counts.get)
            else:
                # If no standard fruit detected, check all labels
                st.warning(f"⚠️ Model detected: {all_labels[:5]} - but no standard fruit found. Please check model labels.")
                detected_fruit_type = None
        
        total_detected = fresh_count + rotten_count
        
        # If we have detections but no standard fruit type, still process
        if total_detected > 0:
            fresh_ratio = fresh_count / total_detected
            avg_confidence = total_confidence / total_detected if total_detected > 0 else 0.5
            initial_fresh_level = fresh_ratio * avg_confidence
        else:
            initial_fresh_level = 0.7
        
        # Determine initial status
        if total_detected > 0 and rotten_count > fresh_count:
            initial_status = "rotten"
        elif initial_fresh_level > 0.5:
            initial_status = "fresh"
        elif initial_fresh_level > 0.2:
            initial_status = "rotting"
        else:
            initial_status = "rotten"
        
        plotted = results.plot()
        detection_info = {
            "fresh_count": fresh_count,
            "rotten_count": rotten_count,
            "total": total_detected,
            "detections": detections,
            "fruit_counts": fruit_counts,
            "all_labels": all_labels  # Debug info
        }
        return plotted, initial_fresh_level, initial_status, detection_info, detected_fruit_type
    
    except Exception as e:
        st.error(f"Error running YOLO: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0.5, "unknown", {"fresh_count": 0, "rotten_count": 0, "total": 0, "detections": []}, None

# 4) Calculate fresh_level decay
def update_fresh_levels(fruits, humidity, temperature, ethylene):
    """
    Update fresh levels and check for status changes (fresh -> rotten).
    Returns updated fruits and whether any status changed.
    """
    updated_fruits = []
    status_changed = False
    
    for fruit in fruits:
        # Ensure fruit is a dict
        if not isinstance(fruit, dict):
            continue
        old_status = fruit.get("status", "fresh")
        fresh_level = fruit.get("fresh_level", 1.0)
        
        decay_rate = (
            (ethylene / 1.8) * 0.25 +
            (humidity / 100.0) * 0.1 +
            (temperature / 12.0) * 0.15
        )
        
        new_fresh_level = max(0.0, fresh_level - decay_rate)
        
        if new_fresh_level > 0.5:
            status = "fresh"
        elif new_fresh_level > 0.2:
            status = "rotting"
        else:
            status = "rotten"
        
        fruit["fresh_level"] = new_fresh_level
        fruit["status"] = status
        
        # If fruit transitions to rotten and was initially fresh, mark for image replacement
        if old_status != "rotten" and status == "rotten":
            # Only mark if it was initially fresh (not added as rotten)
            if fruit.get("initially_fresh", True):
                fruit["should_use_rotten_image"] = True
        
        # Check if status changed (especially fresh -> rotten)
        if old_status != status:
            status_changed = True
        
        updated_fruits.append(fruit)
    
    return updated_fruits, status_changed

# 5) Get product image path
def get_product_image_path(product_type, status=None):
    if product_type in ["apple", "banana", "orange"]:
        # For fruits, use rotten image if rotten, otherwise use uploaded image
        if status == "rotten":
            rotten_path = f"images/fruits/rotten_{product_type}.png"
            if os.path.exists(rotten_path):
                return rotten_path
        # Return uploaded image path (will be in fruit dict)
        return None
    else:
        # For closed products - handle different names
        product_name_map = {
            "orange-juice": "orange-juice",
            "soft-drink": "soft-drink",
            "can": "can",
            "milk": "milk",
            "eggs": "eggs",
            "yogurt": "yogurt"
        }
        mapped_name = product_name_map.get(product_type, product_type)
        product_path = f"images/products/{mapped_name}.png"
        if os.path.exists(product_path):
            return product_path
        # Try alternative paths
        alt_path = f"images/products/{product_type}.png"
        if os.path.exists(alt_path):
            return alt_path
    return None

# 6) Overlay item - NO RESIZING, USE ORIGINAL SIZE
def overlay_item(background, fruit_path, x, y, target_width, target_height):
    """
    Overlay fruit image on background - USE ORIGINAL SIZE, NO SCALING.
    
    Args:
        background: Background image (BGR format, numpy array)
        fruit_path: Path to fruit image
        x, y: Position coordinates (top-left corner)
        target_width: IGNORED - using original width
        target_height: IGNORED - using original height
    
    Returns:
        Updated background image
    """
    if not fruit_path or not os.path.exists(fruit_path):
        return background
    
    try:
        # Load fruit image with alpha channel
        fruit_img = cv2.imread(fruit_path, cv2.IMREAD_UNCHANGED)
        
        if fruit_img is None:
            return background
        
        # Handle different image formats
        if fruit_img.shape[2] == 4:  # Has alpha channel
            fruit_bgra = fruit_img
        elif fruit_img.shape[2] == 3:  # No alpha, add opaque alpha
            fruit_bgra = cv2.cvtColor(fruit_img, cv2.COLOR_BGR2BGRA)
        else:
            return background
        
        # USE ORIGINAL SIZE - NO RESIZING AT ALL!
        h, w = fruit_bgra.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # x, y are top-left coordinates
        paste_x = x
        paste_y = y
        
        x1 = max(0, paste_x)
        y1 = max(0, paste_y)
        x2 = min(bg_w, paste_x + w)
        y2 = min(bg_h, paste_y + h)
        
        # Calculate source region
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1:
            # Convert background to RGBA if needed
            if background.shape[2] == 3:
                background_rgba = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
            else:
                background_rgba = background.copy()
            
            # Extract regions
            bg_region = background_rgba[y1:y2, x1:x2]
            fruit_region = fruit_bgra[src_y1:src_y2, src_x1:src_x2]
            
            # Alpha blending
            alpha = fruit_region[:, :, 3:4] / 255.0
            alpha_3d = np.repeat(alpha, 3, axis=2)
            
            # Blend
            blended = (bg_region[:, :, :3] * (1 - alpha_3d) + 
                       fruit_region[:, :, :3] * alpha_3d).astype(np.uint8)
            
            # Update background
            background_rgba[y1:y2, x1:x2, :3] = blended
            
            # Convert back to BGR
            return cv2.cvtColor(background_rgba, cv2.COLOR_BGRA2BGR)
        
        return background
        
    except Exception as e:
        st.warning(f"Could not overlay image {fruit_path}: {e}")
        return background

# 7) Composite images with OpenCV (for proper alpha blending)
def composite_images_opencv(background_image_path, fruit_image_paths, fruit_positions_and_sizes):
    """
    Composite fruit images onto background using OpenCV with slot-based sizing.
    
    Args:
        background_image_path: Path to background image (fridge)
        fruit_image_paths: List of paths to fruit/product images
        fruit_positions_and_sizes: List of (x, y, target_width, target_height) tuples
    
    Returns:
        Combined image as numpy array (BGR format for OpenCV)
    """
    # Load background
    background = cv2.imread(background_image_path, cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"Could not load background image: {background_image_path}")
    
    # Composite each fruit image with slot-based sizing
    for fruit_path, (x, y, target_width, target_height) in zip(fruit_image_paths, fruit_positions_and_sizes):
        background = overlay_item(background, fruit_path, x, y, target_width, target_height)
    
    return background

# 7) Check if environment values are dangerous
def check_environment_danger(humidity, temperature, ethylene):
    """Check if environment values are in dangerous ranges - returns color codes"""
    warnings = []
    is_dangerous = False
    
    # Color codes: 0=safe(green), 1=warning(yellow), 2=danger(red)
    colors = {"humidity": 0, "temperature": 0, "ethylene": 0}
    
    # Temperature: dangerous if > 10°C or < 3°C
    if temperature > 10:
        warnings.append(f"⚠️ High temperature: {temperature:.1f}°C")
        is_dangerous = True
        colors["temperature"] = 2  # Red
    elif temperature > 8:
        colors["temperature"] = 1  # Yellow
    elif temperature < 3:
        warnings.append(f"⚠️ Low temperature: {temperature:.1f}°C")
        is_dangerous = True
        colors["temperature"] = 2  # Red
    elif temperature < 4:
        colors["temperature"] = 1  # Yellow
    
    # Humidity: dangerous if > 90% or < 45%
    if humidity > 90:
        warnings.append(f"⚠️ High humidity: {humidity:.1f}%")
        is_dangerous = True
        colors["humidity"] = 2  # Red
    elif humidity > 85:
        colors["humidity"] = 1  # Yellow
    elif humidity < 45:
        warnings.append(f"⚠️ Low humidity: {humidity:.1f}%")
        colors["humidity"] = 1  # Yellow
    
    # Ethylene: dangerous if > 1.5
    if ethylene > 1.5:
        warnings.append(f"⚠️ High ethylene: {ethylene:.2f}")
        is_dangerous = True
        colors["ethylene"] = 2  # Red
    elif ethylene > 1.2:
        colors["ethylene"] = 1  # Yellow
    
    return warnings, is_dangerous, colors

# 8) Create gauge chart for environment values
def create_gauge_chart(value, title, min_val, max_val, color_scheme="green", width=200, height=200):
    """
    Create a gauge chart using Plotly.
    
    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        color_scheme: "green", "yellow", or "red"
        width: Chart width in pixels
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    # Color mapping
    colors = {
        "green": ["#00ff00", "#90EE90"],
        "yellow": ["#FFD700", "#FFE4B5"],
        "red": ["#FF0000", "#FFB6C1"]
    }
    
    color = colors.get(color_scheme, colors["green"])
    
    # Normalize value to 0-100 for gauge
    normalized_value = ((value - min_val) / (max_val - min_val)) * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        number = {'font': {'size': 28}, 'suffix': ''},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color[0], 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, (min_val + max_val) * 0.6], 'color': color[1]},
                {'range': [(min_val + max_val) * 0.6, max_val], 'color': color[0]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.85
            }
        }
    ))
    
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "black"}
    )
    
    return fig

# 9) Auto-detect shelf coordinates from fridge image
def auto_detect_shelves(fridge_image_path):
    """
    Automatically detect shelf coordinates from fridge image using computer vision.
    
    Args:
        fridge_image_path: Path to fridge image
    
    Returns:
        Dictionary of shelf coordinates with bounding boxes
    """
    if not os.path.exists(fridge_image_path):
        return None
    
    # Load image
    img = cv2.imread(fridge_image_path)
    if img is None:
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance shelf edges
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny with adjusted parameters
    edges = cv2.Canny(blurred, 30, 100)
    
    # Combine with adaptive threshold for better shelf detection
    combined = cv2.bitwise_or(edges, adaptive)
    
    # Detect horizontal lines using HoughLinesP with adjusted parameters
    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=img_width//8,  # Longer minimum line length
        maxLineGap=30
    )
    
    if lines is None or len(lines) == 0:
        # Fallback to manual coordinates if detection fails
        return get_shelf_coordinates(img_width, img_height)
    
    # Filter and categorize horizontal lines
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if line is approximately horizontal (angle < 10 degrees)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 or angle > 170:
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            horizontal_lines.append({
                'y': center_y,
                'x1': min(x1, x2),
                'x2': max(x1, x2),
                'length': line_length,
                'center_x': center_x
            })
    
    if len(horizontal_lines) == 0:
        return get_shelf_coordinates(img_width, img_height)
    
    # Sort lines by Y position (top to bottom)
    horizontal_lines.sort(key=lambda l: l['y'])
    
    # Categorize lines into left (long) and right (short) shelves
    # Left shelves are typically longer (wider) and on the left side of image
    avg_length = np.mean([l['length'] for l in horizontal_lines])
    mid_x = img_width // 2
    
    # Left shelves: longer lines on the left side
    left_shelf_lines = [l for l in horizontal_lines if l['center_x'] < mid_x and l['length'] > avg_length * 0.6]
    # Right shelves: shorter lines on the right side
    right_shelf_lines = [l for l in horizontal_lines if l['center_x'] > mid_x or (l['center_x'] < mid_x and l['length'] < avg_length * 0.6)]
    
    # Group nearby lines (shelves might have multiple lines)
    def group_lines(lines, threshold=30):
        groups = []
        for line in lines:
            added = False
            for group in groups:
                if abs(line['y'] - np.mean([l['y'] for l in group])) < threshold:
                    group.append(line)
                    added = True
                    break
            if not added:
                groups.append([line])
        return groups
    
    left_groups = group_lines(left_shelf_lines)
    right_groups = group_lines(right_shelf_lines)
    
    # Create shelf bounding boxes
    shelves = {}
    
    # Left shelves
    for i, group in enumerate(left_groups[:3]):  # Max 3 left shelves
        if len(group) == 0:
            continue
        avg_y = int(np.mean([l['y'] for l in group]))
        min_x = min([l['x1'] for l in group])
        max_x = max([l['x2'] for l in group])
        
        shelf_name = ['left_top', 'left_middle', 'left_bottom'][i] if i < 3 else f'left_{i+1}'
        shelves[shelf_name] = {
            'center_x': (min_x + max_x) // 2,
            'center_y': avg_y,
            'width': max_x - min_x,
            'height': img_height * 0.15,  # Default height
            'max_items': 4,
            'rows': 1
        }
    
    # Right shelves
    for i, group in enumerate(right_groups[:3]):  # Max 3 right shelves
        if len(group) == 0:
            continue
        avg_y = int(np.mean([l['y'] for l in group]))
        min_x = min([l['x1'] for l in group])
        max_x = max([l['x2'] for l in group])
        
        shelf_name = ['right_top', 'right_middle', 'right_bottom'][i] if i < 3 else f'right_{i+1}'
        shelves[shelf_name] = {
            'center_x': (min_x + max_x) // 2,
            'center_y': avg_y,
            'width': max_x - min_x,
            'height': img_height * 0.15,  # Default height
            'max_items': 2,
            'rows': 1
        }
    
    # If detection didn't find enough shelves, use manual fallback
    if len(shelves) < 4:
        return get_shelf_coordinates(img_width, img_height)
    
    return shelves

# 9b) Define shelf coordinate systems using fixed pixel coordinates
def get_shelf_coordinates(img_width, img_height):
    """
    Define shelf coordinates scaled to actual fridge image size (4680x5047).
    Coordinates are proportionally scaled based on actual image dimensions.
    """
    # Fridge has 3 left shelves and 4 right door shelves
    SHELVES = {
        # LEFT SHELVES (main compartment) - 3 shelves
        "left_top": {
            "x": int(img_width * 0.05),
            "y": int(img_height * 0.08),
            "width": int(img_width * 0.48),
            "height": int(img_height * 0.12),
            "center_x": int(img_width * 0.29),
            "center_y": int(img_height * 0.14),
            "max_items": 10,
            "rows": 1
        },
        "left_middle": {
            "x": int(img_width * 0.05),
            "y": int(img_height * 0.32),
            "width": int(img_width * 0.48),
            "height": int(img_height * 0.12),
            "center_x": int(img_width * 0.29),
            "center_y": int(img_height * 0.38),
            "max_items": 10,
            "rows": 1
        },
        "left_bottom": {
            "x": int(img_width * 0.05),
            "y": int(img_height * 0.56),
            "width": int(img_width * 0.48),
            "height": int(img_height * 0.12),
            "center_x": int(img_width * 0.29),
            "center_y": int(img_height * 0.62),
            "max_items": 10,
            "rows": 1
        },
        # RIGHT SHELVES (door) - 4 shelves
        "right_top": {
            "x": int(img_width * 0.60),
            "y": int(img_height * 0.08),
            "width": int(img_width * 0.35),
            "height": int(img_height * 0.10),
            "center_x": int(img_width * 0.775),
            "center_y": int(img_height * 0.13),
            "max_items": 5,
            "rows": 1
        },
        "right_middle_top": {
            "x": int(img_width * 0.60),
            "y": int(img_height * 0.25),
            "width": int(img_width * 0.35),
            "height": int(img_height * 0.10),
            "center_x": int(img_width * 0.775),
            "center_y": int(img_height * 0.30),
            "max_items": 5,
            "rows": 1
        },
        "right_middle_bottom": {
            "x": int(img_width * 0.60),
            "y": int(img_height * 0.42),
            "width": int(img_width * 0.35),
            "height": int(img_height * 0.10),
            "center_x": int(img_width * 0.775),
            "center_y": int(img_height * 0.47),
            "max_items": 5,
            "rows": 1
        },
        "right_bottom": {
            "x": int(img_width * 0.60),
            "y": int(img_height * 0.59),
            "width": int(img_width * 0.35),
            "height": int(img_height * 0.10),
            "center_x": int(img_width * 0.775),
            "center_y": int(img_height * 0.64),
            "max_items": 5,
            "rows": 1
        }
    }
    
    return SHELVES

# 10) Place item on specific shelf - ORIGINAL SIZE, SIMPLE PLACEMENT
def place_item_on_shelf(item_index, shelf_location, shelf_coords, fruit_image_path=None, is_fruit=False):
    """
    Place items side-by-side using ORIGINAL IMAGE SIZE.
    
    Returns:
        (x, y, actual_width, actual_height) - using image's original dimensions
    """
    shelf = shelf_coords.get(shelf_location)
    if not shelf:
        return None
    
    shelf_x = shelf['x']
    shelf_y = shelf['y']
    shelf_height = shelf['height']
    
    # Read actual image size
    if fruit_image_path and os.path.exists(fruit_image_path):
        try:
            img = Image.open(fruit_image_path)
            actual_width, actual_height = img.size
        except:
            actual_width, actual_height = 200, 200
    else:
        actual_width, actual_height = 200, 200
    
    # Calculate position: side-by-side with 30px spacing
    spacing = 30
    x = shelf_x + 30 + (item_index * (actual_width + spacing))
    
    # Vertically center in shelf
    y = int(shelf_y + (shelf_height - actual_height) / 2)
    
    return (x, y, actual_width, actual_height)

# 11) Render fridge with products using shelf coordinate system
def render_fridge_with_products():
    base_dir = Path(__file__).resolve().parent
    fridge_path = base_dir / "images" / "fridge.jpg"
    temp_dir = base_dir / "images" / "temp"
    if not fridge_path.exists():
        st.error("Fridge image not found: images/fridge.jpg")
        return None
    
    fridge_img = Image.open(fridge_path)
    img_width, img_height = fridge_img.size
    
    # Get shelf coordinates (manual system)
    shelf_coords = get_shelf_coordinates(img_width, img_height)
    
    # Define shelf order - FRUITS on LEFT (3 shelves), PRODUCTS on RIGHT (4 shelves)
    fruit_shelf_order = [
        'left_top', 'left_middle', 'left_bottom'
    ]
    product_shelf_order = [
        'right_top', 'right_middle_top', 'right_middle_bottom', 'right_bottom'
    ]
    
    # Collect all items first - IMPORTANT: Process ALL items in session state
    all_items = []
    
    # Add ALL fruits from session state (preserve all existing fruits)
    for fruit in st.session_state.fruits:
        if fruit.get("image_path") and os.path.exists(fruit.get("image_path")):
            all_items.append({
                "type": fruit["type"],
                "image_path": fruit.get("image_path"),
                "status": fruit.get("status", "fresh"),
                "is_fruit": True,
                "should_use_rotten_image": fruit.get("should_use_rotten_image", False),
                "initially_fresh": fruit.get("initially_fresh", True)
            })
    
    # Add ALL closed products from session state (preserve all existing products)
    for product in st.session_state.closed_products:
        product_name = product.get("name")
        product_path = get_product_image_path(product_name)
        
        # Debug: Check if product path exists
        if not product_path:
            st.warning(f"⚠️ Product image not found for: {product_name}")
            # Try to find it anyway
            product_path = f"images/products/{product_name}.png"
        
        if product_path and os.path.exists(product_path):
            all_items.append({
                "type": product_name,
                "image_path": product_path,
                "status": product.get("status", "ok"),
                "is_fruit": False
            })
        else:
            st.warning(f"⚠️ Product image path does not exist: {product_path} for product: {product_name}")
    
    # Track items per shelf - SEPARATE tracking for fruits and products
    fruit_shelf_counts = {shelf: 0 for shelf in fruit_shelf_order}
    product_shelf_counts = {shelf: 0 for shelf in product_shelf_order}
    
    # Prepare images and positions for OpenCV compositing using SLOT system
    fruit_image_paths = []
    fruit_positions = []  # Will contain (x, y, target_width, target_height) tuples
    
    # Process ALL items and assign them to SEPARATE shelves
    for idx, item in enumerate(all_items):
        # Find available shelf for this item - DIFFERENT shelves for fruits vs products
        shelf_location = None
        shelf_item_index = None
        
        if item["is_fruit"]:
            # Fruits go to LEFT shelves only
            for shelf_name in fruit_shelf_order:
                shelf = shelf_coords[shelf_name]
                current_count = fruit_shelf_counts[shelf_name]
                # Fruits: max 2 per left shelf
                effective_max_items = 2
                
                if current_count < effective_max_items:
                    shelf_location = shelf_name
                    shelf_item_index = current_count
                    fruit_shelf_counts[shelf_name] += 1
                    break
        else:
            # Closed products go to RIGHT shelves only
            for shelf_name in product_shelf_order:
                shelf = shelf_coords[shelf_name]
                current_count = product_shelf_counts[shelf_name]
                # Products: use full capacity
                effective_max_items = shelf['max_items']
                
                if current_count < effective_max_items:
                    shelf_location = shelf_name
                    shelf_item_index = current_count
                    product_shelf_counts[shelf_name] += 1
                    break
        
        if shelf_location is None:
            # All shelves full, skip this item
            st.warning(f"⚠️ All shelves are full! Cannot add more items.")
            continue
        
        # Get image to place
        if item["is_fruit"]:
            # Only use rotten image if:
            # 1. Status is rotten AND
            # 2. Fruit was initially fresh (not added as rotten)
            # This means: fresh → rotten transition = use rotten image
            # But: directly added as rotten = keep original image
            if item["status"] == "rotten" and item.get("initially_fresh", True):
                # Fruit was fresh and became rotten → use rotten overlay image
                rotten_img_path = f"images/fruits/rotten_{item['type']}.png"
                if os.path.exists(rotten_img_path):
                    img_path = rotten_img_path
                else:
                    # Fallback to original if rotten image doesn't exist
                    img_path = item.get("image_path")
            else:
                # Use original uploaded image:
                # - Fresh/rotting fruits (always original)
                # - Fruits added directly as rotten (keep original)
                img_path = item.get("image_path")
        else:
            img_path = item.get("image_path")
        
        if img_path and os.path.exists(img_path):
            try:
                # Get position from shelf coordinate system
                # Pass is_fruit flag and actual image path to get proper spacing
                position = place_item_on_shelf(shelf_item_index, shelf_location, shelf_coords, img_path, item["is_fruit"])
                
                if position:
                    x, y, target_width, target_height = position
                    
                    # Store image path and position with size (slot-based)
                    fruit_image_paths.append(img_path)
                    fruit_positions.append((x, y, target_width, target_height))
            except Exception as e:
                st.warning(f"Could not process image {img_path}: {e}")
    
    # Debug: Show how many items we're processing
    if len(fruit_image_paths) == 0 and len(all_items) > 0:
        st.warning(f"⚠️ {len(all_items)} items found but no images could be processed. Check image paths.")
        # Show debug info
        for item in all_items:
            img_path = item.get("image_path")
            if img_path:
                exists = os.path.exists(img_path)
                st.write(f"Item: {item.get('type')}, Path: {img_path}, Exists: {exists}")
    
    # Use OpenCV for proper alpha blending
    if len(fruit_image_paths) > 0:
        try:
            # Ensure temp directory exists
            os.makedirs(temp_dir, exist_ok=True)
            
            # Convert PIL image to file for OpenCV (use PNG to avoid JPEG issues)
            temp_bg = temp_dir / "temp_background.png"
            fridge_img.convert("RGB").save(temp_bg, format="PNG")
            
            # Composite using OpenCV with slot-based sizing
            combined_bgr = composite_images_opencv(str(temp_bg), fruit_image_paths, fruit_positions)
            
            # Convert back to PIL Image
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            fridge_img = Image.fromarray(combined_rgb)
            
            # Clean up temp files (only temp background, not original fruit images)
            try:
                if os.path.exists(temp_bg):
                    os.remove(temp_bg)
            except:
                pass  # Ignore cleanup errors
        except Exception as e:
            st.error(f"❌ OpenCV compositing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            # Fallback to PIL method if OpenCV fails
            for idx, (img_path, pos_data) in enumerate(zip(fruit_image_paths, fruit_positions)):
                try:
                    x, y, target_width, target_height = pos_data
                    # Use PIL to paste with slot-based sizing
                    fruit_img = Image.open(img_path).convert("RGBA")
                    fruit_img = fruit_img.resize((target_width, target_height), Image.LANCZOS)
                    fridge_img.paste(fruit_img, (x, y), fruit_img)
                except Exception as e2:
                    st.warning(f"Could not overlay {img_path}: {e2}")
    
    return fridge_img

# 8) Check closed product expiry (now influenced by environment)
def check_product_expiry(products, current_day, humidity, temperature, ethylene):
    """
    Update closed product status based on expiry date and environment.
    
    Environment effect:
    - High temperature (>8°C), high humidity (>80%), and high ethylene (>1.0)
      accelerate spoilage by reducing effective days_left.
    """
    updated_products = []
    current_date = datetime.now().date() + timedelta(days=current_day)
    
    # Simple environment penalty (each factor decreases remaining days)
    temp_penalty = max(0, temperature - 8) * 0.5       # every +2°C ~ -1 day
    hum_penalty = max(0, humidity - 80) / 10.0         # every +10% ~ -1 day
    eth_penalty = max(0, ethylene - 1.0) * 2.0         # high ethylene accelerates
    env_penalty = temp_penalty + hum_penalty + eth_penalty
    
    for product in products:
        expiry_date = product.get("expiry_date")
        if expiry_date:
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            days_left = (expiry_date - current_date).days
            
            # Apply environment penalty (cannot go above the original days_left to negative infinity)
            effective_days_left = days_left - env_penalty
            
            if effective_days_left < 0:
                product["status"] = "expired"
            elif effective_days_left <= 2:
                product["status"] = "expiring_soon"
            else:
                product["status"] = "ok"
        updated_products.append(product)
    return updated_products

# 9) Send to cloud
# Cloudflare Worker URL
CLOUD_URL = "https://wild-meadow-99fd.tlof1844.workers.dev/sim"

# 7a) Create payload for cloud
def create_payload(event_type="update"):
    """
    Create standardized payload for cloud synchronization.
    
    Args:
        event_type: Type of event (optional, for logging)
    
    Returns:
        Dictionary with day, fruits, closed_products, environment
    """
    payload = {
        "day": st.session_state.simulation_day,
        "fruits": [
            {
                "type": fruit.get("type", "unknown") if isinstance(fruit, dict) else "unknown",
                "fresh_level": round(fruit.get("fresh_level", 0.0), 2) if isinstance(fruit, dict) else 0.0,
                "status": fruit.get("status", "fresh") if isinstance(fruit, dict) else "fresh"
            }
            for fruit in st.session_state.fruits
            if isinstance(fruit, dict)  # Only process dict items
        ],
        "closed_products": [
            {
                "type": product.get("name", "unknown") if isinstance(product, dict) else "unknown",
                "expiry": str(product.get("expiry_date", "")) if isinstance(product, dict) else "",
                "status": product.get("status", "ok") if isinstance(product, dict) else "ok"
            }
            for product in st.session_state.closed_products
            if isinstance(product, dict)  # Only process dict items
        ],
        "environment": {
            "humidity": round(st.session_state.humidity, 1),
            "temperature": round(st.session_state.temperature, 1),
            "ethylene": round(st.session_state.ethylene, 2)
        }
    }
    
    return payload

# 7b) Send payload to Cloudflare Worker
def send_to_cloud(payload):
    """
    Send payload to Cloudflare Worker API.
    Robust error handling, doesn't break app if cloud fails.
    """
    try:
        response = requests.post(CLOUD_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print("✓ Cloud synced")
            return True
        else:
            print(f"Cloud error: Status {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("Cloud error: Timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Cloud error: {e}")
        return False
    except Exception as e:
        print(f"Cloud error: {e}")
        return False

# 7c) Remove item from fridge
def remove_item(item_type, is_fruit=True, item_index=None):
    """
    Remove an item from the fridge.
    
    Args:
        item_type: Type of item to remove (fruit type or product name)
        is_fruit: True if fruit, False if closed product
        item_index: Optional index to remove specific item
    
    Returns:
        True if item was removed, False otherwise
    """
    removed = False
    
    if is_fruit:
        if item_index is not None and 0 <= item_index < len(st.session_state.fruits):
            st.session_state.fruits.pop(item_index)
            removed = True
        else:
            # Remove first matching fruit
            for i, fruit in enumerate(st.session_state.fruits):
                if fruit.get("type") == item_type:
                    st.session_state.fruits.pop(i)
                    removed = True
                    break
    else:
        if item_index is not None and 0 <= item_index < len(st.session_state.closed_products):
            st.session_state.closed_products.pop(item_index)
            removed = True
        else:
            # Remove first matching product
            for i, product in enumerate(st.session_state.closed_products):
                if product.get("name") == item_type:
                    st.session_state.closed_products.pop(i)
                    removed = True
                    break
    
    if removed:
        # Send to cloud - item removed event
        payload = create_payload("item_removed")
        send_to_cloud(payload)
    
    return removed

# 8) Main UI
def main():
    st.title("🧊 IoT Smart Fridge Simulation")
    st.markdown("---")
    
    # Top bar: Simulation day and controls
    col_top1, col_top2, col_top3 = st.columns([2, 1, 1])
    with col_top1:
        current_date = datetime.now() + timedelta(days=st.session_state.simulation_day)
        date_str = current_date.strftime("%Y-%m-%d")
        st.metric("📅 Simulation Day", f"{st.session_state.simulation_day} ({date_str})")
    with col_top2:
        if st.button("➡️ Advance Day (+1)", type="primary"):
            st.session_state.simulation_day += 1
            st.session_state.fruits, status_changed = update_fresh_levels(
                st.session_state.fruits,
                st.session_state.humidity,
                st.session_state.temperature,
                st.session_state.ethylene
            )
            st.session_state.closed_products = check_product_expiry(
                st.session_state.closed_products,
                st.session_state.simulation_day,
                st.session_state.humidity,
                st.session_state.temperature,
                st.session_state.ethylene
            )
            # Send to cloud - day advanced event
            payload = create_payload("day_advanced")
            send_to_cloud(payload)
            
            # If any fruit status changed (especially fresh -> rotten), send again
            if status_changed:
                payload = create_payload("status_changed")
                send_to_cloud(payload)
            st.rerun()
    with col_top3:
        if st.button("🔄 Reset"):
            st.session_state.fruits = []
            st.session_state.closed_products = []
            st.session_state.simulation_day = 0
            st.session_state.show_expiry_form = False
            st.session_state.yolo_results = None
            st.rerun()
    
    st.markdown("---")
    
    # Main layout: Fridge on left, controls on right
    col_fridge, col_controls = st.columns([3, 2])
    
    # LEFT: Fridge Display
    with col_fridge:
        st.subheader("🧊 Smart Fridge")
        
        # Debug: Show session state info
        st.caption(f"Debug: {len(st.session_state.fruits)} fruits, {len(st.session_state.closed_products)} products in session")
        
        # Render fridge with products
        fridge_display = render_fridge_with_products()
        if fridge_display:
            st.image(fridge_display)
        else:
            st.error("❌ Could not render fridge image!")
        
        # Show items in fridge
        total_items = len(st.session_state.fruits) + len(st.session_state.closed_products)
        st.caption(f"Items in fridge: {total_items}")
        
        # Debug: Show all fruits
        if st.session_state.fruits:
            with st.expander("🔍 Debug: Fruits in Session State"):
                for i, fruit in enumerate(st.session_state.fruits):
                    st.write(f"{i+1}. {fruit.get('type')} - Path: {fruit.get('image_path')} - Exists: {os.path.exists(fruit.get('image_path', ''))}")
        
        # Environment controls with GAUGE CHARTS (big, visible, adjustable)
        st.markdown("---")
        st.markdown("**🌡️ Environment Status (Gauge Charts)**")
        
        # Get color schemes based on values
        _, _, env_colors = check_environment_danger(
            st.session_state.humidity,
            st.session_state.temperature,
            st.session_state.ethylene
        )
        
        col_gauge1, col_gauge2, col_gauge3 = st.columns(3)
        
        with col_gauge1:
            hum_color = "green" if env_colors["humidity"] == 0 else ("yellow" if env_colors["humidity"] == 1 else "red")
            fig_hum = create_gauge_chart(
                st.session_state.humidity,
                "Humidity (%)",
                40, 95,
                hum_color,
                width=250,
                height=250
            )
            st.plotly_chart(fig_hum, use_container_width=True)
            humidity_value = st.slider(
                "Adjust Humidity",
                min_value=40.0,
                max_value=95.0,
                value=st.session_state.humidity,
                step=1.0,
                key="humidity_slider_fridge"
            )
            st.session_state.humidity = humidity_value
        
        with col_gauge2:
            temp_color = "green" if env_colors["temperature"] == 0 else ("yellow" if env_colors["temperature"] == 1 else "red")
            fig_temp = create_gauge_chart(
                st.session_state.temperature,
                "Temperature (°C)",
                2, 12,
                temp_color,
                width=250,
                height=250
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            temperature_value = st.slider(
                "Adjust Temperature",
                min_value=2.0,
                max_value=12.0,
                value=st.session_state.temperature,
                step=0.5,
                key="temperature_slider_fridge"
            )
            st.session_state.temperature = temperature_value
        
        with col_gauge3:
            eth_color = "green" if env_colors["ethylene"] == 0 else ("yellow" if env_colors["ethylene"] == 1 else "red")
            fig_eth = create_gauge_chart(
                st.session_state.ethylene,
                "Ethylene Gas",
                0.1, 1.8,
                eth_color,
                width=250,
                height=250
            )
            st.plotly_chart(fig_eth, use_container_width=True)
            ethylene_value = st.slider(
                "Adjust Ethylene",
                min_value=0.1,
                max_value=1.8,
                value=st.session_state.ethylene,
                step=0.1,
                key="ethylene_slider_fridge"
            )
            st.session_state.ethylene = ethylene_value
        
        # Auto-apply when sliders change
        if st.button("🔄 Apply Environment Changes", use_container_width=True):
            st.session_state.fruits, status_changed = update_fresh_levels(
                st.session_state.fruits,
                st.session_state.humidity,
                st.session_state.temperature,
                st.session_state.ethylene
            )
            # Send to cloud - environment changed event
            payload = create_payload("environment_changed")
            send_to_cloud(payload)
            
            # If status changed, send again
            if status_changed:
                payload = create_payload("status_changed")
                send_to_cloud(payload)
            
            st.rerun()
    
    # RIGHT: Controls and Info
    with col_controls:
        st.subheader("➕ Add Products")
        
        # Fruits section - Auto-detect fruit type
        st.markdown("**🍎 Fruits (Auto-detect with YOLO)**")
        st.caption("Upload any fruit image - YOLO will automatically detect the type!")
        
        uploaded_file = st.file_uploader(
            "Upload fruit image (apple, banana, or orange)",
            type=['png', 'jpg', 'jpeg'],
            key="fruit_upload"
        )
        
        if uploaded_file is not None:
            upload_dir = "images/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, f"fruit_{len(st.session_state.fruits)}_{uploaded_file.name}")
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Auto-analyze when file is uploaded
            if st.button("🔍 Analyze & Add Fruit", type="primary"):
                with st.spinner("Analyzing with YOLO (auto-detecting fruit type)..."):
                    plotted_img, fresh_level, status, detection_info, detected_fruit_type = analyze_fruit_image(file_path, fruit_type=None)
                    
                    if detected_fruit_type is None:
                        # Show debug info
                        all_labels = detection_info.get("all_labels", [])
                        if all_labels:
                            st.error(f"❌ Model detected labels: {all_labels[:10]} but couldn't match to apple/banana/orange.")
                            st.info("💡 **Debug Info:** Model is detecting objects, but label names might be different. Check if your model uses different naming (e.g., 'apples' instead of 'apple').")
                        else:
                            st.error("❌ No objects detected in image. Please upload a clear image of apple, banana, or orange.")
                        # Show plotted image anyway
                        if plotted_img is not None:
                            st.image(plotted_img, channels="BGR")
                            st.caption("YOLO detection result (even if fruit type not recognized)")
                    else:
                        # Store YOLO results
                        st.session_state.yolo_results = detection_info
                        st.session_state.yolo_results["detected_type"] = detected_fruit_type
                        
                        # Track if fruit was added as fresh (to know if we should replace with rotten image later)
                        initially_fresh = (status != "rotten")
                        
                        new_fruit = {
                            "type": detected_fruit_type,
                            "image_path": file_path,
                            "fresh_level": fresh_level,
                            "status": status,
                            "added_day": st.session_state.simulation_day,
                            "detection_info": detection_info,
                            "initially_fresh": initially_fresh  # Track if added as fresh
                        }
                        st.session_state.fruits.append(new_fruit)
                        
                        # Send to cloud - fruit added event
                        payload = create_payload("fruit_added")
                        send_to_cloud(payload)
                        
                        # Send to cloud - fruit added event
                        payload = create_payload("fruit_added")
                        send_to_cloud(payload)
                        
                        if plotted_img is not None:
                            st.image(plotted_img, channels="BGR")
                        
                        # Show YOLO results
                        st.success(f"✅ {detected_fruit_type.capitalize()} detected and added!")
                        st.info(f"**YOLO Analysis Results:**\n"
                               f"- **Detected Type:** {detected_fruit_type.upper()}\n"
                               f"- **Total detected:** {detection_info['total']}\n"
                               f"- **Fresh:** {detection_info['fresh_count']}\n"
                               f"- **Rotten:** {detection_info['rotten_count']}\n"
                               f"- **Freshness level:** {fresh_level:.1%}\n"
                               f"- **Status:** {status.upper()}")
                        
                        # Show fruit counts if multiple types detected
                        if "fruit_counts" in detection_info:
                            counts = detection_info["fruit_counts"]
                            if sum(counts.values()) > 0:
                                st.write("**All detected fruits:**")
                                for fruit, count in counts.items():
                                    if count > 0:
                                        st.write(f"- {fruit}: {count}")
                        
                        st.rerun()
        
        # Show last YOLO results if available
        if st.session_state.yolo_results:
            with st.expander("📊 Last YOLO Analysis"):
                info = st.session_state.yolo_results
                if "detected_type" in info:
                    st.write(f"**Detected Type:** {info['detected_type'].upper()}")
                st.write(f"**Total Detected:** {info['total']}")
                st.write(f"**Fresh:** {info['fresh_count']}")
                st.write(f"**Rotten:** {info['rotten_count']}")
                if info.get('detections'):
                    st.write("**Details:**")
                    for det in info['detections']:
                        st.write(f"- {det['fruit']}: {det['status']} (confidence: {det['confidence']})")
                if "fruit_counts" in info:
                    counts = info["fruit_counts"]
                    if sum(counts.values()) > 0:
                        st.write("**All detected fruits:**")
                        for fruit, count in counts.items():
                            if count > 0:
                                st.write(f"- {fruit}: {count}")
        
        st.markdown("---")
        
        # Current Items in Fridge section
        st.markdown("**📋 Current Items in Fridge**")
        
        # Fruits list with remove buttons
        if st.session_state.fruits:
            st.markdown("**🍎 Fruits:**")
            for idx, fruit in enumerate(st.session_state.fruits):
                col_item, col_remove = st.columns([4, 1])
                with col_item:
                    status_emoji = "🟢" if fruit.get("status") == "fresh" else ("🟡" if fruit.get("status") == "rotting" else "🔴")
                    st.write(f"{status_emoji} {fruit.get('type', 'unknown').title()} - {fruit.get('status', 'unknown').upper()} (Fresh: {fruit.get('fresh_level', 0):.1%})")
                with col_remove:
                    if st.button("🗑️ Remove", key=f"remove_fruit_{idx}", use_container_width=True):
                        remove_item(fruit.get("type"), is_fruit=True, item_index=idx)
                        st.rerun()
        else:
            st.caption("No fruits in fridge")
        
        # Closed products list with remove buttons
        if st.session_state.closed_products:
            st.markdown("**📦 Closed Products:**")
            for idx, product in enumerate(st.session_state.closed_products):
                col_item, col_remove = st.columns([4, 1])
                with col_item:
                    status_emoji = "✅" if product.get("status") == "ok" else ("⚠️" if product.get("status") == "expiring_soon" else "❌")
                    expiry_str = str(product.get("expiry_date", ""))
                    st.write(f"{status_emoji} {product.get('name', 'unknown').replace('-', ' ').title()} - Expiry: {expiry_str} ({product.get('status', 'unknown').upper()})")
                with col_remove:
                    if st.button("🗑️ Remove", key=f"remove_product_{idx}", use_container_width=True):
                        remove_item(product.get("name"), is_fruit=False, item_index=idx)
                        st.rerun()
        else:
            st.caption("No closed products in fridge")
        
        st.markdown("---")
        
        # Closed products section
        st.markdown("**📦 Add Closed Products (with expiry date)**")
        
        # Product selection buttons - 2 rows
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🥛 Milk", use_container_width=True):
                st.session_state.show_expiry_form = True
                st.session_state.selected_product_type = "milk"
        with col_btn2:
            if st.button("🥚 Eggs", use_container_width=True):
                st.session_state.show_expiry_form = True
                st.session_state.selected_product_type = "eggs"
        with col_btn3:
            if st.button("🥄 Yogurt", use_container_width=True):
                st.session_state.show_expiry_form = True
                st.session_state.selected_product_type = "yogurt"
        
        col_btn4, col_btn5, col_btn6 = st.columns(3)
        with col_btn4:
            if st.button("🥤 Orange Juice", use_container_width=True):
                st.session_state.show_expiry_form = True
                st.session_state.selected_product_type = "orange-juice"
        with col_btn5:
            if st.button("🥤 Soft Drink", use_container_width=True):
                st.session_state.show_expiry_form = True
                st.session_state.selected_product_type = "soft-drink"
        with col_btn6:
            if st.button("🥫 Can", use_container_width=True):
                st.session_state.show_expiry_form = True
                st.session_state.selected_product_type = "can"
        
        # Expiry date form (shows when product button clicked)
        if st.session_state.show_expiry_form and st.session_state.selected_product_type:
            st.markdown(f"**Add {st.session_state.selected_product_type.replace('-', ' ').title()}**")
            expiry_date = st.date_input(
                "Expiry Date",
                min_value=datetime.now().date(),
                key="expiry_input"
            )
            
            col_add, col_cancel = st.columns(2)
            with col_add:
                if st.button("✅ Add Product", type="primary", use_container_width=True):
                    new_product = {
                        "name": st.session_state.selected_product_type,
                        "expiry_date": expiry_date,
                        "added_day": st.session_state.simulation_day,
                        "status": "ok"
                    }
                    
                    # Check expiry
                    current_date = datetime.now().date() + timedelta(days=st.session_state.simulation_day)
                    days_left = (expiry_date - current_date).days
                    if days_left < 0:
                        new_product["status"] = "expired"
                    elif days_left <= 2:
                        new_product["status"] = "expiring_soon"
                    
                    st.session_state.closed_products.append(new_product)
                    st.session_state.show_expiry_form = False
                    st.session_state.selected_product_type = None
                    
                    # Send to cloud - product added event
                    payload = create_payload("product_added")
                    send_to_cloud(payload)
                    st.success(f"{new_product['name'].replace('-', ' ').title()} added!")
                    st.rerun()
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_expiry_form = False
                    st.session_state.selected_product_type = None
                    st.rerun()
        
        st.markdown("---")
        st.caption("💡 **Tip:** Environment controls are now on the fridge panel (left side) for easier access!")
    
    st.markdown("---")
    
    # Bottom: Status tables
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        st.subheader("🍎 Fruits Status")
        if st.session_state.fruits:
            fruit_data = []
            for fruit in st.session_state.fruits:
                fruit_data.append({
                    "Fruit": fruit["type"].upper(),
                    "Freshness": f"{fruit['fresh_level']:.1%}",
                    "Status": fruit["status"].upper(),
                    "Days": st.session_state.simulation_day - fruit.get("added_day", 0)
                })
            st.dataframe(fruit_data, use_container_width=True, hide_index=True)
        else:
            st.info("No fruits in fridge")
    
    with col_status2:
        st.subheader("📦 Products Status")
        if st.session_state.closed_products:
            product_data = []
            current_date = datetime.now().date() + timedelta(days=st.session_state.simulation_day)
            
            for product in st.session_state.closed_products:
                expiry = product.get("expiry_date")
                if expiry:
                    if isinstance(expiry, str):
                        expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
                    days_left = (expiry - current_date).days
                    status_icon = "🔴" if days_left < 0 else "🟡" if days_left <= 2 else "🟢"
                    product_data.append({
                        "Product": product["name"].replace("-", " ").upper(),
                        "Expiry": str(expiry),
                        "Days Left": days_left,
                        "Status": f"{status_icon} {product['status']}"
                    })
            
            if product_data:
                st.dataframe(product_data, use_container_width=True, hide_index=True)
        else:
            st.info("No closed products in fridge")
    
    # Cloud payload viewer
    with st.expander("☁️ Cloud Payload (Last Event)"):
        if st.session_state.fruits or st.session_state.closed_products:
            payload = {
                "day": st.session_state.simulation_day,
                "fruits": [{
                    "type": f["type"],
                    "fresh_level": f["fresh_level"],
                    "status": f["status"]
                } for f in st.session_state.fruits],
                "closed_products": [{
                    "name": p["name"],
                    "expiry_date": str(p.get("expiry_date", "")),
                    "status": p["status"]
                } for p in st.session_state.closed_products],
                "environment": {
                    "humidity": st.session_state.humidity,
                    "temperature": st.session_state.temperature,
                    "ethylene": st.session_state.ethylene
                }
            }
            st.json(payload)
            st.caption("This payload is sent to Cloudflare Worker on every event")

if __name__ == "__main__":
    main()
