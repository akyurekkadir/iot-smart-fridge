# 🧊 IoT Smart Fridge Simulation

A comprehensive Streamlit-based smart refrigerator simulator with YOLO image analysis, sensor simulation, real-time decay modeling, and cloud integration.

## 🚀 Live Demo

The interactive Smart Fridge simulation is live here:

👉 **https://iot-smart-fridge-ld4fjyqgt6adrtfab3oaa3.streamlit.app/**

No local setup needed — model, UI, and environment simulation run in the browser.

## 🎥 Demo Video – How the System Works

Watch a walkthrough demonstrating how the IoT Smart Fridge system operates, including image analysis, sensor simulation, real-time decay modeling, and cloud integration:

👉 **https://youtu.be/3ElFaKVzq6I**

🧠 **System Workflow (Covered in the Video)**  
- YOLO detects and classifies food items inside the fridge  
- Temperature and humidity sensors are simulated in real time  
- Food freshness and decay are dynamically modeled  
- All data and predictions are visualized via a Streamlit cloud interface

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#-live-demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Cloud Integration](#cloud-integration)
- [Architecture](#architecture)

## 🎯 Overview

This application simulates an IoT-enabled smart refrigerator that:
- **Detects fruits** using YOLO object detection (apple, banana, orange)
- **Tracks freshness** with real-time decay simulation based on environmental factors
- **Monitors closed products** with expiry date tracking
- **Visualizes items** on a realistic fridge interface with proper shelf placement
- **Synchronizes data** with a Cloudflare Worker API in real-time

## ✨ Features

### Core Functionality
- **YOLO Fruit Detection**: Automatic fruit type detection and freshness classification
- **Visual Fridge Interface**: Interactive fridge with 7 shelves (3 left, 4 right door shelves)
- **Real-time Decay Simulation**: Fruits decay based on humidity, temperature, and ethylene gas
- **Expiry Date Tracking**: Closed products (milk, eggs, yogurt, etc.) with expiry monitoring
- **Environment Controls**: Adjustable humidity (40-95%), temperature (2-12°C), ethylene (0.1-1.8)
- **Day Progression**: Advance simulation day to see decay and expiry changes
- **Item Management**: Add/remove fruits and products with visual feedback

### Cloud Integration
- **Event-based Synchronization**: Every change triggers a cloud update
- **Standardized Payload**: Consistent JSON format for all events
- **Robust Error Handling**: App continues even if cloud fails

## 📁 Project Structure

```
smart_fridge/
├── app.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── models/
│   └── best.pt              # YOLO model weights
└── images/
    ├── fridge.jpg           # Main fridge background image (4680x5047px)
    ├── fruits/
    │   ├── rotten_apple.png    # Rotten apple overlay
    │   ├── rotten_banana.png   # Rotten banana overlay
    │   └── rotten_orange.png   # Rotten orange overlay
    ├── products/
    │   ├── milk.png            # Milk product image
    │   ├── eggs.png            # Eggs product image
    │   ├── yogurt.png          # Yogurt product image
    │   ├── can.png             # Can product image
    │   ├── orange-juice.png    # Orange juice product image
    │   └── soft-drink.png      # Soft drink product image
    ├── uploads/                # User-uploaded fruit images (auto-created)
    └── temp/                   # Temporary files for image compositing (auto-created)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
cd smart_fridge
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit ultralytics opencv-python pillow numpy requests plotly
```

### Step 2: Verify Model File

Ensure `models/best.pt` exists. This is your trained YOLO model for fruit detection.

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

## 🎮 Usage

### Adding Fruits

1. **Upload Image**: Click "Upload fruit image" and select a PNG/JPG file
2. **Auto-Detection**: Click "🔍 Analyze & Add Fruit"
   - YOLO automatically detects fruit type (apple/banana/orange)
   - Determines freshness status (fresh/rotten)
   - Calculates initial freshness level
3. **Visual Placement**: Fruit appears on the left shelves of the fridge

### Adding Closed Products

1. **Select Product**: Click a product button (Milk, Eggs, Yogurt, etc.)
2. **Set Expiry Date**: Choose expiry date from the date picker
3. **Add**: Click "✅ Add Product"
   - Product appears on the right door shelves
   - Status automatically calculated (ok/expiring_soon/expired)

### Managing Items

- **View Items**: See all items in "📋 Current Items in Fridge" section
- **Remove Items**: Click "🗑️ Remove" button next to any item
- **Status Indicators**: 
  - 🟢 Fresh
  - 🟡 Rotting
  - 🔴 Rotten
  - ✅ OK (products)
  - ⚠️ Expiring Soon
  - ❌ Expired

### Environment Control

1. **Adjust Sliders**: Use humidity, temperature, and ethylene sliders
2. **Apply Changes**: Click "🔄 Apply Environment Changes"
   - Fruits decay rates update immediately
   - Cloud receives environment change event

### Day Progression

1. **Advance Day**: Click "➡️ Advance Day (+1)"
   - Simulation day increments
   - Fruits decay based on current environment
   - Products check expiry dates
   - Cloud receives day_advanced event

## 🔬 Technical Details

### Freshness Decay Calculation

The decay rate is calculated using a weighted formula:

```python
decay_rate = (
    (ethylene / 1.8) * 0.25 +      # 25% weight - ethylene gas
    (humidity / 100.0) * 0.1 +     # 10% weight - humidity
    (temperature / 12.0) * 0.15    # 15% weight - temperature
)

new_fresh_level = max(0.0, current_fresh_level - decay_rate)
```

**Status Classification:**
- `fresh`: fresh_level > 0.5 (50%)
- `rotting`: 0.2 < fresh_level ≤ 0.5 (20-50%)
- `rotten`: fresh_level ≤ 0.2 (≤20%)

### Shelf Coordinate System

The fridge image (4680x5047px) uses proportional coordinates:

**Left Shelves (Main Compartment):**
- `left_top`: x=5%, y=8%, width=48%, height=12%
- `left_middle`: x=5%, y=32%, width=48%, height=12%
- `left_bottom`: x=5%, y=56%, width=48%, height=12%

**Right Shelves (Door):**
- `right_top`: x=60%, y=8%, width=35%, height=10%
- `right_middle_top`: x=60%, y=25%, width=35%, height=10%
- `right_middle_bottom`: x=60%, y=42%, width=35%, height=10%
- `right_bottom`: x=60%, y=59%, width=35%, height=10%

### Item Placement Algorithm

1. **Fruits** → Left shelves (3 shelves, max 10 items each)
2. **Closed Products** → Right shelves (4 shelves, max 5 items each)
3. **Side-by-Side Placement**: Items placed horizontally with 30px spacing
4. **Original Size**: Images maintain original dimensions (no scaling unless too large)
5. **Vertical Centering**: Items centered vertically within shelf height

### YOLO Detection Process

1. **Image Upload**: User uploads fruit image
2. **YOLO Inference**: Model detects objects and classifies freshness
3. **Label Matching**: Flexible matching (handles "rottenapples", "apple", "apples", etc.)
4. **Freshness Calculation**: 
   - `fresh_count / total_count = initial_fresh_level`
   - If rotten detected → status = "rotten", fresh_level = 0.0
   - If fresh detected → status = "fresh", fresh_level = calculated ratio

### Expiry Date Calculation

```python
current_date = datetime.now() + timedelta(days=simulation_day)
days_left = (expiry_date - current_date).days

if days_left < 0:
    status = "expired"
elif days_left <= 2:
    status = "expiring_soon"
else:
    status = "ok"
```

## ☁️ Cloud Integration

### Cloudflare Worker Endpoint

```
https://wild-meadow-99fd.tlof1844.workers.dev/sim
```

### Payload Format

Every event sends a standardized JSON payload:

```json
{
  "day": 5,
  "fruits": [
    {
      "type": "apple",
      "fresh_level": 0.75,
      "status": "fresh"
    },
    {
      "type": "banana",
      "fresh_level": 0.12,
      "status": "rotten"
    }
  ],
  "closed_products": [
    {
      "type": "milk",
      "expiry": "2025-12-15",
      "status": "ok"
    }
  ],
  "environment": {
    "humidity": 65.0,
    "temperature": 6.5,
    "ethylene": 0.8
  }
}
```

### Event Triggers

Cloud synchronization occurs automatically on:

1. **Fruit Added**: New fruit uploaded and analyzed
2. **Product Added**: New closed product with expiry date
3. **Item Removed**: User removes any item from fridge
4. **Day Advanced**: Simulation day increments
5. **Environment Changed**: Sliders adjusted and applied
6. **Status Changed**: Fruit transitions from fresh → rotting → rotten

### Error Handling

- **Timeout**: 5-second timeout for requests
- **Network Errors**: Gracefully handled, app continues
- **Status Codes**: Logged but don't break application
- **Pure Side-Effect**: Cloud sync never modifies application state

## 🏗️ Architecture

### State Management

Uses Streamlit's `st.session_state` for persistence:

```python
st.session_state.fruits = []              # List of fruit dicts
st.session_state.closed_products = []     # List of product dicts
st.session_state.simulation_day = 0       # Current simulation day
st.session_state.humidity = 60.0          # Environment humidity
st.session_state.temperature = 5.0        # Environment temperature
st.session_state.ethylene = 0.5           # Environment ethylene gas
```

### Image Rendering Pipeline

1. **Load Background**: Open `fridge.jpg` with PIL
2. **Collect Items**: Gather all fruits and products from session state
3. **Assign Shelves**: Distribute items to appropriate shelves
4. **Calculate Positions**: Side-by-side placement with spacing
5. **Composite Images**: Use OpenCV for alpha blending
6. **Display**: Render final composite image in Streamlit

### Key Functions

- `load_model()`: Loads YOLO model with caching
- `detect_fruit_type()`: Auto-detects fruit type from image
- `analyze_fruit_image()`: Runs YOLO and calculates freshness
- `update_fresh_levels()`: Simulates decay based on environment
- `check_product_expiry()`: Updates product status based on dates
- `get_shelf_coordinates()`: Returns shelf definitions (proportional to image size)
- `place_item_on_shelf()`: Calculates item position (side-by-side)
- `overlay_item()`: Composites item onto background (OpenCV alpha blending)
- `render_fridge_with_products()`: Main rendering function
- `create_payload()`: Generates standardized cloud payload
- `send_to_cloud()`: Sends payload to Cloudflare Worker
- `remove_item()`: Removes item and triggers cloud sync

## 📊 Environment Impact on Decay

### Humidity (40-95%)
- **Low (40-60%)**: Slower decay
- **Medium (60-80%)**: Normal decay
- **High (80-95%)**: Faster decay

### Temperature (2-12°C)
- **Cold (2-5°C)**: Slower decay
- **Normal (5-8°C)**: Normal decay
- **Warm (8-12°C)**: Faster decay

### Ethylene Gas (0.1-1.8)
- **Low (0.1-0.5)**: Minimal impact
- **Medium (0.5-1.0)**: Moderate decay acceleration
- **High (1.0-1.8)**: Significant decay acceleration

## 🔧 Configuration

### Cloud URL

Edit `CLOUD_URL` in `app.py` to change the endpoint:

```python
CLOUD_URL = "https://your-worker.workers.dev/sim"
```

### Shelf Capacities

Modify `max_items` in `get_shelf_coordinates()`:

```python
"left_top": {
    "max_items": 10,  # Change this value
    ...
}
```

### Decay Formula

Adjust weights in `update_fresh_levels()`:

```python
decay_rate = (
    (ethylene / 1.8) * 0.25 +      # Adjust weight
    (humidity / 100.0) * 0.1 +     # Adjust weight
    (temperature / 12.0) * 0.15    # Adjust weight
)
```

## 🐛 Troubleshooting

### YOLO Not Detecting Fruits
- Check model file exists: `models/best.pt`
- Verify image format (PNG/JPG)
- Check label names match model output

### Items Not Appearing on Fridge
- Verify image paths exist
- Check shelf coordinates match fridge image
- Ensure items are in session state

### Cloud Sync Failing
- Check internet connection
- Verify Cloudflare Worker URL is correct
- Check terminal for error messages

## 📝 Notes

- **Model Requirements**: YOLO model must detect "apple", "banana", "orange" (or variations)
- **Image Formats**: Supports PNG, JPG, JPEG
- **File Cleanup**: Uploaded images stored in `images/uploads/` (not auto-deleted)
- **Temp Files**: Created in `images/temp/` during rendering (auto-cleaned)

## 🎓 Academic Use

This project demonstrates:
- **Computer Vision**: YOLO object detection and classification
- **IoT Simulation**: Sensor data simulation and real-time monitoring
- **Cloud Integration**: Event-driven data synchronization
- **UI/UX Design**: Interactive Streamlit dashboard
- **State Management**: Persistent application state
- **Image Processing**: OpenCV alpha blending and compositing

## 📄 License

This project is for educational/academic purposes.

## 👤 Author

IoT Smart Fridge Simulation - Academic Project
