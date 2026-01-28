# 🚀 Complete Setup Guide: Adding Your ML Models

This guide explains exactly where to add your model weights and how to integrate your custom ML models into this Explainable AI application. Follow each section step-by-step.

---

## 📁 Project Structure Overview

\`\`\`
project-root/
├── backend/                          # Python FastAPI backend
│   ├── main.py                      # Main API server
│   ├── models/
│   │   ├── model_loader.py          # Where to load your models
│   │   ├── weights/                 # Where to put your model files
│   │   │   ├── unet_model.pth       # U-Net weights (you add this)
│   │   │   └── densenet_model.pth   # DenseNet weights (you add this)
│   │   └── inference.py             # Where to add inference logic
│   └── requirements.txt              # Python dependencies
├── app/
│   ├── api/
│   │   └── detect/
│   │       └── route.ts             # Frontend API endpoint
│   └── page.tsx                     # Main page
└── components/
    ├── demo-section.tsx             # Where images are uploaded
    ├── image-uploader.tsx           # Upload component
    └── results-display.tsx          # Where results are shown
\`\`\`

---

## 🎯 Step 1: Prepare Your Model Weights

### What You Need:
- **U-Net Model**: For candidate detection (mitotic figures)
- **DenseNet Model**: For classification (mitotic vs non-mitotic)
- Both should be saved as `.pth` files (PyTorch format)

### Where to Put Them:
1. Create a folder: `backend/models/weights/`
2. Place your model files there:
   - `backend/models/weights/unet_model.pth`
   - `backend/models/weights/densenet_model.pth`

**Example:**
\`\`\`
backend/models/weights/
├── unet_model.pth          ← Your U-Net weights
└── densenet_model.pth      ← Your DenseNet weights
\`\`\`

---

## 🔧 Step 2: Update Model Loader (`backend/models/model_loader.py`)

This file loads your models when the server starts.

### Current Code (with TODOs):
\`\`\`python
def load_unet_model(model_path: str = None):
    # TODO: Implement actual model loading
    return None
\`\`\`

### What You Need to Change:

**Replace the `load_unet_model()` function:**

\`\`\`python
def load_unet_model(model_path: str = None):
    """Load B-COS U-Net model for candidate detection"""
    try:
        if model_path is None:
            model_path = "backend/models/weights/unet_model.pth"
        
        # Load your U-Net model
        model = YourUNetClass()  # Replace with your actual model class
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        logger.info(f"U-Net model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading U-Net model: {str(e)}")
        raise
\`\`\`

**Replace the `load_densenet_model()` function:**

\`\`\`python
def load_densenet_model(model_path: str = None):
    """Load B-COS DenseNet model for classification"""
    try:
        if model_path is None:
            model_path = "backend/models/weights/densenet_model.pth"
        
        # Load your DenseNet model
        model = YourDenseNetClass()  # Replace with your actual model class
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        logger.info(f"DenseNet model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading DenseNet model: {str(e)}")
        raise
\`\`\`

**Key Changes:**
- Replace `YourUNetClass()` with your actual U-Net class name
- Replace `YourDenseNetClass()` with your actual DenseNet class name
- The paths point to `backend/models/weights/` where you put your `.pth` files

---

## 🧠 Step 3: Update Inference Functions (`backend/main.py`)

These functions run your models on uploaded images.

### Location 1: `detect_candidates()` function

**Current Code:**
\`\`\`python
async def detect_candidates(image: np.ndarray) -> List[Dict[str, Any]]:
    """Stage 1: U-Net candidate detection"""
    # TODO: Implement actual U-Net inference
    candidates = [
        {"x": 100, "y": 100, "width": 50, "height": 50, "confidence": 0.85},
    ]
    return candidates
\`\`\`

**What to Replace:**
\`\`\`python
async def detect_candidates(image: np.ndarray) -> List[Dict[str, Any]]:
    """Stage 1: U-Net candidate detection"""
    try:
        # Run U-Net model on image
        with torch.no_grad():
            # Convert image to tensor
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            
            # Run model
            output = model_manager.unet_model(image_tensor)
            
            # Parse output to get bounding boxes
            candidates = parse_unet_output(output)  # Your parsing function
        
        return candidates
    except Exception as e:
        logger.error(f"Error in candidate detection: {str(e)}")
        return []
\`\`\`

### Location 2: `classify_candidates()` function

**Current Code:**
\`\`\`python
async def classify_candidates(image: np.ndarray, candidates: List[Dict]) -> List[Dict[str, Any]]:
    """Stage 2: DenseNet classification"""
    # TODO: Implement actual DenseNet inference
    classifications = []
    for i, candidate in enumerate(candidates):
        classifications.append({
            "id": i,
            "bbox": candidate,
            "class": "mitotic" if i % 2 == 0 else "non_mitotic",
            "confidence": candidate["confidence"],
        })
    return classifications
\`\`\`

**What to Replace:**
\`\`\`python
async def classify_candidates(image: np.ndarray, candidates: List[Dict]) -> List[Dict[str, Any]]:
    """Stage 2: DenseNet classification"""
    classifications = []
    
    try:
        for i, candidate in enumerate(candidates):
            # Extract region from image
            x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
            region = image[y:y+h, x:x+w]
            
            # Run DenseNet model
            with torch.no_grad():
                region_tensor = torch.from_numpy(region).float().unsqueeze(0)
                output = model_manager.densenet_model(region_tensor)
                
                # Get class and confidence
                class_idx = torch.argmax(output).item()
                confidence = torch.softmax(output, dim=1)[0][class_idx].item()
                class_name = "mitotic" if class_idx == 0 else "non_mitotic"
            
            classifications.append({
                "id": i,
                "bbox": candidate,
                "class": class_name,
                "confidence": confidence,
            })
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
    
    return classifications
\`\`\`

---

## 🎨 Step 4: Update Annotation Function (`backend/main.py`)

This function draws boxes on images with your detection results.

### Location: `generate_annotated_image()` function

**Current Code:**
\`\`\`python
def generate_annotated_image(image: np.ndarray, classifications: List[Dict]) -> np.ndarray:
    """Generate annotated image with color-coded overlays"""
    annotated = image.copy()
    
    for classification in classifications:
        bbox = classification["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        # Color: Red for mitotic, Green for non-mitotic
        color = (0, 0, 255) if classification["class"] == "mitotic" else (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    
    return annotated
\`\`\`

**If You Want to Change Colors:**
- Red (mitotic): Change `(0, 0, 255)` to your RGB value
- Green (non-mitotic): Change `(0, 255, 0)` to your RGB value
- Format is `(Blue, Green, Red)` in OpenCV

**Example - Change to Yellow and Cyan:**
\`\`\`python
color = (0, 255, 255) if classification["class"] == "mitotic" else (255, 255, 0)
\`\`\`

---

## 📝 Step 5: Update Class Names (If Different)

If your model detects different classes (not just "mitotic" and "non_mitotic"), update these locations:

### Location 1: `backend/main.py` - `classify_candidates()` function

**Change this line:**
\`\`\`python
class_name = "mitotic" if class_idx == 0 else "non_mitotic"
\`\`\`

**To match your classes:**
\`\`\`python
class_names = ["mitotic", "non_mitotic", "other_class"]  # Your class names
class_name = class_names[class_idx]
\`\`\`

### Location 2: `backend/main.py` - `generate_annotated_image()` function

**Change this line:**
\`\`\`python
color = (0, 0, 255) if classification["class"] == "mitotic" else (0, 255, 0)
\`\`\`

**To handle all your classes:**
\`\`\`python
color_map = {
    "mitotic": (0, 0, 255),        # Red
    "non_mitotic": (0, 255, 0),    # Green
    "other_class": (255, 0, 0),    # Blue
}
color = color_map.get(classification["class"], (255, 255, 255))
\`\`\`

### Location 3: `backend/main.py` - `classification_summary` in `/predict` endpoint

**Change this:**
\`\`\`python
"classification_summary": {
    "total_detections": len(classifications),
    "mitotic_count": sum(1 for c in classifications if c["class"] == "mitotic"),
    "non_mitotic_count": sum(1 for c in classifications if c["class"] == "non_mitotic")
}
\`\`\`

**To:**
\`\`\`python
"classification_summary": {
    "total_detections": len(classifications),
    "mitotic_count": sum(1 for c in classifications if c["class"] == "mitotic"),
    "non_mitotic_count": sum(1 for c in classifications if c["class"] == "non_mitotic"),
    "other_class_count": sum(1 for c in classifications if c["class"] == "other_class"),
}
\`\`\`

---

## 🖼️ Step 6: Update Frontend Display (If Changing Class Names)

If you changed class names, update the frontend to display them correctly.

### Location: `components/results-display.tsx`

**Find this section:**
\`\`\`tsx
<div className="space-y-2">
  {results.detections.map((detection, idx) => (
    <div key={idx} className="flex justify-between items-center p-2 bg-white/5 rounded">
      <span className="font-mono text-sm">{detection.class}</span>
      <span className="text-primary">{(detection.confidence * 100).toFixed(1)}%</span>
    </div>
  ))}
</div>
\`\`\`

**Update the class display if needed:**
\`\`\`tsx
<span className="font-mono text-sm">
  {detection.class === "mitotic" ? "Mitotic Figure" : "Non-Mitotic"}
</span>
\`\`\`

---

## 🖼️ Step 7: Update Image Upload Handling (If Needed)

If you need to preprocess images before sending to the model:

### Location: `components/image-uploader.tsx`

**Find the upload handler:**
\`\`\`tsx
const handleImageUpload = async (file: File) => {
  const formData = new FormData()
  formData.append("image", file)
  
  const response = await fetch("/api/detect", {
    method: "POST",
    body: formData,
  })
}
\`\`\`

**If you need to resize or preprocess:**
\`\`\`tsx
const handleImageUpload = async (file: File) => {
  // Preprocess image if needed
  const canvas = await resizeImage(file, 512, 512)  // Your resize function
  const blob = await canvasToBlob(canvas)
  
  const formData = new FormData()
  formData.append("image", blob)
  
  const response = await fetch("/api/detect", {
    method: "POST",
    body: formData,
  })
}
\`\`\`

---

## 📋 Checklist: All Places to Update

Use this checklist to make sure you've updated everything:

- [ ] **Step 1**: Created `backend/models/weights/` folder
- [ ] **Step 1**: Added `unet_model.pth` to weights folder
- [ ] **Step 1**: Added `densenet_model.pth` to weights folder
- [ ] **Step 2**: Updated `load_unet_model()` in `backend/models/model_loader.py`
- [ ] **Step 2**: Updated `load_densenet_model()` in `backend/models/model_loader.py`
- [ ] **Step 3**: Updated `detect_candidates()` in `backend/main.py`
- [ ] **Step 3**: Updated `classify_candidates()` in `backend/main.py`
- [ ] **Step 4**: Updated colors in `generate_annotated_image()` if needed
- [ ] **Step 5**: Updated class names in `classify_candidates()` if different
- [ ] **Step 5**: Updated class names in `generate_annotated_image()` if different
- [ ] **Step 5**: Updated `classification_summary` in `/predict` endpoint
- [ ] **Step 6**: Updated class display in `components/results-display.tsx` if needed
- [ ] **Step 7**: Updated image preprocessing in `components/image-uploader.tsx` if needed

---

## 🚀 Testing Your Setup

### 1. Start the Backend Server:
\`\`\`bash
cd backend
pip install -r requirements.txt
python main.py
\`\`\`

You should see:
\`\`\`
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Loading U-Net model for candidate detection...
INFO:     Loading DenseNet model for classification...
INFO:     Models loaded successfully
\`\`\`

### 2. Start the Frontend:
\`\`\`bash
npm run dev
\`\`\`

### 3. Test Upload:
- Go to http://localhost:3000
- Click "Try Demo"
- Upload a test image
- Check that detections appear with your model's results

---

## 🐛 Troubleshooting

### Problem: "Models not loaded" error
**Solution**: Check that your `.pth` files are in `backend/models/weights/` and the paths in `model_loader.py` are correct.

### Problem: "Detection failed" error
**Solution**: Check the backend logs for errors. Make sure your model input/output shapes match your inference code.

### Problem: Wrong class names showing
**Solution**: Check that you updated all three locations (classify_candidates, generate_annotated_image, classification_summary).

### Problem: Wrong colors on boxes
**Solution**: Remember OpenCV uses BGR format, not RGB. So (0, 0, 255) is RED, not BLUE.

---

## 📚 Quick Reference: File Locations

| What to Change | File | Function/Section |
|---|---|---|
| Model weights | `backend/models/weights/` | N/A |
| Load U-Net | `backend/models/model_loader.py` | `load_unet_model()` |
| Load DenseNet | `backend/models/model_loader.py` | `load_densenet_model()` |
| U-Net inference | `backend/main.py` | `detect_candidates()` |
| DenseNet inference | `backend/main.py` | `classify_candidates()` |
| Box colors | `backend/main.py` | `generate_annotated_image()` |
| Class names (backend) | `backend/main.py` | `classify_candidates()` |
| Class names (summary) | `backend/main.py` | `/predict` endpoint |
| Class names (frontend) | `components/results-display.tsx` | Detection list display |
| Image preprocessing | `components/image-uploader.tsx` | `handleImageUpload()` |

---

## 💡 Tips

1. **Start Simple**: First get the basic model loading working, then add preprocessing
2. **Check Logs**: Always check backend logs when something fails
3. **Test Locally**: Test with a small image first before using large ones
4. **Keep Backups**: Save your original files before making changes
5. **Use Print Statements**: Add `print()` or `logger.info()` to debug your code

---

Good luck! You've got this! 🎉
