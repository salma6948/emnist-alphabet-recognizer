import gradio as gr
import tensorflow as tf
import cv2
import numpy as np

# Load your model
model = tf.keras.models.load_model("imporoved_model.keras")

def recognize_alpha(image):
    try:
        print(f"Debug - Image type: {type(image)}, Shape: {image.shape if hasattr(image, 'shape') else 'No shape'}")
        
        # Handle empty canvas
        if image is None:
            print("Debug - Image is None")
            return {chr(ord('A') + i): 0.0 for i in range(26)}
        
        # Handle different image formats from Gradio
        if isinstance(image, dict):
            print(f"Debug - Image dict keys: {list(image.keys())}")
            # Check for common keys that might contain the image data
            possible_keys = ['image', 'composite', 'background', 'layers', 'mask']
            image_data = None
            
            for key in possible_keys:
                if key in image and image[key] is not None:
                    image_data = image[key]
                    print(f"Debug - Found image data in key: {key}")
                    break
            
            if image_data is None:
                # Print all values to see what's available
                for key, value in image.items():
                    print(f"Debug - Key '{key}': {type(value)} - {getattr(value, 'shape', 'no shape') if hasattr(value, 'shape') else str(value)[:100]}")
                print("Debug - No valid image data found in dict")
                return {chr(ord('A') + i): 0.0 for i in range(26)}
            
            image = image_data
        
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Check for empty or invalid image
        if image.size == 0 or len(image.shape) == 0:
            print("Debug - Empty image")
            return {chr(ord('A') + i): 0.0 for i in range(26)}
        
        print(f"Debug - Processing image shape: {image.shape}")
        
        # Handle different image shapes
        if len(image.shape) == 4:  # (batch, height, width, channels)
            image = image[0]  # Take first image from batch
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Convert RGBA to RGB first
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            if image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 1:  # Single channel with dimension
                image = image.squeeze()
        
        # Ensure we have a 2D array
        if len(image.shape) != 2:
            print(f"Debug - Wrong image shape after processing: {image.shape}")
            return {chr(ord('A') + i): 0.0 for i in range(26)}
        
        # Check if image is completely empty (all zeros or all same value)
        if np.all(image == image.flat[0]):
            print("Debug - Image is uniform (empty)")
            return {chr(ord('A') + i): 0.0 for i in range(26)}
        
        print(f"Debug - Image range: {image.min()} to {image.max()}")
        
        # Apply Gaussian blur to smooth the strokes (like handwriting)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive threshold for better binarization
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2,2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Find contours and get the main letter area
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get all significant contours (not tiny noise)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            if significant_contours:
                # Get combined bounding box of all significant contours
                all_points = np.vstack([cv2.boundingRect(c)[:2] + (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2], 
                                                                   cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]) 
                                      for c in significant_contours])
                x = np.min(all_points[:, 0])
                y = np.min(all_points[:, 1]) 
                x2 = np.max(all_points[:, 2])
                y2 = np.max(all_points[:, 3])
                w = x2 - x
                h = y2 - y
                
                # Add generous padding (EMNIST has more whitespace)
                padding = max(w, h) // 3
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2*padding)
                h = min(image.shape[0] - y, h + 2*padding)
                
                # Crop to bounding box
                image = image[y:y+h, x:x+w]
                print(f"Debug - Cropped to: {w}x{h}")
            else:
                print("Debug - No significant contours found")
        
        # Make it square by padding with white (important for EMNIST similarity)
        h, w = image.shape
        if h != w:
            size = max(h, w)
            # Create white background (EMNIST style)
            square_image = np.zeros((size, size), dtype=np.uint8)
            # Center the image
            start_h = (size - h) // 2
            start_w = (size - w) // 2
            square_image[start_h:start_h+h, start_w:start_w+w] = image
            image = square_image
        
        # Resize to 28x28 with anti-aliasing
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Apply slight erosion to make strokes thinner (closer to EMNIST style)
        kernel = np.ones((2,2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        
        # Ensure good contrast
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Invert: Paint uses black-on-white, but EMNIST is white-on-black
        image = 255 - image
        
        # Normalize and reshape
        image = image.astype("float32") / 255.0
        image = image.reshape(1, 28, 28, 1)
        
        # Save the processed image for debugging
        debug_image = (image * 255).astype(np.uint8)
        cv2.imwrite('debug_processed.png', debug_image)
        print("Debug - Saved processed image as 'debug_processed.png'")
        
        print(f"Debug - Final image shape: {image.shape}, range: {image.min()} to {image.max()}")
        
        # Predict
        prediction = model.predict(image, verbose=0)[0]
        
        print(f"Debug - Prediction shape: {prediction.shape}, max prob: {prediction.max()}")
        
        # Ensure prediction is valid
        if len(prediction) != 26:
            print(f"Debug - Wrong prediction length: {len(prediction)}")
            return {chr(ord('A') + i): 0.0 for i in range(26)}
        
        # Return probabilities for A-Z
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        result = {}
        for i in range(26):
            # Convert numpy float to Python float and handle any NaN values
            prob = float(prediction[i])
            if np.isnan(prob) or np.isinf(prob):
                prob = 0.0
            result[labels[i]] = prob
        
        # Print top 3 predictions for debugging
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print(f"Debug - Top 3 predictions: {sorted_results[:3]}")
        
        return result
        
    except Exception as e:
        print(f"Error in recognize_alpha: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        # Return default probabilities on error
        return {chr(ord('A') + i): 0.0 for i in range(26)}

# Create the Gradio interface
demo = gr.Interface(
    fn=recognize_alpha,
    inputs=gr.Paint(
        type="numpy"
        # Note: brush_radius and canvas_size not supported in this Gradio version
    ),
    outputs=gr.Label(num_top_classes=5),  # Show top 5 instead of all 26
    live=False,  # Disable real-time prediction
    title="🔤 Draw an Alphabet (A-Z)",
    description="Draw a capital letter. The model will predict which one it is."
)

if __name__ == "__main__":
    demo.launch(share=True)