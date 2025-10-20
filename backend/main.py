import uvicorn
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for TunedCNN activations
import torchvision.transforms as transforms # Use standard transforms
from PIL import Image
# --- End PyTorch Imports ---
import io
import os
from dotenv import load_dotenv

# --- NEW Pydantic Imports ---
from pydantic import BaseModel
from typing import List, Optional
# --- End Pydantic Imports ---


# --- [UPDATED FOR TunedCNN MODEL from new notebook] ---
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
MODEL_PATH = "alzheimer_cnn_model.pth"
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
# --- [END UPDATED] ---

# --- Define the TunedCNN Architecture (Copied from new notebook) ---
class TunedCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.drop1 = nn.Dropout(p=0.2)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.mish(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.mish(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        leaky = nn.LeakyReLU(0.01)
        x = leaky(x)
        x = self.drop1(x)
        x = self.out(x)
        return x
# --- End Model Definition ---


# --- NEW Pydantic Models for Chat History ---
# This defines the structure of a single chat message
class HistoryItem(BaseModel):
    role: str  # Must be "user" or "model"
    parts: List[str]

# This defines the structure of the data your frontend will send
class ChatPayload(BaseModel):
    message: str
    history: List[HistoryItem]
    diagnosis: Optional[str] = None # The diagnosis is optional
# --- END NEW Pydantic Models ---


# Initialize FastAPI app
app = FastAPI(title="Alzheimer's Detection API (Tuned PyTorch)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://ai-doctor-sid.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the PyTorch Model ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = TunedCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Tuned PyTorch Model loaded successfully from {MODEL_PATH} onto {device}")
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    model = None
# --- End Model Loading ---

# Configure the Gemini API
gemini_model = None
try:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please create a .env file.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini model configured successfully")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    gemini_model = None


# --- Define PyTorch Image Transforms ---
image_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
    transforms.ToTensor(), # This scales image to [0.0, 1.0]
])
# --- End Transforms ---

# Helper function to process the image
def process_image_pytorch(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # This applies Grayscale, Resize, and ToTensor()
    image_tensor = image_transforms(image) # Output is (1, 128, 128) float tensor [0.0, 1.0]
    
    # --- THIS IS THE FIX ---
    # Your model was tested on data from 0-255, not 0-1.
    # We must scale the tensor back up to match your notebook.
    image_tensor = image_tensor * 255.0 
    # --- END FIX ---
    
    # Add the batch dimension: (1, 128, 128) -> (1, 1, 128, 128)
    image_tensor = image_tensor.unsqueeze(0) 
    
    return image_tensor

@app.get("/")
def read_root():
    return {"message": "Welcome to the Alzheimer's Detection API (Tuned PyTorch)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # This endpoint is unchanged
    if not model:
        return {"error": "PyTorch ML Model not loaded"}
    try:
        image_bytes = await file.read()
        input_tensor = process_image_pytorch(image_bytes).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class_index = predicted_idx.item()
        if 0 <= predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]
        else:
            print(f"Warning: Predicted index {predicted_class_index} out of bounds for CLASS_NAMES.")
            return {"error": f"Model produced an invalid prediction index: {predicted_class_index}."}
        confidence_score = confidence.item()
        return {
            "diagnosis": predicted_class_name,
            "confidence": round(confidence_score * 100, 2)
        }
    except Exception as e:
        print(f"Prediction Exception: {e}")
        return {"error": f"Prediction failed: {e}. Check image format and model compatibility."}


@app.post("/get_precautions")
async def get_precautions(data: dict):
    # This endpoint is unchanged
    diagnosis = data.get("diagnosis")
    if not diagnosis: return {"error": "No diagnosis provided"}
    if not gemini_model: return {"error": "Chatbot model not configured"}
    try:
        prompt = f"""
        You are a helpful and empathetic health assistant.
        A user's machine learning model has returned a potential diagnosis of: "{diagnosis}".
        Please provide 3-5 general, non-prescription precautions and lifestyle tips for this condition.
        IMPORTANT:
        1.  DO NOT use any medical jargon.
        2.  DO NOT suggest any specific medications or drugs.
        3.  DO NOT give direct medical advice.
        4.  Start your response with: "Based on the model's result, here are some general lifestyle tips that may be helpful:"
        5.  Keep the response concise and easy to understand.
        """
        response = gemini_model.generate_content(prompt)
        return {"response_text": response.text}
    except Exception as e:
        print(f"Chatbot Exception: {e}")
        return {"error": f"Chatbot failed: {e}"}

# --- UPDATED CHATBOT ENDPOINT ---
@app.post("/chat")
async def chat(data: ChatPayload): # Now uses the Pydantic model
    if not gemini_model:
        return {"error": "Chatbot model not configured"}

    try:
        # 1. Define the System Prompt & Safety Rules
        system_prompt = f"""
        You are a helpful and empathetic health assistant.
        Your safety rules are:
        1. DO NOT use any medical jargon.
        2. DO NOT suggest any specific medications or drugs.
        3. DO NOT give direct medical advice.
        4. Keep the response concise and easy to understand.
        """
        
        # 2. Define the context based on the diagnosis
        if data.diagnosis:
            context = f"The user has uploaded an MRI. The model's diagnosis is '{data.diagnosis}'. Answer their questions based on this context."
        else:
            context = "The user has not uploaded an MRI scan yet. Answer their general questions."

        # 3. Build the full history
        # We start with the system prompt and context, then add the user's history
        
        full_history = [
            {"role": "user", "parts": [system_prompt + "\n" + context]},
            {"role": "model", "parts": ["Okay, I understand. I am ready to help the user based on their diagnosis and my safety rules."]}
        ]
        
        # Add the existing history sent from the frontend
        for item in data.history:
            full_history.append(item.dict()) # Add past messages
            
        # 4. Start a new chat session with the *full* history
        chat_session = gemini_model.start_chat(history=full_history)
        
        # 5. Send the user's *new* message
        # Use send_message_async for better performance in FastAPI
        response = await chat_session.send_message_async(data.message)
        
        return {"response_text": response.text}
        
    except Exception as e:
        print(f"Chatbot Exception: {e}")
        return {"error": f"Chatbot failed: {e}"}
# --- END UPDATED ENDPOINT ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)