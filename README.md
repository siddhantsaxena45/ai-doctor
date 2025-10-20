
# Full-Stack Alzheimer's Detection & AI Assistant

This is a full-stack web application designed to provide a preliminary analysis of brain MRI scans for signs of Alzheimer's disease. It uses a PyTorch-based Convolutional Neural Network (CNN) for diagnosis and integrates a Google Gemini-powered AI assistant for follow-up questions.


## Features

  * **MRI Scan Upload:** Upload JPG or PNG images of brain MRI scans.
  * **AI-Powered Diagnosis:** Get a preliminary diagnosis from a custom-trained PyTorch CNN model. The model classifies scans into four categories:
      * Non Demented
      * Very Mild Demented
      * Mild Demented
      * Moderate Demented
  * **Confidence Score:** Receive a confidence percentage for the model's prediction.
  * **Contextual AI Chat:** Chat with a Google Gemini-powered assistant that is aware of the model's diagnosis.
  * **Automated Precautions:** Automatically receive general lifestyle tips and precautions based on the diagnosis.
  * **Responsive UI:** A clean, modern, and responsive user interface built with React, TailwindCSS, and shadcn/ui.

## How It Works

1.  **Frontend (React):** The user visits the web application and uploads an MRI image file.
2.  **Backend (FastAPI):** The React app sends the image to the `/predict` endpoint on the FastAPI server.
3.  **PyTorch Model:** The server processes the image, converts it to the correct format (128x128 grayscale tensor), and feeds it into the pre-trained `TunedCNN` PyTorch model. link -> https://www.kaggle.com/code/siddhantsaxena45/alziemer
4.  **Diagnosis:** The model outputs a prediction. The backend sends this diagnosis (e.g., "Mild Demented") and the confidence score (e.g., 88.42%) back to the frontend.
5.  **Initial Chat:** The frontend displays the diagnosis. It then *immediately* calls the `/get_precautions` endpoint, sending the new diagnosis.
6.  **Gemini AI (Precautions):** The backend uses the Gemini API to generate a set of general, non-medical precautions and tips relevant to the diagnosis, which starts the chat history.
7.  **Conversational Chat:** The user can now ask follow-up questions. Each new message is sent to the `/chat` endpoint, along with the previous chat history, for a fully conversational experience with the AI.

## Tech Stack

| Area | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | **Python** | Core language |
| | **FastAPI** | High-performance web framework for the API |
| | **PyTorch** | Loading and running the `TunedCNN` model for inference |
| | **google-generativeai** | Google Gemini API for the chatbot |
| | **Uvicorn** | ASGI server to run FastAPI |
| **Frontend** | **React** | Core library for building the user interface |
| | **TailwindCSS** | Utility-first CSS framework for styling |
| | **shadcn/ui** | Re-usable UI components (Card, Button, Input) |
| | **lucide-react** | Icon library |

-----

## Setup and Installation

To run this project locally, you will need to set up both the backend server and the frontend application.

### Prerequisites

  * Python 3.9+
  * Node.js v18+ and `npm` (or `yarn`)
  * Git

### 1\. Clone the Repository

```bash
git clone https://github.com/siddhantsaxena45/ai-doctor.git
cd ai-doctor
```

### 2\. Backend Setup (FastAPI)

1.  Navigate to the backend directory (assuming your `main.py` is in a folder named `backend`):

    ```bash
    cd backend
    ```

2.  Create and activate a Python virtual environment:

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  Create a `requirements.txt` file with the following content:

    ```txt
    fastapi
    uvicorn[standard]
    torch
    torchvision
    Pillow
    python-multipart
    google-generativeai
    python-dotenv
    numpy
    pydantic
    ```

4.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Get Model File:**
    Place your trained PyTorch model file (`alzheimer_cnn_model.pth`) in this `backend` directory.

6.  **Configure Environment Variables:**
    Create a file named `.env` in the `backend` directory and add your Google Gemini API key:

    ```ini
    GEMINI_API_KEY=your_google_api_key_goes_here
    ```

7.  Run the backend server:

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

    The server is now running on `http://localhost:8000`.

### 3\. Frontend Setup (React)

1.  Open a **new terminal** and navigate to the frontend directory (assuming your `App.js` is in a folder named `frontend`):

    ```bash
    cd frontend
    ```

2.  Install the Node.js dependencies:

    ```bash
    npm install
    ```

3.  **Check API URLs:**
    The `App.js` file is configured to make API calls to `http://localhost:8000`. This matches the backend server you just started. If you run the backend on a different port, you **must** update the `fetch` URLs in `App.js`.

4.  Run the frontend development server:

    ```bash
    npm run dev
    ```

5.  Open your browser and navigate to the URL provided (usually `http://localhost:5173`). You should see the application running.

https://ai-doctor-sid.netlify.app/

-----

## ⚠️ Disclaimer

This is a proof-of-concept project for educational and demonstration purposes only. **It is NOT a medical-grade diagnostic tool.**

The diagnoses provided are based on a machine learning model and should **never** be considered a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional with any questions or concerns you may have regarding a medical condition.