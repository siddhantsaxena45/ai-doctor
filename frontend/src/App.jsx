import React, { useState, useRef, useEffect } from 'react';
// Import icons from lucide-react
import { Loader2, User, Bot, Image as ImageIcon } from 'lucide-react';

// Import the actual shadcn/ui components
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"; // Adjust path if needed
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";

// Main App Component
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [diagnosis, setDiagnosis] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const [chatHistory, setChatHistory] = useState([]);
  const [userMessage, setUserMessage] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  const fileInputRef = useRef(null);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    // Scroll to the bottom of the chat window when new messages are added
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatHistory]);


  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
      // Reset previous results
      setDiagnosis(null);
      setConfidence(null);
      setError(null);
      setChatHistory([]);
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please select an MRI image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction request failed.');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setDiagnosis(data.diagnosis);
      setConfidence(data.confidence);
      handleGetPrecautions(data.diagnosis); // Fetch precautions after getting diagnosis

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGetPrecautions = async (currentDiagnosis) => {
    setIsChatLoading(true);
    try {
      const response = await fetch('http://localhost:8000/get_precautions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ diagnosis: currentDiagnosis }),
      });

      if (!response.ok) {
        throw new Error('Could not fetch precautions.');
      }

      const data = await response.json();
      if(data.error) {
        throw new Error(data.error);
      }
      
      // Add bot's first message to chat history
      setChatHistory([{ role: 'model', parts: [data.response_text] }]);

    } catch (err) {
      // Add error to chat history
       setChatHistory([{ role: 'model', parts: [`Sorry, I couldn't fetch precautions: ${err.message}`] }]);
    } finally {
       setIsChatLoading(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!userMessage.trim() || isChatLoading) return;

    const newUserMessage = { role: 'user', parts: [userMessage] };
    const updatedHistory = [...chatHistory, newUserMessage];
    
    setChatHistory(updatedHistory);
    setUserMessage('');
    setIsChatLoading(true);

    // Prepare history for API
    const apiHistory = updatedHistory.slice(-11, -1).map(item => ({
        role: item.role,
        parts: item.parts
    }));

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userMessage,
                history: apiHistory,
                diagnosis: diagnosis
            })
        });

        if(!response.ok) {
            throw new Error("The chat service is not responding.");
        }

        const data = await response.json();
        if(data.error) {
            throw new Error(data.error);
        }

        const newBotMessage = { role: 'model', parts: [data.response_text] };
        setChatHistory(prev => [...prev, newBotMessage]);

    } catch(err) {
        const errorMessage = { role: 'model', parts: [`Sorry, I encountered an error: ${err.message}`]};
        setChatHistory(prev => [...prev, errorMessage]);
    } finally {
        setIsChatLoading(false);
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen font-sans">
      <div className="container mx-auto p-4 md:p-8">
        <header className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-800">Alzheimer's Detection Assistant</h1>
          <p className="text-lg text-gray-600 mt-2">Upload an MRI scan to get a preliminary analysis and chat with an AI assistant.</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Left Column: Upload and Diagnosis */}
          <Card>
            <CardHeader>
              <CardTitle>MRI Scan Analysis</CardTitle>
              <CardDescription>Select a JPG or PNG file of a brain MRI scan.</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center text-center cursor-pointer hover:border-blue-500 bg-gray-50 transition-colors"
                onClick={() => fileInputRef.current.click()}
              >
                {preview ? (
                  <img src={preview} alt="MRI Preview" className="max-h-full max-w-full object-contain rounded-md" />
                ) : (
                  <div className="text-gray-500">
                    <ImageIcon className="mx-auto h-12 w-12" />
                    <p className="mt-2">Click to upload an image</p>
                    <p className="text-xs text-gray-400">PNG, JPG up to 10MB</p>
                  </div>
                )}
              </div>
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/png, image/jpeg"
                onChange={handleFileChange}
              />
            </CardContent>
            <CardFooter>
              <Button onClick={handlePredict} disabled={isLoading || !file}>
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {isLoading ? 'Analyzing...' : 'Run Analysis'}
              </Button>
               {file && <p className="text-sm text-gray-500 truncate ml-3">Selected: {file.name}</p>}
            </CardFooter>

            {error && (
                <div className="p-6 border-t">
                    <Alert variant="destructive">
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                </div>
            )}

            {diagnosis && (
                 <div className="p-6 border-t">
                    <Alert>
                        <AlertTitle>Analysis Complete</AlertTitle>
                        <AlertDescription>
                            <p className="text-lg">Diagnosis: <span className="font-semibold text-blue-700">{diagnosis}</span></p>
                            <p className="text-md mt-1">Confidence: <span className="font-semibold">{confidence}%</span></p>
                        </AlertDescription>
                    </Alert>
                </div>
            )}
          </Card>

          {/* Right Column: Chat Assistant */}
          <Card className="flex flex-col h-[70vh]">
            <CardHeader>
              <CardTitle>Chat Assistant</CardTitle>
              <CardDescription>Ask follow-up questions about the diagnosis.</CardDescription>
            </CardHeader>
            <CardContent ref={chatScrollRef} className="flex-grow overflow-y-auto bg-gray-50/50">
              <div className="space-y-4">
                {chatHistory.length === 0 && (
                    <div className="text-center text-gray-500 pt-8">
                        <p>Upload an MRI to begin the chat.</p>
                    </div>
                )}
                {chatHistory.map((msg, index) => (
                  <div key={index} className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                    {msg.role === 'model' && (
                        <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                            <Bot size={16} />
                        </div>
                    )}
                     <div className={`p-3 rounded-lg max-w-md ${msg.role === 'user' ? 'bg-blue-100 text-gray-800' : 'bg-gray-100 text-gray-700'}`}>
                        <p className="text-sm whitespace-pre-wrap">{msg.parts}</p>
                     </div>
                      {msg.role === 'user' && (
                        <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600 font-bold text-sm flex-shrink-0">
                            <User size={16} />
                        </div>
                    )}
                  </div>
                ))}
                 {isChatLoading && chatHistory.length > 0 && (
                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                            <Bot size={16} />
                        </div>
                        <div className="p-3 rounded-lg bg-gray-100 text-gray-700">
                           <div className="flex items-center justify-center space-x-1">
                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                           </div>
                        </div>
                    </div>
                 )}
              </div>
            </CardContent>
            <CardFooter>
              <form onSubmit={handleChatSubmit} className="flex w-full gap-2">
                <Input
                  type="text"
                  placeholder={diagnosis ? "Ask a question..." : "Waiting for analysis..."}
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  disabled={!diagnosis || isChatLoading}
                  autoComplete="off"
                />
                <Button type="submit" disabled={!userMessage.trim() || !diagnosis || isChatLoading}>
                  Send
                </Button>
              </form>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
}
