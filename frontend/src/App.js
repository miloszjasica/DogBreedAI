import { useState } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { Header } from "./components/Header";

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState([]);

  const handleResults = (imageUrl, predictionResults) => {
    setUploadedImage(imageUrl);
    setResults(predictionResults);
  };

  const handleReset = () => {
    setUploadedImage(null);
    setResults([]);
  };

  return (
    <div style={{ padding: 30, fontFamily: "Arial" }}>
      <Header />

      <ImageUploader
        isAnalyzing={isAnalyzing}
        setIsAnalyzing={setIsAnalyzing}
        onResults={handleResults}
        onReset={handleReset}
      />

    </div>
  );
}

export default App;
