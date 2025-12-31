import { Upload } from "lucide-react";
import { useRef, useState } from "react";

export function ImageUploader({ apiUrl = "http://localhost:8000/predict" }) {
  const fileInputRef = useRef(null);

  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendToBackend = async (file, previewUrl) => {
    setLoading(true);
    setPreview(previewUrl);
    setResults(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResults(data.predictions);
    } catch (err) {
      alert("Error connecting to API");
      console.log(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (ev) => sendToBackend(file, ev.target.result);
    reader.readAsDataURL(file);
  };

  const reset = () => {
    setPreview(null);
    setResults(null);
    setLoading(false);
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        paddingTop: 10,
      }}
    >
      {!preview && (
        <div
          onClick={() => fileInputRef.current.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            handleFile(e.dataTransfer.files[0]);
          }}
          style={{
            border: "2px dashed #888",
            padding: "60px 30px",
            borderRadius: "12px",
            textAlign: "center",
            width: 450,
            cursor: "pointer",
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleFile(e.target.files[0])}
            style={{ display: "none" }}
          />

          <Upload size={55} color="#666" style={{ marginBottom: 10 }} />

          <p style={{ fontSize: 18, marginBottom: 5 }}>
            Click or drag & drop a dog photo here
          </p>
        </div>
      )}

      {preview && (
        <div style={{ textAlign: "center" }}>
          <img
            src={preview}
            alt="DogImage"
            style={{
              maxWidth: "500px",
              borderRadius: "20px",
              padding: 5,
              marginTop: 20,
            }}
          />

          {loading && (
            <p style={{ marginTop: 18, fontSize: 18 }}>Photo analyzing...</p>
          )}

          {!loading && results && (
            <div style={{ marginTop: 20, textAlign: "center" }}>
              <h3 style={{ fontSize: 18, marginBottom: 15 }}>Most Probable Breeds:</h3>

              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {results.map((r, i) => {
                  const percentage = (r.probability * 100).toFixed(1);
                  const barWidth = Math.max(percentage, 5);
                  return (
                    <li
                      key={i}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        marginBottom: 8,
                        fontSize: 16,
                      }}
                    >
                      <span style={{
                        width: "240px",
                        textAlign: "right",
                        marginRight: 10,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}>
                        {r.breed}
                      </span>

                      <div
                        style={{
                          flex: 1,
                          height: 16,
                          background: "#eee",
                          borderRadius: 8,
                          overflow: "hidden",
                          marginRight: 10,
                        }}
                      >
                        <div
                          style={{
                            width: `${barWidth}%`,
                            height: "100%",
                            background: "#4f46e5",
                            borderRadius: 8,
                            transition: "width 0.5s ease",
                          }}
                        />
                      </div>

                      <span style={{ width: 50, textAlign: "right", fontWeight: "bold" }}>
                        {percentage}%
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}


          {!loading && (
            <button
              style={{
                marginTop: 25,
                padding: "10px 18px",
                fontSize: 15,
                background: "linear-gradient(90deg, #3b82f6, #6545b1ff)",
                color: "#fff",
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
              }}
              onClick={reset}
            >
              Upload another
            </button>
          )}
        </div>
      )}
    </div>
  );
}
