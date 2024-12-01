import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

// SVG icons for bins
const BlueBin = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="#0d6efd" xmlns="http://www.w3.org/2000/svg">
    <path d="M9 3V4H4V6H5V19C5 19.5304 5.21071 20.0391 5.58579 20.4142C5.96086 20.7893 6.46957 21 7 21H17C17.5304 21 18.0391 20.7893 18.4142 20.4142C18.7893 20.0391 19 19.5304 19 19V6H20V4H15V3H9ZM7 6H17V19H7V6ZM9 8V17H11V8H9ZM13 8V17H15V8H13Z"/>
  </svg>
);

const GrayBin = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="#6c757d" xmlns="http://www.w3.org/2000/svg">
    <path d="M9 3V4H4V6H5V19C5 19.5304 5.21071 20.0391 5.58579 20.4142C5.96086 20.7893 6.46957 21 7 21H17C17.5304 21 18.0391 20.7893 18.4142 20.4142C18.7893 20.0391 19 19.5304 19 19V6H20V4H15V3H9ZM7 6H17V19H7V6ZM9 8V17H11V8H9ZM13 8V17H15V8H13Z"/>
  </svg>
);

const WasteClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError('Error processing image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-5">
      <div className="row justify-content-center">
        <div className="col-md-8">
          <div className="card">
            <div className="card-header bg-primary text-white">
              <h3 className="text-center mb-0">Waste Classification</h3>
            </div>
            <div className="card-body">
              <form onSubmit={handleSubmit}>
                <div className="mb-3">
                  <label className="form-label">Upload Image</label>
                  <input
                    type="file"
                    className="form-control"
                    accept="image/*"
                    onChange={handleFileSelect}
                  />
                </div>

                {preview && (
                  <div className="mb-3 text-center">
                    <img
                      src={preview}
                      alt="Preview"
                      className="img-fluid rounded shadow"
                      style={{ maxHeight: '300px' }}
                    />
                  </div>
                )}

                <button
                  type="submit"
                  className="btn btn-primary w-100"
                  disabled={!selectedFile || loading}
                >
                  {loading ? (
                    <span>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                      Processing...
                    </span>
                  ) : (
                    'Classify Waste'
                  )}
                </button>
              </form>

              {error && (
                <div className="alert alert-danger mt-3" role="alert">
                  {error}
                </div>
              )}

              {prediction && (
                <div className={`alert mt-3 ${prediction.success ? 
                  (prediction.is_recyclable ? 'alert-success' : 'alert-warning') : 
                  'alert-info'}`}>
                  {prediction.success ? (
                    <div>
                      <h4 className="alert-heading">Classification Result</h4>
                      <p><strong>Item Type:</strong> {prediction.label}</p>
                      <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
                      <p className="d-flex align-items-center">
                        <strong>Bin Type:</strong>&nbsp;
                        {prediction.bin_color.toUpperCase()} bin&nbsp;
                        {prediction.bin_color === 'blue' ? <BlueBin /> : <GrayBin />}
                      </p>
                      <hr />
                      <p className="mb-0">{prediction.message}</p>
                    </div>
                  ) : (
                    <p className="mb-0">{prediction.message}</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WasteClassifier;