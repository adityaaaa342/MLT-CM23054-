/**
 * MobileNet Webcam Classifier
 * Handles webcam stream, model loading, and real-time classification.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Elements ----
    const webcamElement = document.getElementById('webcam');
    const toggleBtn = document.getElementById('toggle-btn');
    const btnText = toggleBtn.querySelector('.btn-text');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsContainer = document.getElementById('results-container');
    const predictionsList = document.getElementById('predictions-list');

    // ---- State Variables ----
    let model = null;
    let isClassifying = false;
    let animationId = null;

    // ---- Initialization: Load Model ----
    async function initModel() {
        try {
            console.log("Loading MobileNet model...");
            // Load the model from TensorFlow.js
            model = await mobilenet.load();
            console.log("Model loaded successfully");
            
            // Update UI
            loadingOverlay.classList.add('hidden');
            toggleBtn.disabled = false;
        } catch (error) {
            console.error("Failed to load model:", error);
            loadingOverlay.innerHTML = "<span>Error loading model</span>";
        }
    }

    // Start loading the model immediately
    initModel();

    // Toggle Button Event
    toggleBtn.addEventListener('click', async () => {
        if (!isClassifying) {
            await startWebcam();
        } else {
            stopWebcam();
        }
    });

    // ---- Core Functions ----

    async function startWebcam() {
        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false
            });
            webcamElement.srcObject = stream;
            
            // Wait for video stream to load and set dimensions
            await new Promise((resolve) => {
                webcamElement.onloadedmetadata = () => {
                    // Critical for TFJS: explicitly set the video dimensions
                    webcamElement.width = webcamElement.videoWidth;
                    webcamElement.height = webcamElement.videoHeight;
                    resolve();
                };
            });

            // Make sure the video is playing
            await webcamElement.play();

            // Update state and UI
            isClassifying = true;
            btnText.textContent = "Stop Camera";
            toggleBtn.classList.add('stop-state');
            resultsContainer.classList.remove('hidden');

            // Start classification loop
            classifyFrame();
            
        } catch (error) {
            console.error("Error accessing webcam:", error);
            alert("Could not access the webcam. Please ensure permissions are granted.");
        }
    }

    function stopWebcam() {
        // Stop stream
        if (webcamElement.srcObject) {
            webcamElement.srcObject.getTracks().forEach(track => track.stop());
            webcamElement.srcObject = null;
        }
        
        // Stop inference loop
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // Update state and UI
        isClassifying = false;
        btnText.textContent = "Start Camera & Classify";
        toggleBtn.classList.remove('stop-state');
        resultsContainer.classList.add('hidden');
    }

    async function classifyFrame() {
        if (!isClassifying || !model) return;

        // Ensure video has started playing and is ready
        if (webcamElement.readyState >= 2) {
            try {
                // Classify the current frame from the webcam wrapper
                const predictions = await model.classify(webcamElement, 3);
                renderPredictions(predictions);
            } catch (error) {
                console.error("Classification error:", error);
            }
        }

        // Loop using requestAnimationFrame for real-time inference
        if (isClassifying) {
            // we use requestAnimationFrame to match the screen refresh rate
            animationId = requestAnimationFrame(classifyFrame);
        }
    }

    function renderPredictions(predictions) {
        // Clear previous
        predictionsList.innerHTML = '';

        predictions.forEach((prediction, index) => {
            const percentage = (prediction.probability * 100).toFixed(1);
            const label = prediction.className.split(',')[0];
            
            const li = document.createElement('li');
            li.className = 'prediction-item';
            
            li.innerHTML = `
                <div class="prediction-header">
                    <span class="prediction-label">${label}</span>
                    <span class="prediction-score">${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill ${index === 0 ? 'top-prediction' : ''}" style="width: ${percentage}%;"></div>
                </div>
            `;
            
            predictionsList.appendChild(li);
        });
    }
});
