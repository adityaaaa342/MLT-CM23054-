/**
 * MobileNet Webcam Classifier with Performance Metrics
 * Lab Assignment 3
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Elements ----
    const webcamElement = document.getElementById('webcam');
    const toggleBtn = document.getElementById('toggle-btn');
    const btnText = toggleBtn.querySelector('.btn-text');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsContainer = document.getElementById('results-container');
    const predictionsList = document.getElementById('predictions-list');
    
    // Overlay Elements
    const videoOverlay = document.getElementById('video-overlay');
    const overlayLabel = document.getElementById('overlay-label');

    // Metrics Elements
    const fpsValue = document.getElementById('fps-value');
    const inferenceValue = document.getElementById('inference-value');

    // ---- State Variables ----
    let model = null;
    let isClassifying = false;
    let animationId = null;

    // Performance Variables
    let lastFrameTime = 0;
    let frameCount = 0;
    let lastFpsUpdateTime = 0;
    
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
            videoOverlay.classList.remove('hidden');

            // Reset metrics
            lastFrameTime = performance.now();
            lastFpsUpdateTime = performance.now();
            frameCount = 0;

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
        videoOverlay.classList.add('hidden');
        overlayLabel.textContent = "Detecting...";
        fpsValue.textContent = "--";
        inferenceValue.textContent = "-- ms";
    }

    async function classifyFrame() {
        if (!isClassifying || !model) return;

        // Ensure video has started playing and is ready
        if (webcamElement.readyState >= 2) {
            try {
                // Tracking Inference Time
                const startInference = performance.now();
                
                // Classify the current frame from the webcam wrapper
                const predictions = await model.classify(webcamElement, 3);
                
                const endInference = performance.now();
                const inferenceDuration = endInference - startInference;
                
                // Render UI updates
                renderPredictions(predictions);
                updateMetrics(inferenceDuration);
                
            } catch (error) {
                console.error("Classification error:", error);
            }
        }

        // Loop using requestAnimationFrame for real-time inference
        if (isClassifying) {
            animationId = requestAnimationFrame(classifyFrame);
        }
    }

    function updateMetrics(inferenceDuration) {
        const now = performance.now();
        frameCount++;

        // Calculate FPS every 1 second (1000ms) to smooth out the numbers
        if (now - lastFpsUpdateTime >= 1000) {
            const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
            fpsValue.textContent = fps;
            
            // Reset for next window
            frameCount = 0;
            lastFpsUpdateTime = now;
        }

        // Update Inference time immediately (rounded to nearest decimal)
        inferenceValue.textContent = `${inferenceDuration.toFixed(1)} ms`;
        
        // Change color based on slow inference
        if (inferenceDuration > 100) {
            inferenceValue.style.color = 'var(--danger)'; // Red for slow
        } else if (inferenceDuration > 50) {
            inferenceValue.style.color = 'orange'; // Warning
        } else {
            inferenceValue.style.color = 'var(--success)'; // Green for fast
        }
    }


    function renderPredictions(predictions) {
        // Clear previous list
        predictionsList.innerHTML = '';

        predictions.forEach((prediction, index) => {
            const percentage = (prediction.probability * 100).toFixed(1);
            const label = prediction.className.split(',')[0];
            
            // Overlay the top prediction onto the video feed
            if (index === 0) {
                overlayLabel.textContent = `${label} - ${percentage}%`;
            }

            // Populate the detailed list
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
