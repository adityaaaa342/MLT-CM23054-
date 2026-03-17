document.addEventListener('DOMContentLoaded', async () => {
    // --- UI Elements ---
    const videoElement = document.getElementById('webcam');
    const toggleBtn = document.getElementById('toggle-camera-btn');
    const btnText = document.getElementById('btn-text');
    const loadingOverlay = document.getElementById('loading-overlay');
    const videoOverlay = document.getElementById('video-overlay');
    
    const modelStatusText = document.getElementById('model-status');
    const statusDot = document.getElementById('status-dot');
    
    const predictionsPlaceholder = document.getElementById('predictions-placeholder');
    const predictionsList = document.getElementById('predictions-list');
    
    const fpsCounter = document.getElementById('fps-counter');
    const fpsValue = document.getElementById('fps-value');

    // --- State Variables ---
    let model = null;
    let isWebcamActive = false;
    let stream = null;
    let animationId = null;
    let lastFrameTime = 0;
    let frameCount = 0;
    let lastFpsTime = 0;

    // --- 1. Load the MobileNet Model ---
    async function initModel() {
        try {
            console.log('Loading MobileNet model...');
            model = await mobilenet.load({ version: 2, alpha: 1.0 });
            console.log('Model loaded successfully.');
            
            // Update UI
            loadingOverlay.classList.add('hidden');
            toggleBtn.disabled = false;
            
            modelStatusText.textContent = 'System Ready';
            statusDot.className = 'status-dot ready';
            
        } catch (error) {
            console.error('Failed to load the model', error);
            modelStatusText.textContent = 'Model Load Failed';
            statusDot.className = 'status-dot';
            statusDot.style.backgroundColor = 'var(--danger)';
            loadingOverlay.innerHTML = `
                <i data-lucide="alert-circle" class="overlay-icon" style="color:var(--danger)"></i>
                <p>Failed to load AI model. Please check network.</p>
            `;
            lucide.createIcons();
        }
    }

    // --- 2. Webcam Operations ---
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, 
                audio: false 
            });
            
            videoElement.srcObject = stream;
            
            return new Promise((resolve) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play();
                    resolve();
                };
            });
        } catch (error) {
            console.error('Error accessing webcam', error);
            alert('Could not access the webcam. Please ensure permissions are granted.');
            throw error;
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }
    }

    // --- 3. Prediction Loop ---
    async function predictLoop(timestamp) {
        if (!isWebcamActive || !model) return;

        // Calculate FPS
        if (!lastFpsTime) lastFpsTime = timestamp;
        frameCount++;
        
        if (timestamp - lastFpsTime >= 1000) {
            fpsValue.textContent = frameCount;
            frameCount = 0;
            lastFpsTime = timestamp;
        }

        try {
            // Ensure video has data
            if (videoElement.readyState >= 2) {
                // Throttle inference slightly if needed, but for "real-time", we run as fast as possible
                // MobileNet is usually fast enough for per-frame 
                const predictions = await model.classify(videoElement);
                updatePredictionsUI(predictions);
            }
        } catch (error) {
            console.error('Inference error:', error);
        }

        // Request next frame
        animationId = requestAnimationFrame(predictLoop);
    }

    // --- 4. UI Updates ---
    function updatePredictionsUI(predictions) {
        predictionsList.innerHTML = ''; // Clear prev
        
        // Take top 3
        const topPredictions = predictions.slice(0, 3);
        
        topPredictions.forEach((pred, index) => {
            const probability = Math.round(pred.probability * 100);
            
            // Format class name: separate comma separated terms and capitalize
            const classNameShort = pred.className.split(',')[0].trim();
            
            const li = document.createElement('li');
            li.className = 'prediction-item';
            
            // Slight delay in animation based on rank
            li.style.animationDelay = `${index * 0.1}s`;
            
            li.innerHTML = `
                <div class="prediction-info">
                    <span class="prediction-name">${classNameShort}</span>
                    <span class="prediction-score">${probability}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${probability}%"></div>
                </div>
            `;
            predictionsList.appendChild(li);
        });
    }

    // --- 5. Event Listeners ---
    toggleBtn.addEventListener('click', async () => {
        if (!isWebcamActive) {
            // Turning ON
            try {
                btnText.textContent = 'Starting...';
                toggleBtn.disabled = true;
                
                await startWebcam();
                isWebcamActive = true;
                
                // Update UI state
                toggleBtn.innerHTML = `<i data-lucide="square"></i><span id="btn-text">Stop Camera</span>`;
                toggleBtn.style.backgroundColor = 'var(--danger)';
                toggleBtn.style.boxShadow = '0 4px 14px 0 rgba(239, 68, 68, 0.4)';
                
                videoOverlay.classList.add('hidden');
                
                predictionsPlaceholder.style.display = 'none';
                predictionsList.classList.remove('hidden');
                
                fpsCounter.classList.remove('hidden');
                
                modelStatusText.textContent = 'Detecting Objects...';
                statusDot.className = 'status-dot active';
                
                lucide.createIcons();

                // Start loop
                lastFpsTime = 0;
                frameCount = 0;
                predictLoop(performance.now());
                
            } catch (e) {
                // Revert on fail
                toggleBtn.innerHTML = `<i data-lucide="video"></i><span id="btn-text">Start Camera</span>`;
                lucide.createIcons();
                alert('We had trouble initializing the camera.');
            } finally {
                toggleBtn.disabled = false;
            }
        } else {
            // Turning OFF
            isWebcamActive = false;
            if (animationId) cancelAnimationFrame(animationId);
            
            stopWebcam();
            
            // Revert UI
            toggleBtn.innerHTML = `<i data-lucide="video"></i><span id="btn-text">Start Camera</span>`;
            toggleBtn.style.backgroundColor = 'var(--accent-color)';
            toggleBtn.style.boxShadow = '0 4px 14px 0 var(--accent-glow)';
            
            videoOverlay.classList.remove('hidden');
            
            predictionsPlaceholder.style.display = 'block';
            predictionsList.classList.add('hidden');
            predictionsList.innerHTML = '';
            
            fpsCounter.classList.add('hidden');
            
            modelStatusText.textContent = 'System Ready';
            statusDot.className = 'status-dot ready';
            
            lucide.createIcons();
        }
    });

    // Initialize the app
    initModel();
});
