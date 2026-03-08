/**
 * MobileNet Image Classifier
 * Handles UI interactions, model loading, and image classification logic.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Elements ----
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const classifyBtn = document.getElementById('classify-btn');
    const btnText = classifyBtn.querySelector('.btn-text');
    const spinner = classifyBtn.querySelector('.spinner');
    const resultsContainer = document.getElementById('results-container');
    const predictionsList = document.getElementById('predictions-list');

    // ---- State Variables ----
    let model = null;
    let imageLoaded = false;
    let isClassifying = false;

    // ---- Initialization: Load Model ----
    async function initModel() {
        try {
            console.log("Loading MobileNet model...");
            // Load the model from TensorFlow.js
            model = await mobilenet.load();
            console.log("Model loaded successfully");
            
            // Update UI
            btnText.textContent = "Classify Image";
            // Button stays disabled until an image is loaded
            if (imageLoaded) {
                classifyBtn.disabled = false;
            }
        } catch (error) {
            console.error("Failed to load model:", error);
            btnText.textContent = "Error loading model";
        }
    }

    // Start loading the model immediately
    initModel();

    // ---- Event Listeners for Drag and Drop ----
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Remove Image
    removeBtn.addEventListener('click', () => {
        resetUI();
    });

    // Classify Button
    classifyBtn.addEventListener('click', classifyImage);

    // ---- Core Functions ----

    /**
     * Reads the file and displays it in the preview area
     * @param {File} file 
     */
    function handleFile(file) {
        // Basic validation
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imageLoaded = true;
            
            // Toggle visibility
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsContainer.classList.add('hidden');
            
            // Enable button if model is ready
            if (model && !isClassifying) {
                classifyBtn.disabled = false;
            }
        };
        reader.readAsDataURL(file);
    }

    /**
     * Resets the UI back to the initial upload state
     */
    function resetUI() {
        imagePreview.src = '';
        imageLoaded = false;
        fileInput.value = ''; // Clear input
        
        // Hide/Show containers
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        
        // Disable classify button
        classifyBtn.disabled = true;
    }

    /**
     * Runs Inference using MobileNet and displays results
     */
    async function classifyImage() {
        if (!model || !imageLoaded || isClassifying) return;

        // Set UI to classifying state
        isClassifying = true;
        classifyBtn.disabled = true;
        btnText.textContent = "Classifying...";
        spinner.classList.remove('hidden');
        
        // Clear previous results
        resultsContainer.classList.add('hidden');
        predictionsList.innerHTML = '';

        try {
            // MobileNet format: [{ className: "cat", probability: 0.9 }, ...]
            // tfjs handles the conversion to tensor, resizing (224x224), and normalization internally
            const predictions = await model.classify(imagePreview, 3);
            
            // Render results
            renderPredictions(predictions);
            resultsContainer.classList.remove('hidden');
            
        } catch (error) {
            console.error("Error during classification:", error);
            alert("An error occurred during classification. Check console for details.");
        } finally {
            // Restore UI state
            isClassifying = false;
            btnText.textContent = "Classify Image";
            spinner.classList.add('hidden');
            classifyBtn.disabled = false;
        }
    }

    /**
     * Creates HTML for predictions and inserts into list
     * @param {Array} predictions 
     */
    function renderPredictions(predictions) {
        predictions.forEach(prediction => {
            // Format probability to percentage (e.g., 0.9234 -> 92.3)
            const percentage = (prediction.probability * 100).toFixed(1);
            
            // Clean up class name (mobileNet sometimes returns comma separated like "golden retriever, dog")
            // We'll take the first one or just capitalize the string
            const label = prediction.className.split(',')[0];
            
            const li = document.createElement('li');
            li.className = 'prediction-item';
            
            // Create DOM structure matching our CSS
            li.innerHTML = `
                <div class="prediction-header">
                    <span class="prediction-label">${label}</span>
                    <span class="prediction-score">${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%; transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);"></div>
                </div>
            `;
            
            predictionsList.appendChild(li);
            
            // Trigger animation on next frame to ensure CSS transition works
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    const fill = li.querySelector('.progress-fill');
                    if(fill) {
                        fill.style.width = `${percentage}%`;
                    }
                });
            });
        });
    }
});
