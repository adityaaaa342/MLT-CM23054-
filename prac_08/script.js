// script.js

// Labels for our custom classes
const classes = ['Apple', 'Banana', 'Orange'];

// DOM Elements
const webcamElement = document.getElementById('webcam');
const resultElement = document.getElementById('predictionResult');
const loadingOverlay = document.getElementById('loadingOverlay');
const resetBtn = document.getElementById('resetBtn');

// State variables
let net;           // The MobileNet module
let classifier;    // The KNN Classifier
let isPredicting = false;

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject(error)
            );
        } else {
            reject("Webcam not supported by this browser.");
        }
    });
}

async function app() {
    loadingOverlay.innerText = "Loading MobileNet...";
    
    // Load the base MobileNet model. We won't use it to predict, 
    // but instead use it to get intermediate activation features.
    net = await mobilenet.load();
    
    loadingOverlay.innerText = "Loading KNN Classifier...";
    // Create the KNN classifier
    classifier = knnClassifier.create();
    
    loadingOverlay.innerText = "Setting up Webcam...";
    try {
        await setupWebcam();
    } catch(e) {
        loadingOverlay.innerText = "Error: Please allow webcam access.";
        return;
    }

    loadingOverlay.style.display = 'none';

    // Hook up buttons
    document.querySelectorAll('.train-btn').forEach(btn => {
        // Continuous training while mouse is pressed down
        let trainInterval;
        
        const startTraining = () => {
             const classId = parseInt(btn.getAttribute('data-class'));
             // Run immediately on click
             addExample(classId);
             // And repeat if held down
             trainInterval = setInterval(() => addExample(classId), 100);
        };
        
        const stopTraining = () => {
             clearInterval(trainInterval);
        };

        btn.addEventListener('mousedown', startTraining);
        btn.addEventListener('mouseup', stopTraining);
        btn.addEventListener('mouseleave', stopTraining);
        
        // Mobile support
        btn.addEventListener('touchstart', (e) => { e.preventDefault(); startTraining(); });
        btn.addEventListener('touchend', (e) => { e.preventDefault(); stopTraining(); });
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        classifier.clearAllClasses();
        updateCounts();
        resultElement.innerText = "Waiting for training data...";
        isPredicting = false;
    });
}

/**
 * Grabs a frame from the webcam, gets its features from MobileNet, 
 * and adds it to the KNN Classifier pointing to the specified classId.
 */
function addExample(classId) {
    // Get the intermediate activation (features) of MobileNet from the webcam
    // 'conv_preds' is the feature layer right before the final classification
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier
    classifier.addExample(activation, classId);

    // Update UI counts
    updateCounts();

    // Start predicting if not already
    if (!isPredicting) {
        isPredicting = true;
        predictLoop();
    }
}

/**
 * Recursively predict the current webcam frame.
 */
async function predictLoop() {
    if (classifier.getNumClasses() > 0) {
        // Get features from webcam
        const activation = net.infer(webcamElement, 'conv_preds');
        
        // Predict using KNN
        // k=3 means we look at the 3 nearest neighbors
        const result = await classifier.predictClass(activation, 3);
        
        // Get the predicted label and confidence
        const predictedLabel = classes[result.label];
        const confidence = result.confidences[result.label];
        
        // Only classify if it's confident enough
        if (confidence > 0.5) {
            resultElement.innerText = `${predictedLabel} (${Math.round(confidence * 100)}%)`;
            
            // Apply color based on class
            if (result.label == 0) resultElement.style.color = 'var(--primary)';
            else if (result.label == 1) resultElement.style.color = 'var(--secondary)';
            else if (result.label == 2) resultElement.style.color = 'var(--tertiary)';
        } else {
            resultElement.innerText = "Unsure...";
            resultElement.style.color = "var(--text-color)";
        }
        
        // Clean up memory
        activation.dispose();
    }

    // Loop
    if (isPredicting) {
        // Use requestAnimationFrame so it doesn't freeze the UI 
        await tf.nextFrame();
        predictLoop();
    }
}

/**
 * Updates the example count labels on the buttons
 */
function updateCounts() {
    const counts = classifier.getClassExampleCount();
    document.getElementById('count0').innerText = `${counts[0] || 0} examples`;
    document.getElementById('count1').innerText = `${counts[1] || 0} examples`;
    document.getElementById('count2').innerText = `${counts[2] || 0} examples`;
}

// Start the app!
app();
