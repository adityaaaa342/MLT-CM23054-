// script.js
const trainBtn = document.getElementById('trainBtn');
const trainStatus = document.getElementById('trainStatus');
const loadBtn = document.getElementById('loadBtn');
const predictInput = document.getElementById('predictInput');
const predictBtn = document.getElementById('predictBtn');
const predictStatus = document.getElementById('predictStatus');
const predictionResult = document.getElementById('predictionResult');

// Using localstorage path to save model within the browser
const MODEL_SAVE_PATH = 'localstorage://my-simple-linear-model';
let loadedModel = null;

/**
 * Creates, trains, and saves a simple model
 */
async function createAndTrainModel() {
    trainBtn.disabled = true;
    trainStatus.innerText = 'Initializing model...';

    // 1. Create a simple sequential model
    const model = tf.sequential();
    // A single dense layer with 1 unit and input shape of 1
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // 2. Compile the model
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    // 3. Generate some synthetic data for training (y = 2x - 1)
    const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
    const ys = tf.tensor2d([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], [6, 1]);

    trainStatus.innerText = 'Training model on data (y = 2x - 1)... Please wait.';

    // 4. Train the model
    await model.fit(xs, ys, {
        epochs: 250,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                // Update status every 50 epochs
                if ((epoch + 1) % 50 === 0) {
                    trainStatus.innerText = `Training... Epoch ${epoch + 1}/250 (Loss: ${logs.loss.toFixed(4)})`;
                }
            }
        }
    });

    trainStatus.innerText = 'Training complete! Saving model locally...';

    // 5. Save the model to Local Storage
    try {
        await model.save(MODEL_SAVE_PATH);
        trainStatus.innerText = `Model trained and saved locally at: ${MODEL_SAVE_PATH}`;
        trainStatus.style.borderLeftColor = '#22c55e'; // Green border for success
        trainBtn.disabled = false;
        
        // Enable load button
        loadBtn.disabled = false;
        checkExistingModel(); // Refresh status
    } catch (saveError) {
        trainStatus.innerText = `Error saving model: ${saveError.message}`;
        trainStatus.style.borderLeftColor = '#ef4444'; // Red border for error
        trainBtn.disabled = false;
    }
}

/**
 * Loads the saved model from Local Storage
 */
async function loadModelLocally() {
    loadBtn.disabled = true;
    predictStatus.innerText = 'Loading model from local storage...';

    try {
        // Load the saved model
        loadedModel = await tf.loadLayersModel(MODEL_SAVE_PATH);
        predictStatus.innerText = 'Model loaded successfully! You can now make predictions.';
        predictStatus.style.borderLeftColor = '#22c55e';
        
        // Enable prediction inputs
        predictInput.disabled = false;
        predictBtn.disabled = false;
        predictInput.focus();
    } catch (error) {
        console.error('Error loading model:', error);
        predictStatus.innerText = 'Error loading model. Make sure you train and save it first.';
        predictStatus.style.borderLeftColor = '#ef4444';
        loadBtn.disabled = false;
    }
}

/**
 * Makes a prediction using the loaded model
 */
async function makePrediction() {
    if (!loadedModel) {
        predictStatus.innerText = 'Please load the model first.';
        return;
    }

    const inputValue = parseFloat(predictInput.value);
    if (isNaN(inputValue)) {
        predictionResult.innerHTML = '<div class="result-box" style="border-left-color: #ef4444;">Please enter a valid numeric value.</div>';
        return;
    }

    predictBtn.disabled = true;

    // Use tf.tidy to clean up intermediate tensors automatically and prevent memory leaks
    tf.tidy(() => {
        // Create an input tensor (2D tensor because dense layer expects 2D)
        const inputTensor = tf.tensor2d([inputValue], [1, 1]);
        
        // Run prediction
        const outputTensor = loadedModel.predict(inputTensor);
        
        // Get the predicted value synchronously
        const prediction = outputTensor.dataSync()[0];
        
        // Calculate true expected value for comparison
        const expectedValue = 2 * inputValue - 1;
        
        predictionResult.innerHTML = `
            <div class="result-box">
                <strong>Input (x):</strong> ${inputValue} <br> 
                <strong>Predicted y:</strong> ${prediction.toFixed(4)} <br>
                <small style="color: #64748b;">(Expected approx: ${expectedValue})</small>
            </div>
        `;
    });

    predictBtn.disabled = false;
}

/**
 * Checks if a model already exists in local storage on page load
 */
async function checkExistingModel() {
    try {
        const models = await tf.io.listModels();
        // Check if our local storage model path exists in keys
        if (Object.keys(models).includes(MODEL_SAVE_PATH)) {
            loadBtn.disabled = false;
            trainStatus.innerText = 'Existing model found in local storage. You can load it or retrain it.';
            trainStatus.style.borderLeftColor = '#3b82f6';
        } else {
            loadBtn.disabled = true;
        }
    } catch (e) {
        console.error("Error checking existing models", e);
    }
}

// Attach Event Listeners
trainBtn.addEventListener('click', createAndTrainModel);
loadBtn.addEventListener('click', loadModelLocally);
predictBtn.addEventListener('click', makePrediction);

// Allow pressing 'Enter' to predict
predictInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        makePrediction();
    }
});

// Initialize by checking for saved models
checkExistingModel();
