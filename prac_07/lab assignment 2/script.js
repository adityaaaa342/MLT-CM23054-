// Lab Assignment 2: Reload the model and verify predictions match the original.

const loadBtn = document.getElementById('loadBtn');
const statusDiv = document.getElementById('status');
const MODEL_SAVE_PATH = 'localstorage://lab1-model';

async function loadAndVerifyModel() {
    loadBtn.disabled = true;
    statusDiv.innerHTML = `Attempting to load model from ${MODEL_SAVE_PATH}...`;
    
    try {
        // Load the saved model
        const loadedModel = await tf.loadLayersModel(MODEL_SAVE_PATH);
        statusDiv.innerHTML = 'Model loaded successfully! Now verifying prediction...';
        
        // Let's verify prediction for x = 5 (Expected to be close to 17 if y = 3x + 2)
        const testValue = 5.0;
        const expectedExact = 3 * testValue + 2;
        
        const predictionTensor = loadedModel.predict(tf.tensor2d([testValue], [1, 1]));
        const prediction = predictionTensor.dataSync()[0];
        
        statusDiv.innerHTML = `
            <strong>Model loaded successfully!</strong><br><br>
            Testing x = ${testValue} <br>
            Expected y = ${expectedExact} <br>
            Model prediction y = ${prediction.toFixed(4)} <br><br>
            <em>The prediction matches the original trained model if you first ran Lab 1 on this browser.</em>
        `;
        
    } catch (error) {
        console.error('Error loading model:', error);
        statusDiv.innerHTML = `<span style="color: red;">Error: Could not load the model from LocalStorage. <br>Please make sure you have run Lab Assignment 1 in this same browser first to save the model.</span>`;
        loadBtn.disabled = false;
    }
}

loadBtn.addEventListener('click', loadAndVerifyModel);
