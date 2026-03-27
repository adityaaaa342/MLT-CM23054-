// Lab Assignment 3: Export the model to files and re-import it; test predictions again.

const trainExportBtn = document.getElementById('trainExportBtn');
const exportStatus = document.getElementById('exportStatus');

const modelFilesInput = document.getElementById('modelFiles');
const importPredictBtn = document.getElementById('importPredictBtn');
const importStatus = document.getElementById('importStatus');

// 1. Train and Export (Download)
async function trainAndExportModel() {
    trainExportBtn.disabled = true;
    exportStatus.innerHTML = 'Creating and training model (y = -2x + 5)...';
    
    // Create a simple model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    // Compile model
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    // Data: y = -2x + 5
    const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
    const ys = tf.tensor2d([7.0, 5.0, 3.0, 1.0, -1.0, -3.0], [6, 1]);
    
    // Train
    await model.fit(xs, ys, {epochs: 250});
    exportStatus.innerHTML = 'Model trained. Triggering download of model (.json) and weights (.bin)...';
    
    // Export to downloads
    // This will trigger the browser to download 'my-exported-model.json' and 'my-exported-model.weights.bin'
    await model.save('downloads://my-exported-model');
    
    exportStatus.innerHTML = `Success! Model files downloaded. <br>Please locate them and select them in Step 2.`;
    trainExportBtn.disabled = false;
}

trainExportBtn.addEventListener('click', trainAndExportModel);


// 2. Import and Predict
modelFilesInput.addEventListener('change', () => {
    if (modelFilesInput.files.length === 2) {
        importPredictBtn.disabled = false;
        importStatus.innerHTML = 'Files selected. Click Import and Predict.';
    } else {
        importPredictBtn.disabled = true;
        importStatus.innerHTML = '<span style="color:red;">Please select EXACTLY TWO files (.json and .bin).</span>';
    }
});

async function importAndPredict() {
    importPredictBtn.disabled = true;
    importStatus.innerHTML = 'Importing model from selected files...';
    
    try {
        const files = modelFilesInput.files;
        
        // tf.loadLayersModel taking tf.io.browserFiles(FileList)
        const loadedModel = await tf.loadLayersModel(tf.io.browserFiles(files));
        
        importStatus.innerHTML = 'Model imported successfully! Running prediction...';
        
        // Let's verify prediction for x = 10 (Expected: roughly -15)
        const testValue = 10.0;
        const expectedExact = -2 * testValue + 5;
        
        const predictionTensor = loadedModel.predict(tf.tensor2d([testValue], [1, 1]));
        const prediction = predictionTensor.dataSync()[0];
        
        importStatus.innerHTML = `
            <strong>Import and Prediction Successful!</strong><br><br>
            Testing x = ${testValue} <br>
            Expected y = ${expectedExact} <br>
            Model prediction y = ${prediction.toFixed(4)}
        `;
        
    } catch (error) {
        console.error('Error importing model:', error);
        importStatus.innerHTML = `<span style="color: red;">Error during import: ${error.message}</span>`;
        importPredictBtn.disabled = false;
    }
}

importPredictBtn.addEventListener('click', importAndPredict);
