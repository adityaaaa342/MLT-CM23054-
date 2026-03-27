// Lab Assignment 1: Train a small model and save it to LocalStorage.

const trainBtn = document.getElementById('trainBtn');
const statusDiv = document.getElementById('status');
const MODEL_SAVE_PATH = 'localstorage://lab1-model';

async function trainAndSaveModel() {
    trainBtn.disabled = true;
    statusDiv.innerHTML = 'Creating and training model (y = 3x + 2)...';
    
    // Create a simple model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    // Compile model
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    // Data: y = 3x + 2
    const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
    const ys = tf.tensor2d([-1.0, 2.0, 5.0, 8.0, 11.0, 14.0], [6, 1]);
    
    // Train
    await model.fit(xs, ys, {epochs: 250});
    statusDiv.innerHTML = 'Model trained. Testing prediction for x=5 (Expected: 17)...';
    
    // Predict
    const prediction = model.predict(tf.tensor2d([5.0], [1, 1])).dataSync()[0];
    
    // Save to LocalStorage
    statusDiv.innerHTML = `Model prediction: ${prediction.toFixed(2)}. <br>Saving model to LocalStorage...`;
    await model.save(MODEL_SAVE_PATH);
    
    statusDiv.innerHTML = `Success! Model predicted ${prediction.toFixed(2)} and saved to ${MODEL_SAVE_PATH}.`;
    trainBtn.disabled = false;
}

trainBtn.addEventListener('click', trainAndSaveModel);
