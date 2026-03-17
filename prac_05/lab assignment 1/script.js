// A small synthetic dataset of positive and negative sentences
const rawDataset = [
    { text: "I love this product it is amazing", label: 1 },
    { text: "This is the best experience ever", label: 1 },
    { text: "Great job I am very happy", label: 1 },
    { text: "Awesome fantastic wonderful", label: 1 },
    { text: "I really enjoyed this completely", label: 1 },
    { text: "This looks so good and works perfectly", label: 1 },
    { text: "Simply excellent and brilliant", label: 1 },
    { text: "I am totally satisfied with this", label: 1 },
    { text: "Superb quality highly recommended", label: 1 },
    { text: "A beautiful masterpiece", label: 1 },

    { text: "I hate this it is terrible", label: 0 },
    { text: "This is the worst experience ever", label: 0 },
    { text: "Bad job I am very disappointed", label: 0 },
    { text: "Awful horrible disgusting", label: 0 },
    { text: "I really disliked this completely", label: 0 },
    { text: "This looks so bad and broken", label: 0 },
    { text: "Simply awful and garbage", label: 0 },
    { text: "I am totally unsatisfied with this", label: 0 },
    { text: "Poor quality do not buy", label: 0 },
    { text: "A complete disaster and failure", label: 0 }
];

// Tokenization and Vocabulary Building
let vocabulary = {};
let vocabSize = 0;

function tokenize(text) {
    return text.toLowerCase()
               .replace(/[^\w\s]/g, '') // remove punctuation
               .split(/\s+/)
               .filter(word => word.length > 0);
}

function buildVocabulary(dataset) {
    const vocabSet = new Set();
    dataset.forEach(item => {
        const tokens = tokenize(item.text);
        tokens.forEach(token => vocabSet.add(token));
    });
    
    let index = 0;
    vocabSet.forEach(word => {
        vocabulary[word] = index++;
    });
    vocabSize = index;
    console.log(`Vocabulary built with ${vocabSize} words.`);
}

// Convert sentence to Bag of Words tensor
function textToTensor(text) {
    const tokens = tokenize(text);
    const bow = new Array(vocabSize).fill(0);
    tokens.forEach(token => {
        if (vocabulary[token] !== undefined) {
            bow[vocabulary[token]] = 1;
        }
    });
    return tf.tensor2d([bow]); // Shape [1, vocabSize]
}

function prepareData(dataset) {
    return tf.tidy(() => {
        const xs = [];
        const ys = [];
        dataset.forEach(item => {
            const tokens = tokenize(item.text);
            const bow = new Array(vocabSize).fill(0);
            tokens.forEach(token => {
                if (vocabulary[token] !== undefined) {
                    bow[vocabulary[token]] = 1;
                }
            });
            xs.push(bow);
            ys.push([item.label]);
        });
        
        // Shuffle the data randomly
        const indices = Array.from({length: dataset.length}, (_, i) => i);
        tf.util.shuffle(indices);
        
        const shuffledXs = indices.map(i => xs[i]);
        const shuffledYs = indices.map(i => ys[i]);

        return {
            X: tf.tensor2d(shuffledXs),         // Shape [num_samples, vocabSize]
            y: tf.tensor2d(shuffledYs)          // Shape [num_samples, 1]
        };
    });
}

// Model Setup
let model;

function buildModel() {
    model = tf.sequential();
    
    // First hidden layer
    model.add(tf.layers.dense({
        inputShape: [vocabSize],
        units: 16,
        activation: 'relu'
    }));
    
    // Output layer (Binary classification: sigmoid)
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log("Model built.");
}

// UI Elements
const trainBtn = document.getElementById('train-btn');
const trainStatus = document.getElementById('train-status');
const epochVal = document.getElementById('epoch-val');
const lossVal = document.getElementById('loss-val');
const accVal = document.getElementById('acc-val');
const progressBar = document.getElementById('progress-bar');

const testSection = document.getElementById('test-section');
const testBtn = document.getElementById('test-btn');
const testInput = document.getElementById('test-input');
const predictionLabel = document.getElementById('prediction-label');
const predictionConfidence = document.getElementById('prediction-confidence');
const sentimentIcon = document.getElementById('sentiment-icon');
const posBar = document.getElementById('positive-bar');
const negBar = document.getElementById('negative-bar');

// Training Workflow
trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    trainStatus.textContent = "Processing dataset...";
    
    // 1. Prepare Data
    buildVocabulary(rawDataset);
    const data = prepareData(rawDataset);
    
    // 2. Build Model
    buildModel();
    
    trainStatus.textContent = "Training model...";
    
    const TOTAL_EPOCHS = 50;
    
    // 3. Train Model
    await model.fit(data.X, data.y, {
        epochs: TOTAL_EPOCHS,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                // Update UI
                epochVal.textContent = `${epoch + 1} / ${TOTAL_EPOCHS}`;
                lossVal.textContent = logs.loss.toFixed(4);
                accVal.textContent = logs.acc.toFixed(4);
                
                const progress = ((epoch + 1) / TOTAL_EPOCHS) * 100;
                progressBar.style.width = `${progress}%`;
            }
        }
    });
    
    // Cleanup tensors
    data.X.dispose();
    data.y.dispose();
    
    trainStatus.textContent = "Training complete!";
    trainStatus.classList.remove('text-muted');
    trainStatus.style.color = "var(--success-color)";
    
    // Enable Testing Section
    testSection.classList.remove('disabled');
});

// Testing Workflow
testBtn.addEventListener('click', () => {
    runPrediction();
});

testInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runPrediction();
});

function runPrediction() {
    const text = testInput.value.trim();
    if (!text) return;
    
    const inputTensor = textToTensor(text);
    const prediction = model.predict(inputTensor);
    const score = prediction.dataSync()[0]; // Value between 0 and 1
    
    inputTensor.dispose();
    prediction.dispose();
    
    updateTestUI(score);
}

function updateTestUI(score) {
    const posPercent = (score * 100).toFixed(1);
    const negPercent = ((1 - score) * 100).toFixed(1);
    
    posBar.style.width = `${posPercent}%`;
    negBar.style.width = `${negPercent}%`;
    
    predictionConfidence.textContent = `Confidence: ${(Math.max(score, 1-score)*100).toFixed(1)}%`;
    
    // Bump animation
    sentimentIcon.classList.remove('bump');
    void sentimentIcon.offsetWidth; // Trigger reflow
    sentimentIcon.classList.add('bump');
    
    if (score >= 0.5) {
        sentimentIcon.textContent = "🤩";
        predictionLabel.textContent = "Positive Sentiment";
        predictionLabel.className = "positive-text";
    } else {
        sentimentIcon.textContent = "😠";
        predictionLabel.textContent = "Negative Sentiment";
        predictionLabel.className = "negative-text";
    }
}
