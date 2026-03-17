// Dataset (Toy Sentiment Dataset)
const dataset = [
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

// Tokenizer & Vocabulary setup
const MAX_SEQ_LEN = 10;
let vocabulary = {"<PAD>": 0, "<UNK>": 1};
let vocabSize = 2;

function tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 0);
}

function buildVocabulary() {
    const vocabSet = new Set();
    dataset.forEach(item => tokenize(item.text).forEach(t => vocabSet.add(t)));
    let index = 2; // Reserve 0 and 1
    vocabSet.forEach(word => vocabulary[word] = index++);
    vocabSize = index;
    console.log(`Vocabulary built: ${vocabSize} words.`);
}

function textToSequence(text) {
    const tokens = tokenize(text);
    const seq = tokens.map(t => vocabulary[t] !== undefined ? vocabulary[t] : vocabulary["<UNK>"]);
    
    // Pad or truncate to MAX_SEQ_LEN
    if (seq.length > MAX_SEQ_LEN) {
        return seq.slice(0, MAX_SEQ_LEN); // Truncate
    } else {
        const padding = new Array(MAX_SEQ_LEN - seq.length).fill(vocabulary["<PAD>"]);
        return seq.concat(padding); // Post-pad
    }
}

function prepareData() {
    return tf.tidy(() => {
        const xs = [];
        const ys = [];
        
        dataset.forEach(item => {
            xs.push(textToSequence(item.text));
            ys.push([item.label]);
        });
        
        const indices = Array.from({length: dataset.length}, (_, i) => i);
        tf.util.shuffle(indices);
        
        return {
            X: tf.tensor2d(indices.map(i => xs[i]), [dataset.length, MAX_SEQ_LEN]),
            y: tf.tensor2d(indices.map(i => ys[i]))
        };
    });
}

// Model Definitions
let denseModel;
let rnnModel;

function buildModels() {
    const embedDim = 8;
    
    // 1. Dense Network
    denseModel = tf.sequential();
    denseModel.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embedDim,
        inputLength: MAX_SEQ_LEN
    }));
    denseModel.add(tf.layers.flatten());
    denseModel.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    denseModel.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    denseModel.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // 2. Simple RNN Network
    rnnModel = tf.sequential();
    rnnModel.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embedDim,
        inputLength: MAX_SEQ_LEN
    }));
    rnnModel.add(tf.layers.simpleRNN({ units: 16, returnSequences: false }));
    rnnModel.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    rnnModel.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
}

// UI Elements
const trainBothBtn = document.getElementById('train-both-btn');
const globalStatus = document.getElementById('global-status');
const testSection = document.getElementById('test-section');
const splitResults = document.getElementById('split-results');

const uiState = {
    dense: {
        epoch: document.getElementById('dense-epoch'),
        acc: document.getElementById('dense-acc'),
        time: document.getElementById('dense-time'),
        bar: document.getElementById('dense-progress')
    },
    rnn: {
        epoch: document.getElementById('rnn-epoch'),
        acc: document.getElementById('rnn-acc'),
        time: document.getElementById('rnn-time'),
        bar: document.getElementById('rnn-progress')
    }
};

const TOTAL_EPOCHS = 100;

// Simultaneous Training Logic
trainBothBtn.addEventListener('click', async () => {
    trainBothBtn.disabled = true;
    globalStatus.textContent = "Data preparation...";
    
    buildVocabulary();
    const data = prepareData();
    buildModels();
    
    globalStatus.textContent = "Training models simultaneously...";
    
    // Start both fit routines as promises
    const startTimeDense = performance.now();
    const densePromise = denseModel.fit(data.X, data.y, {
        epochs: TOTAL_EPOCHS,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                uiState.dense.epoch.textContent = `${epoch + 1}/${TOTAL_EPOCHS}`;
                uiState.dense.acc.textContent = logs.acc.toFixed(3);
                uiState.dense.bar.style.width = `${((epoch + 1) / TOTAL_EPOCHS) * 100}%`;
                uiState.dense.time.textContent = `${Math.round(performance.now() - startTimeDense)}ms`;
            }
        }
    });
    
    const startTimeRNN = performance.now();
    const rnnPromise = rnnModel.fit(data.X, data.y, {
        epochs: TOTAL_EPOCHS,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                uiState.rnn.epoch.textContent = `${epoch + 1}/${TOTAL_EPOCHS}`;
                uiState.rnn.acc.textContent = logs.acc.toFixed(3);
                uiState.rnn.bar.style.width = `${((epoch + 1) / TOTAL_EPOCHS) * 100}%`;
                uiState.rnn.time.textContent = `${Math.round(performance.now() - startTimeRNN)}ms`;
            }
        }
    });
    
    await Promise.all([densePromise, rnnPromise]);
    
    data.X.dispose();
    data.y.dispose();
    
    globalStatus.textContent = "Training Complete! Models are ready to test.";
    globalStatus.style.color = "var(--success)";
    
    testSection.classList.remove('disabled');
});

// Testing Logic
document.getElementById('test-btn').addEventListener('click', runPredictions);
document.getElementById('test-input').addEventListener('keypress', (e) => {
    if(e.key === 'Enter') runPredictions();
});

function runPredictions() {
    const text = document.getElementById('test-input').value.trim();
    if (!text) return;
    
    const seq = textToSequence(text);
    const inputTensor = tf.tensor2d([seq]);
    
    // Predict Dense
    const densePred = denseModel.predict(inputTensor).dataSync()[0];
    updateResultUI('dense', densePred);
    
    // Predict RNN
    const rnnPred = rnnModel.predict(inputTensor).dataSync()[0];
    updateResultUI('rnn', rnnPred);
    
    inputTensor.dispose();
    splitResults.classList.remove('hidden');
}

function updateResultUI(modelType, score) {
    const emojiEl = document.getElementById(`${modelType}-emoji`);
    const lblEl = document.getElementById(`${modelType}-pred-lbl`);
    const confEl = document.getElementById(`${modelType}-conf`);
    
    confEl.textContent = `Raw Output: ${score.toFixed(4)}`;
    
    if (score >= 0.6) {
        emojiEl.textContent = "🤩";
        lblEl.textContent = "Positive";
        lblEl.style.color = "var(--success)";
    } else if (score <= 0.4) {
        emojiEl.textContent = "😠";
        lblEl.textContent = "Negative";
        lblEl.style.color = "var(--danger)";
    } else {
        emojiEl.textContent = "🤔";
        lblEl.textContent = "Neutral/Uncertain";
        lblEl.style.color = "#fbbf24";
    }
}
