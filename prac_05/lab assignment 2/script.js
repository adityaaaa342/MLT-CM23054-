// Synthetic dataset from Lab 1
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

// Tokenization and Vocabulary
let vocabulary = {};
let vocabSize = 0;

function tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 0);
}

function buildVocabulary() {
    const vocabSet = new Set();
    dataset.forEach(item => tokenize(item.text).forEach(t => vocabSet.add(t)));
    let index = 0;
    vocabSet.forEach(word => vocabulary[word] = index++);
    vocabSize = index;
}

function prepareData() {
    return tf.tidy(() => {
        const xs = [], ys = [];
        dataset.forEach(item => {
            const bow = new Array(vocabSize).fill(0);
            tokenize(item.text).forEach(token => {
                if (vocabulary[token] !== undefined) bow[vocabulary[token]] = 1;
            });
            xs.push(bow);
            ys.push([item.label]);
        });
        
        const indices = Array.from({length: dataset.length}, (_, i) => i);
        tf.util.shuffle(indices);
        
        return {
            X: tf.tensor2d(indices.map(i => xs[i])),
            y: tf.tensor2d(indices.map(i => ys[i]))
        };
    });
}

// Background Model Training Setup
let model;

async function initBackgroundModel() {
    buildVocabulary();
    const data = prepareData();
    
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [vocabSize], units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({ optimizer: tf.train.adam(0.02), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    
    // Train silently
    await model.fit(data.X, data.y, { epochs: 60, shuffle: true });
    
    data.X.dispose();
    data.y.dispose();
    
    console.log("Background training complete.");
    
    // Hide overlay
    document.getElementById('loading-overlay').style.opacity = '0';
    setTimeout(() => {
        document.getElementById('loading-overlay').style.display = 'none';
    }, 500);
}

// Initialize on load
window.addEventListener('DOMContentLoaded', initBackgroundModel);

// UI Elements & Interaction
const analyzeBtn = document.getElementById('analyze-btn');
const sentenceInput = document.getElementById('sentence-input');
const resultsDashboard = document.getElementById('results-dashboard');

const sentimentEmoji = document.getElementById('sentiment-emoji');
const sentimentLabel = document.getElementById('sentiment-label');
const confidenceBadge = document.getElementById('confidence-badge');
const gaugeMarker = document.querySelector('.gauge-marker');
const interpretText = document.getElementById('interpret-text');
const tokensContainer = document.getElementById('tokens-container');

analyzeBtn.addEventListener('click', () => {
    const text = sentenceInput.value.trim();
    if (!text) return;
    
    // Analyze input vs vocabulary
    const tokens = tokenize(text);
    const bow = new Array(vocabSize).fill(0);
    const knownWords = [];
    const unknownWords = [];
    
    tokens.forEach(token => {
        if (vocabulary[token] !== undefined) {
            bow[vocabulary[token]] = 1;
            knownWords.push(token);
        } else {
            unknownWords.push(token);
        }
    });
    
    // Predict
    const inputTensor = tf.tensor2d([bow]);
    const pred = model.predict(inputTensor);
    const score = pred.dataSync()[0];
    
    inputTensor.dispose();
    pred.dispose();
    
    updateDashboard(score, tokens, knownWords);
});

function updateDashboard(score, allTokens, knownWords) {
    resultsDashboard.classList.remove('hidden');
    
    // 1. Prediction and Gauge
    const percentage = (score * 100).toFixed(1);
    confidenceBadge.textContent = `${percentage}% Confidence (Raw: ${score.toFixed(4)})`;
    gaugeMarker.style.left = `${percentage}%`;
    
    if (score >= 0.75) {
        sentimentEmoji.textContent = "🤩";
        sentimentLabel.textContent = "Strongly Positive";
        sentimentLabel.style.color = "var(--success)";
        interpretText.textContent = "The model is very confident this implies a good sentiment.";
    } else if (score <= 0.25) {
        sentimentEmoji.textContent = "😠";
        sentimentLabel.textContent = "Strongly Negative";
        sentimentLabel.style.color = "var(--danger)";
        interpretText.textContent = "The model is very confident this implies a bad sentiment.";
    } else if (score > 0.4 && score < 0.6) {
        sentimentEmoji.textContent = "🤔";
        sentimentLabel.textContent = "Uncertain / Neutral";
        sentimentLabel.style.color = "var(--warning)";
        interpretText.textContent = "The model is confused. It likely missing strong keyword indicators in its vocabulary.";
    } else if (score > 0.5) {
        sentimentEmoji.textContent = "🙂";
        sentimentLabel.textContent = "Leaning Positive";
        sentimentLabel.style.color = "var(--success)";
        interpretText.textContent = "The model thinks this is positive, but lacks high certainty.";
    } else {
        sentimentEmoji.textContent = "🙁";
        sentimentLabel.textContent = "Leaning Negative";
        sentimentLabel.style.color = "var(--danger)";
        interpretText.textContent = "The model thinks this is negative, but lacks high certainty.";
    }
    
    // 2. Token Analysis Breakdown
    tokensContainer.innerHTML = '';
    
    if (allTokens.length === 0) {
        tokensContainer.innerHTML = '<span class="token text-muted">No valid words detected.</span>';
        return;
    }
    
    allTokens.forEach(token => {
        const span = document.createElement('span');
        span.className = 'token';
        
        if (knownWords.includes(token)) {
            span.classList.add('known');
            span.textContent = token;
        } else {
            span.classList.add('unknown');
            span.textContent = token;
        }
        
        tokensContainer.appendChild(span);
    });
}
