const classes = ['Apple', 'Banana', 'Orange'];
const webcamElement = document.getElementById('webcam');
const statusElement = document.getElementById('status');
const evalBtn = document.getElementById('evalBtn');
const resultsSection = document.getElementById('results');
const accuracyResult = document.getElementById('accuracyResult');
const confusionMatrixTbody = document.querySelector('#confusionMatrix tbody');

let net;
let classifier;
let validationData = []; // Array to store {activation, label}

async function app() {
    statusElement.innerText = "Loading Models...";
    try {
        net = await mobilenet.load();
        classifier = knnClassifier.create();
        
        statusElement.innerText = "Setting up Webcam...";
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamElement.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            webcamElement.onloadeddata = () => {
                resolve();
            };
        });

        statusElement.innerText = "Ready! Add Training Data first, then Validation Data.";
        setupButtons();
    } catch (e) {
        statusElement.innerText = "Error loading webcam or models: " + e.message;
    }
}

function setupButtons() {
    // Training Buttons
    document.querySelectorAll('.train-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const classId = parseInt(btn.getAttribute('data-class'));
            addTrainingExample(classId);
        });
    });

    // Validation Buttons
    document.querySelectorAll('.val-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const classId = parseInt(btn.getAttribute('data-class'));
            addValidationExample(classId);
        });
    });

    // Evaluate Button
    evalBtn.addEventListener('click', evaluateModel);
}

function addTrainingExample(classId) {
    const activation = net.infer(webcamElement, 'conv_preds');
    classifier.addExample(activation, classId);
    
    // Update count
    const counts = classifier.getClassExampleCount();
    document.querySelector(`.count${classId}`).innerText = `(${counts[classId] || 0})`;
    statusElement.innerText = `Added training example for ${classes[classId]}`;
}

function addValidationExample(classId) {
    // Get feature activation but DO NOT add to classifier
    const activation = net.infer(webcamElement, 'conv_preds');
    
    // Store it for later evaluation
    validationData.push({
        activation: activation,
        trueLabel: classId
    });
    
    // Update count manually by counting how many of this class are in validationData
    const count = validationData.filter(item => item.trueLabel === classId).length;
    document.querySelector(`.val-count${classId}`).innerText = `(${count})`;
    statusElement.innerText = `Added validation example for ${classes[classId]}`;
}

async function evaluateModel() {
    if (classifier.getNumClasses() === 0) {
        alert("Please add some training data first!");
        return;
    }
    if (validationData.length === 0) {
        alert("Please add some validation data first!");
        return;
    }

    statusElement.innerText = "Running evaluation...";
    evalBtn.disabled = true;

    // Initialize confusion matrix [trueLabel][predictedLabel]
    let matrix = [
        [0, 0, 0], // True Apple
        [0, 0, 0], // True Banana
        [0, 0, 0]  // True Orange
    ];
    let correct = 0;

    // Predict on validation set
    for (let i = 0; i < validationData.length; i++) {
        const item = validationData[i];
        
        // Predict using the stored activation
        const result = await classifier.predictClass(item.activation, 3);
        const predictedLabel = parseInt(result.label);
        const trueLabel = item.trueLabel;
        
        // Update matrix
        matrix[trueLabel][predictedLabel]++;
        
        if (predictedLabel === trueLabel) {
            correct++;
        }
    }

    // Calculate Accuracy
    const accuracy = (correct / validationData.length) * 100;
    accuracyResult.innerText = `Accuracy: ${accuracy.toFixed(1)}% (${correct}/${validationData.length})`;
    
    // Render Confusion Matrix
    renderMatrix(matrix);
    
    resultsSection.style.display = 'block';
    statusElement.innerText = "Evaluation complete!";
    evalBtn.disabled = false;
}

function renderMatrix(matrix) {
    confusionMatrixTbody.innerHTML = '';
    
    for (let i = 0; i < matrix.length; i++) {
        const row = document.createElement('tr');
        
        // Header cell for true label
        const th = document.createElement('th');
        th.innerText = classes[i] + ' (True)';
        row.appendChild(th);
        
        // Data cells for predictions
        for (let j = 0; j < matrix[i].length; j++) {
            const td = document.createElement('td');
            td.innerText = matrix[i][j];
            
            // Highlight correct predictions across the diagonal
            if (i === j && matrix[i][j] > 0) {
                td.style.backgroundColor = '#d1fae5'; // Light green
                td.style.fontWeight = 'bold';
            } else if (matrix[i][j] > 0) {
                td.style.backgroundColor = '#fee2e2'; // Light red for errors
            }
            
            row.appendChild(td);
        }
        
        confusionMatrixTbody.appendChild(row);
    }
}

app();
