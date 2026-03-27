const webcamElement = document.getElementById('webcam');
const statusElement = document.getElementById('status');
const evalBtn = document.getElementById('evalBtn');
const resultsSection = document.getElementById('results');
const accuracyResult = document.getElementById('accuracyResult');
const matrixHead = document.getElementById('matrixHead');
const matrixBody = document.getElementById('matrixBody');
const trainButtonsContainer = document.getElementById('trainButtonsContainer');
const valButtonsContainer = document.getElementById('valButtonsContainer');
const newClassNameInput = document.getElementById('newClassName');
const addCategoryBtn = document.getElementById('addCategoryBtn');

// Initial classes
let classes = ['Apple', 'Banana', 'Orange'];

let net;
let classifier;
let validationData = []; 

async function app() {
    statusElement.innerText = "Loading Models...";
    try {
        net = await mobilenet.load();
        classifier = knnClassifier.create();
        
        statusElement.innerText = "Setting up Webcam...";
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamElement.srcObject = stream;
        
        await new Promise((resolve) => {
            webcamElement.onloadeddata = () => resolve();
        });

        statusElement.innerText = "Ready! Train initial classes, evaluate, then try adding a new category.";
        renderButtons();
        
        // Setup Evaluate Button
        evalBtn.addEventListener('click', evaluateModel);
        
        // Setup Add Category Button
        addCategoryBtn.addEventListener('click', () => {
            const newName = newClassNameInput.value.trim();
            if (newName) {
                addClass(newName);
                newClassNameInput.value = '';
            }
        });
        
    } catch (e) {
        statusElement.innerText = "Error: " + e.message;
    }
}

function renderButtons() {
    trainButtonsContainer.innerHTML = '';
    valButtonsContainer.innerHTML = '';
    
    classes.forEach((className, classId) => {
        // Create Train Button
        const tBtn = document.createElement('button');
        tBtn.className = 'train-btn';
        tBtn.innerHTML = `Train "${className}" <span class="count${classId}">(0)</span>`;
        tBtn.addEventListener('click', () => addTrainingExample(classId));
        trainButtonsContainer.appendChild(tBtn);
        
        // Create Validation Button
        const vBtn = document.createElement('button');
        vBtn.className = 'val-btn';
        vBtn.innerHTML = `Add Val "${className}" <span class="val-count${classId}">(0)</span>`;
        vBtn.addEventListener('click', () => addValidationExample(classId));
        valButtonsContainer.appendChild(vBtn);
    });
    
    // Update counts on re-render
    updateAllCounts();
}

function addClass(className) {
    if (classes.includes(className)) {
        alert("Class already exists!");
        return;
    }
    classes.push(className);
    statusElement.innerText = `Added new category: ${className}. You can now train it and re-evaluate.`;
    renderButtons();
}

function addTrainingExample(classId) {
    const activation = net.infer(webcamElement, 'conv_preds');
    classifier.addExample(activation, classId);
    updateAllCounts();
    statusElement.innerText = `Added training example for ${classes[classId]}`;
}

function addValidationExample(classId) {
    const activation = net.infer(webcamElement, 'conv_preds');
    validationData.push({ activation: activation, trueLabel: classId });
    updateAllCounts();
    statusElement.innerText = `Added validation example for ${classes[classId]}`;
}

function updateAllCounts() {
    // Train counts from classifier
    const counts = classifier.getNumClasses() > 0 ? classifier.getClassExampleCount() : {};
    
    // Validation counts from array
    const valCounts = {};
    validationData.forEach(item => {
        valCounts[item.trueLabel] = (valCounts[item.trueLabel] || 0) + 1;
    });
    
    classes.forEach((_, classId) => {
        const tSpan = document.querySelector(`.count${classId}`);
        const vSpan = document.querySelector(`.val-count${classId}`);
        if(tSpan) tSpan.innerText = `(${counts[classId] || 0})`;
        if(vSpan) vSpan.innerText = `(${valCounts[classId] || 0})`;
    });
}

async function evaluateModel() {
    const numClasses = classes.length;
    
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

    // Initialize dynamic confusion matrix
    let matrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
    let correct = 0;

    // Predict
    for (let i = 0; i < validationData.length; i++) {
        const item = validationData[i];
        
        // Predict among all available classes
        const result = await classifier.predictClass(item.activation, 3);
        const predictedLabel = parseInt(result.label);
        const trueLabel = item.trueLabel;
        
        // Handle case where predictedLabel might be out of current bounds
        // (Shouldn't happen if array is sized properly and labels match index)
        if (predictedLabel < numClasses && trueLabel < numClasses) {
            matrix[trueLabel][predictedLabel]++;
        }
        
        if (predictedLabel === trueLabel) {
            correct++;
        }
    }

    // Calc Accuracy
    const accuracy = (correct / validationData.length) * 100;
    accuracyResult.innerText = `Accuracy: ${accuracy.toFixed(1)}% (${correct}/${validationData.length})`;
    
    renderMatrix(matrix);
    
    resultsSection.style.display = 'block';
    statusElement.innerText = "Evaluation complete!";
    evalBtn.disabled = false;
}

function renderMatrix(matrix) {
    matrixHead.innerHTML = '';
    matrixBody.innerHTML = '';
    
    // Render Head
    const trHead = document.createElement('tr');
    const thCorner = document.createElement('th');
    thCorner.innerText = 'True \\ Pred';
    trHead.appendChild(thCorner);
    
    classes.forEach(c => {
        const th = document.createElement('th');
        th.innerText = c;
        trHead.appendChild(th);
    });
    matrixHead.appendChild(trHead);
    
    // Render Body
    for (let i = 0; i < matrix.length; i++) {
        const row = document.createElement('tr');
        const thRow = document.createElement('th');
        thRow.innerText = classes[i];
        row.appendChild(thRow);
        
        for (let j = 0; j < matrix[i].length; j++) {
            const td = document.createElement('td');
            td.innerText = matrix[i][j];
            
            if (i === j && matrix[i][j] > 0) {
                td.style.backgroundColor = '#d1fae5';
                td.style.fontWeight = 'bold';
            } else if (matrix[i][j] > 0) {
                td.style.backgroundColor = '#fee2e2';
            }
            row.appendChild(td);
        }
        matrixBody.appendChild(row);
    }
}

app();
