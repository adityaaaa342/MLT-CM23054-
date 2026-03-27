const classes = ['Apple', 'Banana', 'Orange'];
const webcamElement = document.getElementById('webcam');
const resultElement = document.getElementById('predictionResult');

let net, classifier;

async function app() {
    resultElement.innerText = "Loading Models...";
    net = await mobilenet.load();
    classifier = knnClassifier.create();
    
    resultElement.innerText = "Setting up Webcam...";
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamElement.srcObject = stream;
    
    resultElement.innerText = "Waiting for data...";
    
    document.querySelectorAll('.train-btn').forEach(btn => {
        btn.addEventListener('mousedown', () => addExample(parseInt(btn.getAttribute('data-class'))));
    });
}

function addExample(classId) {
    const activation = net.infer(webcamElement, 'conv_preds');
    classifier.addExample(activation, classId);
    
    // Update count
    const counts = classifier.getClassExampleCount();
    document.getElementById(`count${classId}`).innerText = `(${counts[classId] || 0})`;
    
    predict();
}

async function predict() {
    if (classifier.getNumClasses() > 0) {
        const activation = net.infer(webcamElement, 'conv_preds');
        const result = await classifier.predictClass(activation, 3);
        
        if (result.confidences[result.label] > 0.5) {
            resultElement.innerText = `${classes[result.label]} (${Math.round(result.confidences[result.label]*100)}%)`;
        } else {
            resultElement.innerText = "Unsure...";
        }
        activation.dispose();
        
        await tf.nextFrame();
        predict();
    }
}

app();
