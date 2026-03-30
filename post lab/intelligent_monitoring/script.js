const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output');
const canvasCtx = canvasElement.getContext('2d');

const toggleCamBtn = document.getElementById('toggle-cam');
const toggleMonitorBtn = document.getElementById('toggle-monitor');
const systemStatus = document.getElementById('system-status');
const statusIndicator = document.querySelector('.status-indicator');
const loadingOverlay = document.getElementById('loading');

const currentStateEl = document.getElementById('current-state');
const confidenceScoreEl = document.getElementById('confidence-score');
const personCountEl = document.getElementById('person-count');
const alertList = document.getElementById('alert-list');

let net;
let isCameraOn = false;
let isMonitoring = false;
let animationFrameId;
let lastAlertTime = 0;

// Load PoseNet model
async function loadModel() {
    try {
        net = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: 640, height: 480 },
            multiplier: 0.75
        });
        loadingOverlay.classList.add('hidden');
        systemStatus.textContent = 'System Ready';
        console.log("PoseNet Model Loaded");
    } catch (e) {
        console.error("Error loading model", e);
        systemStatus.textContent = 'Model Load Failed';
    }
}

// Setup Webcam
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
            facingMode: 'user',
            width: 640,
            height: 480
        }
    });

    webcamElement.srcObject = stream;
    webcamElement.style.display = 'block';

    return new Promise((resolve) => {
        webcamElement.onloadedmetadata = () => {
            resolve(webcamElement);
        };
    });
}

// Draw keypoints and skeleton
function drawPose(pose) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw Keypoints
    pose.keypoints.forEach(point => {
        if (point.score > 0.5) {
            canvasCtx.beginPath();
            canvasCtx.arc(point.position.x, point.position.y, 5, 0, 2 * Math.PI);
            canvasCtx.fillStyle = '#10b981';
            canvasCtx.fill();
        }
    });

    // Draw Skeleton
    const adjacentKeyPoints = posenet.getAdjacentKeyPoints(pose.keypoints, 0.5);
    adjacentKeyPoints.forEach(keypoints => {
        canvasCtx.beginPath();
        canvasCtx.moveTo(keypoints[0].position.x, keypoints[0].position.y);
        canvasCtx.lineTo(keypoints[1].position.x, keypoints[1].position.y);
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#3b82f6';
        canvasCtx.stroke();
    });
}

// Analyze pose for insights
function analyzePose(pose) {
    // Basic logic to determine behavior
    let state = "Normal Activity";
    let stateClass = "success";
    let confidence = Math.round(pose.score * 100);
    
    // Extract keypoints
    const keypoints = pose.keypoints.reduce((acc, curr) => {
        acc[curr.part] = curr;
        return acc;
    }, {});

    // Check for "Hands Up" / Distress
    if (keypoints.leftWrist && keypoints.rightWrist && keypoints.nose) {
        if (keypoints.leftWrist.score > 0.5 && keypoints.rightWrist.score > 0.5 && keypoints.nose.score > 0.5) {
            if (keypoints.leftWrist.position.y < keypoints.nose.position.y && 
                keypoints.rightWrist.position.y < keypoints.nose.position.y) {
                state = "Distress Signal (Hands Up)";
                stateClass = "danger";
                createAlert("Distress pose detected! Both hands above head.", "danger");
            }
        }
    }

    // Check for potential fall (head drastically low compared to hips)
    if (keypoints.nose && keypoints.leftHip) {
        if (keypoints.nose.score > 0.5 && keypoints.leftHip.score > 0.5) {
            // In a normal standing/sitting position, nose is well above hips
            if (keypoints.nose.position.y > keypoints.leftHip.position.y) {
                state = "Potential Fall Detected";
                stateClass = "warning";
                createAlert("Abnormal posture detected. Potential fall.", "warning");
            }
        }
    }

    // Update UI
    currentStateEl.textContent = state;
    currentStateEl.className = `value ${stateClass}`;
    confidenceScoreEl.textContent = `${confidence}%`;
}

// Create Notification Alert
function createAlert(message, type) {
    const now = Date.now();
    // Throttle alerts
    if (now - lastAlertTime < 3000) return;
    lastAlertTime = now;

    const li = document.createElement('li');
    li.className = `alert-item ${type}`;
    
    const timeText = new Date().toLocaleTimeString();
    
    li.innerHTML = `
        <span class="message">${message}</span>
        <span class="time">${timeText}</span>
    `;
    
    alertList.prepend(li);
    
    // Keep max 5 alerts
    if (alertList.children.length > 5) {
        alertList.lastChild.remove();
    }
}

// Main Detection Loop
async function detectPose() {
    if (!isMonitoring) return;

    personCountEl.textContent = "1"; // Assuming single person for basic model
    const pose = await net.estimateSinglePose(webcamElement, {
        flipHorizontal: false
    });

    drawPose(pose);
    
    if (pose.score > 0.3) {
        analyzePose(pose);
    } else {
        currentStateEl.textContent = "No Person Detected";
        currentStateEl.className = "value";
        confidenceScoreEl.textContent = "0%";
        personCountEl.textContent = "0";
    }

    animationFrameId = requestAnimationFrame(detectPose);
}

// Event Listeners
toggleCamBtn.addEventListener('click', async () => {
    if (!isCameraOn) {
        try {
            toggleCamBtn.textContent = 'Starting...';
            await setupCamera();
            isCameraOn = true;
            toggleCamBtn.textContent = 'Stop Camera';
            toggleCamBtn.classList.replace('primary', 'secondary');
            
            toggleMonitorBtn.disabled = false;
            
        } catch (err) {
            alert('Could not start camera: ' + err.message);
            toggleCamBtn.textContent = 'Start Camera';
        }
    } else {
        const stream = webcamElement.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        webcamElement.srcObject = null;
        webcamElement.style.display = 'none';
        isCameraOn = false;
        
        toggleCamBtn.textContent = 'Start Camera';
        toggleCamBtn.classList.replace('secondary', 'primary');
        
        // Stop monitoring
        if (isMonitoring) toggleMonitorBtn.click();
        toggleMonitorBtn.disabled = true;
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
});

toggleMonitorBtn.addEventListener('click', () => {
    isMonitoring = !isMonitoring;
    if (isMonitoring) {
        toggleMonitorBtn.textContent = 'Stop Monitoring';
        toggleMonitorBtn.classList.replace('secondary', 'primary');
        statusIndicator.classList.add('active');
        systemStatus.textContent = 'Monitoring Active';
        detectPose();
    } else {
        toggleMonitorBtn.textContent = 'Start Monitoring';
        toggleMonitorBtn.classList.replace('primary', 'secondary');
        statusIndicator.classList.remove('active');
        systemStatus.textContent = 'Monitoring Paused';
        cancelAnimationFrame(animationFrameId);
        currentStateEl.textContent = "Paused";
        currentStateEl.className = "value";
        confidenceScoreEl.textContent = "0%";
        
        // Clear canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
});

// Init
loadModel();
