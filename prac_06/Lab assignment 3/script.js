let video;
let singleModel, multiModel;
let activeModelStr = 'single';
let poses = [];
let connections;

function preload() {
    // Load both MoveNet models
    // SINGLEPOSE_LIGHTNING is highly optimized for 1 person (higher framerate/accuracy usually)
    singleModel = ml5.bodyPose("MoveNet", { modelType: "SINGLEPOSE_LIGHTNING" });
    
    // MULTIPOSE_LIGHTNING can track multiple people but might see performance drops
    multiModel = ml5.bodyPose("MoveNet", { modelType: "MULTIPOSE_LIGHTNING" });
}

function setup() {
    // Canvas dimensions
    const canvas = createCanvas(640, 480);
    canvas.parent('canvas-container');

    // Init Webcam
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // Start with the single-pose model
    singleModel.detectStart(video, gotPoses);
    
    // Get skeleton definition (same for both)
    connections = singleModel.getSkeleton();
    
    updateStatus('Models Loaded! Running Single-Pose.', 'success');
}

// Global function to be called from the UI buttons
window.setModel = function(type) {
    if (activeModelStr === type) return; // Prevent unnecessary reloads
    
    // Update active button styling
    document.getElementById('btn-single').classList.remove('active');
    document.getElementById('btn-multi').classList.remove('active');
    document.getElementById('btn-' + type).classList.add('active');
    
    // Stop the previously active model
    if (activeModelStr === 'single') {
        singleModel.detectStop();
    } else {
        multiModel.detectStop();
    }
    
    // Reset state
    poses = [];
    activeModelStr = type;
    
    // Start the new model
    if (type === 'single') {
        singleModel.detectStart(video, gotPoses);
        updateStatus('Running Single-Pose Model.', 'success');
    } else {
        multiModel.detectStart(video, gotPoses);
        updateStatus('Running Multi-Pose Model.', 'success');
    }
};

function gotPoses(results) {
    poses = results;
    
    // --- Metric Calculation ---
    let totalConfidence = 0;
    let keypointCount = 0;
    
    // Loop through all people and all points to get the aggregate accuracy score
    for (let i = 0; i < poses.length; i++) {
        for (let j = 0; j < poses[i].keypoints.length; j++) {
            let kp = poses[i].keypoints[j];
            totalConfidence += kp.confidence;
            keypointCount++;
        }
    }
    
    let avgConf = 0;
    if (keypointCount > 0) {
        avgConf = (totalConfidence / keypointCount) * 100;
    }
    
    // Update the DOM Stats
    document.getElementById('avg-confidence').innerText = avgConf.toFixed(1) + '%';
    document.getElementById('person-count').innerText = poses.length;
}

function draw() {
    // Mirror the video horizontally
    push();
    translate(width, 0);
    scale(-1, 1);
    
    image(video, 0, 0, width, height);

    // Render keypoints and skeleton
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i];
        
        // Draw Skeleton Connections
        for (let j = 0; j < connections.length; j++) {
            let pointAIndex = connections[j][0];
            let pointBIndex = connections[j][1];
            let pointA = pose.keypoints[pointAIndex];
            let pointB = pose.keypoints[pointBIndex];
            
            // Only draw lines if highly confident (minimizes visual noise)
            if (pointA.confidence > 0.1 && pointB.confidence > 0.1) {
                stroke(16, 185, 129); // emerald-500
                strokeWeight(4);
                line(pointA.x, pointA.y, pointB.x, pointB.y);
            }
        }

        // Draw Keypoints
        for (let j = 0; j < pose.keypoints.length; j++) {
            let keypoint = pose.keypoints[j];
            
            if (keypoint.confidence > 0.1) {
                fill(56, 189, 248); // sky-400
                noStroke();
                ellipse(keypoint.x, keypoint.y, 12, 12);
                
                // Add tiny white pupil
                fill(255);
                ellipse(keypoint.x, keypoint.y, 4, 4);
                
                // Optional: render the exact confidence of each point
                // push();
                // translate(keypoint.x, keypoint.y);
                // scale(-1, 1); // fix mirror for text
                // fill(255);
                // textSize(10);
                // text((keypoint.confidence * 100).toFixed(0) + '%', 10, -10);
                // pop();
            }
        }
    }
    pop();
}

// Visual status tracker
function updateStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.innerText = message;
    
    if (type === 'success') {
        statusEl.style.color = '#10b981';
        statusEl.style.background = 'rgba(16, 185, 129, 0.1)';
        statusEl.style.borderColor = 'rgba(16, 185, 129, 0.2)';
    } else {
        statusEl.style.color = '#f59e0b';
        statusEl.style.background = 'rgba(245, 158, 11, 0.1)';
        statusEl.style.borderColor = 'rgba(245, 158, 11, 0.2)';
    }
}
