let video;
let bodyPose;
let poses = [];
let connections;

function preload() {
    // Load the bodyPose model (which uses MoveNet under the hood, standard in newer ml5)
    bodyPose = ml5.bodyPose();
}

function setup() {
    // Create the canvas and attach it to the div
    const canvas = createCanvas(640, 480);
    canvas.parent('canvas-container');

    // Create the video feed and hide it (we will draw it manually in draw loop)
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // Start detecting poses in the webcam video
    bodyPose.detectStart(video, gotPoses);
    
    // Get the skeleton connection information
    connections = bodyPose.getSkeleton();
    
    // Update status to waiting
    updateStatus('Waiting for person...', 'warning');
}

function draw() {
    // Flip the drawing context so the video looks like a mirror
    push();
    translate(width, 0);
    scale(-1, 1);
    
    // Draw the webcam video onto the canvas
    image(video, 0, 0, width, height);

    // Draw the skeleton connections
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i];
        for (let j = 0; j < connections.length; j++) {
            let pointAIndex = connections[j][0];
            let pointBIndex = connections[j][1];
            let pointA = pose.keypoints[pointAIndex];
            let pointB = pose.keypoints[pointBIndex];
            
            // Only draw a line if we have confidence in both points
            if (pointA.confidence > 0.1 && pointB.confidence > 0.1) {
                stroke(16, 185, 129); // emerald-500
                strokeWeight(4);
                line(pointA.x, pointA.y, pointB.x, pointB.y);
            }
        }
    }

    // Draw all the detected keypoints
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i];
        for (let j = 0; j < pose.keypoints.length; j++) {
            let keypoint = pose.keypoints[j];
            
            // Only draw an ellipse if the pose confidence is bigger than 0.1
            if (keypoint.confidence > 0.1) {
                fill(56, 189, 248); // sky-400
                noStroke();
                ellipse(keypoint.x, keypoint.y, 12, 12);
                
                // Add a small white inner dot for aesthetics
                fill(255);
                ellipse(keypoint.x, keypoint.y, 4, 4);
            }
        }
    }
    
    pop(); // Restore context so normal orientations aren't affected
}

// Callback function for when bodyPose outputs data
function gotPoses(results) {
    // Save the output to the poses variable
    poses = results;
    
    // Update status UI based on detection
    if (poses.length > 0) {
        updateStatus('Detection Active!', 'success');
    } else {
        updateStatus('Waiting for person...', 'warning');
    }
}

// Helper to update the UI status badge
function updateStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.innerText = message;
    
    if (type === 'success') {
        statusEl.style.color = '#10b981';
        statusEl.style.background = 'rgba(16, 185, 129, 0.1)';
        statusEl.style.borderColor = 'rgba(16, 185, 129, 0.2)';
    } else if (type === 'warning') {
        statusEl.style.color = '#f59e0b';
        statusEl.style.background = 'rgba(245, 158, 11, 0.1)';
        statusEl.style.borderColor = 'rgba(245, 158, 11, 0.2)';
    }
}
