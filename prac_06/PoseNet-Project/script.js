const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const statusIndicator = document.getElementById('status-indicator');
const overlayUi = document.querySelector('.overlay-ui');
const poseResultOverlay = document.getElementById('pose-result-overlay');
const poseLabelEl = document.getElementById('pose-label');
const poseDotEl = document.querySelector('.pose-dot');

// Premium styling for the poses
const KEYPOINT_COLOR = '#00ff00'; // vibrant neon green
const SKELETON_COLOR = '#00ff00'; // matching neon green for lines
const SKELETON_LINE_WIDTH = 4;
const KEYPOINT_RADIUS = 6;

let net;

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
    }

    // Attempt to get ideal 640x480 resolution from webcam
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
        }
    });
    
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.width = video.videoWidth;
            video.height = video.videoHeight;
            resolve(video);
        };
    });
}

function drawPoint(ctx, y, x, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    
    // Add glow effect slightly to keypoints
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.fill();
    
    // Reset shadow for next strokes
    ctx.shadowBlur = 0;
}

function drawSegment([ay, ax], [by, bx], color, ctx) {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineWidth = SKELETON_LINE_WIDTH;
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    ctx.stroke();
}

function drawSkeleton(keypoints, minConfidence, ctx) {
    const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);

    adjacentKeyPoints.forEach((points) => {
        drawSegment(
            [points[0].position.y, points[0].position.x],
            [points[1].position.y, points[1].position.x],
            SKELETON_COLOR,
            ctx
        );
    });
}

function drawKeypoints(keypoints, minConfidence, ctx) {
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }

        const { y, x } = keypoint.position;
        drawPoint(ctx, y, x, KEYPOINT_RADIUS, KEYPOINT_COLOR);
    }
}

function classifyPosture(pose) {
    const keypoints = pose.keypoints;
    const findPart = (partName) => keypoints.find(k => k.part === partName && k.score > 0.4);

    const leftHip = findPart('leftHip');
    const rightHip = findPart('rightHip');
    const leftKnee = findPart('leftKnee');
    const rightKnee = findPart('rightKnee');
    const leftShoulder = findPart('leftShoulder');
    const rightShoulder = findPart('rightShoulder');

    const hip = leftHip || rightHip;
    const knee = leftKnee || rightKnee;
    const shoulder = leftShoulder || rightShoulder;

    if (!hip || !knee) {
        return "Unknown";
    }

    // Mathematical vertical distance down the screen from hip to knee
    const verticalDistKnee = knee.position.y - hip.position.y;
    
    // Normalize distance using torso length if shoulder is visible
    if (shoulder) {
        const torsoLength = hip.position.y - shoulder.position.y;
        if (torsoLength > 0) {
            // If the hip-to-knee vertical drop is less than ~60% of torso length, 
            // the knees are pointing forward (seated), instead of straight down (standing).
            if (verticalDistKnee < torsoLength * 0.6) {
                return "Sitting";
            } else {
                return "Standing";
            }
        }
    }

    // Absolute fallback if shoulders aren't detected
    if (verticalDistKnee < 80) {
        return "Sitting";
    } else {
        return "Standing";
    }
}

async function detectPoseInRealTime() {
    // Canvas dimensions must match video dimensions for correct mapping
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    async function poseDetectionFrame() {
        const minPoseConfidence = 0.1;
        const minPartConfidence = 0.5;

        // Estimate poses with posenet
        const pose = await net.estimateSinglePose(video, {
            flipHorizontal: false // We visually handle mirroring securely via CSS transform scaleX(-1) on video & canvas
        });

        // Clear previous frame drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (pose.score >= minPoseConfidence) {
            // Draw skeleton before keypoints so points sit on top of lines
            drawSkeleton(pose.keypoints, minPartConfidence, ctx);
            drawKeypoints(pose.keypoints, minPartConfidence, ctx);

            // Classify sitting vs standing
            const posture = classifyPosture(pose);
            
            if (posture === "Sitting" || posture === "Standing") {
                poseResultOverlay.style.display = 'flex';
                poseLabelEl.innerText = posture;
                
                // Update UI colors based on posture class
                if (posture === "Sitting") {
                    poseDotEl.classList.add('sitting');
                    poseDotEl.classList.remove('standing');
                } else {
                    poseDotEl.classList.add('standing');
                    poseDotEl.classList.remove('sitting');
                }
            } else {
                poseLabelEl.innerText = "Tracking...";
                poseDotEl.className = 'pose-dot';
            }
        }

        requestAnimationFrame(poseDetectionFrame);
    }

    // Kick off animation loop
    poseDetectionFrame();
}

function updateStatus(message, isReady = false) {
    statusEl.innerText = message;
    if (isReady) {
        statusIndicator.classList.add('ready');
    } else {
        statusIndicator.classList.remove('ready');
    }
}

async function main() {
    try {
        updateStatus('Requesting camera access...');
        await setupCamera();
        video.play();

        updateStatus('Loading AI Models (PoseNet)...');
        
        // Load PoseNet model from TFJS
        net = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: 640, height: 480 },
            multiplier: 0.75
        });

        updateStatus('AI Active. Ready to track.', true);
        detectPoseInRealTime();
        
        // Hide status UI softly after it's ready so vision is unobstructed
        setTimeout(() => {
            overlayUi.style.opacity = '0';
            setTimeout(() => overlayUi.style.display = 'none', 500);
        }, 3000);

    } catch (e) {
        updateStatus('Error: ' + e.message);
        console.error(e);
        
        statusIndicator.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        statusIndicator.style.background = 'rgba(15, 23, 42, 0.9)';
        
        const spinner = document.querySelector('.spinner');
        if(spinner) spinner.style.display = 'none';
        
        statusIndicator.classList.add('error');
        const style = document.createElement('style');
        style.innerHTML = `
            .status-indicator.error::before {
                content: ''; display: block; width: 10px; height: 10px;
                background: #ef4444; border-radius: 50%; box-shadow: 0 0 10px #ef4444;
            }
        `;
        document.head.appendChild(style);
    }
}

// Start sequence globally when DOM is loaded
window.addEventListener('load', main);
