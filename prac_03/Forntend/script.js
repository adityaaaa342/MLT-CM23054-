/**
 * Sentiment Analysis Frontend Script
 * Handles user interaction and communicates with backend
 */

// Check if TensorFlow.js is loaded properly
console.log("TensorFlow.js loaded:", tf ? "Yes" : "No");

// Backend API URL (Change this if needed)
const API_URL = "http://localhost:8000";

/**
 * Loads example text into the input box
 * @param {string} text - Example text to load
 */
function loadExample(text) {
    document.getElementById('textInput').value = text;
    showMessage("Example loaded! Click 'Analyze Sentiment' to check.", "info");
}

/**
 * Shows temporary message to user
 * @param {string} message - Message to display
 * @param {string} type - Type of message (info/success/error)
 */
function showMessage(message, type = "info") {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.innerHTML = `<i class="fas fa-info-circle"></i> ${message}`;
    
    document.querySelector('.input-section').appendChild(messageDiv);
    
    // Remove message after 3 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
}

/**
 * Main function to analyze sentiment
 * Steps:
 * 1. Get user input
 * 2. Validate input
 * 3. Show loading state
 * 4. Send to backend API
 * 5. Display result
 */
async function analyzeSentiment() {
    // Step 1: Get user input
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    // Step 2: Validate input
    if (!text) {
        showMessage("Please enter some text to analyze!", "error");
        textInput.focus();
        return;
    }
    
    if (text.length < 3) {
        showMessage("Text is too short. Please enter at least 3 characters.", "error");
        return;
    }
    
    // Step 3: Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;
    
    const resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = `
        <div class="loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>AI is analyzing your text...</p>
        </div>
    `;
    resultContainer.className = '';
    
    try {
        // Step 4: Send to backend API
        console.log("Sending text to backend:", text);
        
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });
        
        // Check if response is OK
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        // Parse JSON response
        const data = await response.json();
        console.log("Received response:", data);
        
        // Step 5: Display result
        displayResult(data.sentiment, text);
        
    } catch (error) {
        // Handle errors
        console.error("Error:", error);
        
        resultContainer.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Failed to analyze sentiment.</p>
                <p class="error-detail">Error: ${error.message}</p>
                <p class="hint">Make sure backend server is running on ${API_URL}</p>
            </div>
        `;
        resultContainer.className = 'error';
        
        showMessage("Connection error. Check if backend is running.", "error");
    } finally {
        // Reset button state
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
    }
}

/**
 * Displays the sentiment analysis result
 * @param {string} sentiment - "Positive" or "Negative"
 * @param {string} text - Original input text
 */
function displayResult(sentiment, text) {
    const resultContainer = document.getElementById('resultContainer');
    
    // Set CSS class based on sentiment
    resultContainer.className = sentiment.toLowerCase();
    
    // Create result HTML
    const icon = sentiment === 'Positive' ? 
        '<i class="fas fa-smile-beam"></i>' : 
        '<i class="fas fa-frown"></i>';
    
    const color = sentiment === 'Positive' ? '#28a745' : '#dc3545';
    
    resultContainer.innerHTML = `
        <div class="result-content">
            <div class="sentiment-icon" style="font-size: 4rem; color: ${color}; margin-bottom: 20px;">
                ${icon}
            </div>
            <div class="sentiment-text" style="font-size: 2.5rem; margin-bottom: 10px;">
                ${sentiment.toUpperCase()}
            </div>
            <div class="input-preview" style="font-size: 1rem; color: #666; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <strong>Your text:</strong> "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"
            </div>
            <div class="confidence" style="margin-top: 15px; font-size: 0.9rem; color: #777;">
                <i class="fas fa-robot"></i> Analyzed using AI Model
            </div>
        </div>
    `;
    
    // Show success message
    showMessage(`Analysis complete! Result: ${sentiment}`, "success");
}

/**
 * Simple TensorFlow.js demonstration
 * Shows we can use ML in browser (optional feature)
 */
function demonstrateTensorFlow() {
    if (tf) {
        // Create a simple tensor to show TensorFlow.js is working
        const tensorExample = tf.tensor2d([[1, 2], [3, 4]]);
        console.log("TensorFlow.js is working! Created tensor:", tensorExample.toString());
        
        // Clean up
        tensorExample.dispose();
    }
}

// Initialize when page loads
window.onload = function() {
    demonstrateTensorFlow();
    
    // Allow pressing Enter to analyze (Shift+Enter for new line)
    document.getElementById('textInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeSentiment();
        }
    });
    
    console.log("Sentiment Analysis App initialized!");
    console.log("Backend URL:", API_URL);
};