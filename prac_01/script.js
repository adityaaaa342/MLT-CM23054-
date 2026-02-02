let model;
let chart;

// Synthetic training data
// [experience, role] → salary (LPA)
const trainingInputs = [
    [1,1], [2,1], [3,1],
    [1,2], [2,2], [3,2],
    [2,3], [3,3], [4,3],
    [3,4], [4,4], [5,4]
];

const trainingOutputs = [
    2, 2.5, 3,
    3, 4, 5,
    5.5, 6.5, 7.5,
    7, 8.5, 10
];

// Verify TensorFlow.js
console.log("TensorFlow.js version:", tf.version.tfjs);

async function trainModel() {

    console.log("Training model...");

    const xs = tf.tensor2d(trainingInputs);
    const ys = tf.tensor2d(trainingOutputs, [trainingOutputs.length, 1]);

    model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2]
    }));

    model.compile({
        optimizer: tf.train.sgd(0.01),
        loss: 'meanSquaredError'
    });

    await model.fit(xs, ys, {
        epochs: 500,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch % 100 === 0) {
                    console.log(`Epoch ${epoch} | Loss: ${logs.loss}`);
                }
            }
        }
    });

    console.log("Model training completed.");

    xs.dispose();
    ys.dispose();
}

async function predictSalary() {

    const experience = Number(document.getElementById("experience").value);
    const role = Number(document.getElementById("role").value);

    if (isNaN(experience) || isNaN(role)) {
        document.getElementById("result").innerText =
            "Please enter valid input.";
        return;
    }

    console.log("User Input:");
    console.log("Experience:", experience);
    console.log("Role:", role);

    if (!model) {
        await trainModel();
    }

    const inputTensor = tf.tensor2d([[experience, role]]);
    const prediction = model.predict(inputTensor);

    console.log("Prediction Tensor:");
    prediction.print();

    const salary = prediction.dataSync()[0];

    console.log(`Predicted Salary ≈ ₹${salary.toFixed(2)} LPA`);

    document.getElementById("result").innerText =
        `Predicted Salary ≈ ₹${salary.toFixed(2)} LPA`;

    drawChart(experience, salary);

    inputTensor.dispose();
    prediction.dispose();
}

function drawChart(exp, salary) {

    const ctx = document.getElementById("chart").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: "Training Data",
                    data: trainingInputs.map((v, i) => ({
                        x: v[0],
                        y: trainingOutputs[i]
                    })),
                    backgroundColor: "rgba(54, 162, 235, 0.7)"
                },
                {
                    label: "Predicted Salary",
                    data: [{ x: exp, y: salary }],
                    backgroundColor: "red",
                    pointRadius: 7
                }
            ]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Experience (Years)"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Salary (LPA)"
                    }
                }
            }
        }
    });
}