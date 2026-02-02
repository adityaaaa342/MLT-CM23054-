async function runModel() {

  document.getElementById("result").innerHTML = "⏳ Training in progress...";

 
  const xValues = [];
  const yValues = [];

  for (let x = -10; x <= 10; x += 0.5) {
    xValues.push(x);
    yValues.push(2 * x + 1 + Math.random() * 2);
  }

  const xs = tf.tensor2d(xValues, [xValues.length, 1]);
  const ys = tf.tensor2d(yValues, [yValues.length, 1]);

  
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

 
  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: 'meanSquaredError'
  });

 
  await model.fit(xs, ys, {
    epochs: 100,
    verbose: 0
  });

  
  const weights = model.getWeights();
  const weight = weights[0].dataSync()[0].toFixed(3);
  const bias = weights[1].dataSync()[0].toFixed(3);

  document.getElementById("result").innerHTML = `
    ✅ <b>Training Complete</b><br>
    📌 Learned Weight (m): <b>${weight}</b><br>
    📌 Learned Bias (b): <b>${bias}</b>
  `;

  xs.dispose();
  ys.dispose();
}
