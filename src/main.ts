import "./style.css";
import rawPreTrainModel from "./rawPreTrainedModel";
import usingPreTrainedMLmodelFromTFhub from "./usingPreTrainedMLmodelFromTFhub";
import traningNeuron from "./traningNeuron";
import trainingMultipleNeurons from "./trainingMultipleNeurons";
import multilayerPreceptrons from "./multilayerPerceptrons";
import classificationProblem from "./classificationProblem";
import usingMobileNetModel from "./mobileNet";

// rawPreTrainModel();
// usingPreTrainedMLmodelFromTFhub();
// traningNeuron();
// multilayerPreceptrons();
// trainingMultipleNeurons();
// classificationProblem();
// usingMobileNetModel();

import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();
model.add(tf.layers.flatten({ inputShape: [26, 25, 1] })); // Flatten the input tensor
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 3, activation: "softmax" }));
model.summary();
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});
async function call() {
  //   const input = tf.zeros([1, 28, 28, 3], "float32");
  const cat1 = await loadImageAsTensor("/cat.jpg");
  const cat2 = await loadImageAsTensor("/cat2.jpg");
  const snake = await loadImageAsTensor("/snake.jpg");
  const snake2 = await loadImageAsTensor("/snake2.jpg");
  const snake3 = await loadImageAsTensor("/snake3.jpg");
  const cat3 = await loadImageAsTensor("/cat3.jpg");
  const squirrel = await loadImageAsTensor("/squirrel.jpg");
  const parrot = await loadImageAsTensor("/parrot.jpg");
  const woman = await loadImageAsTensor("/woman.jpg");
  const input = tf.stack([cat1, parrot, snake2, cat2, snake]).div(255);
  const labels = ["cat", "parrot", "snake"];
  //   const targetData = tf.tensor2d([cat, notCat, cat, notCat, notCat]);
  const targetData = tf.oneHot(tf.tensor1d([0, 1, 2, 0, 2], "int32"), 3);
  model
    .fit(input, targetData, {
      batchSize: 2,
      epochs: 10,
      shuffle: true,
      callbacks: {
        onEpochEnd: (num, log) => {
          console.log("Epoch ", num, " : ", log);
        },
      },
    })
    .then(async (res) => {
      console.log(res.history);
      const test = tf.zeros([1, 26, 25, 3]);
      let higest = 0;
      const r = (await model.predict(tf.stack([snake3])).array())[0];
      console.log(r);
      r.forEach((prob, index) => {
        if (prob > higest) higest = index;
      });
      console.log(
        "Predicted ",
        labels[higest],
        " With ",
        (r[higest] * 100).toFixed(2),
        "% Confidence"
      );
    });
}
call();
async function loadImageAsTensor(imageUrl: string) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.src = imageUrl;

    image.onload = () => {
      const tensor = tf.browser.fromPixels(image);
      const grayscaleTensor = tf.image.rgbToGrayscale(tensor);
      resolve(grayscaleTensor);
    };
  });
}
