import * as tf from "@tensorflow/tfjs";
import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";
export default async function classificationProblem() {
  const INPUTS = TRAINING_DATA.inputs;
  const OUTPUTS = TRAINING_DATA.outputs;
  tf.util.shuffleCombo(INPUTS, OUTPUTS);
  const INPUT_TENSORS = tf.tensor2d(INPUTS);
  console.log(INPUT_TENSORS);
  // oneHot take number  of possible classes as 2nd paramter
  const OUTPUT_TENSORS = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

  let model;
  const haveCacheVersionofModel = localStorage.getItem(
    "tensorflowjs_models/tf/handwritten-image-classification/model_metadata"
  );
  if (haveCacheVersionofModel)
    model = await tf.loadLayersModel(
      "localstorage://tf/handwritten-image-classification"
    );
  else {
    console.log("training new model");
    model = createModel();
    //   train the model
    await train(model, INPUT_TENSORS, OUTPUT_TENSORS);
  }
  //   test the model
  evaluate(model, INPUTS, OUTPUTS);
}

function createModel() {
  // create sequential model
  const model = tf.sequential();
  // input layer with shape 784 for 28x28 dims image.
  // unit 30 is not a random number, practice is done to find the good
  model.add(tf.layers.dense({ inputShape: [784], units: 30 }));
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  // softmax activation to convert the output on range 0-1
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  return model;
}

async function train(
  model: tf.LayersModel,
  input: tf.Tensor,
  output: tf.Tensor
) {
  model.compile({
    // adam update learning rate all automatically
    optimizer: "adam",
    // Suited for image problems
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const result = await model.fit(input, output, {
    validationSplit: 0.2,
    shuffle: true,
    batchSize: 512, //update weight after every 512 examples
    epochs: 50,
    callbacks: { onEpochEnd: logProgress },
  });
  input.dispose();
  output.dispose();
  model.save("localstorage://tf/handwritten-image-classification");
  console.log(result.history);
  return result;
}

function logProgress(epoch: EpochTimeStamp, log: tf.Logs) {
  console.log("Data of epoch: " + epoch, Math.sqrt(log.loss));
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

async function evaluate(model: tf.LayersModel, inputs: any, outputs: any) {
  // take a random image from inputs
  const OFFSET = Math.floor(Math.random() * inputs.length);
  let answer = tf.tidy(function () {
    const newInput = tf.tensor1d(inputs[OFFSET]);
    let output = model.predict(newInput.expandDims());
    output.print();
    // squeeze is just as opposite of expandDims
    return output.squeeze().argMax();
  });
  answer.array().then(function (index: number) {
    if (PREDICTION_ELEMENT) {
      PREDICTION_ELEMENT.innerHTML = index.toString();
    }
    PREDICTION_ELEMENT?.setAttribute(
      "class",
      index == outputs[OFFSET] ? "correct" : "wrong"
    );
  });
  answer.dispose();
  drawImage(inputs[OFFSET]);
}

const CANVAS = document.getElementById("canvas") as HTMLCanvasElement;
const CTX = CANVAS.getContext("2d");
function drawImage(digit: number[]) {
  if (CTX) {
    const imageData = CTX.getImageData(0, 0, 28, 28);
    for (let i = 0; i < digit.length; i++) {
      imageData.data[i * 4] = digit[i] * 255;
      imageData.data[i * 4 + 1] = digit[i] * 255;
      imageData.data[i * 4 + 2] = digit[i] * 255;
      imageData.data[i * 4 + 3] = 255;
    }
    CTX.putImageData(imageData, 0, 0);
  }
}
