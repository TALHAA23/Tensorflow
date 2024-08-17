import * as tf from "@tensorflow/tfjs";
const STATUS = document.getElementById("status") as HTMLParagraphElement;
const VIDEO = document.getElementById("webcam") as HTMLVideoElement;
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
const TRAIN_BUTTON = document.getElementById("train");
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES: string[] = [];

let model: undefined | tf.LayersModel;
let dataCollectionButtons = document.querySelectorAll("button.dataCollector");
let mobilenet: undefined | tf.GraphModel = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs: tf.Tensor[] = [];
let trainingDataOutputs: number[] = [];
let examplesCount: number[] = [];
let predict = false;

dataCollectionButtons.forEach((button: Element, index: number) => {
  button.addEventListener("mousedown", gatherDataforClass);
  button.addEventListener("mouseup", gatherDataforClass);
  CLASS_NAMES.push(button.getAttribute("data-name"));
});
ENABLE_CAM_BUTTON?.addEventListener("click", enableCam);
TRAIN_BUTTON?.addEventListener("click", trainAndPredict);
function gatherDataforClass() {
  if (!videoPlaying) {
    alert("Please Enable Cam!");
    return;
  }
  let classNumber = parseInt(this.getAttribute("data-1hot"));
  gatherDataState =
    gatherDataState == STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

function dataGatherLoop() {
  if (!videoPlaying || gatherDataState === STOP_DATA_GATHER) return;
  let imageFeatures = tf.tidy(function () {
    const videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
    let resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    let normalizedTensorFrame = resizedTensorFrame.div(255);
    return mobilenet?.predict(normalizedTensorFrame.expandDims()).squeeze();
  });
  trainingDataInputs.push(imageFeatures);
  trainingDataOutputs.push(gatherDataState);

  if (examplesCount[gatherDataState] == undefined)
    examplesCount[gatherDataState] = 0;

  examplesCount[gatherDataState]++;

  STATUS.innerHTML = "";
  CLASS_NAMES?.map(
    (item, index) =>
      (STATUS.innerHTML += item + " data count: " + examplesCount[index] + ".")
  );
  window.requestAnimationFrame(dataGatherLoop);
}

async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  const outputAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  const oneHotOutputs = tf.oneHot(outputAsTensor, CLASS_NAMES.length);
  const inputAsTensor = tf.stack(trainingDataInputs);
  const result = await model?.fit(inputAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
  });
  outputAsTensor.dispose();
  oneHotOutputs.dispose();
  inputAsTensor.dispose();
  predict = true;
  predictLoop();
}

function predictLoop() {
  if (!predict) return;

  tf.tidy(function () {
    const videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
    const resizedTensor = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    const imageFeatures = mobilenet?.predict(resizedTensor.expandDims());
    const prediction = model?.predict(imageFeatures).squeeze();
    const highestIndex = prediction.argMax().arraySync();
    const predictionArray = prediction.arraySync();
    STATUS.innerHTML = "";
    predictionArray.map(
      (prediction, index) =>
        (STATUS.innerHTML +=
          (index == 0 ? "Class 1 with " : "Class 2 with ") +
          (Math.floor(prediction * 100) || 0) +
          " % Confidence")
    );
    // STATUS.innerHTML =
    //   "Prediction: " +
    //   CLASS_NAMES[highestIndex] +
    //   " with " +
    //   Math.floor(predictionArray[highestIndex] * 100) +
    //   " % Confidence";
  });
  window.requestAnimationFrame(predictLoop);
}

async function loadMobilenetFeaturesModel() {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
  const cachedModel = await tf.loadGraphModel("indexeddb://tf/mobile-net");
  mobilenet =
    cachedModel || (await tf.loadGraphModel(URL, { fromTFHub: true }));
  !cachedModel && mobilenet.save("indexeddb://tf/mobile-net");
  STATUS.innerHTML = "Model loaded!";
  return mobilenet;
}

function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
  );
  model.summary();
  model.compile({
    optimizer: "adam",
    loss:
      CLASS_NAMES.length == 2
        ? "binaryCrossentropy"
        : "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  model.summary();
  return model;
}
export default async function usingMobileNetModel() {
  model = createModel();
  await loadMobilenetFeaturesModel();
}

// webcam

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam(event: MouseEvent) {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640,
      height: 480,
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      if (!VIDEO) return;
      VIDEO.srcObject = stream;
      VIDEO.addEventListener("loadeddata", function () {
        videoPlaying = true;
        ENABLE_CAM_BUTTON?.classList.add("removed");
      });
    });
  } else {
    console.warn("Media not supported");
  }
}
