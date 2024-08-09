// ? When working with images in ML the order of deminsion is
// ? Height, Weight, color and not Weight, Height
import * as tf from "@tensorflow/tfjs";
const MODEL_PATH =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
const WRAPPER = document.querySelector(".image-wrapper");
const EXAMPLE_IMAGE = document.getElementById("exampleImg") as HTMLImageElement;
const PARTS = [
  "nose",
  "left eye",
  "right eye",
  "left ear",
  "right ear",
  "left shoulder",
  "right shoulder",
  "left elbow",
  "right elbow",
  "left wrist",
  "right wrist",
  "left hip",
  "right hip",
  "left knee",
  "right knee",
  "left ankle",
  "right ankle",
];
export default async function usingPreTrainedMLmodelFromTFhub() {
  const model = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });
  model.save("localstorage://tf/movenet-model");
  const imageTensor = tf.browser.fromPixels(EXAMPLE_IMAGE);
  // from where to start cropping y,x axis and 0 represent red to keep all color values
  let cropStartingPoint = [15, 170, 0];
  // next is the crop height, width and color to keep, 3 mean all RGB
  let cropSize = [345, 345, 3];
  // finally cropping
  let croppedTensor = tf.slice(imageTensor, cropStartingPoint, cropSize);
  let resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [192, 192], true)
    .toInt();
  let tensorOutput = model.predict(tf.expandDims(resizedTensor));
  let arrayOutput = await tensorOutput.array();
  console.log(arrayOutput);
  // displaying...
  arrayOutput[0][0].map((prediction: Array<number>, index: number) => {
    const [y, x, score] = prediction;
    if (score < 0.66) return;
    const pointer = document.createElement("div");
    pointer.classList.add("pointer");
    const toolTip = document.createElement("p");
    toolTip.innerHTML = PARTS[index];
    toolTip.classList.add("tool-tip");
    pointer.style.top = y * 345 + 15 + "px";
    pointer.style.left = x * 345 + 170 + "px";
    pointer.appendChild(toolTip);
    WRAPPER?.appendChild(pointer);
  });
}
