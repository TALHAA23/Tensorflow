import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();
model.add(tf.layers.flatten({ inputShape: [26, 25, 1] })); // Flatten the input tensor
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 2, activation: "softmax" }));
model.summary();
model.compile({
  optimizer: "adam",
  loss: "binaryCrossentropy",
  metrics: ["accuracy"],
});
async function call() {
  //   const input = tf.zeros([1, 28, 28, 3], "float32");
  const imageInput = await loadImageAsTensor("/cat.jpg");
  const imageInput2 = await loadImageAsTensor("/cat2.jpg");
  const imageInput3 = await loadImageAsTensor("/snake.jpg");
  const imageInput4 = await loadImageAsTensor("/cat3.jpg");
  const squirrel = await loadImageAsTensor("/squirrel.jpg");
  const parrot = await loadImageAsTensor("/parrot.jpg");
  const woman = await loadImageAsTensor("/woman.jpg");
  const input = tf
    .stack([imageInput, parrot, imageInput2, squirrel, imageInput3])
    .div(255);
  const cat = [1, 0];
  const notCat = [0, 1];
  const targetData = tf.tensor2d([cat, notCat, cat, notCat, notCat]);
  model
    .fit(input, targetData, {
      batchSize: 3,
      epochs: 25,
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
      const r = (await model.predict(tf.stack([woman])).array())[0];
      console.log(r);
      const higest = r[0] > r[1] ? 0 : 1;
      const labels = ["Is a cat", "Not a cat"];
      console.log(labels[higest]);
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
