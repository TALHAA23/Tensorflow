import * as tf from "@tensorflow/tfjs";
const model = tf.sequential();
/**
 * Input shape [1,1] mean [null,1,1] which mean any number of sample each of 1x1 matrix
 * if it could be [1] will meant any number of sample with 1 element long vector i.e 1d
 */
model.add(tf.layers.dense({ inputShape: [1, 1], units: 1 }));
// model.summary();
model.compile({
  optimizer: tf.train.sgd(0.001),
  loss: "meanSquaredError",
});
/**
 * The input tensor should be one lvl higer than the actual data requiement
 * here model require 1x1 matrix which can be done using 2d array but we use 3d
 * that's because we have to count the batch size also
 * 2d tesor will create data but not batch side
 * so we use one rank higer tesnsor to do so
 * atlternativly we can use reshape, stack like methods also
 */
const input = tf.tensor3d([[[1]]]);
console.log("input shape: ", input.shape);
console.log("input size: ", input.size);
// if input is 3d output should also be 3d
const output = tf.tensor3d([[[2]]]);
console.log("output size: ", output.size);

model.fit(input, output).then((res) => {
  console.log(res.history);
});
