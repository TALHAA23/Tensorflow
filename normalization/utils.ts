// This is a practice work on how to normalize value to range 0-1
import * as tf from "@tensorflow/tfjs";

function prac() {
  const inputs = tf.tensor2d([
    [34, 55],
    [1, 2],
    [50, 0],
    [100, 1200],
  ]);
  const outputs = tf.tensor1d([30, 10, 40]);
  const mininput = tf.min(inputs, 0);
  const maxinput = tf.max(inputs, 0);
  const subtractedValues = tf.sub(inputs, mininput);
  const range = tf.sub(maxinput, mininput);
  const normailzed = tf.div(inputs, range);
  normailzed.print();
}
