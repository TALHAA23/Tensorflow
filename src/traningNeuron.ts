// Access dataset
import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";
import * as tf from "@tensorflow/tfjs";
// ? Define model architecture
// srquential: output of one layer is used as input to other
// ? For now its just one layer
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));
model.summary();

export default async function traningNeuron() {
  const INPUTS = TRAINING_DATA.inputs;
  const OUTPUTS = TRAINING_DATA.outputs;
  //   shuffle to discard orders if any, while keep the ipnuts to its corrosponding output.
  tf.util.shuffleCombo(INPUTS, OUTPUTS);
  //   convert JS Array to tensor work with it effectively
  const INPUT_TENSOR = tf.tensor2d(INPUTS);
  INPUT_TENSOR.print();
  const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS);
  const FEATURE_RESULT = normalize(INPUTS);
  INPUT_TENSOR.dispose(); //not in use

  // ? Traning...
  //   ! Here model is always train but we train model only once, save it and use it.
  // rate at which weight and bies changes.
  // not a random value, practice with data to pick a value that train the model fast
  const LEARNING_RATE = 0.01;

  // compile the model to use the details
  model.compile({
    // mathmathical model to train the model
    optimizer: tf.train.sgd(LEARNING_RATE), //sort of back propagation which change weight and baises base on learning rate and loss value
    // declare your own loss fn or use on from tf.
    loss: "meanSquaredError",
  });

  // traning and result
  const result = await model.fit(
    FEATURE_RESULT.NORMALIZED_VALUES,
    OUTPUT_TENSOR,
    {
      validationSplit: 0.15, //data to keep for validation (15%) use only when having larger dataset
      shuffle: true, //shuffle the data so it does not use it in an order which may cause problems
      batchSize: 64, //number after which baises and weights are updated. 1 mean update it for every next tuple
      epochs: 10, //go over the data 10 times
    }
  );

  OUTPUT_TENSOR.dispose();
  FEATURE_RESULT.NORMALIZED_VALUES.dispose();
  console.log(
    "avg error loss:" +
      Math.sqrt(result.history.loss[result.history.loss.length - 1])
  );
  console.log(
    "avg validation error loss:" +
      Math.sqrt(result.history.val_loss[result.history.val_loss.length - 1])
  );
  //   TODO: Downlaod to computer: await model.save("downloads://my-model");
  //   evulation - try the model with data
  tf.tidy(function () {
    let newInput = normalize(
      tf.tensor2d([[750, 1]]), //one house
      FEATURE_RESULT.MIN_VALUES,
      FEATURE_RESULT.MAX_VALUES
    );
    let output = model.predict(newInput.NORMALIZED_VALUES);
    console.log("print: "), output.print();
  });

  FEATURE_RESULT.MIN_VALUES.dispose();
  FEATURE_RESULT.MAX_VALUES.dispose();
  model.dispose();
}

// normalize the value to use range 0-1
function normalize(
  tensor: tf.Tensor2D | tf.Tensor1D,
  min?: tf.Tensor<tf.Rank>,
  max?: tf.Tensor<tf.Rank>
) {
  // use tidy to automatically dispose unuse tensors
  const result = tf.tidy(function () {
    // find min-max so we don't search for it in every tuple
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);
    // subtract min values from each tensor in 2d tensor
    const TENSOR_SUBTRACT_MIN_VALUES = tf.sub(tensor, MIN_VALUES);
    // - ???
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    // Calculating normailzed value...
    // values with range  0-1
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUES, RANGE_SIZE);
    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });
  return result;
}
