//? Feed to model
import * as tf from "@tensorflow/tfjs";
const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";

export default async function rawPreTrainModel() {
  // load layer model
  const loadCacheModel = await tf.loadLayersModel(
    "localstorage://tf/sqftToPropertyPriceModel"
  );
  const model = loadCacheModel || (await tf.loadLayersModel(MODEL_PATH));
  //   save model locally to use without web req
  await model.save("localstorage://tf/sqftToPropertyPriceModel");
  //   see summery info like model type, tensor stucture
  model.summary();

  //   create single value 2d tensor
  const input = tf.tensor2d([[870]]);

  //   create batch of input to work on simultaneously
  const inputBatch = tf.tensor2d([[1], [500], [1100]]);

  // execute model
  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  //   print result
  result.print();
  resultBatch.print();

  //  dispose result

  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
  model.dispose();
}
