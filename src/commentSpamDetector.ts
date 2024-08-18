import * as tf from "@tensorflow/tfjs";
import * as DICTIONARY from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/dictionary.js";
const POST_COMMENT_BTN = document.getElementById("post");
const COMMENT_TEXT = document.getElementById("comment");
const CONMENTS_LIST = document.getElementById("commentList");
const PROCESSING_CLASS = "processing";
const MODEL_URL =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/model.json";
const SPAM_THRESHOLD = 0.75;

export default async function detectCommentSpam() {}

function handleCommentPost() {
  if (POST_COMMENT_BTN?.classList.contains(PROCESSING_CLASS)) return;
  //   POST_COMMENT_BTN?.classList.add(PROCESSING_CLASS);
  COMMENT_TEXT?.classList.add(PROCESSING_CLASS);
  const currentComment = COMMENT_TEXT?.innerText;
  const lowercaseSentenceArray = currentComment
    ?.toLowerCase()
    ?.replace(/[^\w\s]/g, "")
    ?.split(" ") || [""];
  const li = document.createElement("li");
  const p = document.createElement("p");
  p.innerHTML = COMMENT_TEXT?.innerText || "";
  const spanName = document.createElement("span");
  spanName.setAttribute("class", "username");
  spanName.innerHTML = "Username";
  const spanDate = document.createElement("span");
  spanDate.setAttribute("class", "timestamp");
  const curDate = new Date();
  spanDate.innerHTML = curDate.toLocaleString();
  [spanName, spanDate, p].forEach((child) => li.appendChild(child));
  CONMENTS_LIST?.prepend(li);

  //   COMMENT_TEXT?.innerHTML = "";
  loadAndPredictModel(tokenize(lowercaseSentenceArray), li).then(() => {
    POST_COMMENT_BTN?.classList.remove(PROCESSING_CLASS);
    COMMENT_TEXT?.classList.remove(PROCESSING_CLASS);
  });
}

POST_COMMENT_BTN?.addEventListener("click", handleCommentPost);

async function loadAndPredictModel(
  input: tf.Tensor,
  domComment: Element
): Promise<tf.LayersModel> {
  const cachedVersion = await tf
    .loadLayersModel("localstorage://tf/spam-detector")
    .then((res) => res)
    .catch((err) => {
      console.warn("Can't load model from local storage");
    });
  const model = cachedVersion || (await tf.loadLayersModel(MODEL_URL));
  !cachedVersion && model.save("localstorage://tf/spam-detector");
  const result = model.predict(input);
  result.print();
  let dataArray = result.dataSync();
  input.print();
  console.log("Spam value: ", dataArray[1]);
  if (dataArray[1] > SPAM_THRESHOLD) domComment.classList.add("spam");
  return model;
}

// ? Load Dictionary
const ENCODING_LENGTH = 20;
function tokenize(wordArray: string[]) {
  let returnArray = [DICTIONARY.START];
  wordArray.forEach((word) => {
    let encoding = DICTIONARY.LOOKUP[word];
    returnArray.push(encoding || DICTIONARY.UNKNOWN);
  });
  while (returnArray.length < ENCODING_LENGTH) returnArray.push(DICTIONARY.PAD);
  return tf.tensor2d([returnArray]);
}
