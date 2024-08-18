import "./style.css";
import rawPreTrainModel from "./rawPreTrainedModel";
import usingPreTrainedMLmodelFromTFhub from "./usingPreTrainedMLmodelFromTFhub";
import traningNeuron from "./traningNeuron";
import trainingMultipleNeurons from "./trainingMultipleNeurons";
import multilayerPreceptrons from "./multilayerPerceptrons";
import classificationProblem from "./classificationProblem";
import usingMobileNetModel from "./mobileNet";
import detectCommentSpam from "./commentSpamDetector";

// rawPreTrainModel();
// usingPreTrainedMLmodelFromTFhub();
// traningNeuron();
// multilayerPreceptrons();
// trainingMultipleNeurons();
// classificationProblem();
// usingMobileNetModel();
detectCommentSpam();

import * as qna from "@tensorflow-models/qna";

const passage =
  "We belive that Cats are the real reason for revenue of youtube, dog comes on number 2";
const question = "Who is important to youtube";

async function call() {
  const model = await qna.load();
  const answers = await model.findAnswers(question, passage);
  console.log(answers);
}
