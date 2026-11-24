// Builds report/outline.docx: the outline of the final report (sections of the course handout).
//   NODE_PATH=<folder with node_modules> node report/build/outline.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel, AlignmentType,
  LevelFormat, WidthType, ShadingType, BorderStyle,
} = require("docx");

const FONT = "Calibri";
const INK = "0B0B0B", MUTED = "52514E", RULE = "BFBDB5", TINT = "F1F0EC";

const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const p = (text, opts = {}) => new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text, ...opts })] });
const note = (text) => new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text, italics: true, color: MUTED })] });
/** A bullet; `runs` is a string or [[text, {bold}], ...]. */
const b = (runs, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 50 },
  children: (typeof runs === "string" ? [[runs]] : runs).map(([text, o]) => new TextRun({ text, ...(o || {}) })),
});
const B = { bold: true };

function table(widths, rows, header = true) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((cells, r) => new TableRow({
      tableHeader: header && r === 0,
      children: cells.map((text, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        borders,
        margins: { top: 50, bottom: 50, left: 90, right: 90 },
        shading: header && r === 0 ? { fill: TINT, type: ShadingType.CLEAR, color: "auto" } : undefined,
        children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: header && r === 0, size: 20 })] })],
      })),
    })),
  });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Uncertainty-Aware Activity Recognition")] }),
  p("Final report outline. CompE 510, Fall 2026. Robert Ashe and Sirena Rocca.", { color: MUTED }),
  note("Target: 5 to 7 pages, single column, 1 inch margins, 11 pt. Sections and points follow the course handout. Numbers are from five subject folds and three background divisions (15 runs per row), one seed. Items marked TO DO are not done yet."),

  h1("1. Introduction (5 points, about half a page)"),
  b("A wrist-worn exercise recognizer is trained on a fixed list of exercises. In use it also sees every other movement its wearer makes."),
  b("A classifier must answer with one of its classes. On an unfamiliar movement it is often confidently wrong, which gives wrong workout logs and rep counts."),
  b([["Goal: ", B], ["make the recognizer able to say \"unknown\", and compare four ways of measuring its uncertainty."]]),
  b([["What we did: ", B], ["an open-set benchmark on RecoFit (four known exercises, unseen people, unseen exercises); four methods in three training setups; a live app driven by our own wearable."]]),
  b([["Main finding: ", B], ["with uncertainty alone, 60 to 77% of unknown-exercise windows are still accepted. Teaching the models to be uncertain on other movements brings this to 5 to 6% at the same known accuracy (0.96)."]]),

  h1("2. Related work (7 points, about three quarters of a page)"),
  h2("a. Summary of the literature"),
  b([["Morris et al. (2014), RecoFit. ", B], ["An arm-worn inertial sensor that finds exercise periods, recognizes the exercise and counts repetitions. The source of our dataset."]]),
  b([["Hendrycks and Gimpel (2017). ", B], ["The largest softmax probability as a baseline for detecting misclassified and out-of-distribution inputs. Our softmax and logistic-regression score."]]),
  b([["Gal and Ghahramani (2016), MC Dropout. ", B], ["Dropout kept on at test time approximates Bayesian inference; the spread over repeated passes measures model uncertainty."]]),
  b([["Sensoy et al. (2018), evidential deep learning. ", B], ["The network outputs evidence for a Dirichlet distribution over the classes, so uncertainty comes from a single pass."]]),
  b([["Hendrycks et al. (2019), outlier exposure. ", B], ["Training on auxiliary outlier data with a uniform target improves detection of unseen inputs. The basis of our \"taught\" models."]]),
  b([["Huo et al. (2020). ", B], ["Uncertainty quantification for mobile activity recognition and discovery of unknown contexts. TO DO: read before describing its method or results."]]),
  h2("b. How our approach differs"),
  b("Four methods under one protocol on exercise data: two taught in class and two new ones."),
  b("The unknowns are real held-out exercises and non-exercise movement from people not seen in training, not a different dataset."),
  b("We separate \"uncertainty alone\" from \"uncertainty taught with examples of other movements\", and report an explicit \"Other\" class as a reference."),
  b("The models run live on a wearable we built."),
  note("TO DO (both): confirm each statement about prior work against the papers before it goes into the report."),

  h1("3. Problem formulation (10 points, about half a page)"),
  b([["Given: ", B], ["2 s windows of six-axis wrist motion, x in R^(6 x 100) (accelerometer and gyroscope at 50 Hz), with labels from K = 4 known exercises, for the training people. Optionally, unlabelled \"background\" windows of other movement from the same people."]]),
  b([["Goal: ", B], ["a function that gives each window a label in {1, ..., K} and an uncertainty score u(x), and answers \"unknown\" when u(x) is above a threshold tau."]]),
  b([["Constraints: ", B], ["test people are never seen in training; the unknown test exercises are never seen in any form; tau is set on validation data only, so that 95% of known validation windows are kept; one decision per 2 s window, fast enough for live use."]]),
  b([["Measures: ", B], ["known accuracy; share of unknown-exercise windows accepted (headline); share of non-exercise windows accepted; share of known windows kept. AUROC as a threshold-free check."]]),

  h1("4. Methods (10 points, about one and a quarter pages)"),
  b([["Inputs. ", B], ["The three networks share one small 1-D convolutional network. Logistic regression uses 102 handcrafted features per window."]]),
  gap(),
  table([2100, 1500, 2880, 2880], [
    ["Method", "Status", "How it predicts", "Uncertainty score"],
    ["Logistic regression", "taught in class", "linear model on features", "1 - highest class probability"],
    ["Softmax network", "taught in class", "CNN with softmax output", "1 - highest class probability"],
    ["MC Dropout (Sirena)", "new", "30 passes with dropout on, averaged", "entropy of the average; mutual information"],
    ["Evidential network, EDL (Robert)", "new", "CNN outputs evidence for a Dirichlet", "vacuity = classes / total evidence"],
  ]),
  gap(),
  b([["Rejection rule. ", B], ["A window is \"unknown\" when its score is above tau."]]),
  b([["Three training setups. ", B], ["Plain: the four exercises only. Taught: the same, plus background windows with the target \"be uncertain\" (uniform output for softmax, zero evidence for EDL; MC Dropout runs on the taught softmax network). Other: background windows as a fifth class (reference only)."]]),
  b([["EDL details. ", B], ["Expected cross-entropy loss with a KL regularizer whose weight ramps up over the first epochs. The squared-error loss of the paper did not train on our data; say so in one sentence."]]),
  b([["Fine-tuning on our own recordings. ", B], ["Convolution layers frozen, input normalization and output layers re-fitted. TO DO: include only if the recording session gives results."]]),
  note("TO DO: Robert writes the EDL paragraph and Sirena the MC Dropout paragraph in their own words."),

  h1("5. Data description and experimental setup (10 points, about one page)"),
  h2("Dataset"),
  b("RecoFit (Microsoft Research, public): arm-worn accelerometer and gyroscope at 50 Hz, 94 people, labelled gym exercises plus non-exercise periods."),
  b("We keep exercises with at least 25 people and 10 minutes of data: 38 classes. Windows are 2 s with a 1 s step."),
  b("Known: dumbbell curl, shoulder press, lateral raise, jumping jacks. Two same-exercise variants are left out. Of the other 32 exercises, 16 are background (available in training) and 16 are held out as unknown; this division is drawn three times."),
  b("Exploratory findings worth one sentence each: the two RecoFit files contain the same exercise data twice; the people form four groups with different exercise lists, so folds are balanced by exercise; some exercises appear under two labels."),
  note("TO DO: one small table of window counts (train, validation, test known, test unknown, non-exercise) from docs/data-pipeline.md."),
  h2("Setup"),
  b("Five folds by person: about 63 train, 12 validation and 19 test people per fold. No person is in two parts."),
  b("Networks: AdamW, learning rate 0.001, batch 256, up to 60 epochs with early stopping, class-weighted loss. Background: 8,000 exercise and 4,000 non-exercise windows per run."),
  b("Taught models keep the epoch with the best validation score that counts both correct labels and rejected background windows."),
  b("15 runs per result (5 folds x 3 divisions), one seed, CPU only."),
  b("Our wearable: ESP32-S3 with a six-axis IMU at 50 Hz over Bluetooth. TO DO: recording session (four exercises, rest, unknown movements, two people)."),

  h1("6. Experimental results and analyses (20 points, about one and a half pages)"),
  table([1500, 2500, 1500, 1930, 1930], [
    ["Setup", "Method (score)", "Known accuracy", "Unknown exercises accepted", "Non-exercise accepted"],
    ["Plain", "Logistic regression", "0.957", "60%", "54%"],
    ["Plain", "Softmax", "0.955", "65%", "74%"],
    ["Plain", "MC Dropout (mutual information)", "0.955", "77%", "86%"],
    ["Plain", "EDL (vacuity)", "0.961", "63%", "67%"],
    ["Taught", "Softmax", "0.962", "6.3%", "2.9%"],
    ["Taught", "MC Dropout (entropy)", "0.962", "6.1%", "2.7%"],
    ["Taught", "EDL (vacuity)", "0.956", "5.2%", "2.1%"],
    ["Other class", "Softmax", "0.958", "5.1%", "2.6%"],
  ]),
  gap(),
  h2("Findings, in the order to argue them"),
  b("All methods recognize the four known exercises equally well (0.96) on new people."),
  b("Plain models accept most unknown movement. Reason: a four-class model divides all inputs among its four classes and is confident far from its training data."),
  b("Taught models accept 5 to 6%, in every one of the 15 runs lower than their plain versions."),
  b("Taught models match the explicit \"Other\" class, while the rejection is still made by the uncertainty score."),
  b("MC Dropout's mutual information fails when taught (97% accepted): the passes agree on \"unsure\", so there is no disagreement to measure. Entropy of the average works."),
  b("Cost: the taught models keep about 88 to 89% of known windows on new people (plain: 93%)."),
  b("Harder version, 28 known exercises (one fold): accuracy 0.64 to 0.68 and 84 to 93% of unknown windows accepted. Many similar classes make the threshold loose."),
  h2("Strengths and weaknesses"),
  table([2100, 3630, 3630], [
    ["Method", "Strength", "Weakness"],
    ["Logistic regression", "fast, simple, close to the networks on known accuracy", "no taught version; accepts most unknowns"],
    ["Softmax", "simple; works well once taught", "confident on unknowns when plain"],
    ["MC Dropout", "no retraining; best calibrated on the 28-class task", "30 passes per window; mutual information fails when taught"],
    ["EDL", "one pass per window; rejects as well as the others when taught", "under-confident probabilities; training is sensitive to the loss and its schedule"],
  ]),
  gap(),
  b([["Figures. ", B], ["TO DO: (1) bar chart of unknown windows accepted, plain against taught; (2) score histograms of known and unknown windows for one model, plain and taught."]]),
  b([["Our wearable. ", B], ["TO DO: before and after fine-tuning, trained on one person and tested on the other."]]),
  b([["Limits to state. ", B], ["One seed and no significance tests; RecoFit is an arm band, not our wrist kit; taught models depend on which background movements they saw; variants of a known exercise are accepted about 90% of the time."]]),

  h1("7. Application and user interaction (10 points, about three quarters of a page)"),
  b("One command starts a local web page that also serves the predictions."),
  b([["Upload: ", B], ["the wearable streams over Bluetooth, or a recorded file is uploaded. The page shows the signal and how much has arrived."]]),
  b([["Prediction: ", B], ["three models side by side (plain softmax, taught MC Dropout, taught EDL), each with its answer or \"Unknown\"."]]),
  b([["Explanation: ", B], ["each model's uncertainty on a 0 to 100% bar with its limit marked, the name of its measure, and its last 30 answers."]]),
  b("A second page records labelled sets from the wearable for fine-tuning."),
  b("TO DO: one screenshot; one sentence on how AI tools were used for the interface."),

  h1("8. Conclusion (3 points, a short paragraph)"),
  b("Uncertainty scores alone do not make a small exercise recognizer reject unfamiliar movement."),
  b("Showing the model other movement and asking it to be uncertain there does, at the same accuracy; the price is a few more known windows rejected."),
  b("Future work: more seeds and significance tests; more varied background data; evaluation on the wrist kit across more people; running the model on the device."),

  h1("9. Video demo (10 points)"),
  b("TO DO: 1 to 3 minutes: the goal, the live page with the wearable, the main result. Link goes here."),

  h1("10. References"),
  b("Morris, D., Saponas, T. S., Guillory, A., and Kelner, I. (2014). RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises. CHI 2014, 3225-3234. doi:10.1145/2556288.2557116"),
  b("Hendrycks, D., and Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017. arXiv:1610.02136"),
  b("Gal, Y., and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML 2016. arXiv:1506.02142"),
  b("Sensoy, M., Kaplan, L., and Kandemir, M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. NeurIPS 2018. arXiv:1806.01768"),
  b("Hendrycks, D., Mazeika, M., and Dietterich, T. (2019). Deep Anomaly Detection with Outlier Exposure. ICLR 2019. arXiv:1812.04606"),
  b("Huo, Z., PakBin, A., Chen, X., Hurley, N., Yuan, Y., Qian, X., Wang, Z., Huang, S., and Mortazavi, B. (2020). Uncertainty Quantification for Deep Context-Aware Mobile Activity Recognition and Unknown Context Discovery. AISTATS 2020. arXiv:2003.01753"),

  h1("11. Individual contributions (required) and AI use"),
  b([["Robert: ", B], ["EDL; data pipeline and splits; softmax network and training; wearable firmware and recordings; live app. TO DO: complete from docs/contributions.md."]]),
  b([["Sirena: ", B], ["MC Dropout. TO DO: list her other parts (evaluation, logistic regression, review, writing)."]]),
  b([["AI use: ", B], ["required by the handout. Summarize docs/ai-use-log.md: what the assistant produced, and what we decided, checked and wrote ourselves."]]),
];

const doc = new Document({
  creator: "Robert Ashe and Sirena Rocca",
  title: "UAAR final report outline",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22, color: INK } },
      title: { run: { font: FONT, size: 40, bold: true, color: INK }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      heading1: { run: { font: FONT, size: 26, bold: true, color: INK }, paragraph: { spacing: { before: 260, after: 100 }, keepNext: true } },
      heading2: { run: { font: FONT, size: 22, bold: true, color: MUTED }, paragraph: { spacing: { before: 140, after: 60 }, keepNext: true } },
    },
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 240 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 240 } } } },
      ],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

const out = path.join(__dirname, "..", "outline.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
