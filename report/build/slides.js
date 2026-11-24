// Builds report/slides.pptx: the final presentation (7 slides, about 6 minutes), in the style of
// the live demo page (web/demo.css): light page, black ink, cards, uncertainty bars with a limit.
//   NODE_PATH=<folder with node_modules> node report/build/slides.js <path to apply_theme.js>
const path = require("path");
const pptxgen = require("pptxgenjs");

const THEME = {
  name: "UAAR",
  headFontFace: "Bahnschrift",
  bodyFontFace: "Segoe UI",
  colors: {
    dk1: "0B0B0B", lt1: "FCFCFB", dk2: "52514E", lt2: "F9F9F7",
    accent1: "D03B3B", accent2: "2A78D6", accent3: "EB6834", accent4: "1BAF7A", accent5: "EDA100",
    accent6: "898781", hlink: "0B0B0B", folHlink: "52514E",
  },
};
const HAIR = "E1E0D9"; // hairline: not a scheme colour
const GREY = "B9B7B0";
const HEAD = "Bahnschrift";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Uncertainty-aware activity recognition";
pres.author = "Robert Ashe and Sirena Rocca";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const INK = C.text1, INK2 = C.text2, SURFACE = C.background1, PAGE = C.background2;
const RED = C.accent1, MUTED = C.accent6;
const CLASS = { "Jumping jacks": C.accent2, "Lateral raise": C.accent3, "Shoulder press": C.accent4, Curl: C.accent5 };

pres.defineSlideMaster({
  title: "Title",
  background: { color: PAGE },
  objects: [],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: PAGE },
  objects: [
    { text: { text: "", options: { shape: pres.ShapeType.ellipse, x: 0.6, y: 0.5, w: 0.62, h: 0.62, fill: { color: INK }, line: { color: INK } } } },
    { placeholder: { options: { name: "title", type: "title", x: 1.45, y: 0.38, w: 11.2, h: 0.86, fontFace: HEAD, fontSize: 30, bold: true, color: INK, align: "left", valign: "middle", margin: 0 }, text: "Slide title" } },
    { text: { text: "Uncertainty-aware activity recognition", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 0.6, y: 0.5, w: 0.62, h: 0.62, fontFace: HEAD, fontSize: 20, bold: true, color: SURFACE, align: "center", valign: "middle" },
});

// ---- pieces of the demo page, redrawn as shapes ---------------------------------------------------
let n = 0;
const name = (what) => `${what} ${++n}`;
function text(slide, str, o) {
  slide.addText(str, { isTextBox: true, margin: 0, color: INK, fontSize: 14, valign: "top", objectName: name("Text"), ...o });
}
function card(slide, x, y, w, h, fill = SURFACE) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: HAIR, width: 1 }, objectName: name("Card") });
}
function pill(slide, str, x, y, w, { dark = false, h = 0.5, fontSize = 15 } = {}) {
  slide.addText(str, {
    shape: pres.ShapeType.roundRect, rectRadius: h / 2, x, y, w, h, isTextBox: true, margin: 0,
    fill: { color: dark ? INK : SURFACE }, line: { color: INK, width: 1.5 },
    color: dark ? SURFACE : INK, fontFace: HEAD, fontSize, align: "center", valign: "middle", objectName: name("Pill"),
  });
}
function chip(slide, x, y, size, color) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w: size, h: size, rectRadius: 0.03, fill: { color }, line: { color }, objectName: name("Chip") });
}
/** The page's uncertainty bar: a track, a fill (red past the limit) and the limit mark. */
function meter(slide, x, y, w, value, limit, { h = 0.26, caption = true } = {}) {
  const over = value > limit;
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.04, fill: { color: HAIR }, line: { color: HAIR }, objectName: name("Track") });
  if (value > 0.01) slide.addShape(pres.ShapeType.roundRect, { x, y, w: Math.max(0.08, w * value), h, rectRadius: 0.04, fill: { color: over ? RED : INK2 }, line: { color: over ? RED : INK2 }, objectName: name("Fill") });
  slide.addShape(pres.ShapeType.rect, { x: x + w * limit - 0.02, y: y - 0.08, w: 0.04, h: h + 0.16, fill: { color: INK }, line: { color: INK }, objectName: name("Limit") });
  if (caption) text(slide, `limit ${Math.round(100 * limit)}%`, { x: x + w * limit - 0.6, y: y + h + 0.1, w: 1.2, h: 0.22, fontSize: 10, color: INK2, align: "center" });
}
function content(title, notes) {
  const slide = pres.addSlide({ masterName: "Content" });
  slide.addText(title, { placeholder: "title" });
  slide.addNotes(notes);
  return slide;
}

// ---- 1 Title --------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "Title" });
  text(s, "Knowing when it doesn't know", { x: 0.7, y: 1.55, w: 6.3, h: 2.1, fontFace: HEAD, fontSize: 50, bold: true, valign: "top" });
  text(s, "Uncertainty-aware activity recognition on a wrist sensor", { x: 0.7, y: 3.85, w: 6.0, h: 0.9, fontSize: 20, color: INK2 });
  text(s, "Robert Ashe and Sirena Rocca", { x: 0.7, y: 5.3, w: 6, h: 0.4, fontSize: 18, bold: true });
  text(s, "CompE 510, December 3, 2026", { x: 0.7, y: 5.75, w: 6, h: 0.35, fontSize: 14, color: INK2 });

  const mini = (x, title, answer, color, value, limit, measure) => {
    card(s, x, 1.55, 2.75, 3.75);
    text(s, title, { x: x + 0.25, y: 1.75, w: 2.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true });
    chip(s, x + 0.25, 2.72, 0.3, color);
    text(s, answer, { x: x + 0.68, y: 2.55, w: 2.0, h: 0.65, fontFace: HEAD, fontSize: 24, bold: true, color: answer === "Unknown" ? RED : INK, valign: "middle" });
    text(s, measure, { x: x + 0.25, y: 3.75, w: 1.6, h: 0.3, fontSize: 11, color: INK2 });
    text(s, `${Math.round(100 * value)}%`, { x: x + 1.7, y: 3.66, w: 0.8, h: 0.4, fontFace: HEAD, fontSize: 18, align: "right" });
    meter(s, x + 0.25, 4.2, 2.25, value, limit);
  };
  mini(7.3, "Softmax network", "Lateral raise", CLASS["Lateral raise"], 0.15, 0.38, "Uncertainty");
  mini(10.3, "Evidential network", "Unknown", RED, 1.0, 0.78, "Uncertainty");
  text(s, "The wearer is doing a crunch, which neither model was trained on.", { x: 7.3, y: 5.5, w: 5.75, h: 0.6, fontSize: 14, color: INK2 });
  s.addNotes("About 20 seconds. We built an exercise recognizer for a wrist sensor that can say \"I don't know this movement\". On the right is the idea in one picture: the person is doing a crunch, which is not one of the model's exercises. A standard network calls it a lateral raise and is sure of it. The model we trained to know its limits says unknown.");
}

// ---- 2 Background ---------------------------------------------------------------------------------
{
  const s = content("A recognizer always answers, even for a movement it has never seen",
    "About 55 seconds. A wrist recognizer is trained on a fixed list of exercises, here four. In use it sees everything else too: other exercises, walking, resting. A classifier has to pick one of its classes, so it is often confidently wrong. On RecoFit, a standard network that is 96% accurate on its four exercises still accepts 65% of the windows from 16 exercises it never saw. We build on five papers: RecoFit gives the data; the max-softmax baseline, MC Dropout and evidential deep learning are the uncertainty methods; outlier exposure is the training idea we use later. What is new here is the comparison under one protocol with real unknown exercises, and running it live on a wearable we built.");
  pill(s, "Any wrist movement", 0.6, 1.75, 2.7);
  s.addShape(pres.ShapeType.line, { x: 3.4, y: 2.0, w: 0.9, h: 0, line: { color: INK, width: 2, dashType: "dash" }, objectName: name("Flow") });
  pill(s, "4-exercise model", 4.4, 1.75, 2.5, { dark: true });
  s.addShape(pres.ShapeType.line, { x: 7.0, y: 2.0, w: 0.9, h: 0, line: { color: INK, width: 2, dashType: "dash" }, objectName: name("Flow") });
  Object.entries({ Curl: CLASS.Curl, "Shoulder press": CLASS["Shoulder press"], "Lateral raise": CLASS["Lateral raise"], "Jumping jacks": CLASS["Jumping jacks"] }).forEach(([label, color], i) => {
    const x = 8.1 + (i % 2) * 2.5, y = 1.62 + Math.floor(i / 2) * 0.45;
    chip(s, x, y + 0.06, 0.22, color);
    text(s, label, { x: x + 0.34, y, w: 2.1, h: 0.34, fontSize: 15, valign: "middle" });
  });

  text(s, "65%", { x: 0.6, y: 3.0, w: 3.0, h: 1.3, fontFace: HEAD, fontSize: 80, bold: true, color: RED, valign: "middle" });
  text(s, "of windows from 16 exercises the model never saw are accepted as one of its four.", { x: 3.6, y: 3.05, w: 3.3, h: 1.2, fontSize: 16, valign: "middle" });
  text(s, "The same model is 96% accurate on the four exercises it knows.", { x: 0.6, y: 4.5, w: 6.3, h: 0.4, fontSize: 14, color: INK2 });
  text(s, "Our question", { x: 0.6, y: 5.35, w: 6.3, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true });
  text(s, "Can the model answer \"unknown\", and which way of measuring uncertainty does that best?", { x: 0.6, y: 5.72, w: 6.3, h: 0.8, fontSize: 16 });

  card(s, 7.5, 2.85, 5.25, 3.75);
  text(s, "What we build on", { x: 7.75, y: 3.0, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true });
  const refs = [
    ["RecoFit", "Morris et al., 2014", "the exercise data"],
    ["Max-softmax baseline", "Hendrycks and Gimpel, 2017", "confidence as a detector"],
    ["MC Dropout", "Gal and Ghahramani, 2016", "repeat with dropout, look at the spread"],
    ["Evidential deep learning", "Sensoy et al., 2018", "evidence per class, in one pass"],
    ["Outlier exposure", "Hendrycks et al., 2019", "train to be unsure on other data"],
  ];
  refs.forEach(([what, who, why], i) => {
    const y = 3.5 + i * 0.6;
    text(s, [{ text: what, options: { bold: true } }, { text: `  ${who}`, options: { color: INK2 } }], { x: 7.75, y, w: 4.8, h: 0.28, fontSize: 14 });
    text(s, why, { x: 7.75, y: y + 0.27, w: 4.8, h: 0.26, fontSize: 12, color: INK2 });
  });
}

// ---- 3 Problem ------------------------------------------------------------------------------------
{
  const s = content("The problem: label every 2-second window, or answer unknown",
    "About 50 seconds. Formally: we are given two-second windows of six-axis wrist motion, with labels for four known exercises, from the training people. The goal is a model that gives every window a label and an uncertainty score, and answers unknown when the score is over a limit. The constraints are what make it hard: the test people are never seen in training, the unknown exercises are never seen in any form, and the limit is fixed on validation data so that 95% of known windows are kept. We then count how many unknown windows still get accepted. That share is our headline number.");
  const cols = [
    ["Given", ["2-second windows of wrist motion", "accelerometer and gyroscope, 50 Hz", "labels for K = 4 known exercises", "RecoFit: 94 people"]],
    ["Goal", ["a label for every window", "an uncertainty score u(x)", "answer \"unknown\" when u(x) is over a limit"]],
    ["Constraints", ["test people never seen in training", "unknown exercises never seen in any form", "limit set on validation data only", "one answer per window, live"]],
  ];
  cols.forEach(([head, items], i) => {
    const x = 0.6 + i * 4.11;
    card(s, x, 1.6, 3.9, 2.75);
    text(s, head, { x: x + 0.3, y: 1.78, w: 3.3, h: 0.42, fontFace: HEAD, fontSize: 22, bold: true });
    text(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1, paraSpaceAfter: 5 } })), { x: x + 0.3, y: 2.3, w: 3.35, h: 1.9, fontSize: 15 });
  });
  // the decision rule, drawn as the page's uncertainty bar
  text(s, "Uncertainty u(x)", { x: 0.6, y: 4.85, w: 3, h: 0.35, fontFace: HEAD, fontSize: 17, bold: true });
  const bx = 0.6, bw = 12.13, lim = 0.62;
  s.addShape(pres.ShapeType.roundRect, { x: bx, y: 5.3, w: bw * lim, h: 0.42, rectRadius: 0.05, fill: { color: INK2 }, line: { color: INK2 }, objectName: name("Accept") });
  s.addShape(pres.ShapeType.roundRect, { x: bx + bw * lim, y: 5.3, w: bw * (1 - lim), h: 0.42, rectRadius: 0.05, fill: { color: RED }, line: { color: RED }, objectName: name("Reject") });
  s.addShape(pres.ShapeType.rect, { x: bx + bw * lim - 0.03, y: 5.18, w: 0.06, h: 0.66, fill: { color: INK }, line: { color: INK }, objectName: name("Limit") });
  text(s, "below the limit: answer the exercise", { x: bx, y: 5.9, w: 6, h: 0.32, fontSize: 15 });
  text(s, "above the limit: answer \"unknown\"", { x: bx + bw * lim + 0.2, y: 5.9, w: 4.4, h: 0.32, fontSize: 15, color: RED, bold: true });
  text(s, "The limit keeps 95% of known validation windows. Headline measure: the share of unknown windows that are still accepted.", { x: bx, y: 6.35, w: 12.1, h: 0.35, fontSize: 14, color: INK2 });
}

// ---- 4 Methods: four measures ---------------------------------------------------------------------
{
  const s = content("Four ways to measure uncertainty",
    "About 55 seconds. We compare four methods. Two were taught in class: logistic regression on handcrafted features, and a small convolutional network with a softmax output. For both, uncertainty is one minus the highest class probability. The two new ones are ours. MC Dropout, which Sirena implemented, keeps dropout switched on at test time, runs the network thirty times and looks at how spread out the answers are. Evidential deep learning, which Robert implemented, makes the network output evidence for each class; with little evidence in total, the uncertainty, called vacuity, is high, and it needs only one pass. All four get the same data and the same rule for the limit.");
  const methods = [
    ["Taught in class", "Logistic regression", "A linear model on 102 handcrafted features of the window.", "Least confidence", "1 − highest class probability"],
    ["Taught in class", "Softmax network", "A small 1-D convolutional network with a softmax output.", "Least confidence", "1 − highest class probability"],
    ["New: Sirena", "MC Dropout", "The same network run 30 times with dropout left on.", "Predictive entropy", "entropy of the average of the 30 passes"],
    ["New: Robert", "Evidential network", "The network outputs evidence for each class, in one pass.", "Vacuity", "number of classes ÷ total evidence"],
  ];
  methods.forEach(([tag, title, how, measure, formula], i) => {
    const x = 0.6 + i * 3.08;
    const isNew = tag.startsWith("New");
    card(s, x, 1.6, 2.9, 4.45);
    pill(s, tag, x + 0.25, 1.82, isNew ? 1.5 : 1.75, { dark: isNew, h: 0.36, fontSize: 12 });
    text(s, title, { x: x + 0.25, y: 2.38, w: 2.45, h: 0.8, fontFace: HEAD, fontSize: 22, bold: true });
    text(s, how, { x: x + 0.25, y: 3.2, w: 2.45, h: 1.0, fontSize: 14, color: INK2 });
    text(s, "Uncertainty", { x: x + 0.25, y: 4.35, w: 2.45, h: 0.28, fontSize: 12, color: INK2 });
    text(s, measure, { x: x + 0.25, y: 4.62, w: 2.45, h: 0.38, fontFace: HEAD, fontSize: 17, bold: true });
    text(s, formula, { x: x + 0.25, y: 5.02, w: 2.45, h: 0.75, fontSize: 14 });
  });
  text(s, "Same data and the same rule for the limit for all four. The three networks share one architecture.", { x: 0.6, y: 6.3, w: 12.1, h: 0.4, fontSize: 15, color: INK2 });
}

// ---- 5 Methods: three ways to train ---------------------------------------------------------------
{
  const s = content("Three ways to train: plain, taught, and an \"Other\" class",
    "About 55 seconds. The four methods are trained in three ways. Plain: only the four exercises. Taught: we also show the model windows of other movement, sixteen other exercises and rest, and tell it to be uncertain there. For softmax the target is an even split over the four classes; for the evidential network the target is no evidence at all. The model still has four outputs, and the uncertainty score still makes the decision. As a reference we also train a network with a fifth class called Other. The bars show what each one typically does with a movement it does not know. The test is strict: five folds by person, and the sixteen unknown exercises are never seen in any form, not even as background.");
  const setups = [
    ["Plain", "Trained on the four exercises only.", [["Lateral raise", 0.9, CLASS["Lateral raise"]], ["Curl", 0.05, CLASS.Curl], ["Shoulder press", 0.03, CLASS["Shoulder press"]], ["Jumping jacks", 0.02, CLASS["Jumping jacks"]]], "Confident, and wrong", false],
    ["Taught", "Also shown other movement, with the target \"be uncertain here\".", [["Lateral raise", 0.26, CLASS["Lateral raise"]], ["Curl", 0.25, CLASS.Curl], ["Shoulder press", 0.25, CLASS["Shoulder press"]], ["Jumping jacks", 0.24, CLASS["Jumping jacks"]]], "Unsure, so it answers unknown", true],
    ["\"Other\" class", "The same other movement as a fifth class. Reference only.", [["Other", 0.95, GREY], ["Lateral raise", 0.02, CLASS["Lateral raise"]], ["Curl", 0.01, CLASS.Curl], ["Shoulder press", 0.01, CLASS["Shoulder press"]]], "Classified, not uncertain", true],
  ];
  setups.forEach(([title, how, bars, verdict, good], i) => {
    const x = 0.6 + i * 4.11;
    card(s, x, 1.6, 3.9, 3.95);
    text(s, title, { x: x + 0.3, y: 1.75, w: 3.3, h: 0.45, fontFace: HEAD, fontSize: 22, bold: true });
    text(s, how, { x: x + 0.3, y: 2.22, w: 3.35, h: 0.7, fontSize: 14, color: INK2 });
    bars.forEach(([label, v, color], j) => {
      const y = 3.08 + j * 0.37;
      text(s, label, { x: x + 0.3, y, w: 1.45, h: 0.28, fontSize: 12, valign: "middle" });
      s.addShape(pres.ShapeType.roundRect, { x: x + 1.8, y: y + 0.04, w: Math.max(0.05, 1.45 * v), h: 0.2, rectRadius: 0.03, fill: { color }, line: { color }, objectName: name("Bar") });
      text(s, `${Math.round(100 * v)}%`, { x: x + 1.8 + Math.max(0.05, 1.45 * v) + 0.08, y, w: 0.6, h: 0.28, fontSize: 12, color: INK2, valign: "middle" });
    });
    text(s, verdict, { x: x + 0.3, y: 4.85, w: 3.35, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: good ? INK : RED, valign: "middle" });
  });
  text(s, "Typical output for a movement the model does not know (illustration).", { x: 0.6, y: 5.65, w: 8, h: 0.3, fontSize: 12, color: INK2 });
  const facts = [["5 folds", "by person; test people are never seen"], ["16 exercises", "held out as unknown, never seen in any form"], ["16 others + rest", "used as the other movement in training"], ["15 runs", "behind every number"]];
  facts.forEach(([big, small], i) => {
    const x = 0.6 + i * 3.08;
    text(s, big, { x, y: 6.08, w: 2.9, h: 0.38, fontFace: HEAD, fontSize: 18, bold: true });
    text(s, small, { x, y: 6.46, w: 2.9, h: 0.45, fontSize: 12, color: INK2 });
  });
}

// ---- 6 Results ------------------------------------------------------------------------------------
{
  const s = content("Result: taught models reject unknown movement, plain models do not",
    "About 60 seconds. This is the main result. The bars show the share of unknown-exercise windows that each model still accepts as one of its four exercises; lower is better. Grey is plain, black is taught. Plain models accept 60 to 65 percent, whichever uncertainty measure they use. Taught models accept 5 to 6 percent, and this held in every one of the fifteen runs. Known accuracy stays at 96 percent throughout. Two things to note. First, taught models are as good as the explicit Other class, at 5 percent, while still deciding by uncertainty. Second, a warning: MC Dropout's mutual information, which measures disagreement between passes, fails on a taught network, because all passes agree on \"unsure\"; it accepts 97 percent. The entropy of the average works. The cost is that a few more known windows are rejected: 88 percent kept, against 93. One seed, no significance tests yet.");
  s.addChart(pres.charts.BAR, [
    { name: "Plain", labels: ["Logistic regression", "Softmax network", "MC Dropout", "Evidential network"], values: [59.9, 64.7, 63.1, 63.0] },
    { name: "Taught", labels: ["Logistic regression", "Softmax network", "MC Dropout", "Evidential network"], values: [null, 6.3, 6.1, 5.2] },
  ], {
    x: 0.5, y: 1.55, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [GREY, THEME.colors.dk1],
    showTitle: true, title: "Unknown-exercise windows accepted (%), lower is better", titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: THEME.colors.dk1,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: "+mn-lt", dataLabelFontSize: 13, dataLabelColor: THEME.colors.dk1, dataLabelFormatCode: "0",
    catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 13, catAxisLabelColor: THEME.colors.dk1,
    valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 12, valAxisLabelColor: THEME.colors.dk2, valAxisMaxVal: 80, valAxisMinVal: 0, valAxisMajorUnit: 20,
    valGridLine: { color: HAIR, size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontFace: "+mn-lt", legendFontSize: 13, legendColor: THEME.colors.dk1,
    objectName: "Chart: unknown windows accepted",
  });
  text(s, "Logistic regression has no taught version. MC Dropout is scored by entropy here.", { x: 0.6, y: 6.5, w: 7.5, h: 0.3, fontSize: 11, color: INK2 });

  const stats = [
    ["0.96", INK, "known accuracy on new people, plain or taught"],
    ["65% → 6%", INK, "unknown windows accepted by the softmax network, plain to taught"],
    ["97%", RED, "accepted when a taught MC Dropout is scored by mutual information: all passes agree on \"unsure\""],
  ];
  stats.forEach(([big, color, small], i) => {
    const y = 1.6 + i * 1.5;
    card(s, 8.45, y, 4.3, 1.33);
    text(s, big, { x: 8.7, y: y + 0.12, w: 3.8, h: 0.56, fontFace: HEAD, fontSize: 30, bold: true, color, valign: "middle" });
    text(s, small, { x: 8.7, y: y + 0.7, w: 3.85, h: 0.56, fontSize: 12, color: INK2 });
  });
  text(s, "Reference \"Other\" class: 5%. Cost of teaching: 88% of known windows kept, against 93%. One seed.", { x: 8.45, y: 6.15, w: 4.3, h: 0.65, fontSize: 12, color: INK2 });
}

// ---- 7 Application and demo -----------------------------------------------------------------------
{
  const s = content("The app: upload a stream, see each model's answer and uncertainty",
    "About 40 seconds, then the demo. The app is one local web page. Step one, upload: the wearable streams over Bluetooth, or you upload a recorded file. Step two, prediction: three models side by side, a plain softmax network and the taught MC Dropout and evidential models. Each card shows the answer, and the model's own uncertainty as a bar with its limit; over the limit it says unknown. Now live: [click the button]. Rest first: the taught models should say unknown. Then a curl and a shoulder press: all three should agree. Then a movement none of them knows: watch the left card keep naming an exercise while the other two say unknown. If Bluetooth fails, start the page with the replay option and narrate the recording.");
  s.addImage({ path: path.join(__dirname, "..", "assets", "demo_page.png"), x: 0.6, y: 1.6, w: 7.9, h: 4.44, altText: "The live demo page: an upload strip with the signal, and three model cards with their answers and uncertainty bars", objectName: "Screenshot of the demo page" });
  s.addShape(pres.ShapeType.rect, { x: 0.6, y: 1.6, w: 7.9, h: 4.44, fill: { type: "none" }, line: { color: HAIR, width: 1 }, objectName: name("Frame") });
  const steps = [
    ["1", "Upload", "live from the wearable over Bluetooth, or a recorded file"],
    ["2", "Prediction", "three models side by side: plain softmax, taught MC Dropout, taught evidential"],
    ["3", "Uncertainty", "each model's own measure on a bar; over its limit the model answers unknown"],
  ];
  steps.forEach(([num, head, body], i) => {
    const y = 1.62 + i * 1.12;
    s.addText(num, { shape: pres.ShapeType.ellipse, x: 8.85, y: y + 0.03, w: 0.42, h: 0.42, isTextBox: true, margin: 0, fill: { color: INK }, line: { color: INK }, color: SURFACE, fontFace: HEAD, fontSize: 15, bold: true, align: "center", valign: "middle", objectName: name("Step") });
    text(s, head, { x: 9.45, y, w: 3.3, h: 0.42, fontFace: HEAD, fontSize: 19, bold: true, valign: "middle" });
    text(s, body, { x: 9.45, y: y + 0.44, w: 3.3, h: 0.62, fontSize: 13, color: INK2 });
  });
  s.addText([{ text: "Open the live demo", options: { hyperlink: { url: "http://localhost:8000/", tooltip: "The demo page (start it with: uv run scripts/demo.py)" } } }], {
    shape: pres.ShapeType.roundRect, rectRadius: 0.08, x: 8.85, y: 5.18, w: 3.9, h: 0.62, isTextBox: true, margin: 0,
    fill: { color: INK }, line: { color: INK }, color: SURFACE, fontFace: HEAD, fontSize: 20, bold: true, align: "center", valign: "middle", objectName: "Button: open the live demo",
  });
  text(s, "localhost:8000", { x: 8.85, y: 5.84, w: 3.9, h: 0.28, fontSize: 12, color: INK2, align: "center" });
  text(s, "Live: rest, a curl, a shoulder press, then a movement none of the models was trained on.", { x: 0.6, y: 6.25, w: 12.1, h: 0.4, fontSize: 16, bold: true });
}

const out = path.join(__dirname, "..", "slides.pptx");
pres.writeFile({ fileName: out }).then(async () => {
  const { applyTheme } = require(process.argv[2]);
  await applyTheme(out, THEME);
  console.log("wrote", out);
});
