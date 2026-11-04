# Uncertainty-Aware Activity Recognition (UAAR)

CompE 510 · Machine Learning for Engineers · Fall 2026 · Term project
Team: **Robert Ashe**, **Sirena Rocca**

This folder is the working outline for the project. It records what was decided while scoping the idea (brainstorm "Option C", 2026-09-27) and what still has to be checked once the data are loaded. Items marked **(verify)** are expectations, not facts yet.

| File | Contents |
|---|---|
| [README.md](README.md) | Problem, questions, scope, goals, plan, roles, course-requirement map, risks |
| [datasets.md](datasets.md) | RecoFit, MM-Fit, RecGym and our own wearable recordings: sources, formats, units, splits, preprocessing |
| [models.md](models.md) | Shared backbone, the four compared methods, training setup, stretch methods |
| [evaluation.md](evaluation.md) | Protocols, metrics, threshold selection, statistics, planned figures and tables |
| [software-and-firmware.md](software-and-firmware.md) | Repository layout, software stack, demo app spec, wearable hardware/firmware interface and recording protocol |
| [development-spec.md](development-spec.md) | Build plan: repository and tooling, data contracts, module interfaces, scripts and configs, tests, documentation plan, working agreements, week-by-week schedule |
| [references.md](references.md) | Related work and dataset citations |
| [ai-use-log.md](ai-use-log.md) | Running record of AI assistance, for the report's disclosure section |

---

## 1. Problem

A wrist-worn exercise recognizer is trained on a fixed list of exercises. In use it will see movements that were never in that list: a new exercise, a variant of a known one, everyday motion between sets, or the same exercise measured by a different device or a strap worn differently. A standard classifier must still pick one of its known labels, and its softmax confidence is often high when it is wrong. Downstream, that produces wrong rep counts, wrong form feedback and misleading workout logs.

**Engineering goal:** give the recognizer a usable way to say "I don't know," and measure which method does that best, at what accuracy and compute cost, on data from people and devices it was not trained on.

## 2. Research questions and hypotheses

| # | Question | Hypothesis (to test, not assumed) |
|---|---|---|
| Q1 | How well does softmax confidence separate unknown exercises from known ones? | Poorly: the maximum softmax probability stays high on many unknown windows. |
| Q2 | Do evidential deep learning (EDL) and Monte Carlo (MC) Dropout detect unknown exercises better than softmax and logistic regression, without lowering known-class accuracy? | Yes, higher AUROC for unknown detection; known-class macro-F1 within about 2 points of the softmax baseline. |
| Q3 | What does each method cost at inference time? | EDL needs one forward pass; MC Dropout needs T passes (T ≈ 30), so EDL is the better fit for a wearable if detection is comparable. |
| Q4 | Does uncertainty rise when the device changes (forearm band → smartwatch → our wearable)? | Yes, and the windows with the highest uncertainty are the ones most often misclassified. |

## 3. Scope and goals

**Minimum goals (must deliver):**

1. An open-set benchmark on **RecoFit**: test subjects never seen in training, and some exercises held out entirely to act as "unknown."
2. Four methods compared on the same windows: **logistic regression** and a **softmax neural network** (both taught in class), **EDL** (Robert) and **MC Dropout** (Sirena). Report known-class accuracy, unknown detection and calibration.
3. A **Streamlit demo app**: upload an IMU CSV, see per-window predictions, an uncertainty timeline and "unknown" flags.

**Stretch goals (in order):**

1. **Device-shift test:** apply the RecoFit-trained models to the **MM-Fit** smartwatch streams.
2. **Replication:** repeat the open-set benchmark inside **RecGym** (wrist position).
3. **Our wearable:** a short recording session with the BDA 696 wearable (known exercises, new movements, a rotated strap).
4. **Extra methods:** deep ensembles, temperature scaling, SVM with Platt scaling (see [models.md §6](models.md#6-stretch-methods)).

**Out of scope:** on-device (microcontroller) inference, rep counting, form feedback, and anything camera-based. Those belong to the BDA 696 project.

## 4. Relationship to the BDA 696 project (FormCoach)

Robert is also on a BDA 696 team building **FormCoach**, a wrist wearable (XIAO ESP32-S3 + BMI160) that decides when to wake a webcam pose pipeline and gives rule-based form feedback. The CompE 510 syllabus lists "submitting a piece of work you did for one class to another class" as plagiarism, so the boundary is explicit:

| | FormCoach (BDA 696) | UAAR (CompE 510) |
|---|---|---|
| Question | When should the camera run, and is the rep's form correct? | Can the recognizer tell when a movement is unfamiliar? |
| ML task | Binary active/idle gate, closed-set exercise class, pose rules | Open-set recognition and uncertainty estimation |
| Shared | Public datasets (RecoFit, MM-Fit, RecGym); the wearable as a measuring device | Same |
| Not shared | Code, models, results, reports | Separate repository, written for this course |

UAAR does not import FormCoach code. Where FormCoach's documentation recorded facts about the dataset files (download URLs, units, column layouts), those facts are reused in [datasets.md](datasets.md) and cited. State the overlap in both proposals and final reports.

## 5. Plan

Dates come from the CompE 510 syllabus (TTh 8:00–9:15).

| Phase | Dates | Work | Exit criterion | Course events |
|---|---|---|---|---|
| 0 | Sep 29 | Proposal presentation | Slides uploaded, presented | — |
| 1. Data | Sep 30 – Oct 12 | RecoFit loader, 2-s windows, subject folds, known/unknown class splits, exploratory plots | Windows and splits saved; class and subject counts in a table | Exam 1 Oct 8 |
| 2. Baselines | Oct 13 – Oct 25 | Handcrafted features + logistic regression; 1D-CNN backbone + softmax; evaluation harness (all metrics) | Closed-set and open-set tables for both baselines | NN lectures end Oct 15 |
| 3. New methods | Oct 26 – Nov 8 | EDL (Robert), MC Dropout (Sirena), same folds and splits | Four-method comparison table on RecoFit (**minimum goal 1–2 met**) | Exam 2 Nov 5 |
| 4. Shift + wearable | Nov 9 – Nov 22 | MM-Fit loader and device-shift test; RecGym replication; wearable recording session; inference-cost measurements | Shift table; wearable CSVs with labels | — |
| 5. App + write-up | Nov 23 – Dec 2 | Streamlit app, 1–3 min video, final slides (10–15), report draft | App runs from a fresh clone (**minimum goal 3 met**) | Thanksgiving Nov 26 |
| 6. Final | Dec 3 or Dec 8 | Final presentation; report due Dec 8 | Zip submitted: code, report, both slide decks | Exam 3 Dec 10 |

**Decision point, Nov 8:** if either new method is not producing sensible uncertainty by then, freeze the method set, skip stretch goals 2–4 and put the remaining time into analysis of what does work.

## 6. Roles

| | Robert Ashe | Sirena Rocca | Both |
|---|---|---|---|
| New algorithm (course rule) | **EDL** | **MC Dropout** | — |
| Engineering | Data pipeline (RecoFit, MM-Fit loaders, windows, splits); wearable recordings; device-shift test | Evaluation harness (metrics, reliability and risk–coverage plots); Streamlit app; RecGym replication | Baselines (logistic regression, softmax NN) |
| Writing | Methods (EDL), data description | Methods (MC Dropout), results figures | Introduction, related work, conclusion, video, slides |

The split is a proposal; adjust once Sirena's background and preferences are known.

## 7. Course-requirement map

| Requirement (Project_instruction.doc / syllabus) | Where it is met |
|---|---|
| Real engineering problem + public dataset | §1; [datasets.md](datasets.md) |
| Related work: at least two papers, summary + differences | [references.md](references.md) (core papers marked) |
| Formal problem definition: given, goal, constraints | [evaluation.md §1](evaluation.md#1-formal-problem-definition) |
| Each member implements an algorithm not covered in class | EDL (Robert), MC Dropout (Sirena) — neither is on the syllabus topic list |
| Compare with at least one in-class algorithm | Logistic regression and softmax neural network (in-class topics), SVM optional |
| App: upload data, show prediction, explain output | [software-and-firmware.md §3](software-and-firmware.md#3-demo-app-streamlit) |
| Video demo 1–3 min | Phase 5 |
| Report 5–7 pages incl. individual contributions and AI-use disclosure | Phase 5–6; [ai-use-log.md](ai-use-log.md) |
| Proposal 3–5 slides, 3 min (Sep 29); final 10–15 slides, 10 min (Dec 3/8) | Phase 0, Phase 6 |

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| RecoFit class imbalance (a few exercises have minutes of data, idle has hours) | High | Eligibility threshold for known classes; class-weighted loss; report macro-F1 |
| EDL training unstable (evidence collapse, sensitivity to KL annealing) | Medium | Follow Sensoy et al. loss and annealing exactly; ablate evidence activation; fall back to the log-loss variant |
| Unknown classes too easy (very different from known ones) | Medium | Add a hand-picked "near-unknown" split (variants of known exercises) |
| MM-Fit/RecoFit axes and placement differ | High (by design) | That is the shift being measured; rotation augmentation as an ablation, not a fix |
| RecGym units are normalized, not physical | Certain | Use it only inside its own benchmark; never mix with RecoFit/MM-Fit |
| Wearable not ready in November | Medium | Wearable test is a stretch goal; nothing else depends on it |
| Training time on CPU | Medium | ~30 k-parameter network; RTX 3070 eGPU available; fewer seeds if needed |
