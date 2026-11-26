# References

★ = core papers for the report's Related Work section (the course requires at least two, with a summary and a discussion of how our approach differs).

## Uncertainty methods

- ★ Sensoy, M., Kaplan, L., & Kandemir, M. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS. https://arxiv.org/abs/1806.01768 (M3, EDL)
- ★ Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML. https://arxiv.org/abs/1506.02142 (M4, MC Dropout)
- ★ Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR. https://arxiv.org/abs/1610.02136 (M2 score, MSP)
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. https://arxiv.org/abs/1612.01474 (related; not implemented)
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. https://arxiv.org/abs/1706.04599 (ECE)
- Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS. https://arxiv.org/abs/1705.08500 (risk–coverage)
- Malinin, A., & Gales, M. (2018). *Predictive Uncertainty Estimation via Prior Networks.* NeurIPS. https://arxiv.org/abs/1802.10501 (Dirichlet-output relative of EDL)
- Ulmer, D., Hardmeier, C., & Frellsen, J. (2023). *Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods for Uncertainty Estimation.* TMLR. https://arxiv.org/abs/2110.03051 (EDL caveats)
- Bendale, A., & Boult, T. E. (2016). *Towards Open Set Deep Networks* (OpenMax). CVPR. https://arxiv.org/abs/1511.06233

## Experimental method

- Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). *On the Stratification of Multi-Label Data.* ECML PKDD, LNCS 6913, 145–158. https://doi.org/10.1007/978-3-642-23808-6_10 (iterative stratification, used for the subject folds; implemented from the paper's description in `uaar.data.splits`)
- Hendrycks, D., Mazeika, M., & Dietterich, T. (2019). *Deep Anomaly Detection with Outlier Exposure.* ICLR. https://arxiv.org/abs/1812.04606 (training a classifier to be uncertain on auxiliary outlier data; the basis of the "taught" softmax model, [../methods.md](../methods.md). Title, authors and venue checked against the arXiv record on 2026-11-23; the loss was implemented from the description, not from the authors' code)

## Uncertainty and open-set recognition for activity recognition

- ★ Yao, S., Zhao, Y., Shao, H., Zhang, A., Zhang, C., Li, S., & Abdelzaher, T. (2017). *RDeepSense: Reliable Deep Mobile Computing Models with Uncertainty Estimations.* arXiv preprint. https://arxiv.org/abs/1709.02980
- ★ Huo, Z., PakBin, A., Chen, X., Hurley, N., Yuan, Y., Qian, X., Wang, Z., Huang, S., & Mortazavi, B. (2020). *Uncertainty Quantification for Deep Context-Aware Mobile Activity Recognition and Unknown Context Discovery.* AISTATS (PMLR 108). https://arxiv.org/abs/2003.01753
- Gudur, G. K., Sundaramoorthy, P., & Umaashankar, V. (2019). *ActiveHARNet: Towards On-Device Deep Bayesian Active Learning for Human Activity Recognition.* EMDL '19 workshop, ACM MobiSys. https://arxiv.org/abs/1906.00108 (MC Dropout on wearable HAR)
- Cortese, A., Solbiati, S., Scandelli, A., Giudici, A., Antonello, N., Trojaniello, D., Boracchi, G., & Caiani, E. G. (2026). *Open-Set Recognition of Human Activities from Head-Mounted Inertial Sensor.* Sensors 26(3), 1079. https://doi.org/10.3390/s26031079 (compares max-logit, GMM, KDE, OpenMax and nearest-neighbour ratio; AUROC > 0.8)
- McCarthy, C., Quirijnen, L., van Zandwijk, J. P., Geradts, Z., & Worring, M. (2025). *Hi-OSCAR: Hierarchical Open-set Classifier for Human Activity Recognition.* Proc. ACM IMWUT 9(4), Article 199. https://arxiv.org/abs/2510.08635

**How UAAR differs (draft for the report):** prior HAR work usually evaluates one uncertainty method, often on daily-living activities, and rarely separates unknown exercises from non-exercise motion. UAAR compares a single-pass evidential method against sampling-based MC Dropout and two classical baselines under one protocol, in two tiers: plain models, and models taught to be uncertain on background windows (in the manner of outlier exposure), with an explicit "Other" class as a reference. Variants of a known exercise are tested as their own set, not counted as unknown. A test on our own wrist wearable is pending.

## Datasets

- Morris, D., Saponas, T. S., Guillory, A., & Kelner, I. (2014). *RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises.* CHI. https://doi.org/10.1145/2556288.2557116 · data: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
