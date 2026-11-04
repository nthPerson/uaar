# References

★ = core papers for the report's Related Work section (the course requires at least two, with a summary and a discussion of how our approach differs).

## Uncertainty methods

- ★ Sensoy, M., Kaplan, L., & Kandemir, M. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS. https://arxiv.org/abs/1806.01768 (M3, EDL)
- ★ Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML. https://arxiv.org/abs/1506.02142 (M4, MC Dropout)
- ★ Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR. https://arxiv.org/abs/1610.02136 (M2 score, MSP)
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. https://arxiv.org/abs/1612.01474 (stretch)
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. https://arxiv.org/abs/1706.04599 (ECE, temperature scaling)
- Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS. https://arxiv.org/abs/1705.08500 (risk–coverage)
- Malinin, A., & Gales, M. (2018). *Predictive Uncertainty Estimation via Prior Networks.* NeurIPS. https://arxiv.org/abs/1802.10501 (Dirichlet-output relative of EDL)
- Ulmer, D., Hardmeier, C., & Frellsen, J. (2023). *Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods for Uncertainty Estimation.* TMLR. https://arxiv.org/abs/2110.03051 (EDL caveats)
- Bendale, A., & Boult, T. E. (2016). *Towards Open Set Deep Networks* (OpenMax). CVPR. https://arxiv.org/abs/1511.06233

## Uncertainty and open-set recognition for activity recognition

- ★ Yao, S., Zhao, Y., Shao, H., Zhang, A., Zhang, C., Li, S., & Abdelzaher, T. (2018). *RDeepSense: Reliable Deep Mobile Computing Models with Uncertainty Estimations.* https://arxiv.org/abs/1709.02980
- ★ Huo, et al. (2020). *Uncertainty Quantification for Deep Context-Aware Mobile Activity Recognition and Unknown Context Discovery.* AISTATS (PMLR 108). https://arxiv.org/abs/2003.01753
- Gudur, G. K., Sundaramoorthy, P., & Umaashankar, V. (2019). *ActiveHARNet: Towards On-Device Deep Bayesian Active Learning for Human Activity Recognition.* https://arxiv.org/abs/1906.00108 (MC Dropout on wearable HAR)
- *Open-Set Recognition of Human Activities from Head-Mounted Inertial Sensor* (2026). Sensors 26(3), 1079. https://doi.org/10.3390/s26031079 (compares max-logit, GMM, KDE, OpenMax and nearest-neighbour ratio; AUROC > 0.8)
- *Hi-OSCAR: Hierarchical Open-set Classifier for Human Activity Recognition* (2025). https://arxiv.org/abs/2510.08635

**How UAAR differs (draft for the report):** prior HAR work usually evaluates one uncertainty method, often on daily-living activities, and rarely separates the three failure sources we test: unknown exercises, non-exercise motion, and a change of device. UAAR compares a single-pass evidential method against sampling-based MC Dropout and two classical baselines under one protocol, adds a near-unknown split of look-alike exercises, and reports inference cost alongside detection quality.

## Datasets

- Morris, D., Saponas, T. S., Guillory, A., & Kelner, I. (2014). *RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises.* CHI. https://doi.org/10.1145/2556288.2557116 · data: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
- Strömbäck, D., Huang, S., & Radu, V. (2020). *MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices.* IMWUT 4(4). https://doi.org/10.1145/3432701 · data: https://mmfit.github.io/
- Bian, S., et al. *RecGym: Gym Workouts Recognition Dataset with IMU and Capacitive Sensor.* https://arxiv.org/abs/2210.14794 · data: https://zhaxidele.github.io/RecGym/
