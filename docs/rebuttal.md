# Rebuttal

## Overall Rebuttal

Thank you for your thorough review and precious feedback. We have carefully considered your comments and suggestions. Regarding the weakness of the paper, below are the changes and clarifications we have made in the potential revised version of the paper:

1. **Clarifications.** TBA.
1. **Interpretability.** TBA.
1. **Ablation Study.** TBA.
1. **Benchmarking.** TBA.

## Reviewer 1

Thank you for your thorough review and precious feedback. We have carefully considered your comments and suggestions. Regarding the weakness of the paper, below are the changes and clarifications we have made in the potential revised version of the paper:

1. **Mixed Participant Data Split.** In section III.A.3, the 5-fold data split we used does allow participants to be mixed between the training and test sets. We applied constraints to the cross-validation process to ensure that the same participants are included in both the training and test sets, with the exception of participants who have fewer than 5 data points; these participants are included only in the training set.
1. **Prediction Using Mean Value in the Training Data.** We have utilized the average stress score from the training data to predict each participant's stress score in the test data. However, this approach assumes knowledge of the participants' identities for testing, which makes it more comparable to a personalized model. Therefore, while we will discuss the mean value prediction in the related works section, it will not be included in the experimental results. To ensure a fair comparison, our approach should include participant IDs as input, making it consistent with the concept of a personalized model.
1. **Interpretability of Multimodal Deep Learning.** We agree that XGBoost's feature importance is not directly applicable to the multimodal deep learning model. We included it to provide some statistical insights into the time series data. In the revised version of the manuscript, we will relocate the XGBoost feature importance results to the appendix and focus on interpreting the multimodal deep learning model using the feature ablation algorithm provided by Captum. The feature ablation algorithm is a permutation-based method that quantifies the impact of altering a feature on the prediction. Based on the feature ablation results, we have observed that... (details to be added after the experiments).
1. **Single-Channel vs. Multi-Channel.** Thank you for suggesting experiments with a single Fitbit channel for self-supervised multimodal learning. After conducting these experiments, we have observed that using important channels improves model performance compared to using all channels in the LifeSnaps dataset. In contrast, the model performs better when using all channels compared to using only the important channels in the PMData dataset. We will include these results in the revised paper.
1. **Ablation Study on the Amount of Data Used for Pretraining.** Thank you for the suggestion. We have added an ablation study using 1% of the data; however, we will not use 0.1% as it is too small to effectively train the model, given that there are only a few thousand data points in total. We will include the results in the revised paper.
1. **Clarification for Table I.** In Table I, for supervised methods, we utilize 100% of the labeled data points for training. For self-supervised methods, we use 100% of the unlabeled data points for pretraining, followed by fine-tuning the model with all labeled data points. The number of labeled data points in both datasets is mentioned in section IV.A, and the exact number used for training can be calculated by multiplying the percentage of labeled data points used for pretraining by the total number of labeled data points.
1. **Clarification for Table II.** To clarify the confusion in Table II: In the LifeSnaps dataset, the percentages refer to the proportion of unlabeled data points used for pretraining, with all labeled data points being used as well since most of the data in LifeSnaps is unlabeled. In the PMData dataset, where most data points are labeled, the percentages indicate the proportion of labeled data points used for pretraining, with all unlabeled data points also included in pretraining.
1. **Hyperparameter Tuning.** We did not extensively tune hyperparameters as it would be unfair to the baseline methods if we optimized hyperparameters for our model but not for them. Given the time constraints, we may not have been able to tune hyperparameters for all models. While tuning can influence results, it is unlikely to cause significant changes. However, in future work, we plan to use libraries such as Optuna to optimize hyperparameters for all models.
1. **Adding Technical Papers in Related Work.** Thank you for sharing the techinical papers. We have read the technical papers you suggested and will include them in the related work section in the revised paper. However, as it mainly focuses on time series data, we will not compare our method with them in the experiment section. 

About the questions you asked:

1. **No Label in the Prompt.** During the prompt generation, we did include the label information in the prompt. We will clarify this in the revised paper.

## Reviewer 2

Thank you for your thorough review and precious feedback. We have carefully considered your comments and suggestions. Regarding the weakness of the paper, below are the changes and clarifications we have made in the potential revised version of the paper:

Weakness.

1. **Requirements of Big Data.** We agree that the model requires a large amount of data to achieve good performance. However, in stress detection datasets, the number of data points is limited, and the model's performance is affected by the amount of data used for pretraining. We will add a discussion on the limitations of the model in the revised paper.
1. **Small Enhancement of Test AUC on Fully Labeled Dataset.** It is common that the model's performance does not improve significantly when using a fully labeled dataset for fine-tuning. We will add a discussion on this in the revised paper. To improve the model's performance, we will consider how to best employ the representations from general tasks in deep learning and large language models to cope with the limited amount of data in stress detection datasets.
1. **Optimal Prompt Generation.** This paper focuses on the multimodal deep learning model and the self-supervised learning approach. We will consider the optimal prompt generation in future work.

Questions.

1. **Size and Diversity of Datasets Used for Training.** To clarify the confusion in Table II, in LifeSnaps, the proportion of the data refer to unlabeled data points and all the labeled data points are used for pretraining. In PMData, as most data points are labeled, the percentage means the proportion of labeled data points used for pretraining, and all of the unlabeled data points are used for pretraining. In Table I, for supervised methods, we use 100% labeled data points and for self-supervised methods, we use 100% unlabeled data points and then fine-tune the model using labeled data points. The number of labels for both datasets are mentioned in section IV.A, and the number of labels used for training can be computed by multiplying the percentage of labeled data points used for pretraining by the total number of labeled data points.
1. **Handeling Missing Data.** Our method fill missing data with -1, and for prompt generation, we delete the description of the missing feature.
1. **Comparing with SOTA.** Based on our survey, there are no methods for modeling both time series and tabular features for stress detection, and our method is the first to do so. For papers only considering time series data, we will include them in the related work section in the revised paper. But due to the limited time, we may not compare our method with all the methods in the experiment section. We will include this information in the revised paper.
1. **Interpretability in Clinical Setting.** To interpret the decision maded by our multimodal deep learning model, we have used the feature ablation algorithm provided by Captum to explain the model's predictions. We have also observed consistent patterns in the model's behavior across two datasets, LifeSnaps and PMData. We will include the results of the feature ablation algorithm in the revised paper.

## Reviewer 3

Thank you for your thorough review and precious feedback. We have carefully considered your comments and suggestions. Regarding the weakness of the paper, below are the changes and clarifications we have made in the potential revised version of the paper:

Weakness.

1. **Clarification of Fine-tuning.** In Table I, for supervised methods, we use 100% labeled data points to trian the model. For self-supervised methods, we use 100% unlabeled data points to pretrain the model and then fine-tune the model and a fully-connected layer using 100% labeled data points. We will clarify this in the revised paper.
1. **Insignificant Performance Improvement.** It is common that the model's performance does not improve significantly when the amount of data is limited, such as thousands of data points. We will add a discussion on this in the revised paper. To improve the model's performance, we will consider how to best employ the representations from general tasks in deep learning and large language models to cope with the limited amount of data in stress detection datasets. Or we can consider train the model on a combination of multiple datasets to improve the model's performance.
1. **Interpretability of Multimodal Deep Learning.** We agree that XGBoost's feature importance is not directly relevant to the multimodal deep learning model. We included it to gain some insights into the statistics of time series data. To interpret the multimodal deep learning model, we have used the feature ablation algorithm provided by Captum to explain the model's predictions. We have also observed consistent patterns in the model's behavior across two datasets, LifeSnaps and PMData. We will include the results of the feature ablation algorithm in the revised paper.

Questions.

1. **Late Fusion of Tabular and Signal Representations.** We concatenate the representations from the tabular and signal data and feed them into a fully connected layer. We will clarify this in the revised paper.
1. **Prompt Encoding and Contrastive Pre-training.** Yes, the importance of prompt encoding is subject to be further studied. However, to train CLIP, it is required to generate textual descriptions for the input data. For self-supervised mutlimodal learning with time series and tabular data, to our knowledge, there is no such method yet, and that can be a future direction. Generating the best prompt for input data is also a future direction but it is a little beyond the scope of this paper.
1. **Cost and Automation of Prompt Generation.** Generating the prompt is an automated process as we defined the pattern of the prompt. The use of large language models is alos helpful so it is not costly. And it is even cheaper than labeling by human.
1. **LEAVES on the PMData Dataset.** We think the reason why LEAVES did not perform well on the PMData is because in PMData, there are around 50 tabular features and they provide more information than the time series data. 
1. **Invariant Performance for Ablation Study on LifeSnaps.** We do not think it is because of the label distribution because it is not serverly imbalanced. 80 are stressed and 160 are relaxed.
1. **Common Signals on LifeSnaps and PMData.** Common signals are steps, heart_rate, calories, and distance.
1. **Comparing with Other Methods.** Papers [1, 27, 16, 31] did not provide code. For [1, 27, 31], the multimodal refer to multi channel time series. For [16], the multimodal refer to multi channel time series and images. Therefore, we did not compare with them because our method is time series and tabular data. We will include this information in the revised paper.
