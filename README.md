# An active learning budget-based oversampling approach for partially labeled multi-class imbalanced data streams

This repository provides the source code, algorithms, experimental setup, and results for the budget-based overamspling method paper presented at the 38th ACM/SIGAPP Symposium On Applied Computing. The manuscript is available at [paper](https://www.researchgate.net/profile/Gabriel-Aguiar-3/publication/367992827_An_active_learning_budget-based_oversampling_approach_for_partially_labeled_multi-class_imbalanced_data_streams/links/63dbd32164fc8606380b21bb/An-active-learning-budget-based-oversampling-approach-for-partially-labeled-multi-class-imbalanced-data-streams.pdf).

## Abstract 
Learning classification models from multi-class imbalanced data streams is a challenging task in machine learning. 
Moreover, there is a common assumption that all instances are labeled and available for the training phase. However, this is not realistic in real-world
scenarios when learning from partially labeled data. In this work, we
propose an active learning method based on labeling budget that can
tackle multi-class imbalance data, concept drift, and limited access
to labels. The proposed method combines information from budget
constraints and dynamic class ratios to generate new relevant
instances. We performed experiments on 18 real-world data streams
and 11 semi-synthetic data streams, under different labeling budgets,
in order to evaluate the performance of the proposed method under
a varied set of scenarios. The experimental study showed that our
oversampling method was able to improve the performance of stateof-the-art classifiers for multi-class imbalanced data streams under
strict budgets and outperforms previously proposed oversampling
methods in the domain.


## Usage of BB

Download the pre-compiled jar files or import the project source code into [MOA](https://github.com/Waikato/moa). See the src/main/java/experiments folder to reproduce our research. We use the [MOA framework](https://moa.cms.waikato.ac.nz/) and its class hierarchy. Adding a new algorithm, generator, or evaluator is the same as adding it in MOA (see [MOA documentation](https://moa.cms.waikato.ac.nz/documentation/)).

The package `src/main/java/experiments/active` provides the scripts for the experiments experiments. Use any of the scripts provided at `src/main/java/experiments/active` for the different groups of experiments and add your algorithm, generator, or evaluator. These scripts will generate the command lines used to run the experiments

## Citation
```
@article{aguiar2023active,
  title={An active learning budget-based oversampling approach for partially labeled multi-class imbalanced data streams},
  author={Aguiar, Gabriel J and Cano, Alberto},
  year={2023}
}
```

