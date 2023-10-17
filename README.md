# RULSTM - Extended for Benchmarking with FUTR

This repository contains a modified version of the RULSTM model. By adding functionalities and conducting  experiments, we aim to compare the capabilities of LSTM (RULSTM) with Transformer architectures (FUTR) in predicting human actions.

Our main goal was to compare LSTM with Transformer frameworks in action anticipation, exploring both through egocentric and third-person datasets. Additionally, we've attempted to extend RULSTM's functionality by integrating the i3d modality, which was originally used by the FUTR model.

The adapted FUTR repository used for comparison with RULSTM can be found here: https://github.com/lucas-ps/FUTR.

## Modifications

**i3d Modality**: RULSTM is designed only to use the TSN modality for RGB features. This project also implements the i3d modality to allow for fair comparison with FUTR.

**Data Compatibility**: RULSTM was designed for the Epic-Kitchens and EGTEA datasets. Significant modifications were added to enable loading of features and action labels from the Breakfast and 50-Salads datasets.

**Feature and action label generation**: To generate the TSN features for 50-Salads and Breakfast, the provided FEATEXT feature extraction code was modified, and an action label converter was implemented.

## Usage

- Begin by extracting TSN features using `extract_rgb.py` from the FEATEXT repository.
- For action label conversion, use `convert_action_labels.py`.
- Use the following terminal commands can be used for testing and training the models.

Feature extraction and labed conversion:
```
python extract_rgb.py --dataset_path "/50-Salads/" --dataset 50-Salads --generate_jpgs --extract_features 
python extract_rgb.py --dataset_path "/BreakfastII_15fps_qvga_sync/" --dataset Breakfast --generate_jpgs --extract_features
```

Model training:
```
python main.py train data/50-salads models/50-salads --modality rgb --task anticipation --sequence_completion 
python main.py train data/50-salads models/50-salads --modality rgb --task anticipation 

python main.py train data/breakfast models/breakfast --modality rgb --task anticipation --sequence_completion 
python main.py train data/breakfast models/breakfast --modality rgb --task anticipation 

python main.py train data/ek55 models/ek55 --modality rgb --task anticipation --sequence_completion 
python main.py train data/ek55 models/ek55 --modality rgb --task anticipation --ek100 
```

Model testing/validation:
```
python main.py validate data/50-salads models/50-salads --modality rgb --task anticipation 
python main.py test data/50-salads models/50-salads --modality rgb --task anticipation --json_directory jsons/50-salads

python main.py validate data/breakfast models/breakfast --modality rgb --task anticipation 
python main.py test data/breakfast models/breakfast --modality rgb --task anticipation --json_directory jsons/breakfast

python main.py validate data/ek55 models/ek55 --modality rgb --task anticipation 
python main.py test data/ek55 models/ek55 --modality rgb --task anticipation --json_directory jsons/ek55
```

## Results Summary

**Training Duration**:
- 50-Salads: 7 minutes
- Breakfast: 20 minutes
- Epic-Kitchens: 50 minutes

When compared to the transformer model FUTR, it was observed that the LSTM-based model RULSTM was about 4x faster in training 50-Salads and Breakfast datasets using the same TSN features.

**Anticipation Results**:

**Breakfast Dataset**: The RULSTM model achieved varying accuracy rates, peaking at 10.46% for Top-1 Accuracy, 30.88% for Top-5 Accuracy, and 89.98% for Mean Top-5 Recall.
   
**50-Salads Dataset**: The results indicated a gap between Top-1 and Top-5 accuracy, suggesting the correct action might not always be the top choice, but is still among the top five possibilities.
   
**Epic-Kitchens-55 Dataset**: This dataset produced slightly lower accuracies, likely due to the increased complexity of egocentric data. Notably, the Top-5 recall is generally higher than the Top-1 precision across the datasets, demonstrating the model's strength in considering a broad set of potentially relevant actions over making a precise top prediction.

**Comparison with FUTR**: 
RULSTM outperformed FUTR on the Breakfast dataset, while FUTR (using i3d features) excelled over RULSTM on the 50-Salads dataset. When using FUTR, i3d features consistently delivered better performance than TSN features across datasets.

For a more comprehensive analysis, please refer to the main paper.

## Acknowledgements

We would like to express gratitude towards the creators of the original FUTR and RULSTM models. Our modifications are built upon their foundational work.

We would also like to acknowledge the extensive support from our supervisor - Dr Sareh Rowlands of the Univeristy of Exeter
