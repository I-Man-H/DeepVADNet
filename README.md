# DeepVADNet

Deep Learning Model for Simultaneous Prediction of Quantitative and Qualitative Emotion using Viseual and Bio-sensing Data

DeepVADNet is a novel end-to-end deep learning framework designed for multi-modal emotion recognition. This model leverages both visual data (facial expressions) and physiological signals (e.g., EEG, ECG) to predict both quantitative emotions (valence, arousal, dominance) and qualitative emotions (discrete labels like happiness, sadness, etc.) in a single forward pass.

The model achieves state-of-the-art results on popular datasets, including DEAP and MAHNOB-HCI, with significant improvements in emotion recognition accuracy. You can find the full paper with detailed methodologies, experiments, and results in the link below.

You can read the full paper [here](https://www.sciencedirect.com/science/article/pii/S1077314224002029).


## Framework & Architecture

The architecture of our proposed DeepVADNet and the workflow for combined quantitative and qualitative emotion recognition is:

![DeepVADNet Architecture-1](https://github.com/user-attachments/assets/9e51ed03-016e-4b9a-8cfb-f716bbb40640)



## Datasets

In this research, we used two publicly available datasets to train and test DeepVADNet. These datasets provide multi-modal emotion data, including both visual and physiological signals, enabling comprehensive emotion recognition tasks.


| Dataset         | Number of Subjects | Modalities                                        | Signals                                                    | Labels                                   | Access Link |
|-----------------|--------------------|---------------------------------------------------|------------------------------------------------------------|------------------------------------------|-------------|
| **DEAP** \[1\]       | 32                 | EEG, peripheral physiological signals, face videos | 32-channel EEG, EMG, GSR, respiration belt, and more        | Valence, arousal, dominance, liking (9-point scale) | [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html) |
| **MAHNOB-HCI** \[2\] | 27                 | Vision, EEG, ECG, eye gaze, peripheral signals    | 32-channel EEG, ECG, GSR, respiration belt, eye gaze data   | Valence, arousal, dominance, 9 discrete emotions | [MAHNOB-HCI Dataset](http://mahnob-db.eu/hci-tagging) |


## Summary of Results

The comparison of our model with previously established methods are discussed here. Classification accuracy (%) and mean square error (MSE) are presented on the left and right side respectively.

| Study                                        | Modalities          | Features                                               | Valence                  | Arousal                  | Dominance                 |
|----------------------------------------------|---------------------|--------------------------------------------------------|--------------------------|--------------------------|---------------------------|
|                             |                     |         **DEAP Dataset**                                                |                          |                          |                           |
| Keoltra et al. (2012)                        | EEG, Peripheral      | PSD, Statistic                                          | 62.70%                   | 62.00%                   | -                         |
| Tang et al. (2017)                           | Bio-sensing          | Differential entropy, Statistic                         | 83.82%                   | 83.23%                   | -                         |
| Yang et al. (2018)                           | EEG                 | EEG deep learning-based                                 | 90.80%                   | 91.03%                   | -                         |
| Anubhav et al. (2020)                        | EEG                 | EEG band power                                          | 94.69%                   | 93.13%                   | -                         |
| Zhang et al. (2022)                          | EEG, Peripheral      | Bio-sensing deep learning-based                         | 90.46%                   | 93.22%                   | -                         |
| Zhang et al. (2022)                          | EEG                 | EEG transformer-based                                   | 72.89%                   | 77.03%                   | -                         |
| Wang et al. (2022)                           | EEG                 | EEG Spatial Transformer-based                           | 66.51%                   | 65.75%                   | -                         |
| Li et al. (2022)                             | EEG                 | EEG deep learning-based                                 | 97.41%                   | 97.25%                   | 98.35%                    |
| Li et al. (2022)                             | EEG                 | Spatial and frequency domain deep learning              | 95.15%                   | 95.76%                   | 95.64%                    |
| Ding et al. (2022)                           | EEG                 | EEG deep learning-based                                 | 61.27%                   | 60.61%                   | -                         |
| Kang et al. (2022)                           | PPG, GSR            | PPG and GSR deep learning-based                         | 80.25%                   | 81.33%                   | -                         |
| Saffaryazdi et al. (2022)                    | Vision, Bio-sensing  | Face appearance, Bio-sensing deep learning-based        | 62.40%                   | 65.10%                   | -                         |
| Wang et al. (2023)                           | Vision, EEG         | Face appearance, EEG deep learning-based                | 96.63%                   | 97.15%                   | -                         |
| Peng et al. (2023)                           | EEG                 | EEG Temporal transformer-based                          | 95.18%                   | 95.58%                   | 95.78%                    |
| Li et al. (2023)                             | Bio-sensing         | Spatial Frequency Bio-sensing deep learning-based       | 94.99%                   | 95.89%                   | -                         |
| Yuvaraj et al. (2023)                        | EEG, Peripheral     | Statistical, WT, Fractal, Hjorth, HOS                   | 78.18%                   | 79.90%                   | -                         |
| Gong et al. (2024)                           | EEG, Peripheral     | Bio-sensing transformer-based                           | 97.97%                   | 98.02%                   | -                         |
| **This work**                                | Vision, Bio-sensing | Face appearance, Bio-sensing deep learning-based        | **98.89%/5e-4**          | **99.08%/8e-4**          | **98.82%/4e-4**           |

| Study                                        | Modalities          | Features                                               | Valence                  | Arousal                  | Dominance                 |
|----------------------------------------------|---------------------|--------------------------------------------------------|--------------------------|--------------------------|---------------------------|
|                       |                     |        **MAHNOB-HCI Dataset**                                                 |                          |                          |                           |
| Soleymani et al. (2012)                      | Bio-sensing          | PSD, Statistic, etc.                                    | 57.00%                   | 52.40%                   | -                         |
| Huang et al. (2016)                          | Vision, Bio-sensing  | Face appearance, PSD                                    | 57.47%                   | 58.62%                   | -                         |
| Wiem et al. (2017)                           | Bio-sensing          | PSD, Statistic                                          | 68.75%                   | 64.23%                   | -                         |
| Siddharth et al. (2019)                      | Vision, Bio-sensing  | Face appearance & geometric, PSD, Statistic             | 85.49%                   | 82.93%                   | -                         |
| Zhang et al. (2022)                          | EEG                 | EEG transformer-based                                   | 79.90%                   | 81.37%                   | -                         |
| Wang et al. (2022)                           | EEG                 | EEG Transformer-based                                   | 66.63%                   | 66.20%                   | -                         |
| Kaur et al. (2022)                           | EEG                 | EEG deep learning-based                                 | 86.97%                   | 87.07%                   | -                         |
| Ding et al. (2022)                           | EEG                 | EEG deep learning-based                                 | 62.27%                   | 63.75%                   | -                         |
| Lopez et al. (2023)                          | EEG, Peripheral     | Bio-sensing deep learning-based                         | 44.30%                   | 41.56%                   | -                         |
| Yuvaraj et al. (2023)                        | EEG, Peripheral     | Statistical, WT, Fractal, Hjorth, HOS                   | 83.98%                   | 85.58%                   | -                         |
| Ouguz et al. (2023)                          | ECG                 | Morphological, HRV                                      | 78.28%                   | 83.61%                   | -                         |
| Mellouk et al. (2023)                        | Peripheral          | Deep learning-based                                     | 73.33%                   | 60.00%                   | -                         |
| **This work**                                | Vision, Bio-sensing | Face appearance, Bio-sensing deep learning-based        | **89.98%/1.09**          | **88.60%/0.83**          | **88.13%/0.98**           |





## Dependencies
The following environments are required 

+ Python 3.8
+ PyTorch
+ torchvision
+ numpy
+ pandas
+ Pillow
+ scipy


## Dataset Preprocessing
* data_preprocess.py contains the functions used for data pre-process. It also provides a preprocess_demo() to preprocess DEAP dataset.
After running preprocess_demo.py, the face and bio-sensing data of each subject should be compressed to .zip format.    
The final organization should be like follows:    
./data/    
　　-- DEAP/    
　　　　-- faces/    
　　　　　-- s{subject_id}.zip    
　　　　-- bio/    
　　　　　-- s{subject_id}.zip    
　　　　-- labels/    
　　　　　-- participant_ratings.csv    
　　-- MAHNOB/    
　　　　-- faces/    
　　　　　-- s{subject_id}.zip    
　　　　-- bio/    
　　　　　-- s{subject_id}.zip    
　　　　-- labels/    
　　　　　-- mahnob_labels.npy
     
## Demo
Train and test the model using per-subject experiments with the following arguments:

| Arguments| Description | Default |
|---|---|---|
| --modal | Data modality | face_bio |
| --dataset | The dataset used for evaluation | DEAP |
| --task | Emotion Classification Task | VADClassification |
| --epoch | The number of epochs in training| 50 |
| --lr | Learn rate in training| 0.0005 |
| --batch_size | The batch size used in training | 64 |
| --face_feature_size | Face feature size | 16 |
| --bio_feature_size | Bio-sensing feature size| 64 |
| --use_gpu | Use gpu or not | Flase |
| --save_model | Save trained model | True
| --mse_weight | mean squared error weight | 0.01 |


## Citation

If this repository is helpful to you, please consider citing our original paper:

"[Deep learning model for simultaneous recognition of quantitative and qualitative emotion using visual and bio-sensing data](https://www.sciencedirect.com/science/article/pii/S1077314224002029)"

```bibtex
@article{hosseini2024deep,
  title={Deep learning model for simultaneous recognition of quantitative and qualitative emotion using visual and bio-sensing data},
  author={Hosseini, Iman and Hossain, Md Zakir and Zhang, Yuhao and Rahman, Shafin},
  journal={Computer Vision and Image Understanding},
  volume={248},
  pages={104121},
  year={2024},
  publisher={Elsevier}
}

```
## References
\[1\] Koelstra, S., Muhl, C., Soleymani, M., Lee, J.S., Yazdani, A., Ebrahimi, T., Pun,
T., Nijholt, A., Patras, L.: Deap: A database for emotion analysis using physiolog-
ical signals. IEEE Transactions on Affective Computing 3(1), 18–31 (2012)    
\[2\] Soleymani, M., Lichtenauer, J., Pun, T., Pantic, M.: A multimodal database for affect recognition and implicit tagging. IEEE Transactions on Affective Computing 3(1), 42–55 (2012)
