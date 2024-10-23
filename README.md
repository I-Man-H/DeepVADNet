# DeepVADNet

Deep Learning Model for Simultaneous Prediction of Quantitative and Qualitative Emotion using Viseual and Bio-sensing Data

## Framework & Architecture

![DeepVADNet Architecture-1](https://github.com/user-attachments/assets/9e51ed03-016e-4b9a-8cfb-f716bbb40640)


## Dependencies
+ Python 3.8
+ PyTorch
+ torchvision
+ numpy
+ pandas
+ Pillow
+ scipy

## Datasets
* Two public datasets have been used in this paper to train and test the model. 
  + DEAP \[1\]: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html
  + MAHNOB-HCI \[2\]: http://mahnob-db.eu/hci-tagging

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
