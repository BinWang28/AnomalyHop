# AnomalyHop: An SSL-based Image Anomaly Localization Method

Implementation for our paper. The paper can be found at [AnomalyHop](https://arxiv.org/).

## 1. Install Requirements / Environmen

We provide the environment setup information for virtual environment (anacodna). The environment can be easily install with (conda required):
```
conda env create -f environment.yml
conda activate AnomalyHop
```

## 2. Dataset Preparation

MVTec AD datasets can be downloaded from: [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

It can also be downloaded by using our following commands:
```
cd datasets
wget ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
cd ..
```

## 3. Reproduce our result

An example for carpet class:
```
python ./src/main.py --kernel 7 6 3 2 4 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.2 0.4 0.5 0.1 --class_names carpet
```

To reproduce all results from the paper:
```
chmod +x run.sh
./run.sh
```

## 4. Our performance


## 5. Citation

If you find our model is useful in your research, please consider cite our paper: [AnomalyHop: An SSL-based Image Anomaly Localization Method](https://arxiv.org/):

```
@article{anomalyhop,
    title = {{AnomalyHop}: An SSL-based Image Anomaly Localization Method},
    author = {tobe filled},
    journal={tobe filled},
    year={2021}
}
```


Contact person: Bin Wang, bwang28c@gmail.com
http://mcl.usc.edu/

## Acknowledgement

The code is partially adapted from [PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)

