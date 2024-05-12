# Configuration

### create a separate environment

- miniconda:

```
conda create -n py310 python==3.10.13 -y
conda activate py310

```

### install packages

```
pip install --upgrade pip
pip install -r requirements.txt

```

### dataset

- get dataset: 
[Automated Cardiac Diagnosis Challenge (ACDC)](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html)

```
- database/
    - training/*
    - testing/*
```

you need change `data_acdc.py` to make png images currectly. 

`data_acdc.py`: `__main__`


### config

change config file to train.
`traincfg.py`


### start train 

1. use cmd line

```
python main.py

```

### dir structure

```
- acdc_aDl2apple
    - model
    - img
    - logdir
    - metricdata
    - *.py

```


