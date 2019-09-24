# A Pure Keras Implementation of KGCN

This is a pure keras implementation of Knowledge Graph Convolution Network (KGCN) as described in 
the paper [Wang et.al. Knowledge Graph Convolution Networks for Recommender Systems. WWW2019](https://dl.acm.org/citation.cfm?id=3313417). 
Author's tensorflow implementation is available in this [repo](https://github.com/hwwang55/KGCN).  

## Requirements 
- python==3.6
- Keras==2.2.4

## Run

### Prepare data
All the data are copied from author's [repo](https://github.com/hwwang55/KGCN#files-in-the-folder). 
The rating file of MovieLens-20M is still needed to download first:  
```Shell
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
mv ml-20m/ratings.csv raw_data/movie/
```

### Preprocess
```python
python3 preprocess.py
```

The processed data will be stored in `data` dir.

### Train
```python
python3 main.py
```

## Performance