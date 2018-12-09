# Fake news classification (Mark II)
https://www.kaggle.com/c/fake-news-pair-classification-challenge
* deadline: 2018/12/13
* team: 機器學習小組

# Preprocess
Read csv then return these attribute
* titleD2V
* titleFlag
   
# Run
Decompress the zip
```{terminal}
unzip dataset.zip
```
Execute file run.py
```{python}
train = Preprocess("dataset/train.csv", nrows=168)
```
