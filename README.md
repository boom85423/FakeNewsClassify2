# Fake news classification (Mark II)
https://www.kaggle.com/c/fake-news-pair-classification-challenge
* deadline: 2018/12/13
* team: 機器學習小組

# Preprocess
Read csv then return these attribute
* titleD2V
* titleFlag
* title_B_Keyword
* [Pickle2](https://drive.google.com/open?id=16VcR1b-W5qhH0hQL6F6_M5Ae6rrVC4LI)
   
# Run
Decompress the zip
```{terminal}
unzip dataset.zip
unzip pickle2.zip
```
Execute file run.py
```{python}
train = Preprocess("dataset/train.csv", nrows=168)
```
