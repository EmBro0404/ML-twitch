# ML-twitch
The data for machine learning project for classification of 9 twitch streamers based on the chatters. 

Folder "messages" contain the data used for training.

"lookup.csv" contain the tokenization ID rules, not including 0, where 0 is saved for any word not in it.

Folder "frequencies" and "combined_vocabulary_frequency.csv" are interesting but not used in my model.

### What to touch

RUN train_data.R once to save model into your folder.

USE only predict.R to predict chat messsages. 

- train_data.R (Save model)
- predict.R (Predict values)

Do not use any other .R or .py file. These are sourced automatically.

#### Streamers:
  cinna 0,
  dantes 1,
  extraemily 2,
  ishowspeed 3,
  jasontheween 4,
  ludwig 5,
  marlon 6,
  moistcr1tikal 7,
  xqc 8
