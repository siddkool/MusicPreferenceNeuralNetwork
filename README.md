# Music Preference Neural Network

A feedforward neural network (pure NumPy) that predicts preference across the
five MUSIC dimensions (mellow, unpretentious, sophisticated, intense,
contemporary) from age, empathy (EQ), and systemizing (SQ).

## Run

```
pip install -r requirements.txt
python neural_net.py
```

## Data

`data.tsv` holds survey responses: the five music ratings (1-5), three EQ
items, three SQ items, and age. EQ and SQ are averaged into composite scores;
all inputs are scaled to 0-1.

## Model

Layers 3 -> 32 -> 16 -> 5, ReLU hidden activations, sigmoid output, Adam
optimizer, MSE loss. Backpropagation is implemented by hand. Evaluated with
5-fold cross-validation against a predict-the-mean baseline.

## Results (n=125)

Within 1 point: ~56%. Mean absolute error: ~1.0 point. The model does not beat
the mean baseline out of sample at this sample size, which is expected: the
sample is small and skews young. The pipeline is built to run unchanged on a
larger dataset.
