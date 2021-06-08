# Convolutional Signature Method
This is the implementation for experiments in CNN-Sig project.
    
## Installment
Install required packages with correct version
    
```
# need to maintain correct versions of packages
pip install -r requirements
```

## Experiments
We implement the CNN-Sig algorithm to different experiments.

### High Dimensional Time Series (HDTS) Classification
Use the following code to run (HDTS) with signature depth 3, training epochs 60, batch size 16. By specifying "--rocket True", we also implement the ROCKET method as a benchmark.

```
# by change the file hdts_classification.py, one can easily add desired 
# time series that want to be classified
python3 hdts_classification.py --depth 3 --epochs 60 --batch_size 16 --rocket True
``` 

### NLP sentimental analysis
First need to download english model from spacy
```
python -m spacy download en_core_web_sm
```
