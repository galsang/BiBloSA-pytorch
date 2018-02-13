# BiBloSA-pytorch
Re-implementation of [Bi-Directional Block Self-Attention for Fast and Memory-Efficient Sequence Modeling (T. Shen et al., ICLR 2018)](https://openreview.net/pdf?id=H1cWzoxA-) on Pytorch.

## Results

Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| **Re-implementation (600D Bi-BloSAN)**            |   **?**   |  
| Baseline from the paper (480D Bi-BloSAN)          |   85.7    |    

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2
- Pytorch: 0.3.0

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    nltk==3.2.4
    tensorboardX==1.0
    torch==0.3.0
    torchtext==0.2.1


## Training

> python train.py --help

	usage: train.py [-h] [--batch-size BATCH_SIZE] [--block-size BLOCK_SIZE]
                [--data-type DATA_TYPE] [--dropout DROPOUT] [--epoch EPOCH]
                [--gpu GPU] [--learning-rate LEARNING_RATE]
                [--mSA-scalar MSA_SCALAR] [--print-freq PRINT_FREQ]
                [--weight-decay WEIGHT_DECAY] [--word-dim WORD_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --block-size BLOCK_SIZE
      --data-type DATA_TYPE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --learning-rate LEARNING_RATE
      --mSA-scalar MSA_SCALAR
      --print-freq PRINT_FREQ
      --weight-decay WEIGHT_DECAY
      --word-dim WORD_DIM 

**Note:** 

- The two of most important hyperparameters are **block-size** (_r_ in the paper) and **mSA-scalar** (_c_ in the paper).
The paper suggests some heuristics to decide the _r_ (in the Appendix) but there's no mention about _c_.
In this implementation, they are chosen as **_r_=3** and **_c_=5**, following the settings of the authors' code. 
But you can utilize the heuristic for _r_ if you set **_r_ < 0**.
- The Dropout technique also exists in this model, but the strategy for it is not specified. 
Therefore, to be naive, the dropout is adapted to normal fully connected layers, except for attention weights.
- Furthermore, there're no details about what **480D** Bi-BloSAN is, which is suggested in the paper. Hence, the result is only reported for the **600D(300D-Forward + 300D-Backward)** Bi-BloSAN.

## Test

> python test.py --help

	usage: test.py [-h] [--batch-size BATCH_SIZE] [--block-size BLOCK_SIZE]
               [--data-type DATA_TYPE] [--dropout DROPOUT] [--epoch EPOCH]
               [--gpu GPU] [--mSA-scalar MSA_SCALAR] [--print-freq PRINT_FREQ]
               [--word-dim WORD_DIM] --model-path MODEL_PATH

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --block-size BLOCK_SIZE
      --data-type DATA_TYPE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --mSA-scalar MSA_SCALAR
      --print-freq PRINT_FREQ
      --word-dim WORD_DIM
      --model-path MODEL_PATH
 
**Note:** You should execute **test.py** with the same hyperparameters, which are used for training the model you want to run.    
