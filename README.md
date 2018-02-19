# BiBloSA-pytorch
Re-implementation of [Bi-Directional Block Self-Attention for Fast and Memory-Efficient Sequence Modeling (T. Shen et al., ICLR 2018)](https://openreview.net/pdf?id=H1cWzoxA-) on Pytorch.

## Results

Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| **Re-implementation (600D Bi-BloSAN)**            |   **84.1**   |  
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

- The two of most important hyperparameters are **block-size** (**_r_** in the paper) and **mSA-scalar** (**_c_** in the paper).
The paper suggests a heuristic to decide the **_r_** (in the Appendix) but there's no mention about **_c_**.
In this implementation, **_r_ is computed by the suggested heuristic** and **_c_** is set to **5**, following the settings of the authors.
But you can also assign values to them manually.
- The Dropout technique also exists in this model, but it is not specified that how the dropout is applied. 
Therefore, to be naive, the dropout is adapted to layers for SNLI (**NN4SNLI** class) only.
- Furthermore, there're no details about **480D Bi-BloSAN**, whose result is reported in the paper. 
Hence, the result reported here is based on **600D(300D-Forward + 300D-Backward) Bi-BloSAN**. 
Note that hyperparameter tuning hasn't been done thoroughly. The result can be improved with fine-tuning.

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
