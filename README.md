This project started as an attempt to clone the voice of late SP Balasubramaniam as part of my graduation project in 2022. While there are numerous models, both open-source and paid, for speech synthesis in English, we couldn't find pretrained models for Telugu. At that stage, we had three options for my project:
1. Transfer Learning - Use Tacotron-2 trained on English corpus and train it further on Telugu corpus we prepared from SP balasubramaniam's audio clips to clone his voice for Telugu speech. However, we quickly realised this wouldn't work as well as we would have expected since the pretrained English does not capture the diction and the nuances of Telugu.
2. Transfer Learning - Use Tacotron-2 trained on English corpus and train it further on English corpus we prepared from SP Balasubramaniam's audio clips to clone his voice for English speech. We came across multiple works which used Tacotromn-2 to clone personalities such as Donald Trump and Martin Luther King Jr. This was easily achievable but that lead us to realise how there is no TTS for our language of Telugu. Even the existing models use English embeddings and do not produce natural speech.
3. Train a model from base step on Telugu Corpus. While our college management and professor advised us on how training a model from base would be beyond the scope of a graduation project, we were suggested to choose this option given the potential learning curve in it for us.


In an attempt to create a TTS system for Telugu, an Indian language, as part of my graduation project in 2022, I stumbled on Tacotron-2 and the master branch of this repo. While the project didn't reach the required alignment and we are a long way off from synthesizing natural-sounding speech in Telugu (mostly because of lack of quality audio and transcripts when we started this in 2022), it was still a great opportunity to understand the implementation of a Deep Learning Architecture using tensorflow. 

In addition to the training modules present in the repo, we implemented synthesise/inference modules to accept text and generate audio output for the same.

The project also explored making various changes to the architecture of Tacotron-2 to adapt it to Telugu. The language of Telugu is phoenitic i.e., the words are pronunced the way they are written. This is unlike languages such as English and French which are rule-based.
For example, 
1. I 'read' books
2. I 'read' The Origin by Dan Brown.
3. I am now 'reading' The 'Red' Rising.

The pronunciation of the word 'read' in 1 and 2 differs but pronunciation of 'Read' and 'Red' from 2 and 3 is same. This introduces quite a bit of complexity and requires larger dataset for any model to pick up on these subtle variations. The same is not the case with Telugu.

Unfortunately, I lost the source code and this repo is an attempt to redo the project after migrating the existing code to Tensorflow-2.11. The synthesize file will be updated shortly.

References:
1. Documentation and implementation of tacotron 2 in https://github.com/Rayhane-mamah/Tacotron-2 also helped me understand the paper.
2. Tensorflow Documentation. 

# Tacotron 2 Explained

This repository is meant to teach the intricacies of writing advanced Recurrent Neural Networks in Tensorflow. The code is used as a guide, in weekly Deep Learning meetings at Ohio State University, for teaching -
1. How to read a paper
2. How to implement it in Tensorflow

I choose Tacotron 2 because -
1. Encoder-Decoder architectures contain more complexities then standard DNNs. Implementing one helps you master concepts you would otherwise overlook
2. Tachotron 2 was released less than a year ago (as of 2018) and is a relatively simple model (compared to something like GNTM). The associated paper explains the architecture well
3. Other public implementations offer a benchmark to compare results
4. Public datasets are available to achieve state of the art results
4. Training requires ~10 days given access to a GPU (comparable to GTX 1080)

Note: This code has no affiliation with the companies I worked at. I used none of the proprietery knowledge of any of those companies to write this code. This was purely an exercise in self study.

The paper followed in this repository is - [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884). The repository only implements the Text to Mel Spectrogram part (called Tacotron 2). The repository does not include the vocoder used to synthesize audio.

This is a production grade code which can be used as state of the art TTS frontend. The blog post \[TODO\] shows some audio samples synthesized with a Griffin Lin vocoder. But the code has excess comments to aid a novice Tensorflow user which could be a hindrance. To read the code, start from ```train.py```

The repository also uses Tensorflow's tf.data API for pre-processing and \[TODO\] Estimator API for modularity
 

## Directory Structure
The directory structure followed is as specified in [Stanford's CS230 Notes on Tensorflow](https://cs230-stanford.github.io/tensorflow-getting-started.html). We modify the structure a bit to suite our needs.
```
data/ (Contains all data)
model/ (Contains model architecture)
    input_fn.py (Input data pipeline)
    model_fn.py (Main model)
    utils.py (Utility functions)
    loss.py (Model loss)
    wrappers.py (Wrappers for RNN cells)
    helpers.py (Decoder helpers)
    external/ (Code adapted from other repositories)
        attention.py (Location sensitive attention)
        zoneout_wrapper.py (Zoneout)
train.py (Run training)
config.json (Hyper parameters)
synthesize_results.py (Generate Mels from text)
```

## Requirements
The repository uses Tensorflow 1.8.0. Some code may be incompatible with older versions of Tensorflow (specifically the Location Sensitive Attention Wrapper).

## Setup
1. Setup python 3 virtual environment. If you dont have ```virtualenv```, install it with

```
pip install virtualenv
```

2. Then create the environment with

```
virtualenv -p $(which python3) env
```

3. Activate the environment

```
source env/bin/activate
```

4. Install tensorflow

```
pip install tensorflow==1.8.0
```

5. Clone the repository

```
git clone https://gitlab.com/codetendolkar/tacotron-2-explained.git
```

6. Run the training script

```
cd tacotron2
python train.py
```

## Generate Mels from Text

## Synthesize Audio from Mels

## Credits and References
1. "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"
Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu
[arXiv:1712.05884]()
2. Location Sensitive Attention adapted from Tacotron 2 implementation by [Keith Ito](https://github.com/keithito) - [GitHub link](https://github.com/keithito/tacotron/tree/c94ab2757d52e4294dcd6a8da03f49d251b2dec4)
3. Zoneout Wrapper for RNNCell adapted from Tensorflow's official repository for [MaskGan](https://github.com/tensorflow/models/tree/master/research/maskgan). The code contributed by [A Dai](https://github.com/a-dai) - [GitHub link](https://github.com/tensorflow/models/blob/master/research/maskgan/regularization/zoneout.py)
4. And obviously - all the contributors of [Tensorflow](https://github.com/tensorflow)
5. Internet
