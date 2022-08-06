## Translator

This is my implementation of the Transformer architecture, introduced on Vaswani et al. [2017], on a translation task.

### Getting started

For using this code, clone the repository and install the requirements on a virtual environment.

```bash
git clone git@github.com:eramdiaz/translator.git
cd translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Project structure.

The repository consists on four folders: 
- `translator` contains the python files where all the layers, models, trainers and data classes used are defined.
- `data` contains trained tokenizers ready to use inside the `tokenizers` folder and training data. Currently, the only training set which is available is the `IWSLMT2016` dataset.
- `checkpoints` contains trained models available to load and use. These models are organized in folders whose names follow the format `$language_model_translates_from-$language_model_translates_to$other_info-$date`.
- `tests` contains the tests for the project code. 

### Structure of the code

The folder `translator`, where all the functions and classes needed for train and apply a translator are defined, contains the files.
- `tokenizer.py`: defines the functions `load_tokenizer` and `train_tokenizer` which load and train respectively a `sentencepiece.SentencePieceProcessor` tokenizer.
- `blocks.py`: defines the individual layers which build the decoder and encoder cells of the Transformer.
- `learning_rate.py`: contains classes that allow a dinamic learning rate during the training. The class `translator.learning_rate.WarmUpLr` implements the learning rate scheldule proposed on Vaswani et al. [2017].
- `data.py`: contains a subclass of `torch.utils.data.Dataset` adapted to the translator training task.
- `models.py`: this file defines the Transformer architecture and its encoder and decoder cells.
- `train.py`: contains the `Trainer`, a trainer class for a `translator.models.Transformer` object and implements a bleu score method.
- `tools.py`: defines utilities for easily instantiating, training and loading models.

### Basic use

#### How to load a model and use it

The way to load a model is to pass the folder where it is stored to the function `translator.tools.load_model`. At the moment, the only translator available in the repository is `checkpoints/en-de-base-2807`, which is a english-german translator that achieves a bleu score of 0.19 on the dataset `IWSLT1026`. Once we have loaded a translator, we can apply it by inputing a sentence as a string to its `predict` method. 

```python
>>> from translator.tools import load_model
>>> model = load_model('checkpoints/en-de-base-2807')
Loading model checkpoints/en-de-base-2807, which achieves a bleu score of 0.1939 on IWSLT2016
Loading tokenizer...
>>> model.predict('This is a sentence for the translator')
'Das ist ein Satz für den Übersetzer'
```

### How to instantiate a standard Transformer model

The function `translator.tools.get_standard_model` creates a Transformer architecture with the default parameters proposed on Vaswani et al. [2017]. The only different is a vocabulary size of 12000 instead of 38000.

```python
>>> from translator.tools import get_standard_model
>>> model = get_standard_model()
Loading tokenizer...
```

### How to train a translator

The easiest way to train a model is to get a `translator.train.Trainer` object by calling the function `translator.tools.get_standard_trainer` and invoque its `train` method as shown below.

```python
>>> from translator.tools import get_standard_trainer
>>> trainer = get_standard_trainer()
Loading tokenizer...
>>> trainer.train()
Starting training
...
```

It's also possible to train a model with customized data, model or settings by determining the arguments of `translator.tools.get_standard_trainer` or `translator.train.Trainer`.

