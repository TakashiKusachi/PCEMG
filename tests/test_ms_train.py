import unittest
import pytest
import tempfile
import git
from pathlib import Path

from pcemg.scripts.ms_train import ms_train

class Test_ms_train(unittest.TestCase):
    def setUp(self):
        pass

    def test_ms_train(self):
        ms_train(vocab_path="/home/kusachi/workspace/jtvae/MS_vocab.txt",
                    dataset_path="/home/kusachi/workspace/jtvae/massbank.pkl",
                    load_model="/home/kusachi/workspace/jtvae/vae_model/model.iter-160000",
                    model_config="./model_config.ini")