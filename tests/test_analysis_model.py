import unittest
import pytest
import tempfile
import git
from pathlib import Path

from pcemg.scripts.analysis_model import analysisModel

class Test_analysisModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_analysisModel(self):
        analysisModel(trained_result_path="/home/kusachi/workspace/jtvae/result-20200316-0255.tar.gz")