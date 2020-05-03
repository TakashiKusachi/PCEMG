import unittest
import pytest
import tempfile
import git
from pathlib import Path

from pcemg.parser.massbank_parser import MassBankParser

class Test_massbank_parser(unittest.TestCase):
    def setUp(self):
        pass

    def test_parser(self):
        temp_path = tempfile.TemporaryDirectory()
        print('')
        try:
            print("temp dir: "+temp_path.name)
            git.Repo.clone_from('https://github.com/MassBank/MassBank-data.git',
                temp_path.name)

            path = Path(temp_path.name)
            file_list = list(path.glob('*/*.txt'))
            print("number of dataset :{}".format(len(file_list)))
            ret = MassBankParser()(file_list)
            print("{}".format(len([None for one in ret if one['instrument_type'] == 'LC-ESI-QTOF'])))

        finally:
            temp_path.cleanup()
        