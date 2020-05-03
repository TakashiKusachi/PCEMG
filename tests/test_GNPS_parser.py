import unittest
import pytest
import tempfile
import urllib.request
from pathlib import Path
import collections

from pcemg.parser.GNPS_parser import GNPSParser


class Test_GNPS_parser(unittest.TestCase):
    def setUp(self):
        pass

    def test_parser(self):
        temp_path = tempfile.TemporaryDirectory()
        print('')
        try:
            print("temp dir: "+temp_path.name)
            with urllib.request.urlopen('ftp://ccms-ftp.ucsd.edu/Spectral_Libraries/ALL_GNPS.mgf') as u:
                with open(Path(temp_path.name)/"temp.mgf",'wb') as f:
                    f.write(u.read())

            path = Path(temp_path.name)
            file_list = list(path.glob('*.mgf'))
            print("number of dataset :{}".format(len(file_list)))
            ret = GNPSParser()(file_list)
            print(collections.Counter([one['SOURCE_INSTRUMENT'] for one in ret]))

        finally:
            temp_path.cleanup()
        