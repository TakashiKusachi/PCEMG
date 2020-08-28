import sys
import os
from logging.config import fileConfig

d = os.path.dirname(sys.modules['pcemg'].__file__)
fileConfig(os.path.join(d,"data/logger_config.ini"))