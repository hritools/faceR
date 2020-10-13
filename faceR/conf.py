"""Reads the conf file, is further imported by other packages"""

import logging
import logging.config

import yaml


def setup(config_filename, logging_config):
    with open(config_filename, 'r') as ymlfile:
        global cfg
        cfg = yaml.load(ymlfile)

    with open(logging_config, 'r') as stream:
        config = yaml.load(stream)

    logging.config.dictConfig(config)
    logger = logging.getLogger('conf')
    logger.setLevel(logging.INFO)


def get_entry(entry):
    return cfg[entry]
