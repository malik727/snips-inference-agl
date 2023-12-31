from __future__ import print_function, unicode_literals

import logging
import sys
from enum import Enum, unique

import requests

import snips_inference_agl
from snips_inference_agl import __about__


@unique
class PrettyPrintLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    SUCCESS = 3


FMT = "[%(levelname)s][%(asctime)s.%(msecs)03d][%(name)s]: %(message)s"
DATE_FMT = "%H:%M:%S"


def pretty_print(*texts, **kwargs):
    """Print formatted message

    Args:
        *texts (str): Texts to print. Each argument is rendered as paragraph.
        **kwargs: 'title' becomes coloured headline. exits=True performs sys
        exit.
    """
    exits = kwargs.get("exits")
    title = kwargs.get("title")
    level = kwargs.get("level", PrettyPrintLevel.INFO)
    title_color = _color_from_level(level)
    if title:
        title = "\033[{color}m{title}\033[0m\n".format(title=title,
                                                       color=title_color)
    else:
        title = ""
    message = "\n\n".join([text for text in texts])
    print("\n{title}{message}\n".format(title=title, message=message))
    if exits is not None:
        sys.exit(exits)


def _color_from_level(level):
    if level == PrettyPrintLevel.INFO:
        return "92"
    if level == PrettyPrintLevel.WARNING:
        return "93"
    if level == PrettyPrintLevel.ERROR:
        return "91"
    if level == PrettyPrintLevel.SUCCESS:
        return "92"
    else:
        raise ValueError("Unknown PrettyPrintLevel: %s" % level)


def get_json(url, desc):
    r = requests.get(url)
    if r.status_code != 200:
        raise OSError("%s: Received status code %s when fetching the resource"
                      % (desc, r.status_code))
    return r.json()


def set_nlu_logger(level=logging.INFO):
    logger = logging.getLogger(snips_inference_agl.__name__)
    logger.setLevel(level)

    formatter = logging.Formatter(FMT, DATE_FMT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)
