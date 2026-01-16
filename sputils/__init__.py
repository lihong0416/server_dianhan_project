#!/usr/bin/env python
# -*- coding:utf-8 -*-

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


from .data_definition import *
from .data_utils import *
from .datetime_utils import *
from .files_utils import *
from .ini_utils import *
from .logger_utils import *
from .math_utils import *
from .phy_utils import *
from .read_rui import *
from .rui_params import *
from .url_utils import *
