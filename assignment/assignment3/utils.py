"""
utils.py
This file contains helper functions for Assignment 3.
Author: Sebastian D. Goodfellow
"""

# 3rd party imports
import json


def json_print(obj):
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)
