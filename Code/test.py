import pandas as pd
import json
import torch
import re
import replicate

if __name__ == '__main__':
    parent = 'The parent step that requires mutation is: '
    l = int(3.5 * len(parent.split(' ')))
    print(l)