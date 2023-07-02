#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-

# notice, you should add the former path into sys.path. or you will
# not import the former path. but the same directory and the lower directory need not to
# add the path. and you should add the __init__.py file in any directory.
import sys
sys.path.append("..")
from sample.general.general import GeneralOperation


if __name__ == "__main__":
    generalOperation = GeneralOperation()
    result = generalOperation.imshow_absolute_path(0, 0, 1)
    print(result)