"""

https://codelikes.com/python-version/

"""
#
# python version check
#
print('=' * 50, 'version')
import platform
exp_ver = '3.8.10'
# exp_ver = '3.10.9'

ver = platform.python_version()
if ver[:3] == exp_ver[:3]:
    print("python version ok [{}]".format(exp_ver))
else:
    print("[Error] plrease check python version. should be {}.".format(exp_ver))
print('=' * 50, 'version')

#
# GPU check
#
print('=' * 50, 'CPU')
import tensorflow as tf
tf.config.list_physical_devices('CPU')
print('=' * 50, 'GPU')
tf.config.list_physical_devices('GPU')
print('=' * 50, 'Done')

import sys
import pprint
pprint.pprint(sys.path)