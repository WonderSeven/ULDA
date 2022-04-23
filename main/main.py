'''
@ Author: Tiexin
@ email: tiexinqin@163.com
@Data: 2019-8-14
'''

from engine.configs.parser import BaseOptions
# import engine.fsl_trainer as trainer
import engine.ssl_trainer as trainer

import sys
sys.dont_write_bytecode = True

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

if __name__ == '__main__':
    # Load experiment setting
    opts = BaseOptions().opts
    trainer = trainer.Trainer(opts)
    trainer.train()



