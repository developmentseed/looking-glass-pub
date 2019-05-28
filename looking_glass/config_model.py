"""
Configuration paramters for exporting a model

@author: Development Seed
"""
import os
import os.path as op

IMG_SIZE = (256, 256, 3)
PCT_CUTOFF = 98.0
N_GPU = 0

model_start_time = '0524_215014'
model_dir = op.join(os.environ['BUILDS_DIR'], 'looking-glass-pub', 'models')
export_dir = op.join(os.environ['BUILDS_DIR'], 'looking-glass-pub',
                     'looking_glass_export', '001')
