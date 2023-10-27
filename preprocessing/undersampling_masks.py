from preproccessing.utils.undersamp_utils import *


# Generating poisson masks

generate_masks((200, 256), '/home/timothy/PycharmProjects/undsamp/masks/', count=100)

'''
# Testing the under-sampling
image_path = '/home/timothy/PycharmProjects/undsamp/data-samp/simulated/'
mask_path = '/home/timothy/PycharmProjects/undsamp/masks/'
destin_path = '/home/timothy/PycharmProjects/undsamp/data-samp/undersamp_test/'
undersamp (image_path, mask_path, destin_path)
'''
