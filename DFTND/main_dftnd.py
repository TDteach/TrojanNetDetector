# import shutil
import os

os.environ['NOTEBOOK_MODE'] = '1'
# import sys
import torch as ch
# import numpy as np
# import seaborn as sns
# from scipy import stats
# from tqdm import tqdm, tqdm_notebook
# import matplotlib.pyplot as plt
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
# from robustness.tools.constants import CLASS_DICT
# from user_constants import DATA_PATH_DICT
from dataset_wrapper import wrapper

# model name
model = 'cifarpoison'

# read data
# If you want to use other imgs instead of the given samples, then do not comment the following line out
# ds = wrapper()
# matplotlib inline

# Load model

model_kwargs = {
    'arch': 'resnet18',
    'dataset': datasets.CIFAR("cifar10"),
    # 'resume_path': f'./models/{model}.pt',
    'resume_path': None,
    'model_path': '../../checkpoint/box_4x4_resnet18.pth',
}

model_kwargs['state_dict_path'] = 'model'
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()
pass


# Custom loss to maximize activation
def activation_loss(mod, inp, targ):
    _, rep = mod(inp, with_latent=True, fake_relu=True)
    return rep, None


# If you want to use other imgs instead of the given samples, then use choice1
# choice1: clean validation input
# x_batch, y_batch = ds.eval_data.get_next_batch(10,
#                                                          multiple_passes=True)
# x_batch = x_batch / 255.0
# im = ch.from_numpy(x_batch.astype(np.float32).transpose((0, 3, 1, 2))).cuda()


# choice2: using the given img samples
im = ch.load('cifarimg.pt') # [10,3,32,32]

# If you want to use noise inputs instead of the given samples, then use choice3
# If you use noise inputs, please comment out the related lines in vis_tool.py for better visualization
# choice3: using random noise input (together with choice 1)
# im = ch.load('cifar10noise.pt')


for i in range(3, 4):
    # PGD Parameters
    kwargs = {
        'criterion': ch.nn.CrossEntropyLoss(),
        'custom_loss': activation_loss,
        'constraint': 'inf',
        'eps': 100,
        'step_size': 0.1,
        'iterations': 1000,
        'targeted': False,
        'gamma': 0.000001 * (10 ** i)
    }
    # Add features to seed images

    # recovered images
    outputs, im_feat = model(im, 0, make_adv=True, fake_relu=True, **kwargs)
    show_image_row([im_feat.detach().cpu()], [f'Activation {0}'], fontsize=18, baseline=[im.cpu()])
    _, predicted = outputs.max(1)
    # print out the logits of the recovered images
    print(outputs)
    output_rec = outputs

    # seed images
    outputs, im_feat = model(im, 0, make_adv=False, **kwargs)
    _, predicted = outputs.max(1)
    print(predicted)
    # print out the logits of the original images
    print(outputs)
    output_ori = outputs

    log_increase = ch.mean(output_rec - output_ori, 0)
    print(log_increase)

    # check whether the model is a Trojan model and the target label
    T = 100  # a preset threshold
    target = -1
    for i in range(10):
        if log_increase[i] > T:
            print('The model is a Trojan model and the target label is: {}'.format(i))
            target = i
    if target == -1:
        print('The model is a clean model')