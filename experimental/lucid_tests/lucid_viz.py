import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
from lucid.optvis import objectives, param, render, transform
import matplotlib.pyplot as plt

### Sanity check for lucid install
model = models.InceptionV1()
model.load_graphdef()

# rendered
rendered = render.render_vis(model, 'mixed4a_pre_relu:476')

plt.imshow(rendered)

## Load custom model
#TODO: Perhaps use argparse?

