
# coding: utf-8

# # Single Model Interface

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:
print("Starting")

import tensorflow as tf
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')


import analysis
import pipeline
import model

tf.logging.set_verbosity(0)

OUT = "out/"

# #### Data Parameters

# In[3]:


# file_names = ['data/may052018_float32_kernel68.tfrecord']
# tiff_files =  [
#     "data/2017-01-01_MOD09GA_background_removal_zero_inputated_image_with_cf_50perc_grid_size10-0000017664-0000000000.tif",
# ]
tiff_files = ["foo.tif"]
img_width = 64
n_bands = 7


# In[4]:


# data = pipeline.load_dataset(file_names, img_width, n_bands)
data = (tf.data.Dataset
        .from_generator(
            pipeline.read_tiff_gen(tiff_files, img_width),
            tf.float32,
            (img_width, img_width, n_bands)
        )
        .apply(tf.contrib.data.shuffle_and_repeat(100))
       )


# In[5]:

print("Starting session to extract some examples")

examples = 10
x = data.make_one_shot_iterator().get_next()
xs = []
with tf.Session() as sess:
    for _ in range(examples):
        print(".")
        xs.append(sess.run(x))

print("Samples extracted, making subplot")
fig, ax = plt.subplots(figsize=(20,20), nrows=examples, ncols=n_bands)

for i in range(examples):
    for j in range(n_bands):
        ax[i][j].imshow(xs[i][:, :, j])

print("Subplots made, saving")
fig.savefig(OUT + "examples.png")



# #### Training Parameters

# In[6]:


optimizer = 'adam'
loss = 'mean_squared_error'
metrics = ['accuracy']


# In[7]:

print("compiling autoencoder")

en, ae = model.autoencoder((img_width, img_width, n_bands))

en.compile('adam', loss='mse')
ae.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

ae.summary()


# In[ ]:
print("Model compiled. Fitting...")

history = ae.fit(
    x=data.zip((data, data)).batch(32),
    epochs=10,
    steps_per_epoch=700,
    verbose=2,
)

print("Model fit. Pulling data back...")

# ##### Get data

# In[157]:


x = (data
    .batch(700)
    .make_one_shot_iterator()
    .get_next())

with tf.Session() as sess:
    x = sess.run(x)

print("Data back in numpy. Predicting with autoencoder...")

# # In[158]:


en = tf.keras.models.Model(
    inputs=ae.input,
    outputs=ae.get_layer('conv2d_3').output
)
e = en.predict(x)
y = ae.predict(x)

print("encoded and decoded states in numpy. Plotting pca projections")

# # Analysis

# #### AE results
# Each column is a band, each pair of rows is the input and autoencoded output.

# In[184]:


analysis.plot_ae_output(x, y, 2, n_bands)
plt.savefig(OUT + "ae_output.png")

print("ae output saved. Plotting kmeans and encoded state pca projected")

# ### PCA Projection
# * Below are the images projected onto the two principle components of the encoded space.
# * The solid points represent clusters found by K Means (also projected down to principle components)
# * Note that K-Means performs poorly in high dimensions

# In[251]:




# In[ ]:


K = 3 # Number of means


# In[250]:

e_ = e - e.mean(axis=)

pc = analysis.PCA(e_)
proj = pc.project(e_, 2)
codebook, distortion = kmeans(e_, K)

fig , ax = analysis.img_scatter(proj, x.mean(axis=3), zoom = 0.5)

xs, ys = pc.project(codebook,2).transpose()

ax.scatter(xs, ys, s=100, c='r', zorder=1000)

fig.savefig(OUT + "pca.png")
print("Image saved")


# #### T-SNE

# In[199]:


# get_ipython().run_line_magic('pinfo', 'vq.kmeans')
