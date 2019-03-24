
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot= True)


# In[6]:


type(mnist)


# In[7]:


mnist.train.images


# In[8]:


mnist.train.num_examples


# In[9]:


mnist.test.num_examples


# In[10]:


mnist.train.images.shape


# In[11]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


single_image = mnist.train.images[1].reshape(28,28)


# In[13]:


plt.imshow(single_image)


# In[28]:


plt.imshow(single_image, cmap="gist_gray")


# In[29]:


single_image.min()


# In[30]:


single_image.max()


# In[31]:


x = tf.placeholder(tf.float32,shape=[None, 784])


# In[32]:


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[33]:


y = tf.matmul(x,W) + b


# In[34]:


y_true = tf.placeholder(tf.float32,[None,10])


# In[36]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits = y))


# In[37]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)


# In[39]:


init = tf.global_variables_initializer()


# In[43]:


with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train,feed_dict = {x:batch_x, y_true:batch_y})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))

