from transformers import Transformer
import tensorflow as tf


xformer = Transformer(3,num_layers=4,num_heads=4,dff=256,d_model=32)
pred = xformer(tf.random.normal((1,24,4)),tf.random.normal((1,6,3)),False)
# xformer.load_weights("xformer.h5")
print(pred.shape)
xformer.summary()