# example-1.basic.py
import tensorflow as tf
from datetime import datetime

LOG_PATH = './tmp/example-1/' + datetime.now().isoformat()

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')

y = tf.add(a, b, name='y')

sess = tf.Session()

tf.summary.scalar('Value of y', y)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_PATH, graph=sess.graph)

_a = 1
_b = 2
for i in range(10):
    # perform calculation 
    summary, _y = sess.run([merged, y], feed_dict={a: _a, b: _b})
    writer.add_summary(summary, i)
    
    _a = _b
    _b = _y
    
    print(_y)

# tensorboard --logdir=./tmp/example-1 --port=8001
