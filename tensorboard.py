import tensorflow as tf

# X and Y data
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

with tf.name_scope("layer") as scope:
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    w_hist = tf.summary.histogram("weight", W)
    b_hist = tf.summary.histogram("bias", b)

    # XW+b
    hypothesis = X * W + b
    hypothesis_summ = tf.summary.histogram("hypothesis", hypothesis)

# cost function
with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    cost_summ = tf.summary.scalar("cost", cost)

# Minimize cost
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

with tf.Session() as sess:
    # tensorboard --logdir=./logs/
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/")
    writer.add_graph(sess.graph)  # Show the graph

    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    print("step\tcost\t\tW\t\tb")
    # Fit the line
    for step in range(2001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
        writer.add_summary(summary, global_step=step)


    # Testing our model
    print("\nPred 1:", sess.run(hypothesis, feed_dict={X: [5]}))
    print("Pred 2:", sess.run(hypothesis, feed_dict={X: [2.5]}))
    print("Pred 3:", sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
