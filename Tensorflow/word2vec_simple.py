import collections
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


def get_dictionaries(words):
    # Count/Frequency of Each Word.
    count = collections.Counter(words).most_common()
    # Build Dictionary
    # <List> Word of Most Frequent word in descending order
    rdic = [i[0] for i in count]
    # Giving rank to each of those most frequent words. Rank starts with 0.
    # E.g. Most frequent word has a rank 0
    dic = {w: i for i, w in enumerate(rdic)}
    # Vocabulary Size: Number of Unique words
    voc_size = len(dic)
    return count, rdic, dic, voc_size


def get_cbow_pairs(data, window_size):
    cbow_pairs = []
    for i in range(window_size, len(data) - window_size):
        cbow_pairs.append([[data[i-1], data[i+1]], data[i]])
    return cbow_pairs


def get_skip_gram_pairs(cbow_pairs):
    skip_gram_pairs = []
    for c in cbow_pairs:
        skip_gram_pairs.append([c[1], c[0][0]])
        skip_gram_pairs.append([c[1], c[0][1]])
    return skip_gram_pairs


def generate_batch(skip_gram_pairs, size):
    assert size < len(skip_gram_pairs)
    x_data, y_data = [], []
    r = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)
    for i in r:
        x_data.append(skip_gram_pairs[i][0])    # n dim
        y_data.append([skip_gram_pairs[i][1]])  # n, 1 dim
    return x_data, y_data

if __name__ == "__main__":
    # Configuration
    batch_size = 20
    embedding_size = 2
    num_sampled = 15   # Number of negative examples to sample

    # Sample sentences
    sentences = ["the quick brown fox jumped over the lazy dog",
                 "I love cats and dogs",
                 "we all love cats and dogs",
                 "cats and dogs are great",
                 "sung likes cats",
                 "she loves dogs",
                 "cats can be very independent",
                 "cats are great companions when they want to be",
                 "cats are playful",
                 "cats are natural hunters",
                 "It's raining cats and dogs",
                 "dogs and cats are animals",
                 "Cats do not like dogs"]

    # Split sentences to words
    words = " ".join(sentences).split()

    count, rdic, dic, voc_size = get_dictionaries(words)

    # Make indexed word data (ordered). The words of the sentences are
    # replaced by their ranks
    data = [dic[word] for word in words]

    # Prepare Training data
    # Continuous Bag of Words Pairs.
    # E.g. ([the, brown], quick), ([quick, fox], brown), ([brown, jumped],
    # fox), ...
    cbow_pairs = get_cbow_pairs(data, window_size=1)

    # Skip Gram Pairs.
    # E.g. (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
    skip_gram_pairs = get_skip_gram_pairs(cbow_pairs)

    x, y = generate_batch(skip_gram_pairs, 3)

    # Input Data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Train
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        embedding_layer = tf.nn.embedding_lookup(embeddings, train_inputs)  # lookup table

    weights = tf.Variable(tf.random_normal([voc_size, embedding_size], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([voc_size]))

    # Compute average NCE loss for the batch
    loss = tf.reduce_mean(tf.nn.nce_loss(weights, biases, embedding_layer, train_labels, num_sampled, voc_size))

    # Optimiser
    opt = tf.train.AdamOptimizer().minimize(loss)

    # Running the Graph
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for step in range(3000):
            batch_inputs, batch_labels = generate_batch(skip_gram_pairs, batch_size)
            _, loss_val = sess.run([opt, loss], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
            if step % 200 == 0:
                print("Step: {}, Loss: {}").format(step, loss_val)
        trained_embeddings = embeddings.eval()

    # Plot Results
    # Show word2vec if dim is 2
    if trained_embeddings.shape[1] == 2:
        labels = rdic[:20]  # Show top 20 words
        for i, label in enumerate(labels):
            x, y = trained_embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom')
        plt.show()
