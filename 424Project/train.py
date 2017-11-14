import tensorflow as tf
import preproc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#depth = 36
#one_hot_labels = []
#one_hot_labels = tf.one_hot(preproc.data_labels, depth))


data_tensor = tf.convert_to_tensor(preproc.data_list)
labels_tensor = tf.convert_to_tensor(preproc.data_labels)

tf.cast(data_tensor, tf.float32)
tf.cast(labels_tensor, tf.float32)


nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500

num_classes = 36

x = tf.placeholder('tf.float32')
y = tf.placeholder('tf.float32')

def network_model(data):
   
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([1024, nodes_layer1])), 'biases':tf.Variable(tf.random_normal([nodes_layer1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([nodes_layer1, nodes_layer2])), 'biases':tf.Variable(tf.random_normal([nodes_layer2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([nodes_layer2, nodes_layer3])), 'biases':tf.Variable(tf.random_normal([nodes_layer3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([hidden_3_layer, num_classes])), 'biases':tf.Variable(tf.random_normal([num_classes]))}


    layer1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3, output_layer['weights']), output_layer['biases'])

    return output


def train_network(x,y):
    prediction = network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 1
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_x = data_array
            epoch_y = labels_array
            c = sess.run([optimizer,cost], feed_dict = {x:epoch_x, y:epoch_y})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', epoch_loss)

        #correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))




train_network(x)

