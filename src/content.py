import tensorflow as tf


def content_layer_loss(p, x, content_loss_function=1):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def sum_content_losses(sess, net, content_img, config):
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(config.content_layers, config.content_layer_weights):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p, x) * weight
    content_loss /= float(len(config.content_layers))
    return content_loss
