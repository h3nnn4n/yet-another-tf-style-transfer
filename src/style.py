import tensorflow as tf


def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))

    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)

    return G


def sum_style_losses(sess, net, style_imgs, config):
    total_style_loss = 0.
    weights = [1.0]

    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.

        for layer, weight in zip(config.style_layers, config.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight

        style_loss /= float(len(config.style_layers))
        total_style_loss += (style_loss * img_weight)

    total_style_loss /= float(len(style_imgs))

    return total_style_loss
