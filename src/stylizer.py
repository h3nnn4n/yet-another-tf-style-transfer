import tensorflow as tf
import numpy as np
import time

import utils
import model
import content
import style
import images


def run(config):
    content_img = images.get_content_image(config.content_image_name)
    style_imgs = images.get_style_images(content_img, [config.style_image_name])
    with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = images.get_noise_image(1.0, content_img)

        tick = time.time()
        stylize(content_img, style_imgs, init_img, config)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))

    print(config.output_name)
    #Image(filename=output_name)


def stylize(content_img, style_imgs, init_img, config):
    with tf.device(config.device), tf.Session() as sess:
        net = model.build_model(content_img)

        L_style = style.sum_style_losses(sess, net, style_imgs, config)
        L_content = content.sum_content_losses(sess, net, content_img, config)
        L_tv = tf.image.total_variation(net['input'])

        alpha = config.content_weight
        beta  = config.style_weight
        theta = config.tv_weight

        L_total  = alpha * L_content + beta * L_style + theta * L_tv

        optimizer = get_optimizer(L_total, config)

        if config.optimizer_to_use == 'adam':
            minimize_with_adam(sess, net, optimizer, init_img, L_total, config)
        elif config.optimizer_to_use == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer, init_img, config)

        output_img = sess.run(net['input'])

        if config.original_colors:
            output_img = images.convert_to_original_colors(np.copy(content_img), output_img)

        utils.write_image_output(output_img, content_img, style_imgs, init_img)


def minimize_with_lbfgs(sess, net, optimizer, init_img):
    if True:
        print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)


def minimize_with_adam(sess, net, optimizer, init_img, loss, config):
    if True:
        print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')

    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < config.max_iterations):
        sess.run(train_op)
        if iterations % 50 == 0 or iterations == config.max_iterations - 1:
            curr_loss = loss.eval()
            print("At iterate {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1


def get_optimizer(loss, config):
    print_iterations = 50
    if config.optimizer_to_use == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B',
            options={'maxiter': config.max_iterations, 'disp': print_iterations})
    elif config.optimizer_to_use == 'adam':
        optimizer = tf.train.AdamOptimizer(10)
    return optimizer
