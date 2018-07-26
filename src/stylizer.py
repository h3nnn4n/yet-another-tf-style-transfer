import tensorflow as tf
import time

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
        stylize(content_img, style_imgs, init_img)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))

    print(config.output_name)
    #Image(filename=output_name)


def stylize(content_img, style_imgs, init_img, config):
    with tf.device('/gpu:0'), tf.Session() as sess:
        net = model.build_model(content_img)

        L_style = style.sum_style_losses(sess, net, style_imgs)
        L_content = content.sum_content_losses(sess, net, content_img)
        L_tv = tf.image.total_variation(net['input'])

        alpha = config.content_weight
        beta  = config.style_weight
        theta = config.tv_weight

        L_total  = alpha * L_content + beta * L_style + theta * L_tv

        optimizer = get_optimizer(L_total)

        if optimizer_to_use == 'adam':
            minimize_with_adam(sess, net, optimizer, init_img, L_total)
        elif optimizer_to_use == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer, init_img)

        output_img = sess.run(net['input'])

        if original_colors:
            output_img = convert_to_original_colors(np.copy(content_img), output_img)

        write_image_output(output_img, content_img, style_imgs, init_img)


def minimize_with_lbfgs(sess, net, optimizer, init_img):
    if True:
        print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)


def minimize_with_adam(sess, net, optimizer, init_img, loss, max_iterations):
    if True:
        print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')

    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < max_iterations):
        sess.run(train_op)
        if iterations % 50 == 0 or iterations == max_iterations - 1:
            curr_loss = loss.eval()
            print("At iterate {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1


def get_optimizer(loss, max_iterations, optimizer_to_use='adam'):
    print_iterations = 50
    if optimizer_to_use == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B',
            options={'maxiter': max_iterations, 'disp': print_iterations})
    elif optimizer_to_use == 'adam':
        optimizer = tf.train.AdamOptimizer(10)
    return optimizer
