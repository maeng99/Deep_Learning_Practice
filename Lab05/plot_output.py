import numpy as np
import matplotlib.pyplot as plt

# Visualization function
def plot_latent_space(vae, n=15, figsize=15):
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.get_layer("decoder").predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    plt.xticks(pixel_range, np.round(grid_x, 1))
    plt.yticks(pixel_range, np.round(grid_y, 1))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


# 생성된 이미지 시각화
def gan_images(generator, epoch, noise_dim, rows=5, cols=5):
    noise = np.random.normal(0, 1, (rows * cols, noise_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = gen_imgs.reshape(rows * cols, 28, 28)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

# 결과 시각화
def plot_denoising_results(model, noisy_data, clean_data):
    decoded_imgs = model.predict(noisy_data)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(noisy_data[i].reshape(28, 28), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(clean_data[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
    plt.show()