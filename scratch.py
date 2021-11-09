import numpy as np

def format_input_img(imgs):
    new_imgs = np.empty([imgs.shape[0], imgs.shape[2], imgs.shape[3], imgs.shape[1]], dtype=imgs.dtype)
    for i in range(0, imgs.shape[0]):
        img = imgs[i, :, :, :]
        for j in range(0, imgs.shape[1]):
            band = np.reshape(img[j, :, :], img[0].shape + (1,))
            if (j == 0):
                new_img = band;
            else:
                new_img = np.concatenate((new_img, band), axis=new_img.ndim-1)

        new_imgs[i] = new_img 

    return new_imgs
a = np.arange(36).reshape(2,2,3,3)
print(f"a: {a}")
a_ = format_input_img(a)
print(f"a_: {a_}")
