import scipy
from glob import glob
import numpy as np
# from extension import extension


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128), is_zeroMean = True):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.is_zeroMean = is_zeroMean

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        if self.is_zeroMean:
            imgs = np.array(imgs)/127.5 - 1.
        else:
            imgs = np.array(imgs)/255.0

        return imgs

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)

        if self.is_zeroMean:
            img = img/127.5 - 1.
        else:
            img = img/255.0

        return img[np.newaxis, :, :, :]

    def load_full_img(self, path, blockwise = False, is_ext = False):
        img = self.imread(path)

        # if is_ext:
        #     mask = img[:,:,2] > 10
        #     img = extension(img, mask)

        # img = scipy.misc.imresize(img, 0.33)
        # for i in range(0,3):
        #     img[:,:,i] = scipy.ndimage.filters.median_filter(img[:,:,i], 7)

        p = 32
        # print('Input:', img.shape)

        self.padX = (p - img.shape[0] % p)
        self.padY = (p - img.shape[1] % p)
        temp = np.zeros((img.shape[0] + self.padX, img.shape[1] + self.padY, img.shape[2]), dtype=np.uint8)
        temp[:img.shape[0], :img.shape[1], :] = img
        # print('Input: after pad', temp.shape)
        img = temp

        if blockwise:
            self.pad = 80
            temp = self.crop(img, img.shape[0]/2, img.shape[1]/2)
            print('After crop Array shape : ', temp.shape)

            img_bw = np.zeros((4, img.shape[0]/2 + self.pad, img.shape[1]/2 + self.pad, 3), dtype=np.uint8)
            for bs in range(0,4):
                for dp in range(0,3):
                    img_bw[bs,:,:,dp] = np.lib.pad(temp[bs,:,:,dp], (self.pad/2), 'edge')

            img = img_bw
        else:
            img = img[np.newaxis, :, :, :]

        if self.is_zeroMean:
            img = img/127.5 - 1.
        else:
            img = img/255.0

        # print('Input: preprocessed shape ', img.shape, np.max(img), np.min(img), np.mean(img))

        return img

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

    def imwrite(self, path, img, blockwise = False):
        # print('Predit:', path, img.shape)
        # print('Predit: After stitch: ', img.shape, np.max(img), np.min(img), np.mean(img))
        if blockwise:
            img = self.stitch(img)

        if self.is_zeroMean:
            img = 0.5 * img + 0.5

        # img = np.clip(img, 0, np.max([1, np.max(img)-0.2]))
        img = (img - np.min(img))/(np.max(img) - np.min(img))

        img = np.uint8(np.squeeze(img)*255)
        img = img[:-self.padX, :-self.padY, :]
        return scipy.misc.imsave(path, img)

    def crop(self, im, h, w):
        k = 0
        a = np.zeros((4, h, w, 3), dtype=np.uint8)
        for i in range(0,im.shape[0]-h+1, h):
            for j in range(0,im.shape[1]-w+1, w):
                a[k,:,:,:] = im[i:i+h, j:j+w, :]
                k +=1
        return a

    def stitch(self, img):
        img = img[:, self.pad/2:-self.pad/2, self.pad/2:-self.pad/2, :]
        bs,h,w,dp = img.shape
        # print('Predit: After removing pad: ', img.shape)

        temp = np.zeros((h*2, w*2, dp), dtype=np.float32)
        temp[:h,:w,:] = img[0]
        temp[:h,w:,:] = img[1]
        temp[h:,:w,:] = img[2]
        temp[h:,w:,:] = img[3]

        return temp
