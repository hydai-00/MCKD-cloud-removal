import numpy as np
import skimage.io as skio
import logging


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_img(filename, img, cloud_data, sar_vh_data, sar_vv_data, target_img):
    img = np.round((img.copy() * 255.0)).astype('uint8')
    target_img = np.round((target_img.copy() * 255.0)).astype('uint8')
    img = np.concatenate((img, cloud_data, sar_vh_data, sar_vv_data, target_img), axis=1)
    skio.imsave(filename, img)


def write_rslt(filename, img):
    img = np.round((img.copy() * 10000.0)).astype('uint16')
    skio.imsave(filename, img)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def initialize_logger(file_dir):
    """
    Return a logger that accepts INFO and above and is unaffected by anything else.
    """
    # 1. a dedicated logger (any name will do, so long as it does not collide)
    logger = logging.getLogger("my_app_logger")
    logger.handlers.clear()          # drop handlers left over from a previous call
    logger.setLevel(logging.INFO)    # let INFO and above through

    # 2. attach exactly one FileHandler to this logger
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.INFO)  # again: INFO and above only
    logger.addHandler(fhandler)

    # 3. do not propagate to root, whose handlers would emit unrelated records
    logger.propagate = False

    return logger


def record_loss(loss_csv, epoch, epoch_time, lr, train_loss, val_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{}\n'.format(epoch, epoch_time, lr, train_loss, val_loss))
    loss_csv.flush()
    loss_csv.close
