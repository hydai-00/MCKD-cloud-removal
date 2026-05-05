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
    返回一个只接收 INFO 级别、且不受外部干扰的 logger。
    """
    # 1. 创建“专属” logger（名字随便取，不与根 logger 冲突即可）
    logger = logging.getLogger("my_app_logger")
    logger.handlers.clear()          # 防止重复 addHandler
    logger.setLevel(logging.INFO)    # 只让 >= INFO 的日志进来

    # 2. 只给这个 logger 加一个 FileHandler
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.INFO)  # 再次确认只收 INFO 及以上
    logger.addHandler(fhandler)

    # 3. 禁止向上传播到 root，避免 root 的 handler 输出别的日志
    logger.propagate = False

    return logger


def record_loss(loss_csv, epoch, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{}\n'.format(epoch, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
