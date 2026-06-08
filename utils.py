import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import sys
import logging
import logging.handlers
from matplotlib import pyplot as plt

from dataset.artifact_curriculum import get_artifact_curriculum_lr_scale


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(stream_handler)

    return logger


def format_section(title, width=80):
    title = f' {title} '
    return title.center(width, '=')


def log_section(logger, title):
    logger.info(format_section(title))


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_section(logger, 'Config Info')
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR',
                          'ArtifactCurriculumLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
                epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'ArtifactCurriculumLR':
        warm_up_epochs = getattr(config, 'warm_up_epochs', 20)
        eta_min = getattr(config, 'eta_min', 1e-5)
        base_lr = max(float(getattr(config, 'lr', 1e-3)), 1e-12)
        min_factor = eta_min / base_lr

        def lr_func(epoch):
            effective_epoch = max(epoch + 1, 1)
            if warm_up_epochs > 0 and effective_epoch <= warm_up_epochs:
                return effective_epoch / warm_up_epochs
            denom = max(config.epochs - warm_up_epochs, 1)
            progress = min(max((effective_epoch - warm_up_epochs) / denom, 0.0), 1.0)
            cosine = min_factor + (1.0 - min_factor) * 0.5 * (math.cos(progress * math.pi) + 1.0)
            return cosine * get_artifact_curriculum_lr_scale(effective_epoch)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png')
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, data):
        if len(data) == 2:
            out, target = data
            bcediceloss = self.bcedice(out, target)
            return bcediceloss
        elif len(data) == 3:
            gt_pre, out, target = data
            bcediceloss = self.bcedice(out, target)
            gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
            gt_loss = (self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 +
                       self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 +
                       self.bcedice(gt_pre1, target) * 0.5)
            return bcediceloss + gt_loss


class SimpleBceDiceLoss(nn.Module):
    """
    Direct (pred, target) criterion for models without deep supervision.
    Scale factor 2.5 matches the effective loss magnitude of GT_BceDiceLoss
    with its deep-supervision weights (1 + 0.5+0.4+0.3+0.2+0.1 = 2.5).
    """

    def __init__(self, wb=1, wd=1, scale=2.5):
        super().__init__()
        self.bcedice = BceDiceLoss(wb, wd)
        self.scale = scale

    def forward(self, pred, target):
        return self.scale * self.bcedice(pred, target)


class SoftBoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super(SoftBoundaryLoss, self).__init__()
        self.kernel_size = int(kernel_size)
        self.smooth = 1.0

    def _boundary(self, x):
        padding = self.kernel_size // 2
        dilation = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=padding)
        erosion = -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=padding)
        return torch.clamp(dilation - erosion, 0.0, 1.0)

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        pred_boundary = self._boundary(pred)
        target_boundary = self._boundary(target)

        size = pred.size(0)
        pred_flat = pred_boundary.view(size, -1)
        target_flat = target_boundary.view(size, -1)
        intersection = pred_flat * target_flat
        dice = (2 * intersection.sum(1) + self.smooth) / (
            pred_flat.sum(1) + target_flat.sum(1) + self.smooth
        )
        return 1 - dice.mean()


class SoftShapeConsistencyLoss(nn.Module):
    def __init__(self, kernel_size=15):
        super(SoftShapeConsistencyLoss, self).__init__()
        self.kernel_size = int(kernel_size)
        self.smooth = 1.0

    def _smooth_shape(self, x):
        padding = self.kernel_size // 2
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=padding)

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        pred_shape = self._smooth_shape(pred)
        target_shape = self._smooth_shape(target)

        mse = F.mse_loss(pred_shape, target_shape)
        size = pred.size(0)
        pred_flat = pred_shape.view(size, -1)
        target_flat = target_shape.view(size, -1)
        intersection = pred_flat * target_flat
        dice = 1 - ((2 * intersection.sum(1) + self.smooth) / (
            pred_flat.sum(1) + target_flat.sum(1) + self.smooth
        )).mean()
        return mse + dice


class BoundaryShapeGT_BceDiceLoss(nn.Module):
    def __init__(
        self,
        wb=1,
        wd=1,
        boundary_weight=0.10,
        shape_weight=0.05,
        boundary_kernel_size=5,
    ):
        super(BoundaryShapeGT_BceDiceLoss, self).__init__()
        self.base = GT_BceDiceLoss(wb=wb, wd=wd)
        self.boundary = SoftBoundaryLoss(kernel_size=boundary_kernel_size)
        self.shape = SoftShapeConsistencyLoss(kernel_size=max(boundary_kernel_size * 3, 15))
        self.boundary_weight = float(boundary_weight)
        self.shape_weight = float(shape_weight)

    def forward(self, data):
        base_loss = self.base(data)
        if len(data) == 2:
            out, target = data
        elif len(data) == 3:
            _, out, target = data
        else:
            raise ValueError(f'Unsupported data tuple length for boundary-shape loss: {len(data)}')

        boundary_loss = self.boundary(out, target)
        shape_loss = self.shape(out, target)
        return base_loss + self.boundary_weight * boundary_loss + self.shape_weight * shape_loss


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[-180, 180]):
        self.degree = degree
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            return TF.rotate(image, angle), TF.rotate(mask, angle)
        return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk
