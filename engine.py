import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs

from medpy.metric.binary import hd95


def calculate_hd95_list(preds_list, gts_list, threshold=0.5):
    hd95_scores = []

    for batch_pred, batch_gt in zip(preds_list, gts_list):
        for i in range(batch_pred.shape[0]):
            p = (batch_pred[i] >= threshold).astype(int)
            g = (batch_gt[i] >= threshold).astype(int)

            if np.sum(p) > 0 and np.sum(g) > 0:
                try:
                    res = hd95(p, g)
                    hd95_scores.append(res)
                except RuntimeError:
                    pass

    if len(hd95_scores) > 0:
        return np.mean(hd95_scores)
    else:
        return 0.0


def _format_train_log(epoch, iter_idx, loss, lr):
    return f'train | epoch {epoch:03d} | iter {iter_idx:05d} | loss {loss:8.4f} | lr {lr:.6e}'


def _format_eval_log(phase, epoch, loss, metrics=None):
    log_info = f'{phase:<5} | epoch {epoch:03d} | loss {loss:8.4f}'
    if metrics is None:
        return log_info

    metric_info = ' | '.join([
        f'miou {metrics["miou"]:.4f}',
        f'f1/dsc {metrics["f1_or_dsc"]:.4f}',
        f'acc {metrics["accuracy"]:.4f}',
        f'spec {metrics["specificity"]:.4f}',
        f'sens {metrics["sensitivity"]:.4f}',
        f'hd95 {metrics["hd95"]:.4f}',
    ])
    return f'{log_info} | {metric_info}'


def _format_table(title, rows):
    name_width = max(len('Metric'), *(len(name) for name, _ in rows))
    value_width = max(len('Value'), *(len(value) for _, value in rows))
    border = f'+-{"-" * name_width}-+-{"-" * value_width}-+'

    lines = [
        title,
        border,
        f'| {"Metric":<{name_width}} | {"Value":>{value_width}} |',
        border,
    ]
    for name, value in rows:
        lines.append(f'| {name:<{name_width}} | {value:>{value_width}} |')
    lines.append(border)

    return '\n'.join(lines)


def _format_confusion_matrix_table(confusion):
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    rows = [
        ('GT 0', str(TN), str(FP)),
        ('GT 1', str(FN), str(TP)),
    ]
    label_width = max(len(''), *(len(label) for label, _, _ in rows))
    pred0_width = max(len('Pred 0'), *(len(pred0) for _, pred0, _ in rows))
    pred1_width = max(len('Pred 1'), *(len(pred1) for _, _, pred1 in rows))
    border = f'+-{"-" * label_width}-+-{"-" * pred0_width}-+-{"-" * pred1_width}-+'

    lines = [
        'Confusion Matrix',
        border,
        f'| {"":<{label_width}} | {"Pred 0":>{pred0_width}} | {"Pred 1":>{pred1_width}} |',
        border,
    ]
    for label, pred0, pred1 in rows:
        lines.append(f'| {label:<{label_width}} | {pred0:>{pred0_width}} | {pred1:>{pred1_width}} |')
    lines.append(border)

    return '\n'.join(lines)


def _format_test_summary(loss, metrics, confusion, test_data_name=None):
    rows = [
        ('Dataset', test_data_name) if test_data_name is not None else None,
        ('Loss', f'{loss:.4f}'),
        ('mIoU', f'{metrics["miou"]:.4f}'),
        ('F1/DSC', f'{metrics["f1_or_dsc"]:.4f}'),
        ('Accuracy', f'{metrics["accuracy"]:.4f}'),
        ('Specificity', f'{metrics["specificity"]:.4f}'),
        ('Sensitivity', f'{metrics["sensitivity"]:.4f}'),
        ('HD95', f'{metrics["hd95"]:.4f}'),
    ]
    rows = [row for row in rows if row is not None]
    return '\n'.join([
        _format_table('Test Summary', rows),
        _format_confusion_matrix_table(confusion),
    ])


def _weighted_bce_dice_loss(pred, target, weight, eps=1e-6):
    pred = torch.clamp(pred, eps, 1.0 - eps)
    target = target.float()
    weight = weight.float()

    bce = F.binary_cross_entropy(pred, target, reduction='none')
    bce = (bce * weight).sum() / (weight.sum() + eps)

    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    target_flat = target.view(size, -1)
    weight_flat = weight.view(size, -1)
    intersection = (pred_flat * target_flat * weight_flat).sum(1)
    pred_sum = (pred_flat * weight_flat).sum(1)
    target_sum = (target_flat * weight_flat).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)).mean()
    return bce + dice


def _artifact_consistency_loss(model, criterion, data, config):
    clean_images, artifact_images, targets, artifact_masks = data
    clean_images = clean_images.cuda(non_blocking=True).float()
    artifact_images = artifact_images.cuda(non_blocking=True).float()
    targets = targets.cuda(non_blocking=True).float()
    artifact_masks = artifact_masks.cuda(non_blocking=True).float()

    gt_clean, out_clean = model(clean_images)
    _, out_artifact = model(artifact_images)

    if gt_clean is None:
        loss_clean = criterion(out_clean, targets)
    else:
        loss_clean = criterion((gt_clean, out_clean, targets))
    loss_cons = F.mse_loss(out_artifact, out_clean.detach())

    beta = float(getattr(config, 'artifact_consistency_downweight_beta', 0.7))
    weight = torch.clamp(1.0 - beta * artifact_masks, min=0.05, max=1.0)
    loss_artifact = _weighted_bce_dice_loss(out_artifact, targets, weight)

    lambda_cons = float(getattr(config, 'artifact_consistency_lambda', 0.2))
    lambda_artifact = float(getattr(config, 'artifact_consistency_artifact_lambda', 0.1))
    return loss_clean + lambda_cons * loss_cons + lambda_artifact * loss_artifact


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        artifact_consistency = getattr(config, 'train_data_mode', 'clean') == 'artifact_consistency'
        if config.amp:
            with autocast():
                if artifact_consistency:
                    loss = _artifact_consistency_loss(model, criterion, data, config)
                else:
                    images, targets = data
                    images = images.cuda(non_blocking=True).float()
                    targets = targets.cuda(non_blocking=True).float()
                    gt_pre, out = model(images)
                    if gt_pre is None:
                        loss = criterion(out, targets)
                    else:
                        loss = criterion((gt_pre, out, targets))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if artifact_consistency:
                loss = _artifact_consistency_loss(model, criterion, data, config)
            else:
                images, targets = data
                images = images.cuda(non_blocking=True).float()
                targets = targets.cuda(non_blocking=True).float()
                gt_pre, out = model(images)
                if gt_pre is None:
                    loss = criterion(out, targets)
                else:
                    loss = criterion((gt_pre, out, targets))
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            logger.info(_format_train_log(epoch, iter, np.mean(loss_list), now_lr))
    scheduler.step()


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            aux, out = model(img)
            if aux is None:
                loss = criterion(out, msk)
            else:
                loss = criterion((out, msk))

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    metrics = None
    if epoch % config.val_interval == 0:
        avg_hd95 = calculate_hd95_list(preds, gts, config.threshold)
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        metrics = {
            'miou': miou,
            'f1_or_dsc': f1_or_dsc,
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'hd95': avg_hd95,
        }
        logger.info(_format_eval_log('val', epoch, np.mean(loss_list), metrics))

    else:
        logger.info(_format_eval_log('val', epoch, np.mean(loss_list)))

    return np.mean(loss_list), metrics


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            aux, out = model(img)
            if aux is None:
                loss = criterion(out, msk)
            else:
                loss = criterion((out, msk))

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        avg_hd95 = calculate_hd95_list(preds, gts, config.threshold)
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        metrics = {
            'miou': miou,
            'f1_or_dsc': f1_or_dsc,
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'hd95': avg_hd95,
        }
        logger.info(_format_test_summary(np.mean(loss_list), metrics, confusion, test_data_name))

    return np.mean(loss_list)
