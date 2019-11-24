import cv2
from src.data.transforms import Rescale, RandomCrop
from src.models.joint_face_det_model import JointDetectionModule
from src.data.Dataset import LS3DDataset
from src.models.loss import JointLoss
from src.models.metrics import map_eval, get_bbox
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
import albumentations as albu
import albumentations.pytorch.transforms
import torch
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter
import datetime

BATCH_SIZE = 10
LEARNING_RATE = 10 ** (-4)
LR_FACTOR = 10
EPOCHS = 7
logger.add('/root/3dface_aelita/log/train.log', format="\n{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
now = datetime.datetime.now()
writer = SummaryWriter(f'/root/3dface_aelita/log/tensorboard/{now.day}-{now.month}-{now.hour}-{now.minute}')
TAG ='fix'

def run(mode, epoch, dataloader, model, criterion, optimizer, scheduler, device=None):
    running_loss = 0.0
    intermediate_loss = 0.0
    nme_sum = 0.0
    map_sum = 0.0
    # try:
    for batch_i, batch in tqdm(enumerate(dataloader), desc='batch', total=dataloader.__len__()):
        optimizer.zero_grad()
        data = batch['image'].to(device)
        landmarks_target = batch['landmarks_2d'].to(device)
        bbox_target = batch['bbox'].to(device)
        if mode == 'eval':
            with torch.no_grad():
                predicted = model(data)

                nme, loc_loss, conf_loss, suppressed = criterion(predicted, (bbox_target, landmarks_target))
                map = map_eval(get_bbox(landmarks_target), suppressed[0].tolist())
        else:
            predicted = model(data)

            nme, loc_loss, conf_loss, suppressed = criterion(predicted, (bbox_target, landmarks_target))
            map = map_eval(get_bbox(landmarks_target), suppressed[0].tolist())
        loss = nme + loc_loss + conf_loss
        if mode == 'train':
            loss.backward()
            optimizer.step()
        i = dataloader.__len__() * epoch + batch_i
        writer.add_scalar(f'{mode}/nme', nme.item(), i)
        writer.add_scalar(f'{mode}/loc_loss', loc_loss.item(), i)
        writer.add_scalar(f'{mode}/conf_loss', conf_loss.item(), i)
        writer.add_scalar(f'{mode}/loss', loss.item(), i)
        writer.add_scalar(f'{mode}/map', map, i)

        intermediate_loss += loss.item()
        running_loss += loss.item()
        map_sum += map
        nme_sum += nme
        if mode == 'train':
            if batch_i == 0:
                print('\n', loss.item(), '\n')
            if (batch_i + 1) % 900 == 0:
                logger.info(f'Batch {batch_i}: training loss:{intermediate_loss / 900}')
                intermediate_loss = 0.0

    # except Exception as exc:
    #     logger.info(f'Exception {exc} caught')
    #     save_model(model, optimizer, scheduler, running_loss / (batch_i + 1), epoch, f'model_exc_{TAG}')
    #     if batch_i < dataloader.__len__() // 2:
    #         return None
    #     return model, running_loss / (batch_i + 1), map_sum / (batch_i + 1), nme_sum / (batch_i + 1)

    return model, running_loss / dataloader.__len__(), map_sum / dataloader.__len__(), nme_sum / dataloader.__len__()


def save_model(model, optimizer, scheduler, loss, epoch, model_name):
    path = f'/root/3dface_aelita/models/joint_det_models/{model_name}.pth.tar'
    logger.info('Saving model...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss

    }, path)
    logger.info(f'Model saved in {path}')


def load_model(model_name):
    path = f'/root/3dface_aelita/models/joint_det_models/{model_name}.pth.tar'

    model = JointDetectionModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    skip_train = 'train' in model_name
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        if 'val' in model_name:
            epoch += 1
        loss = checkpoint['loss']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f'Model loaded from "models/joint_det_models/{model_name}.pth.tar". Curent loss:{loss}, epoch:{epoch}')
        return skip_train, model, loss, epoch, optimizer, scheduler
    except FileNotFoundError:
        logger.info(f'No saved model')
        return False, model, float("inf"), 0, optimizer, scheduler


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


def main():
    train_dataset = LS3DDataset(csv_file='/root/data/train.csv',
                                root_dir='/root/data/LS3D-W',
                                transformations=[RandomCrop(0.1), Rescale(288)],
                                albu_transformations=[
                                    albu.PadIfNeeded(288, 288, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                                    # albu.Rotate(border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                                    albu.pytorch.transforms.ToTensor()])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    val_dataset = LS3DDataset(csv_file='/root/data/test.csv',
                              root_dir='/root/data/LS3D-W',
                              transformations=[RandomCrop(0.1), Rescale(288)],
                              albu_transformations=[
                                  albu.PadIfNeeded(288, 288, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                                  # albu.Rotate(border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                                  albu.pytorch.transforms.ToTensor()])

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device is set: {device}')
    criterion = JointLoss(device)
    skip_train, model, min_loss, start_epoch, optimizer, scheduler = load_model(f'model_train_{TAG}')
    model.to(device)
    scheduler_to(scheduler, device)
    optimizer_to(optimizer, device)
    for epoch in tqdm(range(start_epoch, EPOCHS), desc='epochs', total=EPOCHS):
        logger.info(f'Epoch {epoch} started')
        if skip_train:
            logger.info(f'Already trained for epoch {epoch} model loaded, training skipped')
            skip_train = False
        else:
            logger.info(f'Training started')
            model.train()
            model, loss, map, nme = run('train', epoch, train_loader, model, criterion, optimizer, scheduler,
                                        device=device)
            save_model(model, optimizer, scheduler, loss, epoch, f'model_train_{TAG}')

            logger.info(f'Training finished with average loss: {loss}, MAP: {map}, NME:{nme}')

        logger.info(f'Evaluation started')
        model.eval()
        model, val_loss, map, nme = run('eval', epoch, val_loader, model, criterion, optimizer, scheduler,
                                        device=device)
        scheduler.step(val_loss)
        logger.info(f'Evaluation finished. MAP {map}, NME:{nme}, val_loss:{val_loss}')
        save_model(model, optimizer, scheduler, val_loss, epoch, f'model_val_{TAG}')
        logger.info(f'Epoch {epoch} finished')
    writer.close()


if __name__ == '__main__':
    main()
