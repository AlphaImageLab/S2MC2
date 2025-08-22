# from main._init_paths import *
import _init_paths
import sys
from lib.loss.loss_factory import Loss
from lib.dataset import *
from lib.config import cfg, update_config
from lib.utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
)
from lib.utils.sampler import get_sampler, get_category_list
from lib.core.function import train_model, valid_model
from lib.core.combiner import *
from lib.evaluate.evaluate_factory import EvaluateMetric
from lib.data_transform.transform_factory import Transform
from lib.net.net_factory import Net
import torch
import os, json, shutil
from torch.utils.data import DataLoader
import warnings
import click
from tensorboardX import SummaryWriter
from main.parser import parser
import pysnooper

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    update_config(cfg, args)
    # gpus = [int(i) for i in args.gpu.split(',')]
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")

    train_set = eval(cfg.DATASET.DATASET)("train", cfg, Transform.get(cfg, cfg.TRANSFORMS.TRAIN_TRANSFORMS))
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg, Transform.get(cfg, cfg.TRANSFORMS.TRAIN_TRANSFORMS))
    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()

    num_class_list, _ = ([None, None])

    if not cfg.LOSS.WEIGHT_FILE == "":
        logger.info("Loading loss weights from {}".format(cfg.LOSS.WEIGHT_FILE))
        with open(cfg.LOSS.WEIGHT_FILE, "r") as f:
            weight = json.load(f)
    else:
        weight = None
    para_dict = {
        "num_classes": num_classes,
        "weight": weight,
        "num_class_list": num_class_list,
        "cfg": cfg,
    }

    criterions = Loss.get(cfg, para_dict)
    epoch_number = cfg.TRAIN.MAX_EPOCH
    device = torch.device('cuda' if not cfg.CPU_MODE and torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    # ----- BEGIN MODEL BUILDER -----
    sampler = get_sampler(cfg, annotations, num_classes)
    model = Net.get(cfg, num_classes, device, logger)
    combiner = eval(cfg.COMBINER)(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    metrics = EvaluateMetric.get(cfg.METRICS)
    # ----- END MODEL BUILDER -----

    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    validLoader = DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # 实验闭环机制
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    # code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
        ):
            exit(0)
        # shutil.rmtree(code_dir)
        # if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*"
    )
#    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)
    if tensorboard_dir is not None:
        # dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        # writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input,))
    else:
        writer = None

    best_result, best_epoch, start_epoch = -9999, 0, 1
    # ----- BEGIN RESUME ---------
    if cfg.RESUME_MODEL != "":
        logger.info("Loading checkpoint from {}...".format(cfg.RESUME_MODEL))
        checkpoint = torch.load(
            cfg.RESUME_MODEL, map_location="cpu" if cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = checkpoint['state_dict']
        if cfg.CPU_MODE:
            model.load_model(pretrain_dict)
        else:
            model.module.load_model(pretrain_dict)

        if cfg.RESUME_MODE == 'all':
            start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
    # ----- END RESUME ---------

    logger.info(
        "-------------------Train start :{}  {}  {}  {}-------------------".format(
            cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.CLASSIFIER.TYPE, cfg.COMBINER
        )
    )
    for epoch in range(start_epoch, epoch_number + 1):
        scheduler.step()
        logger.info('----------Learning Rate:{}-----------'.format(scheduler.get_lr()))
        train_result, train_loss = train_model(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            combiner,
            criterions,
            cfg,
            logger,
            metrics,
            writer=writer,
        )
        model_save_path = os.path.join(
            model_dir,
            "{}_{}_{}_epoch_{}.pth".format(
                cfg.DATASET.DATASET, cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, epoch
            ),
        )

        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_result': best_result,
            'best_epoch': best_epoch,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_save_path)

        loss_dict, result_dict = {"train_loss": train_loss}, {"train_result": train_result}

        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_result, valid_loss = valid_model(
                validLoader, epoch, model, cfg, criterions, logger, device, combiner, metrics, writer=writer
            )

            loss_dict["valid_loss"], result_dict["valid_result"] = valid_loss, valid_result
            if valid_result > best_result:
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, '{:.4f}_best_model.pth'.format(valid_result)))
                prev_path = os.path.join(model_dir, '{:.4f}_best_model.pth'.format(best_result))
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                best_result, best_epoch = valid_result, epoch
            logger.info(
                "--------------Best_Epoch:{:>3d}    Best_result:{:>5.2f}%--------------".format(
                    best_epoch, best_result * 100
                )
            )
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/result", result_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
