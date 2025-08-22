import _init_paths
from lib.net.net_factory import Net
from lib.config import cfg, update_config
from lib.dataset import *
from main.parser import parser
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.utils.utils import write_json_file
from lib.evaluate.tools import *
from lib.evaluate.evaluate_factory import EvaluateMetric
import click

# 暂停使用共享内存
import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]
#####


def foo(pred_dict, gt_dict, pred_count, gt_count, check_img_names):
    for ind, check_img_name in enumerate(check_img_names):
        if check_img_name not in pred_dict:
            pred_dict[check_img_name] = []
        if check_img_name not in gt_dict:
            gt_dict[check_img_name] = []
        pred_dict[check_img_name].append(int(pred_count[ind].item()))
        gt_dict[check_img_name].append(gt_count[ind].item())


def test_model(dataLoader, model, cfg, device, metrics):
    pred_dict = {}
    gt_dict = {}

    name = cfg.NAME
    save_dir = os.path.join(cfg.OUTPUT_DIR, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fp_dict = {}
    length = 0
    with torch.no_grad():
        for ind, (image, image_labels, meta) in enumerate(tqdm(dataLoader)):
            check_image = image.to(device)
            # for i in range(len(image_labels)):
            image_labels = image_labels.to(device, non_blocking=True)
            # single_image = meta['single_img'].to(device, non_blocking=True)
            # gt_count = meta['gt_count']
            # label_names = meta['label_name']
            # check_image_names = meta['check_img_name']
            outputs = model(check_image)
            # pred_count = get_counts(outputs[-1])
            metrics.update(outputs, image_labels)
            if ind % cfg.SHOW_STEP == 0:
                pbar_str = "Test:Batch:{:>3d}/{}\t\t".format(
                    ind, len(dataLoader)) + metrics.__str__()
                print(pbar_str)

            # foo(pred_dict, gt_dict, pred_count, gt_count, check_image_names)
            # for i in range(len(label_names)):
            #     # from lib.utils.visualization import visual
            #     # visual(os.path.join(cfg.DATASET.TEST_ROOT, meta['check_img_name'][i]),
            #     #        os.path.join('/data/rpc/retail_product_checkout/train2019/',
            #     #                     meta['single_img_name'][i][:-3] + 'jpg'),
            #     #        outputs[-1][i][0], '/data/ActivityNet/Counting_CodeBase/', '{}'.format(i))
            #
            #     if pred_count[i].item() == gt_count[i].item():
            #         continue
            #     label_name = label_names[i]
            #     check_image_name = check_image_names[i]
            #     if check_image_name not in fp_dict:
            #         fp_dict[check_image_name] = []
            #     fp_dict[check_image_name].append(label_name)
            #     length += 1

        print('-----------' + metrics.__str__() + '--------------')
        # print("fp length: {}".format(length))

        # f = open('/data/ActivityNet/Counting_CodeBase/out1.txt', 'w')
        # for name in sorted(list(pred_dict.keys())):
        #     print(name, end='\t', file=f)
        #     for i in pred_dict[name]:
        #         print(i, file=f, end='\t')
        #     print(file=f)
        #     print(name, end='\t', file=f)
        #     for i in gt_dict[name]:
        #         print(i, file=f, end='\t')
        #     print(file=f)

        if not click.confirm(
                "\033[1;31;40mContinue and save the fp json?\033[0m",
                default=False,
        ):
            exit(0)
        # if cfg.TEST.JSON_FILE != '':
        #     write_json_file(fp_dict, os.path.join(save_dir, cfg.TEST.JSON_FILE))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    update_config(cfg, args)
    device = torch.device('cuda' if not cfg.CPU_MODE and torch.cuda.is_available() else 'cpu')
    test_set = eval(cfg.DATASET.DATASET)('test', cfg)
    metrics = EvaluateMetric.get(cfg.METRICS)
    num_classes = test_set.get_num_classes()

    model = Net.get(cfg, num_classes, device)
    print('load model from {}'.format(cfg.TEST.MODEL_FILE))
    if device.type == 'cpu':
        model.load_model(torch.load(cfg.TEST.MODEL_FILE, map_location=device)['state_dict'])
    else:
        model.module.load_model(torch.load(cfg.TEST.MODEL_FILE, map_location=device)['state_dict'])
    model.eval()
    testLoader = DataLoader(test_set,
                            batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.TEST.NUM_WORKERS,
                            pin_memory=False)
    test_model(testLoader, model, cfg, device, metrics)
