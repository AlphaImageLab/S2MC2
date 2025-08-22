import _init_paths
from lib.modules import ChannelAttention, SpatialAttention
from lib.net.net_factory import Net
from lib.config import cfg, update_config
from lib.dataset import *
from main.parser import parser
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.evaluate.tools import *
from lib.utils.visualization import *
from lib.utils.visualize_attention_map_V2 import visulize_attention_ratio
import torch
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def foo(pred_dict, gt_dict, pred_count, gt_count, check_img_names):
    for ind, check_img_name in enumerate(check_img_names):
        if check_img_name not in pred_dict:
            pred_dict[check_img_name] = []
        if check_img_name not in gt_dict:
            gt_dict[check_img_name] = []
        pred_dict[check_img_name].append(int(pred_count[ind].item()))
        gt_dict[check_img_name].append(gt_count[ind].item())


def visual_model(dataLoader, model, cfg, device):
    pred_dict = {}
    with torch.no_grad():
        for ind, (image, image_labels, meta) in enumerate(dataLoader):
            print(ind)
            if ind > 30:
                break
            check_image = image.to(device)
            for i in range(len(image_labels)):
                image_labels[i] = image_labels[i].to(device, non_blocking=True)
            single_image = meta['single_img'].to(device, non_blocking=True)
            gt_count = meta['gt_count']
            label_names = meta['label_name']
            check_image_names = meta['check_img_name']
            outputs = model(check_image, single_image)
            pred_count = get_counts(outputs[-1])
            # foo(pred_dict, gt_dict, pred_count, gt_count, check_image_names)
            for i in range(len(label_names)):
                if check_image_names[i] not in pred_dict:
                    pred_dict[check_image_names[i]] = []
                pred_dict[check_image_names[i]].append(
                    {'gt_count': gt_count[i].item(), 'pred_count': pred_count[i].item(), 'label_name': label_names[i],
                     'pred_map': outputs[-1][i][0].cpu()})

        for key, value in pred_dict.items():
            check_img = plt.imread(os.path.join(cfg.DATASET.TEST_ROOT, key)) / 255.0
            # heatmap = np.zeros((*check_img.shape[:2], 3))
            label_dict = {}
            pred_map = 0
            for item in value:
                # norm_pred_map = norm_features(item['pred_map'])
                pred_map += item['pred_map']
                # heatmap += Visualization.get_heatmap(norm_pred_map, check_img.shape[:2])
                label_dict[item['label_name']] = [item['gt_count'], item['pred_count']]
            # merge_img = 0.7 * check_img + 0.3 * heatmap
            pred_map_norm = norm_features(pred_map)
            print(pred_map_norm)
            heatmap_s = Visualization.get_heatmap(pred_map_norm, check_img.shape[:2])
            print(heatmap_s)
            # plt.imshow(heatmap_s)
            # plt.show()
            # plt.close()
            merge_img_s = 0.7 * check_img + 0.3 * heatmap_s
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, cfg.DATASET.LEVEL, key[:-4])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # plt.imsave(os.path.join(save_path, 'merge.jpg'), merge_img, format="png")
            plt.imsave(os.path.join(save_path, 'merge_s.jpg'), merge_img_s, format="jpg")
            plt.imsave(os.path.join(save_path, '{}.jpg').format(key[:-4]), check_img, format="jpg")
            with open(os.path.join(save_path, 'gt.txt'), 'w') as f:
                for label_name, value in label_dict.items():
                    print(label_name, value[0], value[1], file=f)
            with open(os.path.join(save_path, '{}_gt.txt').format(key[:-4]), 'w') as f:
                for label_name, value in label_dict.items():
                    print(label_name, value[0], file=f)


def da_visual_model(dataLoader, model, cfg, device):
    from sklearn.manifold import TSNE
    num = 5000
    single_features = []
    singleLoader = DataLoader(RPCSingleFeature(cfg=cfg),
                              batch_size=cfg.TEST.BATCH_SIZE,
                              shuffle=False,
                              num_workers=cfg.TEST.NUM_WORKERS,
                              pin_memory=False)
    for single_image in tqdm(singleLoader):
        single_features.append(single_image)

    check_features = []
    with torch.no_grad():
        for ind, (image, image_labels, meta) in enumerate(tqdm(dataLoader)):
            check_image = image.to(device)
            single_image = meta['single_img'].to(device, non_blocking=True)
            outputs = model(check_image, single_image, extract_feature=True)
            outputs = outputs[0]
            image_labels = image_labels[0].to(device, non_blocking=True)
            b, c, h, w = outputs.shape
            mask = image_labels.gt(0)
            mask = mask.view(b, 1, -1)
            mask = mask.transpose(1, 2).contiguous()
            mask = mask.view(-1, 1).squeeze(-1)
            outputs = outputs.view(b, c, -1)
            outputs = outputs.transpose(1, 2).contiguous()
            outputs = outputs.view(-1, c)
            outputs = outputs[mask].mean(0, keepdim=True)
            # outputs = F.adaptive_avg_pool2d(outputs, (1, 1))

            # outputs = outputs.view(b, -1)
            # outputs = outputs.transpose(1, 2).contiguous()
            # outputs = outputs.view(-1, c)
            check_features.append(outputs)

        single_features = torch.cat(single_features, 0).data.cpu().numpy()
        check_features = torch.cat(check_features, 0).data.cpu().numpy()

        check_features_index = np.random.randint(0, check_features.shape[0], num)
        single_features_index = np.random.randint(0, single_features.shape[0], num)

        sample_single = single_features[single_features_index]
        sample_check = check_features[check_features_index]
        # check = TSNE(n_components=2, learning_rate=100, verbose=1).fit_transform(sample_check)
        # single = TSNE(n_components=2, learning_rate=100, verbose=1).fit_transform(sample_single)
        all_array = np.vstack([sample_check, sample_single])
        result = TSNE(n_components=2, learning_rate=100, verbose=1).fit_transform(all_array)
        check = result[:num]
        single = result[num:]
        plt.figure()
        plt.scatter(single[:, 0], single[:, 1], c='b')
        plt.scatter(check[:, 0], check[:, 1], c='r')
        # plt.scatter(single[:, 0], single[:, 1], c='b')
        plt.savefig('/home/hao/Desktop/Counting_CodeBase/output/test_4.jpg')


def atten_visual_model(dataLoader, model, cfg, device):
    # check_feature = []
    with torch.no_grad():
        for ind, (image, image_labels, meta) in enumerate(tqdm(dataLoader)):
            print(ind)
            if ind > 300:
                break
            check_image = image.to(device)
            single_image = meta['single_img'].to(device, non_blocking=True)
            check_image_names = meta['check_img_name']
            check_feature = model(check_image, single_image, extract_feature=True)
            # outputs = outputs[0]
            # b, c, h, w = outputs.shape
            # check_features.append(outputs)
            chan_atten = ChannelAttention(check_feature.shape[1]).cuda()
            spat_atten = SpatialAttention().cuda()
            atten_map1 = spat_atten(check_feature)
            atten_map2 = atten_map1 * check_feature
            atten_map3 = chan_atten(atten_map2)
            atten_map4 = atten_map3 * atten_map2
            atten_map2_ = atten_map2.mean(axis=1, keepdim=False)
            chan_idx = torch.max(atten_map3, 1)[1].data.squeeze()
            atten_map4_ = torch.index_select(atten_map4, 1, chan_idx).squeeze()
            atten_map4_1 = atten_map4.mean(axis=1, keepdim=False)

            layout_atten = atten_map2_[0].cpu()
            detial_atten = atten_map4_[0].cpu()
            detial_atten_1 = atten_map4_1[0].cpu()

            check_img = plt.imread(os.path.join(cfg.DATASET.TEST_ROOT, check_image_names[0])) / 255.0
            layout_atten_norm = norm_features(layout_atten)
            detial_atten_norm = norm_features(detial_atten)
            detial_atten_norm_1 = norm_features(detial_atten_1)
            heatmap_l = Visualization.get_heatmap(layout_atten_norm, check_img.shape[:2])
            heatmap_d = Visualization.get_heatmap(detial_atten_norm, check_img.shape[:2])
            heatmap_d_1 = Visualization.get_heatmap(detial_atten_norm_1, check_img.shape[:2])
            merge_img_l = 0.5 * check_img + 0.5 * heatmap_l
            merge_img_d = 0.5 * check_img + 0.5 * heatmap_d
            merge_img_d_1 = 0.5 * check_img + 0.5 * heatmap_d_1
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, cfg.DATASET.LEVEL, check_image_names[0][:-4])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(os.path.join(save_path, 'merge_layout.jpg'), merge_img_l, format="jpg")
            plt.imsave(os.path.join(save_path, 'merge_detial.jpg'), merge_img_d, format="jpg")
            plt.imsave(os.path.join(save_path, 'merge_detial_1.jpg'), merge_img_d_1, format="jpg")
            plt.imsave(os.path.join(save_path, '{}.jpg').format(check_image_names[0][:-4]), check_img, format="jpg")


def atten_visual_model_1(dataLoader, model, cfg, device):
    # check_feature = []
    with torch.no_grad():
        for ind, (image, image_labels, meta) in enumerate(tqdm(dataLoader)):
            print(ind)
            if ind > 300:
                break
            check_image = image.to(device)
            single_image = meta['single_img'].to(device, non_blocking=True)
            check_image_names = meta['check_img_name']
            check_feature = model(check_image, single_image, extract_feature=True)
            # check_feature = model.module.backbone.extract_features(check_image)

            chan_atten = ChannelAttention(check_feature.shape[1]).cuda()
            spat_atten = SpatialAttention().cuda()

            atten_map1 = spat_atten(check_feature)
            atten_map2 = atten_map1 * check_feature
            atten_map3 = chan_atten(atten_map2)
            atten_map4 = atten_map3 * atten_map2

            atten_map2_ = atten_map2.mean(axis=1, keepdim=False)
            # print(atten_map2_)
            chan_idx = torch.max(atten_map3, 1)[1].data.squeeze()
            atten_map4_ = torch.index_select(atten_map4, 1, chan_idx).squeeze()
            atten_map4_1 = atten_map4.mean(axis=1, keepdim=False)

            img_path = os.path.join(cfg.DATASET.TEST_ROOT, check_image_names[0])
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, cfg.DATASET.LEVEL, check_image_names[0][:-4])
            visulize_attention_ratio(img_path=img_path, save_path=save_path, attention_mask=atten_map1.squeeze().cpu().numpy(),
                                     save_image=True,
                                     save_original_image=True)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    update_config(cfg, args)
    device = torch.device('cuda' if not cfg.CPU_MODE and torch.cuda.is_available() else 'cpu')
    test_set = eval(cfg.DATASET.DATASET)('test', cfg)
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
    # visual_model(testLoader, model, cfg, device)
    # da_visual_model(testLoader, model, cfg, device)
    # atten_visual_model(testLoader, model, cfg, device)
    atten_visual_model_1(testLoader, model, cfg, device)
