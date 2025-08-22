import _init_paths
from lib.net.net_factory import Net
from lib.config import cfg, update_config
from lib.dataset import *
from main.parser import parser
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.utils.utils import write_json_file


def test_model(dataLoader, model, cfg, device):
    name = cfg.NAME
    save_dir = os.path.join(cfg.OUTPUT_DIR, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    anno_file = os.path.join(save_dir, 'single_feature.json')
    feature_dir = os.path.join(save_dir, 'features')
    print('Annotation file will be saved to {}'.format(anno_file))
    print('Feature file will be saved to {}'.format(feature_dir))
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    anno_list = []
    # torch.save(model, '/home/hao/Desktop/Counting_CodeBase/feature_extractor.pth')

    num = torch.zeros([200])
    prototype = torch.zeros([200, 2048])
    prototype_sum = torch.zeros([200, 2048])
    prototype_dir = os.path.join(save_dir, 'prototype.pth')

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(tqdm(dataLoader)):
            image = image.to(device)
            output = model(image, feature_flag=True)
            file_names = meta['file_name']
            label_names = meta['label_name']
            output_numpy = output.data.cpu().numpy()
            for i, file_name in enumerate(file_names):
                feature_name = file_name[:-4] + '.npy'
                feature_file = os.path.join(feature_dir, feature_name)
                label_name = label_names[i]
                anno_list.append({'file_name': feature_name, 'labels': [{'label': label_name}]})
                np.save(feature_file, output_numpy[i])

            for j in range(len(output)):
                num[image_labels[j].item()] += 1
                prototype_sum[image_labels[j].item()] += output[j].cpu()
        for j in range(len(prototype)):
            prototype[j] = prototype_sum[j]/num[j]
        prototype = prototype[~torch.isnan(prototype).any(axis=1)]
        torch.save(prototype, prototype_dir)

        write_json_file(anno_list, anno_file)


if __name__ == '__main__':
    args = parser.parse_args()
    update_config(cfg, args)
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    test_set = eval(cfg.DATASET.DATASET)('test', cfg)
    num_classes = test_set.get_num_classes()

    model = Net.get(cfg, num_classes, device)
    if cfg.CPU_MODE:
        model.load_model(torch.load(cfg.TEST.MODEL_FILE, map_location=device)['state_dict'])
    else:
        model.module.load_model(torch.load(cfg.TEST.MODEL_FILE, map_location=device)['state_dict'])
    # torch.save(model, '/home/hao/Desktop/Counting_CodeBase/feature_extractor.pth')
    model.eval()
    testLoader = DataLoader(test_set,
                            batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.TEST.NUM_WORKERS,
                            pin_memory=False)
    test_model(testLoader, model, cfg, device)
