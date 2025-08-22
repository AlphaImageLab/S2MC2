from lib.utils.utils import *
import random
import numpy as np


def select_sample_list(data_list, cat_set):
    sample_list, others = [], []
    for data in data_list:
        flag = 0
        for label in data['labels']:
            if label['label'] not in cat_set:
                flag = 1
                break
        if flag == 1:
            others.append(data)
        else:
            sample_list.append(data)
    return sample_list, others


def select(check_list, single_list, categories, num):
    all_categories_set = set([cat['name'] for cat in categories])
    while True:
        sample_categories = random.sample(categories, num)
        sample_categories_set = set([cat['name'] for cat in sample_categories])
        print('check image length:{}\t single image length:{}'.format(len(check_list), len(single_list)))
        sample_check_list, sample_check_test_list = select_sample_list(check_list, sample_categories_set)
        if len(sample_check_list) >= 5300:
            break
    sample_single_list, sample_single_test_list = select_sample_list(single_list, sample_categories_set)
    print('sample check image length:{}\t sample single image length:{}'.format(len(sample_check_list),
                                                                                len(sample_single_list)))
    print('sample test check image length:{}\t sample test single image length:{}'.format(len(sample_check_test_list),
                                                                                          len(sample_single_test_list)))

    sample_check = list2json(sample_check_list, categories)
    sample_check_test = list2json(sample_check_test_list, categories)

    sample_single = list2json(sample_single_list, categories)
    sample_single_test = list2json(sample_single_test_list, categories)

    return sample_check, sample_check_test, sample_single, sample_single_test


if __name__ == '__main__':
    num = 183
    # non_cls = [4, 17, 25, 32, 46, 59, 78, 80, 97, 114, 127, 138, 145, 160, 167, 181, 199]

    # num = 175
    single_json = '/media/hao/Data/RPC/instances_train2019.json'
    check_json = '/media/hao/Data/RPC/instances_val2019.json'
    sample_check_out = '/media/hao/Data/RPC/sample/sub/{}_check_train.json'.format(num)
    sample_check_val_out = '/media/hao/Data/RPC/sample/sub/{}_check_train_r.json'.format(num)
    R_sample_check_out = '/media/hao/Data/RPC/sample/sub/{}_R_check_train.json'.format(num)
    sample_single_out = '/media/hao/Data/RPC/sample/sub/{}_single.json'.format(num)
    sample_single_test_out = '/media/hao/Data/RPC/sample/sub/{}_single_r.json'.format(num)

    single_data = read_json_file(single_json)
    check_data = read_json_file(check_json)
    categories = check_data['categories']
    check_list = json2list(check_data)
    single_list = json2list(single_data)
    # sample_categories = []
    while True:
        # for i in range(len(categories)):
        #     if i+1 not in non_cls:
        #         sample_categories.append(categories[i])
        sample_categories = random.sample(categories, num)
        # sample_categories = categories[:150]
        sample_categories_set = set([cat['name'] for cat in sample_categories])
        all_categories_set = set([cat['name'] for cat in categories])

        print('check image length:{}\t single image length:{}'.format(len(check_list), len(single_list)))

        sample_check_list, sample_check_val_list = select_sample_list(check_list, sample_categories_set)
        if len(sample_check_list) >= 250:
            break
    sample_single_list, sample_single_test_list = select_sample_list(single_list, sample_categories_set)

    R_sample_check_list, _ = select_sample_list(check_list, all_categories_set - sample_categories_set)

    print('sample check image length:{}\t sample single image length:{}'.format(len(sample_check_list),
                                                                                len(sample_single_list)))
    print('sample test check image length:{}\t sample test single image length:{}'.format(len(sample_check_val_list),
                                                                                          len(sample_single_test_list)))
    print('R sample check image length:{}'.format(len(R_sample_check_list)))

    sample_check = list2json(sample_check_list, categories)
    sample_check_val = list2json(sample_check_val_list, categories)
    R_sample_check = list2json(R_sample_check_list, categories)
    sample_single = list2json(sample_single_list, categories)
    sample_single_test = list2json(sample_single_test_list, categories)

    write_json_file(sample_check, sample_check_out)
    write_json_file(sample_check_val, sample_check_val_out)
    write_json_file(R_sample_check, R_sample_check_out)
    write_json_file(sample_single, sample_single_out)
    write_json_file(sample_single_test, sample_single_test_out)

    single_json = '/media/hao/Data/RPC/sample/sub/{}_single.json'.format(num)
    check_json = '/media/hao/Data/RPC/instances_test2019.json'
    out_json = '/media/hao/Data/RPC/sample/sub/{}_check_test_.json'.format(num)
    sample_check_test_out = '/media/hao/Data/RPC/sample/sub/{}_check_test_r.json'.format(num)
    single_data = read_json_file(single_json)
    check_data = read_json_file(check_json)
    single_list = json2list(single_data)
    check_list = json2list(check_data)
    categories = set()
    for item in single_list:
        categories.add(item['labels'][0]['label'])
    sample_check_list, sample_check_test_list = select_sample_list(check_list, categories)
    print('sample check image length:{}\t sample check test image length:{}'.format(len(sample_check_list),
                                                                                    len(sample_check_test_list)))
    sample_check = list2json(sample_check_list, check_data['categories'])
    sample_check_test = list2json(sample_check_test_list, check_data['categories'])
    write_json_file(sample_check, out_json)
    write_json_file(sample_check_test, sample_check_test_out)


    # num = 150
    # single_json = '/media/hao/Data/RPC/instances_train2019.json'
    # val_json = '/media/hao/Data/RPC/instances_val2019.json'
    # test_json = '/media/hao/Data/RPC/instances_test2019.json'
    # sample_check_out = '/media/hao/Data/RPC/sample/{}_check.json'.format(num)
    # sample_check_test_out = '/media/hao/Data/RPC/sample/{}_check_test.json'.format(num)
    # sample_single_out = '/media/hao/Data/RPC/sample/{}_single.json'.format(num)
    # sample_single_test_out = '/media/hao/Data/RPC/sample/{}_single_test.json'.format(num)
    # single_data = read_json_file(single_json)
    # val_data = read_json_file(val_json)
    # test_data = read_json_file(test_json)
    # categories = val_data['categories']
    # val_list = json2list(val_data)
    # test_list = json2list(test_data)
    # single_list = json2list(single_data)
    # check_list = []
    # check_list.extend(val_list)
    # check_list.extend(test_list)
    # all_categories_set = set([cat['name'] for cat in categories])
    # sample_check, sample_check_test, sample_single, sample_single_test = select(check_list, single_list, categories, num)
    # write_json_file(sample_check, sample_check_out)
    # write_json_file(sample_check_test, sample_check_test_out)
    # write_json_file(sample_single, sample_single_out)
    # write_json_file(sample_single_test, sample_single_test_out)
