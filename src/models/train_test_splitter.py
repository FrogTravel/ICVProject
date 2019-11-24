import os
import torchfile
import random
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Split dataset on train and test parts')
    parser.add_argument('-tts', type=float, help='coefficient which define the size of train', default=0.7)
    parser.add_argument('-path', '-p', type=str, help='data directory', default='/' + os.path.join('root', 'data'))
    parser.add_argument('-rand', '-r', type=int, help='random state', default=17)
    parser.add_argument('-train', type=str, help='path to train.csv', default=None)
    parser.add_argument('-test', type=str, help='path to test.csv', default=None)
    args = parser.parse_args()
    if not args.test:
        args.test = os.path.join(args.path, 'test.csv')
    if not args.train:
        args.train = os.path.join(args.path, 'train.csv')
    return args


def walk_300vw_3d(tts, path):
    """
    walk throw 300VW_3D dataset
    :param tts: train/test coefficient
    :param path: path to 300VW_3D dataset
    :return: [train, test] landmarks for train and test dataset
    """
    train, test = '', ''
    for (dir_path, dir_names, file_names) in os.walk(path):
        landmarks = []
        for file_name in file_names:
            # search for torch files
            file_path = os.path.join(dir_path, file_name)
            if not file_path.endswith('.t7'):
                continue

            # load the points
            points = torchfile.load(file_path)

            # search for image name
            if os.path.isfile(file_path[:-2] + 'jpg'):
                csv_list = [file_path[:-2] + 'jpg']
            elif os.path.isfile(file_path[:-2] + 'png'):
                csv_list = [file_path[:-2] + 'png']
            else:
                continue

            # add landmarks to our list
            for p in points:
                for n in p:
                    csv_list.append(str(n))
            landmarks.append(f"{','.join(csv_list)}")

        # add landmarks from hole directory to train or path (no two files from one subset in different groups)
        if len(landmarks) > 0:
            if random.random() > tts:
                test += '\n'.join(landmarks)
                test += '\n'
            else:
                train += '\n'.join(landmarks)
                train += '\n'
    return train, test


def main(args):
    header = 'file_name,2d_0_x,2d_0_y,2d_1_x,2d_1_y,2d_2_x,2d_2_y,2d_3_x,2d_3_y,2d_4_x,2d_4_y,2d_5_x,2d_5_y,' \
             '2d_6_x,2d_6_y,2d_7_x,2d_7_y,2d_8_x,2d_8_y,2d_9_x,2d_9_y,2d_10_x,2d_10_y,2d_11_x,2d_11_y,2d_12_x,' \
             '2d_12_y,2d_13_x,2d_13_y,2d_14_x,2d_14_y,2d_15_x,2d_15_y,2d_16_x,2d_16_y,2d_17_x,2d_17_y,2d_18_x,' \
             '2d_18_y,2d_19_x,2d_19_y,2d_20_x,2d_20_y,2d_21_x,2d_21_y,2d_22_x,2d_22_y,2d_23_x,2d_23_y,2d_24_x,' \
             '2d_24_y,2d_25_x,2d_25_y,2d_26_x,2d_26_y,2d_27_x,2d_27_y,2d_28_x,2d_28_y,2d_29_x,2d_29_y,2d_30_x,' \
             '2d_30_y,2d_31_x,2d_31_y,2d_32_x,2d_32_y,2d_33_x,2d_33_y,2d_34_x,2d_34_y,2d_35_x,2d_35_y,2d_36_x,' \
             '2d_36_y,2d_37_x,2d_37_y,2d_38_x,2d_38_y,2d_39_x,2d_39_y,2d_40_x,2d_40_y,2d_41_x,2d_41_y,2d_42_x,' \
             '2d_42_y,2d_43_x,2d_43_y,2d_44_x,2d_44_y,2d_45_x,2d_45_y,2d_46_x,2d_46_y,2d_47_x,2d_47_y,2d_48_x,' \
             '2d_48_y,2d_49_x,2d_49_y,2d_50_x,2d_50_y,2d_51_x,2d_51_y,2d_52_x,2d_52_y,2d_53_x,2d_53_y,2d_54_x,' \
             '2d_54_y,2d_55_x,2d_55_y,2d_56_x,2d_56_y,2d_57_x,2d_57_y,2d_58_x,2d_58_y,2d_59_x,2d_59_y,2d_60_x,' \
             '2d_60_y,2d_61_x,2d_61_y,2d_62_x,2d_62_y,2d_63_x,2d_63_y,2d_64_x,2d_64_y,2d_65_x,2d_65_y,2d_66_x,' \
             '2d_66_y,2d_67_x,2d_67_y'
    random.seed(args.rand)
    try:
        train = open(args.train, 'w')
        test = open(args.test, 'w')
    except FileNotFoundError as e:
        print(e)
        return 0
    train.write(f'{header}\n')
    test.write(f'{header}\n')

    # go throw 300VW_3D
    tr, te = walk_300vw_3d(args.tts, os.path.join(args.path, 'LS3D-W', '300VW-3D'))
    train.write(f'{tr}\n')
    test.write(f'{te}\n')

    # go throw the all other directories in path
    for (dir_path, dir_names, file_names) in os.walk(args.path):
        if dir_path.find('300VW-3D') > -1:
            continue
        landmarks = []
        for file_name in file_names:
            # search for torch files
            file_path = os.path.join(dir_path, file_name)
            if not file_path.endswith('.t7'):
                continue

            # load the points
            points = torchfile.load(file_path)

            # search for image name
            if os.path.isfile(file_path[:-2] + 'jpg'):
                csv_list = [file_path[:-2] + 'jpg']
            elif os.path.isfile(file_path[:-2] + 'png'):
                csv_list = [file_path[:-2] + 'png']
            else:
                continue

            # add landmarks to our list
            for p in points:
                for n in p:
                    csv_list.append(str(n))
            landmarks.append(f"{','.join(csv_list)}")

        # add landmarks from to train or test
        for landmark in landmarks:
            if random.random() > args.tts:
                test.write(landmark + '\n')
            else:
                train.write(landmark + '\n')
    test.close()
    train.close()


if __name__ == '__main__':
    arg = parse_arguments()
    main(arg)
