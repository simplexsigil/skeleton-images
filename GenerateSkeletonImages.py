import argparse
import glob
import multiprocessing as mp
import os
from typing import List

import ImgType


def get_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument('--data_path', type=str, help='directory containing the skeleton data')
    parser.add_argument('--img_type', type=int, choices=[1, 2, 3], help='Image type to compute'
                                                                        '1 - CaetanoMagnitude (SkeleMotion - AVSS2019)'
                                                                        '2 - CaetanoOrientation (SkeleMotion - AVSS2019)'
                                                                        '3 - CaetanoTSRJI (TSRJI - SIBGRAPI2019)',
                        default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Workers')

    parser.add_argument('--temp_dist', nargs='+', type=int, help='Temporal distance between frames', default=1)
    parser.add_argument('--path_to_save', type=str, help='directory to save the skeleton images')
    parser.add_argument('--no-resize', nargs='?', type=bool, const=True,
                        help='If true, images will be resized to a fixed width of 100, independent of frame count.',
                        default=False)
    parser.add_argument('--force', nargs='?', type=bool, const=True,
                        help='If true, images will be replaced although they already exist.', default=False)
    parser.add_argument('--input_format', type=str, choices=["nturgbd_csv", "openpose_json"], default="openpose_json",
                        help='If true, images will be resized to a fixed width of 100, independent of frame count.')
    args = parser.parse_args()
    print(args)
    return args


def save_extraction_list(list_extraction: List[str], path_to_save: str = '') -> None:
    file = open(os.path.join(path_to_save, 'extraction_list.txt'), 'w')
    file.writelines("%s\n" % l for l in list_extraction)
    file.close()


def get_skeleton_files_csv(data_path: str) -> List[str]:
    file_list = []
    for file in glob.glob(os.path.join(data_path, '*.skeleton')):
        file_list.append(file)
    return file_list


def get_skeleton_files_json(data_path: str) -> List[str]:
    file_list = []
    for file in glob.glob(os.path.join(data_path, '**', '*.json')):
        file_list.append(file)
    return file_list


def check_path(path_to_check: str) -> None:
    try:
        if not os.path.exists(path_to_check):
            print('Creating path: ' + path_to_check)
            os.makedirs(path_to_check)
            print('Path ' + path_to_check + ' OK')
    except OSError:
        print('Error: Creating directory. ' + path_to_check)


def worker(args: tuple):
    obj, method_name, skl_file, path_to_save, temp_dist, resize = args
    getattr(obj, 'set_temporal_scale')([temp_dist])
    getattr(obj, method_name)(skl_file, path_to_save, resize)
    del obj


def main(parser: argparse.ArgumentParser) -> None:
    args = get_arguments(parser)
    print(args)
    check_path(args.path_to_save)
    skl_list = []
    if args.input_format == "nturgbd_csv":
        skl_list = get_skeleton_files_csv(args.data_path)
    elif args.input_format == "openpose_json":
        skl_list = get_skeleton_files_json(args.data_path)
        if not args.force:
            ending = ""
            if args.img_type == 1:
                ending = "_CaetanoMagnitude.json.npz"
            elif args.img_type == 2:
                ending = "_CaetanoOrientation.json.npz"

            existing_files = glob.glob(f"{args.path_to_save}/**/*{ending}")
            existing_ids = set(map(lambda fl: os.path.split(fl)[-1][:11], existing_files))
            sk_ids = list(map(lambda fl: os.path.split(fl)[-1][:11], skl_list))

            count = len(skl_list)
            skl_list = [fl for fl, fl_id in zip(skl_list, sk_ids) if fl_id not in existing_ids]
            print(f"Skipped {count - len(skl_list)} files due to preexisting content on location.")
            print(f"{len(skl_list)} files are remaining.")
    else:
        raise ValueError

    obj_list = [ImgType.class_img_types[args.img_type](args.input_format) for _ in range(0, len(skl_list))]
    pool = mp.Pool(args.num_workers, maxtasksperchild=100)

    pool.map(worker, ((obj, 'process_skl_file', skl_file, args.path_to_save, args.temp_dist, not args.no_resize) for
                      obj, skl_file in zip(obj_list, skl_list)))
    pool.close()
    pool.join()
    pool.terminate()
    print('End')


if __name__ == '__main__':
    main(argparse.ArgumentParser())
