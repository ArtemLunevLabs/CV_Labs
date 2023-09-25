import os

import cv2
import numpy as np
import pandas as pd

from solution_example import get_foreground_mask, get_foreground_mask_with_watershed


def evaluate_iou(image_dir: str, anno_dir: str, with_watershed: bool = False) -> tuple:
    """
    Метод для расчёта mean IoU на датасете
    :param image_dir - каталог с фото для анализа
    :param anno_dir - каталог с аннотациями для расчета метрики
    :param with_watershed - выбор функции запуска
    :return значение mean IoU с сохранением оценки по каждому фото в csv-файл
    """

    iou_data = dict()
    total_time_mp = 0
    for _, _, files in os.walk(image_dir):
        assert files is not None, 'no files read'
        for image_name in files:
            # print(f'Processing file {image_name}')
            sample_name = image_name[:image_name.find('.jpg')]

            # acquiring ground truth mask data
            mask_true = cv2.imread(os.path.join(anno_dir, f'{sample_name}.png'))
            assert mask_true is not None, 'mask is None'
            true_points = cv2.cvtColor(mask_true, cv2.COLOR_BGR2GRAY)
            true_points[true_points != 2] = 1
            true_points[true_points == 2] = 0
            true_points = np.argwhere(true_points)
            true_points_set = set([tuple(x) for x in true_points])

            # acquiring predicted mask
            processor = get_foreground_mask if not with_watershed else get_foreground_mask_with_watershed
            pred_points, working_time = processor(image_path=os.path.join(image_dir, image_name))
            assert pred_points is not None, 'pred_points is None'
            total_time_mp += working_time
            pred_points_set = set([tuple(x) for x in pred_points])

            # calculating IoU
            iou = len(true_points_set.intersection(pred_points_set)) / len(true_points_set.union(pred_points_set))

            image_names = iou_data.get('image_names', [])
            image_names.append(sample_name)
            iou_data['image_names'] = image_names

            iou_values = iou_data.get('iou_values', [])
            iou_values.append(iou)
            iou_data['iou_values'] = iou_values

        pd.DataFrame(data=iou_data).to_csv('detailed_results.csv')
        return np.mean(iou_data['iou_values']), total_time_mp / len(files)
