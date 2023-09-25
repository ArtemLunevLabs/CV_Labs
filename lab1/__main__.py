from metric_evaluation import evaluate_iou

if __name__ == '__main__':
    metric_res, time_mp = evaluate_iou(image_dir='./images', anno_dir='./annotations', with_watershed=False)
    print(f'Metric result - {metric_res}, speed - {time_mp} (s / mp)')
