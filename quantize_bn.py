import argparse
import os
import time

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import tqdm
import copy

import model.yolov3
import model.yolov3_quant
import utils.datasets
import utils.utils

#from test import test_evaluate
backend = "qnnpack"  # running on a x86 CPU. Use "qnnpack" if running on ARM.
def evaluate(model, path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_workers, device):
    # 모델을 evaluation mode로 설정
    model.eval()

    # 데이터셋, 데이터로더 설정
    dataset = utils.datasets.ListDataset(path, image_size, augment=False, multiscale=False)
    indices = torch.randperm(len(dataset))
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            sampler=sampler,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            collate_fn=dataset.collate_fn)


    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    entire_time = 0
    for _, images, targets in tqdm.tqdm(dataloader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        # Extract labels
        labels.extend(targets[:, 1].tolist())

        # Rescale targets
        targets[:, 2:] = utils.utils.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        # Predict objects
        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = utils.utils.non_max_suppression(outputs, conf_thres, nms_thres)
        entire_time += time.time() - start_time

        # Compute true positives, predicted scores and predicted labels per batch
        sample_metrics.extend(utils.utils.get_batch_statistics(outputs, targets, iou_thres))

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # Compute AP
    precision, recall, AP, f1, ap_class = utils.utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute inference time and fps
    inference_time = entire_time / dataset.__len__()
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    return precision, recall, AP, f1, ap_class, inference_time, fps

def fuse_model(model):
    print('\nFusing model...\n')
    modules_to_fuse = ["0", "1"]
    #fused_model = copy.deepcopy(model)
    model.eval()
    for n0, m0 in model.named_children():
        for n1, m1 in m0.named_children():
            if isinstance(m1, nn.Sequential):
                for n2, m2 in m1.named_children():
                    if isinstance(m2, nn.Sequential):
                        torch.ao.quantization.fuse_modules(m2, modules_to_fuse, inplace = True)
            if isinstance(m1, nn.Sequential) and len(m1) == 3:
                torch.ao.quantization.fuse_modules(m1, modules_to_fuse, inplace = True)
    return model

def quant_model(model, path, image_size, batch_size, num_workers, device):
    model = fuse_model(model)
    print('\nInserting observers...\n')
    #print(model)
    for n, m0 in model.named_children():
        if "final" in n:
            pass
        elif "yolo_layer" in n:
            pass
        else:
            m0.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(model, inplace = True)
    #print("-----------------------------------\n")

    # 모델을 evaluation mode로 설정
    model.eval()

    # 데이터셋, 데이터로더 설정
    dataset = utils.datasets.ListDataset(path, image_size, augment=False, multiscale=False)

    # trainset에서 무작위 1000장 추출
    indices = torch.randperm(len(dataset))[:2000]
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            sampler=sampler,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            collate_fn=dataset.collate_fn)
    print('\nCalibrating model...\n')
    for _, images, targets in tqdm.tqdm(dataloader, desc='Calibration', leave=False):
        if targets is None:
            continue
        with torch.inference_mode():
                images = images.to(device)
                model(images)
    print('\nQuantizing model...\n')
    model=model.to('cpu')
    model.eval()
    

    torch.ao.quantization.convert(model, inplace=True)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_voc.pth",
                        help="path to pretrained weights file")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    args = parser.parse_args()
    print(args)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))

    # 데이터셋 설정값을 가져오기
    data_config = utils.utils.parse_data_config(args.data_config)
    train_path = data_config['train']
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = utils.utils.load_classes(data_config['names'])

    # 모델 준비하기
    original_model = model.yolov3.YOLOv3(args.image_size, num_classes).to(device)
    quantized_model = model.yolov3_quant.YOLOv3(args.image_size, num_classes).to(device)
    if args.pretrained_weights.endswith('.pth'):
        original_model.load_state_dict(torch.load(args.pretrained_weights))
        quantized_model.load_state_dict(torch.load(args.pretrained_weights))

    else:
        original_model.load_darknet_weights(args.pretrained_weights)
        quantized_model.load_darknet_weights(args.pretrained_weights)

    # Fuse and Quantize model
    quantized_model = quant_model(quantized_model,
                                    path=train_path,
                                    image_size=args.image_size,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    device=device)

    # checkpoint file 저장
    save_dir = os.path.join('quantized', now)
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = os.path.split(args.data_config)[-1].split('.')[0]
    torch.save(quantized_model.state_dict(), os.path.join(save_dir, 'yolov3_{}_PTQ.pth'.format(dataset_name)))

    # 검증 데이터셋으로 양자화 모델을 평가
    quantized_model=quantized_model.to("cpu")
    print('\nEvaluate quantized model...\n')
    precision, recall, AP, f1, ap_class, inference_time, fps = evaluate(quantized_model,
                                                                        path=valid_path,
                                                                        iou_thres=args.iou_thres,
                                                                        conf_thres=args.conf_thres,
                                                                        nms_thres=args.nms_thres,
                                                                        image_size=args.image_size,
                                                                        batch_size=args.batch_size,
                                                                        num_workers=args.num_workers,
                                                                        device="cpu")

    # AP, mAP, inference_time 출력
    print('Quantized Model\'s Average Precisions:')
    for i, class_num in enumerate(ap_class):
        print('\tClass {} ({}) - AP: {:.02f}'.format(class_num, class_names[class_num], AP[i] * 100))
    print('mAP: {:.02f}'.format(AP.mean() * 100))
    print('Inference_time (ms): {:.02f}'.format(inference_time))
    print('FPS: {:.02f}'.format(fps))