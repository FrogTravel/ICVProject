import numpy as np


def _iou(bb1, bb2):
    """
    Calculate intersection over union of two boxes
    :param bb1: array of 4 integer numbers [left_top_X, left_top_Y, right_bottom_X, right_bottom_Y]
    :param bb2: same format as bb1
    :return: one float number in interval between 0 and 1
    """
    # calculate intersection box area
    bbi = [max(bb1[0], bb2[0]), min(bb1[1], bb2[1]), min(bb1[2], bb2[2]), max(bb1[3], bb2[3])]
    wi, hi = bbi[2] - bbi[0], bbi[1] - bbi[3]
    intersection_area = wi * hi

    # calculate union box area as sum of area of each box subtract intersection area
    union_area = (bb1[2] - bb1[0]) * (bb1[1] - bb1[3]) + (bb2[2] - bb2[0]) * (bb2[1] - bb2[3]) - intersection_area

    return intersection_area * 1.0 / (union_area + 1e-6)


def _rec(tp, n_gt):
    """
    calculate the recall array
    :param tp: array of size of detected boxes, which contain 1 if the box is true positive
               and 0 if the box is false positive
    :param n_gt: number of ground truth boxes (can not be higher then the number of 1 in tp
    :return: array of tp size with recall value
    """
    return np.cumsum(tp) / n_gt


def _prec(tp):
    """
    calculate the recall array
    :param tp: array of size of detected boxes, which contain 1 if the box is true positive and 0 otherwise
    :return: array of tp size with precision value
    """
    tp = np.cumsum(tp)
    for i in range(1, len(tp)):
        tp[i] /= (i + 1)
    return tp


def _average_precision(rec, prec):
    """
    calculate average precision given recall and precision array's
    :param rec: sorted array of values from [0;1]
    :param prec: sorted array of values from [0;1]
    :return: average precision (value from [0;1]) or None if some trouble happens
    """
    # extend rec and prec to be sure the boundary values are 0 and 1 for recall, and both 0 for precision
    rec = np.concatenate(([0], rec, [1]))
    prec = np.concatenate(([0], prec, [0]))

    # smooth out the zigzag for precision
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # find indexes where recall change it's values
    ind_rec_change = []
    for i in range(1, len(rec)):
        if rec[i] != rec[i - 1]:
            ind_rec_change.append(i)

    # calculate ap
    ap = 0.0
    for i in ind_rec_change:
        ap += (rec[i] - rec[i - 1]) * prec[i]
    return ap


def map_eval(gt_boxes, det_boxes, iou_threshold=0.55):
    """
    calculate mean average precision.
    True positives are the boxes with the IOU over the threshold. Count once for each gt box.
    All other boxes are false positives.
    :param gt_boxes: array of ground truth boxes:
                [[x_top_left_0, y_top_left_0, x_bottom_right_0, y_bottom_right_0], ...]
    :param det_boxes: array of detected boxes with additional field for confidence:
                [[x_top_left_0, y_top_left_0, x_bottom_right_0, y_bottom_right_0, confidence], ...]
    :param iou_threshold: threshold for true positive boxes
    :return: float number between 0 and 1
    """
    # convert gt_boxes and det_boxes to np.array
    gt_boxes = np.array(gt_boxes)
    det_boxes.sort(key=lambda _: -_[4])
    det_boxes = np.array(det_boxes)
    det_boxes = det_boxes[:, :-1]

    # let n_det be the number of detected boxes, and n_gt - the number of gt boxes
    n_det = len(det_boxes)
    n_gt = len(gt_boxes)

    # tp - array of size of det_boxes. 1 - if this box counted as true positive, 0 otherwise
    tp = np.array([0] * n_det)

    # gt_check - array of size of gt_boxes. 1 - if this box used to count some det box as true positive, 0 otherwise
    gt_check = np.array([0] * n_gt)

    # calculate matrix with pairwise mAP value between detected and ground truth boxes
    iou_matrix = np.array([[0.0] * n_gt for _ in range(n_det)])
    for i in range(n_det):
        for j in range(n_gt):
            iou_matrix[i][j] = _iou(det_boxes[i], gt_boxes[j])

    # calculate tp (true positive)
    for i in range(n_det):
        ind_best_gt = np.argmax(iou_matrix[i, :])
        if iou_matrix[ind_best_gt, i] > iou_threshold and gt_check[ind_best_gt] == 0:
            tp[i] = 1
            gt_check[ind_best_gt] = 1

    # calculate req and prec
    rec = _rec(tp, n_gt)
    prec = _prec(tp)

    return _average_precision(rec, prec)


def format_bbox(landmarks_batch):
    """
    Finds bounding box coordinates in format [[x_top_left_0, y_top_left_0, x_bottom_right_0, y_bottom_right_0], ...]
     from landmarks
    :param landmarks: torch.size(batch_size, num_of_faces,68,2)
    :return: torch.size(batch_size,4)
    """
    gt_boxes = []
    for landmarks_set in landmarks_batch:
        for landmarks in landmarks_set:
            x = landmarks[:, 0]
            y = landmarks[:, 1]

            xmin, xmax = min(x), max(x)
            ymin, ymax = min(y), max(y)

            gt_boxes.append([xmin, ymin, xmax, ymax])
    return gt_boxes
