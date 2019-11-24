import math
import torch
import torch.nn as nn


class JointLoss(nn.Module):
    def __init__(self, device, image_size=288, anchors_dims=[[0.24, 0.24],
                                                             [0.12, 0.12],
                                                             [0.08, 0.08],
                                                             [0.28, 0.28],
                                                             [0.15, 0.15]],
                 lambda_noobj=0.5, lambda_coor=5, lambda_landmarks=2):
        """

        :param anchors_dims: (list) of size num_of_anchor_box
                                     containing [width, height]  anchors
        :param lambda_noobj: from YOLO
        :param lambda_coor: from YOLO
        """
        super(JointLoss, self).__init__()
        self.device = device
        self.anchors_dims = anchors_dims
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

        self.lambda_noobj = lambda_noobj
        self.lambda_coor = lambda_coor
        self.image_size = image_size
        self.lambda_landmarks = lambda_landmarks

    def forward(self, prediction, target):
        """
        Joint loss for landmarks(NME) and bounding boxes(localization loss, confidence loss)

        :param prediction: (tuple) containing predictions for bounding boxes and landmarks
        bounding boxes prediction:  torch.size(batch_size, grid_size, grid_size, num_anchor_box, 4+1)
        landmarks prediction: torch.size(batch_size, grid_size, grid_size, num_anchor_box, 68, 2 or 3)

        :param target: (tuple) containing target values  for bounding boxes and landmarks
        bounding boxes targets: torch.size(batch_size, max_num_of_faces, 4+1)
        landmarks targets: torch.size(batch_size, max_num_of_faces, 68, 2 or 3)

        :return: normalized mean error(float), localization loss(float), confidence loss(float),
        best predicted bbox matched with gt (torch.tensor)
        """
        bbox_prediction, landmarks_prediction = prediction
        bbox_prediction, landmarks_prediction = bbox_prediction.to(self.device), landmarks_prediction.to(self.device)
        gt_boxes, gt_conf, obj_mask, noobj_mask, gt_landmarks = self.get_mask(prediction, target)
        # choose only landmarks which corresponds to cells with face in it
        landmarks_pred = landmarks_prediction[obj_mask]
        gt_landmarks = gt_landmarks[obj_mask]
        # calculate Normalized Mean Error
        nme = self.lambda_landmarks * self.nme(gt_landmarks, landmarks_pred, gt_boxes[:, :, :, :, 2:4][obj_mask])
        # calculate localization error
        bbox_pred = bbox_prediction[:, :, :, :, :4][obj_mask].to(self.device)
        gt_boxes = gt_boxes[obj_mask]
        loc_loss = self.lambda_coor * self.smooth_l1_loss(bbox_pred, gt_boxes)
        # calculate confidence loss
        # get conf mask where gt and where there is no gt
        conf_pred = bbox_prediction[:, :, :, :, 4]  # torch.Size([batch_size, grid_size, grid_size, num_of_anchors])
        conf_loss = self.lambda_noobj * self.mse_loss(conf_pred[noobj_mask],
                                                      gt_conf[noobj_mask]) + self.mse_loss(
            conf_pred[obj_mask], gt_conf[obj_mask])

        return nme, loc_loss, conf_loss, self.non_maximum_suppression(bbox_prediction, landmarks_prediction)

    def get_mask(self, prediction, target):
        """
        Tool for calculating the loss
        Calculates masks (filter then used as indexes) for prediction,
         build target values with the same shape as prediction
         and best predicted bboxes
        :param prediction: (tuple) the same as in forward method
        :param target: (tuple) the same as in forward method
        :return: gt_boxes: torch.size(batch_size, grid_size, grid_size, num_anchors, 4),
                 gt_conf: torch.size(batch_size, grid_size, grid_size, num_anchors),
                 mask: torch.size(batch_size, grid_size, grid_size, num_anchors),
                 conf_mask: torch.size(batch_size, grid_size, grid_size, num_anchors),
                 gt_landmarks: torch.size(batch_size, grid_size, grid_size, num_anchors, 68, 2 or 3)
        """
        """
        ground truth bbox - [c_x, c_y, w, h] all values in [0,1] w.r.t the whole image 
        anchor boxes - [0.5, 0.5, w, h] 0.5 w.r.t cell ; w,h in [0,1] w.r.t the whole image
                        store all values w.r.t the whole image 
                        (transform 0.5 to value in [0,1] w.r.t. the whole image in the 'self._get_anchor_boxes)
        predicted bbox - [ln(c_x), ln(c_y), ln(w), ln(h)] c_x, c_y w.r.t. cell w, h w.r.t. image

        1) IoU between gt and anchor, gt -> anchor form: 
            [c_x, c_y, w, h] -> [c_x/9 - int(c_x/9), c_y - int(c_y/9), w, h]  
      
        """
        bbox_target, landmarks_target = target
        bbox_prediction, landmarks_prediction = prediction

        batch_size, grid_size, num_anchors = bbox_prediction.size(0), bbox_prediction.size(1), bbox_prediction.size(3)

        # to mark bboxes with high IoU between predicted and target
        # in other words, mark (with 1) bbox with the face in it w.r.t. ground truth(gt)
        obj_mask = torch.zeros(batch_size, grid_size, grid_size, num_anchors)
        noobj_mask = torch.ones(batch_size, grid_size, grid_size, num_anchors)
        # to store ground truth confidence scores and box coordinates
        gt_conf = torch.zeros(batch_size, grid_size, grid_size, num_anchors)
        gt_boxes = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 4)
        gt_landmarks = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 68 * 2)

        for batch_idx in range(batch_size):
            for target_idx in range(bbox_target.shape[1]):
                # there is no target, continue
                if bbox_target[batch_idx, target_idx].sum() == 0:
                    continue

                # get ground truth box coordinates
                gt_x = bbox_target[batch_idx, target_idx, 0]
                gt_y = bbox_target[batch_idx, target_idx, 1]
                gt_w = bbox_target[batch_idx, target_idx, 2]
                gt_h = bbox_target[batch_idx, target_idx, 3]

                # get grid box indices of ground truth box
                # coordinates gt_x*grid_size and gt_y*grid_size w.r.t. cell size (one cell 1x1)
                gt_i = int(gt_x * grid_size)
                gt_j = int(gt_y * grid_size)
                gt_box = torch.tensor([gt_x, gt_y, gt_w, gt_h]).unsqueeze(0).to(
                    self.device)  # torch.size(0,4)
                # get anchor box that has the highest iou with ground truth
                anchor_boxes = self._get_anchor_boxes(gt_i, gt_j, grid_size)
                anchor_iou = self._get_iou(gt_box, anchor_boxes)
                # best matching anchor box
                best_anchor_idx = torch.argmax(anchor_iou)

                # mark best predicted box
                obj_mask[batch_idx, gt_j, gt_i, best_anchor_idx] = 1
                noobj_mask[batch_idx, gt_j, gt_i, best_anchor_idx] = 0

                gt_conf[batch_idx, gt_j, gt_i, best_anchor_idx] = 1
                gt_boxes[batch_idx, gt_j, gt_i, best_anchor_idx] = torch.log1p(gt_box)
                gt_landmarks[batch_idx, gt_j, gt_i, best_anchor_idx] = landmarks_target[batch_idx, target_idx].view(
                    68 * 2)

        obj_mask = obj_mask.byte()  # to use then as indexes of tensor
        noobj_mask = noobj_mask.byte()  # to use then as indexes of tensor

        return gt_boxes.to(self.device), gt_conf.to(self.device), obj_mask, noobj_mask, gt_landmarks.to(
            self.device)

    def nme(self, gt_landmarks, pred_landmarks, boxes_shapes):
        """
        Normalized mean error (NME) defined as the Euclidean distance
        between the predicted and ground truth 2D landmarks averaged over
        68 landmarks and normalized by the bounding box dimensions

        :param gt_landmarks: torch.size(batch_size, 68*2)
        :param pred_landmarks: torch.size(batch_size, 68*2)
        :param boxes_shapes: [[width, height], ...]
        :return: (float)
        """
        nme = 0.0
        batch_size = gt_landmarks.shape[0]
        gt_landmarks = gt_landmarks.view(batch_size, 68, 2)
        pred_landmarks = pred_landmarks.view(batch_size, 68, 2)
        for batch_idx in range(batch_size):
            sum = 0
            for i in range(68):
                euclidean_dist = torch.dist(gt_landmarks[batch_idx, i], pred_landmarks[batch_idx, i], 2)
                sum += euclidean_dist
            normalization_factor = math.sqrt(
                boxes_shapes[batch_idx][0] * boxes_shapes[batch_idx][1])
            nme += sum / (normalization_factor * 68 * batch_size)
        return nme

    def _get_iou(self, box1, box2):
        """
        Calculates IoU for two tensors of bboxes
        :param box1: torch.size(num_of_boxes_1, 4)
        :param box2: torch.size(num_of_boxes_2, 4)
        :return: torch.size(max(num_of_boxes_1, num_of_boxes_2), 4)
        """

        b1 = self._format_bbox(box1)
        b2 = self._format_bbox(box2)
        b1_x1, b1_x2, b1_y1, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        b2_x1, b2_x2, b2_y1, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        intersect_x1 = torch.max(b1_x1, b2_x1)
        intersect_y1 = torch.max(b1_y1, b2_y1)
        intersect_x2 = torch.min(b1_x2, b2_x2)
        intersect_y2 = torch.min(b1_y2, b2_y2)

        intersect_area = (intersect_x2 - intersect_x1 + 1) * (intersect_y2 - intersect_y1 + 1)

        # union area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = intersect_area / (b1_area + b2_area - intersect_area + 1e-16)
        return iou

    def _get_anchor_boxes(self, g_i, g_j, grid_size, center_x=0.5, center_y=0.5):
        """
        Creates list of anchor boxes with given dimensions (height, width) w.r.t image size
        anchor box =  [center_x, center_y, width, height] w.r.t. the whole image

        :param center_x: x coordinate of the center of the anchor box w.r.t. cell
        :param center_y: y coordinate of the center of the anchor box w.r.t. cell
        :return: (tensor) torch.size(len(of anchor_aspect_ratios), 4)
        """

        anchors = []
        center_x = (center_x + g_i) / grid_size
        center_y = (center_y + g_j) / grid_size
        for dims in self.anchors_dims:
            anchors.append([center_x, center_y, dims[0], dims[1]])
        return torch.tensor(anchors).to(self.device)

    def _format_bbox(self, box):
        """
        Convert [[c_x, c_y, w, h], ...] to [[x_top_left_0, y_top_left_0, x_bottom_right_0, y_bottom_right_0], ...]
        :param box: (torch.tensor) [[c_x, c_y, w, h], ...]
        :return: (torch.tensor) [[x_top_left_0, y_top_left_0, x_bottom_right_0, y_bottom_right_0], ...]
        """

        x1, x2 = (box[:, 0] - box[:, 2] / 2).unsqueeze(0), (
                box[:, 0] + box[:, 2] / 2).unsqueeze(0)
        y1, y2 = (box[:, 1] - box[:, 3] / 2).unsqueeze(0), (
                box[:, 1] + box[:, 3] / 2).unsqueeze(0)
        return torch.cat((torch.t(x1 * self.image_size), torch.t(x2 * self.image_size), torch.t(y1 * self.image_size),
                          torch.t(y2 * self.image_size)), 1)

    def non_maximum_suppression(self, bbox_prediction, landmarks_prediction, conf_thresh=0.5, iou_thresh=0.5):
        batch_size, grid_size, num_anchors = bbox_prediction.size(0), bbox_prediction.size(1), bbox_prediction.size(3)

        bbox_prediction = bbox_prediction.view(batch_size, grid_size * grid_size * num_anchors, 4 + 1)
        bbox_prediction = torch.cat((torch.expm1(bbox_prediction[:, :, :4]), bbox_prediction[:, :, 4:]), dim=2)

        landmarks_prediction = landmarks_prediction.view(batch_size, grid_size * grid_size * num_anchors, 68 * 2)

        conf_mask = (bbox_prediction[:, :, 4] > conf_thresh).float().unsqueeze(2)
        bbox_pred = bbox_prediction * conf_mask
        landmarks_pred = landmarks_prediction * conf_mask

        bbox = []
        landmarks = []
        for i in range(batch_size):
            image_bbox = bbox_pred[i]
            image_landmarks = landmarks_pred[i]
            max_conf_idx = torch.argmax(image_bbox[:, 4])
            ious = self._get_iou(image_bbox[max_conf_idx].unsqueeze(0), image_bbox)
            mask = ious < iou_thresh
            bbox.append(image_bbox[mask.byte()])
            bbox.append(image_bbox[max_conf_idx].unsqueeze(0))
            landmarks.append(image_landmarks[mask.byte()])
            landmarks.append(image_landmarks[max_conf_idx].unsqueeze(0))
        bbox = torch.cat(bbox, dim=0)
        bbox = torch.cat((self._format_bbox(bbox), bbox[:, 4:]), dim=1)
        landmarks = torch.cat(landmarks, dim=0)
        return bbox, landmarks.view(-1, 68, 2)
