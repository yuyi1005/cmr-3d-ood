# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmcv.ops import box_iou_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox


@METRICS.register_module()
class CMRMetric(BaseMetric):
    """DOTA evaluation metric.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the DOTA
    large images, which can be found at: ``tools/data/dota/split``.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Only support
            'mAP' now. If is list, the first setting in the list will
             be used to evaluate metric.
        predict_box_type (str): Box type of model results. If the QuadriBoxes
            is used, you need to specify 'qbox'. Defaults to 'rbox'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format. Defaults to False.
        outfile_prefix (str, optional): The prefix of json/zip files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        merge_patches (bool): Generate the full image's results by merging
            patches' results.
        iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
            patches. Defaults to 0.1.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'. Defaults to '11points'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'dota'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = [0.5, 0.75],
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = 'area',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        # voc evaluation metrics
        if isinstance(metric, str):
            metric = [metric]
        for m in metric:
            if m not in ['mAP', 'angle-dev', 'mIoU']:
                raise KeyError(f"metric should be one of ['mAP', 'angle-dev', 'mIoU'], but got {metric}.")
        self.metric = metric
        self.predict_box_type = predict_box_type

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr

        self.use_07_metric = True if eval_mode == '11points' else False

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i].tolist()
                data['score'] = float(scores[i])
                label = int(label) if label.ndim == 0 else label.tolist()
                data['category_id'] = label
                bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        return result_files

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['arrows'] = pred['arrows'].cpu().numpy()

            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape(
                        (-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        for i in range(len(gts)):
            if len(gts[i]['labels'].shape) == 2:
                gts[i]['arrows'] = gts[i]['labels'][:, 1] / 1e5
                gts[i]['labels'] = gts[i]['labels'][:, 0]
            if len(gts[i]['labels_ignore'].shape) == 2:
                gts[i]['labels_ignore'] = gts[i]['labels_ignore'][:, 0]

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()

        # convert predictions to coco format and dump to json file
        _ = self.results2json(preds, outfile_prefix)
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results
        
        # Arrow metric
        if 'angle-dev' in self.metric:
            angle_err = []
            for i in range(len(gts)):
                gt_rbox = gts[i]['bboxes']
                pd_rbox = preds[i]['bboxes']
                gt_a = gts[i]['arrows']
                pd_a = preds[i]['arrows']
                if len(gt_a) == 1 and len(pd_a) == 1:
                    gt_a = np.clip(gt_a[0], 0.0, 1.0)
                    pd_a = np.clip(pd_a[0], 0.0, 1.0)
                    k = np.clip((1 - gt_a**2), 0.0, 1.0)**0.5
                    vec_gt = np.array([np.cos(gt_rbox[0][4]) * k, np.sin(gt_rbox[0][4]) * k, gt_a])
                    k = np.clip((1 - pd_a**2), 0.0, 1.0)**0.5
                    vec_pd = np.array([np.cos(pd_rbox[0][4]) * k, np.sin(pd_rbox[0][4]) * k, pd_a])
                    cos_theta = np.dot(vec_gt, vec_pd)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
                    theta_deg = np.degrees(np.arccos(cos_theta))
                    if not np.isnan(theta_deg):
                        angle_err.append(theta_deg)
                        
                    logger.info(
                        f"{preds[i]['img_id']} "
                        f"{vec_gt:} {vec_pd:} "
                        f"{theta_deg:}"
                        )
            # logger.info(f"Angle errors: {angle_err}")

            if len(angle_err) > 0:
                angle_err_arr = np.array(angle_err)
                mean_err = np.mean(angle_err_arr)
                std_err = np.std(angle_err_arr, ddof=1)  # sample std
            else:
                mean_err, std_err = np.array(-1), np.array(-1)
            eval_results['angle-dev'] = [round(mean_err.item(), 4), round(std_err.item(), 4)]
            logger.info(
                f"\n{'-' * 42}\n"
                f"Mean Angle Deviation: {eval_results['angle-dev'][0]:.2f} "
                f"+/- {eval_results['angle-dev'][1]:.2f}"
            )

        if 'mIoU' in self.metric:
            iou_list = []
            for i in range(len(gts)):
                gt_rbox = gts[i]['bboxes']     # shape (1, 5) expected
                pd_rbox = preds[i]['bboxes']   # shape (1, 5) expected
                if len(gt_rbox) == 1 and len(pd_rbox) == 1:
                    gt_bboxes = np.array(gt_rbox, dtype=np.float32)
                    det_bboxes = np.array(pd_rbox, dtype=np.float32)
                    # Compute rotated IoU
                    ious = box_iou_rotated(
                        torch.from_numpy(det_bboxes).float(),
                        torch.from_numpy(gt_bboxes).float(),
                        aligned=True  # since 1-to-1
                    ).numpy()
                    iou_list.append(float(ious[0]))

            if len(iou_list) > 0:
                mean_iou = float(np.mean(iou_list))
            else:
                mean_iou = 0.0
            eval_results['mIoU'] = mean_iou
            logger.info(
                f"\n{'-' * 42}\n"
                f"Mean IoU: {eval_results['mIoU']:.4f}"
            )

        if 'mAP' in self.metric:
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 4)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results
