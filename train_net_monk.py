#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

from train.detectron2.mapper import MonkDatasetMapper
from trains import Task

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

from monk.config.classes import Classes
from monk.config.models import get_config_detectron2 as get_cfg
from monk.data.catalog import DatasetCatalog
from monk.data.tables.fill_database import FilesDB
from monk.evaluation.evaluators import CustomF1Evaluator, Instance2SemEvaluator, NumInstancesEvaluator  # TODO: add AP
from monk.evaluation.utils import Monk2DetectronEvaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_types = cfg.TEST.EVALUATOR_TYPES
        for evaluator_type in evaluator_types:
            if evaluator_type == "instance2sem":
                evaluator_list.append(
                    Monk2DetectronEvaluator(
                        Instance2SemEvaluator,
                        dataset_name=dataset_name,
                        distributed=cfg.TEST.DISTRIBUTED,
                        output_dir=output_folder,
                        conf_thresholds=cfg.TEST.CONF_THRESHOLDS,
                        contour_ignore=cfg.TEST.INSTANCE2SEM.CONTOUR_IGNORE,
                        downsize_longer_side=cfg.TEST.DOWNSIZE_LONGER_SIDE,
                        crop_car=cfg.INPUT.CROP_CAR,
                        crop_car_extend_pct=cfg.INPUT.CROP_CAR_EXTEND_BBOX,
                        process_mode=cfg.TEST.PROCESS_MODE,
                    )
                )
            elif evaluator_type == "customf1":
                evaluator_list.append(
                    Monk2DetectronEvaluator(
                        CustomF1Evaluator,
                        dataset_name=dataset_name,
                        distributed=cfg.TEST.DISTRIBUTED,
                        output_dir=output_folder,
                        conf_thresholds=cfg.TEST.CONF_THRESHOLDS,
                        iou_thresh=dict(cfg.TEST.CUSTOMF1.IOU_THRESHOLDS),
                        do_avg_metrics=cfg.TEST.CUSTOMF1.AVG_METRICS,
                        weights=dict(cfg.TEST.CUSTOMF1.WEIGHTS),
                        downsize_longer_side=cfg.TEST.DOWNSIZE_LONGER_SIDE,
                        groupping_distance_thresh=cfg.TEST.GROUPPING_THRESH,
                        crop_car=cfg.INPUT.CROP_CAR,
                        crop_car_extend_pct=cfg.INPUT.CROP_CAR_EXTEND_BBOX,
                        process_mode=cfg.TEST.PROCESS_MODE,
                    )
                )
            elif evaluator_type == "ap":
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, cfg.TEST.DISTRIBUTED, output_folder))
            elif evaluator_type == "num_instances":
                evaluator_list.append(
                    Monk2DetectronEvaluator(
                        NumInstancesEvaluator,
                        dataset_name=dataset_name,
                        distributed=cfg.TEST.DISTRIBUTED,
                        output_dir=output_folder,
                        conf_thresholds=cfg.TEST.CONF_THRESHOLDS,
                        crop_car=cfg.INPUT.CROP_CAR,
                        crop_car_extend_pct=cfg.INPUT.CROP_CAR_EXTEND_BBOX,
                        process_mode=cfg.TEST.PROCESS_MODE,
                    )
                )
            else:
                raise NotImplementedError
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        hooks_ = super().build_hooks()
        return hooks_

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = MonkDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = MonkDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup_allegro():
    training_name = os.getenv('TRAINING_NAME', "NO_NAME")
    expfolder = os.getenv('EXPFOLDER', "?")
    return Task.init(
        project_name="Monk Detectron2",
        task_name=" ".join([training_name, expfolder]),
        task_type=Task.TaskTypes.training,
    )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(Path(args.config_file).parent)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    task = setup_allegro()
    task.connect(cfg, 'cfg')
    register_monk_datasets(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    return trainer.train()


def _get_monk_instances_meta(classes):
    continous_classes = classes.to_continous_classes()
    cats = continous_classes.to_detectron2()
    for cat in cats:
        cat["color"] = (0, 0, 0)
    thing_colors = [k["color"] for k in cats if k["isthing"] == 1]

    thing_dataset_id_to_contiguous_id = classes.map(continous_classes)
    thing_classes = [k["name"] for k in cats if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "classes": continous_classes,
    }
    return ret


def register_monk_datasets(cfg):
    args_dataset = {}
    if cfg.DATASETS.MIN_SIZE_SCRATCH != 0:
        args_dataset["min_size_scratch"] = cfg.DATASETS.MIN_SIZE_SCRATCH
    if cfg.DATASETS.MIN_SIZE_DENT != 0:
        args_dataset["min_size_dent"] = cfg.DATASETS.MIN_SIZE_DENT

    versions = {"train": cfg.DATASETS.TRAIN_VERSION, "test": cfg.DATASETS.TEST_VERSION}
    datasets = {x: "train" for x in cfg.DATASETS.TRAIN}
    datasets.update({x: "test" for x in cfg.DATASETS.TEST})

    # Read classes metadata from first annotation file
    dataset = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else cfg.DATASETS.TEST[0]
    if dataset in DatasetCatalog.CUSTOM_DATASETS:
        ann_file = DatasetCatalog.CUSTOM_DATASETS[dataset]
    elif dataset.startswith("/"):
        ann_file = dataset
    else:
        dataset_name, dataset_set = dataset.rsplit("_", 1)
        dset = "train" if cfg.DATASETS.TRAIN else "test"
        version = versions[dset]
        ann_file = DatasetCatalog.get_ann_file(
            dataset_name=dataset_name,
            dataset_task=cfg.DATASETS.TASK,
            dataset_set=dataset_set,
            dataset_subcats=cfg.DATASETS.SUBCATS,
            dataset_version=version,
            **args_dataset,
            create_if_not_exists=True,
        )

    with open(ann_file, "r") as f:
        data = json.load(f)
    dataset_classes = Classes.from_coco_categories(data["categories"])
    metadata = _get_monk_instances_meta(dataset_classes)

    if cfg.MODEL.ROI_HEADS.NUM_CLASSES != len(dataset_classes):
        raise ValueError(
            f"Number of classes in dataset: {len(dataset_classes)} but cfg.MODEL.ROI_HEADS.NUM_CLASSES"
            " says {cfg.MODEL.ROI_HEADS.NUM_CLASSES}"
        )
    metadata["classes"].to_file(Path(cfg.OUTPUT_DIR) / "classes.json")

    # Register datasets
    for dataset, mode in datasets.items():
        if dataset in DatasetCatalog.CUSTOM_DATASETS:
            ann_file = DatasetCatalog.CUSTOM_DATASETS[dataset]
        elif dataset.startswith("/"):
            ann_file = dataset
        else:
            dataset_name, dataset_set = dataset.rsplit("_", 1)
            ann_file = DatasetCatalog.get_ann_file(
                dataset_name=dataset_name,
                dataset_task=cfg.DATASETS.TASK,
                dataset_set=dataset_set,
                dataset_subcats=cfg.DATASETS.SUBCATS,
                dataset_version=versions[mode],
                create_if_not_exists=True,
                **args_dataset,
            )
        register_coco_instances(dataset, metadata, str(ann_file), str(FilesDB.IMAGES_ROOT))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
