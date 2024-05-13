import os
import copy
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
import pickle as pkl

if __name__ == '__main__':
    config = 'projects/configs/safead_sparse4dv3_temporal_r50_1x8_bs6_256x704_with_tl.py'
    result_file = '/home/goel/carnet/explore/Sparse4D/results/safead_base_with_tl.pkl'

    cfg = Config.fromfile(config)

    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    val_data = copy.deepcopy(cfg.data.val)
    val_dataset = build_dataset(val_data)

    with open(result_file, 'rb') as f:
        results = pkl.load(f)

    val_dataset.show(results=results, pipeline=cfg.vis_pipeline)
