from ..core.loader import load_model, hash_model_file
from ..core.vram import AutoWrappedModule
from ..configs import MODEL_CONFIGS, VRAM_MANAGEMENT_MODULE_MAPS, VERSION_CHECKER_MAPS
import importlib, json, os, torch


class ModelPool:
    def __init__(self):
        self.model = []
        self.model_name = []
        self.model_path = []
        
    def import_model_class(self, model_class):
        split = model_class.rfind(".")
        model_resource, model_class = model_class[:split], model_class[split+1:]
        model_class = importlib.import_module(model_resource).__getattribute__(model_class)
        return model_class
    
    def need_to_enable_vram_management(self, vram_config):
        return vram_config["offload_dtype"] is not None and vram_config["offload_device"] is not None
    
    def fetch_module_map(self, model_class, vram_config):
        if self.need_to_enable_vram_management(vram_config):
            if model_class in VRAM_MANAGEMENT_MODULE_MAPS:
                vram_module_map = VRAM_MANAGEMENT_MODULE_MAPS[model_class] if model_class not in VERSION_CHECKER_MAPS else VERSION_CHECKER_MAPS[model_class]()
                module_map = {self.import_model_class(source): self.import_model_class(target) for source, target in vram_module_map.items()}
            else:
                module_map = {self.import_model_class(model_class): AutoWrappedModule}
        else:
            module_map = None
        return module_map
    
    def load_model_file(self, config, path, vram_config, vram_limit=None, state_dict=None):
        model_class = self.import_model_class(config["model_class"])
        model_config = config.get("extra_kwargs", {})
        if "state_dict_converter" in config:
            state_dict_converter = self.import_model_class(config["state_dict_converter"])
        else:
            state_dict_converter = None
        module_map = self.fetch_module_map(config["model_class"], vram_config)
        model = load_model(
            model_class, path, model_config,
            vram_config["computation_dtype"], vram_config["computation_device"],
            state_dict_converter,
            use_disk_map=True,
            vram_config=vram_config, module_map=module_map, vram_limit=vram_limit,
            state_dict=state_dict,
        )
        return model
    
    def default_vram_config(self):
        vram_config = {
            "offload_dtype": None,
            "offload_device": None,
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cpu",
        }
        return vram_config
    
    def auto_load_model(self, path, vram_config=None, vram_limit=None, clear_parameters=False, state_dict=None):
        print(f"Loading models from: {json.dumps(path, indent=4)}")
        if vram_config is None:
            vram_config = self.default_vram_config()

        # Minimal override for OmniShow audio tasks:
        # If `DIFFSYNTH_OMNISHOW_FORCE_DIT` is set to:
        # - a DiT checkpoint file path: exact match against current `path`, OR
        # - a model directory: prefix match against sharded DiT weight lists
        # then skip hash auto-detection and load it as WanModel with `omnishow_enable_audio_inject=True`.
        force_target = os.environ.get("DIFFSYNTH_OMNISHOW_FORCE_DIT", "").strip()
        if force_target:
            paths = path if isinstance(path, list) else [path]
            # Normalize to an absolute, real path to avoid relative/absolute mismatch.
            tgt = os.path.realpath(os.path.abspath(os.path.normpath(force_target)))
            tgt_prefix = tgt.rstrip(os.sep) + os.sep
            # NOTE: `tgt` might be a model directory that also contains T5/VAE/CLIP files.
            # Only treat *DiT weights* as matched to avoid accidentally forcing those other modules.
            exact_ckpt = any(
                isinstance(p, str)
                and p.endswith(".safetensors")
                and os.path.realpath(os.path.abspath(os.path.normpath(p))) == tgt
                for p in paths
            )
            dit_shards_in_dir = any(
                isinstance(p, str)
                and p.endswith(".safetensors")
                and ("diffusion_pytorch_model" in os.path.basename(p))
                and os.path.realpath(os.path.abspath(os.path.normpath(p))).startswith(tgt_prefix)
                for p in paths
            )
            matched = exact_ckpt or dit_shards_in_dir
            if matched:
                # Align force branch with the hash branch: pick the same MODEL_CONFIGS entry
                # that the hash branch would resolve for local Wan2.1-I2V-14B base shards
                # (hash=6bfcfb3b342cb286ce886889d519a77e, no state_dict_converter). Both 480P
                # and 720P local shards resolve to this same entry on this machine.
                variant = os.environ.get("DIFFSYNTH_OMNISHOW_FORCE_DIT_VARIANT", "").strip().lower()
                if variant not in {"i2v_14b_480", "i2v_14b_720"}:
                    raise ValueError(
                        "OmniShow audio only supports Wan2.1-I2V-14B-480P / 720P, "
                        f"but got DIFFSYNTH_OMNISHOW_FORCE_DIT_VARIANT={variant!r}."
                    )
                TARGET_HASH = "6bfcfb3b342cb286ce886889d519a77e"
                base = None
                for c in MODEL_CONFIGS:
                    if c.get("model_hash") == TARGET_HASH \
                            and c.get("model_name") == "wan_video_dit" \
                            and c.get("model_class") == "diffsynth.models.wan_video_dit.WanModel":
                        base = c
                        break
                if base is None:
                    raise ValueError(
                        f"OmniShow audio force branch cannot find MODEL_CONFIGS entry with hash={TARGET_HASH}."
                    )
                print(
                    "[ModelPool] DIFFSYNTH_OMNISHOW_FORCE_DIT matched; "
                    f"variant={variant} -> reuse MODEL_CONFIGS[hash={TARGET_HASH}] "
                    f"extra_kwargs={base.get('extra_kwargs')}, forcing omnishow_enable_audio_inject=True"
                )
                cfg = dict(base)
                extra = dict(base.get("extra_kwargs", {}) or {})
                extra["omnishow_enable_audio_inject"] = True
                cfg["extra_kwargs"] = extra
                model = self.load_model_file(cfg, path, vram_config, vram_limit=vram_limit, state_dict=state_dict)
                if clear_parameters:
                    self.clear_parameters(model)
                self.model.append(model)
                self.model_name.append(cfg["model_name"])
                self.model_path.append(path)
                model_info = {"model_name": cfg["model_name"], "model_class": cfg["model_class"], "extra_kwargs": cfg.get("extra_kwargs")}
                print(f"Loaded model: {json.dumps(model_info, indent=4)}")
                return
    
        model_hash = hash_model_file(path)
        loaded = False
        for config in MODEL_CONFIGS:
            if config["model_hash"] == model_hash:
                model = self.load_model_file(config, path, vram_config, vram_limit=vram_limit, state_dict=state_dict)
                if clear_parameters: self.clear_parameters(model)
                self.model.append(model)
                model_name = config["model_name"]
                self.model_name.append(model_name)
                self.model_path.append(path)
                model_info = {"model_name": model_name, "model_class": config["model_class"], "extra_kwargs": config.get("extra_kwargs")}
                print(f"Loaded model: {json.dumps(model_info, indent=4)}")
                loaded = True
        if not loaded:
            raise ValueError(f"Cannot detect the model type. File: {path}. Model hash: {model_hash}")
    
    def fetch_model(self, model_name, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available. This is not an error.")
            model = None
        elif len(fetched_models) == 1:
            print(f"Using {model_name} from {json.dumps(fetched_model_paths[0], indent=4)}.")
            model = fetched_models[0]
        else:
            if index is None:
                model = fetched_models[0]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths[0], indent=4)}.")
            elif isinstance(index, int):
                model = fetched_models[:index]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths[:index], indent=4)}.")
            else:
                model = fetched_models
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths, indent=4)}.")
        return model

    def clear_parameters(self, model: torch.nn.Module):
        for name, module in model.named_children():
            self.clear_parameters(module)
        for name, param in model.named_parameters(recurse=False):
            setattr(model, name, None)
