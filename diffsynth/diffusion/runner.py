import os, sys, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _should_enable_tqdm(accelerator: Accelerator):
    return accelerator.is_local_main_process and sys.stdout.isatty() and sys.stderr.isatty()


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    if hasattr(model, "optimizer_param_groups"):
        optimizer_param_groups = model.optimizer_param_groups(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            args=args,
        )
        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    steps_per_epoch = len(dataloader)
    accelerator.print(
        f"[train] start training: epochs={num_epochs}, steps_per_epoch={steps_per_epoch}, "
        f"save_steps={save_steps}, grad_accum={accelerator.gradient_accumulation_steps}, "
        f"lr={learning_rate}, output_path={model_logger.output_path}"
    )
    enable_tqdm = _should_enable_tqdm(accelerator)
    epoch_bar = tqdm(
        range(num_epochs),
        disable=not enable_tqdm,
        desc="Epochs",
        dynamic_ncols=True,
    )
    for epoch_id in epoch_bar:
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        step_bar = tqdm(
            dataloader,
            disable=not enable_tqdm,
            desc=f"Epoch {epoch_id + 1}/{num_epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for step_in_epoch, data in enumerate(step_bar, start=1):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                loss_value = float(loss.detach().float().item())
                accelerator.backward(loss)
                optimizer.step()
                current_step = model_logger.num_steps + 1
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                epoch_loss_sum += loss_value
                epoch_step_count += 1
                avg_loss = epoch_loss_sum / epoch_step_count
                current_lr = scheduler.get_last_lr()[0]
                if enable_tqdm:
                    step_bar.set_postfix(
                        step=current_step,
                        loss=f"{loss_value:.6f}",
                        avg_loss=f"{avg_loss:.6f}",
                        lr=f"{current_lr:.2e}",
                    )
                if accelerator.is_main_process and (
                    current_step == 1 or current_step % 10 == 0 or (save_steps is not None and current_step % save_steps == 0)
                ):
                    accelerator.print(
                        f"[train] epoch={epoch_id + 1}/{num_epochs} step={current_step} "
                        f"epoch_step={step_in_epoch}/{steps_per_epoch} loss={loss_value:.6f} "
                        f"avg_loss={avg_loss:.6f} lr={current_lr:.2e}"
                    )
                if accelerator.is_main_process and save_steps is not None and current_step % save_steps == 0:
                    accelerator.print(
                        f"[train] checkpoint saved at step={current_step}: "
                        f"{os.path.join(model_logger.output_path, f'step-{current_step}.safetensors')}"
                    )
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            if accelerator.is_main_process:
                accelerator.print(
                    f"[train] checkpoint saved at epoch={epoch_id}: "
                    f"{os.path.join(model_logger.output_path, f'epoch-{epoch_id}.safetensors')}"
                )
        if accelerator.is_main_process and epoch_step_count > 0:
            accelerator.print(
                f"[train] epoch {epoch_id + 1}/{num_epochs} finished, avg_loss={epoch_loss_sum / epoch_step_count:.6f}"
            )
    model_logger.on_training_end(accelerator, model, save_steps)
    if accelerator.is_main_process and save_steps is not None and model_logger.num_steps % save_steps != 0:
        accelerator.print(
            f"[train] final checkpoint saved at step={model_logger.num_steps}: "
            f"{os.path.join(model_logger.output_path, f'step-{model_logger.num_steps}.safetensors')}"
        )
    accelerator.print("[train] training finished")


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    enable_tqdm = _should_enable_tqdm(accelerator)
    
    for data_id, data in enumerate(tqdm(dataloader, disable=not enable_tqdm, dynamic_ncols=True)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
