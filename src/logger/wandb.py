from datetime import datetime

import numpy as np
import pandas as pd


class WandBWriter:
    """
    Class for experiment tracking via Weights & Biases.

    The writer supports two kinds of logging:

    1. Batch-level logging:
       loss_train, TRAIN_Accuracy_train, learning_rate_train, etc.

    2. Epoch-level logging:
       loss_epoch, val_loss_epoch, test_loss_epoch, etc.

    Epoch-level charts use epoch_index as their x-axis.
    """

    EPOCH_METRIC_NAMES = [
        "loss_epoch",
        "grad_norm_epoch",
        "TRAIN_Accuracy_epoch",
        "TRAIN_EER_epoch",
        "val_loss_epoch",
        "val_Accuracy_epoch",
        "val_EER_epoch",
        "test_loss_epoch",
        "test_Accuracy_epoch",
        "test_EER_epoch",
    ]

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        Initialize W&B experiment tracking.

        Args:
            logger: project logger.
            project_config (dict): configuration of the current experiment.
            project_name (str): W&B project name.
            entity (str | None): W&B user/team entity.
            run_id (str | None): existing W&B run ID for resume.
            run_name (str | None): human-readable run name.
            mode (str): "online", "offline" or "disabled".
        """
        self.logger = logger
        self.wandb = None
        self.run = None

        try:
            import wandb

            wandb.login()

            self.run_id = run_id

            self.run = wandb.init(
                project=project_name,
                entity=entity,
                config=project_config,
                name=run_name,
                resume="allow",
                id=self.run_id,
                mode=mode,
                save_code=kwargs.get("save_code", False),
            )

            self.wandb = wandb

            # Independent x-axis for epoch-level charts.
            self.wandb.define_metric("epoch_index")

            # W&B 0.28 does not accept patterns such as "*_epoch",
            # so every expected epoch metric is registered explicitly.
            for metric_name in self.EPOCH_METRIC_NAMES:
                self.wandb.define_metric(
                    metric_name,
                    step_metric="epoch_index",
                )

        except ImportError:
            logger.warning(
                "W&B is not installed. Install it with:\n"
                "\tpip install wandb"
            )

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define the current global batch step and logging mode.

        Args:
            step (int): current global training step.
            mode (str): train, val, test, epoch, etc.
        """
        self.mode = mode

        previous_step = self.step
        self.step = step

        if step == 0:
            self.timer = datetime.now()
            return

        duration = datetime.now() - self.timer
        duration_seconds = duration.total_seconds()

        if duration_seconds > 0 and step > previous_step:
            steps_per_second = (
                self.step - previous_step
            ) / duration_seconds

            self.add_scalar(
                "steps_per_sec",
                steps_per_second,
            )

        self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Add the current mode to a logged object name.

        Example:
            loss + train -> loss_train
        """
        if self.mode:
            return f"{object_name}_{self.mode}"

        return object_name

    def _is_active(self):
        """
        Check that the W&B module and run are available.
        """
        return self.wandb is not None and self.run is not None

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Upload a checkpoint file to the active W&B run.
        """
        if not self._is_active():
            return

        self.wandb.save(
            checkpoint_path,
            base_path=save_dir,
        )

    def add_scalar(self, scalar_name, scalar):
        """
        Log one scalar using the current global batch step.
        """
        if not self._is_active():
            return

        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars using the current global batch step.
        """
        if not self._is_active():
            return

        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_epoch_scalars(self, epoch, scalars):
        """
        Log consolidated train/validation/test values once per epoch.

        These metrics use epoch_index as the x-axis and do not depend
        on the batch-level global step.

        Args:
            epoch (int): epoch number starting from 1.
            scalars (dict): epoch summary metrics.
        """
        if not self._is_active():
            return

        payload = {
            "epoch_index": int(epoch),
        }

        for scalar_name, scalar_value in scalars.items():
            epoch_metric_name = f"{scalar_name}_epoch"
            payload[epoch_metric_name] = scalar_value

            # Register unexpected numeric metrics dynamically.
            # Known metrics were already registered in __init__.
            if epoch_metric_name not in self.EPOCH_METRIC_NAMES:
                self.wandb.define_metric(
                    epoch_metric_name,
                    step_metric="epoch_index",
                )

        self.wandb.log(payload)

    def add_image(self, image_name, image):
        """
        Log one image.
        """
        if not self._is_active():
            return

        self.wandb.log(
            {
                self._object_name(image_name): self.wandb.Image(image),
            },
            step=self.step,
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log one audio sample.
        """
        if not self._is_active():
            return

        audio = audio.detach().cpu().numpy().T

        self.wandb.log(
            {
                self._object_name(audio_name): self.wandb.Audio(
                    audio,
                    sample_rate=sample_rate,
                ),
            },
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log HTML/text content.
        """
        if not self._is_active():
            return

        self.wandb.log(
            {
                self._object_name(text_name): self.wandb.Html(text),
            },
            step=self.step,
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log a histogram.
        """
        if not self._is_active():
            return

        values_for_hist = values_for_hist.detach().cpu().numpy()
        np_hist = np.histogram(values_for_hist, bins=bins)

        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(
                values_for_hist,
                bins=512,
            )

        histogram = self.wandb.Histogram(
            np_histogram=np_hist,
        )

        self.wandb.log(
            {
                self._object_name(hist_name): histogram,
            },
            step=self.step,
        )

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log a pandas DataFrame as a W&B table.
        """
        if not self._is_active():
            return

        self.wandb.log(
            {
                self._object_name(table_name): self.wandb.Table(
                    dataframe=table
                ),
            },
            step=self.step,
        )

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()

    def finish(self):
        """
        Finish the active W&B run cleanly.

        This makes the run appear as Finished instead of Crashed.
        """
        if not self._is_active():
            return

        self.wandb.finish()

        self.run = None