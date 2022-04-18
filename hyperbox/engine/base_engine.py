

class BaseEngine:
    """
    Base class for all engines.
    """

    def __init__(
        self,
        trainer,
        model,
        datamodule,
        cfg
    ):
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        self.cfg = cfg

    def run(self):
        raise NotImplementedError
