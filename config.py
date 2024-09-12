class Config:
    def __init__(
        self,
        vocab_size: int = 30256,
        vocab_path: str = "data/vocab.txt",
        weights_path: str = "TO_BE_DEFINED",
        embedding_size: int = 512,
        context_size: int = 512,
        nb_blocks: int = 6,
        batch_size: int = 64,
        warmup_steps: int = 4000,
        beta1: float = 0.9,
        beta2: float = 0.98,
        epsilon: float = 1e-9,
        dropout_rate: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        self.vocab_path = vocab_path
        self.weights_path = weights_path

        # constants
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.nb_blocks = nb_blocks

        # training
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        # adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # ~ 1e5 steps for training

    def get_model_size(self) -> int:
        return 1

    def get_lr(self, step: int) -> float:
        return (self.embedding_size**-0.5) * min(
            step**-0.5, step * (self.warmup_steps**-1.5)
        )


config = Config()
print(config.get_lr(2000))
