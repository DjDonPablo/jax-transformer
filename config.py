class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 10256,
        vocab_path: str = "data/vocab.txt",
        weights_path: str = "TO_BE_DEFINED",
        embedding_dim: int = 512,
        key_query_dim: int = 64,
        context_size: int = 512,
        nb_blocks: int = 6,
        nb_heads: int = 8,
        batch_size: int = 64,
        warmup_steps: int = 4000,
        beta1: float = 0.9,
        beta2: float = 0.98,
        epsilon: float = 1e-9,
        dropout_rate: float = 0.1,
        label_smoothing: float = 0.1,
        top_k: int = 5,
        temp: float = 1.0,
    ) -> None:
        self.vocab_path = vocab_path
        self.weights_path = weights_path

        # constants
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.key_query_dim = key_query_dim
        self.context_size = context_size
        self.nb_blocks = nb_blocks
        self.nb_heads = nb_heads

        # training
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        # adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # inference
        self.top_k = top_k
        self.temp = temp  # acts on softmax

        # ~ 1e5 steps for training

    def get_model_size(self) -> int:
        return 1

    def get_lr(self, step: int) -> float:
        return (self.embedding_size**-0.5) * min(
            step**-0.5, step * (self.warmup_steps**-1.5)
        )


config = TransformerConfig()
print(config.get_lr(2000))
