class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 10256,
        vocab_path: str = "data/vocab.txt",
        weights_path: str = "TO_BE_DEFINED",
        embedding_dim: int = 512,
        key_query_dim: int = 64,
        context_size: int = 512,
        nb_layers: int = 6,
        nb_heads: int = 8,
        batch_size: int = 64,
        warmup_steps: int = 2000,
        beta1: float = 0.9,
        beta2: float = 0.98,
        epsilon: float = 1e-9,
        dropout_rate: float = 0.1,
        label_smoothing: float = 0.1,
        top_k: int = 5,
        temp: float = 1.0,
        training: bool = True,
        use_existing_weights: bool = False,
    ) -> None:
        self.vocab_path = vocab_path
        self.weights_path = weights_path

        # constants
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.key_query_dim = key_query_dim
        self.context_size = context_size
        self.nb_layers = nb_layers
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

        # else
        self.training = training
        self.use_existing_weights = use_existing_weights

        # ~ 1e5 steps for training
        # input shape : (batch_size, context_size)

    def get_model_size(self) -> int:
        embedding = unembedding = self.vocab_size * self.embedding_dim
        query = key = value = (
            (self.embedding_dim * self.key_query_dim) * self.nb_heads * self.nb_layers
        )
        output = (
            self.nb_heads * self.key_query_dim * self.embedding_dim * self.nb_layers
        )
        mlp_weight = (4 * self.embedding_dim) * self.embedding_dim * 2 * self.nb_layers
        mlp_bias = (4 * self.embedding_dim + self.embedding_dim) * self.nb_layers
        return (
            embedding
            + unembedding
            + key
            + query
            + value
            + output
            + mlp_weight
            + mlp_bias
        )

    def get_lr(self, step: int) -> float:
        return (self.embedding_dim**-0.5) * min(
            step**-0.5, step * (self.warmup_steps**-1.5)
        )
