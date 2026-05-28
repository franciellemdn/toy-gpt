# Default configuration parameters for the Toy GPT model and training loop.
# These can be customized or overridden via command line flags when training.

# Model Hyperparameters
BLOCK_SIZE = 128    # Context length: how many tokens the model sees at once
N_EMBED    = 192    # Embedding dimension / model width
N_HEADS    = 6      # Number of attention heads (N_EMBED must be divisible by N_HEADS)
N_LAYERS   = 4      # Number of transformer blocks stacked
DROPOUT    = 0.1

# Training Hyperparameters
BATCH_SIZE    = 32     # Batch size: how many independent sequences per step
LEARNING_RATE = 3e-4
MAX_ITERS     = 3000   # Total training iterations
EVAL_INTERVAL = 250    # Interval to estimate loss and save checkpoints
EVAL_ITERS    = 50     # Number of batches to average for loss estimation
SEED          = 1337

# Learning Rate Decay Scheduler
DECAY_LR     = True
WARMUP_ITERS = 100    # Steps to linearly warm up learning rate
MIN_LR       = 3e-5   # Minimum learning rate (usually LEARNING_RATE / 10)
WEIGHT_DECAY = 1e-1   # Weight decay coefficient
