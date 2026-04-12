'''Main module to encode and decode messages using the Stegosaurus algorithm.'''

import argparse
import functools
import json
import logging
import os
from logging.handlers import RotatingFileHandler
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get('MODEL', 'Qwen/Qwen3-0.6B')

# Load per-model configuration
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'model_config.json')

with open(_CONFIG_PATH) as _f:
    _ALL_CONFIGS = json.load(_f)

if MODEL_NAME not in _ALL_CONFIGS:
    raise ValueError(
        f'Model "{MODEL_NAME}" not found in model_config.json. '
        f'Supported models: {list(_ALL_CONFIGS)}'
    )

_MODEL_CONFIG = _ALL_CONFIGS[MODEL_NAME]

TOP_K = int(os.environ.get('TOP_K', 20))            # Number of top tokens to consider at each step
N_PARTITIONS = int(os.environ.get('N_PARTITIONS', 2))  # Must be a power of 2; bits per token = log2(N_PARTITIONS)
EOM = [1, 1, 1, 1, 1, 1, 1, 1]  # 0xFF - never valid UTF-8; marks end of message
PROMPT = _MODEL_CONFIG['default_prompt']

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

logger = logging.getLogger('stegosaurus')
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())

_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)

_file_handler = RotatingFileHandler(
    os.path.join(_LOG_DIR, 'stegosaurus.log'),
    mode='w',        # truncate on each startup
    maxBytes=1_000_000,  # 1 MB per file
    backupCount=3,
)
_file_handler.setFormatter(_formatter)

logger.addHandler(_stream_handler)
logger.addHandler(_file_handler)

# ---------------------------------------------------------------------------
# Shared model loading (lazy, module-level cache)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_device = None

def _load_model():
    '''Preps model and tokenizer, caching them in module-level variables 
    for reuse across calls.'''

    global _tokenizer, _model, _device

    if _model is None:

        # Set device to GPU if available, otherwise CPU
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Device: %s (cuda available: %s, torch CUDA: %s)', _device, torch.cuda.is_available(), torch.version.cuda)

        # Use bfloat16 on GPU, float32 on CPU
        _dtype = torch.bfloat16 if _device.type == 'cuda' else torch.float32
        logger.info('Using dtype %s', _dtype)

        # Load the tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=_MODEL_CONFIG['trust_remote_code'],
        )

        # Load the model
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=_dtype,
            trust_remote_code=_MODEL_CONFIG['trust_remote_code'],
        ).to(_device)

        # Put model in evaluation mode (disables dropout, etc.)
        _model.eval()

        # Clear the BPE-safety cache so stale results from a previous model
        # are not carried over if _load_model is ever called again.
        _is_bpe_safe.cache_clear()

        logger.info('Loaded model "%s"', MODEL_NAME)

    return _tokenizer, _model, _device


# ---------------------------------------------------------------------------
# Text <-> bits
# ---------------------------------------------------------------------------

def _encode_message(message: str) -> list[int]:
    '''Encode a UTF-8 string as a flat list of bits (MSB first).'''

    # Collector for encoded result
    bits = []

    # Loop over each byte in the UTF-8 encoding of the message
    for byte in message.encode('utf-8'):

        # Loop over each bit in the byte, starting with the most significant bit
        for i in range(7, -1, -1):

            # Extract the bit at position i and append it to the result list
            bits.append((byte >> i) & 1)

    return bits


def _decode_bits(bits: list[int]) -> str:
    '''Decode a flat list of bits (MSB first) back to a UTF-8 string.'''

    # Check for valid input length (must be a multiple of 8)
    if len(bits) % 8 != 0:
        raise ValueError(f'Bit list length must be a multiple of 8, got {len(bits)}')

    # Collector for decoded bytes
    byte_array = bytearray()

    # Loop over the bits in chunks of 8 to reconstruct each byte
    for i in range(0, len(bits), 8):

        # Set up an accumulator for the byte value
        byte = 0

        # Loop over the 8 bits for this byte, shifting and combining them
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit

        # Append the reconstructed byte to the byte array
        byte_array.append(byte)

    # Return the decoded string by interpreting the byte array as UTF-8
    return byte_array.decode('utf-8')


# ---------------------------------------------------------------------------
# Partition function
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _is_bpe_safe(prev_token_id: int | None, new_token_id: int) -> bool:
    '''
    Return True if appending new_token_id after prev_token_id produces a
    text that re-tokenizes back to exactly [prev_token_id, new_token_id].

    GPT-2 uses byte-level BPE, so adjacent tokens can merge when decoded
    to a string and re-encoded.  Filtering unsafe tokens from the candidate
    set ensures the cover-text token sequence is always recoverable.

    Results are memoized: the same token pair always gives the same answer,
    so repeated checks across generation steps hit the cache instead of
    calling the tokenizer again.
    '''

    if prev_token_id is None:
        return True

    pair_text = _tokenizer.decode([prev_token_id, new_token_id])

    # add_special_tokens=False prevents the tokenizer from prepending a BOS
    # token, which would make the re-encoded list longer than the pair.
    # Whether this matters is model-specific (see model_config.json).
    ats = _MODEL_CONFIG['bpe_check_add_special_tokens']
    return _tokenizer.encode(pair_text, add_special_tokens=ats) == [prev_token_id, new_token_id]


def _partition_top_k(probs, indices, top_k, n_partitions, prev_token_id, tokenizer):
    '''
    Partition the top_k tokens into n_partitions bins with approximately
    equal probability mass using greedy assignment.

    Only tokens that are BPE-safe (i.e. do not merge with prev_token_id
    when decoded to text and re-encoded) are included.  Special tokens
    (e.g. <|im_end|>) are also excluded so they never appear in cover text.

    Returns a list of n_partitions lists, each containing (token_id, prob) pairs.
    '''

    # Get the top_k probabilities and their corresponding token indices
    top_probs = probs[:top_k]
    top_indices = indices[:top_k]

    # Build a set of special token ids for fast membership testing
    special_ids = set(tokenizer.all_special_ids)

    # Create empty partitions and track their total mass
    partitions = [[] for _ in range(n_partitions)]
    partition_mass = [0.0] * n_partitions

    # Loop on the top_k tokens in descending order of probability
    for token_id, prob in zip(top_indices.tolist(), top_probs.tolist()):

        # Skip special tokens (e.g. <|im_end|>, <|endoftext|>) so they never
        # appear in cover text and don't interfere with the partition assignment
        if token_id in special_ids:
            continue

        # Skip tokens that would merge with the previous token under BPE re-encoding
        if not _is_bpe_safe(prev_token_id, token_id):
            continue

        # Assign to the partition with the lowest current mass
        target = partition_mass.index(min(partition_mass))
        partitions[target].append((token_id, prob))
        partition_mass[target] += prob

    return partitions


# ---------------------------------------------------------------------------
# Next-token helpers
# ---------------------------------------------------------------------------

def _get_probs(input_ids, model, device, past_key_values=None):
    '''Run a forward pass and return (sorted_probs, sorted_indices, past_key_values)
    for the next token.

    Pass past_key_values from the previous call to avoid recomputing attention
    over the full context at every step. When provided, input_ids should contain
    only the single new token rather than the entire sequence.
    '''

    # Run single forward pass with gradient accumulation turned off
    with torch.no_grad():
        outputs = model(input_ids.to(device), past_key_values=past_key_values, use_cache=True)

    # Get the logits from the last token, convert to double precision
    logits = outputs.logits[0, -1, :].double()

    # Get sorted logits and their corresponding token indices
    logits, indices = logits.sort(descending=True)

    # Convert to probabilities
    probs = F.softmax(logits, dim=0)

    return probs, indices, outputs.past_key_values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(
        message: str,
        prompt: str = PROMPT,
        top_k: int = TOP_K,
        n_partitions: int = N_PARTITIONS
) -> str:
    '''
    Encode a secret message into cover text.

    Each bit (or group of bits) of the UTF-8 binary representation of
    `message` selects a partition of the top-k next-token distribution.
    The highest-probability token in the chosen partition is appended to
    the cover text, embedding the secret bit(s) invisibly.

    A fixed 8-bit EOM marker (0xFF) is appended after the message bits.
    0xFF is never a valid UTF-8 byte, so it is unambiguous as a sentinel.

    After all message bits have been encoded, generation continues greedily
    (using the highest-probability non-special token at each step) until the
    cover text ends with a sentence-terminal character (`.`, `!`, or `?`),
    capped at 200 extra tokens.  This prevents the cover text from cutting
    off mid-sentence without affecting the hidden payload.
    '''

    # Load model and tokenizer (cached across calls)
    tokenizer, model, device = _load_model()

    # Encode the message as a list of bits and append the EOM marker
    message_bits = _encode_message(message)
    bits = message_bits + EOM

    # Calculate how many bits we can encode per token based on the number of partitions
    bits_per_token = n_partitions.bit_length() - 1  # log2(n_partitions)

    logger.info('Encoding message (%d chars) with top_k=%d, n_partitions=%d', len(message), top_k, n_partitions)

    # Tokenize the prompt and prime the KV cache; get probs for the first token
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    probs, indices, past_key_values = _get_probs(input_ids, model, device)

    cover_ids = []
    prev_token_id = None  # track the last generated token for BPE safety checks

    # Iterate over the bits in chunks of bits_per_token, selecting tokens accordingly
    bit_idx = 0

    while bit_idx < len(bits):

        partitions = _partition_top_k(
            probs, indices, top_k, n_partitions,
            prev_token_id, tokenizer
        )

        # Read the next bits_per_token bits as an integer partition index
        chunk = bits[bit_idx:bit_idx + bits_per_token]

        # Pad the last chunk with zeros if it's shorter than bits_per_token
        if len(chunk) < bits_per_token:
            chunk += [0] * (bits_per_token - len(chunk))  # pad last chunk

        # Convert the chunk of bits to an integer partition index
        partition_idx = 0
        for b in chunk:
            partition_idx = (partition_idx << 1) | b

        # Pick the highest-probability token in the selected partition
        chosen_id = partitions[partition_idx][0][0]
        cover_ids.append(chosen_id)
        prev_token_id = chosen_id

        # Move to the next chunk of bits
        bit_idx += bits_per_token

        # Extend the KV cache with the chosen token for the next iteration
        if bit_idx < len(bits):
            next_input = torch.tensor([[chosen_id]], dtype=torch.long)
            probs, indices, past_key_values = _get_probs(next_input, model, device, past_key_values)

    # Continue generating greedily until we reach the end of a sentence, so the
    # cover text doesn't cut off mid-sentence. Cap at 200 tokens to avoid runaway.
    special_ids = set(tokenizer.all_special_ids)
    for _ in range(200):

        # Advance the KV cache with the last encoded token
        next_input = torch.tensor([[chosen_id]], dtype=torch.long)
        probs, indices, past_key_values = _get_probs(next_input, model, device, past_key_values)

        # Pick the highest-probability non-special token
        for token_id, _ in zip(indices.tolist(), probs.tolist()):
            if token_id not in special_ids:
                chosen_id = token_id
                break

        cover_ids.append(chosen_id)

        # Stop at the end of a sentence
        if tokenizer.decode(cover_ids).rstrip().endswith(('.', '!', '?')):
            break

    cover_text = tokenizer.decode(cover_ids)
    logger.info('Encoding complete: %d tokens generated', len(cover_ids))
    return cover_text


def decode(
        cover_text: str,
        prompt: str = PROMPT,
        top_k: int = TOP_K,
        n_partitions: int = N_PARTITIONS
) -> str:
    '''
    Decode a secret message from cover text.

    Re-runs the same greedy partition process over the cover text tokens
    to determine which partition each token belongs to, recovering the
    hidden bits and reconstructing the original message.

    Reads tokens until the recovered bit stream ends with the 8-bit EOM
    marker (0xFF). No out-of-band metadata is required.
    '''

    # Load model and tokenizer (cached across calls)
    tokenizer, model, device = _load_model()

    # Calculate how many bits we can encode per token based on the number of partitions
    bits_per_token = n_partitions.bit_length() - 1

    logger.info('Decoding cover text (%d chars) with top_k=%d, n_partitions=%d', len(cover_text), top_k, n_partitions)

    # Tokenize the prompt and cover text
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')

    # The prompt is tokenized with add_special_tokens=True (default) so the
    # model's BOS token is prepended once. The cover text must not get another
    # BOS, so we use the per-model setting from model_config.json.
    cover_ids = tokenizer.encode(
        cover_text,
        add_special_tokens=_MODEL_CONFIG['cover_add_special_tokens'],
        return_tensors='pt',
    )[0].tolist()

    # Prime the KV cache with the prompt; get probs for the first cover token
    recovered_bits = []
    prev_token_id = None   # track the last cover token for BPE safety checks
    probs, indices, past_key_values = _get_probs(prompt_ids, model, device)

    # Iterate over each token in the cover text, determining which partition it belongs to
    for token_id in cover_ids:

        partitions = _partition_top_k(
            probs, indices, top_k, n_partitions,
            prev_token_id, tokenizer
        )

        # Find which partition this token belongs to
        partition_idx = None

        # Loop over partitions to find the one containing the current token_id
        for p_idx, partition in enumerate(partitions):
            if any(tid == token_id for tid, _ in partition):
                partition_idx = p_idx
                break

        # If the token_id was not found in any partition, it means the cover text is invalid or was generated with different settings
        if partition_idx is None:
            raise ValueError(
                f'Token "{tokenizer.decode(token_id)}" (id={token_id}) '
                f'was not found in the top-{top_k} at this position. '
                'The cover text may have been altered or used different settings.'
            )

        # Convert partition index back to bits
        chunk = [(partition_idx >> (bits_per_token - 1 - i)) & 1
                 for i in range(bits_per_token)]

        # Append the recovered bits from this token to the result list
        recovered_bits.extend(chunk)
        prev_token_id = token_id

        # Stop as soon as the recovered bit stream ends with the EOM marker.
        # Only check at byte boundaries (EOM is 8 bits; message is always
        # byte-aligned UTF-8, so a false positive is impossible).
        if len(recovered_bits) >= 8 and len(recovered_bits) % 8 == 0 and recovered_bits[-8:] == EOM:
            break

        # Extend the KV cache with this token for the next step
        next_input = torch.tensor([[token_id]], dtype=torch.long)
        probs, indices, past_key_values = _get_probs(next_input, model, device, past_key_values)

    # Strip the EOM marker and decode the remaining bits to a string
    message_bits = recovered_bits[:-8]
    message = _decode_bits(message_bits)
    logger.info('Decoding complete: recovered %d chars', len(message))
    return message


if __name__ == '__main__':

    import sys

    # Take message or cover text as command line arguments
    parser = argparse.ArgumentParser(description='Command line stegosaurus entry-point')
    parser.add_argument('-e', '--encode', type=str, help='Message to be encoded')
    parser.add_argument('-d', '--decode', action='store_true',
                        help='Decode cover text read from stdin')
    args = parser.parse_args()

    if args.encode:
        print(encode(args.encode))

    if args.decode:
        cover_text = sys.stdin.read().rstrip('\n')
        print(decode(cover_text))
