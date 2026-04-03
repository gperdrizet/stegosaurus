'''Main module to encode and decode messages using the Stegosaurus algorithm.'''

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = 'openai-community/gpt2-large'
TOP_K = 50          # Number of top tokens to consider at each step
N_PARTITIONS = 2    # Must be a power of 2; bits per token = log2(N_PARTITIONS)
PROMPT = '<|endoftext|>A turtle and a bird were walking in the forest one day. The turtle said, "'
HEADER_BITS = 32    # Fixed-width header encoding the message length in bits

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

        # Load model and tokenizer from Hugging Face
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(_device)

        # Put model in evaluation mode (disables dropout, etc.)
        _model.eval()

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

def _partition_top_k(probs, indices, top_k, n_partitions):
    '''
    Partition the top_k tokens into n_partitions bins with approximately
    equal probability mass using greedy assignment.

    Returns a list of n_partitions lists, each containing (token_id, prob) pairs.
    '''

    # Get the top_k probabilities and their corresponding token indices
    top_probs = probs[:top_k]
    top_indices = indices[:top_k]

    # Create empty partitions and track their total mass
    partitions = [[] for _ in range(n_partitions)]
    partition_mass = [0.0] * n_partitions

    # Loop on the top_k tokens in descending order of probability
    for token_id, prob in zip(top_indices.tolist(), top_probs.tolist()):

        # Assign to the partition with the lowest current mass
        target = partition_mass.index(min(partition_mass))
        partitions[target].append((token_id, prob))
        partition_mass[target] += prob

    return partitions


# ---------------------------------------------------------------------------
# Next-token helpers
# ---------------------------------------------------------------------------

def _get_probs(input_ids, model, device):
    '''Run a forward pass and return (sorted_probs, sorted_indices) for
    the next token.'''

    # Run single forward pass with gradient accumulation turned off
    with torch.no_grad():
        outputs = model(input_ids.to(device))

    # Get the logits from the last token, convert to double precision
    logits = outputs.logits[0, -1, :].double()

    # Get sorted logits and their corresponding token indices
    logits, indices = logits.sort(descending=True)

    # Convert to probabilities
    probs = F.softmax(logits, dim=0)

    return probs, indices


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
    '''

    # Load model and tokenizer (cached across calls)
    tokenizer, model, device = _load_model()

    # Encode the message as a list of bits
    message_bits = _encode_message(message)

    # Prepend a fixed-width header encoding the number of message bits
    n_message_bits = len(message_bits)
    header_bits = [(n_message_bits >> (HEADER_BITS - 1 - i)) & 1
                   for i in range(HEADER_BITS)]
    bits = header_bits + message_bits

    # Calculate how many bits we can encode per token based on the number of partitions
    bits_per_token = n_partitions.bit_length() - 1  # log2(n_partitions)

    # Tokenize the prompt and initialize the generated token sequence
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_ids = input_ids.clone()

    # Iterate over the bits in chunks of bits_per_token, selecting tokens accordingly
    bit_idx = 0

    while bit_idx < len(bits):

        # Get the next-token probabilities and their sorted indices
        probs, indices = _get_probs(generated_ids, model, device)
        partitions = _partition_top_k(probs, indices, top_k, n_partitions)

        # Read the next bits_per_token bits as an integer partition index
        chunk = bits[bit_idx:bit_idx + bits_per_token]
    
        # Pad the last chunk with zeros if it's shorter than bits_per_token
        if len(chunk) < bits_per_token:
            chunk += [0] * (bits_per_token - len(chunk))  # pad last chunk

        # Convert the chunk of bits to an integer partition index
        partition_idx = int(''.join(str(b) for b in chunk), 2)

        # Pick the highest-probability token in the selected partition
        chosen_id = partitions[partition_idx][0][0]
        chosen_tensor = torch.tensor([[chosen_id]], dtype=torch.long)
        generated_ids = torch.cat([generated_ids, chosen_tensor], dim=1)

        # Move to the next chunk of bits
        bit_idx += bits_per_token

    # Decode everything after the prompt
    prompt_len = input_ids.shape[1]
    cover_ids = generated_ids[0, prompt_len:].tolist()

    return tokenizer.decode(cover_ids)


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

    The first HEADER_BITS tokens encode the message length, so no
    out-of-band metadata is required.
    '''

    # Load model and tokenizer (cached across calls)
    tokenizer, model, device = _load_model()

    # Calculate how many bits we can encode per token based on the number of partitions
    bits_per_token = n_partitions.bit_length() - 1

    # Tokenize the prompt and cover text
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    cover_ids = tokenizer.encode(cover_text, return_tensors='pt')[0].tolist()

    # Collector for recovered bits and context for next-token prediction
    recovered_bits = []
    context = prompt_ids.clone()
    n_message_bits = None  # learned from header

    # Iterate over each token in the cover text, determining which partition it belongs to
    for token_id in cover_ids:

        # Stop once we have the header + all message bits
        total_bits_needed = HEADER_BITS + (n_message_bits if n_message_bits is not None else 0)
        
        if len(recovered_bits) >= total_bits_needed and n_message_bits is not None:
            break
        
        # Get the next-token probabilities and their sorted indices for the current context
        probs, indices = _get_probs(context, model, device)
        partitions = _partition_top_k(probs, indices, top_k, n_partitions)

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

        # Once we have enough bits for the header, decode the message length
        if n_message_bits is None and len(recovered_bits) >= HEADER_BITS:
            header = recovered_bits[:HEADER_BITS]
            n_message_bits = int(''.join(str(b) for b in header), 2)

        # Extend context with this token for the next step
        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        context = torch.cat([context, token_tensor], dim=1)

    # Extract the message bits from the recovered bits, skipping the header
    message_bits = recovered_bits[HEADER_BITS:HEADER_BITS + n_message_bits]

    # Decode the message bits back to a string and return it
    return _decode_bits(message_bits)


if __name__ == '__main__':

    # Take message or cover text as command line arguments
    parser = argparse.ArgumentParser(description='Command line stegosaurus entry-point')
    parser.add_argument('-e', '--encode', type=str, help='Message to be encoded')
    parser.add_argument('-d', '--decode', type=str, help='Cover text to be decoded')
    args = parser.parse_args()

    if args.encode:
        print(encode(args.encode))

    if args.decode:
        print(decode(args.decode))
