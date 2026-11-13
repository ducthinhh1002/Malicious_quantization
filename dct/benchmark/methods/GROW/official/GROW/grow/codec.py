"""Bit-level message codec used to (de)serialize the watermark payload."""

from typing import List


def message_to_bits(message: str) -> List[int]:
    """Encode a UTF-8 message into a flat list of bits (MSB first per byte)."""
    bits: List[int] = []
    for byte in message.encode("utf-8"):
        bits.extend(int(b) for b in f"{byte:08b}")
    return bits


def bits_to_message(bits: List[int]) -> str:
    """Decode a flat list of bits back into a UTF-8 string.

    Returns a sentinel error string on malformed input so that callers in
    the attack-evaluation loop can robustly score success.
    """
    if not bits or len(bits) % 8 != 0:
        return "[ERROR: Invalid bit length]"
    try:
        byte_array = bytearray(
            int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8)
        )
        return byte_array.decode("utf-8", errors="ignore")
    except Exception:
        return "[ERROR: Decode failed]"
