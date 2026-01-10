import uuid
from datetime import datetime


def uuid7(user_uuid: str = None, user_datetime=None) -> uuid.UUID:
    """
    Convert a UUIDv4 into a UUIDv7-like UUID using timestamp from the given datetime.
    This is required for opik.
    """
    # Validate input UUID is version 4
    if user_uuid and user_uuid.version != 4:
        raise ValueError('Input UUID must be version 4')

    uuidv4 = user_uuid or uuid.UUID(str(uuid.uuid4()))

    user_datetime = user_datetime or datetime.now()
    # Convert timestamp to milliseconds
    unix_ts_ms = int(user_datetime.timestamp() * 1000)

    # Build UUID bytes
    uuid_bytes = bytearray(16)

    # First 6 bytes: 48-bit Unix timestamp in milliseconds
    uuid_bytes[0:6] = unix_ts_ms.to_bytes(6, byteorder='big')

    # Byte 6: version (7 in high 4 bits) + low 4 bits from uuid4
    uuid_bytes[6] = 0x70 | (uuidv4.bytes[6] & 0x0F)

    # Byte 7: copied from uuid4 randomness
    uuid_bytes[7] = uuidv4.bytes[7]

    # Byte 8: variant (10xxxxxx)
    uuid_bytes[8] = 0x80 | (uuidv4.bytes[8] & 0x3F)

    # Bytes 9–15: copied from uuid4 randomness
    uuid_bytes[9:] = uuidv4.bytes[9:]

    return uuid.UUID(bytes=bytes(uuid_bytes))
