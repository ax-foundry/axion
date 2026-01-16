import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set, TextIO, Tuple, Union

from axion._core.environment import settings
from axion._core.logging import get_logger

logger = get_logger(__name__)


LOG_LEVEL = logging._nameToLevel[settings.log_level]


#################
#################

# Customized extension of json_repair package
# https://github.com/mangiucugna/json_repair/blob/main/src/json_repair/json_parser.py

#################
#################


class StringFileWrapper:
    """A wrapper for file-like objects that makes them array-like."""

    def __init__(self, fd: TextIO, chunk_length: int = 1024):
        self.fd = fd
        self.chunk_length = chunk_length if chunk_length > 0 else 1024
        self.buffer = ''
        self.position = 0
        self.eof = False

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            result = ''
            for i in range(start, stop, step):
                result += self[i]
            return result

        if index < 0:
            # Negative indexing requires knowing the total length
            # which we can't efficiently do with a file stream
            index = self.position + index
            if index < 0:
                raise IndexError('Negative index out of range')

        if index < self.position:
            # Requesting an item that was already read
            raise IndexError('Cannot access previously read items')

        while index >= self.position + len(self.buffer) and not self.eof:
            chunk = self.fd.read(self.chunk_length)
            if not chunk:
                self.eof = True
                break
            self.buffer += chunk

        if index >= self.position + len(self.buffer):
            raise IndexError('Index out of range')

        char = self.buffer[index - self.position]

        # If we're accessing the last character in our buffer, shift the window
        if index - self.position == len(self.buffer) - 1:
            self.position = index + 1
            self.buffer = ''

        return char

    def __len__(self):
        if self.eof:
            return self.position + len(self.buffer)
        # This is a rough approximation as we can't know the file length without reading it all
        return float('inf')


class JsonContext:
    """Context manager for JSON parsing."""

    class ContextValues:
        """Enum-like class for context values."""

        OBJECT_KEY = 'object_key'
        OBJECT_VALUE = 'object_value'
        ARRAY = 'array'

    def __init__(self):
        self.context: Set[str] = set()
        self.current = None

    def set(self, value: str) -> None:
        """Set the current context value."""
        self.context.add(value)
        self.current = value

    def reset(self) -> None:
        """Reset the current context value."""
        self.current = None

    @property
    def empty(self) -> bool:
        """Check if the context is empty."""
        return len(self.context) == 0


class ObjectComparer:
    """Utility for comparing JSON objects."""

    @staticmethod
    def is_same_object(obj1: Any, obj2: Any) -> bool:
        """Compare two objects for structural similarity."""
        if type(obj1) is not type(obj2):
            return False

        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            # Check just the first key for efficiency
            if obj1 and list(obj1.keys())[0] in obj2:
                return True
            return False

        if isinstance(obj1, list):
            # For lists, we check length and first element if present
            if len(obj1) != len(obj2):
                return False
            if obj1 and obj2:
                return ObjectComparer.is_same_object(obj1[0], obj2[0])
            return True

        # For primitives, direct comparison
        return obj1 == obj2


# JSONReturnType for type hints
JSONReturnType = Union[Dict[str, Any], List[Any], str, float, int, bool, None]


class JSONParser:
    """
    A robust JSON parser combining the best features of two implementations:
    1. Detailed, character-by-character parsing for high tolerance to malformed input
    2. Repair capabilities for common JSON errors
    3. Extraction of JSON from markdown, comments, etc.

    Features:
    - High tolerance for invalid JSON with repair capabilities
    - Detailed context tracking for accurate parsing
    - Support for streaming input from files
    - Extensive logging for debugging
    - Multiple entry points for different use cases
    """

    # Constants
    STRING_DELIMITERS = ['"', "'", """, """]
    NUMBER_CHARS = set('0123456789-.eE/,')
    ESCAPE_SEQUENCES = {
        '\\\\': '\\',  # Backslash
        '\\"': '"',  # Double quote
        "\\'": "'",  # Single quote
        '\\n': '\n',  # Newline
        '\\r': '\r',  # Carriage return
        '\\t': '\t',  # Tab
        '\\b': '\b',  # Backspace
        '\\f': '\f',  # Form feed
    }
    JSON_MARKERS = {
        'markdown': ['```json', '```', '~~~json', '~~~', '`json', '`'],
        'xml': ['<json>', '</json>'],
        'comments': ['/*', '*/'],
    }

    def __init__(
        self,
        json_str: Optional[Union[str, StringFileWrapper]] = None,
        json_fd: Optional[TextIO] = None,
        json_fd_chunk_length: int = 1024,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the JSON parser.

        Args:
            json_str: String to parse, or None to use json_fd
            json_fd: File descriptor with JSON content, or None to use json_str
            json_fd_chunk_length: Chunk length when reading from file
            strict_mode: If True, throw exceptions for malformed JSON rather than trying to repair
        """
        # The string to parse
        self.json_str: Union[str, StringFileWrapper] = json_str or ''

        # Alternatively, use file descriptor
        if json_fd:
            self.json_str = StringFileWrapper(json_fd, json_fd_chunk_length)

        # Index is our iterator for character-by-character parsing
        self.index: int = 0

        # Context tracking for special parsing cases
        self.context = JsonContext()

        # Strictness setting
        self.strict_mode = strict_mode

        # Statistics for repair operations
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset the repair statistics."""
        self.stats = {
            'quotes_fixed': 0,
            'braces_balanced': 0,
            'commas_fixed': 0,
            'control_chars_removed': 0,
            'unicode_escapes_fixed': 0,
            'single_quotes_converted': 0,
        }

    # Main entry points for parsing

    def parse(
        self,
    ) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
        """
        Parse the JSON input, handling multiple JSON objects if present.

        Returns:
            Parsed JSON object(s), and optionally logging information if logging is enabled
        """
        json = self.parse_json()

        # Check if there are multiple JSON objects
        if self.index < len(self.json_str):
            logger.log(
                LOG_LEVEL,
                "The parser returned early, checking if there's more json elements",
            )
            json = [json]
            last_index = self.index

            while self.index < len(self.json_str):
                j = self.parse_json()
                if j != '':
                    if ObjectComparer.is_same_object(json[-1], j):
                        # Replace the last entry if it appears to be an update
                        json.pop()
                    json.append(j)

                if self.index == last_index:
                    self.index += 1
                last_index = self.index

            # If only one object was found, don't return an array
            if len(json) == 1:
                logger.log(
                    LOG_LEVEL,
                    'There were no more elements, returning the element without the array',
                )
                json = json[0]

        return json

    def parse_text(self, text: str) -> Optional[Union[Dict, List]]:
        """
        Parse JSON from a text string, with automatic repair of common issues.

        Args:
            text: Input text containing JSON

        Returns:
            Parsed JSON object or None if parsing fails
        """
        if not text:
            return None

        try:
            # First try standard JSON parsing
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.debug(f'Initial JSON parsing failed: {e}')

            if self.strict_mode:
                raise e

            # Extract JSON if embedded in other content
            extracted_json = self.extract_json(text)

            # Try to repair the JSON and parse again
            repaired_json = self.repair_json(extracted_json)

            try:
                result = json.loads(repaired_json)
                logger.info(
                    f'Successfully repaired and parsed JSON. Stats: {self.stats}'
                )
                return result
            except json.JSONDecodeError as e2:
                logger.warning(f'Failed to parse even after repair: {e2}')
                return None

    @staticmethod
    def parse_prompt_template_json(text: str) -> str:
        """
        Parse JSON used in prompt templates with specific formatting requirements.

        Args:
            text: Input text containing template JSON

        Returns:
            Cleaned and formatted JSON string
        """
        # Remove outer single quotes if present
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Replace double braces with single braces
        text = text.replace('{{', '{')

        # Split at the ### separator to remove the question part
        parts = text.split('\n\n###')
        json_part = parts[0]

        # Fix the end - replace the extra brace
        fixed_json = json_part.replace('}}', '}')

        return fixed_json

    # JSON extraction methods

    def extract_json(self, text: str) -> str:
        """
        Extract the first JSON structure from a text blob.

        This function identifies and extracts the first valid JSON structure,
        looking for markup indicators first, then balancing brackets.

        Args:
            text: Input text that may contain JSON

        Returns:
            Extracted JSON structure or the original text if no JSON is found
        """
        if not text:
            return ''

        # Check for JSON in markdown blocks, XML tags, etc.
        extracted = self._extract_from_markup(text)
        if extracted:
            return extracted

        # Find the first opening bracket or brace
        open_chars = {'{': text.find('{'), '[': text.find('[')}
        open_chars = {k: v for k, v in open_chars.items() if v != -1}

        if not open_chars:
            # No JSON delimiters found
            return text

        # Start from the earliest opening character
        open_char = min(open_chars, key=open_chars.get)
        start_idx = open_chars[open_char]
        close_char = '}' if open_char == '{' else ']'

        # Extract by balancing brackets/braces
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Track string boundaries to avoid counting brackets in strings
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1

                    # Return complete JSON when all pairs are matched
                    if depth == 0:
                        return text[start_idx : i + 1]

        # Return original text if no valid JSON structure found
        return text

    def _extract_from_markup(self, text: str) -> str:
        """Extract JSON from common markup formats like Markdown code blocks or XML tags."""
        for format_type, markers in self.JSON_MARKERS.items():
            if format_type in ('markdown', 'comments'):
                # For paired markers like ```json and ```
                for i in range(0, len(markers), 2):
                    start_marker = markers[i]
                    end_marker = markers[i + 1] if i + 1 < len(markers) else None

                    start_idx = text.find(start_marker)
                    if start_idx != -1:
                        start_content = start_idx + len(start_marker)
                        if end_marker:
                            end_idx = text.find(end_marker, start_content)
                            if end_idx != -1:
                                return text[start_content:end_idx].strip()
            elif format_type == 'xml':
                # For XML-style tags
                start_marker, end_marker = markers
                start_idx = text.find(start_marker)
                if start_idx != -1:
                    start_content = start_idx + len(start_marker)
                    end_idx = text.find(end_marker, start_content)
                    if end_idx != -1:
                        return text[start_content:end_idx].strip()

        return ''

    # Character-by-character parsing methods

    def parse_json(self) -> JSONReturnType:
        """
        Parse the next JSON value from the current position.

        Returns:
            Parsed JSON value or empty string if nothing found
        """
        while True:
            char = self.get_char_at()

            # End of string
            if char is False:
                return ''

            # Object starts with '{'
            elif char == '{':
                self.index += 1
                return self.parse_object()

            # Array starts with '['
            elif char == '[':
                self.index += 1
                return self.parse_array()

            # Handle edge case of missing value at end of object
            elif (
                self.context.current == JsonContext.ContextValues.OBJECT_VALUE
                and char == '}'
            ):
                logger.log(
                    LOG_LEVEL,
                    'At the end of an object we found a key with missing value, skipping',
                )
                return ''

            # String starts with a quote or alpha character if in context
            elif not self.context.empty and (
                char in self.STRING_DELIMITERS or char.isalpha()
            ):
                return self.parse_string()

            # Number starts with digit, minus, or period
            elif not self.context.empty and (
                char.isdigit() or char == '-' or char == '.'
            ):
                return self.parse_number()

            # Comments start with # or /
            elif char in ['#', '/']:
                return self.parse_comment()

            # Skip other characters and continue
            else:
                self.index += 1

    def parse_object(self) -> Dict[str, JSONReturnType]:
        """
        Parse a JSON object (dictionary).

        Returns:
            Parsed dictionary object
        """
        obj = {}

        # Parse until closing brace or end of input
        while (self.get_char_at() or '}') != '}':
            # Skip whitespace
            self.skip_whitespaces_at()

            # Handle case of colon before key
            if (self.get_char_at() or '') == ':':
                logger.log(
                    LOG_LEVEL,
                    'While parsing an object we found a : before a key, ignoring',
                )
                self.index += 1

            # Set context for parsing the key
            self.context.set(JsonContext.ContextValues.OBJECT_KEY)

            # Save index for potential rollback
            rollback_index = self.index

            # Parse the key
            key = ''
            while self.get_char_at():
                rollback_index = self.index
                key = str(self.parse_string())
                if key == '':
                    self.skip_whitespaces_at()
                if key != '' or (key == '' and self.get_char_at() in [':', '}']):
                    break

            # Handle duplicate keys in arrays
            if JsonContext.ContextValues.ARRAY in self.context.context and key in obj:
                logger.log(
                    LOG_LEVEL,
                    'While parsing an object we found a duplicate key, closing the object here and rolling back the index',
                )
                self.index = rollback_index - 1
                # Add opening brace to make this work
                self.json_str = (
                    self.json_str[: self.index + 1]
                    + '{'
                    + self.json_str[self.index + 1 :]
                )
                break

            # Skip whitespace
            self.skip_whitespaces_at()

            # Handle end of object
            if (self.get_char_at() or '}') == '}':
                continue

            self.skip_whitespaces_at()

            # Handle missing colon after key
            if (self.get_char_at() or '') != ':':
                logger.log(
                    LOG_LEVEL, 'While parsing an object we missed a : after a key'
                )

            # Move past colon
            self.index += 1
            self.context.reset()
            self.context.set(JsonContext.ContextValues.OBJECT_VALUE)

            # Parse the value
            value = self.parse_json()

            # Reset context
            self.context.reset()
            obj[key] = value

            # Skip trailing comma or quotes
            if (self.get_char_at() or '') in [',', "'", '"']:
                self.index += 1

            # Skip trailing whitespace
            self.skip_whitespaces_at()

        # Move past closing brace
        self.index += 1
        return obj

    def parse_array(self) -> List[JSONReturnType]:
        """
        Parse a JSON array (list).

        Returns:
            Parsed list object
        """
        arr = []
        self.context.set(JsonContext.ContextValues.ARRAY)

        # Parse until closing bracket or end of input
        char = self.get_char_at()
        while char and char not in [']', '}']:
            self.skip_whitespaces_at()
            value = self.parse_json()

            # Handle various special cases
            if value == '':
                self.index += 1
            elif value == '...' and self.get_char_at(-1) == '.':
                logger.log(
                    LOG_LEVEL,
                    "While parsing an array, found a stray '...'; ignoring it",
                )
            else:
                arr.append(value)

            # Skip whitespace and commas
            char = self.get_char_at()
            while char and (char.isspace() or char == ','):
                self.index += 1
                char = self.get_char_at()

        # Handle missing closing bracket
        if char and char != ']':
            logger.log(
                LOG_LEVEL, 'While parsing an array we missed the closing ], ignoring it'
            )

        # Move past closing bracket
        self.index += 1

        self.context.reset()
        return arr

    def parse_string(self) -> Union[str, bool, None]:
        """
        Parse a JSON string, handling various edge cases.

        Returns:
            Parsed string, boolean, or null value
        """
        # Track special parsing cases
        missing_quotes = False
        doubled_quotes = False
        lstring_delimiter = rstring_delimiter = '"'

        # Handle comments
        char = self.get_char_at()
        if char in ['#', '/']:
            return self.parse_comment()

        # Skip non-alphanumeric characters that aren't quotes
        while char and char not in self.STRING_DELIMITERS and not char.isalnum():
            self.index += 1
            char = self.get_char_at()

        if not char:
            # Empty string
            return ''

        # Determine the string delimiters
        if char == "'":
            lstring_delimiter = rstring_delimiter = "'"
        elif char == '"':
            lstring_delimiter = rstring_delimiter = '"'
        elif char.isalnum():
            # Check for boolean/null values
            if (
                char.lower() in ['t', 'f', 'n']
                and self.context.current != JsonContext.ContextValues.OBJECT_KEY
            ):
                value = self.parse_boolean_or_null()
                if value != '':
                    return value

            logger.log(
                LOG_LEVEL,
                'While parsing a string, we found a literal instead of a quote',
            )
            missing_quotes = True

        # Move past opening quote if present
        if not missing_quotes:
            self.index += 1

        # Handle doubled quotes
        if self.get_char_at() in self.STRING_DELIMITERS:
            # Check if next character is same delimiter
            if self.get_char_at() == lstring_delimiter:
                # Handle empty key
                if (
                    self.context.current == JsonContext.ContextValues.OBJECT_KEY
                    and self.get_char_at(1) == ':'
                ):
                    self.index += 1
                    return ''

                # Handle multiple consecutive quotes
                if self.get_char_at(1) == lstring_delimiter:
                    logger.log(
                        LOG_LEVEL,
                        'While parsing a string, we found a doubled quote and then a quote again, ignoring it',
                    )
                    return ''

                # Find next delimiter
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)

                # Check for doubled closing quotes
                if next_c and (self.get_char_at(i + 1) or '') == rstring_delimiter:
                    logger.log(
                        LOG_LEVEL,
                        'While parsing a string, we found a valid starting doubled quote',
                    )
                    doubled_quotes = True
                    self.index += 1
                else:
                    # Check if this is an empty string
                    i = self.skip_whitespaces_at(idx=1, move_main_index=False)
                    next_c = self.get_char_at(i)
                    if next_c in self.STRING_DELIMITERS + ['{', '[']:
                        logger.log(
                            LOG_LEVEL,
                            'While parsing a string, we found a doubled quote but also another quote afterwards, ignoring it',
                        )
                        self.index += 1
                        return ''
                    elif next_c not in [',', ']', '}']:
                        logger.log(
                            LOG_LEVEL,
                            'While parsing a string, we found a doubled quote but it was a mistake, removing one quote',
                        )
                        self.index += 1
            else:
                # Check if this delimiter appears again
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if not next_c:
                    logger.log(
                        LOG_LEVEL,
                        'While parsing a string, we found a quote but it was a mistake, ignoring it',
                    )
                    return ''

        # Parse the string content
        string_acc = ''
        char = self.get_char_at()
        unmatched_delimiter = False

        # Complex string parsing loop with many edge cases
        while char and char != rstring_delimiter:
            # Handle missing quotes in object keys
            if (
                missing_quotes
                and self.context.current == JsonContext.ContextValues.OBJECT_KEY
                and (char == ':' or char.isspace())
            ):
                logger.log(
                    LOG_LEVEL,
                    'While parsing a string missing the left delimiter in object key context, we found a :, stopping here',
                )
                break

            # Handle missing quotes in object values
            if (
                self.context.current == JsonContext.ContextValues.OBJECT_VALUE
                and char
                in [
                    ',',
                    '}',
                ]
            ):
                rstring_delimiter_missing = True

                # Various checks to determine if the delimiter is actually missing
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)

                if next_c:
                    i += 1
                    i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                    next_c = self.get_char_at(i)

                    if not next_c or next_c in [',', '}']:
                        rstring_delimiter_missing = False
                    else:
                        # Check for a new string delimiter
                        i = self.skip_to_character(character=lstring_delimiter, idx=i)
                        if doubled_quotes:
                            i = self.skip_to_character(
                                character=lstring_delimiter, idx=i
                            )
                        next_c = self.get_char_at(i)

                        if not next_c:
                            rstring_delimiter_missing = False
                        else:
                            # Check for key-value pattern
                            i = self.skip_whitespaces_at(
                                idx=i + 1, move_main_index=False
                            )
                            next_c = self.get_char_at(i)
                            if next_c and next_c != ':':
                                rstring_delimiter_missing = False
                else:
                    # Check for systemic missing delimiters
                    i = self.skip_to_character(character=':', idx=1)
                    next_c = self.get_char_at(i)

                    if next_c:
                        # Systemic issue with delimiters
                        break
                    else:
                        # Check if this is the last string in an object
                        i = self.skip_whitespaces_at(idx=1, move_main_index=False)
                        j = self.skip_to_character(character='}', idx=i)

                        if j - i > 1:
                            rstring_delimiter_missing = False
                        elif self.get_char_at(j):
                            # Check for unmatched opening brace
                            for c in reversed(string_acc):
                                if c == '{':
                                    rstring_delimiter_missing = False
                                    break
                                elif c == '}':
                                    break

                if rstring_delimiter_missing:
                    logger.log(
                        LOG_LEVEL,
                        "While parsing a string missing the left delimiter in object value context, we found a , or } and we couldn't determine that a right delimiter was present. Stopping here",
                    )
                    break

            # Handle array closing brackets
            if char == ']' and JsonContext.ContextValues.ARRAY in self.context.context:
                i = self.skip_to_character(rstring_delimiter)
                if not self.get_char_at(i):
                    break

            # Accumulate the character
            string_acc += char
            self.index += 1
            char = self.get_char_at()

            # Handle escape sequences
            if char and string_acc[-1] == '\\':
                logger.log(LOG_LEVEL, 'Found an escape sequence, normalizing it')
                # Remove the backslash from string
                string_acc = string_acc[:-1]

                # Handle all standard JSON escape sequences
                escape_seqs = {
                    '"': '"',
                    "'": "'",
                    '\\': '\\',
                    '/': '/',
                    'b': '\b',
                    'f': '\f',
                    'n': '\n',
                    'r': '\r',
                    't': '\t',
                }

                if char in escape_seqs:
                    string_acc += escape_seqs[char]
                elif char == 'u' and self.index + 4 < len(self.json_str):
                    # Handle Unicode escape sequences \uXXXX
                    try:
                        # Get the next 4 characters
                        hex_chars = ''
                        for i in range(1, 5):
                            hex_char = self.get_char_at(i)
                            if hex_char is False:
                                break
                            hex_chars += hex_char

                        # Convert to Unicode character
                        if len(hex_chars) == 4:
                            unicode_char = chr(int(hex_chars, 16))
                            string_acc += unicode_char
                            # Skip the 4 hex characters
                            self.index += 4
                        else:
                            # Just add the 'u' if we can't parse the full sequence
                            string_acc += char
                    except ValueError:
                        # If conversion fails, just add the 'u'
                        string_acc += char
                else:
                    # For unknown escape sequences, keep the character as is
                    string_acc += char

                self.index += 1
                char = self.get_char_at()

            # Handle colon after object key
            if (
                char == ':'
                and not missing_quotes
                and self.context.current == JsonContext.ContextValues.OBJECT_KEY
            ):
                # Check if this is a missing right quote
                i = self.skip_to_character(character=lstring_delimiter, idx=1)
                next_c = self.get_char_at(i)

                if next_c:
                    i += 1
                    i = self.skip_to_character(character=rstring_delimiter, idx=i)
                    next_c = self.get_char_at(i)

                    if next_c:
                        i += 1
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)

                        if next_c and next_c in [',', '}']:
                            logger.log(
                                LOG_LEVEL,
                                'While parsing a string missing the right delimiter in object key context, we found a :, stopping here',
                            )
                            break
                else:
                    logger.log(
                        LOG_LEVEL,
                        'While parsing a string missing the right delimiter in object key context, we found a :, stopping here',
                    )
                    break

            # Handle quotes within strings
            if char == rstring_delimiter:
                # Handle doubled quotes
                if doubled_quotes and self.get_char_at(1) == rstring_delimiter:
                    logger.log(
                        LOG_LEVEL,
                        'While parsing a string, we found a doubled quote, ignoring it',
                    )
                    self.index += 1
                elif (
                    missing_quotes
                    and self.context.current == JsonContext.ContextValues.OBJECT_VALUE
                ):
                    # Check if delimiter is end of value or start of key
                    i = 1
                    next_c = self.get_char_at(i)

                    while next_c and next_c not in [
                        rstring_delimiter,
                        lstring_delimiter,
                    ]:
                        i += 1
                        next_c = self.get_char_at(i)

                    if next_c:
                        i += 1
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)

                        if next_c and next_c == ':':
                            self.index -= 1
                            char = self.get_char_at()
                            logger.log(
                                LOG_LEVEL,
                                'In a string with missing quotes and object value context, I found a delimiter but it turns out it was the beginning on the next key. Stopping here.',
                            )
                            break
                elif unmatched_delimiter:
                    unmatched_delimiter = False
                    string_acc += str(char)
                    self.index += 1
                    char = self.get_char_at()
                else:
                    # Check for another delimiter later in the string
                    i = 1
                    next_c = self.get_char_at(i)
                    check_comma_in_object_value = True

                    while next_c and next_c not in [
                        rstring_delimiter,
                        lstring_delimiter,
                    ]:
                        if check_comma_in_object_value and next_c.isalpha():
                            check_comma_in_object_value = False

                        # Check for relevant context delimiters
                        if (
                            (
                                JsonContext.ContextValues.OBJECT_KEY
                                in self.context.context
                                and next_c in [':', '}']
                            )
                            or (
                                JsonContext.ContextValues.OBJECT_VALUE
                                in self.context.context
                                and next_c == '}'
                            )
                            or (
                                JsonContext.ContextValues.ARRAY in self.context.context
                                and next_c in [']', ',']
                            )
                            or (
                                check_comma_in_object_value
                                and self.context.current
                                == JsonContext.ContextValues.OBJECT_VALUE
                                and next_c == ','
                            )
                        ):
                            break
                        i += 1
                        next_c = self.get_char_at(i)

                    # Check for comma in object value
                    if (
                        next_c == ','
                        and self.context.current
                        == JsonContext.ContextValues.OBJECT_VALUE
                    ):
                        i += 1
                        i = self.skip_to_character(character=rstring_delimiter, idx=i)
                        next_c = self.get_char_at(i)

                        # Check for closing brace after delimiter
                        i += 1
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)

                        if next_c == '}':
                            # Valid string with misplaced quotes
                            logger.log(
                                LOG_LEVEL,
                                'While parsing a string, we misplaced a quote that would have closed the string but has a different meaning here since this is the last element of the object, ignoring it',
                            )
                            unmatched_delimiter = not unmatched_delimiter
                            string_acc += str(char)
                            self.index += 1
                            char = self.get_char_at()
                    elif (
                        next_c == rstring_delimiter and self.get_char_at(i - 1) != '\\'
                    ):
                        # Check if path to next delimiter is just whitespace
                        if all(
                            str(self.get_char_at(j)).isspace()
                            for j in range(1, i)
                            if self.get_char_at(j)
                        ):
                            break

                        if (
                            self.context.current
                            == JsonContext.ContextValues.OBJECT_VALUE
                        ):
                            # Check if this is a key with missing comma
                            i = self.skip_to_character(
                                character=rstring_delimiter, idx=i + 1
                            )
                            i += 1
                            next_c = self.get_char_at(i)

                            while next_c and next_c != ':':
                                if next_c in [',', ']', '}'] or (
                                    next_c == rstring_delimiter
                                    and self.get_char_at(i - 1) != '\\'
                                ):
                                    break
                                i += 1
                                next_c = self.get_char_at(i)

                            # Only if colon not found, this is a misplaced quote
                            if next_c != ':':
                                logger.log(
                                    LOG_LEVEL,
                                    'While parsing a string, we a misplaced quote that would have closed the string but has a different meaning here, ignoring it',
                                )
                                unmatched_delimiter = not unmatched_delimiter
                                string_acc += str(char)
                                self.index += 1
                                char = self.get_char_at()
                        elif self.context.current == JsonContext.ContextValues.ARRAY:
                            # Handle quotes in array strings
                            logger.log(
                                LOG_LEVEL,
                                'While parsing a string in Array context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it',
                            )
                            unmatched_delimiter = not unmatched_delimiter
                            string_acc += str(char)
                            self.index += 1
                            char = self.get_char_at()
                        elif (
                            self.context.current == JsonContext.ContextValues.OBJECT_KEY
                        ):
                            # Handle misplaced quotes in object keys
                            logger.log(
                                LOG_LEVEL,
                                'While parsing a string in Object Key context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it',
                            )
                            string_acc += str(char)
                            self.index += 1
                            char = self.get_char_at()

        # Handle special case of comment instead of valid string
        if (
            char
            and missing_quotes
            and self.context.current == JsonContext.ContextValues.OBJECT_KEY
            and char.isspace()
        ):
            logger.log(
                LOG_LEVEL,
                'While parsing a string, handling an extreme corner case in which the LLM added a comment instead of valid string, invalidate the string and return an empty value',
            )
            self.skip_whitespaces_at()
            if self.get_char_at() not in [':', ',']:
                return ''

        # Update index only if closing quote found
        if char != rstring_delimiter:
            logger.log(
                LOG_LEVEL,
                'While parsing a string, we missed the closing quote, ignoring',
            )
            string_acc = string_acc.rstrip()
        else:
            self.index += 1

        # Clean whitespace in special cases
        if missing_quotes or (string_acc and string_acc[-1] == '\n'):
            string_acc = string_acc.rstrip()

        return string_acc

    def parse_number(self) -> Union[float, int, str, JSONReturnType]:
        """
        Parse a JSON number.

        Returns:
            Parsed number as int, float, or string if not a valid number
        """
        number_str = ''
        char = self.get_char_at()
        is_array = self.context.current == JsonContext.ContextValues.ARRAY

        # Accumulate number characters
        while char and char in self.NUMBER_CHARS and (not is_array or char != ','):
            number_str += char
            self.index += 1
            char = self.get_char_at()

        # Handle invalid number endings
        if number_str and number_str[-1] in '-eE/,':
            number_str = number_str[:-1]
            self.index -= 1

        try:
            # Parse according to number type
            if ',' in number_str:
                return str(number_str)
            if '.' in number_str or 'e' in number_str or 'E' in number_str:
                return float(number_str)
            elif number_str == '-':
                # Handle stray minus sign
                return self.parse_json()
            else:
                return int(number_str)
        except ValueError:
            return number_str

    def parse_boolean_or_null(self) -> Union[bool, str, None]:
        """
        Parse a JSON boolean or null value.

        Returns:
            Parsed boolean, null, or empty string if not a valid value
        """
        starting_index = self.index
        char = (self.get_char_at() or '').lower()

        # Determine the potential value
        value: Optional[Tuple[str, Optional[bool]]] = None
        if char == 't':
            value = ('true', True)
        elif char == 'f':
            value = ('false', False)
        elif char == 'n':
            value = ('null', None)

        if value:
            # Try to match the entire token
            i = 0
            while char and i < len(value[0]) and char == value[0][i]:
                i += 1
                self.index += 1
                char = (self.get_char_at() or '').lower()

            if i == len(value[0]):
                return value[1]

        # Reset if no match
        self.index = starting_index
        return ''

    def parse_comment(self) -> str:
        """
        Parse a comment (# or // or /* */).

        Returns:
            Empty string (comments are ignored in JSON)
        """
        char = self.get_char_at()

        # Define characters that terminate line comments
        termination_characters = ['\n', '\r']
        if JsonContext.ContextValues.ARRAY in self.context.context:
            termination_characters.append(']')
        if JsonContext.ContextValues.OBJECT_VALUE in self.context.context:
            termination_characters.append('}')
        if JsonContext.ContextValues.OBJECT_KEY in self.context.context:
            termination_characters.append(':')

        # Parse line comment starting with #
        if char == '#':
            comment = ''
            while char and char not in termination_characters:
                comment += char
                self.index += 1
                char = self.get_char_at()
            logger.log(LOG_LEVEL, f'Found line comment: {comment}')
            return ''

        # Parse comments starting with /
        elif char == '/':
            next_char = self.get_char_at(1)

            # Line comment //
            if next_char == '/':
                comment = '//'
                self.index += 2  # Skip both slashes
                char = self.get_char_at()

                while char and char not in termination_characters:
                    comment += char
                    self.index += 1
                    char = self.get_char_at()

                logger.log(LOG_LEVEL, f'Found line comment: {comment}')
                return ''

            # Block comment /* */
            elif next_char == '*':
                comment = '/*'
                self.index += 2  # Skip /*

                while True:
                    char = self.get_char_at()
                    if not char:
                        logger.log(
                            LOG_LEVEL,
                            'Reached end-of-string while parsing block comment; unclosed block comment.',
                        )
                        break

                    comment += char
                    self.index += 1

                    if comment.endswith('*/'):
                        break

                logger.log(LOG_LEVEL, f'Found block comment: {comment}')
                return ''
            else:
                # Not a recognized comment pattern
                self.index += 1
                return ''
        else:
            # Not a comment, skip current character
            self.index += 1
            return ''

    # Helper methods

    def get_char_at(self, count: int = 0) -> Union[str, Literal[False]]:
        """
        Get character at current index + count.

        Args:
            count: Offset from current index

        Returns:
            Character at position or False if out of bounds
        """
        try:
            return self.json_str[self.index + count]
        except IndexError:
            return False

    def skip_whitespaces_at(self, idx: int = 0, move_main_index=True) -> int:
        """
        Skip over whitespace characters.

        Args:
            idx: Starting offset from current index
            move_main_index: Whether to update the main index

        Returns:
            Number of characters skipped
        """
        try:
            char = self.json_str[self.index + idx]
        except IndexError:
            return idx

        while char.isspace():
            if move_main_index:
                self.index += 1
            else:
                idx += 1

            try:
                char = self.json_str[self.index + idx]
            except IndexError:
                return idx

        return idx

    def skip_to_character(self, character: str, idx: int = 0) -> int:
        """
        Skip to the next occurrence of a specific character.

        Args:
            character: Character to find
            idx: Starting offset from current index

        Returns:
            Index of the character relative to current position, or length if not found
        """
        try:
            char = self.json_str[self.index + idx]
        except IndexError:
            return idx

        while char != character:
            idx += 1
            try:
                char = self.json_str[self.index + idx]
            except IndexError:
                return idx

        # Check for escaped character
        if self.index + idx > 0 and self.json_str[self.index + idx - 1] == '\\':
            # Skip escaped character and continue searching
            return self.skip_to_character(character=character, idx=idx + 1)

        return idx

    def repair_json(self, text: str) -> str:
        """
        Repair common JSON formatting issues.

        Args:
            text: JSON text to repair

        Returns:
            Repaired JSON text
        """
        if not text:
            return text

        # Reset stats for this repair operation
        self.reset_stats()

        # Series of repair operations
        text = self._remove_control_chars(text)
        text = self._fix_unicode_escapes(text)
        text = self._convert_single_quotes(text)
        text = self._fix_trailing_commas(text)
        text = self._balance_braces(text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """
        Remove or escape control characters that would break JSON parsing.

        Args:
            text: JSON text to clean

        Returns:
            Cleaned JSON text
        """
        # Remove control characters that aren't valid in JSON strings
        control_chars_pattern = re.compile(r'[\x00-\x1F\x7F]')
        result = control_chars_pattern.sub('', text)

        if result != text:
            self.stats['control_chars_removed'] += 1

        return result

    def _fix_unicode_escapes(self, text: str) -> str:
        """
        Fix malformed Unicode escape sequences.

        Args:
            text: JSON text to fix

        Returns:
            Fixed JSON text
        """
        # Fix incorrect Unicode escapes like \u00 (should be \u0000)
        bad_unicode = re.compile(r'\\u([0-9a-fA-F]{1,3})[^\d\w]')

        def fix_unicode_match(match):
            self.stats['unicode_escapes_fixed'] += 1
            code = match.group(1)
            # Pad with zeros to get 4 characters
            padded = code.zfill(4)
            # Get the character after the code
            after = match.string[match.end() - 1]
            return f'\\u{padded}{after}'

        return bad_unicode.sub(fix_unicode_match, text)

    def _convert_single_quotes(self, text: str) -> str:
        """
        Convert single quotes to double quotes for JSON compatibility.

        Args:
            text: JSON text to fix

        Returns:
            Fixed JSON text
        """
        if "'" not in text:
            return text

        # Skip if already using double quotes predominantly
        if text.count('"') > text.count("'"):
            return text

        result = ''
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                result += char
                continue

            if char == '\\':
                escape_next = True
                result += char
                continue

            # Convert single quotes that delimit strings
            if char == "'":
                # Check if string delimiter or apostrophe
                if not in_string:
                    # Starting a string
                    in_string = True
                    result += '"'
                    self.stats['single_quotes_converted'] += 1
                else:
                    # Check if ending a string based on next character
                    if i + 1 < len(text) and text[i + 1] in ',]}:':
                        # Ending a string
                        in_string = False
                        result += '"'
                        self.stats['single_quotes_converted'] += 1
                    else:
                        # Likely an apostrophe
                        result += "'"
            else:
                # Handle double quotes inside single-quoted strings
                if char == '"' and in_string:
                    # Check if escaped
                    if i > 0 and text[i - 1] == '\\':
                        # Remove escape since converting to double quotes
                        result = result[:-1] + char
                    else:
                        # Add escape for double quote inside string
                        result += '\\' + char
                else:
                    result += char

                    # Update string state
                    if char == '"' and not escape_next:
                        in_string = not in_string

        return result

    def _fix_trailing_commas(self, text: str) -> str:
        """
        Remove trailing commas before closing brackets or braces.

        Args:
            text: JSON text to fix

        Returns:
            Fixed JSON text
        """
        # Pattern for trailing commas
        pattern = re.compile(r',\s*([}\]])')

        def remove_comma(match):
            self.stats['commas_fixed'] += 1
            return match.group(1)

        return pattern.sub(remove_comma, text)

    def _balance_braces(self, text: str) -> str:
        """
        Add missing closing brackets or remove extra closing brackets.

        Args:
            text: JSON text to fix

        Returns:
            Fixed JSON text
        """
        # Count opening and closing brackets
        opens = {'[': 0, '{': 0}
        closes = {']': 0, '}': 0}

        # Count respecting string boundaries
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char in opens:
                    opens[char] += 1
                elif char in closes:
                    closes[char] += 1

        # Add missing closing brackets
        result = text
        if opens['['] > closes[']']:
            result += ']' * (opens['['] - closes[']'])
            self.stats['braces_balanced'] += 1

        if opens['{'] > closes['}']:
            result += '}' * (opens['{'] - closes['}'])
            self.stats['braces_balanced'] += 1

        # Log extra closing brackets
        if closes[']'] > opens['['] or closes['}'] > opens['{']:
            logger.warning('Extra closing brackets detected - complex repair needed')

        return result


def parse_json_string(
    json_string: str,
    strict_mode: bool = False,
) -> Any:
    """
    Parse a JSON string using the enhanced parser.

    Args:
        json_string: JSON string to parse
        strict_mode: Whether to use strict mode

    Returns:
        Parsed JSON object
    """
    parser = JSONParser(json_str=json_string, strict_mode=strict_mode)
    return parser.parse()


def repair_json_string(json_string: str) -> str:
    """
    Repair a malformed JSON string.

    Args:
        json_string: JSON string to repair

    Returns:
        Repaired JSON string
    """
    parser = JSONParser()
    return parser.repair_json(json_string)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that may contain other content.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON string
    """
    parser = JSONParser()
    return parser.extract_json(text)
