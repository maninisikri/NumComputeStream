import os
import io as _io
from typing import Generator
import numpy as np

def load_csv(
    filepath: str,
    delimiter: str = ',',
    dtype: type = float,
    missing_values: str = '',
    filling_values: float = np.nan,
) -> np.ndarray:
    """
    Load a CSV file into a 2-D NumPy array.
    Uses numpy.genfromtxt under the hood. Missing cells are filled
    with filling_values which defaults to np.nan.
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str
        Column separator. Default comma.
    dtype : data-type
        NumPy dtype for the output array. Default float.
    missing_values : str
        Token that marks a missing cell. Default empty string.
    filling_values : float
        Value used to fill missing entries. Default np.nan.
    Returns
    -------
    np.ndarray
        2-D array of shape (n_rows, n_cols).
    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If the file is empty.
    Time complexity: O(n)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File is empty: '{filepath}'")
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=filling_values,
        autostrip=True,
    )
    if data.ndim == 0:
        raise ValueError(f"File could not be parsed: '{filepath}'")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def load_csv_chunked(
    filepath: str,
    chunksize: int,
    delimiter: str = ',',
    dtype: type = float,
    missing_values: str = '',
    filling_values: float = np.nan,
) -> Generator[np.ndarray, None, None]:
    """
    Stream a large CSV file in fixed-size row chunks.
    Reads line by line so the full file never has to sit in memory.
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    chunksize : int
        Maximum rows per chunk. Must be at least 1.
    delimiter : str
        Column separator. Default comma.
    dtype : data-type
        NumPy dtype for output arrays. Default float.
    missing_values : str
        Token that marks a missing cell.
    filling_values : float
        Value used to fill missing entries.
    Yields
    ------
    np.ndarray
        2-D array of shape (<=chunksize, n_cols).
    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If chunksize is less than 1.
    Time complexity: O(chunksize * n_cols) per chunk
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}")
    with open(filepath, 'r', encoding='utf-8') as fh:
        buffer: list[str] = []
        for line in fh:
            buffer.append(line)
            if len(buffer) == chunksize:
                yield _parse_lines(buffer, delimiter, dtype, missing_values, filling_values)
                buffer = []
        if buffer:
            yield _parse_lines(buffer, delimiter, dtype, missing_values, filling_values)

def save_csv(
    array: np.ndarray,
    filepath: str,
    delimiter: str = ',',
    header: str = None,
) -> None:
    """
    Write a NumPy array to a CSV file.
    Parameters
    ----------
    array : np.ndarray
        1-D or 2-D array to save.
    filepath : str
        Destination file path.
    delimiter : str
        Column separator. Default comma.
    header : str or None
        Optional header row. Default None.
    Raises
    ------
    ValueError
        If array is not 1-D or 2-D.
    Time complexity: O(n)
    """
    if array.ndim not in (1, 2):
        raise ValueError(f"array must be 1-D or 2-D, got {array.ndim}-D.")
    np.savetxt(
        filepath,
        array,
        delimiter=delimiter,
        header=header if header is not None else '',
        comments='',
    )

def _parse_lines(lines, delimiter, dtype, missing_values, filling_values):
    """Parse a list of raw CSV lines into a 2-D NumPy array."""
    block = ''.join(lines)
    arr = np.genfromtxt(
        _io.StringIO(block),
        delimiter=delimiter,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=filling_values,
        autostrip=True,
    )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr