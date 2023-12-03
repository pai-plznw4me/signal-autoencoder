import numpy as np
import pandas as pd
from tqdm import tqdm
def slice_dataframe(df, interval, stride, output_type='numpy'):
    """
    주어진 DataFrame을 일정한 간격과 스트라이드에 따라 잘라주는 함수.


    Parameters:
    - df (pd.DataFrame): 잘라낼 DataFrame
    - interval (int): 간격(몇 개의 row를 하나의 slice로 할지)
    - stride (int): 스트라이드(얼마나 건너뛸지)
    - output_type (str): 출력 형식 선택 ('numpy'), numpy 로 지정되지 않으면 dataframe 형식으로 반환

    Returns:
    - sliced_dfs (list): 잘린 DataFrame들을 포함한 리스트

    Example:
    >>> df = pd.DataFrame({'value': range(1, 21)})
    >>> interval = 3
    >>> stride = 2
    >>> result = slice_dataframe(df, interval, stride, 'numpy')

    Example:
    >>> df = pd.DataFrame({'value': range(1, 21)})
    >>> interval = 3
    >>> stride = 2
    >>> result = slice_dataframe(df, interval, stride)

    """
    sliced_dfs = []

    for i in tqdm(range(0, len(df) - interval + 1, stride)):
        sliced_df = df.iloc[i:i + interval]
        if output_type == 'numpy':
            sliced_df = sliced_df.values
        sliced_dfs.append(sliced_df)
    return sliced_dfs


def split_sequence(sequence, n_steps):
    """
    주어진 시퀀스를 입력과 출력 부분으로 나누어주는 함수.

    Parameters:
    - sequence (list or array): 입력 시퀀스
    - n_steps (int): 각 패턴의 길이, 즉 입력 시퀀스의 크기

    Returns:
    - X (array): 입력 패턴들의 배열
    - y (array): 출력 패턴들의 배열

    Examples:
    >>> sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> n_steps = 3
    >>> X, y = split_sequence(sequence, n_steps)
    >>> print(X)
    [[1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]
     [5 6 7]
     [6 7 8]]
    >>> print(y)
    [4 5 6 7 8 9]
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
