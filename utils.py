import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_yhat_y(ys, ys_hat, names, save_dir, xs=None, show=True):

    """
     주어진 데이터의 실제 값(ys)과 예측 값(ys_hat)을 그래프로 시각화하는 함수입니다.

     Parameters:
     - ys (np.ndarray): 실제 값의 배열. Shape는 (N_data, N_col)이어야 합니다.
     - ys_hat (np.ndarray): 예측 값의 배열. Shape는 (N_data, N_col)이어야 합니다.
     - names (bool): 각 그래프에 대한 이름(라벨)의 리스트. 없을 경우 인덱스로 대체됩니다.
     - save_dir (str): 그래프를 저장할 디렉토리 경로. None일 경우 저장하지 않습니다.
     - xs (np.ndarray, optional): x 축 값의 배열. None일 경우 데이터 포인트의 인덱스를 사용합니다.
     - show (bool, optional): 그래프를 화면에 표시할지 여부 (기본값은 True).

     Returns:
     - None

     예시:
     plot_yhat_y(ys_data, ys_hat_data, names=['Feature 1', 'Feature 2'], save_dir='./plots', xs=time_steps, show=True)
     """
    assert len(ys.shape) == 2, 'ys shape (N_data, N_col) 이여야 합니다. 현재 입려된 shape {}'.format(ys.shape)
    assert len(ys_hat.shape) == 2, 'ys_hat shape (N_data, N_col) 이여야 합니다. 현재 입려된 shape {}'.format(ys_hat.shape)
    if not xs:
        assert len(xs.shape) == 1, '입력된 xs shape 는 N_data 이여야 합니다. 현재 입려된 shape {}'.format(xs.shape)
        xs = np.arange(len(ys))
    n_col = ys.shape[1]

    for idx in range(n_col):
        plt.plot(xs, ys[:, idx], label='y', alpha=0.5)
        plt.plot(xs, ys_hat[:, idx], label='y_hat', alpha=0.5)
        plt.legend()
        if not names:
            plt.title(idx)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_ext = '.png'
            save_path = os.path.join(save_dir, idx + save_ext)
            plt.savefig(save_path)
        if show:
            plt.show()


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
