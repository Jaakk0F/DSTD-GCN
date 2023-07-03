import torch


def st_transform(data):
    # (batch, time, space) -> (batch, space, time)
    data = data.permute(0, 2, 1).contiguous()
    return data


def st_inverse(data):
    # (batch, space, time) -> (batch, time, space)
    data = data.permute(0, 2, 1).contiguous()
    return data


def cst_transform(data, c=3):
    # (batch, time, space * c) -> (batch, time, space, c)
    B, T, SC = data.shape
    assert SC % c == 0
    S = SC // c
    data = data.view(B, T, S, c)
    data = data.permute(0, 3, 2, 1).contiguous()
    return data


def cst_inverse(data, c=3):
    # (batch, time, space * c) -> (batch, time, space, c)
    B, C, S, T = data.shape
    assert C == c
    data = data.permute(0, 3, 2, 1).contiguous()
    data = data.view(B, T, S * c)
    return data


def tsc_transform(data, c=3):
    # (batch, time, space * c) -> (batch, time, space, c)
    B, T, SC = data.shape
    assert SC % c == 0
    S = SC // c
    data = data.view(B, T, S, c)
    return data


def tsc_inverse(data, c=3):
    # (batch, time, space, c) ->  (batch, time, space * c)
    B, T, S, C = data.shape
    assert C == c
    data = data.view(B, T, S * C)
    return data


def tscr_h36m_transform(data, c=3):
    # (batch, time, space * c) -> (batch, time, space, c)
    B, T, SC = data.shape
    assert SC % c == 0
    S = SC // c
    # h36m
    orders = [
        21, 20, 19, 18, 17, 12, 13, 14, 15, 16, 11, 10, 9, 8, 4, 5, 6, 7, 0, 1,
        2, 3
    ]
    data = data.view(B, T, S, c)
    assert S == len(orders)
    # data = data.gather(dim=-2, index=torch.Tensor(orders).to(
    #     data.device)).contiguous()
    data = data[:, :, orders, :].contiguous()
    return data


def tscr_h36m_inverse(data, c=3):
    # (batch, time, space, c) ->  (batch, time, space * c)
    B, T, S, C = data.shape
    assert C == c
    # h36m
    orders = [
        18, 19, 20, 21, 14, 15, 16, 17, 13, 12, 11, 10, 5, 6, 7, 8, 9, 4, 3, 2,
        1, 0
    ]
    # data = data.gather(dim=-2, index=torch.Tensor(orders).to(
    #     data.device)).contiguous()
    data = data[:, :, orders, :].contiguous()
    data = data.view(B, T, S * C)
    return data


def tscr_cmu_transform(data, c=3):
    # (batch, time, space * c) -> (batch, time, space, c)
    B, T, SC = data.shape
    assert SC % c == 0
    S = SC // c
    # cmu
    orders = [
        23, 21, 20, 14, 15, 17, 12, 11, 9, 5, 6, 7, 1, 2, 3, 0, 4, 8, 10, 13,
        19, 16, 18, 22, 24
    ]
    data = data.view(B, T, S, c)
    assert S == len(orders)
    # data = data.gather(dim=-2, index=torch.Tensor(orders).to(
    #     data.device)).contiguous()
    data = data[:, :, orders, :].contiguous()
    return data


def tscr_cmu_inverse(data, c=3):
    # (batch, time, space, c) ->  (batch, time, space * c)
    B, T, S, C = data.shape
    assert C == c
    # cmu
    orders = [
        15, 12, 13, 14, 16, 9, 10, 11, 17, 8, 18, 7, 6, 19, 3, 4, 21, 5, 22,
        20, 2, 1, 23, 0, 24
    ]
    # data = data.gather(dim=-2, index=torch.Tensor(orders).to(
    #     data.device)).contiguous()
    data = data[:, :, orders, :].contiguous()
    data = data.view(B, T, S * C)
    return data


def tscr_3dpw_transform(data, c=3):
    # (batch, time, space * c) -> (batch, time, space, c)
    B, T, SC = data.shape
    assert SC % c == 0
    S = SC // c
    # cmu
    orders = [
        22,20,18,16,13,12,15,17,19,21,14,11,8,5,2,1,4,7,10,0,3,6,9
    ]
    data = data.view(B, T, S, c)
    assert S == len(orders)
    # data = data.gather(dim=-2, index=torch.Tensor(orders).to(
    #     data.device)).contiguous()
    data = data[:, :, orders, :].contiguous()
    return data


def tscr_3dpw_inverse(data, c=3):
    # (batch, time, space, c) ->  (batch, time, space * c)
    B, T, S, C = data.shape
    assert C == c
    # cmu
    orders = [
        19,15,14,20,16,13,21,17,12,22,18,11,5,4,10,6,3,7,2,8,1,9,0
    ]
    # data = data.gather(dim=-2, index=torch.Tensor(orders).to(
    #     data.device)).contiguous()
    data = data[:, :, orders, :].contiguous()
    data = data.view(B, T, S * C)
    return data