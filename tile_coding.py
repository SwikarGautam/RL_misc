import numpy as np
from itertools import permutations
from itertools import combinations


def tilings(value_low_high, num_tiles, num_tilings, dis_vec_values=None):

    value_range = np.transpose(np.array(value_low_high).tolist())

    if dis_vec_values is None:
        dis_vec_values = []
        for i in range(1, len(value_range)+1):
            value = 2*i-1
            if value < num_tiles:
                dis_vec_values.append(2*i-1)

    if not isinstance(num_tilings, list):
        assert isinstance(num_tilings, int)
        num_tilings_list = []
        for _ in range(len(value_range)):
            num_tilings_list.append(num_tilings)
        num_tilings = num_tilings_list

    ranges = [high - low for (low, high) in value_range]
    tile_ranges = [a*b/(b-1) for a, b in zip(ranges, num_tilings)]

    offset_unit = np.array(
        [a/b for a, b in zip(tile_ranges, num_tilings)])/num_tiles

    dis_vecs = [perm for perm, _ in zip(
        permutations(dis_vec_values, len(value_range)), range((num_tiles-1)*5))]

    offsets = [np.zeros(len(num_tilings))]
    tiles = []

    for _ in range(num_tiles-1):

        while True:
            i = np.random.choice([a for a in range(len(offsets))])
            base = offsets[i]

            i = np.random.choice([a for a in range(len(dis_vecs))])
            dis_vec = np.array(dis_vecs[i], dtype=int)

            new_tile_offset = base + dis_vec

            if np.any(new_tile_offset >= num_tiles):
                continue

            flag = False

            for o in offsets:
                if np.array_equal(o, new_tile_offset):
                    flag = True
                    break
            if flag:
                continue

            offsets.append(new_tile_offset)
            break

    for offset in offsets:

        offset_value = offset*offset_unit
        start = np.array([a[0] for a in value_range]) - offset_value
        tile = []

        for i, (s, r, n) in enumerate(zip(start, tile_ranges, num_tilings)):
            tile.append(np.linspace(s, s+r, n+1))

        tiles.append(np.array(tile))

    return np.array(tiles)


def multiply(a_list):
    result = 1
    for i in a_list:
        result *= i
    return result


def features(tiles, observations):
    index = []
    index_list = []
    stride = [1]

    for n, tiling in enumerate(tiles):
        for dim, o in zip(tiling[::-1], observations[::-1]):
            for i in range(len(dim)):
                if dim[i+1] > o:
                    index_list.append(i)
                    stride.append(len(dim)-1)
                    break
        index_list.append(n)
        mul_list = [multiply(stride[:j]) for j in range(1, len(stride)+1)]

        result = 0

        for i, m in zip(index_list, mul_list):
            result += i*m

        index.append(result)
        stride = [1]
        index_list = []

    return index


if __name__ == "__main__":
    print('running')
    tiles = tilings([[-2.5, -3.5, -0.3, -4],
                     [2.5, 3.5, 0.3, 4]], 8, 4)

    print(tiles)
    print(4*4*4*4)
