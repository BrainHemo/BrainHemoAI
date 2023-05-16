import numpy as np
from skimage import measure, morphology


def remove_small_object(data: np.ndarray) -> np.ndarray:
    assert data.ndim == 3
    nz = data.shape[0]
    result = np.zeros_like(data)
    for z in range(nz):
        mask = data[z] != 0
        mask = morphology.remove_small_objects(mask, min_size=100)
        result[z][mask] = data[z][mask]
    return result


def closing(data: np.ndarray) -> np.ndarray:
    assert data.ndim == 3
    nz = data.shape[0]
    result = np.zeros_like(data)
    for z in range(nz):
        mask = data[z] != 0
        mask = morphology.closing(mask, morphology.disk(3))
        result[z][mask] = data[z][mask]
        pad = np.logical_and(mask, data[z] == 0)
        result[z][pad] = 4
    return result


def find_mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask != 0
    nx, ny = mask.shape
    points = []
    for x in range(nx):
        for y in range(ny):
            if mask[x, y]:
                if x == 0 or not mask[x - 1, y]:
                    points.append([x, y])
                    continue
                if y == 0 or not mask[x, y - 1]:
                    points.append([x, y])
                    continue
                if x == nx - 1 or not mask[x + 1, y]:
                    points.append([x, y])
                    continue
                if y == ny - 1 or not mask[x, y + 1]:
                    points.append([x, y])
                    continue
    avg_point = np.average(points, axis=0)
    cx, cy = avg_point[0], avg_point[1]
    points.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    points = np.array(points)

    return points


def label_region(labelmap: np.ndarray) -> np.ndarray:
    types = np.unique(labelmap)
    result = np.zeros_like(labelmap)
    for t in types:
        if t == 0:
            continue
        labels = measure.label(labelmap == t)
        labels[labels != 0] += np.max(result)  # shift the label value
        result = result + labels
    return result


def region_connection(labelmap: np.ndarray) -> np.ndarray:
    """
    This function needs to be run twice
    1111111112223333 -> 1111111111113333 => 3333 will not change because it's the largest region locally.
    """
    labels = label_region(labelmap)
    region_property = [{'area': 0, 'neighbors': set(), 'type': 0} for _ in range(np.max(labels) + 1)]
    nx, ny = labelmap.shape
    for x in range(nx):
        for y in range(ny):
            if labels[x, y] == 0:
                continue
            region_property[labels[x, y]]['area'] += 1
            region_property[labels[x, y]]['type'] = labelmap[x, y]
            direction = [[-1, 0], [1, 0], [0, -1], [0, 1],
                         [-1, -1], [1, -1], [-1, 1], [1, 1]]
            for direct in direction:
                tx, ty = x + direct[0], y + direct[1]
                if 0 <= tx < nx and 0 <= ty < ny and labels[x, y] != labels[tx, ty] and labels[tx, ty] != 0:
                    region_property[labels[x, y]]['neighbors'].add(labels[tx, ty])
    relabel = [i for i in range(len(region_property))]

    total_area = 0
    for i in range(len(region_property)):
        total_area += region_property[i]['area']

    for i in range(1, len(region_property)):
        if region_property[i]['area'] / total_area < 0.25 or region_property[i]['type'] == 4:
            # connect
            max_neighbor_id = -1
            max_neighbor_area = 0
            for neighbor_id in region_property[i]['neighbors']:
                if region_property[neighbor_id]['area'] > region_property[i]['area']:
                    if region_property[neighbor_id]['area'] > max_neighbor_area:
                        max_neighbor_area = region_property[neighbor_id]['area']
                        max_neighbor_id = neighbor_id
            if max_neighbor_id != -1:
                relabel[i] = max_neighbor_id

    result = labelmap.copy()
    for i in range(len(relabel)):
        l = relabel[i]
        # find the root： 0 1 1 2 => 0 1 1 1
        while l != relabel[l]:
            l = relabel[l]
        # result[result == region_property[i]['type']] = region_property[l]['type']
        result[labels == i] = region_property[l]['type']

    return result


def rectify_blood_type(blood: np.ndarray) -> np.ndarray:
    nz, nx, ny = blood.shape
    result = np.zeros_like(blood)
    for z in range(nz):
        labels = measure.label(blood[z] != 0)
        regions = measure.regionprops(labels)
        for region in regions:
            # rectify the blood for every type
            temp = blood[z].copy()
            temp[labels != region.label] = 0
            foreground = temp != 0
            types = np.unique(temp[foreground])
            if len(types) < 2:
                # single type (background and single blood)
                result[z][foreground] = temp[foreground]
            else:
                result[z][foreground] = region_connection(temp)[foreground]
    return result


def rectify(src: str, dst: str):
    data = np.load(src)
    brain: np.ndarray = data['brain']
    blood: np.ndarray = data['blood']
    preds: np.ndarray = data['preds']
    if brain.ndim == 4:
        brain = brain[0]
        blood = blood[0]
        preds = preds[0]

    # index = data['index']
    index = 0
    brain = (brain - np.min(brain)) / (np.max(brain) - np.min(brain) + 1e-6)

    nz, _, ny = brain.shape

    preds = remove_small_object(preds)

    # run twice
    preds = rectify_blood_type(preds.copy())
    preds = rectify_blood_type(preds.copy())

    if dst is not None:
        np.savez_compressed(dst, brain=np.array([brain]), blood=np.array([blood]),
                            preds=np.array([preds]), index=index)
    return preds
