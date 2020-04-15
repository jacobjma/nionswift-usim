# standard libraries
import abc
import gettext
import numbers
import os
import random
import typing

import numpy
import numpy as np
import scipy.ndimage
from nion.data import Image
from nion.utils import Geometry

from .radial_functions import interpolate_radial_functions

_ = gettext.gettext


class Feature:

    def __init__(self, position_m, size_m, edges, plasmon_eV, plurality):
        self.position_m = position_m
        self.size_m = size_m
        self.edges = edges
        self.plasmon_eV = plasmon_eV
        self.plurality = plurality

    def get_scan_rect_m(self, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize,
                        center_nm: Geometry.FloatPoint) -> Geometry.FloatRect:
        scan_size_m = Geometry.FloatSize(height=fov_nm.height, width=fov_nm.width) / 1E9
        scan_rect_m = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint.make(center_nm) / 1E9, scan_size_m)
        scan_rect_m -= offset_m
        return scan_rect_m

    def get_feature_rect_m(self) -> Geometry.FloatRect:
        return Geometry.FloatRect.from_center_and_size(self.position_m, self.size_m)

    def intersects(self, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint,
                   probe_position: Geometry.FloatPoint) -> bool:
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        probe_position_m = Geometry.FloatPoint(y=probe_position.y * scan_rect_m.height + scan_rect_m.top,
                                               x=probe_position.x * scan_rect_m.width + scan_rect_m.left)
        return scan_rect_m.intersects_rect(feature_rect_m) and feature_rect_m.contains_point(probe_position_m)

    def plot(self, data: numpy.ndarray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize,
             center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        # TODO: how does center_nm interact with stage position?
        # TODO: take into account feature angle
        # TODO: take into account frame parameters angle
        # TODO: expand features to other shapes than rectangle
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        sum = 0
        if scan_rect_m.intersects_rect(feature_rect_m):
            feature_rect_top_px = int(shape[0] * (feature_rect_m.top - scan_rect_m.top) / scan_rect_m.height)
            feature_rect_left_px = int(shape[1] * (feature_rect_m.left - scan_rect_m.left) / scan_rect_m.width)
            feature_rect_height_px = int(shape[0] * feature_rect_m.height / scan_rect_m.height)
            feature_rect_width_px = int(shape[1] * feature_rect_m.width / scan_rect_m.width)
            if feature_rect_top_px < 0:
                feature_rect_height_px += feature_rect_top_px
                feature_rect_top_px = 0
            if feature_rect_left_px < 0:
                feature_rect_width_px += feature_rect_left_px
                feature_rect_left_px = 0
            if feature_rect_top_px + feature_rect_height_px > shape[0]:
                feature_rect_height_px = shape[0] - feature_rect_top_px
            if feature_rect_left_px + feature_rect_width_px > shape[1]:
                feature_rect_width_px = shape[1] - feature_rect_left_px
            feature_rect_origin_px = Geometry.IntPoint(y=feature_rect_top_px, x=feature_rect_left_px)
            feature_rect_size_px = Geometry.IntSize(height=feature_rect_height_px, width=feature_rect_width_px)
            feature_rect_px = Geometry.IntRect(feature_rect_origin_px, feature_rect_size_px)
            data[feature_rect_px.top:feature_rect_px.bottom, feature_rect_px.left:feature_rect_px.right] += 1.0
            sum += (feature_rect_px.bottom - feature_rect_px.top) * (feature_rect_px.right - feature_rect_px.left)
        return sum


class Sample(abc.ABC):

    @property
    @abc.abstractmethod
    def title(self) -> str: ...

    @property
    @abc.abstractmethod
    def features(self) -> typing.List[Feature]: ...

    @abc.abstractmethod
    def plot_features(self, data: numpy.ndarray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize,
                      extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint,
                      used_size: Geometry.IntSize) -> None: ...


class RectangleFlakeSample:

    def __init__(self, stage_size_nm: float):
        self.__features = list()
        sample_size_m = Geometry.FloatSize(height=20 * stage_size_nm / 100, width=20 * stage_size_nm / 100) / 1E9
        feature_percentage = 0.3
        random_state = random.getstate()
        random.seed(1)
        energies = [[(68, 30), (855, 50), (872, 50)], [(29, 15), (1217, 50), (1248, 50)],
                    [(1839, 5), (99, 50)]]  # Ni, Ge, Si
        plasmons = [20, 16.2, 16.8]
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height,
                                             x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height,
                                                             width=random.random() * sample_size_m.width)
            self.__features.append(
                Feature(position_m, size_m, energies[i % len(energies)], plasmons[i % len(plasmons)], 4))
        random.setstate(random_state)

    @property
    def title(self) -> str:
        return _("Flake")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: numpy.ndarray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize,
                      extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint,
                      used_size: Geometry.IntSize) -> None:
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)


class AmorphousSample(Sample):

    def __init__(self):
        self.__amorphous = numpy.random.RandomState(1).randn(2048, 2048) * 2 + 1

    @property
    def title(self) -> str:
        return "Amorphous"

    @property
    def features(self) -> typing.List[Feature]:
        return list()

    def plot_features(self, data: numpy.ndarray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize,
                      extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint,
                      used_size: Geometry.IntSize) -> None:
        range_nm = 80

        # print(f"{offset_m * 1E9}, {fov_size_nm} {extra_nm}")

        # calculate destination bounds in nm
        left_nm = -offset_m.x * 1E9 - (fov_size_nm.width + extra_nm.x) / 2
        top_nm = -offset_m.y * 1E9 - (fov_size_nm.height + extra_nm.y) / 2
        right_nm = left_nm + fov_size_nm.width + extra_nm.x
        bottom_nm = top_nm + fov_size_nm.height + extra_nm.y

        # print(f"{left_nm}, {top_nm} x {right_nm - left_nm}, {bottom_nm - top_nm}")

        # map into fractional coordinates (0, 1) of source area where (-range_nm, range_nm) is the range in both axes
        intersection_left_nm = max(left_nm, -range_nm)
        intersection_top_nm = max(top_nm, -range_nm)
        intersection_right_nm = min(right_nm, range_nm)
        intersection_bottom_nm = min(bottom_nm, range_nm)

        # print(f"{intersection_left_nm}, {intersection_top_nm} x {intersection_right_nm - intersection_left_nm}, {intersection_bottom_nm - intersection_top_nm}")

        if intersection_left_nm < intersection_right_nm and intersection_top_nm < intersection_bottom_nm:
            src_left = int(self.__amorphous.shape[1] * max((intersection_left_nm + range_nm) / (range_nm * 2), 0))
            src_top = int(self.__amorphous.shape[0] * max((intersection_top_nm + range_nm) / (range_nm * 2), 0))
            src_right = int(self.__amorphous.shape[1] * min((intersection_right_nm + range_nm) / (range_nm * 2), 1))
            src_bottom = int(self.__amorphous.shape[0] * min((intersection_bottom_nm + range_nm) / (range_nm * 2), 1))
            dst_left = int(data.shape[1] * max((intersection_left_nm - left_nm) / (right_nm - left_nm), 0))
            dst_top = int(data.shape[0] * max((intersection_top_nm - top_nm) / (bottom_nm - top_nm), 0))
            dst_right = int(data.shape[1] * min((intersection_right_nm - left_nm) / (right_nm - left_nm), 1))
            dst_bottom = int(data.shape[0] * min((intersection_bottom_nm - top_nm) / (bottom_nm - top_nm), 1))

            # print(f"{src_left}, {src_top} x {src_right - src_left}, {src_bottom - src_top} => {dst_left}, {dst_top} x {dst_right - dst_left}, {dst_bottom - dst_top}")

            src = self.__amorphous[src_top:src_bottom, src_left:src_right]
            src = scipy.ndimage.gaussian_filter(src, 3)
            # may be faster, but doesn't work for non-square src
            # src = numpy.fft.ifft2(scipy.ndimage.fourier_gaussian(src * 1j, 3)).real
            src = 4 * (src - numpy.amin(src)) / numpy.ptp(src)
            data[dst_top:dst_bottom, dst_left:dst_right] = Image.scaled(src,
                                                                        (dst_bottom - dst_top, dst_right - dst_left))

            # print(f"src min/max {numpy.amin(src)} / {numpy.amax(src)}")

        # print(f"data min/max {numpy.amin(data)} / {numpy.amax(data)}")


class LabelledPoints:

    def __init__(self, positions=None, cell=None, labels=None, dimensions=2):
        if positions is None:
            positions = np.zeros((0, dimensions), dtype=np.float)

        positions = np.array(positions, dtype=np.float)

        if (len(positions.shape) != dimensions) | (positions.shape[1] != dimensions):
            raise RuntimeError()

        self._positions = positions

        if labels is None:
            labels = np.zeros(len(positions), dtype=np.int)
        else:
            labels = np.array(labels).astype(np.int)

        self._labels = labels

        self._cell = np.zeros((dimensions, dimensions), np.float)

        if cell is not None:
            self._cell[:] = cell

    def __len__(self):
        return len(self.positions)

    @property
    def positions(self):
        return self._positions

    @property
    def labels(self):
        return self._labels

    @property
    def cell(self):
        return self._cell

    def repeat(self, n, m):
        N = len(self)

        n0, n1 = 0, n
        m0, m1 = 0, m
        new_positions = np.zeros((n * m * N, 2), dtype=np.float)

        positions = self.positions.copy()
        new_positions[:N] = self.positions

        k = N
        for i in range(n0, n1):
            for j in range(m0, m1):
                if i + j != 0:
                    l = k + N
                    new_positions[k:l] = positions + np.dot((i, j), self.cell)
                    k = l

        labels = np.tile(self.labels, (n * m,))
        cell = self.cell.copy() * (n, m)
        return LabelledPoints(new_positions, cell=cell, labels=labels)


def fill_rectangle(points, extent, origin=None, margin=0., eps=1e-12):
    if origin is None:
        origin = np.zeros(2)
    else:
        origin = np.array(origin)

    extent = np.array(extent)
    original_cell = points.cell.copy()

    P_inv = np.linalg.inv(original_cell)

    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower_corner = np.dot(origin_t, original_cell)
    upper_corner = lower_corner + extent

    corners = np.array([[-margin - eps, -margin - eps],
                        [upper_corner[0] + margin + eps, -margin - eps],
                        [upper_corner[0] + margin + eps, upper_corner[1] + margin + eps],
                        [-margin - eps, upper_corner[1] + margin + eps]])

    n0, m0 = 0, 0
    n1, m1 = 0, 0
    for corner in corners:
        new_n, new_m = np.ceil(np.dot(corner, P_inv)).astype(np.int)
        n0 = max(n0, new_n)
        m0 = max(m0, new_m)
        new_n, new_m = np.floor(np.dot(corner, P_inv)).astype(np.int)
        n1 = min(n1, new_n)
        m1 = min(m1, new_m)

    repeated = points.repeat(1 + n0 - n1, 1 + m0 - m1)

    positions = repeated.positions.copy()

    positions = positions + original_cell[0] * n1 + original_cell[1] * m1

    inside = ((positions[:, 0] > lower_corner[0] - eps - margin) &
              (positions[:, 1] > lower_corner[1] - eps - margin) &
              (positions[:, 0] < upper_corner[0] + margin) &
              (positions[:, 1] < upper_corner[1] + margin))
    new_positions = positions[inside] - lower_corner
    labels = repeated.labels[inside]

    return LabelledPoints(new_positions, cell=extent, labels=labels)


def gaussians(points, width, gpts):
    if isinstance(gpts, numbers.Number):
        gpts = (gpts,) * 2

    gpts = np.array(gpts)
    extent = np.diag(points.cell)
    sampling = extent / gpts
    markers = np.zeros(gpts)

    r = np.linspace(0, 4 * width, 100)
    values = np.exp(-r ** 2 / (2 * width ** 2))
    interpolate_radial_functions(markers, r, values, points.positions[points.labels == 0], sampling)
    interpolate_radial_functions(markers, r, 4.6*values, points.positions[points.labels == 1], sampling)
    #interpolate_radial_functions(markers, r, values, points.positions, sampling)
    return markers


class GrapheneSample(Sample):

    def __init__(self, instrument):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'graphene.npz')
        npzfile = np.load(path)
        self._points = LabelledPoints(positions=npzfile['positions'], labels=npzfile['labels'], cell=npzfile['cell'])
        self._instrument = instrument

    @property
    def title(self) -> str:
        return "Graphene"

    @property
    def features(self) -> typing.List[Feature]:
        return list()

    def plot_features(self, data: np.ndarray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize,
                      extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint,
                      used_size: Geometry.IntSize) -> None:
        left_nm = -offset_m.x * 1e9 - (fov_size_nm.width + extra_nm.x) / 2.
        top_nm = -offset_m.y * 1e9 - (fov_size_nm.height + extra_nm.y) / 2.
        right_nm = left_nm + fov_size_nm.width + extra_nm.x
        bottom_nm = top_nm + fov_size_nm.height + extra_nm.y

        extent = np.array([bottom_nm - top_nm, right_nm - left_nm])
        origin = np.array([-offset_m.y, -offset_m.x]) * 1e9

        points = fill_rectangle(self._points, extent, origin, 2)

        image = gaussians(points, .5, np.array(data.shape))

        drift = self._instrument.GetVal2D('Drift')
        drift_x = drift.x
        drift_y = drift.y

        self._instrument.change_stage_position(dy = drift_y, dx = drift_x)
        
        # superposition = gaussian_superposition(self.__positions, np.array(data.shape), origin, extent, .05)
        data[:, :] = image * self._instrument.GetVal("BeamCurrent") / 4e-10  # superposition / 2
