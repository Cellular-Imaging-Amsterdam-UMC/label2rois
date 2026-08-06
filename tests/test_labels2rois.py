import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

# Importing omero-py on Windows eagerly loads thousands of generated classes.
# These unit tests exercise pure mapping/plane logic, so provide the tiny API
# surface needed to import the script instead of booting a full OMERO client.
omero = ModuleType("omero")
omero_scripts = ModuleType("omero.scripts")
omero_gateway = ModuleType("omero.gateway")
omero_rtypes = ModuleType("omero.rtypes")
omero_cmd = ModuleType("omero.cmd")
omero_model = ModuleType("omero.model")
omero_gateway.BlitzGateway = Mock
omero_rtypes.rint = lambda value: value
omero_rtypes.rstring = lambda value: value
omero_rtypes.rlong = lambda value: value
omero_cmd.Delete2 = Mock
omero_model.RoiI = Mock
omero.model = omero_model
sys.modules.update({
    "omero": omero,
    "omero.scripts": omero_scripts,
    "omero.gateway": omero_gateway,
    "omero.rtypes": omero_rtypes,
    "omero.cmd": omero_cmd,
    "omero.model": omero_model,
    "ezomero": ModuleType("ezomero"),
    "omero_rois": ModuleType("omero_rois"),
})

import Labels2Rois as labels


def test_get_label_values_supports_sparse_labels():
    image = np.array([[0, 2, 2], [1000, 0, 1000]], dtype=np.uint16)

    assert labels.get_label_values(image) == [2, 1000]


def test_explicit_pair_lists_must_have_equal_lengths():
    with pytest.raises(ValueError, match="same number"):
        labels.process_explicit_image_pairs(
            [11, 12], [21], Mock(), False, "", "Polygon", False
        )


def test_explicit_pairs_continue_after_pair_failure(monkeypatch):
    images = {
        11: Mock(name="label-11"),
        12: Mock(name="label-12"),
        21: Mock(name="target-21"),
        22: Mock(name="target-22"),
    }
    conn = Mock()
    conn.getObject.side_effect = lambda object_type, image_id: images.get(image_id)

    def process(label_image, target_id, *args, **kwargs):
        if target_id == 21:
            raise RuntimeError("bad label image")
        return [901, 902]

    monkeypatch.setattr(labels, "process_single_label_image", process)

    rois, processed, warnings = labels.process_explicit_image_pairs(
        [11, 12], [21, 22], conn, False, "", "Polygon", False
    )

    assert rois == [901, 902]
    assert processed == 1
    assert len(warnings) == 1
    assert "11 -> target image 21" in warnings[0]


def test_explicit_pairs_forward_provenance_prefix(monkeypatch):
    images = {
        11: Mock(name="label-11"),
        21: Mock(name="target-21"),
    }
    conn = Mock()
    conn.getObject.side_effect = lambda object_type, image_id: images.get(image_id)
    calls = []

    def process(*args, **kwargs):
        calls.append((args, kwargs))
        return [901]

    monkeypatch.setattr(labels, "process_single_label_image", process)

    rois, processed, warnings = labels.process_explicit_image_pairs(
        [11], [21], conn, False, "", "Polygon", False,
        "cellpose__11111111-2222-3333-4444-555555555555",
    )

    assert rois == [901]
    assert processed == 1
    assert warnings == []
    assert calls[0][1]["roi_name_prefix"] == (
        "cellpose__11111111-2222-3333-4444-555555555555"
    )


def test_process_single_label_image_preserves_z_and_t(monkeypatch):
    planes = {
        (0, 0): np.array([[0, 1], [0, 0]]),
        (0, 1): np.array([[0, 0], [2, 0]]),
        (1, 0): np.array([[3, 0], [0, 0]]),
        (1, 1): np.array([[0, 4], [0, 0]]),
    }
    label_pixels = Mock()
    label_pixels.getSizeZ.return_value = 2
    label_pixels.getSizeT.return_value = 2
    label_pixels.getPlane.side_effect = lambda z, c, t: planes[(z, t)]
    target_pixels = Mock()
    target_pixels.getSizeZ.return_value = 2
    target_pixels.getSizeT.return_value = 2

    label_image = Mock()
    label_image.name = "source-label.tif"
    label_image.getPrimaryPixels.return_value = label_pixels
    target_image = Mock()
    target_image.name = "source.tif"
    target_image.getPrimaryPixels.return_value = target_pixels
    conn = Mock()
    conn.getObject.return_value = target_image

    uploads = []
    monkeypatch.setattr(
        labels,
        "create_contours",
        lambda plane, algorithm: ({int(plane.max()): [[0, 0], [1, 0], [1, 1]]}, 0),
    )

    def upload(*args, **kwargs):
        uploads.append((kwargs["z"], kwargs["t"]))
        return [100 + len(uploads)], 0

    monkeypatch.setattr(labels, "upload_rois", upload)

    rois = labels.process_single_label_image(
        label_image, 21, "Polygon", conn, "-label", False
    )

    assert uploads == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert rois == [101, 102, 103, 104]


def test_process_single_label_image_uses_provenance_prefix(monkeypatch):
    pixels = Mock()
    pixels.getSizeZ.return_value = 1
    pixels.getSizeT.return_value = 1
    pixels.getPlane.return_value = np.array([[0, 1], [0, 0]])
    label_image = Mock()
    label_image.name = "source-label.tif"
    label_image.getPrimaryPixels.return_value = pixels
    target_image = Mock()
    target_image.name = "source.tif"
    target_image.getPrimaryPixels.return_value = pixels
    conn = Mock()
    conn.getObject.return_value = target_image

    monkeypatch.setattr(
        labels,
        "create_contours",
        lambda plane, algorithm: ({1: [[0, 0], [1, 0], [1, 1]]}, 0),
    )
    prefixes = []

    def upload(*args, **kwargs):
        prefixes.append(kwargs["roi_name_prefix"])
        return [101], 0

    monkeypatch.setattr(labels, "upload_rois", upload)

    labels.process_single_label_image(
        label_image, 21, "Polygon", conn, "-label", False,
        roi_name_prefix="cellpose__run-uuid",
    )

    assert prefixes == ["cellpose__run-uuid"]


def test_upload_rois_combines_provenance_and_label_prefix(monkeypatch):
    label_image = Mock()
    label_image.name = "source-label.tif"
    label_image.getPrimaryPixels.return_value.getSizeZ.return_value = 1
    label_image.getPrimaryPixels.return_value.getSizeT.return_value = 1
    target_image = Mock()
    target_image.name = "source.tif"
    prefixes = []

    def upload(contours, parent_id, conn, clean_suffix, z, t):
        prefixes.append(clean_suffix)
        return [501]

    monkeypatch.setattr(labels, "upload_polygon_rois", upload)

    roi_ids, _ = labels.upload_rois(
        {1: [[0, 0], [1, 0], [1, 1]]},
        21,
        "Polygon",
        Mock(),
        label_image,
        target_image,
        roi_name_prefix="cellpose__run-uuid",
    )

    assert prefixes == ["cellpose__run-uuid__label"]
    assert roi_ids == [501]


def test_mask_shapes_are_attached_to_the_processed_plane(monkeypatch):
    shape = Mock()
    saved_roi = Mock()
    saved_roi.id.val = 501
    update = Mock()
    update.saveAndReturnObject.return_value = saved_roi
    target = Mock()
    conn = Mock()
    conn.getUpdateService.return_value = update
    conn.getObject.return_value = target
    monkeypatch.setattr(labels, "RoiI", Mock)

    roi_ids = labels.upload_mask_rois(
        {7: shape}, 21, conn, "cells", z=2, t=3
    )

    shape.setTheZ.assert_called_once_with(2)
    shape.setTheT.assert_called_once_with(3)
    shape.setTheC.assert_called_once_with(0)
    assert roi_ids == [501]
