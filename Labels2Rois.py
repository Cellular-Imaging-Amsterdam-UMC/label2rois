#!/usr/bin/env python
# -*- coding: utf-8 -*-

import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.rtypes import rstring, rlong
import ezomero as ez
import numpy as np
from omero.cmd import Delete2
from omero.model import RoiI
import time
from collections import deque
import sys

try:
    from skimage.measure import find_contours
except ImportError:
    print("!! Could not find python package 'scikit-image' !!\n+++++++++++++++++")

try:
    import omero_rois as omeroi
except ImportError:
    print("!! Could not find python package 'omero_rois' !!\n+++++++++++++++++")


def labels2rois(script_params, conn):
    """
    Main function to convert label images to ROIs.
    
    Args:
        script_params: Dictionary containing script parameters
        conn: OMERO connection object
    
    Returns:
        Tuple of (newRois, imagesProcessed)
    """
    input_type = script_params["Data_Type"]
    input_ids = script_params["IDs"]
    delete_label_image = script_params["Delete_Label_Image"]
    algorithm = script_params["ROI_type"]
    
    label_suffix = script_params.get("Label_Suffix", "-label")
    label_dataset_id = script_params.get("Label_Dataset_ID", None)
    search_mode = script_params.get("Search_Mode", "Same Dataset")
    clear_rois = script_params.get("Clear_Existing_ROIs", False)
    clear_filter = script_params.get("Clear_ROI_Filter", "")

    new_rois = []
    images_processed = 0

    if input_type == "Dataset":
        new_rois, images_processed = process_dataset_input(
            input_ids, conn, label_suffix, label_dataset_id, search_mode, 
            clear_rois, clear_filter, algorithm, delete_label_image
        )
    elif input_type == "Image":
        new_rois, images_processed = process_image_input(
            input_ids, conn, label_suffix, label_dataset_id, search_mode,
            clear_rois, clear_filter, algorithm, delete_label_image
        )

    return new_rois, images_processed


def process_dataset_input(input_ids, conn, label_suffix, label_dataset_id, search_mode, 
                         clear_rois, clear_filter, algorithm, delete_label_image):
    """Process dataset input to create ROIs from label images."""
    new_rois = []
    images_processed = 0

    for dataset_id in input_ids:
        dataset = conn.getObject("Dataset", dataset_id)
        dataset_images = list(dataset.listChildren())

        # Always build image_dict from the target dataset (where we want to create ROIs)
        image_dict = build_image_dict(dataset_images, label_suffix)
        if image_dict:
            print("image dict:\n", image_dict)

        # Get label images from the search dataset
        search_images = get_search_dataset_images(dataset, label_dataset_id, search_mode, conn)
        label_images = [img for img in search_images if is_label_image(img.name, label_suffix)]
        
        print(f"Found {len(label_images)} label images: {[img.name for img in label_images]}")

        # Group labels by target - this now works across datasets
        target_to_labels = group_labels_by_target(label_images, image_dict, label_suffix)
        
        print(f"Grouped into {len(target_to_labels)} target groups")

        for target_id, target_labels in target_to_labels.items():
            print("--------------------------------------")
            print(f"Processing target image ID {target_id} with {len(target_labels)} label image(s)")
            
            # Get target image object
            target_image = conn.getObject("Image", target_id)
            
            if clear_rois:
                filter_to_use = clear_filter if clear_filter.strip() else None
                cleared_count = clear_existing_rois(target_id, conn, filter_to_use)
                print(f"Cleared {cleared_count} existing ROIs from target image {target_id}")

            for label_image in target_labels:
                print(f"processing label image '{label_image.name}' from Dataset '{label_image.getAncestry()[0].name}'")

                plane = get_label_image_as_array(label_image)
                print(f"shape of plane z=0/t=0/c=0: {plane.shape}")
                print(f"min: {plane.min()}, max: {plane.max()}, pixel type: {plane.dtype.name}")

                contour_dict, contour_time = create_contours(plane, algorithm)
                created_rois, roi_time = upload_rois(contour_dict, target_id, algorithm, conn, label_image, target_image, label_suffix)
                new_rois.extend(created_rois)
                images_processed += 1

                if delete_label_image:
                    delete_image(label_image, conn)

                print(f"{int(contour_time)}s to create contours and {int(roi_time)}s to upload ROIs")

    return new_rois, images_processed


def process_image_input(input_ids, conn, label_suffix, label_dataset_id, search_mode,
                       clear_rois, clear_filter, algorithm, delete_label_image):
    """Process individual image input to create ROIs from corresponding label images."""
    new_rois = []
    images_processed = 0

    for image_id in input_ids:
        target_image = conn.getObject("Image", image_id)
        dataset = target_image.getAncestry()[0]
        dataset_images = list(dataset.listChildren())

        image_dict = build_image_dict(dataset_images, label_suffix, exclude_id=image_id)
        if image_dict:
            print("image dict:\n", image_dict)

        search_images = get_search_dataset_images(dataset, label_dataset_id, search_mode, conn)
        label_image = find_most_precise_label_image(target_image, search_images, label_suffix)

        if not label_image:
            print(f"!! Warning: No label image found for '{target_image.name}' !!")
            continue

        print("--------------------------------------")
        print(f"processing label image '{label_image.name}' for target '{target_image.name}' from Dataset '{dataset.name}'")

        if clear_rois:
            filter_to_use = clear_filter if clear_filter.strip() else None
            clear_existing_rois(target_image.id, conn, filter_to_use)

        roi_count = process_single_label_image(
            label_image, target_image.id, algorithm, conn, label_suffix, delete_label_image
        )
        new_rois.extend(roi_count)
        images_processed += 1

    return new_rois, images_processed


def process_single_label_image(label_image, target_id, algorithm, conn, label_suffix, delete_label_image):
    """Process a single label image to create ROIs."""
    print(f"processing label image '{label_image.name}' from Dataset '{label_image.getAncestry()[0].name}'")

    plane = get_label_image_as_array(label_image)
    print(f"shape of plane z=0/t=0/c=0: {plane.shape}")
    print(f"min: {plane.min()}, max: {plane.max()}, pixel type: {plane.dtype.name}")

    contour_dict, contour_time = create_contours(plane, algorithm)
    
    # Get the target image object for ROI naming
    target_image = conn.getObject("Image", target_id)
    created_rois, roi_time = upload_rois(contour_dict, target_id, algorithm, conn, label_image, target_image, label_suffix)

    if delete_label_image:
        delete_image(label_image, conn)

    print(f"{int(contour_time)}s to create contours and {int(roi_time)}s to upload ROIs")
    return created_rois


def build_image_dict(dataset_images, label_suffix, exclude_id=None):
    """Build dictionary of non-label images."""
    image_dict = {}
    for img in dataset_images:
        if exclude_id and img.id == exclude_id:
            continue
        if not is_label_image(img.name, label_suffix):
            image_dict[img.name] = img.id
    return image_dict


def group_labels_by_target(label_images, image_dict, label_suffix):
    """Group label images by their target image to avoid multiple ROI clearing."""
    target_to_labels = {}
    for label_image in label_images:
        print(f"DEBUG: Trying to match label '{label_image.name}' with suffix '{label_suffix}'")
        target_id = get_target_image_id(image_dict, label_image.name, label_suffix)
        print(f"DEBUG: Found target_id: {target_id}")
        if target_id != 0:
            if target_id not in target_to_labels:
                target_to_labels[target_id] = []
            target_to_labels[target_id].append(label_image)
    return target_to_labels


def get_search_dataset_images(target_dataset, label_dataset_id, search_mode, conn):
    """Get images from the appropriate dataset based on search mode."""
    if search_mode == "Same Dataset":
        return list(target_dataset.listChildren())
    elif search_mode == "Specific Dataset" and label_dataset_id:
        label_dataset = conn.getObject("Dataset", label_dataset_id)
        if label_dataset:
            return list(label_dataset.listChildren())
        else:
            print(f"!! Error: Label dataset with ID {label_dataset_id} not found !!")
            return []
    return []


def find_most_precise_label_image(target_image, dataset_images, label_suffix):
    """
    Find the most precise label image for a target image.
    Returns the label image with the shortest name that matches the target.
    """
    target_base = remove_extension(target_image.name)
    label_candidates = []

    for img in dataset_images:
        if is_label_image(img.name, label_suffix) and img.name.startswith(target_base):
            label_candidates.append(img)

    if not label_candidates:
        return None

    label_candidates.sort(key=lambda x: len(x.name))

    if len(label_candidates) > 1:
        print(f"Found {len(label_candidates)} label candidates for '{target_image.name}':")
        for candidate in label_candidates:
            print(f"  - {candidate.name}")
        print(f"Selected most precise: {label_candidates[0].name}")

    return label_candidates[0]


def get_target_image_id(image_dict, label_name, label_suffix="-label"):
    """
    Find target image ID for a given label image name.
    Prioritizes precision while maintaining recall.
    """
    label_base = remove_extension(label_name)
    possible_targets = []
    temp_name = label_base
    
    while label_suffix in temp_name:
        suffix_index = temp_name.rfind(label_suffix)
        target_base = temp_name[:suffix_index]
        if target_base and target_base not in possible_targets:
            possible_targets.append(target_base)
        temp_name = target_base

    if not possible_targets:
        return 0

    valid_matches = []
    for name, img_id in image_dict.items():
        target_base = remove_extension(name)
        if target_base in possible_targets:
            valid_matches.append((target_base, name, img_id))

    if not valid_matches:
        return 0

    valid_matches.sort(key=lambda x: len(x[0]), reverse=True)
    best_match = valid_matches[0]

    print(f"found matching image '{best_match[1]}' for label '{label_name}' (base: '{best_match[0]}')")
    return best_match[2]


def remove_extension(filename):
    """Get basename without any extension."""
    return filename[:filename.index('.')] if '.' in filename else filename


def is_label_image(name, label_suffix="-label"):
    """Check if image name indicates it's a label image."""
    return label_suffix in name


def clear_existing_rois(image_id, conn, roi_name_filter=None):
    """Clear existing ROIs from an image."""
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(image_id, None)

    if not result or not result.rois:
        return 0

    rois_to_delete = []
    for roi in result.rois:
        roi_name = roi.name.val if roi.name else ""
        if roi_name_filter is None or roi_name_filter in roi_name:
            rois_to_delete.append(roi.id.val)

    if rois_to_delete:
        delete = Delete2(targetObjects={"Roi": rois_to_delete})
        conn.c.submit(delete, loops=5, ms=2000)

    return len(rois_to_delete)


def get_label_image_as_array(image):
    """Get the image as a numpy array."""
    z, t, c = 0, 0, 0
    pixels = image.getPrimaryPixels()
    return pixels.getPlane(z, c, t)


def delete_image(image, conn):
    """Delete an image from OMERO."""
    delete = Delete2(targetObjects={"Image": [image.id]})
    conn.c.submit(delete, loops=5, ms=2000)


def get_cropped_mask(mask):
    """Get a cropped sub-mask from a boolean mask."""
    xmask = mask.sum(0).nonzero()[0]
    ymask = mask.sum(1).nonzero()[0]
    x0 = max(0, min(xmask) - 1)
    w = max(xmask) - x0 + 2
    y0 = max(0, min(ymask) - 1)
    h = max(ymask) - y0 + 2
    return mask[y0:(y0 + h), x0:(x0 + w)], x0, y0


def create_contours(label_image, algorithm):
    """Create contours that will be uploaded as ROIs from label images."""
    contour_dict = {}
    start = time.time()
    
    if algorithm == "Mask":
        contour_dict = create_mask_contours(label_image)
    elif algorithm == "Polygon":
        contour_dict = create_polygon_contours(label_image)
    
    contour_time = time.time() - start
    return contour_dict, contour_time


def create_mask_contours(label_image):
    """Create mask-based contours."""
    contour_dict = {}
    for i in range(1, label_image.max() + 1):
        mask = (label_image == i)
        contour = omeroi.mask_from_binary_image(mask, text=str(i))
        contour_dict[i] = contour
    
    assert len(contour_dict) == label_image.max(), \
        f"Expected {label_image.max()} ROIs, found {len(contour_dict)}."
    
    return contour_dict


def create_polygon_contours(label_image):
    """Create polygon-based contours."""
    contour_dict = {}
    multiple_contours = []
    
    for i in range(1, label_image.max() + 1):
        mask = (label_image == i)
        cropped, x_offset, y_offset = get_cropped_mask(mask)
        
        assert np.array_equal(np.unique(cropped), [0, 1]), \
            "The cropped array does not contain both 0s and 1s"
        
        if "skimage" in sys.modules:
            contours = find_contours(cropped, level=0)
        else:
            contours = own_find_contours(cropped)
        
        if len(contours) > 1:
            multiple_contours.append(i)
            contours = sorted(contours, key=len, reverse=True)
            overlength_contours = [len(contour) for contour in contours[1:] if len(contour) >= 6]
            if overlength_contours:
                print(f"    for grey value {i} found {len(overlength_contours)} 'overlength' contour(s) with length(s): {overlength_contours}")
        
        contour_dict[i] = [[x + y_offset, y + x_offset] for [x, y] in contours[0]]
    
    if multiple_contours:
        print(f"found multiple contours for the grey value(s) {multiple_contours}")
    
    return contour_dict


def upload_rois(contour_dict, parent_id, algorithm, conn, label_image, target_image, label_suffix="-label"):
    """Upload ROIs to OMERO."""
    start = time.time()
    new_rois = []
    
    # Get unique prefix from label filename
    roi_prefix = get_roi_name_prefix(label_image.name, target_image.name, label_suffix)

    if algorithm == "Mask":
        new_rois = upload_mask_rois(contour_dict, parent_id, conn, roi_prefix)
    elif algorithm == "Polygon":
        new_rois = upload_polygon_rois(contour_dict, parent_id, conn, roi_prefix)

    roi_time = time.time() - start
    print(f"created new Rois: {new_rois}")
    return new_rois, roi_time


def upload_mask_rois(contour_dict, parent_id, conn, clean_suffix):
    """Upload mask-based ROIs."""
    new_rois = []
    update = conn.getUpdateService()
    
    for grey_value, shape in contour_dict.items():
        roi = RoiI()
        roi.name = rstring(f"{clean_suffix}_{grey_value}")
        roi.image = conn.getObject("Image", parent_id)._obj
        roi.addShape(shape)
        roi = update.saveAndReturnObject(roi)
        new_rois.append(roi.id.val)
    
    return new_rois


def upload_polygon_rois(contour_dict, parent_id, conn, clean_suffix):
    """Upload polygon-based ROIs."""
    new_rois = []
    
    for grey_value, coordinates in contour_dict.items():
        flipped = np.flip(coordinates)
        roi_name = f"{clean_suffix}_{grey_value}"
        shape = [ez.rois.Polygon(flipped, label=roi_name)]
        roi_id = ez.post_roi(conn, int(parent_id), shape, name=roi_name)
        new_rois.append(roi_id)
    
    return new_rois


def own_find_contours(image):
    """Custom implementation of skimage.measure.find_contours()."""
    segments = _get_contour_segments(image.astype(np.float64))
    contours = _assemble_contours(segments)
    return contours


def _get_fraction(from_value, to_value):
    """Calculate fraction for contour interpolation."""
    return 0 if to_value == from_value else (0 - from_value) / (to_value - from_value)


def _get_contour_segments(array):
    """Get contour segments from array."""
    segments = []
    
    for r0 in range(array.shape[0] - 1):
        for c0 in range(array.shape[1] - 1):
            r1, c1 = r0 + 1, c0 + 1
            
            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]
            
            square_case = (ul > 0) + 2 * (ur > 0) + 4 * (ll > 0) + 8 * (lr > 0)
            
            if square_case in [0, 15]:
                continue
            
            top = r0, c0 + _get_fraction(ul, ur)
            bottom = r1, c0 + _get_fraction(ll, lr)
            left = r0 + _get_fraction(ul, ll), c0
            right = r0 + _get_fraction(ur, lr), c1
            
            segment_map = {
                1: (top, left), 2: (right, top), 3: (right, left), 4: (left, bottom),
                5: (top, bottom), 6: [(right, top), (left, bottom)], 7: (right, bottom),
                8: (bottom, right), 9: [(top, left), (bottom, right)], 10: (bottom, top),
                11: (bottom, left), 12: (left, right), 13: (top, right), 14: (left, top)
            }
            
            segment = segment_map.get(square_case)
            if isinstance(segment, list):
                segments.extend(segment)
            else:
                segments.append(segment)
    
    return segments


def _assemble_contours(segments):
    """Assemble contour segments into complete contours."""
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    
    for from_point, to_point in segments:
        if from_point == to_point:
            continue
        
        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))
        
        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            else:
                if tail_num > head_num:
                    head.extend(tail)
                    contours.pop(tail_num, None)
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:
                    tail.extendleft(reversed(head))
                    starts.pop(head[0], None)
                    contours.pop(head_num, None)
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)
    
    return [np.array(contour) for _, contour in sorted(contours.items())]


def get_roi_name_prefix(label_name, target_name, label_suffix):
    """
    Get a unique ROI name prefix based on the label filename.
    For label 'test_nuc_cells.tif' with target 'test.tif', returns 'nuc_cells'
    For label 'test_nuc.tif' with target 'test.tif', returns 'nuc'
    """
    label_base = remove_extension(label_name)
    target_base = remove_extension(target_name)
    
    # Remove the target base from the beginning
    if label_base.startswith(target_base):
        unique_part = label_base[len(target_base):]
        # Remove leading underscores or other separators
        unique_part = unique_part.lstrip('_-.')
        return unique_part if unique_part else 'label'
    
    # Fallback to removing suffix
    if label_suffix in label_base:
        return label_base.replace(label_suffix, '').strip('_-.')
    
    return 'label'


def run_script():
    """Main script entry point."""
    data_types = [rstring('Dataset'), rstring('Image')]
    shape_types = [rstring("Polygon"), rstring("Mask")]
    search_modes = [rstring("Same Dataset"), rstring("Specific Dataset")]

    client = scripts.client(
        'Labels2Rois',
        """
        Creates (named) ROIs from label images.
        
        For correct mapping of the ROIs, the label image must have
        the same name as the target image and contain the specified suffix
        (default: '-label'). Label images can be in the same dataset
        or a specific dataset. Optionally clear existing ROIs.
        """,

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose source of label images",
            values=data_types, default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of IDs").ofType(rlong(0)),

        scripts.String(
            "ROI_type", optional=False, grouping="3",
            description="Select 'Polygon' or 'Mask'. A 'Mask' shape will cover the segmented region, a 'Polygon' will create an outline around it. It also determines which algorithm will be used. The 'Mask' algorithm is faster if the ROIs do not touch.",
            values=shape_types, default="Polygon"),

        scripts.String(
            "Search_Mode", optional=False, grouping="4",
            description="Where to search for label images",
            values=search_modes, default="Same Dataset"),

        scripts.Long(
            "Label_Dataset_ID", optional=True, grouping="4.1",
            description="Dataset ID for label images (when using 'Specific Dataset')"),

        scripts.String(
            "Label_Suffix", optional=True, grouping="5", default="-label",
            description="Suffix that identifies label images (default: '-label')"),

        scripts.Bool(
            "Clear_Existing_ROIs", optional=False, grouping="6", default=False,
            description="Delete existing ROIs before adding new ones"),

        scripts.String(
            "Clear_ROI_Filter", optional=True, grouping="6.1", default="",
            description="Only delete ROIs containing this text (leave empty for all)"),

        scripts.Bool(
            "Delete_Label_Image", optional=False, grouping="7", default=False,
            description="Delete the label image(s) after conversion to ROIs is complete"),

        authors=["Jens Wendt"],
        contact="https://forum.image.sc/tag/omero, jens.wendt@uni-muenster.de",
        version="0.5"
    )

    try:
        script_params = client.getInputs(unwrap=True)
        conn = BlitzGateway(client_obj=client)
        new_rois, images_processed = labels2rois(script_params, conn)
        message = f"created {len(new_rois)} ROIs in {images_processed} images"
        client.setOutput("Message", rstring(message))
    finally:
        client.closeSession()


if __name__ == "__main__":
    run_script()
