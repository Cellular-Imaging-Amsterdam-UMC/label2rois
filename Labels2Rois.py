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
# try imports of "extra" packages
try:
    from skimage.measure import find_contours
except:
    print("!! Could not find python package 'scikit-image' !!\n+++++++++++++++++")
try:
    import omero_rois as omeroi
except:
    print("!! Could not find python package 'omero_rois' !!\n+++++++++++++++++")

# main function
def labels2rois(script_params, conn):
    inputType = script_params["Data_Type"]
    inputIds = script_params["IDs"]
    deleteLabelImage = script_params["Delete_Label_Image"]
    algorithm = script_params["ROI_type"]
    
    # New parameters
    label_suffix = script_params.get("Label_Suffix", "-label")
    label_dataset_id = script_params.get("Label_Dataset_ID", None)
    search_mode = script_params.get("Search_Mode", "Same Dataset")
    clear_rois = script_params.get("Clear_Existing_ROIs", False)
    clear_filter = script_params.get("Clear_ROI_Filter", "")

    newRois = []
    imagesProcessed = 0
    imageDict = {}

    if inputType == "Dataset":
        for id in inputIds:
            dataset = conn.getObject("Dataset", id)
            # get the image dict
            for img in list(dataset.listChildren()):
                if not isLabelImage(img.name, label_suffix):
                    imageDict[img.name] = img.id
            if len(imageDict)>0:
                print("image dict:\n",imageDict)
                
            # Get label images based on search mode
            if search_mode == "Same Dataset":
                label_images = [img for img in list(dataset.listChildren()) if isLabelImage(img.name, label_suffix)]
            elif search_mode == "Specific Dataset" and label_dataset_id:
                label_dataset = conn.getObject("Dataset", label_dataset_id)
                label_images = [img for img in list(label_dataset.listChildren()) if isLabelImage(img.name, label_suffix)] if label_dataset else []
            else:
                label_images = []
                
            # main loop
            for image in label_images:
                # get label image as numpy array
                plane = get_label_image_as_array(image)
                print("--------------------------------------")
                print(f"processing image '{image.name}' from Dataset '{image.getAncestry()[0].name}'")
                print(f"shape of plane z=0/t=0/c=0: ", plane.shape)
                print("min:", plane.min(), " max:", plane.max(),\
                    "pixel type:", plane.dtype.name)
                # get the contours
                contour_dict, contourTime = create_contours(plane, algorithm)
                # find the target image
                targetId = get_target_image_id(imageDict, image.name, label_suffix)
                if targetId == 0:
                    print(f"!! Warning: No matching target image found for label '{image.name}' !!")
                # upload the rois
                else:
                    # Clear existing ROIs if requested
                    if clear_rois:
                        filter_to_use = clear_filter if clear_filter.strip() else None
                        clear_existing_rois(targetId, conn, filter_to_use)
                    
                    createdRois, roiTime = upload_ROIs(contour_dict, targetId, algorithm, conn, label_suffix)
                    newRois = newRois + createdRois
                imagesProcessed += 1
                if deleteLabelImage:
                    delete_image(image, conn)
                print(f"{int(contourTime)}s to create contours and {int(roiTime)}s to upload ROIs")

            # empty out the image dictionary
            imageDict.clear()
        
    elif inputType == "Image":
        imageDict = {}
        for id in inputIds:
            image = conn.getObject("Image", id)
            # assuming that not all images have the same Dataset
            dataset = image.getAncestry()[0]
            for img in list(dataset.listChildren()):
                if not id == img.id and not isLabelImage(img.name, label_suffix):
                    imageDict[img.name] = img.id
            if len(imageDict)>0:
                print("image dict:\n",imageDict)
                
            # Find corresponding label image
            if search_mode == "Same Dataset":
                target_base = remove_extension(image.name)
                label_image = None
                for img in list(dataset.listChildren()):
                    if label_suffix in img.name and img.name.startswith(target_base):
                        label_image = img
                        break
            elif search_mode == "Specific Dataset" and label_dataset_id:
                label_dataset = conn.getObject("Dataset", label_dataset_id)
                target_base = remove_extension(image.name)
                label_image = None
                if label_dataset:
                    for img in list(label_dataset.listChildren()):
                        if label_suffix in img.name and img.name.startswith(target_base):
                            label_image = img
                            break
            else:
                label_image = image  # fallback to original behavior
                
            if not label_image or label_image == image:
                print(f"!! Warning: No label image found for '{image.name}' !!")
                continue
                
            # Clear existing ROIs if requested
            if clear_rois:
                filter_to_use = clear_filter if clear_filter.strip() else None
                clear_existing_rois(image.id, conn, filter_to_use)
                
            plane = get_label_image_as_array(label_image)
            print("--------------------------------------")
            print(f"processing image '{label_image.name}' from Dataset '{dataset.name}'")
            print(f"shape of plane z=0/t=0/c=0: ", plane.shape)
            print("min:", plane.min(), " max:", plane.max(),\
                "pixel type:", plane.dtype.name)
            # get the contours
            contour_dict, contourTime = create_contours(plane, algorithm)
            # upload the rois
            createdRois, roiTime = upload_ROIs(contour_dict, image.id, algorithm, conn, label_suffix)
            newRois = newRois + createdRois
            imagesProcessed += 1
            if deleteLabelImage:
                delete_image(label_image, conn)
            print(f"{int(contourTime)}s to create contours and {int(roiTime)}s to upload ROIs")
        imageDict.clear() 

    return newRois, imagesProcessed

def remove_extension(filename):
    """Get basename without any extension."""
    if '.' in filename:
        return filename[:filename.index('.')]
    return filename

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

# helper function to get the image as an numpy array
def get_label_image_as_array(image):
    z, t, c = 0, 0, 0  # first plane of the image
    pixels = image.getPrimaryPixels()
    # get a numpy array
    plane = pixels.getPlane(z, c, t)
    return plane


# helper function to delete an image
def delete_image(image, conn):
    delete = Delete2(targetObjects={"Image": [image.id]})
    conn.c.submit(delete, loops=5, ms=2000)


# get a cropped sub-mask from a bool-mask
def get_cropped_mask(mask):
    # adapted from omero_rois package
    xmask = mask.sum(0).nonzero()[0]
    ymask = mask.sum(1).nonzero()[0]
    x0 = min(xmask)
    # padd everything by one pixel to
    # enable find_contours to work better
    if x0 != 0:
        x0 -= 1
    w = max(xmask) - x0 + 2
    y0 = min(ymask)
    if y0 != 0:
        y0 -= 1
    h = max(ymask) - y0 + 2
    submask = mask[y0: (y0 + h), x0: (x0 + w)]

    return submask, x0, y0


# custom implementation of skimage.measure.find_contours()
def own_find_contours(image):
    segments = _get_contour_segments(image.astype(np.float64))
    contours = _assemble_contours(segments)
    return contours

################################################################################################
# from scikit-image find_contours() Cython->Python Conversion                                  #
# original code:                                                                               #
# https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours_cy.pyx #
################################################################################################


def _get_fraction(from_value, to_value):
    if to_value == from_value:
        return 0
    return (0 - from_value) / (to_value - from_value)


def _get_contour_segments(array):
    segments = []

    for r0 in range(array.shape[0] - 1):
        for c0 in range(array.shape[1] - 1):
            r1, c1 = r0 + 1, c0 + 1

            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]

            square_case = 0
            if ul > 0:
                square_case += 1
            if ur > 0:
                square_case += 2
            if ll > 0:
                square_case += 4
            if lr > 0:
                square_case += 8

            if square_case in [0, 15]:
                continue

            top = r0, c0 + _get_fraction(ul, ur)
            bottom = r1, c0 + _get_fraction(ll, lr)
            left = r0 + _get_fraction(ul, ll), c0
            right = r0 + _get_fraction(ur, lr), c1

            if (square_case == 1):
                # top to left
                segments.append((top, left))
            elif (square_case == 2):
                # right to top
                segments.append((right, top))
            elif (square_case == 3):
                # right to left
                segments.append((right, left))
            elif (square_case == 4):
                # left to bottom
                segments.append((left, bottom))
            elif (square_case == 5):
                # top to bottom
                segments.append((top, bottom))
            elif (square_case == 6):
                segments.append((right, top))
                segments.append((left, bottom))
            elif (square_case == 7):
                # right to bottom
                segments.append((right, bottom))
            elif (square_case == 8):
                # bottom to right
                segments.append((bottom, right))
            elif (square_case == 9):
                segments.append((top, left))
                segments.append((bottom, right))
            elif (square_case == 10):
                # bottom to top
                segments.append((bottom, top))
            elif (square_case == 11):
                # bottom to left
                segments.append((bottom, left))
            elif (square_case == 12):
                # lef to right
                segments.append((left, right))
            elif (square_case == 13):
                # top to right
                segments.append((top, right))
            elif (square_case == 14):
                # left to top
                segments.append((left, top))

    return segments


def _assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        # Ignore degenerate segments.
        # This happens when (and only when) one vertex of the square is
        # exactly the contour level, and the rest are above or below.
        # This degenerate vertex will be picked up later by neighboring
        # squares.
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            # We need to connect these two contours.
            if tail is head:
                # We need to closed a contour: add the end point
                head.append(to_point)
            else:  # tail is not head
                # We need to join two distinct contours.
                # We want to keep the first contour segment created, so that
                # the final contours are ordered left->right, top->bottom.
                if tail_num > head_num:
                    # tail was created second. Append tail to head.
                    head.extend(tail)
                    # Remove tail from the detected contours
                    contours.pop(tail_num, None)
                    # Update starts and ends
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:  # tail_num <= head_num
                    # head was created second. Prepend head to tail.
                    tail.extendleft(reversed(head))
                    # Remove head from the detected contours
                    starts.pop(head[0], None)  # head[0] can be == to_point!
                    contours.pop(head_num, None)
                    # Update starts and ends
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            # We need to add a new contour
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:  # tail is not None
            # tail first element is to_point: the new segment should be
            # prepended.
            tail.appendleft(from_point)
            # Update starts
            starts[from_point] = (tail, tail_num)
        else:  # tail is None and head is not None:
            # head last element is from_point: the new segment should be
            # appended
            head.append(to_point)
            # Update ends
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]


# Modified function to determine if an image name comes from a label image
def isLabelImage(name, label_suffix="-label"):
    """Check if image name indicates it's a label image."""
    # For label detection, check if the suffix appears anywhere in the filename
    return label_suffix in name


# Fixed function to find label image for a specific target image
def find_label_for_image(target_image, label_dataset_id, label_suffix, search_mode, conn):
    """Find the corresponding label image for a target image."""
    target_base = remove_extension(target_image.name)  # test.tif -> test
    
    print(f"Looking for label image containing '{label_suffix}' for target '{target_image.name}' (base: '{target_base}')")
    
    search_datasets = []
    
    if search_mode == "Same Dataset":
        search_datasets = [target_image.getAncestry()[0]]
    elif search_mode == "Specific Dataset" and label_dataset_id:
        label_dataset = conn.getObject("Dataset", label_dataset_id)
        if label_dataset:
            search_datasets = [label_dataset]
        else:
            print(f"!! Error: Label dataset with ID {label_dataset_id} not found !!")
            return None
    elif search_mode == "All Datasets":
        search_datasets = list(conn.getObjects("Dataset"))
    
    # Search through the datasets
    for dataset in search_datasets:
        print(f"Searching in dataset '{dataset.name}' (ID: {dataset.id})")
        for image in list(dataset.listChildren()):
            # Check if this image could be a label for our target
            # Label should: 1) contain the suffix, 2) start with target basename
            if label_suffix in image.name and image.name.startswith(target_base):
                print(f"Found matching label image: '{image.name}'")
                return image
            print(f"  Checking image '{image.name}' -> no match")
    
    print(f"!! No label image found containing '{label_suffix}' and starting with '{target_base}'")
    return None


# Modified function to get the matching image id
def get_target_image_id(imageDict, labelName, label_suffix="-label"):
    """Find target image ID for a given label image name."""
    # Label should start with target basename and contain the suffix
    for name, id in imageDict.items():
        target_base = remove_extension(name)  # test.tif -> test
        # Check if label starts with this target base and contains suffix
        if labelName.startswith(target_base) and label_suffix in labelName:
            print(f"found matching image '{name}' for label '{labelName}'")
            return id
    
    print(f"!! No target found for label '{labelName}' with suffix '{label_suffix}'")
    print(f"!! Available images: {list(imageDict.keys())}")
    return 0


# Fixed function to get label images based on search mode
def get_label_images(target_dataset, label_dataset_id, label_suffix, search_mode, conn):
    """Get label images based on the specified search mode."""
    label_images = []
    
    if search_mode == "Same Dataset":
        search_datasets = [target_dataset]
    elif search_mode == "Specific Dataset" and label_dataset_id:
        label_dataset = conn.getObject("Dataset", label_dataset_id)
        if not label_dataset:
            print(f"!! Error: Label dataset with ID {label_dataset_id} not found !!")
            return []
        search_datasets = [label_dataset]
    elif search_mode == "All Datasets":
        search_datasets = list(conn.getObjects("Dataset"))
    else:
        return []
    
    for dataset in search_datasets:
        print(f"Searching for label images in dataset '{dataset.name}' (ID: {dataset.id})")
        for image in list(dataset.listChildren()):
            if isLabelImage(image.name, label_suffix):
                print(f"Found label image: '{image.name}'")
                label_images.append(image)
    
    return label_images


# Create contours that will be uploaded as ROIs from label images
def create_contours(labelimage, algorithm):
    contourDict = {}
    start = time.time()
    if algorithm == "Mask":

        for i in range(1, labelimage.max() + 1):
            mask = (labelimage == i)
            contour = omeroi.mask_from_binary_image(mask, text=str(i))
            contourDict[i] = contour
        # check if number of contours equals number of grey values
        # assuming that each grey value got correctly converted to a contour
        # this will (most likely) only work if the labeled regions do not touch
        assert len(contourDict) == labelimage.max(
        ), f"skimage.find_contours() found {len(contourDict)} ROIs instead of {labelimage.max()}."
        contourTime = time.time() - start
        return contourDict, contourTime

    elif algorithm == "Polygon":
        multipleContours = []
        for i in range(1, labelimage.max() + 1):
            mask = (labelimage == i)
            cropped, xOffset, yOffset = get_cropped_mask(mask)
            # make sure some signal is in the cropped mask
            assert np.array_equal(np.unique(cropped), [
                                  0, 1]), "The cropped array does not both contain 0s and 1s"
            # check if scikit-image package has been imported
            # otherwise use own function
            if "skimage" in sys.modules:
                contours = find_contours(cropped, level=0)
            else:
                contours = own_find_contours(cropped)
            # find_contours tends to find "extra" small contours
            # this serves only as a debug option to make sure everything got
            # recognized correctly
            if len(contours) > 1:
                # sort contours to make sure the relevant contour is at the start
                multipleContours.append(i)
                contours = sorted(contours, key=len, reverse=True)
                overlengthContours = []
                for counter, contour in enumerate(contours):
                    if counter == 0:
                        continue
                    # I chose length of 6 as this seemed the most sensible
                    # threshold after some testing
                    elif len(contour) < 6:
                        continue
                    else:
                        overlengthContours.append(len(contour))
                if len(overlengthContours) > 0:
                    print(
                        f"    for grey value {i} found {len(overlengthContours)} 'overlength' contour(s) with length(s): {overlengthContours}")
            contourDict[i] = [[x+yOffset, y+xOffset] for [x, y] in contours[0]]
        contourTime = time.time() - start
        if len(multipleContours) > 0:
            print(
                f"found multiple contours for the grey value(s) {multipleContours}")

        return contourDict, contourTime


# Modified upload_ROIs function to include suffix in ROI names
def upload_ROIs(contour_dict, parent_id, algorithm, conn, label_suffix="-label"):
    start = time.time()
    newRois = []
    
    # Clean up the suffix for ROI naming
    clean_suffix = label_suffix.lstrip('-_.')
    
    if algorithm == "Mask":
        # the Mask Shapes are already omero.model.ShapeI objects
        update = conn.getUpdateService()
        for greyValue, shape in contour_dict.items():
            roi = RoiI()
            roi.name = rstring(f"{clean_suffix}_{greyValue}")
            roi.image = conn.getObject("Image",parent_id)._obj
            roi.addShape(shape)
            update = conn.getUpdateService()
            roi = update.saveAndReturnObject(roi)
            newRois.append(roi.id.val)

    elif algorithm == "Polygon":
        # Polygon objects are lists of tuples of x,y coordinates
        for greyValue, coordinates in contour_dict.items():
        # create polygon shape for each
            flipped = np.flip(coordinates)
            roi_name = f"{clean_suffix}_{greyValue}"
            shape = [ez.rois.Polygon(flipped, label=roi_name)]
            # create roi and link shape to roi
            roi_id = ez.post_roi(conn, int(parent_id), shape, name=roi_name)
            newRois.append(roi_id)
    
    roiTime = time.time() - start
    print(f"created new Rois: {newRois}")

    return newRois, roiTime

# get the matching image id
def get_target_image_id(imageDict, labelName, label_suffix="-label"):
    for name, id in imageDict.items():
        target_base = remove_extension(name)
        if labelName.startswith(target_base) and label_suffix in labelName:
            print(f"found matching image '{name}'")
            return id
    return 0

# determine if an image name comes from a label image
def isLabelImage(name, label_suffix="-label"):
    return label_suffix in name

# Update the script parameters in run_script()
def run_script():
    data_types = [rstring('Dataset'),rstring('Image')]
    shape_types = [rstring("Polygon"), rstring("Mask")]
    search_modes = [rstring("Same Dataset"), rstring("Specific Dataset")]

    client = scripts.client(
        'Labels2Rois',
        """
        Creates (named) Rois from Label images.\n
        For correct mapping of the Rois the Label image must have\n
        the same name as the target image and contain the specified suffix\n
        (default: '-label'). Label images can be in the same dataset\n
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
            description="Select 'Polygon' or 'Mask'." +
            " A 'Mask' Shape will cover the segmented region, a 'Polygon'" +
            " will create an outline around it.\nIt also determines which " +
            "algorithm will be used. The 'Mask' algorithm is faster if the " +
            "ROIs do not touch.", values=shape_types, default="Polygon"),

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
            description="Deletes the Label image(s) after the conversion to Rois is done."),
       
        authors=["Jens Wendt"],
        contact="https://forum.image.sc/tag/omero, jens.wendt@uni-muenster.de",
        version="0.4"
        )

    try:
        script_params = client.getInputs(unwrap=True)
        conn = BlitzGateway(client_obj=client)
        # main function
        newRois, imagesProcessed  = labels2rois(script_params, conn)

        message = f"created {len(newRois)} ROIs in {imagesProcessed} images"
        client.setOutput("Message", rstring(message))

    finally:
        client.closeSession()


if __name__ == "__main__":
    run_script()
