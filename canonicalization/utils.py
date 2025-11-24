from typing import List, Tuple

import torch
from torchvision import transforms


def flip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """
    Flips bounding boxes horizontally.

    Args:
        boxes (torch.Tensor): The bounding boxes to flip.
        width (int): The width of the image.

    Returns:
        torch.Tensor: The flipped bounding boxes.
    """
    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return boxes


def flip_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Flips masks horizontally.

    Args:
        masks (torch.Tensor): The masks to flip.

    Returns:
        torch.Tensor: The flipped masks.
    """
    return masks.flip(-1)


def rotate_masks(masks: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotates masks by a specified angle.

    Args:
        masks (torch.Tensor): The masks to rotate.
        angle (torch.Tensor): The angle to rotate the masks by.

    Returns:
        torch.Tensor: The rotated masks.
    """
    return transforms.functional.rotate(masks, angle)


def rotate_boxes(boxes: torch.Tensor, angle: torch.Tensor, width: int) -> torch.Tensor:
    """
    Rotates bounding boxes by a specified angle.

    Args:
        boxes (torch.Tensor): The bounding boxes to rotate.
        angle (torch.Tensor): The angle to rotate the bounding boxes by.
        width (int): The width of the image.

    Returns:
        torch.Tensor: The rotated bounding boxes.
    """
    # rotate points
    origin: List[float] = [width / 2, width / 2]
    x_min_rot, y_min_rot = rotate_points(origin, boxes[:, :2].T, torch.deg2rad(angle))
    x_max_rot, y_max_rot = rotate_points(origin, boxes[:, 2:].T, torch.deg2rad(angle))

    # rearrange the max and mins to get rotated boxes
    x_min_rot, x_max_rot = torch.min(x_min_rot, x_max_rot), torch.max(
        x_min_rot, x_max_rot
    )
    y_min_rot, y_max_rot = torch.min(y_min_rot, y_max_rot), torch.max(
        y_min_rot, y_max_rot
    )
    rotated_boxes = torch.stack([x_min_rot, y_min_rot, x_max_rot, y_max_rot], dim=-1)

    return rotated_boxes
