import math
import torch

import numpy as np
import kornia as K

from omegaconf import DictConfig
from torch.nn import functional as F
from torchvision import transforms
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from eqbot.canonicalization.basecanonicalization import DiscreteGroupCanonicalization
from eqbot.canonicalization.utils import (
    flip_boxes,
    flip_masks,
    rotate_boxes,
    rotate_masks,
)


def square_pad(img, target_dim=368):
    """
    Pads an image to make it square, with per-channel constant values.
    Args:
        img: Input image of shape (H, W, C).
        target_dim: The target dimension for the square padded image.
    Returns:
        np.ndarray: Padded image of shape (target_dim, target_dim, C).
    """
    assert img.ndim == 3, "image_tensor shape must be (H, W, C)"
    assert img.shape[2] == 6, "image_tensor must have 6 channels"

    H, W, C = img.shape

    if H == target_dim and W == target_dim:
        return img

    # Create the padded image initialized to zeros with the same dtype as the input image
    padded_img = np.zeros((target_dim, target_dim, C), dtype=img.dtype)

    # Copy the original image into the center of the padded image, and remove the border
    pad_top = (target_dim - H) // 2
    pad_left = (target_dim - W) // 2
    padded_img[pad_top+1:pad_top+H-1, pad_left+1:pad_left+W-1, :] = img[1:H-1, 1:W-1, :]
    
    return padded_img


class DiscreteGroupImageCanonicalization(DiscreteGroupCanonicalization):
    """
    This class represents a discrete group image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.
    Other discrete group canonicalizers can be derived from this class.

    Methods:
        __init__: Initializes the DiscreteGroupImageCanonicalization instance.
        groupactivations_to_groupelement: Takes the activations for each group element as input and returns the group element.
        get_groupelement: Maps the input image to a group element.
        transformations_before_canonicalization_network_forward: Applies transformations to the input images before passing it through the canonicalization network.
        canonicalize: Canonicalizes the input images.
        invert_canonicalization: Inverts the canonicalization of the output of the canonicalized image.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the DiscreteGroupImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(canonicalization_network)

        self.beta = canonicalization_hyperparams.beta

        assert (
            len(in_shape) == 3
        ), "Input shape should be in the format (channels, height, width)"

        # Define all the image transformations here which are used during canonicalization
        # pad and crop the input image if it is not rotated MNIST
        is_grayscale = in_shape[0] == 1
        
        self.pad_ratio = 0.1
        self.pad = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.Pad(math.ceil(in_shape[-1] * self.pad_ratio), padding_mode="edge")
        )
        self.crop = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.CenterCrop((in_shape[-2], in_shape[-1]))
        )

        self.crop_canonization = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.CenterCrop(
                (
                    math.ceil(
                        in_shape[-2] * canonicalization_hyperparams.input_crop_ratio
                    ),
                    math.ceil(
                        in_shape[-1] * canonicalization_hyperparams.input_crop_ratio
                    ),
                )
            )
        )

        self.resize_canonization = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.Resize(size=canonicalization_hyperparams.resize_shape)
        )

    def groupactivations_to_groupelement(self, group_activations: torch.Tensor) -> dict:
        """
        This method takes the activations for each group element as input and returns the group element

        Args:
            group_activations (torch.Tensor): activations for each group element.

        Returns:
            dict: group element.
        """
        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        group_elements_one_hot = self.groupactivations_to_groupelementonehot(
            group_activations
        )

        angles = torch.linspace(0.0, 360.0, self.num_rotations + 1)[
            : self.num_rotations
        ].to(self.device)
        group_elements_rot_comp = (
            torch.cat([angles, angles], dim=0)
            if self.group_type == "roto-reflection"
            else angles
        )

        group_element_dict = {}

        group_element_rot_comp = torch.sum(
            group_elements_one_hot * group_elements_rot_comp, dim=-1
        )
        group_element_dict["rotation"] = group_element_rot_comp

        if self.group_type == "roto-reflection":
            reflect_identifier_vector = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)], dim=0
            ).to(self.device)
            group_element_reflect_comp = torch.sum(
                group_elements_one_hot * reflect_identifier_vector, dim=-1
            )
            group_element_dict["reflection"] = group_element_reflect_comp

        return group_element_dict

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input images.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The group activations.
        """
        raise NotImplementedError(
            "get_group_activations is not implemented for"
            "the DiscreteGroupImageCanonicalization class"
        )

    def get_groupelement(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps the input image to a group element.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            dict[str, torch.Tensor]: The corresponding group elements.
        """
        group_activations = self.get_group_activations(x)
        group_element_dict = self.groupactivations_to_groupelement(group_activations)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        self.canonicalization_info_dict["group_element"] = group_element_dict  # type: ignore
        self.canonicalization_info_dict["group_activations"] = group_activations

        return group_element_dict

    def transformations_before_canonicalization_network_forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies transformations to the input images before passing it through the canonicalization network.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The pre-canonicalized image.
        """
        x = self.crop_canonization(x)
        x = self.resize_canonization(x)
        return x

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalizes the input images.

        Args:
            x (torch.Tensor): The input images.
            targets (Optional[List], optional): The targets for instance segmentation. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized image, and optionally the targets.
        """
        self.device = x.device

        group_element_dict = self.get_groupelement(x)

        x = self.pad(x)

        if "reflection" in group_element_dict.keys():
            reflect_indicator = group_element_dict["reflection"][:, None, None, None]
            x = (1 - reflect_indicator) * x + reflect_indicator * K.geometry.hflip(x)

        # x = K.geometry.rotate(x, -group_element_dict["rotation"])
        # === Added: affine rotation ===
        B, _, H, W = x.shape
        device = x.device
        center = torch.tensor([[W / 2.0, H / 2.0]] * B, device=device)
        # Note: this is 'canonicalization', so use a negative angle
        angle = -group_element_dict["rotation"]
        M = K.geometry.get_rotation_matrix2d(center, angle, torch.ones(B, 2, device=device))
        x = K.geometry.warp_affine(x, M, dsize=(H, W), flags='bilinear', padding_mode='border', align_corners=True)

        x = self.crop(x)

        if targets:
            # canonicalize the targets (for instance segmentation, masks and boxes)
            image_width = x.shape[-1]

            if "reflection" in group_element_dict.keys():
                # flip masks and boxes
                for t in range(len(targets)):
                    targets[t]["boxes"] = flip_boxes(targets[t]["boxes"], image_width)
                    targets[t]["masks"] = flip_masks(targets[t]["masks"])

            # rotate masks and boxes
            for t in range(len(targets)):
                targets[t]["boxes"] = rotate_boxes(
                    targets[t]["boxes"], group_element_dict["rotation"][t], image_width
                )
                targets[t]["masks"] = rotate_masks(
                    targets[t]["masks"], -group_element_dict["rotation"][t].item()  # type: ignore
                )

            return x, targets

        return x

    def invert_canonicalize(self, x_canonicalized_out: torch.Tensor) -> torch.Tensor:
        """
        Inverts the canonicalization of the output of the canonicalized image.

        Args:
            x_canonicalized_out (torch.Tensor): The output of the canonicalized image.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: The output corresponding to the original image.
        """
        group_element_dict=self.canonicalization_info_dict["group_element"]
        angle = group_element_dict["rotation"]

        x_canonicalized_out = self.pad(x_canonicalized_out)
        
        # x_out = K.geometry.rotate(x_canonicalized_out, angles)
        # === Added: affine inverse rotation ===
        B, _, H, W = x_canonicalized_out.shape
        device = x_canonicalized_out.device

        # Rotation center is the image center
        center = torch.tensor([[W / 2.0, H / 2.0]] * B, device=device)
        # angle is already the inverse rotation (positive direction), no need to negate
        M = K.geometry.get_rotation_matrix2d(center, angle, torch.ones(B, 2, device=device))
        x_out = K.geometry.warp_affine(x_canonicalized_out, M, dsize=(H, W), flags='bilinear', padding_mode='border', align_corners=True)

        if "reflection" in group_element_dict:
            reflect_indicator = group_element_dict["reflection"]
            x_out_reflected = K.geometry.hflip(x_out)
            x_out = x_out * reflect_indicator[:, None, None, None] + x_out_reflected * (
                1 - reflect_indicator[:, None, None, None]
            )
            
        x_out = self.crop(x_out)

        return x_out

    def point_canonicalize(
        self, 
        p_on_fixed_input_img: Tuple[float, float], 
        fixed_input_image_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Transforms a point from the coordinate system of a fixed-size input image 
        (which is fed to self.canonicalize()) to the final canonicalized image space.

        This method simulates the internal operations of self.canonicalize():
        1. Internal padding (self.pad effect) based on fixed_input_image_size.
        2. Optional reflection (if any) based on group_element_dict.
        3. Rotation based on group_element_dict.
        4. Cropping (self.crop effect) back to fixed_input_image_size.

        Args:
            p_on_fixed_input_img (Tuple[float, float]): 
                The point (u, v) or (row, col) on the fixed-size image
                that is passed as input to the self.canonicalize() method.
                Example: A point on a 368x368 image.
            fixed_input_image_size (Tuple[int, int]): 
                The spatial dimensions (H_fixed, W_fixed) of the input image
                to self.canonicalize(). Example: (368, 368).

        Returns:
            Tuple[float, float]: The corresponding point (u_canon, v_canon) in the 
                                 final canonicalized image space. The dimensions of this space
                                 are also (H_fixed, W_fixed) due to the self.crop operation.
        """
        H_fixed, W_fixed = fixed_input_image_size
        u_fixed, v_fixed = p_on_fixed_input_img

        # --- 1. Internal Padding (self.pad effect) ---
        pad_pixels_internal: int
        if isinstance(self.pad, torch.nn.Identity): # Check if self.pad is defined and is Identity
            pad_pixels_internal = 0
        else:
            pad_pixels_internal = math.ceil(W_fixed * self.pad_ratio) # Assumes square fixed input for simplicity of self.pad

        # Point coordinates after internal padding
        u_after_internal_pad = u_fixed + pad_pixels_internal
        v_after_internal_pad = v_fixed + pad_pixels_internal
        
        # Dimensions of the "large canvas" after self.pad
        H_large = H_fixed + 2 * pad_pixels_internal
        W_large = W_fixed + 2 * pad_pixels_internal

        point_on_large_canvas = (u_after_internal_pad, v_after_internal_pad)

        # --- 2. Reflection (if any) ---
        if not hasattr(self, 'canonicalization_info_dict') or "group_element" not in self.canonicalization_info_dict:
            raise RuntimeError("canonicalization_info_dict or group_element not found. "
                               "Ensure self.canonicalize() was called on an image first.")
        group_element_dict = self.canonicalization_info_dict["group_element"]
        
        current_reflection = 0.0
        if "reflection" in group_element_dict:
            # Assuming batch size 1 for the image that set this dict, so index [0]
            current_reflection = group_element_dict["reflection"][0].item() 
            if current_reflection > 0.5: # If reflection was applied (typically 1.0)
                # Horizontal flip with respect to the width of the large canvas
                point_on_large_canvas = (point_on_large_canvas[0], W_large - 1.0 - point_on_large_canvas[1])
        
        # --- 3. Rotation ---
        current_rotation_angle_deg = -group_element_dict["rotation"][0].item() # Angle in degrees

        # Rotation center is the center of the large canvas
        center_u_large = H_large / 2.0
        center_v_large = W_large / 2.0

        u_p, v_p = point_on_large_canvas # Point to be rotated
        
        theta_rad = math.radians(current_rotation_angle_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        # Standard 2D rotation:
        # Original relative coordinates
        v_p_original_rel = v_p - center_v_large
        u_p_original_rel = u_p - center_u_large

        # To match Kornia's CCW image content rotation by theta_rad:
        v_rotated_component = v_p_original_rel * cos_t + u_p_original_rel * sin_t
        u_rotated_component = -v_p_original_rel * sin_t + u_p_original_rel * cos_t
 
        v_rot = center_v_large + v_rotated_component
        u_rot = center_u_large + u_rotated_component
        
        point_rotated_on_large_canvas = (u_rot, v_rot)

        # --- 4. Cropping (self.crop effect) ---
        u_final_canonical = point_rotated_on_large_canvas[0] - pad_pixels_internal
        v_final_canonical = point_rotated_on_large_canvas[1] - pad_pixels_internal
        p_final_canonical_coords = (u_final_canonical, v_final_canonical)
        
        return p_final_canonical_coords


class GroupEquivariantImageCanonicalization(DiscreteGroupImageCanonicalization):
    """
    This class represents a discrete group equivariant image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.

    Methods:
        __init__: Initializes the GroupEquivariantImageCanonicalization instance.
        get_group_activations: Gets the group activations for the input images.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the GroupEquivariantImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(
            canonicalization_network, canonicalization_hyperparams, in_shape
        )
        self.group_type = canonicalization_network.group_type
        self.num_rotations = canonicalization_network.num_rotations
        self.num_group = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.group_info_dict = {
            "num_rotations": self.num_rotations,
            "num_group": self.num_group,
        }

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input image.

        This method takes an image as input, applies transformations before forwarding it through the canonicalization network,
        and then returns the group activations.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The group activations.
        """
        x = self.transformations_before_canonicalization_network_forward(x)
        group_activations = self.canonicalization_network(x)
        return group_activations


class OptimizedGroupEquivariantImageCanonicalization(
    DiscreteGroupImageCanonicalization
):
    """
    This class represents an optimized (discrete) group equivariant image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.

    Methods:
        __init__: Initializes the OptimizedGroupEquivariantImageCanonicalization instance.
        rotate_and_maybe_reflect: Rotate and maybe reflect the input images.
        group_augment: Augment the input images by applying group transformations (rotations and reflections).
        get_group_activations: Gets the group activations for the input images.
        get_optimization_specific_loss: Gets the loss specific to the optimization process.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the OptimizedGroupEquivariantImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(
            canonicalization_network, canonicalization_hyperparams, in_shape
        )
        self.group_type = canonicalization_hyperparams.group_type
        self.num_rotations = canonicalization_hyperparams.num_rotations
        self.artifact_err_wt = canonicalization_hyperparams.artifact_err_wt
        self.num_group = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.out_vector_size = canonicalization_network.out_vector_size

        # group optimization specific cropping and padding (required for group_augment())
        group_augment_in_shape = canonicalization_hyperparams.resize_shape
        self.crop_group_augment = (
            torch.nn.Identity()
            if in_shape[0] == 1
            else transforms.CenterCrop(group_augment_in_shape)
        )
        self.pad_group_augment = (
            torch.nn.Identity()
            if in_shape[0] == 1
            else transforms.Pad(
                math.ceil(group_augment_in_shape * 0.5), padding_mode="edge"
            )
        )

        self.reference_vector = torch.nn.Parameter(
            torch.randn(1, self.out_vector_size),
            requires_grad=canonicalization_hyperparams.learn_ref_vec,
        )
        self.group_info_dict = {
            "num_rotations": self.num_rotations,
            "num_group": self.num_group,
        }

    def rotate_and_maybe_reflect(
        self, x: torch.Tensor, degrees: torch.Tensor, reflect: bool = False
    ) -> List[torch.Tensor]:
        """
        Rotate and maybe reflect the input images.

        Args:
            x (torch.Tensor): The input image.
            degrees (torch.Tensor): The degrees of rotation.
            reflect (bool, optional): Whether to reflect the image. Defaults to False.

        Returns:
            List[torch.Tensor]: The list of rotated and maybe reflected images.
        """
        x_augmented_list = []
        for degree in degrees:
            x_rot = self.pad_group_augment(x)
            x_rot = K.geometry.rotate(x_rot, -degree)
            if reflect:
                x_rot = K.geometry.hflip(x_rot)
            x_rot = self.crop_group_augment(x_rot)
            x_augmented_list.append(x_rot)
        return x_augmented_list

    def group_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment the input images by applying group transformations (rotations and reflections).

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The augmented image.
        """
        degrees = torch.linspace(0, 360, self.num_rotations + 1)[:-1].to(self.device)
        x_augmented_list = self.rotate_and_maybe_reflect(x, degrees)

        if self.group_type == "roto-reflection":
            x_augmented_list += self.rotate_and_maybe_reflect(x, degrees, reflect=True)

        return torch.cat(x_augmented_list, dim=0)

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input image.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The group activations.
        """
        x = self.transformations_before_canonicalization_network_forward(x)
        x_augmented = self.group_augment(
            x
        )  # size (batch_size * group_size, in_channels, height, width)

        vector_out = self.canonicalization_network(
            x_augmented
        )  # size (batch_size * group_size, reference_vector_size)
        self.canonicalization_info_dict = {"vector_out": vector_out}

        if self.artifact_err_wt:
            # select a random rotation for each image in the batch
            rotation_indices = torch.randint(
                0, self.num_rotations, (x_augmented.shape[0],)
            ).to(self.device)

            # apply the rotation degree to the images
            x_dummy = self.pad_group_augment(x_augmented)
            x_dummy = K.geometry.rotate(
                x_dummy, -rotation_indices * 360 / self.num_rotations
            )
            x_dummy = self.crop_group_augment(x_dummy)

            # invert the image back to the original orientation
            x_dummy = self.pad_group_augment(x_dummy)
            x_dummy = K.geometry.rotate(
                x_dummy, rotation_indices * 360 / self.num_rotations
            )
            x_dummy = self.crop_group_augment(x_dummy)

            vector_out_dummy = self.canonicalization_network(
                x_dummy
            )  # size (batch_size * group_size, reference_vector_size)
            self.canonicalization_info_dict.update(
                {"vector_out_dummy": vector_out_dummy}
            )

        scalar_out = F.cosine_similarity(
            self.reference_vector.repeat(vector_out.shape[0], 1), vector_out
        )  # size (batch_size * group_size, 1)
        group_activations = scalar_out.reshape(
            self.num_group, -1
        ).T  # size (batch_size, group_size)
        return group_activations

    def get_optimization_specific_loss(self) -> torch.Tensor:
        """
        Gets the loss specific to the optimization process.

        Returns:
            torch.Tensor: The loss.
        """
        vectors = self.canonicalization_info_dict["vector_out"]

        # compute error to reduce rotation artifacts
        rotation_artifact_error = 0
        if self.artifact_err_wt:
            vectors_dummy = self.canonicalization_info_dict["vector_out_dummy"]
            rotation_artifact_error = torch.nn.functional.mse_loss(
                vectors_dummy, vectors
            )  # type: ignore

        # error to ensure that the vectors are (as much as possible) orthogonal
        vectors = vectors.reshape(self.num_group, -1, self.out_vector_size).permute(
            (1, 0, 2)
        )  # (batch_size, group_size, vector_out_size)
        distances = vectors @ vectors.permute((0, 2, 1))
        mask = 1.0 - torch.eye(self.num_group).to(
            self.device
        )  # (group_size, group_size)

        return (
            torch.abs(distances * mask).mean()
            + self.artifact_err_wt * rotation_artifact_error
        )


class InverseActionCanonicalizationNetwork(torch.nn.Module):
    """
    Learn a simple inverse canonicalization network to map VLA output actions from the canonical space back to the original space.
    
    The network concatenates each action chunk predicted by VLA with the canonicalization information (rotation angles), 
    then uses a shared MLP to process each concatenated vector and predict the inverse-canonicalized action for each chunk.
    """
    
    def __init__(self, action_dim: int = 7, hidden_dim: int = 256, action_chunks: int = 8):
        """
        Initialize the simple inverse canonicalization network.
        
        Args:
            action_dim (int): Action dimension, default 7 (x, y, z, rx, ry, rz, gripper)
            hidden_dim (int): Hidden layer size
            action_chunks (int): Number of action chunks; OpenVLA defaults to 8.
        """
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_chunks = action_chunks
        
        # Input dimension = single action chunk dimension + rotation angles
        input_dim = self.action_dim + 2

        # Define a shared MLP applied independently to each action chunk
        self.shared_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.action_dim)
        )   # Output is the inverse-canonicalized action for one chunk
    
    def forward(self, predicted_actions: torch.Tensor, 
                canonicalization_info_1: Dict[str, torch.Tensor],
                canonicalization_info_2: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass performing inverse canonicalization.
        
        Args:
            predicted_actions (torch.Tensor): VLA-predicted actions of shape (batch_size, action_chunk, action_dim)
            canonicalization_info_1 (Dict[str, torch.Tensor]): First camera canonicalization info
            canonicalization_info_2 (Dict[str, torch.Tensor]): Second camera canonicalization info
            
        Returns:
            torch.Tensor: Inverse-canonicalized actions
        """
        batch_size, num_chunks, action_dim = predicted_actions.shape
        assert num_chunks == self.action_chunks and action_dim == self.action_dim, "Input action shape mismatch"

        # 1. Extract and normalize rotation angles
        angle1 = canonicalization_info_1["group_element"]["rotation"].unsqueeze(1) / 360.0 # (B, 1)
        angle2 = canonicalization_info_2["group_element"]["rotation"].unsqueeze(1) / 360.0 # (B, 1)
        rotation_info = torch.cat([angle1, angle2], dim=1) # (B, 2)

        # 2. Expand rotation info to match the number of action chunks
        # From (B, 2) to (B, num_chunks, 2)
        expanded_rotation_info = rotation_info.unsqueeze(1).expand(-1, num_chunks, -1)

        # 3. Concatenate each action chunk with rotation info
        # predicted_actions: (B, C, D) | expanded_rotation_info: (B, C, 2)
        # combined_input: (B, C, D + 2)
        combined_input = torch.cat([predicted_actions, expanded_rotation_info], dim=2)

        # 4. Reshape tensor for shared MLP processing
        # From (B, C, D + 2) to (B * C, D + 2)
        reshaped_input = combined_input.reshape(batch_size * num_chunks, -1)

        # 5. Apply the shared MLP to predict inverse-canonicalized actions for all chunks
        # Output shape: (B * C, D)
        output_actions_flat = self.shared_mlp(reshaped_input)

        # 6. Reshape output back to sequence format
        # From (B * C, D) to (B, C, D)
        output_actions = output_actions_flat.reshape(batch_size, num_chunks, -1)
        
        # 7. Return the MLP-predicted inverse-canonicalized actions
        return output_actions


class GroupEquivariantImageCanonicalization(torch.nn.Module):
    """
    Group-equivariant image canonicalization system.
    
    Manages independent canonicalization networks for the different cameras,
    and provides a unified interface for canonicalization and inverse-canonicalization operations.
    """
    
    def __init__(self, 
                 canonicalization_network_type: Type[torch.nn.Module],
                 canonicalization_network_1: torch.nn.Module,
                 canonicalization_network_2: torch.nn.Module,
                 canonicalization_hyperparams: DictConfig,
                 in_shape: tuple,
                 action_chunks: int = 8):
        """
        Initialize the group-equivariant canonicalization system.
        
        Args:
            canonicalization_network_1: First camera canonicalization network
            canonicalization_network_2: Second camera canonicalization network
            canonicalization_hyperparams: Canonicalization hyperparameters
            in_shape: Input image shape
            action_chunks (int): Number of action chunks
        """
        super().__init__()
        
        # Create two independent canonicalizers
        self.canonicalizer_1 = canonicalization_network_type(
            canonicalization_network_1,
            canonicalization_hyperparams,
            in_shape
        )
        
        self.canonicalizer_2 = canonicalization_network_type(
            canonicalization_network_2,
            canonicalization_hyperparams,
            in_shape
        )
        
        # Create inverse canonicalization network
        self.inverse_action_network = InverseActionCanonicalizationNetwork(
            action_dim=7,  # adjustable as needed
            hidden_dim=256,
            action_chunks=action_chunks
        )
        
        # Store canonicalization info
        self.canonicalization_info_first = {}
        self.canonicalization_info_second = {}
    
    def canonicalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Canonicalize input images.
        
        Args:
            images (torch.Tensor): Input of shape (batch_size, total_channels, height, width)
            
        Returns:
            torch.Tensor: Canonicalized images
        """
        batch_size, total_channels, height, width = images.shape
        assert total_channels == 12, "Input images should have 12 channels (6 for first camera, 6 for second camera)"

        # Separate first and second camera images
        first_images = images[:, :6, :, :]  # (B, 6, H, W)
        second_images = images[:, 6:, :, :]  # (B, 6, H, W)
        
        # Canonicalize separately
        canonicalized_first = self.canonicalizer_1.canonicalize(first_images)
        canonicalized_second = self.canonicalizer_2.canonicalize(second_images)
        
        # Store canonicalization info
        self.canonicalization_info_first = self.canonicalizer_1.canonicalization_info_dict
        self.canonicalization_info_second = self.canonicalizer_2.canonicalization_info_dict
        
        # Merge canonicalized images
        canonicalized_images = torch.cat([canonicalized_first, canonicalized_second], dim=1)
        
        return canonicalized_images
    
    def inverse_canonicalize_actions(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        """
        Inverse-canonicalize predicted actions.
        
        Args:
            predicted_actions (torch.Tensor): VLA-predicted actions of shape (batch_size, action_chunk, action_dim)
            
        Returns:
            torch.Tensor: Inverse-canonicalized actions
        """
        return self.inverse_action_network(
            predicted_actions,
            self.canonicalization_info_first,
            self.canonicalization_info_second
        )
    
    def get_prior_regularization_loss(self) -> torch.Tensor:
        """
        Get prior regularization loss.
        
        Returns:
            torch.Tensor: Total prior regularization loss
        """
        loss_first = self.canonicalizer_1.get_prior_regularization_loss()
        loss_second = self.canonicalizer_2.get_prior_regularization_loss()
        return loss_first + loss_second
