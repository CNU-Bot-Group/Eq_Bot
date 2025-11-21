import e2cnn
import torch
from e2cnn import gspaces


class ESCNNEquivariantNetwork(torch.nn.Module):
    """
    This class represents an Equivariant Convolutional Neural Network (Equivariant CNN).

    The network is equivariant to a group of transformations, which can be either rotations or roto-reflections.
    The network consists of a sequence of equivariant convolutional layers, each followed by batch normalization, a ReLU activation function, and dropout. The number of output channels of the convolutional layers is the same for all layers.

    Methods:
        __init__: Initializes the ESCNNEquivariantNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int,
        group_type: str = "rotation",
        num_rotations: int = 4,
        num_layers: int = 1,
    ):
        """
        Initializes the ESCNNEquivariantNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the convolutional layers.
            kernel_size (int): The size of the kernel of the convolutional layers.
            group_type (str, optional): The type of the group of transformations. It can be either "rotation" or "roto-reflection". Defaults to "rotation".
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            num_layers (int, optional): The number of convolutional layers. Defaults to 1.
        """
        super().__init__()

        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations

        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        r1 = e2cnn.nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.in_channels
        )
        r2 = e2cnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * out_channels)

        self.in_type = r1
        self.out_type = r2

        modules = [
            e2cnn.nn.R2Conv(self.in_type, self.out_type, kernel_size),
            e2cnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            e2cnn.nn.ReLU(self.out_type, inplace=True),
            e2cnn.nn.PointwiseDropout(self.out_type, p=0.5),
        ]
        for _ in range(num_layers - 2):
            modules.append(
                e2cnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),
            )
            modules.append(
                e2cnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            )
            modules.append(
                e2cnn.nn.ReLU(self.out_type, inplace=True),
            )
            modules.append(
                e2cnn.nn.PointwiseDropout(self.out_type, p=0.5),
            )

        modules.append(
            e2cnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),
        )

        self.eqv_network = e2cnn.nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, num_group_elements).
        """
        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(
            feature_map.shape[0],
            self.out_channels,
            self.num_group_elements,
            feature_map.shape[-2],
            feature_map.shape[-1],
        )

        group_activations = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activations

