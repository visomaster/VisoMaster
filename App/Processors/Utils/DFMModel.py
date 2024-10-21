import torch
import onnxruntime
import App.Processors.Utils.FaceUtil as faceutil
import numpy as np

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1

class DFMModel:
    def __init__(self, model_path: str, providers, device='cuda'):

        self._model_path = model_path
        self.providers = providers
        self.device = device
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=device)
        
        sess = self._sess = onnxruntime.InferenceSession(str(model_path), providers=self.providers)
        inputs = sess.get_inputs()

        if len(inputs) == 0 or 'in_face' not in inputs[0].name:
            raise Exception(f'Invalid model {model_path}')

        self._input_height, self._input_width = inputs[0].shape[1:3]
        self._model_type = 1

        if len(inputs) == 2:
            if 'morph_value' not in inputs[1].name:
                raise Exception(f'Invalid model {model_path}')
            self._model_type = 2

        elif len(inputs) > 2:
            raise Exception(f'Invalid model {model_path}')

        # Mapping function from ONNX Runtime data types to PyTorch dtypes (you may need to adjust based on your actual usage)
        self.onnx_to_torch_dtype = {
            'tensor(float)': torch.float32,
            'tensor(double)': torch.float64,
            'tensor(int32)': torch.int32,
            'tensor(int64)': torch.int64,
            # Add other necessary dtype mappings as needed
        }

        # Mapping function from ONNX Runtime data types to NumPy dtypes for binding
        self.onnx_to_numpy_dtype = {
            'tensor(float)': np.float32,
            'tensor(double)': np.float64,
            'tensor(int32)': np.int32,
            'tensor(int64)': np.int64,
            # Add other necessary dtype mappings as needed
        }

    def get_model_path(self):
        return self._model_path

    def get_input_res(self):
        return self._input_width, self._input_height

    def has_morph_value(self) -> bool:
        return self._model_type == 2

    def convert(self, img, morph_factor=0.75, rct=False):
        """
        img    torch.Tensor  CHW uint8,float32
        morph_factor   float   used if model supports it
        returns:
         img        NHW3  same dtype as img
         celeb_mask NHW1  same dtype as img
         face_mask  NHW1  same dtype as img
        """
        dtype = img.dtype

        # Normalize img and transform to NCHW
        img = self.to_ufloat32(img)
        img = torch.unsqueeze(img, 0)  # Add batch dimension

        # Resize to the input shape
        img = torch.nn.functional.interpolate(img, size=(self._input_height, self._input_width), mode="bilinear", align_corners=False)

        # Convert from RGB to BGR Format (assuming input is RGB)
        img = img[:, [2, 1, 0], :, :]  # Reverse the channel dimension (C)

        # Transform from NCHW to NHWC and ensure all bytes are contiguous
        img = img.permute(0, 2, 3, 1).contiguous()

        N, H, W, C = img.shape

        io_binding = self._sess.io_binding()

        # Bind input image tensor
        io_binding.bind_input(name='in_face:0', device_type=self.device, device_id=0, element_type=np.float32, shape=img.shape, buffer_ptr=img.data_ptr())

        # Bind morph factor if the model supports it
        if self._model_type == 2:
            morph_factor_t = torch.tensor([morph_factor], dtype=torch.float32, device=self.device)
            io_binding.bind_input(name='morph_value:0', device_type=self.device, device_id=0, element_type=np.float32, shape=morph_factor_t.shape, buffer_ptr=morph_factor_t.data_ptr())

        # Prepare output tensors and bind them
        outputs = self._sess.get_outputs()
        binding_outputs = []

        for idx, output in enumerate(outputs):
            # Convert shape to a valid tuple of integers
            shape = self.convert_shape(output.shape)

            # Create a torch tensor with the shape and dtype of the output
            torch_dtype = self.onnx_to_torch_dtype[output.type]
            tensor_output = torch.empty(shape, dtype=torch_dtype, device=self.device).contiguous()

            # Append the tensor to the list
            binding_outputs.append(tensor_output)

            # Bind the output using ONNX Runtime's io_binding
            io_binding.bind_output(
                name=output.name,
                device_type=self.device,
                device_id=0,
                element_type=self.onnx_to_numpy_dtype[output.type],  # Use NumPy dtype for element_type
                shape=shape,
                buffer_ptr=binding_outputs[idx].data_ptr()
            )

        # Run the model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self._sess.run_with_iobinding(io_binding)

        # Process outputs (resize, clip channels, and convert back to original dtype)
        out_face_mask = self.to_dtype(self.ch(self.resize(binding_outputs[0], (W, H)), 1), dtype)

        # NHWC to HWC
        out_face_mask = torch.squeeze(out_face_mask, dim=0)

        # Process outputs (resize, clip channels, and convert back to original dtype)
        out_celeb = self.to_dtype(self.ch(self.resize(binding_outputs[1], (W, H)), 3), dtype)

        # Process outputs (resize, clip channels, and convert back to original dtype)
        out_celeb_mask = self.to_dtype(self.ch(self.resize(binding_outputs[2], (W, H)), 1), dtype)

        # If rct is enabled, further processing is needed
        if rct:
            # convert img back to original dtype
            img = self.to_dtype(img, dtype)
            # apply rct
            out_celeb = self.rct(out_celeb, img, out_celeb_mask, out_celeb_mask, 0.3)
            # NHWC to HWC
            out_celeb = torch.squeeze(out_celeb, dim=0)
            if out_celeb.shape[-1] == 3:  # Check if there are 3 channels
                out_celeb = out_celeb[..., [2, 1, 0]]  # Safe way to reorder channels from BGR to RGB
        else:
            # NHWC to HWC
            out_celeb = torch.squeeze(out_celeb, dim=0)
            if out_celeb.shape[-1] == 3:  # Check if there are 3 channels
                out_celeb = out_celeb[..., [2, 1, 0]]  # Safe way to reorder channels from BGR to RGB

        # NHWC to HWC
        out_celeb_mask = torch.squeeze(out_celeb_mask, dim=0)

        return out_celeb, out_celeb_mask, out_face_mask

    def rct(self, img: torch.Tensor, like: torch.Tensor, mask: torch.Tensor = None, like_mask: torch.Tensor = None, mask_cutoff=0.5):
        """
        Transfer color using the RCT method.

        Args:
            img (torch.Tensor): [N, H, W, 3] torch.uint8/torch.float32
            like (torch.Tensor): [N, H, W, 3] torch.uint8/torch.float32
            mask (torch.Tensor, optional): [N, H, W, 1] torch.uint8/torch.float32
            like_mask (torch.Tensor, optional): [N, H, W, 1] torch.uint8/torch.float32
            mask_cutoff (float, optional): Cutoff value for masks. Defaults to 0.5.

        Returns:
            torch.Tensor: The color-transferred image. [N, C, H, W]
        """
        dtype = img.dtype
        N = img.shape[0]  # Batch size

        # Convert images to float32 and normalize to [0, 1]
        img = self.to_ufloat32(img).permute(0, 3, 1, 2)  # Convert to (N, 3, H, W)
        like_for_stat = self.to_ufloat32(like).permute(0, 3, 1, 2)  # Convert to (N, 3, H, W)

        # Convert to LAB color space for each image in the batch
        img_lab = torch.stack([faceutil.rgb_to_lab(img[i], False) for i in range(N)])  # Convert to LAB in (N, 3, H, W)
        like_lab = torch.stack([faceutil.rgb_to_lab(like_for_stat[i], False) for i in range(N)])  # Convert to LAB in (N, 3, H, W)

        # Apply like mask
        if like_mask is not None:
            like_mask = self.get_image(self.ch(self.to_ufloat32(like_mask), 1), 'NHW')  # Convert to (N, H, W)
            like_mask = like_mask.unsqueeze(1).expand(-1, 3, -1, -1)  # Convert to (N, 3, H, W) to match (N, 3, H, W)
            like_lab = like_lab.clone()
            like_lab[like_mask < mask_cutoff] = 0  # Zero out regions below cutoff

        # Apply img mask
        img_for_stat = img_lab.clone()
        if mask is not None:
            mask = self.get_image(self.ch(self.to_ufloat32(mask), 1), 'NHW')  # Convert to (N, H, W)
            mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)  # Convert to (N, 3, H, W) to match (N, 3, H, W)
            img_for_stat = img_for_stat.clone()
            img_for_stat[mask < mask_cutoff] = 0  # Zero out regions below cutoff

        # Initialize the output tensor
        img_out = torch.zeros_like(img_lab)

        # Process each image in the batch
        for i in range(N):
            # Compute statistics for LAB channels in (3, H, W)
            source_l_mean, source_l_std = img_for_stat[i, 0].mean(), img_for_stat[i, 0].std()
            source_a_mean, source_a_std = img_for_stat[i, 1].mean(), img_for_stat[i, 1].std()
            source_b_mean, source_b_std = img_for_stat[i, 2].mean(), img_for_stat[i, 2].std()

            like_l_mean, like_l_std = like_lab[i, 0].mean(), like_lab[i, 0].std()
            like_a_mean, like_a_std = like_lab[i, 1].mean(), like_lab[i, 1].std()
            like_b_mean, like_b_std = like_lab[i, 2].mean(), like_lab[i, 2].std()

            # Perform color transfer adjustments in (3, H, W)
            source_l = img_lab[i, 0]
            source_a = img_lab[i, 1]
            source_b = img_lab[i, 2]

            # Adjust L, A, B channels
            source_l = (source_l - source_l_mean) * (like_l_std / (source_l_std + 1e-6)) + like_l_mean
            source_a = (source_a - source_a_mean) * (like_a_std / (source_a_std + 1e-6)) + like_a_mean
            source_b = (source_b - source_b_mean) * (like_b_std / (source_b_std + 1e-6)) + like_b_mean

            # Clip the adjusted channels to valid LAB ranges
            source_l = torch.clamp(source_l, 0, 100)
            source_a = torch.clamp(source_a, -127, 127)
            source_b = torch.clamp(source_b, -127, 127)

            # Stack channels back together in (3, H, W)
            img_out[i] = torch.stack([source_l, source_a, source_b], dim=0)

        # Convert back to RGB for each image in the batch
        img_out = torch.stack([faceutil.lab_to_rgb(img_out[i], False) for i in range(N)])  # Convert from LAB to RGB directly in (N, 3, H, W)

        # Convert back to the original data type
        img_out = self.to_dtype(img_out, dtype).permute(0, 2, 3, 1)  # Convert back to (N, H, W, 3)
        return img_out

    def convert_shape(self, shape):
        # Iterate over each dimension in the shape
        return tuple(
            int(dim) if isinstance(dim, int) or (isinstance(dim, str) and dim.isdigit()) else 1
            for dim in shape
        )

    def to_dtype(self, img, dtype, from_tanh=False):
        if dtype == torch.float32:
            return self.to_ufloat32(img, from_tanh=from_tanh)
        elif dtype == torch.uint8:
            return self.to_uint8(img, from_tanh=from_tanh)
        else:
            raise ValueError('unsupported dtype')

    def to_ufloat32(self, img, as_tanh=False, from_tanh=False):
        """
        Convert to uniform float32
        """
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
            if as_tanh:
                img /= 127.5
                img -= 1.0
            else:
                img /= 255.0
        elif img.dtype in [torch.float32, torch.float64]:
            if from_tanh:
                img += 1.0
                img /= 2.0

        return img

    def to_uint8(self, img, from_tanh=False):
        """
        Convert to uint8

        if current image dtype is float32/64, then image will be multiplied by *255
        """
        if img.dtype in [torch.float32, torch.float64]:
            if from_tanh:
                img += 1.0
                img /= 2.0

            img *= 255.0
            img = torch.clamp(img, 0, 255)

        return img.to(torch.uint8)

    def get_image(self, img, format):
        """
        Returns a PyTorch tensor image with the desired format.

        Args:
            img (torch.Tensor): Input image tensor.
            format (str): Desired format, e.g., 'NHWC', 'HWCN', 'NHW'.

        Returns:
            torch.Tensor: The image tensor in the desired format.
        """
        format = format.upper()

        # First slice missing dims
        N_slice = 0 if 'N' not in format else slice(None)
        H_slice = 0 if 'H' not in format else slice(None)
        W_slice = 0 if 'W' not in format else slice(None)
        C_slice = 0 if 'C' not in format else slice(None)
        img = img[N_slice, H_slice, W_slice, C_slice]

        f = ''
        if 'N' in format: f += 'N'
        if 'H' in format: f += 'H'
        if 'W' in format: f += 'W'
        if 'C' in format: f += 'C'

        if f != format:
            # Transpose to target
            d = {s: i for i, s in enumerate(f)}
            transpose_order = [d[s] for s in format]
            img = img.permute(transpose_order)  # PyTorch uses permute for transpose-like operations

        return img.contiguous()  # Ensures that the tensor is contiguous in memory

    def ch(self, img, TC: int):
        """
        Clips or expands the channel dimension to the target number of channels.

        Args:
            img (torch.Tensor): Input image tensor with shape (N, H, W, C).
            TC (int): Target number of channels, must be >= 1.

        Returns:
            torch.Tensor: Image tensor with the target number of channels.
        """
        N, H, W, C = img.shape

        if TC <= 0:
            raise ValueError(f'channels must be positive value, not {TC}')

        if TC > C:
            # Channel expansion
            img = img[..., 0:1]  # Clip to single channel first.
            img = img.repeat(1, 1, 1, TC)  # Expand by repeat along the last dimension
        elif TC < C:
            # Channel reduction by clipping
            img = img[..., :TC]

        return img

    def resize(self, img: torch.Tensor, size: tuple, interpolation: str = 'bilinear'):
        """
        Resize a PyTorch tensor image to the target size (W, H).

        Args:
            img (torch.Tensor): Input image tensor with shape (N, H, W, C).
            size (Tuple[int, int]): Target size as (width, height).
            interpolation (str): Interpolation method, can be 'nearest', 'bilinear', or 'bicubic'.

        Returns:
            torch.Tensor: Resized image tensor with shape (N, TH, TW, C).
        """
        # Ensure the interpolation method is supported
        supported_interpolations = {'nearest', 'bilinear', 'bicubic'}
        if interpolation not in supported_interpolations:
            raise ValueError(f"Interpolation '{interpolation}' not supported. Choose from {supported_interpolations}.")

        # Input image shape
        N, H, W, C = img.shape
        TW, TH = size  # Target Width and Height

        if W != TW or H != TH:
            # PyTorch expects channel dimension to be the second (N, C, H, W) format
            img = img.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)

            # Resize using torch.nn.functional.interpolate
            img = torch.nn.functional.interpolate(img, size=(TH, TW), mode=interpolation, align_corners=False)

            # Convert back to original shape (N, H, W, C)
            img = img.permute(0, 2, 3, 1)

        return img