import torch
import numpy as np
import math
from torchvision.transforms import v2
from functools import partial
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.Processors.ModelsProcessor import ModelsProcessor

def run_enhance_frame_tile_process(self: 'ModelsProcessor', img, enhancer_type, tile_size=256, scale=1):
    _, _, height, width = img.shape

    # Calcolo del numero di tile necessari
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # Calcolo del padding necessario per adattare l'immagine alle dimensioni dei tile
    pad_right = (tile_size - (width % tile_size)) % tile_size
    pad_bottom = (tile_size - (height % tile_size)) % tile_size

    # Padding dell'immagine se necessario
    if pad_right != 0 or pad_bottom != 0:
        img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), 'constant', 0)

    # Creazione di un output tensor vuoto
    b, c, h, w = img.shape
    output = torch.empty((b, c, h * scale, w * scale), dtype=torch.float32, device=self.device).contiguous()

    # Selezione della funzione di upscaling in base al tipo
    upscaler_functions = {
        'RealEsrgan-x2-Plus': run_realesrganx2,
        'RealEsrgan-x4-Plus': run_realesrganx4,
        'BSRGan-x2': run_bsrganx2,
        'BSRGan-x4': run_bsrganx4,
        'UltraSharp-x4': run_ultrasharpx4,
        'UltraMix-x4': run_ultramixx4,
        'RealEsr-General-x4v3': run_realesrx4v3
    }

    fn_upscaler = upscaler_functions.get(enhancer_type)

    if not fn_upscaler:  # Se il tipo di enhancer non è valido
        if pad_right != 0 or pad_bottom != 0:
            img = v2.functional.crop(img, 0, 0, height, width)
        return img

    with torch.no_grad():  # Disabilita il calcolo del gradiente
        # Elaborazione dei tile
        for j in range(tiles_y):
            for i in range(tiles_x):
                x_start, y_start = i * tile_size, j * tile_size
                x_end, y_end = x_start + tile_size, y_start + tile_size

                # Estrazione del tile di input
                input_tile = img[:, :, y_start:y_end, x_start:x_end].contiguous()
                output_tile = torch.empty((input_tile.shape[0], input_tile.shape[1], input_tile.shape[2] * scale, input_tile.shape[3] * scale), dtype=torch.float32, device=self.device).contiguous()

                # Upscaling del tile
                fn_upscaler(self, input_tile, output_tile)

                # Inserimento del tile upscalato nel tensor di output
                output_y_start, output_x_start = y_start * scale, x_start * scale
                output_y_end, output_x_end = output_y_start + output_tile.shape[2], output_x_start + output_tile.shape[3]
                output[:, :, output_y_start:output_y_end, output_x_start:output_x_end] = output_tile

        # Ritaglio dell'output per rimuovere il padding aggiunto
        if pad_right != 0 or pad_bottom != 0:
            output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

    return output

def run_realesrganx2(self: 'ModelsProcessor', image, output):
    if not self.models['RealEsrganx2Plus']:
        self.models['RealEsrganx2Plus'] = self.load_model('RealEsrganx2Plus')

    io_binding = self.models['RealEsrganx2Plus'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['RealEsrganx2Plus'].run_with_iobinding(io_binding)

def run_realesrganx4(self: 'ModelsProcessor', image, output):
    if not self.models['RealEsrganx4Plus']:
        self.models['RealEsrganx4Plus'] = self.load_model('RealEsrganx4Plus')

    io_binding = self.models['RealEsrganx4Plus'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['RealEsrganx4Plus'].run_with_iobinding(io_binding)

def run_realesrx4v3(self: 'ModelsProcessor', image, output):
    if not self.models['RealEsrx4v3']:
        self.models['RealEsrx4v3'] = self.load_model('RealEsrx4v3')

    io_binding = self.models['RealEsrx4v3'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['RealEsrx4v3'].run_with_iobinding(io_binding)

def run_bsrganx2(self: 'ModelsProcessor', image, output):
    if not self.models['BSRGANx2']:
        self.models['BSRGANx2'] = self.load_model('BSRGANx2')

    io_binding = self.models['BSRGANx2'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['BSRGANx2'].run_with_iobinding(io_binding)

def run_bsrganx4(self: 'ModelsProcessor', image, output):
    if not self.models['BSRGANx4']:
        self.models['BSRGANx4'] = self.load_model('BSRGANx4')

    io_binding = self.models['BSRGANx4'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['BSRGANx4'].run_with_iobinding(io_binding)

def run_ultrasharpx4(self: 'ModelsProcessor', image, output):
    if not self.models['UltraSharpx4']:
        self.models['UltraSharpx4'] = self.load_model('UltraSharpx4')

    io_binding = self.models['UltraSharpx4'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['UltraSharpx4'].run_with_iobinding(io_binding)

def run_ultramixx4(self: 'ModelsProcessor', image, output):
    if not self.models['UltraMixx4']:
        self.models['UltraMixx4'] = self.load_model('UltraMixx4')

    io_binding = self.models['UltraMixx4'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['UltraMixx4'].run_with_iobinding(io_binding)

def run_deoldify_artistic(self: 'ModelsProcessor', image, output):
    if not self.models['DeoldifyArt']:
        self.models['DeoldifyArt'] = self.load_model('DeoldifyArt')

    io_binding = self.models['DeoldifyArt'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['DeoldifyArt'].run_with_iobinding(io_binding)

def run_deoldify_stable(self: 'ModelsProcessor', image, output):
    if not self.models['DeoldifyStable']:
        self.models['DeoldifyStable'] = self.load_model('DeoldifyStable')

    io_binding = self.models['DeoldifyStable'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['DeoldifyStable'].run_with_iobinding(io_binding)

def run_deoldify_video(self: 'ModelsProcessor', image, output):
    if not self.models['DeoldifyVideo']:
        self.models['DeoldifyVideo'] = self.load_model('DeoldifyVideo')

    io_binding = self.models['DeoldifyVideo'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['DeoldifyVideo'].run_with_iobinding(io_binding)

def run_ddcolor_artistic(self: 'ModelsProcessor', image, output):
    if not self.models['DDColorArt']:
        self.models['DDColorArt'] = self.load_model('DDColorArt')

    io_binding = self.models['DDColorArt'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['DDColorArt'].run_with_iobinding(io_binding)

def run_ddcolor(self: 'ModelsProcessor', image, output):
    if not self.models['DDcolor']:
        self.models['DDcolor'] = self.load_model('DDcolor')

    io_binding = self.models['DDcolor'].io_binding()
    io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
    io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

    if self.device == "cuda":
        torch.cuda.synchronize()
    elif self.device != "cpu":
        self.syncvec.cpu()
    self.models['DDcolor'].run_with_iobinding(io_binding)