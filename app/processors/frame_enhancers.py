import torch
import numpy as np
import math
from torchvision.transforms import v2
from functools import partial
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

class FrameEnhancers:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
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
        output = torch.empty((b, c, h * scale, w * scale), dtype=torch.float32, device=self.models_processor.device).contiguous()

        # Selezione della funzione di upscaling in base al tipo
        upscaler_functions = {
            'RealEsrgan-x2-Plus': self.run_realesrganx2,
            'RealEsrgan-x4-Plus': self.run_realesrganx4,
            'BSRGan-x2': self.run_bsrganx2,
            'BSRGan-x4': self.run_bsrganx4,
            'UltraSharp-x4': self.run_ultrasharpx4,
            'UltraMix-x4': self.run_ultramixx4,
            'RealEsr-General-x4v3': self.run_realesrx4v3
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
                    output_tile = torch.empty((input_tile.shape[0], input_tile.shape[1], input_tile.shape[2] * scale, input_tile.shape[3] * scale), dtype=torch.float32, device=self.models_processor.device).contiguous()

                    # Upscaling del tile
                    fn_upscaler(input_tile, output_tile)

                    # Inserimento del tile upscalato nel tensor di output
                    output_y_start, output_x_start = y_start * scale, x_start * scale
                    output_y_end, output_x_end = output_y_start + output_tile.shape[2], output_x_start + output_tile.shape[3]
                    output[:, :, output_y_start:output_y_end, output_x_start:output_x_end] = output_tile

            # Ritaglio dell'output per rimuovere il padding aggiunto
            if pad_right != 0 or pad_bottom != 0:
                output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

        return output

    def run_realesrganx2(self, image, output):
        if not self.models_processor.models['RealEsrganx2Plus']:
            self.models_processor.models['RealEsrganx2Plus'] = self.models_processor.load_model('RealEsrganx2Plus')

        io_binding = self.models_processor.models['RealEsrganx2Plus'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['RealEsrganx2Plus'].run_with_iobinding(io_binding)

    def run_realesrganx4(self, image, output):
        if not self.models_processor.models['RealEsrganx4Plus']:
            self.models_processor.models['RealEsrganx4Plus'] = self.models_processor.load_model('RealEsrganx4Plus')

        io_binding = self.models_processor.models['RealEsrganx4Plus'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['RealEsrganx4Plus'].run_with_iobinding(io_binding)

    def run_realesrx4v3(self, image, output):
        if not self.models_processor.models['RealEsrx4v3']:
            self.models_processor.models['RealEsrx4v3'] = self.models_processor.load_model('RealEsrx4v3')

        io_binding = self.models_processor.models['RealEsrx4v3'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['RealEsrx4v3'].run_with_iobinding(io_binding)

    def run_bsrganx2(self, image, output):
        if not self.models_processor.models['BSRGANx2']:
            self.models_processor.models['BSRGANx2'] = self.models_processor.load_model('BSRGANx2')

        io_binding = self.models_processor.models['BSRGANx2'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['BSRGANx2'].run_with_iobinding(io_binding)

    def run_bsrganx4(self, image, output):
        if not self.models_processor.models['BSRGANx4']:
            self.models_processor.models['BSRGANx4'] = self.models_processor.load_model('BSRGANx4')

        io_binding = self.models_processor.models['BSRGANx4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['BSRGANx4'].run_with_iobinding(io_binding)

    def run_ultrasharpx4(self, image, output):
        if not self.models_processor.models['UltraSharpx4']:
            self.models_processor.models['UltraSharpx4'] = self.models_processor.load_model('UltraSharpx4')

        io_binding = self.models_processor.models['UltraSharpx4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['UltraSharpx4'].run_with_iobinding(io_binding)

    def run_ultramixx4(self, image, output):
        if not self.models_processor.models['UltraMixx4']:
            self.models_processor.models['UltraMixx4'] = self.models_processor.load_model('UltraMixx4')

        io_binding = self.models_processor.models['UltraMixx4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['UltraMixx4'].run_with_iobinding(io_binding)

    def run_deoldify_artistic(self, image, output):
        if not self.models_processor.models['DeoldifyArt']:
            self.models_processor.models['DeoldifyArt'] = self.models_processor.load_model('DeoldifyArt')

        io_binding = self.models_processor.models['DeoldifyArt'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['DeoldifyArt'].run_with_iobinding(io_binding)

    def run_deoldify_stable(self, image, output):
        if not self.models_processor.models['DeoldifyStable']:
            self.models_processor.models['DeoldifyStable'] = self.models_processor.load_model('DeoldifyStable')

        io_binding = self.models_processor.models['DeoldifyStable'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['DeoldifyStable'].run_with_iobinding(io_binding)

    def run_deoldify_video(self, image, output):
        if not self.models_processor.models['DeoldifyVideo']:
            self.models_processor.models['DeoldifyVideo'] = self.models_processor.load_model('DeoldifyVideo')

        io_binding = self.models_processor.models['DeoldifyVideo'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['DeoldifyVideo'].run_with_iobinding(io_binding)

    def run_ddcolor_artistic(self, image, output):
        if not self.models_processor.models['DDColorArt']:
            self.models_processor.models['DDColorArt'] = self.models_processor.load_model('DDColorArt')

        io_binding = self.models_processor.models['DDColorArt'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['DDColorArt'].run_with_iobinding(io_binding)

    def run_ddcolor(self, image, output):
        if not self.models_processor.models['DDcolor']:
            self.models_processor.models['DDcolor'] = self.models_processor.load_model('DDcolor')

        io_binding = self.models_processor.models['DDcolor'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['DDcolor'].run_with_iobinding(io_binding)