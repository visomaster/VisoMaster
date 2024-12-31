import numpy as np
import torch
from collections import OrderedDict
import platform
from queue import Queue
from threading import Lock

try:
    from torch.cuda import nvtx
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    pass

# Dizionario per la conversione dei tipi di dati numpy a torch
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

if 'trt' in globals():
    # Creazione di un'istanza globale di logger di TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
else:
    TRT_LOGGER = {}

# imported from https://github.com/warmshao/FasterLivePortrait/blob/master/src/models/predictor.py
# adjusted to work with TensorRT 10.3.0
class TensorRTPredictor:
    """
    Implements inference for the TensorRT engine with a pool of execution contexts.
    """

    def __init__(self, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        :param pool_size: The size of the pool of execution contexts.
        """
        global TRT_LOGGER

        # Inizializzazione del modello TensorRT
        self.engine = None
        self.context_pool = None
        self.lock = Lock()
        self.device = kwargs.get("device", 'cuda')

        custom_plugin_path = kwargs.get("custom_plugin_path", None)
        if custom_plugin_path is not None:
            # Carica il plugin personalizzato solo una volta
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)

        # Load TRT engine
        engine_path = kwargs.get("model_path", None)
        self.debug = kwargs.get("debug", False)
        self.pool_size = kwargs.get("pool_size", 10)
        assert engine_path, f"model:{engine_path} must exist!"

        # Caricamento dell'engine TensorRT
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine

        # Setup I/O bindings e contesto
        self.inputs = []
        self.outputs = []
        self.tensors = OrderedDict()

        # Gestione dei tensori dinamici
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            binding = {
                "index": idx,
                "name": name,
                "dtype": dtype,
                "shape": list(shape)
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        self.allocate_max_buffers()

        # Creazione del pool di contesti di esecuzione
        self.context_pool = Queue(maxsize=self.pool_size)
        for _ in range(self.pool_size):
            self.context_pool.put(self.engine.create_execution_context())

    def allocate_max_buffers(self):
        nvtx.range_push("allocate_max_buffers")
        # Supporto per batch dinamico
        batch_size = 1
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine.get_tensor_name(idx)
            shape = self.engine.get_tensor_shape(binding)
            is_input = self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
            if -1 in shape:
                if is_input:
                    shape = self.engine.get_tensor_profile_shape(binding, 0)[-1]
                    batch_size = shape[0]
                else:
                    shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=self.device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {o['name']} -> {o['shape']} -> {o['dtype']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']} -> {o['shape']} -> {o['dtype']}")
        return specs

    def adjust_buffer(self, feed_dict, context):
        """
        Adjust input buffer sizes and set input shapes in the given execution context.
        :param feed_dict: A dictionary of inputs as numpy arrays.
        :param context: The TensorRT execution context to set input shapes.
        """
        nvtx.range_push("adjust_buffer")
        for name, buf in feed_dict.items():
            input_tensor = self.tensors[name]
            current_shape = list(buf.shape)
            slices = tuple(slice(0, dim) for dim in current_shape)
            input_tensor[slices].copy_(buf)
            # Imposta la forma di input nel contesto fornito
            context.set_input_shape(name, current_shape)
        nvtx.range_pop()

    def predict(self, feed_dict):
        """
        Execute inference on a batch of images in synchronous mode using execute_v2.
        :param feed_dict: A dictionary of inputs as numpy arrays.
        :return: A dictionary of outputs as PyTorch tensors.
        """
        # Ottieni un contesto dal pool
        with self.lock:
            context = self.context_pool.get()

        try:
            nvtx.range_push("set_tensors")
            # Passa il contesto a adjust_buffer
            self.adjust_buffer(feed_dict, context)

            for name, tensor in self.tensors.items():
                assert tensor.dtype == torch.float32, f"Tensor '{name}' should be torch.float32 but is {tensor.dtype}"
                context.set_tensor_address(name, tensor.data_ptr())

            nvtx.range_pop()

            # Prepara i binding per execute_v2()
            bindings = [tensor.data_ptr() for tensor in self.tensors.values()]

            # Esecuzione sincrona con execute_v2()
            nvtx.range_push("execute")
            noerror = context.execute_v2(bindings)
            if not noerror:
                raise ValueError("ERROR: inference failed.")
            nvtx.range_pop()

            return self.tensors

        finally:
            # Sincronizza il flusso CUDA prima di restituire il contesto
            torch.cuda.synchronize()  # Sincronizza il default stream
            # Restituisci il contesto al pool dopo l'uso
            with self.lock:
                self.context_pool.put(context)

    def predict_async(self, feed_dict, stream):
        """
        Execute inference on a batch of images in asynchronous mode using execute_async_v3.
        :param feed_dict: A dictionary of inputs as numpy arrays.
        :param stream: A CUDA stream for asynchronous execution.
        :return: A dictionary of outputs as PyTorch tensors.
        """
        # Ottieni un contesto dal pool
        with self.lock:
            context = self.context_pool.get()

        try:
            nvtx.range_push("set_tensors")
            # Passa il contesto a adjust_buffer
            self.adjust_buffer(feed_dict, context)

            for name, tensor in self.tensors.items():
                assert tensor.dtype == torch.float32, f"Tensor '{name}' should be torch.float32 but is {tensor.dtype}"
                context.set_tensor_address(name, tensor.data_ptr())

            nvtx.range_pop()

            # Creare un evento CUDA per tracciare il consumo dell'input
            input_consumed_event = torch.cuda.Event()

            # Impostare l'evento per l'input consumato
            context.set_input_consumed_event(input_consumed_event.cuda_event)
        
            # Esecuzione asincrona con execute_async_v3()
            nvtx.range_push("execute_async")
            noerror = context.execute_async_v3(stream.cuda_stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")
            nvtx.range_pop()

            # Sincronizzare l'evento dell'input consumato (se necessario)
            input_consumed_event.synchronize()

            return self.tensors

        finally:
            # Sincronizza il flusso CUDA prima di restituire il contesto
            if stream != torch.cuda.current_stream():
                stream.synchronize()  # Sincronizza lo stream personalizzato
            else:
                torch.cuda.synchronize()  # Sincronizza il default stream

            # Restituisci il contesto al pool dopo l'uso
            with self.lock:
                self.context_pool.put(context)

    def cleanup(self):
        """
        Clean up all resources associated with the TensorRTPredictor.
        This method should be called explicitly before deleting the object.
        """
        # Pulisci l'engine TensorRT
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine  # Libera l'engine di TensorRT
            self.engine = None  # Imposta a None per assicurarti che il GC lo raccolga

        # Pulisci il pool di contesti di esecuzione
        if hasattr(self, 'context_pool') and self.context_pool is not None:
            while not self.context_pool.empty():
                context = self.context_pool.get()
                del context  # Libera ogni contesto
            self.context_pool = None  # Imposta a None per il GC

        # Imposta gli attributi su None per garantire la pulizia
        self.inputs = None
        self.outputs = None
        self.tensors = None
        self.pool_size = None

    def __del__(self):
        # Richiama il metodo cleanup nel distruttore per maggiore sicurezza
        self.cleanup()