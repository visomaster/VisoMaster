import numpy as np
import torch
from collections import OrderedDict
import platform
from queue import Queue
from threading import Lock
from typing import Dict, Any, OrderedDict as OrderedDictType

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
    # Creazione di un’istanza globale di logger di TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
else:
    TRT_LOGGER = None


class TensorRTPredictor:
    """
    Implementa l'inferenza su un engine TensorRT, utilizzando un pool di execution context
    ognuno dei quali possiede i propri buffer per garantire la sicurezza in ambiente multithread.
    """

    def __init__(self, **kwargs) -> None:
        """
        :param model_path: Percorso al file dell'engine serializzato.
        :param pool_size: Numero di execution context da mantenere nel pool.
        :param custom_plugin_path: (Opzionale) percorso a eventuali plugin personalizzati.
        :param device: Device su cui allocare i tensori (default 'cuda').
        :param debug: Se True, stampa informazioni di debug.
        """
        self.device = kwargs.get("device", 'cuda')
        self.debug = kwargs.get("debug", False)
        self.pool_size = kwargs.get("pool_size", 10)

        # Caricamento del plugin personalizzato (se fornito)
        custom_plugin_path = kwargs.get("custom_plugin_path", None)
        if custom_plugin_path is not None:
            try:
                if platform.system().lower() == 'linux':
                    ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
                else:
                    # Su Windows eventualmente usare WinDLL o parametri specifici
                    ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
            except Exception as e:
                raise RuntimeError(f"Errore nel caricamento del plugin personalizzato: {e}")

        # Verifica che il percorso del modello sia fornito
        engine_path = kwargs.get("model_path", None)
        if not engine_path:
            raise ValueError("Il parametro 'model_path' è obbligatorio.")

        # Caricamento dell'engine TensorRT
        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            raise RuntimeError(f"Errore nella deserializzazione dell'engine: {e}")

        if self.engine is None:
            raise RuntimeError("La deserializzazione dell'engine è fallita.")

        # Setup delle specifiche di I/O (input e output)
        self.inputs = []
        self.outputs = []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            binding = {
                "index": idx,
                "name": name,
                "dtype": dtype,
                "shape": shape,
            }
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        if len(self.inputs) == 0 or len(self.outputs) == 0:
            raise RuntimeError("L'engine deve avere almeno un input e un output.")

        # Creazione del pool di execution context
        self.context_pool = Queue(maxsize=self.pool_size)
        # (Opzionale) Lock per eventuali operazioni critiche
        self.lock = Lock()
        for _ in range(self.pool_size):
            context = self.engine.create_execution_context()
            buffers = self._allocate_buffers()
            self.context_pool.put({"context": context, "buffers": buffers})

    def _allocate_buffers(self) -> OrderedDictType[str, torch.Tensor]:
        """
        Alloca un dizionario di tensori per tutti gli I/O del modello, tenendo conto di eventuali
        dimensioni dinamiche. Viene restituito un OrderedDict in cui la chiave è il nome del tensore.
        """
        nvtx.range_push("allocate_max_buffers")
        buffers = OrderedDict()
        # Batch size predefinito
        batch_size = 1
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = list(self.engine.get_tensor_shape(name))  # assicuriamoci di avere una lista
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if -1 in shape:
                if is_input:
                    # Ottiene la shape massima per il profilo 0
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)[-1]
                    shape = list(profile_shape)
                    batch_size = shape[0]
                else:
                    shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if dtype not in numpy_to_torch_dtype_dict:
                raise TypeError(f"Tipo numpy non supportato: {dtype}")
            tensor = torch.empty(tuple(shape),
                                 dtype=numpy_to_torch_dtype_dict[dtype],
                                 device=self.device)
            buffers[name] = tensor
        nvtx.range_pop()
        return buffers

    def input_spec(self) -> list:
        """
        Restituisce le specifiche degli input (nome, shape, dtype) utili per preparare gli array.
        """
        specs = []
        for i, inp in enumerate(self.inputs):
            specs.append((inp["name"], inp["shape"], inp["dtype"]))
            if self.debug:
                print(f"trt input {i} -> {inp['name']} -> {inp['shape']} -> {inp['dtype']}")
        return specs

    def output_spec(self) -> list:
        """
        Restituisce le specifiche degli output (nome, shape, dtype) utili per preparare gli array.
        """
        specs = []
        for i, out in enumerate(self.outputs):
            specs.append((out["name"], out["shape"], out["dtype"]))
            if self.debug:
                print(f"trt output {i} -> {out['name']} -> {out['shape']} -> {out['dtype']}")
        return specs

    def adjust_buffer(self, feed_dict: Dict[str, Any], context: Any, buffers: OrderedDictType[str, torch.Tensor]) -> None:
        """
        Regola le dimensioni dei buffer di input e copia i dati dal feed_dict nei tensori allocati.
        Se l’input è un array NumPy, lo converte in tensore Torch (sul device corretto).
        Imposta inoltre la shape di input nel contesto di esecuzione.
        """
        nvtx.range_push("adjust_buffer")
        for name, buf in feed_dict.items():
            if name not in buffers:
                raise KeyError(f"Input '{name}' non trovato nei buffer allocati.")
            input_tensor = buffers[name]
            # Converte in tensore se necessario
            if isinstance(buf, np.ndarray):
                buf_tensor = torch.from_numpy(buf).to(input_tensor.device)
            elif isinstance(buf, torch.Tensor):
                buf_tensor = buf.to(input_tensor.device)
            else:
                raise TypeError(f"Tipo di dato per '{name}' non supportato: {type(buf)}")
            current_shape = list(buf_tensor.shape)
            # Copia solo la porzione effettivamente utilizzata nel buffer preallocato
            slices = tuple(slice(0, dim) for dim in current_shape)
            input_tensor[slices].copy_(buf_tensor)
            # Imposta la shape dell'input nel contesto
            context.set_input_shape(name, current_shape)
        nvtx.range_pop()

    def predict(self, feed_dict: Dict[str, Any]) -> OrderedDictType[str, torch.Tensor]:
        """
        Esegue l'inferenza in modalità sincrona usando execute_v2().

        :param feed_dict: Dizionario di input (array numpy o tensori Torch).
        :return: Dizionario dei tensori (input e output) aggiornati.
        """
        pool_entry = self.context_pool.get()  # La Queue è thread-safe
        context = pool_entry["context"]
        buffers = pool_entry["buffers"]

        try:
            nvtx.range_push("set_tensors")
            self.adjust_buffer(feed_dict, context, buffers)
            # Imposta gli indirizzi dei buffer
            for name, tensor in buffers.items():
                # Se necessario, si può controllare che il tipo del tensore sia quello atteso
                context.set_tensor_address(name, tensor.data_ptr())
            nvtx.range_pop()

            # Prepara i binding (lista degli indirizzi dei buffer)
            bindings = [tensor.data_ptr() for tensor in buffers.values()]

            nvtx.range_push("execute")
            noerror = context.execute_v2(bindings)
            nvtx.range_pop()
            if not noerror:
                raise RuntimeError("ERROR: inference failed.")

            # (Opzionalmente, si potrebbero restituire solo gli output)
            return buffers

        finally:
            # Sincronizza il flusso CUDA prima di restituire il contesto
            torch.cuda.synchronize()
            self.context_pool.put(pool_entry)

    def predict_async(self, feed_dict: Dict[str, Any], stream: torch.cuda.Stream) -> OrderedDictType[str, torch.Tensor]:
        """
        Esegue l'inferenza in modalità asincrona usando execute_async_v3().

        :param feed_dict: Dizionario di input (array numpy o tensori Torch).
        :param stream: Un CUDA stream per l'esecuzione asincrona.
        :return: Dizionario dei tensori (input e output) aggiornati.
        """
        pool_entry = self.context_pool.get()
        context = pool_entry["context"]
        buffers = pool_entry["buffers"]

        try:
            nvtx.range_push("set_tensors")
            self.adjust_buffer(feed_dict, context, buffers)
            for name, tensor in buffers.items():
                context.set_tensor_address(name, tensor.data_ptr())
            nvtx.range_pop()

            # Creazione di un evento CUDA per monitorare il consumo dell'input
            input_consumed_event = torch.cuda.Event()
            context.set_input_consumed_event(input_consumed_event.cuda_event)

            nvtx.range_push("execute_async")
            noerror = context.execute_async_v3(stream.cuda_stream)
            nvtx.range_pop()
            if not noerror:
                raise RuntimeError("ERROR: inference failed.")

            input_consumed_event.synchronize()

            return buffers

        finally:
            # Sincronizza lo stream usato se diverso da quello corrente
            if stream != torch.cuda.current_stream():
                stream.synchronize()
            else:
                torch.cuda.synchronize()
            self.context_pool.put(pool_entry)

    def cleanup(self) -> None:
        """
        Libera tutte le risorse associate al TensorRTPredictor.
        Questo metodo deve essere chiamato esplicitamente prima di eliminare l'oggetto.
        """
        # Libera l'engine TensorRT
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
            self.engine = None

        # Libera il pool di execution context e relativi buffer
        if hasattr(self, 'context_pool') and self.context_pool is not None:
            while not self.context_pool.empty():
                pool_entry = self.context_pool.get()
                context = pool_entry.get("context", None)
                buffers = pool_entry.get("buffers", None)
                if context is not None:
                    del context
                if buffers is not None:
                    for t in buffers.values():
                        del t
            self.context_pool = None

        self.inputs = None
        self.outputs = None
        self.pool_size = None

    def __del__(self) -> None:
        # Per maggiore sicurezza, chiama cleanup nel distruttore
        self.cleanup()
