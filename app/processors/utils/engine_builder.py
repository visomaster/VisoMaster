# pylint: disable=no-member

import os
import sys
import logging
import platform
import ctypes
from pathlib import Path

try:
    import tensorrt as trt
except ModuleNotFoundError:
    pass

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

if 'trt' in globals():
    # Creazione di un'istanza globale di logger di TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.INFO) # pylint: disable=no-member
else:
    TRT_LOGGER = {}

# imported from https://github.com/warmshao/FasterLivePortrait/blob/master/scripts/onnx2trt.py
# adjusted to work with TensorRT 10.3.0
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, custom_plugin_path=None, builder_optimization_level=3):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param custom_plugin_path: Path to the custom plugin library (DLL or SO).
        """
        if verbose:
            TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

        # Inizializza i plugin di TensorRT
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

        # Costruisce il builder di TensorRT e la configurazione usando lo stesso logger
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        # Imposta il limite di memoria del pool di lavoro a 3 GB
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 * (2 ** 30))  # 3 GB

        # Imposta il livello di ottimizzazione del builder (se fornito)
        self.config.builder_optimization_level = builder_optimization_level

        # Crea un profilo di ottimizzazione, se necessario
        profile = self.builder.create_optimization_profile()
        self.config.add_optimization_profile(profile)

        self.batch_size = None
        self.network = None
        self.parser = None

        # Carica plugin personalizzati se specificato
        if custom_plugin_path is not None:
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: %s", onnx_path)
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for net_input in inputs:
            self.batch_size = net_input.shape[0]
            log.info("Input '%s' with shape %s and dtype %s", net_input.name, net_input.shape, net_input.dtype)
        for net_output in outputs:
            log.info("Output %s' with shape %s and dtype %s", net_output.name, net_output.shape, net_output.dtype)

    def create_engine(self, engine_path, precision):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building %s Engine in %s", precision, engine_path)

        # Forza TensorRT a rispettare i vincoli di precisione
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        # Costruzione del motore serializzato
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)

        # Verifica che il motore sia stato serializzato correttamente
        if serialized_engine is None:
            raise RuntimeError("Errore nella costruzione del motore TensorRT!")

        # Scrittura del motore serializzato su disco
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: %s", engine_path)
            f.write(serialized_engine)

def change_extension(file_path, new_extension, version=None):
    """
    Change the extension of the file path and optionally prepend a version.
    """
    # Remove leading '.' from the new extension if present
    new_extension = new_extension.lstrip('.')

    # Create the new file path with the version before the extension, if provided
    if version:
        new_file_path = Path(file_path).with_suffix(f'.{version}.{new_extension}')
    else:
        new_file_path = Path(file_path).with_suffix(f'.{new_extension}')

    return str(new_file_path)

def onnx_to_trt(onnx_model_path, trt_model_path=None, precision="fp16", custom_plugin_path=None, verbose=False):
    # The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'"

    if trt_model_path is None:
        trt_version = trt.__version__
        trt_model_path = change_extension(onnx_model_path, "trt", version=trt_version)
    builder = EngineBuilder(verbose=verbose, custom_plugin_path=custom_plugin_path)

    builder.create_network(onnx_model_path)
    builder.create_engine(trt_model_path, precision)
