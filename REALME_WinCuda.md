# 🎭 VisoMaster v0.1.6 — Guía de Instalación y Uso

> Alternativa ligera a FaceFusion | Gratis | Open Source  
> Repo oficial: `github.com/visomaster/VisoMaster`

---

## ⚙️ Requisitos

- Windows 10/11 64-bit
- GPU NVIDIA con mínimo **6GB VRAM** (RTX 3050 ✅)
- Miniconda / Anaconda instalado
- Git instalado

---

## 📦 Instalación (Anaconda Prompt)

### Paso 1 — Clonar el repositorio

```bash
cd /d "F:\PROYECTO IDE\AAA Proyectos ALD\Codelatin"
git clone https://github.com/visomaster/VisoMaster.git
cd VisoMaster
```

### Paso 2 — Crear entorno conda

```bash
conda create -n visomaster python=3.10.13 -y
conda activate visomaster
```

> ✅ Debes ver `(visomaster)` al inicio de la línea

### Paso 3 — Instalar CUDA y dependencias

```bash
conda install -c nvidia/label/cuda-12.4.1 cuda-runtime -y
conda install -c conda-forge cudnn -y
conda install scikit-image -y
pip install -r requirements_cu124.txt
```

### Paso 4 — Descargar modelos (~6-9 GB)

```bash
python download_models.py
```

> ⚠️ Tarda varios minutos según tu internet. No cierres la ventana.

### Paso 5 — Corregir error KMP (solo primera vez)

Abre `Start.bat` con el Bloc de notas y agrega esta línea **antes** de `python main.py`:

```bat
SET KMP_DUPLICATE_LIB_OK=TRUE
```

El archivo debe quedar así:

```bat
call conda activate visomaster
call app/ui/core/convert_ui_to_py.bat
SET APP_ROOT=%~dp0
SET APP_ROOT=%APP_ROOT:~0,-1%
SET DEPENDENCIES=%APP_ROOT%\dependencies
echo %DEPENDENCIES%
SET PATH=%DEPENDENCIES%;%PATH%
SET KMP_DUPLICATE_LIB_OK=TRUE
python main.py
pause
```

---

## Iniciar VisoMaster

```bash
cd /d "F:\PROYECTO IDE\AAA Proyectos ALD\Codelatin\VisoMaster"
conda activate visomaster
Start.bat
```

---

## 🎬 Cómo hacer un Face Swap

### 1. Cargar archivos
- **File → Load Target Image/Video Files** → selecciona tu video
- **File → Load Source Image Files** → selecciona la foto de la cara

### 2. Detectar y activar swap
- Clic en **Find Faces**
- Clic en **Swap Faces**
- Verifica el preview en pantalla

### 3. Grabar/exportar
- Presiona **`R`** para iniciar la grabación
- El video se reproduce y graba automáticamente frame a frame
- Presiona **`R`** de nuevo para detener
- El archivo se guarda en tu **Output Directory**

---

## Configuración recomendada (Settings)

| Opción | Valor recomendado |
|--------|------------------|
| Output Directory | Carpeta de tu elección |
| Providers Priority | **CUDA** |
| Number of Threads | **2** (evita crashes con 6GB VRAM) |
| Auto Swap | **Activado** |
| Face Detect Model | RetinaFace |

---

## Configuración Face Swap recomendada

| Opción | Valor recomendado |
|--------|------------------|
| Swapper Model | **InStyleSwapper256 Version A** |
| Swapper Resolution | 256 |
| Face Parser Mask | **Activado** |
| Similarity Threshold | 60 |

---

## Atajos de teclado útiles

| Tecla | Acción |
|-------|--------|
| `R` | Iniciar/Detener grabación |
| `Space` | Play/Pause video |
| `S` | Activar/Desactivar swap |
| `F` | Agregar marcador de frame |
| `V` | Avanzar 1 frame |
| `C` | Retroceder 1 frame |
| `F11` | Pantalla completa |

---

## VisoMaster vs FaceFusion

| | FaceFusion | VisoMaster |
|--|-----------|------------|
| Interfaz | Browser (Gradio) | App de escritorio |
| Instalación | Compleja | Más simple |
| Audio en output | ⚠️ A veces se pierde | ✅ Automático |
| Velocidad GPU | CUDA estándar | TensorRT optimizado |
| Preview en vivo | Limitado | ✅ Tiempo real |
| Precio | Gratis | Gratis |

---

## Notas importantes

- La **primera vez** que usas un modelo, VisoMaster lo compila para tu GPU (TensorRT). Tarda varios minutos — es normal.
- No uses más de **2 hilos** con 6GB de VRAM para evitar crashes.
- El audio del video original **se conserva automáticamente**.
- Los modelos ocupan ~6-9GB en disco.

---

## Estructura de carpetas

```
VisoMaster/
├── model_assets/       # Modelos de IA descargados
├── dependencies/       # Dependencias del sistema
├── app/                # Código de la aplicación
├── output/             # Videos exportados
├── Start.bat           # ← Ejecutar esto para iniciar
└── download_models.py  # Script de descarga de modelos
```

---

*Guía creada para RTX 3050 6GB Laptop — Lima, Perú 🇵🇪*