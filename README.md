
<img src="app/ui/core/media/visomaster_full.png" height="250"/>

# VisoMaster 
VisoMaster is a powerful yet easy-to-use tool for face swapping and editing in images and videos. It utilizes AI to produce natural-looking results with minimal effort, making it ideal for both casual users and professionals.  

---

## **Features**  
✅ High-quality **AI-powered face swapping** for images and videos  
✅ **Easy-to-use** interface with simple controls  
✅ Supports **multiple formats** for input and output  
✅ Efficient processing with **GPU acceleration** (CUDA support)  
✅ **Customizable** models and fine-tuning options  


## **Installation Guide (Nvidia)**

Follow the steps below to install and run **VisoMaster** on your system.

## **Prerequisites**
Before proceeding, ensure you have the following installed on your system:
- **Git** ([Download](https://git-scm.com/downloads))
- **Miniconda** ([Download](https://www.anaconda.com/download))

---

## **Installation Steps**

### **1. Clone the Repository**  
Open a terminal or command prompt and run:  
```sh
git clone https://github.com/visomaster/VisoMaster.git
```
```sh
cd VisoMaster
```

### **2. Create and Activate a Conda Environment**  
```sh
conda create -n visomaster python=3.10.13 -y
```
```sh
conda activate visomaster
```

### **3. Install CUDA and cuDNN**  
```sh
conda install -c nvidia/label/cuda-12.4.1 cuda-runtime
```
```sh
conda install -c conda-forge cudnn
```

### **4. Install Additional Dependencies**  
```sh
conda install scikit-image
```
```sh
pip install -r requirements_cu124.txt
```

### **5. Download Models and Other Dependencies**  
1. Download all the required models
```sh
python download_models.py
```
2. Download all the files from this [page](https://github.com/visomaster/visomaster-assets/releases/tag/v0.1.0_dp) and copy it to the ***dependencies/*** folder.

  **Note**: You do not need to download the Source code (zip) and Source code (tar.gz) files 
### **6. Run the Application**  
Once everything is set up, start the application by opening the **Start.bat** file.
On Linux just run `python main.py`.
---

## **Troubleshooting**
- If you face CUDA-related issues, ensure your GPU drivers are up to date.
- For missing models, double-check that all models are placed in the correct directories.

## [Join Discord](https://discord.gg/5rx4SQuDbp)

Now you're ready to use **VisoMaster**! 🚀
