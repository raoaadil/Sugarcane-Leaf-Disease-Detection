# AI-Driven Sugarcane Diseases Detection, Classification & Remedy Suggestion

## Project Overview

This project enables identification of abnormalities on sugarcane leaves, accurate classification of diseases, and suggesting appropriate remedies with direct purchase options.

***

## Dataset Details

- Dataset collected from local sugarcane farms.
- Total 224 labeled images.
- Three classes:
  - Healthy: 75 images
  - Red Rot: 74 images
  - Red Rust: 75 images

***

## Methodology

- Utilized Transfer Learning with DenseNet201 architecture for feature extraction.
- Final classification performed using Support Vector Machine (SVM).
- Achieved training accuracy of **98%** and validation accuracy of **97.78%**.

***

## Running the Project

1. Install Anaconda Python Package  
   

2. Open Anaconda Prompt and navigate to project folder:
   ```
   cd path_to_your_project_folder

   Example if project is present in D drive use following command
	C:\>D:
    	C:\>cd D:\Sugarcane-Leaf-Disease-Detection
   ```

3. Create and activate a virtual environment:
   ```
   conda create -n sdd python=3.9
   conda activate sdd
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. (Optional) To train the model:
   ```
   jupyter notebook
   ```
   Open and run all cells in `Sugarcane-Leaf-Disease-Detection.ipynb`.

6. To launch the web app:
   ```
   python app.py
   ```
