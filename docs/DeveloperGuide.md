# How to Set Up the Conda Environment for YOLO Project

## 🔽 Step 1: Clone the Repo
- Lauinch conda terminal or your preferred terminal application with conda installed.
- git clone https://github.com/danielmusselwhite/NuclearLobeDetection
- cd into the cloned directory:
  - cd NuclearLobeDetection

## 📦 Step 2: Create the Environment

- Use the provided environment.yml file to create the Conda environment:
  - **conda env create -f environment.yml**
- This will create an environment (usually with the same name yolo_env).

## ✅ Step 3: Activate the Environment
- conda activate yolo_env

- Now you're ready to use the environment!

## 🛠 Troubleshooting

- If the name yolo_env already exists, Conda will show an error. You can rename the environment during creation:
  - **conda env create -f environment.yml -n my_custom_env**

- If you want to update an existing environment (eg if the environment.yml file has changed), use:
    - conda env update -f environment.yml --prune