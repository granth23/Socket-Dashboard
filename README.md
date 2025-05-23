# Flask Web App

This is a basic Flask web application running in a Conda environment.

---

## 🔧 Setup Instructions

### 1. Install Miniconda (Recommended)

Download and install Miniconda from:  
👉 https://docs.conda.io/en/latest/miniconda.html

Follow the platform-specific installation guide provided there.

---

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

---

### 3. Create and Activate the Conda Environment

Create the environment using the provided `environment.yaml`:

```bash
conda env create -f environment.yaml
```

Activate the environment:

```bash
conda activate flask-app
```

> Replace `flask-app` with the name defined in your `environment.yaml` if it’s different.

---

### 4. Run the Application

Start the Flask app:

```bash
python app.py
```

By default, the app will be available at:
🌐 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📝 Notes

* Make sure `app.py` is in the root directory or adjust the command accordingly.
* To deactivate the environment later, run:

  ```bash
  conda deactivate
  ```
