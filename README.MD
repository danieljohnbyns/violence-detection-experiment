# Violence Detection AI (Experimental)
> **Note**
> Use python 3.11 to run this project.

This is an experimental project to detect violence in images and videos using AI. The project is still in the early stages of development and is not yet ready for production use.

## File Structure
```
violence-detection-experiment/
├── README.MD
├── dataset/
│   ├── non-violence/
│   │   ├── 1.jpg
│   │   ├── 2.mp4
│   │   └── ...
│   └── violence/
│       ├── 1.mp4
│       ├── 2.jpg
│       └── ...
├── trained_model/
├── README.MD
├── .gitignore
├── requirements.txt
├── test_image.jpg
├── test_image2.jpg
├── detect_image.py
├── evaluator.py
└── trainer.py
```

## Usage
1. Install the required packages using the following command:
```
pip install -r requirements.txt
```

2. Train the model using the following command:
```
python trainer.py
```

3. Evaluate the model using the following command:
```
python evaluator.py
```

4. Detect violence in an image using the following command:
```
python detect_image.p
```


