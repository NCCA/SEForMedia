# Realtime ASL Translator

This is work in progress and needs lots more fine tuning. The idea is to use a webcam to capture images of ASL signs and translate them into text. The current implementation is a simple demo that uses a pre-trained model from our notebooks in the above directory. The model is a simple CNN that was trained on the ASL alphabet dataset from kaggle. The model is loaded and used to predict the sign from the webcam feed. Whilst the model is good, I need to add hand detection and tracking and better image processing to send the correct image to the model.


The GUI uses [PySide6](https://doc.qt.io/qtforpython-6/) and the demo also needs [OpenCV](https://pypi.org/project/opencv-python/) for the webcam, which isn't in the project dependencies, so add it when you run it from this folder.

```bash
uv run --with opencv-python Capture.py
```

If you change `MainWindow.ui` re-generate `MainWindow.py` with `uv run pyside6-uic MainWindow.ui -o MainWindow.py`.
