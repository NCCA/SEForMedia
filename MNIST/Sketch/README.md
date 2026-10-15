# Sketch

This tool allows the user to sketch on the canvas and allow our trained model to determine what number you have written.

![Sketch](Sketch.apng)

The tool is written using [PySide6](https://doc.qt.io/qtforpython-6/). It loads the whole trained model (not just the weights) from `mnist_model.pth` in this folder, and is run from here.

```bash
uv run SketchNumbers.py
```

The GUI is designed in Qt Designer (`uv run pyside6-designer MainWindow.ui`) and the .ui file is converted to `MainWindow.py` using `pyside6-uic`. The generated .py file is then imported into the main script, so if you change the .ui file re-generate it like this.

```bash
uv run pyside6-uic MainWindow.ui -o MainWindow.py
```

The simple SketchWidget can be re-used in other projects by importing the SketchWidget class, it currently allows black on white drawing but changing this is quite simple.

The GraphWidget is a custom widget that allows the user to use Matplotlib to plot a graph in a Qt application and this can be modified to suit your needs.

## Controls

The user interface allows you to select either a Center of Mass image or Bounding Box crop. The border of the bounding box is controlled by the slider.  The small 28x28 image is what is actually fed to the model for processing. 

The slider for pen size will change the size of the pen used to draw on the canvas, and the space bar or clear button will clear the canvas. 

When the mouse is released the current image is send to the prediction model and the result is displayed in the LCD display.

The graph shows the relative probabilities of the prediction model for each number 0-9. 

