# Sketch

This tool allows the user to sketch on the canvas and allow our trained model to determine what number you have written.

![Sketch](Sketch.apng)

The tool is written in PyQt using the qtpy module to allow for cross compatibility between  PyQt5, and PySide2.

The GUI is generated using the Qt Designer tool and the .ui file is converted to a .py file using the pyuic5 tool. The generated .py file is then imported into the main script.

The simple SketchWidget can be re-used in other projects by importing the SketchWidget class, it currently allows black on white drawing but changing this is quite simple.

The GraphWidget is a custom widget that allows the user to use Matplotlib to plot a graph in a Qt application and this can be modified to suit your needs.

## Controls

The user interface allows you to select either a Center of Mass image or Bounding Box crop. The border of the bounding box is controlled by the slider.  The small 28x28 image is what is actually fed to the model for processing. 

The slider for pen size will change the size of the pen used to draw on the canvas, and the space bar or clear button will clear the canvas. 

When the mouse is released the current image is send to the prediction model and the result is displayed in the LCD display.

The graph shows the relative probabilities of the prediction model for each number 0-9. 

