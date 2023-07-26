# Getting Started with vision-explanation-methods

This documentation provides a guide on how to get started with the vision-explanation-methods package. 

## Installation

To install the vision-explanation-methods package, you need to run the setup file. The setup file for the vision-explanation-methods package is located at `python/setup.py`. This file contains metadata including the name and version of the package. 

The dependencies required for the package are listed in the setup file. These include:

- numpy
- tqdm
- matplotlib<3.7.0
- ml_wrappers

To install the package, navigate to the directory containing the setup file and run the following command:

```bash
python setup.py install
```

## Usage

The vision-explanation-methods package provides a variety of explanation evaluation tools for image explanation methods. 

### Generating Saliency Maps

The `DRISE_runner.py` file contains the method for generating saliency maps for object detection models. The `get_drise_saliency_map` function is used to generate the saliency map. This function takes in parameters such as the image location, model, number of classes, save name, and maximum figures. 

The function returns a tuple of figure, location, and labels. The figure is a list of base64 encoded strings representing the saliency maps. The location is the path where the saliency maps are saved. The labels are a list of labels for the detected objects.

### Error Labeling

The `error_labeling.py` file defines the Error Labeling Manager class. This class is used to label errors in the predictions of the model. The types of errors that can be labeled include class name errors, duplicate detection errors, background errors, and missing detection errors.

## Contributing

Contributions to the vision-explanation-methods package are welcome. Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to the project.

## Code of Conduct

The vision-explanation-methods package has adopted the Microsoft Open Source Code of Conduct. Please refer to the `CODE_OF_CONDUCT.md` file for more information.

## Support

For help and questions about using the vision-explanation-methods package, please refer to the `SUPPORT.md` file. This file provides information on how to file issues and get help.

## License

The vision-explanation-methods package is licensed under the MIT License. Please refer to the `LICENSE.txt` file for more information.