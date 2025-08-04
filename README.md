# cv\_nish\_gif: Create GIFs from Nishika 8000 Scans

`cv_nish_gif` is a Python script designed to process images captured with quadrascopic film cameras, such as the Nishika 8000. It takes a single image containing N frames or N images containing 1 frame each, separates them, and generates a seamless, looping GIF.

The program uses computer vision techniques to ensure the frames are properly aligned and sequenced, creating a smooth and compelling animated effect.

## Features

  * **Quadrascopic Image Processing**: Automatically splits a single scan into its four individual frames.

  * **GIF Generation**: Combines the four frames into a single GIF.

## Prerequisites

Before running the script, you will need to have Python 3 installed. The program relies on a few external libraries, which can be installed using `pip`.

```
pip install opencv-python numpy imageio

```

  * `opencv-python`: For image processing and frame splitting.

  * `numpy`: A core dependency for numerical operations, used by OpenCV and other libraries.

  * `imageio`: Used to create the final GIF file from the processed frames.

## Usage

The script is run from the command line and requires an input image file and an output filename.

### Basic Usage

To create a GIF with default settings, simply specify the input image and the desired output file.

```
python main.py --frames 4 --input path/to/your/scan.jpg --output output.gif
```

or if you have 4 images named scan_1.jpg, scan_2.jpg, scan_3.jpg, scan_4.jpg

```
python main.py --frames 4 --pattern path/to/your/scan --output output.gif
```

This will pop up a preview window, where you may select the point of interest.

### Options

-i OR -p is required.

The following command-line arguments are available for customizing the output:

  * `-i`, `--input`: Path to the input image file containing the four frames.

  * `-p`, `--pattern`: Path to input files of pattern <myfilename>_i.jpg, from i between 1 and n'

  * `-o`, `--output`: (default 'out.gif') Path and filename for the output GIF.

  * `-y`, `--height`: (default 1400) scaled height in pixels of the output image. (width is inferred).

  * `-z`, `--boomerang`: (default true) Whether the image should boomerang back and forth

### Example

To create a GIF named `my_nishika_gif.gif` from `nishika_scan.png` with a frame duration of 200ms:

```
python main.py --frames 4 --input path/to/your/scan.jpg --output output.gif
```
