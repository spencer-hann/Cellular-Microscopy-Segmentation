import numpy as np

from src.data import data
from src.data import Image
from PIL import ImageShow
import matplotlib.pyplot as plt



if __name__ == "__main__":
    image = next(data.load_images())
    mask = data.load_nuclei_by_index(image.index)
    image = image.add_mask(mask, "nuclei")
    #image = Image.MaskedImage(image, mask, ('nuclei',))
    mask = data.load_mitochondria_by_index(image.index)
    image = image.add_mask(mask, "mitochondria")

    print('colored_image')
    image.highlight_plot(show=True)

    #image = next(data.load_images())
    #print("raw image")
    #plt.imshow(image.im); plt.show()
    print("exiting...")

