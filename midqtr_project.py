"""
DSC 20 Mid-Quarter Project
Name: Joshua Brusewitz
PID:  A16383198
"""

# Part 1: RGB Image #
class RGBImage:
    """
    Object representation of an image, with colors encoded in RGB format.
    """



    def __init__(self, pixels):
        """
        Creates a new RGBImage object.
        
        Parameters:
        pixels (list): Three-dimensional list. Represents an image by storing
        the intensity values for each color at each location on the image.
        """
        #Specifications clarify that this is guaranteed to be formatted right
        
        self.pixels = pixels

    def size(self):
        """
        Returns the size of the image as a tuple where the first entry is the
        number of rows and the second is the number of columns.
        """
        
        num_rows = len(self.pixels[0])
        num_cols = len(self.pixels[0][0])
        return (num_rows, num_cols)

    def get_pixels(self):
        """
        Returns a deep copy of the RGBImage object's pixels matrix.
        """
        
        return [[list(row) for row in channel] for channel in self.pixels]

    def copy(self):
        """
        Returns a deep copy of the RGBImage object.
        """
        
        return RGBImage(self.get_pixels())
        
        
    def validate_location_input (self, row, col):
        """
        Verifies that a given row / column pair is valid input for 
        get_pixel or set_pixel.
        """
        
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        elif (row >= self.size()[0] or row < 0 or
                col >= self.size()[1] or col < 0):
            raise ValueError()
            
    def get_pixel(self, row, col):
        """
        Returns the color of a pixel at a specified position on the image.
        """
            
        self.validate_location_input(row, col)
        
        RED_IDX, GRN_IDX, BLU_IDX = 0, 1, 2
        
        pxls = self.pixels
        red, green, blue = pxls[RED_IDX], pxls[GRN_IDX], pxls[BLU_IDX]
        return (red[row][col], green[row][col], blue[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Updates the color of a pixel at a specified position on the image.
        """
        
        self.validate_location_input(row, col)
        
        #RED_IDX, GRN_IDX, BLU_IDX = 0, 1, 2

        for channel, intensity in enumerate(new_color):
            if intensity != -1:
                self.pixels[channel][row][col] = intensity
    


# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    Class containing methods used to process and edit images.
    """

    @staticmethod
    def negate(image):
        """
        Takes an image and returns its negative (version with all colors
        inverted).
        """
        
        RGB_MAX = 255
        
        negated_image = [
            [[RGB_MAX - intensity for intensity in row] for row in channel] 
            for channel in image.get_pixels()
        ]
        return RGBImage(negated_image)

    @staticmethod
    def tint(image, color):
        """
        Takes an image and a color value and returns the image tinted by
        the color value.
        """
        
        RGB = (0, 1, 2)
        HALF = 2
        
        pxls = image.get_pixels()
        tinted_image = [
            [[(intensity + color[channel]) // HALF for intensity in row] 
            for row in pxls[channel]]
            for channel in RGB
        ]
        return RGBImage(tinted_image)

    @staticmethod
    def clear_channel(image, channel):
        """
        Takes an image and a color channel and returns a copy of the image in
        which all intensity values in that channel are reset to zero.
        """
        
        pxls = image.get_pixels()
        height, length = image.size()[0], image.size()[1]
        
        pxls[channel] = [[0 for i in range(length)] for i in range(height)]
        
        return RGBImage(pxls)
        

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        Takes an image and crops it to a certain area based off of the
        information specified in the input.
        """
        
        RGB = (0, 1, 2)
        max_rows = image.size()[0] - tl_row
        max_cols = image.size()[1] - tl_col
        num_rows = min(target_size[0], max_rows)
        num_cols = min(target_size[1], max_cols)
        pxls = image.get_pixels()
        
        cropped_img = [
            [
                [
                    pxls[channel][row][col]
                    for col in range(tl_col, tl_col + num_cols)
                ]
                for row in range(tl_row, tl_row + num_rows)
            ]
            for channel in RGB
        ]
        
        return RGBImage(cropped_img)


    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        Takes an image and a background and keys all pixels of a given
        color to be the matching ones in background_image.
        """
        
        #check that chroma_image and background_image are both RGBImages
        if not (
            isinstance(chroma_image, RGBImage) and 
            isinstance(background_image, RGBImage)
        ):
            raise TypeError()
        #check that chroma_image and background_image are the same size
        if chroma_image.size() != background_image.size():
            raise ValueError()
        
        NUM_CHANNELS = 3
        
        height, length = chroma_image.size()[0], chroma_image.size()[1]
        keyed_image = chroma_image.copy()
        for row in range(height):
            for col in range(height):
                if chroma_image.get_pixel(row, col) == color:
                    keyed_image.set_pixel(
                        row, 
                        col,
                        background_image.get_pixel(row, col)
                    )
                    
        return keyed_image

    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        Takes an image and returns a copy of it in which it is flipped
        upside down.
        """
        
        pxls = image.get_pixels()
        height = image.size()[0]
        
        rotated_image = [
            [row[::-1] for row in channel][::-1] 
            for channel in pxls
        ]
        
        return RGBImage(rotated_image)


# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    kNN classifier for guessing labels of provided images.
    """

    def __init__(self, n_neighbors):
        """
        Creates a new ImageKNNClassifier object.
        
        Parameters:
        n_neighbors (int): Size of nearest neighborhood used to assign
        labels in predictions.
        """
        
        self.n_neighbors = n_neighbors
        self.data = None

    def fit(self, data):
        """
        Takes training data as input and stores it inside of the classifier.
        """
        
        #make sure size of data is greater than self.n_neighbors
        if not len(data) > self.n_neighbors:
            raise ValueError()
        #make sure there isn't already training data stored
        if self.data:
            raise ValueError()
        
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        Calculates the distance between two images for use in prediction.
        """
        
        #Known bug - only returns zero
        #make sure both arguments are RGBImages
        if not (
            isinstance(image1, RGBImage) and 
            isinstance(image2, RGBImage)
        ):
            raise TypeError()
        #make sure sizes of arguments match
        if not image1.size() == image2.size():
            raise ValueError()
            
        RGB, SQUARED = (0, 1, 2), 2
        height, length = image1.size()[0], image1.size()[1]
        
        #sum squared differences of intensity values
        dist = sum([
            sum([
                sum([
                    (
                        image1.get_pixel(row, col)[channel] - 
                        image2.get_pixel(row, col)[channel]
                    ) ** SQUARED
                    for col in range(length)
                ])
                for row in range(height)
            ])
            for channel in RGB
        ])
        dist **= 0.5
        
        return dist

    @staticmethod
    def vote(candidates):
        """
        Takes a list of candidates and returns the most popular label
        among them.
        """
        
        #dict to hold labels
        labels = {}
        
        #find how much of each label there is
        for candidate in candidates:
            if candidate not in labels:
                labels[candidate] = 1
            else:
                labels[candidate] += 1
        #pick out most popular label
        popular = max(labels, key=lambda label: labels[label])
        return popular
            

    def predict(self, image):
        """
        Predicts the label of an image based off of the training data.
        """
        
        #make sure we have data fitted
        if not self.data:
            raise ValueError()
        
        #make list of tuples of distances from our image plus their labels
        closest_labels = [
            (ImageKNNClassifier.distance(image, item), label) 
            for item, label in self.data
        ]
        #sort by distance
        closest_labels.sort(key=lambda pair: pair[0])
        #get the most popular label from k best candidates
        winning_label = ImageKNNClassifier.vote(
            [closest_labels[i][1] for i in range(self.n_neighbors)]
        )
        
        return winning_label
