import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans

class Quantizer:
    """Perform quantization using kmeans. The frames are divided into `h_blocks` x `w_blocks`.
    Each block of the frames are assigned a symbol corresponding to nearest cluster. This 
    reduce a high dimensional block into a scalar value. If a frame is divided
    into 25 blocks. The frame will be reduced to 25 dimensional vector.
    """

    def __init__(self, h_blocks: int, w_blocks: int, num_points: int):
        """Construct a qunatizer with given number of quantizers and clusters

        Parameters
        ----------
        h_blocks : int
            Number of blocks along axis 0
        w_block : int
            Number of blocks along axis 1
        num_points : int
            Number of possible point for each block. If num_clusters = 40, then there would
            be 40 possible values for each block
        """
        self.h = h_blocks
        self.w = w_blocks
        self.num_quantizer = self.h * self.w
        self.kmeans = [
            KMeans(n_clusters = num_points, random_state=42) for i in range(self.num_quantizer)
        ]
    
    def __split(self, frame: np.array,h: int, w: int) -> np.array:
        """Split a frame into hxw blocks

        Parameters
        ----------
        frame : np.array
            One frame
        h : int
            Number of block along axis 0
        w : int
            Number of block along axis 1

        Returns
        -------
        np.array
            Array with frames splitted into blocks
        """
        assert frame.shape[0]%h == 0, "The height of frame is not divisible by given value of h_blocks"
        assert frame.shape[1]%w == 0, "The width of frame is not divisible by given value of w_blocks"

        frame = np.array(np.split(frame,h,0))
        frame = np.vstack(np.split(frame,w,2))
        return frame
    
    def fit(self, X: np.array) -> None:
        """Fit a quantizer

        Parameters
        ----------
        X : np.array of shape (n_frames, height, weigth)
            Array of frames
        """
        X = np.array([self.__split(frame, self.h, self.w) for frame in X])
        X = np.reshape(X, (X.shape[0], X.shape[1], -1))
        for i in tqdm(range(self.num_quantizer), desc="Learning mappings for blocks", leave=False):
            self.kmeans[i].fit(X[:,i,:])
    
    def quantize(self, X) -> np.array:
        """Quantize a frame with n blocks to vector with n-dim

        Parameters
        ----------
        X : np.array of shape (n_frames, height, weigth)
            Array of frames

        Returns
        -------
        np.array of shape (n_frames, h_blocks * w_blocks)
            Quantized vectors
        """
        X = np.array([self.__split(frame, self.h, self.w) for frame in X])
        X = np.reshape(X, (X.shape[0], X.shape[1], -1))
        vectors = [self.kmeans[i].predict(X[:,i,:]) for i in range(self.num_quantizer)]
        vectors = np.stack(vectors).T
        return vectors