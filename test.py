import sys
from enum import Enum, auto
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import f_oneway


class Method(Enum):
    '''Available methods for processing an image
                    REGION_BASED: segment the skin using the HSV and YCbCr colorspaces, followed by the Watershed algorithm'''
    REGION_BASED = auto()


class SkinDetector():

    def __init__(self, image_path: str) -> None:
        self.image = cv2.imread(image_path)
        self.image_mask = None
        self.skin = None

    def find_skin(self, method=Method.REGION_BASED) -> None:
        '''function to process the image based on some method '''
        if (method == Method.REGION_BASED):
            self.__color_segmentation()
            self.__region_based_segmentation()

    def get_resulting_images(self) -> tuple:
        """Returns the processed images
                        [0] = The original image
                        [1] = The resulting image mask containing the skin
                        [2] = The result image after a bitwise_and of [1] and [0]"""

        return self.image, self.image_mask, self.skin

    def __color_segmentation(self) -> None:
        '''Apply a threshold to an HSV and YCbCr images, the used values were based on current research papers along with some empirical tests and visual evaluation'''

        HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)

        lower_HSV_values = np.array([0, 30, 50], dtype="uint8")
        upper_HSV_values = np.array([180, 180, 160], dtype="uint8")

        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(
            YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)

        self.skin_mask = cv2.add(mask_HSV, mask_YCbCr)

    def __region_based_segmentation(self) -> None:
        '''Function that applies Watershed and morphological operations on the thresholded image morphological operations'''

        image_foreground = cv2.erode(
            self.skin_mask, None, iterations=3)  # remove noise

        # The background region is reduced a little because of the dilate operation
        dilated_binary_image = cv2.dilate(self.skin_mask, None, iterations=3)
        # set all background regions to 128
        _, image_background = cv2.threshold(
            dilated_binary_image, 1, 128, cv2.THRESH_BINARY)

        # add both foreground and background, forming markers. The markers are "seeds" of the future image regions.
        image_marker = cv2.add(image_foreground, image_background)
        image_marker32 = np.int32(image_marker)  # convert to 32SC1 format

        image_marker32 = cv2.watershed(self.image, image_marker32)
        m = cv2.convertScaleAbs(image_marker32)  # convert back to uint8

        # bitwise of the mask with the input image
        _, self.image_mask = cv2.threshold(
            m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.skin = cv2.bitwise_and(
            self.image, self.image, mask=self.image_mask)

    def show_all_images(self, window_title="Original Image | Skin Mask | Result") -> None:
        '''Show all processed images concatenated along the 1 axis using imshow '''
        rgb_mask = cv2.cvtColor(self.image_mask, cv2.COLOR_GRAY2RGB)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        # Convert to HSV
        hsv_img = cv2.cvtColor(self.skin, cv2.COLOR_BGR2HSV)

        # Increase hue and saturation
        hue_shift = 15
        saturation_scale = 1.5
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
        hsv_img[:, :, 1] = hsv_img[:, :, 1] * saturation_scale

        # Convert back to BGR
        yellow_face = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # Display result
        # cv2.imshow('Yellow Face', yellow_face)
        cv2.imshow(window_title, np.concatenate(
            (self.image, rgb_mask, self.skin, yellow_face), axis=1))

        plt.subplot(1, 2, 1)
        blue_color = cv2.calcHist([self.skin], [0], None, [256], [1, 256])
        red_color = cv2.calcHist([self.skin], [1], None, [256], [1, 256])
        green_color = cv2.calcHist([self.skin], [2], None, [256], [1, 256])
        # plt.plot(abc)
        plt.plot(blue_color, color="blue")
        plt.plot(green_color, color="green")
        plt.plot(red_color, color="red")

        plt.subplot(1, 2, 2)
        blue_color = cv2.calcHist([yellow_face], [0], None, [256], [1, 256])
        red_color = cv2.calcHist([yellow_face], [1], None, [256], [1, 256])
        green_color = cv2.calcHist([yellow_face], [2], None, [256], [1, 256])
        # plt.plot(abc)
        plt.plot(blue_color, color="blue")
        plt.plot(green_color, color="green")
        plt.plot(red_color, color="red")

       # Convert the image to a NumPy array
        data = np.array(yellow_face)
        # print('abc',np.array(self.skin[:,:,0]),np.array(self.skin[:,:,1]),np.array(self.skin[:,:,2]))
        # group1_features = np.array(self.skin[:,:,0])
        # group2_features = np.array(self.skin[:,:,1])
        # group3_features = np.array(self.skin[:,:,2])
        # all_features = np.concatenate([group1_features, group2_features, group3_features])
        # labels = ['Group 1'] * len(group1_features) + ['Group 2'] * len(group2_features) + ['Group 3'] * len(group3_features)
        # f_value, p_value = f_oneway(group1_features)
        # print("F-value:", f_value)
        # print("p-value:", p_value)
        pixels = self.skin.reshape(-1, 3)
        pixels1 = yellow_face.reshape(-1, 3)

        # Separate the pixel values for each RGB channel
        red_channel = pixels[:, 0]
        red_channel1 = pixels1[:, 0]
        green_channel = pixels[:, 1]
        green_channel1 = pixels1[:, 1]
        blue_channel = pixels[:, 2]
        blue_channel1 = pixels1[:, 2]

        # Perform ANOVA test for each channel
        _, red_pvalue = f_oneway(red_channel,red_channel1)
        _, green_pvalue =f_oneway(green_channel,green_channel1)
        _, blue_pvalue = f_oneway(blue_channel,blue_channel1)

        # Print the p-values
        print("Red channel p-value:", red_pvalue)
        print("Green channel p-value:", green_pvalue)
        print("Blue channel p-value:", blue_pvalue)

        # Calculate the histograms for each color channel
        # hist_red, bins_red = np.histogram(self.skin[:,:,0], bins='auto', range=[0, 256])
        # hist_green, bins_green = np.histogram(self.skin[:,:,1], bins='auto', range=[0, 256])
        # hist_blue, bins_blue = np.histogram(self.skin[:,:,2], bins='auto', range=[0, 256])

        # # Calculate the area under the curve for each channel
        # area_red = np.sum(np.diff(bins_red) * hist_red)
        # area_green = np.sum(np.diff(bins_green) * hist_green)
        # area_blue = np.sum(np.diff(bins_blue) * hist_blue)

        # # Plot the histograms for each color channel
        # # plt.figure(figsize=(10, 4))
        # # plt.subplot(1, 3, 1)
        # # plt.hist(data[:, :, 0].flatten(), bins=256, range=[0, 256], color='red')
        # # plt.title('Red Channel')
        # # plt.subplot(1, 3, 2)
        # # plt.hist(data[:, :, 1].flatten(), bins=256, range=[0, 256], color='green')
        # # plt.title('Green Channel')
        # # plt.subplot(1, 3, 3)
        # # plt.hist(data[:, :, 2].flatten(), bins=256, range=[0, 256], color='blue')
        # # plt.title('Blue Channel')

        # # Display the areas under the curves
        # plt.text(0.5, 0.9, f"SELF Red Area = {area_red:.2f}", transform=plt.gca().transAxes, color='red')
        # plt.text(0.5, 0.8, f"Green Area = {area_green:.2f}", transform=plt.gca().transAxes, color='green')
        # plt.text(0.5, 0.7, f"Blue Area = {area_blue:.2f}", transform=plt.gca().transAxes, color='blue')

        # # Show the plots
        # plt.tight_layout()
        # plt.show()
        
             
        # # Convert the image to a NumPy array
        # data = np.array(yellow_face)

        # # Calculate the histograms for each color channel
        # hist_red, bins_red = np.histogram(yellow_face[:,:,0], bins='auto', range=[0, 256])
        # hist_green, bins_green = np.histogram(yellow_face[:,:,1], bins='auto', range=[0, 256])
        # hist_blue, bins_blue = np.histogram(yellow_face[:,:,2], bins='auto', range=[0, 256])

        # # Calculate the area under the curve for each channel
        # area_red = np.sum(np.diff(bins_red) * hist_red)
        # area_green = np.sum(np.diff(bins_green) * hist_green)
        # area_blue = np.sum(np.diff(bins_blue) * hist_blue)

        # # Plot the histograms for each color channel
        # # plt.figure(figsize=(10, 4))
        # # plt.subplot(1, 3, 1)
        # # plt.hist(data[:, :, 0].flatten(), bins=256, range=[0, 256], color='red')
        # # plt.title('Red Channel')
        # # plt.subplot(1, 3, 2)
        # # plt.hist(data[:, :, 1].flatten(), bins=256, range=[0, 256], color='green')
        # # plt.title('Green Channel')
        # # plt.subplot(1, 3, 3)
        # # plt.hist(data[:, :, 2].flatten(), bins=256, range=[0, 256], color='blue')
        # # plt.title('Blue Channel')

        # # Display the areas under the curves
        # plt.text(0.5, 0.9, f"YELLOW Red Area = {area_red:.2f}", transform=plt.gca().transAxes, color='red')
        # plt.text(0.5, 0.8, f"Green Area = {area_green:.2f}", transform=plt.gca().transAxes, color='green')
        # plt.text(0.5, 0.7, f"Blue Area = {area_blue:.2f}", transform=plt.gca().transAxes, color='blue')

        # # Show the plots
        # plt.tight_layout()
        # plt.show()
        
        
        mean_self=np.mean(self.skin)
        mean_yellow=np.mean(yellow_face)
        print(mean_self,mean_yellow)
        plt.show()
        cv2.waitKey(0)


'''This is an example script for running the default library processing method'''
# from trial1 import SkinDetector


imageName = sys.argv[1]

detector = SkinDetector(
    image_path="C:/Users/sjawre/Desktop/onboarding/Personal/Dissertattion/Caucasian/Caucasian/00000488_resized.png")
detector.find_skin()
detector.show_all_images()
