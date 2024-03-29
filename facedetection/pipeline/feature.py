from .rectangle import Rectangle
from constants import PADDING, WINDOW_SIZE

class Feature:
    def __init__(self, rect_lists : list[Rectangle]) -> None:
        self.rect_lists = rect_lists
        pass

    def compute_feature(self, integral_image_window, scale):
        value = 0
        for rect in self.rect_lists:
            # get the new rectangle after scaling the original rectangle by scale variable
            rect_after_scaling = rect.scale_rectangle(scale)
            x, y, rect_width, rect_height, rect_weight = rect_after_scaling.x,  rect_after_scaling.y, rect_after_scaling.rect_width, rect_after_scaling.rect_height, rect_after_scaling.rect_weight

            # VERY IMPORTANT !!!
            # take care x,y are with respect to the window (24,24)
            # not wrt to the integral image (25,25)
            # therefore if you needed for ex from (0,0) and  width, height = 2,2  --- (0,0 -> 1,1)
            # this mean (0,0) will be reflected in the integral image to be (1,1) due to padding made
            # since width, height = 2,2  -> therefore (1,1) -> (2,2) in integral image
            # but we will also need to add prev row and col therefore from (0,0) -> (2,2). WHY ?
            # because if you didnt add it you will get wrong calculations of integral image by removing top left element
            # which could cause wrong classfication
            
            # therefore to get required rect with PADDING = [y: y+ height + PADDING , x: x+ width + PADDING]

            # take previous row and column to compute window correctly
            # example if integral window 
            # 0 0 0
            # 0 1 2
            # 0 3 4
            # and the rect x,y is 1,1 and width, height = 2,2
            # this means you are addressing the window 
            # 1 1
            # 1 1
            # but you will need the previous column and row to correctly get the sum of the window
            rect_window = integral_image_window[y: y + rect_height + PADDING, x: x + rect_width + PADDING]
            x_end ,y_end  = rect_width + PADDING - 1, rect_height + PADDING - 1
            # calculating value of the rectangle
            value += (rect_window[y_end, x_end] + rect_window[0, 0] - rect_window[0, x_end] - rect_window[y_end, 0]) * rect_weight
            
        # normalize the feature by window area 
        # also casted each dimension scaled to int since (scale * dimension) may got a non-integer which could
        # make our calculations wrong
        return value / (int(WINDOW_SIZE[0] * scale) * int(WINDOW_SIZE[1] * scale))
