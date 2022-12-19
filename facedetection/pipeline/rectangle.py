
class Rectangle:
    def __init__(self, x: int, y: int, rect_width:int, rect_height: int, rect_weight: int) -> None:
        self.x = x
        self.y = y
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.rect_weight = rect_weight
        return

    # since haar features we got all depend on a window size of 24 so we will need to try other window sizes
    # since some faces may be smaller than that window or maybe larger so we will add a scale factor
    def scale_rectangle(self, scale_factor):
        # top left coord won't be changed (x,y)

        # all others will be scaled
        rect_x_scaled, rect_y_scaled = round(self.x * scale_factor) , round(self.y * scale_factor)
        rect_width_scaled , rect_height_scaled = round(scale_factor * self.rect_width), round(scale_factor * self.rect_height)
        # should i scale the weight or keep it since its same ratio
        # rect_weight_scaled = scale_factor * self.rect_weight

        return Rectangle(rect_x_scaled, rect_y_scaled, rect_width_scaled, rect_height_scaled, self.rect_weight)