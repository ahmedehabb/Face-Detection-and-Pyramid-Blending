from .rectangle import Rectangle

class Feature:
    def __init__(self, rect_lists : list[Rectangle]) -> None:
        self.rect_lists = rect_lists
        pass