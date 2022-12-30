# padding used in integral image
PADDING = 1
# original window size
WINDOW_SIZE = 24,24
# the scale by which scale variable starts by
STARTING_SCALE = 1
# factor by which window scale increase each time
SCALE_FACTOR = 1.25

# Subsequent locations are obtained by shifting the window some number of pixels delta. 
# This shifting process is affected by the scale of the detector: 
# if the current scale is s , then the window is shifted by round(s*delta)

# The results we present are for delta =1. 
# We can achieve a significant speedup by setting delta = 1.5 setting with only a slight decrease in accuracy.
DELTA_SHIFT = 1