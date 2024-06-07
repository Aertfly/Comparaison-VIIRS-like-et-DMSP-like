from plp import lit_pixel
from plp import lit_pixel_resized
import time
def get_max(country):
    before = time.time()
    plp = lit_pixel(country,"VIIRS")
    max = plp.getMax()
    elapsed =  time.strftime('%H:%M:%S', time.gmtime(time.time() - before))
    print("elapsed",elapsed,max)
    return max

def get_max_resized(country):
    before = time.time()
    plp = lit_pixel_resized(country,"VIIRS")
    max = plp.getMax()
    elapsed =  time.strftime('%H:%M:%S', time.gmtime(time.time() - before))
    print("elapsed",elapsed,max)
    return max