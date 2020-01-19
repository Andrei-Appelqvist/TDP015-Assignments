"""
Convert a list of greyscale values into a PNG image.
"""

import zlib
import struct

def make_png(width, height, data):
    def B1(value):
        # network (= big-endian) unsigned char (from 0 to 2^8-1)
        return struct.pack("!B", value & (2 ** 8 - 1))
    def B4(value):
        # network (= big-endian) unsigned int (from 0 to 2^32-1)
        return struct.pack("!I", value & (2 ** 32 - 1))
    def make_chunk(ctype, cdata):
        body = ctype.encode('ascii') + cdata
        return B4(len(cdata)) + body + B4(zlib.crc32(body))
    
    png = b"\x89" + "PNG\r\n\x1A\n".encode('ascii')
    
    IHDR = B4(width) + B4(height) + B1(8) + B1(0) + B1(0) + B1(0) + B1(0)
    png += make_chunk("IHDR", IHDR)
    
    IDAT = b""
    for i in range(0, len(data), width):
        IDAT += b"\0" + b"".join(map(lambda b: B1(b), data[i:i+width]))
    png += make_chunk("IDAT", zlib.compress(IDAT))
    
    IEND = b""
    png += make_chunk("IEND", IEND)
    
    return png

def save_png(width, height, data, file):
    with open(file, 'wb') as f:
        f.write(make_png(width, height, data))

if __name__ == "__main__":
    import mnist
    for i, (x, y) in enumerate(mnist.read_test_data()[:10]):
        save_png(28, 28, map(lambda p: 255 - p, x), "mnist-%d.png" % i)
