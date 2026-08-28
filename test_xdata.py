import ezdxf

doc = ezdxf.new('R2010')
msp = doc.modelspace()

if "DXF_CONVERTER" not in doc.appids:
    doc.appids.add("DXF_CONVERTER")

pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
polyline = msp.add_lwpolyline(pts, close=True)

xdata = [
    (1000, "ID:5"),
    (1040, 100.5)
]
polyline.set_xdata("DXF_CONVERTER", xdata)
doc.saveas("output/test_xdata.dxf")
print("Success")
