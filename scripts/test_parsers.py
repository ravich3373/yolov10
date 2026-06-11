#!/usr/bin/env python
"""Unit checks for every annotation parser, using fixtures taken from the real
formats (CCPD filename grammar, CRPD label lines, OpenALPR lines, UC3M JSON,
VOC XML, Roboflow name mangling, YOLO label round-trip)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lpr.data.datasets.ccpd import parse_ccpd_filename  # noqa: E402
from lpr.data.datasets.crpd import parse_crpd_label  # noqa: E402
from lpr.data.datasets.openalpr import parse_openalpr_line  # noqa: E402
from lpr.data.datasets.uc3m_lp import parse_uc3m_json  # noqa: E402
from lpr.data.datasets.kaggle_andrewmvd import parse_voc_xml  # noqa: E402
from lpr.data.datasets.roboflow_sets import roboflow_group_key  # noqa: E402
from lpr.data.datasets.base import write_yolo_label, read_yolo_label  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


# CCPD: real filename from the dataset README
stem = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15"
parsed = parse_ccpd_filename(stem)
check("ccpd: parses", parsed is not None)
boxes, plate_id = parsed
check("ccpd: bbox == (154,383,386,473)", boxes == [(154.0, 383.0, 386.0, 473.0)])
check("ccpd: plate identity field", plate_id == "0_0_22_27_27_33_16")
check("ccpd: non-conforming name (ccpd_np) -> None", parse_ccpd_filename("1005") is None)

# CRPD: corners in arbitrary order + Chinese plate string; one empty line
crpd = "120 200 320 200 320 260 120 260 0 皖A12345\n  \n500.5 100 480 100 480 140.25 500.5 140 1 苏B67890"
plates = parse_crpd_label(crpd)
check("crpd: two plates", len(plates) == 2)
check("crpd: bbox via min/max (order-invariant)", plates[0][0] == (120.0, 200.0, 320.0, 260.0))
check("crpd: plate string captured", plates[0][1] == "皖A12345")
check("crpd: ccw corners still correct", plates[1][0] == (480.0, 100.0, 500.5, 140.25))

# OpenALPR: 'file x y w h plate' (x,y,w,h -> xyxy)
line = "wts-lg-000238.jpg\t935\t362\t99\t49\tYG9X2G"
parsed = parse_openalpr_line(line)
check("openalpr: bbox xywh->xyxy", parsed[0] == (935.0, 362.0, 1034.0, 411.0))
check("openalpr: plate text", parsed[1] == "YG9X2G")
check("openalpr: malformed -> None", parse_openalpr_line("garbage") is None)

# UC3M: polygon corners -> bbox; trust pairing not imagePath
uc3m = """{"imagePath": "wrong_name.jpg", "imageWidth": 5184, "imageHeight": 3888,
"lps": [{"lp_id": "DN1234**", "poly_coord": [[100, 200], [510, 210], [505, 300], [98, 290]],
"characters": []}]}"""
boxes, w, h = parse_uc3m_json(uc3m)
check("uc3m: poly -> bbox min/max", boxes == [(98, 200, 510, 300)])
check("uc3m: size from json", (w, h) == (5184, 3888))

# VOC XML (andrewmvd)
voc = """<annotation><object><name>licence</name>
<bndbox><xmin>226</xmin><ymin>125</ymin><xmax>419</xmax><ymax>173</ymax></bndbox></object>
<object><name>licence</name>
<bndbox><xmin>250</xmin><ymin>142</ymin><xmax>400</xmax><ymax>150</ymax></bndbox></object></annotation>"""
check("voc: two boxes, first correct", parse_voc_xml(voc)[0] == (226.0, 125.0, 419.0, 173.0))

# Roboflow group keys: strip export mangling so copies share a group
check("roboflow: strips .rf hash + _jpg", roboflow_group_key("scan0001_jpg.rf.0a1b2c3d") == "scan0001")
check("roboflow: plain stem unchanged", roboflow_group_key("IMG_1234") == "IMG_1234")

# YOLO label round-trip incl. clipping and degenerate-box dropping
import tempfile

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.txt"
    write_yolo_label(p, [(10, 20, 110, 70), (-5, 0, 50, 40), (3, 3, 4, 4)], 200, 100)
    back = read_yolo_label(p, 200, 100)
    check("yolo: degenerate box dropped (2 of 3 kept)", len(back) == 2)
    check("yolo: round-trip box 1", all(abs(a - b) < 0.05 for a, b in zip(back[0], (10, 20, 110, 70))))
    check("yolo: clipped to image bounds", back[1][0] >= 0)
    write_yolo_label(p, [], 200, 100)
    check("yolo: negative image -> empty file, zero boxes", p.read_text() == "" and read_yolo_label(p, 200, 100) == [])

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'}")
sys.exit(1 if failures else 0)
