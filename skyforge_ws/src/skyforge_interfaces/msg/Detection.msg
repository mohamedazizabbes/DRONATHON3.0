# One 2D detection (common format of the jury and team vision modules)
string cls            # debris | mine | weapon | injured | hostile_unit
float32 conf
float32[4] bbox       # x1, y1, x2, y2 in pixels
string source         # jury | team
string attrs_json     # optional: {"temp_k":309.5,"posture":"lying","moving":false}
