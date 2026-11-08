# Confirmed or tentative 3D track (ENU metres, ground plane z = 0)
string id
string cls
float64 x
float64 y
float64 vx
float64 vy
float32 sigma         # 1-sigma horizontal position uncertainty, m
float32 conf
bool confirmed
bool floating         # in water: drifts with current + leeway
string category       # START: IMM | DEL | MIN | EXP | UNK
float32 urgency
float64 first_seen
float64 last_seen
string attrs_json     # decision-relevant attributes: likely_deceased, marked (kit delivered),
                      # layers + depth (mines), category_why, detection_radius (hostiles)
