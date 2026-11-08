# Exclusion zone (hard constraint for planner and reflex)
string id
string cls
float64 x
float64 y
float32 radius
float64 vx
float64 vy
float32 soft_radius   # hostile line-of-sight ring (soft cost), 0 if none
string source         # perception | station | params | handover | provisional
