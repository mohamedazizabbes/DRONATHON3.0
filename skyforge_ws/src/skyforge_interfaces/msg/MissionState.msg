std_msgs/Header header
string phase          # INIT EXPLORE DETECT TRIAGE NAVIGATE DELIVER RETURN_BASE LANDED MAYDAY
string phase_fr       # INIT EXPLORER DÉTECTER TRIAGE NAVIGUER LIVRER RETOUR BASE POSÉ MAYDAY
float64[3] position   # ENU
float32 battery_pct
float32 battery_wh
float32 rth_threshold_wh
float32 coverage
bool link_up
bool gnss_ok
string nav_source
int32 kits
string target_id
float32 decision_ms
