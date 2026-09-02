# Aegis Incident CP-7009

Incident ID GV-2026-095. A development bucket applied the obsolete seven-day upload retention instead of 24 hours. CP-7009 blocked promotion before production data arrived. Infrastructure corrected the policy and added retention checks to deployment CI.
