# Data Quality and Validation Report

- **Validation Status:** `PASSED WITH WARNINGS`
- **Total Records Analyzed:** 110,527
- **Target No-Show Rate:** 20.19%
- **Duplicate Records:** 0
- **Impossible Age Count (<0 or >115):** 1
- **Invalid Appointment Date Count (Appt < Sched):** 5

## Target Distribution
- `No`: 88,208
- `Yes`: 22,319

## Identified Anomalies
- [WARNING] Detected 1 impossible age values (< 0 or > 115).
- [WARNING] Detected 5 records where AppointmentDay precedes ScheduledDay.
