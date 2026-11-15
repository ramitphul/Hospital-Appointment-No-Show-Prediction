# Exploratory Data Analysis (EDA) Report
## Brazilian Medical Appointment Attendance Dataset

### 1. Executive Summary & Core Statistics
- **Total Appointments:** 110,521
- **Attended Appointments:** 88,207 (79.81%)
- **Missed Appointments (No-Shows):** 22,314 (20.19%)
- **Patient Mean Age:** 37.1 years
- **Median Scheduling Lead Time:** 4.0 days

### 2. Key Findings & Operational Insights

#### A. The Dominant Lead Time Factor (Waiting Days)
- **Same-Day Bookings:** Patients who book and have an appointment on the exact same day have an extraordinarily low no-show rate of **4.65%**.
- **Long Lead Times:** When the appointment is scheduled more than 15 days in advance, the no-show rate surges above **32%**.
- **Operational Takeaway:** Shortening booking horizons and offering rapid fill-in slots can directly reduce unfulfilled appointment capacity.

#### B. Confounding Variable Analysis: The SMS Reminder Paradox
- **No-Show Rate with SMS Received:** 27.57%
- **No-Show Rate without SMS:** 16.7%
- **Important Healthcare Context:** At first glance, receiving an SMS appears to correlate with *higher* no-shows. However, this is a classic example of **confounding by indication (Simpson's Paradox)**: hospital protocol only dispatched SMS reminders to patients whose appointments were booked multiple days in advance. Same-day appointments (which almost never miss) never received SMS messages. Controlling for waiting days reveals that SMS reminders do not induce no-shows; rather, long waiting days drive both the receipt of an SMS and the likelihood of missing the appointment.

#### C. Demographics & Chronic Disease Associations
- **Age Groups:** Young adults (20-39 years) exhibit the highest no-show rate (~24.6%), whereas seniors (60+) demonstrate significantly higher compliance (~15.3%).
- **Socioeconomic Support:** Patients enrolled in the *Bolsa Família* welfare scholarship program experience a moderately higher no-show rate (~23.7% vs 19.8%), often reflecting transportation, childcare, or hourly wage work constraints.

### 3. Generated Visualizations
- `target_distribution.png`: Binary class balance analysis
- `age_distribution.png`: Histogram & density curve by attendance
- `noshow_by_age_group.png`: Breakdown across life stages
- `noshow_by_weekday.png`: Weekly operational attendance patterns
- `sms_reminder_comparison.png`: Confounding factor inspection
- `waiting_time_vs_noshow.png`: Lead time sensitivity analysis
- `correlation_heatmap.png`: Spearman/Pearson correlation matrix
- `neighbourhood_noshow_rates.png`: Spatial variation across clinic locations
