# PharmaForecast
## Demand Forecasting for Supply Chain Risk Management

This is a Flask Data Product which uses scss, jinja2, etc for the base. It also includes regression models from scikit-learn and other relevant tools listed in requirements.txt

---

### Home Page

![alt text](pages/Screenshot%202025-07-10%20172413.png)

### Dataset

![alt text](pages/Screenshot%202025-07-10%20172150.png)

### Data Entry

![alt text](pages/Screenshot%202025-07-10%20172219.png)

### Demand Forecasts

![alt text](pages/Screenshot%202025-07-10%20172100.png)

### Consumption & Stock Overview

![alt text](pages/Screenshot%202025-07-01%20131406.png)

### Supply Chain Risks Dashboard

![alt text](pages/Screenshot%202025-07-01%20125736.png)
![alt text](pages/Screenshot%202025-07-01%20125641.png)


---

## 🧼 Project Structure

```
PharmaForecast/
├── env/                  # virtual environment (excluded by .gitignore)
├── instance/             # database folder (excluded by .gitignore)
│   └── *.db
├── uploads/             # dataset folder (excluded by .gitignore)
│   └── *.xlsx
│   └── *.txt
├── templates/            # HTML files
│   ├── base.html
│   ├── index.html
│   ├── dataset.html
│   ├── data_entry.html
│   ├── dashboard_forecasting.html
│   ├── dashboard_consumption.html
│   ├── dashboard_suppliers.html
│   ├── performance.html
│   └── edit.html
├── static/               # SCSS/CSS or JS files
│   └── images/
│       ├── *.png
│       └── *.png
│   ├── styles.css
│   ├── styles.css.map
│   └── styles.scss
├── app.py                # main Flask app
├── requirements.txt      # dependencies
├── .gitignore
└── README.md
```

---

Created by Rafia H
