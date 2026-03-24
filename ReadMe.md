## Project Title

The Billion Dollar Question: Is Vanguard's New UI Worth It?

## Content

- [Project Title](#project-title)
- [Content](#content)
- [Contributors](#contributors)
- [Datasets](#datasets)
- [About Vanguard](#about-vanguard)
- [Business question](#business-question)
- [Hypotheses](#hypotheses)
- [Data Cleaning \& Obstacles](#data-cleaning--obstacles)
- [Methodology](#methodology)
- [Results](#results)
- [Additional insights](#additional-insights)
- [Business Recommendations](#business-recommendations)
- [Tech Stack](#tech-stack)
- [URL to Slides](#url-to-slides)
- [Repository content](#repository-content)
- [Installation instruction](#installation-instruction)

## Contributors

- Ofelia Akopian, 
- Alex Mateu, 
- Sarah Jane Nede, 
- Bruno Sousa

## Datasets
We started out with 4 Datasets from Vanguard:
- df_final_demo.txt
- df_final_experiment_clients.txt
- df_final_web_date_pt_1.txt
- df_final_webdata_pt_2.txt

The most important columns for our project:
From df_final_demo.txt
- client_id
- logon_6_mnth
- clnt_tenure_yr
  
From df_final_experiment_clients
- client_id
- Variation (whether the clients were in the test or the control group)

From df_final_web_data_pt_1.txt & df_final_webdata_pt_2.txt
- client_id
- process_step
  
## About Vanguard
Vanguard is one of the world's largest investment management companies, founded in 1975 and serving over 50 million clients worldwide. As part of their ongoing commitment to improving the client experience, Vanguard conducted an A/B test to evaluate whether a new user interface design leads to higher process completion rates among their clients. 

## Business question
Does the new Vanguard UI design drive higher process completion rates, and if so, does the improvement justify the investment?

## Hypotheses 
1: Completion Rate
H₀: The new design does not increase the completion rate compared to the old design. 
The completion rate in the Test group is less than or equal to the completion rate in 
the Control group.

H₁: The new design increases the completion rate compared to the old design. 
The completion rate in the Test group is higher than the completion rate in the Control group. 

2: Completion Rate with a Cost-Effectiveness Threshold
H₀: The new process design does not achieve the minimum required increase in completion rate. The difference in completion rate between the test group and the control group is less than 5%, meaning the new design is not cost-effective

H₁: The new process design achieves or exceeds the minimum required increase in completion rate. The difference in completion rate between the test group and the control group is 5% or more, meaning the new design is cost-effective and its benefits outweigh the associated costs of design, development, testing, staff training and potential user disruptions.

3: Completion Rate according to tenure years
H₀: The completion rate of the new process does not differ significantly from the old process across tenure groups

H₁: The completion rate of the new process differs significantly from the old process across at least one tenure group

## Data Cleaning & Obstacles

The dataset was relatively clean and therefore did not present any significant obstacles. We removed rows that contained only null values, with the exception of the client_id, and merged the datasets df_final_web_data_pt_1 and df_final_web_data_pt_2.

The primary challenges we encountered were related to alignment, communication, and mutual understanding within the team. Although communication was ongoing, some misunderstandings arose, and different approaches were taken to address errors encountered during the process. While this was not a major issue, it was ultimately insightful to observe that the overall error rate was relatively low. However, it did require us to realign at a later stage of the project.

## Methodology

To compare completion rates between the test and control groups, a Chi-square test was used as completion is a categorical variable (completed/not completed). To verify that the groups were balanced in terms of client tenure, an independent samples T-test was applied, preceded by Levene's test for equality of variances and a normality check. A significance level of α = 0.05 was used throughout the analysis.

## Results

The results show:
- We have to reject the first H₀: Completion Rate.
H₀: The new design does not increase the completion rate compared to the old design. 
The completion rate in the Test group is less than or equal to the completion rate in 
the Control group.
    The new design does increase the completion rate:

    | | Completion Rate |
    |---|---|
    | Control (Old UI) | 64.6% |
    | Test (New UI) | 68.0% |
    | Difference | +3.4% |
    

- The second H₀: Completion Rate with a Cost-Effectiveness Threshold we fail to reject.
H₀: The new process design does not achieve the minimum required increase in completion rate. The difference in completion rate between the test group and the control group is less than 5%, meaning the new design is not cost-effective.
The new UI performs better but only by 3.4%.


- The last H₀: Completion Rate according to tenure years we have to reject.
H₀: The completion rate of the new process does not differ significantly from the old process across tenure groups
Especially under the newer clients the new UI does better with up to 3.9% better in the tenure group 6-10 years. 


## Additional insights
 
 - 31+ year tenure clients perform better with old UI 
 - Biggest drop-off at start (control 14% vs test 9.1%) 
 - Step 1 loses more clients in test group (control 6.2% vs test 7.3%)
 - 33 clients without start event — minor tracking issue 
 - 1,412 clients confirmed without completing all steps 

## Business Recommendations

The new UI shows promise but does not yet meet the 5% cost-effectiveness threshold (observed improvement: 3.4%). We recommend against full implementation at this stage.

Key actions before re-evaluation:
• Investigate and reduce process drop-offs
• Resolve process integrity errors (clients confirming without completing all steps)
• Develop a tailored approach for long-tenure clients (31+ years) who perform better under the old process

We are confident that addressing these issues could push completion rates above the 5% threshold, making the new UI a worthwhile investment.


## Tech Stack

For this project we used:
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
- ![Python](https://img.shields.io/badge/Python-3.13-blue)
- ![Pandas](https://img.shields.io/badge/Pandas-3.0.1-150458?logo=pandas)
- ![NumPy](https://img.shields.io/badge/NumPy-2.4.3-013243?logo=numpy)
- ![Tableau](https://img.shields.io/badge/Tableau-blue?logo=tableau)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-0orange)
- ![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-orange)
- ![Scipy](https://img.shields.io/badge/Scipy-1.17.1-red)
 

## URL to Slides

https://docs.google.com/presentation/d/1MtAjua1jVTg8QEi5DEEQXuwsui0Agx2kUaQ2pm36Hrk/edit?slide=id.g3d15ee1e420_0_1735#slide=id.g3d15ee1e420_0_1735

## Repository content

```
2nd-project/
│
├── data/
│   ├── clean/
│   │   ├── df_final_experiment_clients.csv
│   │   ├── df_final_web_data.csv
│   │   ├── df_web.csv
│   │   ├── error_free_df_web.csv
│   │   ├── joined_demo_expi_df.csv
│   │   └── test_group_df.csv
│   │
│   └── raw/
│       ├── df_final_demo.txt
│       ├── df_final_experiment_clients.txt
│       ├── df_final_web_data_pt_1.txt
│       └── df_final_web_data_pt_2.txt
│
├── Figures/
│   ├── call_per_tenure_group.png
│   ├── calls_by_age.png
│   ├── client_tenure_yr.png
│   ├── completion_rates_tenure_group.png
│   ├── completion_rates_test_vs_control.png
│   ├── contingency_table_heatmap.png
│   ├── logons_by_age.png
│   ├── logons_per_tenure_group.png
│   └── tenure_distribution_test_vs_control.png
│
├── Notebooks/
│   ├── alex.ipynb
│   ├── bruno.ipynb
│   ├── sarah_tenure.ipynb
│   └── sarah.ipynb
│
├── .python-version
├── config.yaml
├── pyproject.toml
└── README.md
```
## Installation instruction

1. clone the repository:
```
git clone https://github.com/BMAS30/2nd-project
```

2. Install UV

if you're a MacOS/Linux user:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

if you're a Windows user open an Anaconda Powershell Prompt and enter:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Create an environment:
``` 
uv venv
```

4. Activate the environment:
if you're a MacOS/Linux user type:
```
source ./venv/bin/activate
```

if you're a Windows user enter in Anaconda Powershell Prompt:
```
.\.venv\Scripts\activate
```

1. Install dependencies:
```
uv pip install -r requirements.txt
```