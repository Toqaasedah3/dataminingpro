import re
from typing import Dict, List, Optional
import pandas as pd

COLUMN_ALIASES = {
    "employee_id": ["employee_id", "emp_id", "id", "employee number", "employee_number", "EmployeeNumber"],
    "salary": ["salary", "income", "monthly_income", "MonthlyIncome", "wage", "pay", "compensation"],
    "satisfaction": ["satisfaction", "satisfaction_level", "job_satisfaction", "JobSatisfaction", "happiness"],
    "overtime": ["overtime", "over_time", "OverTime", "extra_hours", "overtime_hours"],
    "department": ["department", "Department", "dept", "team", "unit", "division"],
    "performance": ["performance", "performance_score", "rating", "PerformanceRating", "evaluation", "last_evaluation"],
    "attrition": ["attrition", "Attrition", "left", "resigned", "turnover", "quit"],
    "years_at_company": ["years_at_company", "YearsAtCompany", "tenure", "experience", "years_company", "time_spend_company"],
    "work_life_balance": ["work_life_balance", "WorkLifeBalance", "worklifebalance", "balance"],
}

def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def detect_hr_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    normalized_map = {normalize_column_name(col): col for col in df.columns}
    detected = {}

    for target, aliases in COLUMN_ALIASES.items():
        detected[target] = None
        normalized_aliases = [normalize_column_name(a) for a in aliases]

        for norm_col, original_col in normalized_map.items():
            if norm_col in normalized_aliases:
                detected[target] = original_col
                break

        if detected[target] is None:
            for norm_col, original_col in normalized_map.items():
                if any(alias in norm_col for alias in normalized_aliases):
                    detected[target] = original_col
                    break

    return detected

def missing_important_columns(detected: Dict[str, Optional[str]]) -> List[str]:
    important = ["salary", "satisfaction", "overtime", "department", "performance"]
    return [col for col in important if detected.get(col) is None]