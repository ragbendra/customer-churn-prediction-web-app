# **Machine Learning Workflow Notes Step-by-Step:**

## Task-1: Pre-EDA
  ### What I Did:
  - Loaded dataset files which was in zip(containing train and test csv files),
  - Checked the dtypes, shape, sample, duplicate, nulls of each columns,
  - Calculated no. of catgeorical features and its values by objects and category dtype,
  - Converted the object dtype columns into category dtype columns according to their values,
  ### Reasoning:
  - Could have loaded after extraction but I want to make so that even zips are usable for my workflow,
  - Wanted to know the dataset values, types, and others,

Key Validation Checks We Need:

Schema Validation: Required columns present?

Data Type Validation: MonthlyRevenue is numeric?

Value Range Validation: Age between 18-100?

Business Rule Validation: CustomerID unique per company?

Data Quality: Missing value thresholds?